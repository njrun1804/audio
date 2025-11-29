"""Audio ingestion, normalization, and chunking."""

import atexit
import hashlib
import logging
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from asr.config import ASRConfig, VADConfig

logger = logging.getLogger(__name__)

# Global VAD model cache to avoid reloading on every call
# Protected by lock for thread safety
# Can contain either (model, utils) on success or Exception on failure
_VAD_MODEL_CACHE: Optional[tuple | Exception] = None
_VAD_MODEL_LOCK = threading.Lock()

# Track temp directories for cleanup
# Protected by lock for thread safety
_TEMP_DIRS_TO_CLEANUP: list[Path] = []
_TEMP_DIRS_LOCK = threading.Lock()


def _cleanup_temp_dirs():
    """Cleanup temporary directories on exit."""
    with _TEMP_DIRS_LOCK:
        for temp_dir in _TEMP_DIRS_TO_CLEANUP:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.debug(f"Failed to cleanup temp directory {temp_dir}: {e}")
        _TEMP_DIRS_TO_CLEANUP.clear()


# Register cleanup handler
atexit.register(_cleanup_temp_dirs)


@dataclass
class AudioChunk:
    """A chunk of audio ready for transcription."""

    samples: np.ndarray  # 16kHz mono float32
    start_time: float  # Original file offset in seconds
    duration: float
    path: Path  # Temp WAV path for backends that need file input
    difficulty_score: float = 0.0  # 0=easy, 1=hard; used for adaptive processing


def check_ffmpeg() -> None:
    """Check if ffmpeg is installed."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found. Install with: brew install ffmpeg"
        )


# =============================================================================
# Subprocess Helper (DRY refactor)
# =============================================================================

def _run_subprocess(
    cmd: list[str],
    timeout: int,
    context: str,
) -> subprocess.CompletedProcess:
    """Run subprocess with consistent error handling.

    Args:
        cmd: Command and arguments
        timeout: Timeout in seconds
        context: Context for error messages (e.g., "ffmpeg", "ffprobe")

    Returns:
        CompletedProcess on success

    Raises:
        RuntimeError: On timeout or non-zero exit
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{context} timed out after {timeout}s")

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "unknown error"
        raise RuntimeError(f"{context} failed: {error_msg}")

    return result


def get_file_hash(path: Path) -> str:
    """Get SHA256 hash of file for caching."""
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    except (FileNotFoundError, PermissionError, IOError) as e:
        raise RuntimeError(f"Failed to hash file {path}: {e}")


def build_audio_filter_chain(
    loudness_enabled: bool = True,
    loudness_target_lufs: int = -16,
    highpass_enabled: bool = True,
    highpass_freq: int = 80,
    noise_reduction: str = "off",
) -> str | None:
    """Build ffmpeg audio filter chain for pre-processing.

    Returns filter string or None if no filters enabled.
    """
    # SECURITY: Validate numeric parameters to prevent injection
    if not isinstance(loudness_target_lufs, int) or not (-70 <= loudness_target_lufs <= 0):
        raise ValueError(
            f"loudness_target_lufs must be integer between -70 and 0, got {loudness_target_lufs}"
        )
    if not isinstance(highpass_freq, int) or not (20 <= highpass_freq <= 2000):
        raise ValueError(
            f"highpass_freq must be integer between 20 and 2000 Hz, got {highpass_freq}"
        )
    if noise_reduction not in ("off", "light", "moderate"):
        raise ValueError(
            f"noise_reduction must be 'off', 'light', or 'moderate', got {noise_reduction}"
        )

    filters = []

    # High-pass filter first (removes rumble before other processing)
    if highpass_enabled:
        filters.append(f"highpass=f={highpass_freq}:poles=2")

    # Noise reduction (optional - can degrade clean audio)
    if noise_reduction != "off":
        strength_map = {"light": 0.3, "moderate": 0.5}
        strength = strength_map.get(noise_reduction, 0.3)
        filters.append(f"anlmdn=s={strength}")

    # Loudness normalization last (LUFS targeting)
    if loudness_enabled:
        filters.append(f"loudnorm=I={loudness_target_lufs}:TP=-1.5:LRA=11")

    return ",".join(filters) if filters else None


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file in seconds using ffprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds

    Raises:
        RuntimeError: If ffprobe fails or returns invalid duration
    """
    check_ffmpeg()

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    result = _run_subprocess(cmd, 30, f"ffprobe for {audio_path}")

    duration_str = result.stdout.strip()
    if not duration_str:
        raise RuntimeError(f"ffprobe returned empty duration for {audio_path}")

    try:
        return float(duration_str)
    except ValueError:
        raise RuntimeError(f"ffprobe returned invalid duration: {duration_str}")


def normalize_audio(
    input_path: Path,
    output_path: Path,
    timeout: int = 300,
    loudness_enabled: bool = True,
    loudness_target_lufs: int = -16,
    highpass_enabled: bool = True,
    highpass_freq: int = 80,
    noise_reduction: str = "off",
) -> float:
    """
    Convert any audio format to 16kHz mono WAV using ffmpeg.

    Applies optional audio enhancement filters:
    - High-pass filter (removes low-frequency rumble)
    - Loudness normalization (LUFS targeting for consistent levels)
    - Noise reduction (optional, off by default)

    Returns duration in seconds.
    """
    check_ffmpeg()

    # Build filter chain
    audio_filter = build_audio_filter_chain(
        loudness_enabled=loudness_enabled,
        loudness_target_lufs=loudness_target_lufs,
        highpass_enabled=highpass_enabled,
        highpass_freq=highpass_freq,
        noise_reduction=noise_reduction,
    )

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-threads", "4",  # Limit to P-cores, reduce contention with GPU
        "-i", str(input_path),
    ]

    # Add audio filter if any enabled
    if audio_filter:
        cmd.extend(["-af", audio_filter])

    cmd.extend([
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # Mono
        "-c:a", "pcm_s16le",  # 16-bit PCM
        str(output_path),
    ])

    # Run ffmpeg with helper (DRY refactor)
    try:
        _run_subprocess(cmd, timeout, f"ffmpeg processing {input_path}")
    except RuntimeError:
        # Clean up partial output file if ffmpeg failed
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception as cleanup_error:
                # Log cleanup failures but don't mask the original error
                logger.warning(
                    f"Failed to cleanup {output_path} after ffmpeg error: {cleanup_error}"
                )
        raise

    # Get duration using ffprobe (DRY refactor)
    duration_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(output_path),
    ]
    try:
        duration_result = _run_subprocess(duration_cmd, 30, f"ffprobe for {output_path}")
    except RuntimeError:
        # Clean up output file if ffprobe failed (file may be corrupt)
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception as cleanup_error:
                # Log cleanup failures but don't mask the original error
                logger.warning(
                    f"Failed to cleanup {output_path} after ffprobe error: {cleanup_error}"
                )
        raise

    duration_str = duration_result.stdout.strip()
    if not duration_str:
        raise RuntimeError(f"ffprobe returned empty duration for {output_path}")

    try:
        return float(duration_str)
    except ValueError:
        raise RuntimeError(f"ffprobe returned invalid duration: {duration_str}")


# Maximum audio file size to load (prevents memory exhaustion)
# 2GB of 16-bit mono 16kHz audio = ~18 hours, more than enough for any practical use
MAX_AUDIO_BYTES = 2 * 1024**3


def load_audio_samples(path: Path) -> np.ndarray:
    """Load WAV file as float32 numpy array.

    Validates file size before loading to prevent memory exhaustion.
    """
    import wave

    try:
        # Check file size before loading to prevent memory exhaustion
        file_size = path.stat().st_size
        if file_size > MAX_AUDIO_BYTES:
            raise RuntimeError(
                f"Audio file too large: {file_size / 1024**3:.1f}GB "
                f"(max {MAX_AUDIO_BYTES / 1024**3:.0f}GB). "
                "Split the file into smaller chunks first."
            )

        with wave.open(str(path), "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                raise RuntimeError(f"Audio file is empty: {path}")

            # Validate that claimed frame count doesn't exceed reasonable limits
            # based on file size (prevent malformed files from causing memory exhaustion)
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            expected_bytes = n_frames * sample_width * channels

            # Allow up to 10% overhead for WAV headers/metadata
            if expected_bytes > file_size * 1.1:
                raise RuntimeError(
                    f"WAV file appears corrupt: claims {n_frames} frames "
                    f"({expected_bytes} bytes) but file is only {file_size} bytes"
                )

            frames = wf.readframes(n_frames)

            # Validate actual bytes read matches expected
            if len(frames) != expected_bytes:
                raise RuntimeError(
                    f"WAV file read mismatch: expected {expected_bytes} bytes "
                    f"but got {len(frames)} bytes"
                )

            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            samples /= 32768.0  # Normalize to [-1, 1]
        return samples
    except wave.Error as e:
        raise RuntimeError(f"Failed to read WAV file {path}: {e}")
    except (FileNotFoundError, PermissionError, IOError) as e:
        raise RuntimeError(f"Failed to open audio file {path}: {e}")


def _get_vad_model():
    """Get or load the Silero VAD model (cached).

    Uses double-checked locking pattern for thread-safe lazy initialization.
    Caches both successful loads and failures to avoid repeated failed attempts.
    """
    global _VAD_MODEL_CACHE

    # Fast path: return cached model if available, or raise cached error
    if _VAD_MODEL_CACHE is not None:
        if isinstance(_VAD_MODEL_CACHE, Exception):
            raise _VAD_MODEL_CACHE
        return _VAD_MODEL_CACHE

    # Slow path: acquire lock and load model
    with _VAD_MODEL_LOCK:
        # Double-check after acquiring lock (another thread may have loaded it)
        if _VAD_MODEL_CACHE is not None:
            if isinstance(_VAD_MODEL_CACHE, Exception):
                raise _VAD_MODEL_CACHE
            return _VAD_MODEL_CACHE

        import torch

        try:
            # Set a timeout for model download (use environment variable as workaround)
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )

            # Validate utils structure
            if not isinstance(utils, (tuple, list)) or len(utils) < 5:
                raise RuntimeError("Silero VAD utils has unexpected structure")

            _VAD_MODEL_CACHE = (model, utils)
            return _VAD_MODEL_CACHE

        except Exception as e:
            # Cache the error to avoid repeated failed load attempts
            error = RuntimeError(f"Failed to load Silero VAD model: {e}")
            _VAD_MODEL_CACHE = error
            raise error


def detect_speech_segments(
    samples: np.ndarray,
    sample_rate: int = 16000,
    vad_config: VADConfig | None = None,
) -> list[tuple[float, float]]:
    """
    Detect speech segments using Silero VAD with tuned parameters.

    Args:
        samples: Audio samples as numpy array
        sample_rate: Sample rate in Hz (must be positive)
        vad_config: VAD configuration (uses defaults if None)

    Returns list of (start, end) tuples in seconds.
    """
    # Validate sample_rate to prevent division by zero
    # Accept int or float but ensure it's positive and non-zero
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive number, got {sample_rate}")
    sample_rate = int(sample_rate)  # Ensure integer for later calculations

    # Use defaults if no config provided
    if vad_config is None:
        vad_config = VADConfig()

    try:
        import torch
    except ImportError:
        # If torch not available, return full audio as single segment
        return [(0.0, len(samples) / sample_rate)]

    # Validate samples
    if samples.size == 0:
        return [(0.0, 0.0)]

    total_duration = len(samples) / sample_rate
    audio_tensor = None  # Initialize to None for safe cleanup

    try:
        # Load Silero VAD model (cached)
        model, utils = _get_vad_model()
        (get_speech_timestamps, _, _, _, _) = utils

        # Use Metal GPU on Apple Silicon if available
        device = "cpu"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"

        # Convert to torch tensor and move to device
        audio_tensor = torch.from_numpy(samples).to(device)

        # Move model to same device if not already (and supported)
        if device == "mps" and hasattr(model, 'to'):
            try:
                model = model.to(device)
            except Exception as e:
                # Some models don't support MPS, fall back to CPU
                logger.debug(f"MPS device transfer failed (falling back to CPU): {e}")
                audio_tensor = audio_tensor.to("cpu")

        try:
            # Get speech timestamps with tuned parameters
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                model,
                sampling_rate=sample_rate,
                threshold=vad_config.threshold,
                min_speech_duration_ms=int(vad_config.min_speech_duration * 1000),
                min_silence_duration_ms=int(vad_config.min_silence_duration * 1000),
            )
        finally:
            # Cleanup tensor to free memory (only if created)
            if audio_tensor is not None:
                del audio_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Also cleanup MPS cache on Apple Silicon
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception as e:
                    logger.debug(f"MPS cache cleanup failed (non-critical): {e}")

        # Convert to seconds
        segments = [
            (ts["start"] / sample_rate, ts["end"] / sample_rate)
            for ts in speech_timestamps
        ]

        # Merge segments that are close together
        if not segments:
            return [(0.0, total_duration)]

        merged = [segments[0]]
        for start, end in segments[1:]:
            if not merged:  # Defensive check
                merged.append((start, end))
                continue
            prev_end = merged[-1][1]
            if start - prev_end < vad_config.min_silence_duration:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        # Apply boundary padding to catch word onsets/offsets
        # This is critical for preventing chopped words at segment boundaries
        padded = []
        for start, end in merged:
            padded_start = max(0.0, start - vad_config.boundary_pad)
            padded_end = min(total_duration, end + vad_config.boundary_pad)
            padded.append((padded_start, padded_end))

        # Re-merge any segments that now overlap due to padding
        if len(padded) > 1:
            final = [padded[0]]
            for start, end in padded[1:]:
                if start <= final[-1][1]:  # Overlapping
                    final[-1] = (final[-1][0], max(final[-1][1], end))
                else:
                    final.append((start, end))
            return final

        return padded

    except Exception as e:
        # Fall back to treating entire audio as one segment
        import sys
        print(f"Warning: VAD failed ({e}), using full audio as single segment", file=sys.stderr)
        return [(0.0, total_duration)]


def _estimate_chunk_difficulty(
    samples: np.ndarray,
    sample_rate: int,
    vad_probs: list[float] | None = None,
) -> float:
    """Estimate difficulty score for an audio chunk.

    Uses RMS energy and VAD probabilities to estimate how difficult
    the audio will be to transcribe accurately.

    Args:
        samples: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        vad_probs: Optional VAD frame probabilities (0-1)

    Returns:
        Difficulty score from 0.0 (easy) to 1.0 (hard)
    """
    # RMS energy
    rms = np.sqrt(np.mean(samples ** 2))

    # If we have VAD frame probabilities, compute SNR proxy
    if vad_probs:
        # High variance in VAD = uncertain speech boundaries = harder
        vad_variance = np.var(vad_probs)
        # Low average VAD confidence = harder
        vad_avg = np.mean(vad_probs)
        difficulty = (1 - vad_avg) * 0.5 + vad_variance * 0.5
    else:
        # Use RMS-based heuristic (very quiet or very loud = harder)
        # Optimal RMS is around 0.1-0.3 for normalized audio
        optimal_rms = 0.2
        difficulty = min(1.0, abs(rms - optimal_rms) / optimal_rms)

    return float(np.clip(difficulty, 0.0, 1.0))


def _merge_small_segments(
    segments: list[tuple[float, float]],
    min_chunk_duration: float,
    max_chunk_seconds: float,
    max_gap_seconds: float = 2.0,
    soft_cap_seconds: float = 15.0,
) -> list[tuple[float, float]]:
    """Merge adjacent small segments with smart caps.

    VAD can create many small segments from conversational speech.
    This merges them into larger chunks with:
    - Soft cap (~15s): Prefer to stay under this for optimal context
    - Hard cap (20s): Never exceed this to fit in memory
    - Gap limit (2s): Don't merge across long silences (topic boundaries)

    Args:
        segments: List of (start, end) tuples from VAD
        min_chunk_duration: Merge segments shorter than this
        max_chunk_seconds: Hard cap - never exceed (typically 20s)
        max_gap_seconds: Don't merge segments with gaps larger than this
        soft_cap_seconds: Soft cap - prefer to stay under (typically 15s)

    Returns:
        Merged segment list with fewer, larger segments
    """
    if not segments or len(segments) <= 1:
        return segments

    merged = []
    current_start, current_end = segments[0]

    for next_start, next_end in segments[1:]:
        current_duration = current_end - current_start
        next_duration = next_end - next_start
        gap = next_start - current_end  # Gap between segments
        merged_duration = next_end - current_start  # Total if we merge

        # Decision logic:
        # 1. Never merge if gap is too large (likely topic boundary)
        # 2. Never merge if it would exceed hard cap
        # 3. Prefer not to merge if already at soft cap (unless next is tiny)
        # 4. Always merge if current or next is below minimum

        too_large_gap = gap > max_gap_seconds
        would_exceed_hard_cap = merged_duration > max_chunk_seconds
        at_soft_cap = current_duration >= soft_cap_seconds
        next_is_tiny = next_duration < 3.0  # Very short segment
        needs_merge = (current_duration < min_chunk_duration or
                       next_duration < min_chunk_duration)

        # Decide whether to merge
        should_merge = (
            not too_large_gap
            and not would_exceed_hard_cap
            and (needs_merge or (not at_soft_cap) or next_is_tiny)
        )

        if should_merge:
            # Extend current segment to include next
            current_end = next_end
        else:
            # Finalize current segment, start new one
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    # Don't forget the last segment
    merged.append((current_start, current_end))

    return merged


def chunk_audio(
    samples: np.ndarray,
    speech_segments: list[tuple[float, float]],
    max_chunk_seconds: float = 60.0,
    overlap_seconds: float = 2.0,
    inter_segment_overlap: float = 0.25,
    min_chunk_duration: float = 20.0,
    soft_cap_seconds: float = 15.0,
    max_gap_seconds: float = 2.0,
    sample_rate: int = 16000,
    temp_dir: Path | None = None,
) -> list[AudioChunk]:
    """
    Split audio into chunks for processing.

    Respects speech segment boundaries when possible.
    Merges small segments to reduce chunk count (GPU overhead optimization).
    Adds inter-segment overlap to prevent chopped words at VAD boundaries.

    Args:
        samples: Audio samples as numpy array
        speech_segments: List of (start, end) tuples from VAD
        max_chunk_seconds: Maximum chunk duration (hard cap)
        overlap_seconds: Overlap when splitting long segments (>max_chunk_seconds)
        inter_segment_overlap: Overlap added between consecutive VAD segments
        min_chunk_duration: Merge segments shorter than this with neighbors
        soft_cap_seconds: Prefer chunks under this duration (soft cap)
        max_gap_seconds: Don't merge across silences longer than this
        sample_rate: Sample rate in Hz
        temp_dir: Directory for temporary chunk files
    """
    # Validate sample_rate to prevent division by zero
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate}")

    # Validate max_chunk_seconds to prevent infinite loops
    if max_chunk_seconds <= 0:
        raise ValueError(f"max_chunk_seconds must be positive, got {max_chunk_seconds}")

    # Validate overlap isn't >= max_chunk_seconds (would prevent progress)
    if overlap_seconds >= max_chunk_seconds:
        raise ValueError(
            f"overlap_seconds ({overlap_seconds}) must be less than "
            f"max_chunk_seconds ({max_chunk_seconds})"
        )

    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="asr_"))
        # Track for cleanup on exit (thread-safe)
        with _TEMP_DIRS_LOCK:
            _TEMP_DIRS_TO_CLEANUP.append(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)

    # Validate input
    if not speech_segments:
        return []

    # POST-VAD MERGING: Combine small adjacent segments to reduce chunk count
    # This is critical for performance - each chunk has MLX GPU setup overhead
    merged_segments = _merge_small_segments(
        speech_segments,
        min_chunk_duration,
        max_chunk_seconds,
        max_gap_seconds=max_gap_seconds,
        soft_cap_seconds=soft_cap_seconds,
    )

    total_samples = len(samples)
    total_duration = total_samples / sample_rate
    chunks = []
    chunk_id = 0

    for i, (seg_start, seg_end) in enumerate(merged_segments):
        # Add inter-segment overlap at the start (except for first segment)
        # This helps Whisper handle words that might be cut at VAD boundaries
        if i > 0 and inter_segment_overlap > 0:
            overlap_start = max(0.0, seg_start - inter_segment_overlap)
            # Only add overlap if it doesn't significantly overlap with previous chunk
            if chunks and overlap_start < chunks[-1].start_time + chunks[-1].duration - 0.1:
                overlap_start = seg_start  # Skip overlap, segments are adjacent
            seg_start = overlap_start

        # Add inter-segment overlap at the end (except for last segment)
        if i < len(merged_segments) - 1 and inter_segment_overlap > 0:
            seg_end = min(total_duration, seg_end + inter_segment_overlap)

        seg_duration = seg_end - seg_start

        # Skip segments that became invalid due to overlap adjustments
        if seg_duration <= 0:
            continue

        # Adaptive chunk sizing: use smaller chunks for hard segments
        # Pre-estimate difficulty from segment audio to decide chunk size
        HARD_SEGMENT_THRESHOLD = 0.35
        HARD_SEGMENT_MAX_CHUNK = 30.0  # seconds (vs default 60s)

        start_sample = int(seg_start * sample_rate)
        end_sample = min(int(seg_end * sample_rate), total_samples)
        seg_samples = samples[start_sample:end_sample]
        seg_difficulty = _estimate_chunk_difficulty(seg_samples, sample_rate)

        # Use smaller chunks for difficult audio (reduces memory, improves stability)
        effective_max_chunk = max_chunk_seconds
        if seg_difficulty > HARD_SEGMENT_THRESHOLD:
            effective_max_chunk = min(max_chunk_seconds, HARD_SEGMENT_MAX_CHUNK)

        if seg_duration <= effective_max_chunk:
            # Segment fits in one chunk
            start_sample = int(seg_start * sample_rate)
            end_sample = min(int(seg_end * sample_rate), total_samples)
            chunk_samples = samples[start_sample:end_sample]

            if len(chunk_samples) == 0:
                continue  # Skip empty chunks

            chunk_path = temp_dir / f"chunk_{chunk_id:04d}.wav"
            save_wav(chunk_samples, chunk_path, sample_rate)

            # Estimate difficulty score for adaptive processing
            difficulty = _estimate_chunk_difficulty(chunk_samples, sample_rate)

            chunks.append(AudioChunk(
                samples=chunk_samples,
                start_time=seg_start,
                duration=len(chunk_samples) / sample_rate,
                path=chunk_path,
                difficulty_score=difficulty,
            ))
            chunk_id += 1
        else:
            # Split long segment with overlap
            # Validate overlap isn't larger than chunk size (would cause infinite loop)
            # Ensure at least 1 second of progress per iteration to prevent infinite loops
            min_progress = 1.0  # seconds
            effective_overlap = min(overlap_seconds, effective_max_chunk - min_progress)

            current_start = seg_start
            iteration_count = 0
            max_iterations = int((seg_end - seg_start) / min_progress) + 10  # Safety limit

            while current_start < seg_end:
                iteration_count += 1
                if iteration_count > max_iterations:
                    logger.warning(
                        f"Chunk splitting exceeded max iterations ({max_iterations}), "
                        f"stopping to prevent infinite loop. seg_start={seg_start}, "
                        f"seg_end={seg_end}, current_start={current_start}"
                    )
                    break

                current_end = min(current_start + effective_max_chunk, seg_end)

                start_sample = int(current_start * sample_rate)
                end_sample = min(int(current_end * sample_rate), total_samples)
                chunk_samples = samples[start_sample:end_sample]

                if len(chunk_samples) == 0:
                    break  # Skip empty chunks

                chunk_path = temp_dir / f"chunk_{chunk_id:04d}.wav"
                save_wav(chunk_samples, chunk_path, sample_rate)

                # Estimate difficulty score for adaptive processing
                difficulty = _estimate_chunk_difficulty(chunk_samples, sample_rate)

                chunks.append(AudioChunk(
                    samples=chunk_samples,
                    start_time=current_start,
                    duration=len(chunk_samples) / sample_rate,
                    path=chunk_path,
                    difficulty_score=difficulty,
                ))
                chunk_id += 1

                # Move forward with overlap (ensure progress to prevent infinite loop)
                next_start = current_end - effective_overlap
                # Check loop termination condition on the updated value
                if next_start >= seg_end or current_end >= seg_end:
                    break
                # Ensure we always make forward progress (at least min_progress)
                if next_start <= current_start + min_progress:
                    # Force minimum progress to prevent infinite loop
                    next_start = current_start + min_progress
                    if next_start >= seg_end:
                        break
                current_start = next_start

    return chunks


def cleanup_temp_audio():
    """Manually trigger cleanup of temporary audio directories."""
    _cleanup_temp_dirs()


def save_wav(samples: np.ndarray, path: Path, sample_rate: int = 16000) -> None:
    """Save numpy array as WAV file."""
    import wave

    # Validate inputs
    if not isinstance(samples, np.ndarray):
        raise TypeError(f"samples must be numpy array, got {type(samples)}")
    if samples.size == 0:
        raise ValueError("Cannot save empty audio samples")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive integer, got {sample_rate}")

    # Convert back to int16 with clipping to prevent overflow
    # Clip to [-1, 1] range before scaling to prevent int16 overflow
    clipped_samples = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped_samples * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())

    # SECURITY: Set restrictive permissions on audio chunk files
    path.chmod(0o600)


def prepare_audio(
    path: Path,
    config: ASRConfig,
    use_cache: bool = True,
) -> tuple[list[AudioChunk], float]:
    """
    Main entry point for audio preparation.

    Returns (chunks, total_duration).
    """
    check_ffmpeg()

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Check cache
    file_hash = get_file_hash(path)
    cache_dir = config.cache_dir / file_hash
    normalized_path = cache_dir / "normalized.wav"

    if use_cache and normalized_path.exists():
        duration = get_audio_duration(normalized_path)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # SECURITY: Set restrictive permissions on cache directory
        cache_dir.chmod(0o700)
        # Apply audio enhancement filters from config
        ae = config.audio_enhancement
        duration = normalize_audio(
            path,
            normalized_path,
            loudness_enabled=ae.loudness_enabled,
            loudness_target_lufs=ae.loudness_target_lufs,
            highpass_enabled=ae.highpass_enabled,
            highpass_freq=ae.highpass_freq,
            noise_reduction=ae.noise_reduction,
        )

    # Load samples
    samples = load_audio_samples(normalized_path)

    # Detect speech if VAD enabled
    if config.use_vad:
        speech_segments = detect_speech_segments(samples, vad_config=config.vad)
    else:
        speech_segments = [(0.0, duration)]

    # Create chunks with post-VAD merging for fewer, larger chunks
    # This is critical for performance - MLX has setup overhead per chunk
    chunks = chunk_audio(
        samples=samples,
        speech_segments=speech_segments,
        max_chunk_seconds=config.max_chunk_seconds,
        overlap_seconds=config.chunk_overlap_seconds,
        inter_segment_overlap=config.vad.inter_segment_overlap,
        min_chunk_duration=config.vad.min_chunk_duration,
        soft_cap_seconds=config.vad.soft_cap_seconds,
        max_gap_seconds=config.vad.max_gap_seconds,
        temp_dir=cache_dir / "chunks",
    )

    return chunks, duration

"""CrisperWhisper MLX backend - single-model, word-precision focused.

CrisperWhisper is a Whisper Large v3 fine-tuned for word-by-word precision,
making it ideal for transcription where exact words matter (interviews,
dictation, voice automation, etc.)

Model: kyr0/crisperwhisper-unsloth-mlx
License: CC-BY-NC-4.0 (non-commercial)
"""

import logging
import math
import threading
from pathlib import Path

from asr.audio.ingest import AudioChunk
from asr.backends.base import BaseEngine
from asr.logging import get_logger
from asr.models.transcript import CorrectionInfo, Segment, WordTiming

logger = logging.getLogger(__name__)

# Global lock to serialize MLX GPU operations
# Metal command buffers are not thread-safe for concurrent transcriptions
_MLX_LOCK = threading.Lock()

# Whisper transcription parameters (tuned for CrisperWhisper accuracy)
TEMPERATURE = 0.0  # Deterministic, most accurate
HALLUCINATION_SILENCE_THRESHOLD = 0.5  # Filter hallucinations in silence
COMPRESSION_RATIO_THRESHOLD = 2.4  # Detect repetitive hallucinations
LOGPROB_THRESHOLD = -1.0  # Confidence cutoff for output
NO_SPEECH_THRESHOLD = 0.6  # Silence detection threshold

# Model path resolution order:
# 1. ~/.asr/models/crisperwhisper (user data directory, preferred)
# 2. ./models/crisperwhisper (local dev fallback)
def _resolve_model_path() -> Path:
    """Resolve CrisperWhisper model path with fallback."""
    user_path = Path.home() / ".asr" / "models" / "crisperwhisper"
    if user_path.exists():
        return user_path
    local_path = Path.cwd() / "models" / "crisperwhisper"
    if local_path.exists():
        return local_path
    # Default to user path even if missing (will fail gracefully in is_available)
    return user_path


DEFAULT_MODEL_PATH = _resolve_model_path()


class CrisperWhisperEngine(BaseEngine):
    """CrisperWhisper engine optimized for M4 MacBook Air.

    This model is fine-tuned for word-by-word precision, making it ideal
    for transcription where exact words matter (interviews, dictation, etc.)

    Memory configuration tuned for 24GB unified memory:
    - 21GB MLX limit (allows headroom for OS lazy mapping)
    - 3GB tensor cache (sufficient, reduces fragmentation)
    """

    def __init__(self, model_path: Path | str | None = None, warm_up: bool = False):
        """Initialize the CrisperWhisper engine.

        Args:
            model_path: Path to local model directory. If None, resolves in order:
                       1. ~/.asr/models/crisperwhisper (preferred)
                       2. ./models/crisperwhisper (local fallback)
            warm_up: Pre-compile Metal kernels to reduce first-chunk latency
        """
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._configure_mlx_memory()

        if warm_up:
            self._warm_up_model()

    def _warm_up_model(self) -> None:
        """Pre-compile Metal kernels to eliminate first-call latency.

        Runs a minimal transcription (1 second silence) to trigger JIT
        compilation of Metal shaders. This moves the ~1s startup cost
        from the first real transcription to initialization.
        """
        import mlx_whisper
        import numpy as np

        try:
            # 1 second of silence triggers kernel compilation without output
            dummy_audio = np.zeros(16000, dtype=np.float32)
            with _MLX_LOCK:
                mlx_whisper.transcribe(
                    dummy_audio,
                    path_or_hf_repo=str(self._model_path),
                    language="en",
                    word_timestamps=False,
                    verbose=False,
                )
        except Exception as e:
            logger.debug(f"Warm-up failed (non-critical): {e}")

    def _configure_mlx_memory(self):
        """Configure MLX for M4 24GB: Optimized for CrisperWhisper Large v3.

        Working set breakdown:
        - Model weights (FP16): ~1.5GB
        - Activations: ~2-3GB
        - KV cache (60s chunks): ~800MB
        - Scratch/intermediate: ~1-2GB
        - Total working set: ~5.3-7.3GB

        Memory strategy:
        - 21GB limit: Allows headroom for OS lazy mapping
        - 3GB cache: Sufficient for intermediates, reduces fragmentation
        """
        try:
            import mlx.core as mx
            # Allow MLX to use more memory (OS uses lazy mapping)
            mx.set_memory_limit(21 * 1024**3)
            # Reduce cache to minimize fragmentation (3GB sufficient)
            mx.set_cache_limit(3 * 1024**3)
        except Exception as e:
            logger.debug(f"MLX memory config not available: {e}")

    @property
    def name(self) -> str:
        return "crisperwhisper"

    def is_available(self) -> bool:
        """Check if CrisperWhisper is available (MLX installed + model exists)."""
        try:
            import mlx.core  # noqa: F401
            import mlx_whisper  # noqa: F401
            return self._model_path.exists()
        except ImportError:
            return False

    def transcribe(
        self,
        audio: AudioChunk,
        word_timestamps: bool,
        language: str | None,
        initial_prompt: str | None = None,
    ) -> list[Segment]:
        """Transcribe an audio chunk using CrisperWhisper FP16.

        Args:
            audio: Audio chunk to transcribe
            word_timestamps: Include word-level timestamps with confidence
            language: Language code (e.g., "en") or None for auto-detect
            initial_prompt: Context prompt with names, vocabulary, domain hints

        Returns:
            List of transcribed segments
        """
        import mlx_whisper

        # Build transcription kwargs
        kwargs = dict(
            path_or_hf_repo=str(self._model_path),
            language=language,
            word_timestamps=word_timestamps,
            temperature=TEMPERATURE,
            condition_on_previous_text=True,  # Use context from previous segments
            hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=LOGPROB_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            initial_prompt=initial_prompt,
        )

        # Use lock to serialize Metal GPU operations
        # Pass numpy array directly to avoid disk I/O (mlx_whisper supports np.ndarray)
        try:
            with _MLX_LOCK:
                result = mlx_whisper.transcribe(audio.samples, **kwargs)
        finally:
            # Clean up MLX memory cache to prevent memory leaks
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except Exception as e:
                logger.debug(f"MLX cache cleanup failed (non-critical): {e}")

        # Generate chunk_id from start_time for debugging (stable hash within int range)
        chunk_id = hash(audio.start_time) % (2**31)
        return self._parse_result(result, audio.start_time, word_timestamps, chunk_id=chunk_id)

    def transcribe_file(
        self,
        audio_path: Path,
        word_timestamps: bool,
        language: str | None,
        initial_prompt: str | None = None,
    ) -> list[Segment]:
        """Transcribe an entire audio file (for VAD-disabled mode).

        Args:
            audio_path: Path to audio file
            word_timestamps: Include word-level timestamps
            language: Language code or None for auto-detect
            initial_prompt: Context prompt

        Returns:
            List of segments covering the entire file
        """
        import mlx_whisper

        kwargs = dict(
            path_or_hf_repo=str(self._model_path),
            language=language,
            word_timestamps=word_timestamps,
            temperature=TEMPERATURE,
            condition_on_previous_text=True,
            hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=LOGPROB_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            initial_prompt=initial_prompt,
        )

        try:
            with _MLX_LOCK:
                result = mlx_whisper.transcribe(str(audio_path), **kwargs)
        finally:
            # Clean up MLX memory cache to prevent memory leaks
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except Exception as e:
                logger.debug(f"MLX cache cleanup failed (non-critical): {e}")

        return self._parse_result(result, 0.0, word_timestamps)

    def _parse_result(
        self,
        result: dict,
        start_offset: float,
        word_timestamps: bool,
        chunk_id: int | None = None,
    ) -> list[Segment]:
        """Parse mlx_whisper result into Segment models.

        Args:
            result: mlx_whisper transcription result
            start_offset: Time offset to add to all timestamps
            word_timestamps: Whether word-level timestamps are included
            chunk_id: Optional chunk ID for debugging VAD pathology
        """
        segments = []

        for i, seg in enumerate(result.get("segments", [])):
            # Parse word-level timestamps if available
            words = []
            if word_timestamps and "words" in seg:
                for word_data in seg["words"]:
                    if not word_data:
                        continue
                    try:
                        # Safely extract word data with type coercion and validation
                        raw_word = word_data.get("word")
                        word_text = str(raw_word) if raw_word is not None else ""

                        # Validate and convert timing values
                        raw_start = word_data.get("start", 0)
                        raw_end = word_data.get("end", 0)
                        raw_prob = word_data.get("probability", 0.9)

                        if raw_start is None or raw_end is None:
                            logger.debug(f"Skipping word with missing timing: {word_text}")
                            continue

                        word_start = float(raw_start)
                        word_end = float(raw_end)
                        word_prob = float(raw_prob) if raw_prob is not None else 0.9

                        # Validate timing makes sense
                        if word_start < 0 or word_end < 0 or word_end < word_start:
                            logger.debug(
                                f"Skipping word with invalid timing: {word_text} "
                                f"({word_start}-{word_end})"
                            )
                            continue

                        words.append(WordTiming(
                            word=word_text,
                            start=start_offset + word_start,
                            end=start_offset + word_end,
                            confidence=min(1.0, max(0.0, word_prob)),
                        ))
                    except (ValueError, TypeError) as e:
                        # Skip malformed word data but don't fail the entire transcription
                        logger.debug(f"Skipping malformed word data: {e}")

            text = str(seg.get("text", "") or "").strip()

            # Safe extraction with type coercion for segment timing
            try:
                raw_seg_start = seg.get("start", 0)
                raw_seg_end = seg.get("end", 0)
                seg_start = float(raw_seg_start if raw_seg_start is not None else 0)
                seg_end = float(raw_seg_end if raw_seg_end is not None else 0)

                # Validate segment timing makes sense
                if seg_start < 0:
                    logger.debug(f"Invalid segment start time {seg_start}, using 0")
                    seg_start = 0.0
                if seg_end < seg_start:
                    logger.debug(
                        f"Invalid segment end time {seg_end} < start {seg_start}, using start"
                    )
                    seg_end = seg_start
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse segment timing, using defaults: {e}")
                seg_start = 0.0
                seg_end = 0.0

            # Safe confidence calculation from avg_logprob
            # avg_logprob is log probability, so exp() converts to probability [0, 1]
            avg_logprob = seg.get("avg_logprob")
            if avg_logprob is None or not isinstance(avg_logprob, (int, float)):
                avg_logprob = -0.5
            try:
                # Clamp logprob to reasonable range to avoid overflow
                clamped_logprob = max(-10.0, min(0.0, float(avg_logprob)))
                confidence = min(1.0, max(0.0, math.exp(clamped_logprob)))
            except (ValueError, OverflowError) as e:
                logger.debug(f"Failed to calculate confidence from logprob, using default: {e}")
                confidence = 0.5

            segments.append(Segment(
                id=i,
                chunk_id=chunk_id,
                start=start_offset + seg_start,
                end=start_offset + seg_end,
                speaker=None,
                raw_text=text,
                text=text,
                confidence=confidence,
                words=words,
                corrections=CorrectionInfo(),
            ))

        # Log chunk transcription with confidence stats
        self._log_transcription(segments, start_offset)

        return segments

    def _log_transcription(self, segments: list[Segment], start_offset: float) -> None:
        """Log transcription stats to session logger."""
        logger = get_logger()
        if not logger or not segments:
            return

        # Collect all words from all segments for this chunk
        all_words = []
        for seg in segments:
            if seg.words:
                for word in seg.words:
                    all_words.append({
                        "word": word.word,
                        "probability": word.confidence,
                    })

        if all_words and segments:
            # Calculate duration from segments
            duration = segments[-1].end - start_offset
            # Use hash of start_time to create stable chunk_id within int range
            chunk_id = hash(start_offset) % (2**31)
            logger.log_chunk_transcribed(
                chunk_id=chunk_id,
                words=all_words,
                start_time=start_offset,
                duration=duration,
            )


# Singleton instance with thread-safe initialization
_engine: CrisperWhisperEngine | None = None
_engine_lock = threading.Lock()


def get_crisperwhisper_engine(
    model_path: Path | str | None = None,
    warm_up: bool = False,
) -> CrisperWhisperEngine:
    """Get a CrisperWhisper engine instance (singleton, thread-safe).

    Args:
        model_path: Optional override for model path (only used on first call)
        warm_up: Pre-compile Metal kernels (only affects first call)

    Returns:
        CrisperWhisperEngine instance

    Note:
        The singleton pattern means model_path is only used on the first call.
        Subsequent calls with different model_path will return the existing engine.
        This is by design to avoid reloading the model on every call.
    """
    global _engine

    # Fast path: return cached engine if available
    if _engine is not None:
        # Warn if model_path is specified but differs from cached engine
        if model_path is not None:
            requested_path = Path(model_path)
            if requested_path != _engine._model_path:
                logger.warning(
                    f"Ignoring model_path {requested_path} - engine already initialized "
                    f"with {_engine._model_path}. Singleton pattern prevents reloading."
                )
        return _engine

    # Slow path: acquire lock and create engine
    with _engine_lock:
        # Double-check after acquiring lock (another thread may have created it)
        if _engine is None:
            _engine = CrisperWhisperEngine(model_path, warm_up=warm_up)
        return _engine

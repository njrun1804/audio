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
        with _MLX_LOCK:
            result = mlx_whisper.transcribe(audio.samples, **kwargs)

        return self._parse_result(result, audio.start_time, word_timestamps)

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

        with _MLX_LOCK:
            result = mlx_whisper.transcribe(str(audio_path), **kwargs)

        return self._parse_result(result, 0.0, word_timestamps)

    def _parse_result(
        self,
        result: dict,
        start_offset: float,
        word_timestamps: bool,
    ) -> list[Segment]:
        """Parse mlx_whisper result into Segment models."""
        segments = []

        for i, seg in enumerate(result.get("segments", [])):
            # Parse word-level timestamps if available
            words = []
            if word_timestamps and "words" in seg:
                for word_data in seg["words"]:
                    if not word_data:
                        continue
                    # Safely extract word data with type coercion
                    word_text = str(word_data.get("word", "")) if word_data.get("word") is not None else ""
                    word_start = float(word_data.get("start", 0))
                    word_end = float(word_data.get("end", 0))
                    word_prob = float(word_data.get("probability", 0.9))
                    words.append(WordTiming(
                        word=word_text,
                        start=start_offset + word_start,
                        end=start_offset + word_end,
                        confidence=min(1.0, max(0.0, word_prob)),
                    ))

            text = str(seg.get("text", "") or "").strip()

            # Safe extraction with type coercion for segment timing
            seg_start = float(seg.get("start", 0) or 0)
            seg_end = float(seg.get("end", 0) or 0)

            # Safe confidence calculation from avg_logprob
            # avg_logprob is log probability, so exp() converts to probability [0, 1]
            avg_logprob = seg.get("avg_logprob")
            if avg_logprob is None or not isinstance(avg_logprob, (int, float)):
                avg_logprob = -0.5
            # Clamp logprob to reasonable range to avoid overflow
            clamped_logprob = max(-10.0, min(0.0, float(avg_logprob)))
            confidence = min(1.0, max(0.0, math.exp(clamped_logprob)))

            segments.append(Segment(
                id=i,
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

        if all_words:
            # Calculate duration from segments
            duration = segments[-1].end - start_offset if segments else 0
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
        model_path: Optional override for model path
        warm_up: Pre-compile Metal kernels (only affects first call)

    Returns:
        CrisperWhisperEngine instance
    """
    global _engine

    # Fast path: return cached engine if available
    if _engine is not None:
        return _engine

    # Slow path: acquire lock and create engine
    with _engine_lock:
        # Double-check after acquiring lock (another thread may have created it)
        if _engine is None:
            _engine = CrisperWhisperEngine(model_path, warm_up=warm_up)
        return _engine

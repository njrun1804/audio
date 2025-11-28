"""Abstract base class for transcription backends."""

from abc import ABC, abstractmethod
from pathlib import Path

from asr.audio.ingest import AudioChunk
from asr.models.transcript import Segment


class BaseEngine(ABC):
    """Abstract base class for ASR engines."""

    @abstractmethod
    def transcribe(
        self,
        audio: AudioChunk,
        word_timestamps: bool,
        language: str | None,
        initial_prompt: str | None = None,
    ) -> list[Segment]:
        """
        Transcribe an audio chunk.

        Args:
            audio: Audio chunk to transcribe
            word_timestamps: Whether to include word-level timestamps
            language: Language code (e.g., "en") or None for auto-detect
            initial_prompt: Context prompt with names, vocabulary hints

        Returns:
            List of transcribed segments
        """
        ...

    @abstractmethod
    def transcribe_file(
        self,
        audio_path: Path,
        word_timestamps: bool,
        language: str | None,
        initial_prompt: str | None = None,
    ) -> list[Segment]:
        """Transcribe an entire audio file.

        Alternative entry point for file-based inputs, typically used
        when VAD is disabled.

        Args:
            audio_path: Path to audio file
            word_timestamps: Include word-level timestamps
            language: Language code or None for auto-detect
            initial_prompt: Context prompt

        Returns:
            List of segments covering the entire file
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for display."""
        ...

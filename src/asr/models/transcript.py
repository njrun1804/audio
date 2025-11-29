"""Pydantic models for ASR transcript output."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class WordTiming(BaseModel):
    """Word-level timing information."""

    word: str
    start: float
    end: float
    confidence: float = Field(ge=0.0, le=1.0)


class CorrectionInfo(BaseModel):
    """Information about ASR error corrections applied."""

    applied: bool = False
    source: Literal["local", "claude", "none"] = "none"
    changes: list[str] = Field(default_factory=list)


class Segment(BaseModel):
    """A transcribed segment of audio.

    Note: The `words` array contains word-level timestamps from the original
    Whisper transcription. After correction, `text` is updated but `words`
    remains unchanged (pre-correction). Segment-level timestamps (`start`/`end`)
    are accurate; word-level timestamps may not align with corrected text.
    """

    id: int
    start: float
    end: float
    chunk_id: int | None = None  # Source chunk for debugging VAD pathology
    speaker: str | None = None
    raw_text: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    words: list[WordTiming] = Field(default_factory=list)
    corrections: CorrectionInfo = Field(default_factory=CorrectionInfo)


class TranscriptConfig(BaseModel):
    """Configuration used for transcription."""

    model: str = "crisperwhisper"
    backend: Literal["mlx"] = "mlx"
    quantization: str = "fp16"  # CrisperWhisper is FP16 only
    language: str = "en"


class TranscriptMetadata(BaseModel):
    """Optional metadata extracted from transcript."""

    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)


class Transcript(BaseModel):
    """Complete transcript output."""

    version: str = "1.0"
    audio_path: str
    duration_seconds: float
    created_at: datetime = Field(default_factory=datetime.now)
    config: TranscriptConfig
    segments: list[Segment]
    metadata: TranscriptMetadata = Field(default_factory=TranscriptMetadata)

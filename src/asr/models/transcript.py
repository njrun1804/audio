"""Pydantic models for ASR transcript output."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WordTiming(BaseModel):
    """Word-level timing information."""

    word: str
    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(ge=0.0, description="End time in seconds")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")

    @field_validator("end")
    @classmethod
    def validate_end_after_start(cls, v: float, info) -> float:
        """Ensure end time is after start time."""
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end time must be >= start time")
        return v


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

    id: int = Field(ge=0, description="Segment ID")
    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(ge=0.0, description="End time in seconds")
    chunk_id: int | None = Field(default=None, description="Source chunk for VAD debugging")
    speaker: str | None = Field(default=None, min_length=1, description="Speaker name if known")
    raw_text: str = Field(description="Original ASR output text")
    text: str = Field(description="Corrected text (or raw_text if no correction)")
    confidence: float = Field(ge=0.0, le=1.0, description="Average confidence score (0.0-1.0)")
    words: list[WordTiming] = Field(default_factory=list, description="Word-level timing info")
    corrections: CorrectionInfo = Field(
        default_factory=lambda: CorrectionInfo(), description="Correction metadata"
    )

    @field_validator("end")
    @classmethod
    def validate_end_after_start(cls, v: float, info) -> float:
        """Ensure end time is after start time."""
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end time must be >= start time")
        return v

    @field_validator("speaker")
    @classmethod
    def validate_speaker_not_empty(cls, v: str | None) -> str | None:
        """Ensure speaker name is not empty string."""
        if v is not None and not v.strip():
            return None
        return v


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

    model_config = ConfigDict(
        # Enable JSON serialization mode for datetime objects
        ser_json_timedelta="iso8601",
        # Validate default values
        validate_default=True,
        # Validate assignments after creation
        validate_assignment=True,
    )

    version: str = Field(default="1.0", min_length=1, description="Transcript schema version")
    audio_path: str = Field(min_length=1, description="Path to source audio file")
    duration_seconds: float = Field(gt=0.0, description="Total audio duration in seconds")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Transcript creation timestamp"
    )
    config: TranscriptConfig = Field(description="Transcription configuration used")
    segments: list[Segment] = Field(description="Transcribed segments")
    metadata: TranscriptMetadata = Field(
        default_factory=lambda: TranscriptMetadata(), description="Optional metadata"
    )

    @field_validator("audio_path")
    @classmethod
    def validate_audio_path_not_empty(cls, v: str) -> str:
        """Ensure audio path is not empty or whitespace."""
        if not v.strip():
            raise ValueError("audio_path cannot be empty")
        return v

    @field_validator("segments")
    @classmethod
    def validate_segments_not_empty(cls, v: list[Segment]) -> list[Segment]:
        """Warn if segments list is empty (likely an error)."""
        # Allow empty list but could add warning in future
        return v

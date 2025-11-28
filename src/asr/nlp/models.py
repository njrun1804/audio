"""Models for ASR correction pipeline."""

from typing import Literal

from pydantic import BaseModel, Field


class CorrectionChange(BaseModel):
    """A single correction change."""

    original: str
    corrected: str
    reason: str


class SegmentCorrection(BaseModel):
    """Correction result for a single segment."""

    id: int
    corrected: str
    changes: list[CorrectionChange] = Field(default_factory=list)


class Pass1Result(BaseModel):
    """Result from Pass 1 (correction)."""

    corrected_segments: list[SegmentCorrection] = Field(default_factory=list)


class ConsistencyFix(BaseModel):
    """A consistency fix from Pass 2."""

    segment_id: int
    original: str
    corrected: str
    reason: str


class Pass2Result(BaseModel):
    """Result from Pass 2 (consistency)."""

    consistency_fixes: list[ConsistencyFix] = Field(default_factory=list)
    entity_map: dict[str, list[str]] = Field(default_factory=dict)
    flags: list[dict] = Field(default_factory=list)


class CorrectionConfig(BaseModel):
    """Configuration for the correction pipeline.

    Tuned for CrisperWhisper's verbatim transcription output.
    Uses Sonnet 4.5 with extended thinking for best accuracy.
    """

    model: str = "claude-sonnet-4-5-20250929"
    passes: Literal[1, 2] = 2
    vocabulary: list[str] = Field(default_factory=list)
    domain: str | None = None
    # Conservative by default - CrisperWhisper is already accurate
    aggressiveness: Literal["conservative", "moderate", "aggressive"] = "conservative"
    batch_size: int = 5  # Segments per API call

    # Safety settings - tuned for CrisperWhisper's higher accuracy
    # Research consensus: 0.70 is optimal for Whisper word-level confidence
    # Fragment grouping in prompts.py handles proper noun detection
    low_confidence_threshold: float = 0.70
    context_window: int = 5  # Segments before/after for context
    strict_diff_gating: bool = True  # Reject changes outside low_conf regions

    # API behavior (temperature only used when NOT using extended thinking)
    temperature: float = 0.0  # Deterministic output

    # Extended thinking - always enabled for best accuracy
    use_extended_thinking: bool = True  # Let Claude reason before correcting
    thinking_budget_tokens: int = 4096  # Token budget for thinking (min 1024)

    # Show detailed word confidence scores in prompt (not just <low_conf> tags)
    show_word_confidences: bool = True  # Include numeric confidence per word


class CorrectionResult(BaseModel):
    """Full result from the correction pipeline."""

    passes_completed: int
    total_changes: int
    changes_by_segment: dict[int, list[CorrectionChange]] = Field(default_factory=dict)
    entity_map: dict[str, list[str]] = Field(default_factory=dict)
    flags: list[dict] = Field(default_factory=list)

"""Models for ASR correction pipeline."""

from typing import Literal

from pydantic import BaseModel, Field


class CorrectionChange(BaseModel):
    """A single correction change."""

    original: str = Field(..., description="Original text before correction")
    corrected: str = Field(..., description="Corrected text")
    reason: str = Field(..., description="Reason for the correction")


class SegmentCorrection(BaseModel):
    """Correction result for a single segment."""

    id: int = Field(..., description="Segment ID from the original transcript")
    corrected: str = Field(..., description="Corrected text for this segment")
    changes: list[CorrectionChange] = Field(default_factory=list, description="Changes made")


class Pass1Result(BaseModel):
    """Result from Pass 1 (correction)."""

    corrected_segments: list[SegmentCorrection] = Field(default_factory=list, description="List of corrected segments")


class ConsistencyFix(BaseModel):
    """A consistency fix from Pass 2."""

    segment_id: int = Field(..., description="ID of the segment to fix")
    original: str = Field(..., description="Original text before fix")
    corrected: str = Field(..., description="Corrected text")
    reason: str = Field(..., description="Reason for consistency fix")


class Pass2Result(BaseModel):
    """Result from Pass 2 (consistency)."""

    consistency_fixes: list[ConsistencyFix] = Field(default_factory=list, description="List of consistency fixes applied")
    entity_map: dict[str, list[str]] = Field(default_factory=dict, description="Map of canonical entities to variants found")
    flags: list[dict] = Field(default_factory=list, description="Quality flags for manual review")


class KickerFix(BaseModel):
    """A fix from the final kicker pass."""

    segment_id: int = Field(..., description="ID of the segment to fix")
    original: str = Field(..., description="Original text before fix")
    corrected: str = Field(..., description="Corrected text")
    reason: str = Field(..., description="Reason for the fix")
    confidence: Literal["low", "medium", "high"] = Field(default="high", description="Confidence level of the fix")


class KickerResult(BaseModel):
    """Result from the final kicker pass (Sonnet/Opus with thinking)."""

    final_fixes: list[KickerFix] = Field(default_factory=list, description="List of final fixes from kicker pass")
    quality_assessment: str = Field(default="", description="Overall quality assessment of the transcript")


class CorrectionConfig(BaseModel):
    """Configuration for the correction pipeline.

    Architecture:
    - Pass 1-2: Haiku 4.5 (fast, no thinking) for bulk corrections
    - Final kicker: Sonnet 4.5 WITH thinking for polish (optional but recommended)

    This gives speed + accuracy: fast passes catch obvious errors,
    final thinking pass catches subtle issues with full context.
    """

    # Fast model: Haiku 4.5 for passes 1-2 (pattern matching, spelling)
    model: str = "claude-haiku-4-5-20251001"
    passes: Literal[1, 2] = 2

    # Kicker model: Final polish pass with thinking (full transcript context)
    kicker_model: str = "claude-sonnet-4-5-20250929"  # Or claude-opus-4-5-20250430 for max
    use_kicker: bool = True  # Run final thinking pass for best accuracy
    kicker_thinking_budget: int = Field(
        default=8192, ge=1024, le=16384, description="Token budget for kicker thinking"
    )
    vocabulary: list[str] = Field(default_factory=list, description="Custom vocabulary terms")
    domain: str | None = Field(default=None, description="Domain context (biography, tech, etc.)")
    # Conservative by default - CrisperWhisper is already accurate
    aggressiveness: Literal["conservative", "moderate", "aggressive"] = "conservative"
    batch_size: int = Field(default=100, ge=1, le=1000, description="Segments per API call")

    # Safety settings - tuned for CrisperWhisper's higher accuracy
    # Research consensus: 0.70 is optimal for Whisper word-level confidence
    # Fragment grouping in prompts.py handles proper noun detection
    low_confidence_threshold: float = Field(
        default=0.70, ge=0.0, le=1.0, description="Threshold for marking low-confidence words"
    )
    context_window: int = Field(default=2, ge=0, le=10, description="Segments before/after")
    strict_diff_gating: bool = True  # Reject changes outside low_conf regions

    # API behavior
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")

    # Extended thinking - disabled for Haiku 4.5 (ASR correction is pattern matching)
    # Enable for complex reasoning tasks, but adds significant latency
    use_extended_thinking: bool = False
    thinking_budget_tokens: int = Field(
        default=1024, ge=256, le=16384, description="Token budget for thinking"
    )

    # Show detailed word confidence scores in prompt (not just <low_conf> tags)
    # Disabled: <low_conf> tags are sufficient for Claude with extended thinking
    show_word_confidences: bool = False


class CorrectionResult(BaseModel):
    """Full result from the correction pipeline."""

    passes_completed: int = Field(default=0, ge=0, le=3, description="Number of passes (0-3)")
    total_changes: int = Field(default=0, ge=0, description="Total number of changes made")
    changes_by_segment: dict[int, list[CorrectionChange]] = Field(
        default_factory=dict, description="Changes grouped by segment ID"
    )
    entity_map: dict[str, list[str]] = Field(
        default_factory=dict, description="Map of canonical entities to variants"
    )
    flags: list[dict] = Field(default_factory=list, description="Quality flags from Pass 2")

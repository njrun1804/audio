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


class KickerFix(BaseModel):
    """A fix from the final kicker pass."""

    segment_id: int
    original: str
    corrected: str
    reason: str
    confidence: str = "high"


class KickerResult(BaseModel):
    """Result from the final kicker pass (Sonnet/Opus with thinking)."""

    final_fixes: list[KickerFix] = Field(default_factory=list)
    quality_assessment: str = ""


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
    kicker_thinking_budget: int = 8192  # Generous thinking for final pass
    vocabulary: list[str] = Field(default_factory=list)
    domain: str | None = None
    # Conservative by default - CrisperWhisper is already accurate
    aggressiveness: Literal["conservative", "moderate", "aggressive"] = "conservative"
    batch_size: int = 100  # Segments per API call - larger = fewer calls, more context

    # Safety settings - tuned for CrisperWhisper's higher accuracy
    # Research consensus: 0.70 is optimal for Whisper word-level confidence
    # Fragment grouping in prompts.py handles proper noun detection
    low_confidence_threshold: float = 0.70
    context_window: int = 2  # Segments before/after for context (was 5 - redundant in batches)
    strict_diff_gating: bool = True  # Reject changes outside low_conf regions

    # API behavior
    temperature: float = 0.0  # Deterministic output

    # Extended thinking - disabled for Haiku 4.5 (ASR correction is pattern matching)
    # Enable for complex reasoning tasks, but adds significant latency
    use_extended_thinking: bool = False
    thinking_budget_tokens: int = 1024  # Minimal if enabled

    # Show detailed word confidence scores in prompt (not just <low_conf> tags)
    # Disabled: <low_conf> tags are sufficient for Claude with extended thinking
    show_word_confidences: bool = False


class CorrectionResult(BaseModel):
    """Full result from the correction pipeline."""

    passes_completed: int
    total_changes: int
    changes_by_segment: dict[int, list[CorrectionChange]] = Field(default_factory=dict)
    entity_map: dict[str, list[str]] = Field(default_factory=dict)
    flags: list[dict] = Field(default_factory=list)

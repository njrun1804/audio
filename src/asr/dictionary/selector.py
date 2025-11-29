"""Context-aware bias list selection for ASR dictionary.

This module provides BiasListSelector for selecting and ranking dictionary
entries based on context, tier, recency, and other factors.

Context profiles stored at: ~/.asr/dictionaries/contexts/{name}.json
"""

import json
import logging
from datetime import datetime

from pydantic import ValidationError

from asr.config import CONFIG_DIR, _file_lock
from asr.dictionary.db import (
    get_all_entries,
    get_entries_by_context,
    get_recent_entries,
)
from asr.dictionary.models import (
    ContextProfile,
    EntryWithRelations,
    TierLevel,
    TIER_WEIGHTS,
)

logger = logging.getLogger(__name__)

# Storage paths
DICTIONARIES_DIR = CONFIG_DIR / "dictionaries"
CONTEXTS_DIR = DICTIONARIES_DIR / "contexts"

# Tier ordering for comparisons
TIER_ORDER: list[TierLevel] = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _ensure_dirs() -> None:
    """Ensure dictionary storage directories exist."""
    DICTIONARIES_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)


# Context keyword mappings for auto-detection
# Maps context name -> list of keywords that trigger that context
CONTEXT_KEYWORDS: dict[str, list[str]] = {
    "running": [
        "strava", "running", "marathon", "garmin", "pace", "splits",
        "tempo", "interval", "fartlek", "cadence", "vo2max", "threshold",
        "easy run", "long run", "track", "5k", "10k", "half marathon",
        "ultra", "trail", "race", "pr", "personal record", "mileage",
    ],
    "work": [
        "veeva", "vault", "gxp", "capa", "deviation", "qms", "quality",
        "regulatory", "compliance", "validation", "audit", "sop",
        "change control", "batch record", "clinical", "pharma",
        "fda", "ema", "ich", "gmp", "glp", "gcp",
    ],
    "asr_dev": [
        "whisper", "mlx", "claude", "transcription", "asr", "speech",
        "recognition", "diarization", "vad", "voice activity",
        "crisperwhisper", "anthropic", "openai", "huggingface",
        "transformer", "attention", "encoder", "decoder", "mel",
        "spectrogram", "wer", "word error rate", "ctc", "rnnt",
    ],
    "cycling": [
        "zwift", "peloton", "watts", "ftp", "power meter", "cadence",
        "trainer", "indoor cycling", "spin", "bike", "bicycle",
        "climbing", "descent", "gear", "derailleur", "cassette",
    ],
    "tech": [
        "python", "typescript", "javascript", "rust", "golang",
        "kubernetes", "docker", "aws", "gcp", "azure", "terraform",
        "api", "rest", "graphql", "database", "postgres", "redis",
        "github", "gitlab", "cicd", "devops", "microservice",
    ],
    "medical": [
        "diagnosis", "treatment", "medication", "prescription",
        "symptoms", "patient", "doctor", "hospital", "clinic",
        "surgery", "therapy", "chronic", "acute", "mg", "ml",
    ],
    "finance": [
        "stock", "bond", "investment", "portfolio", "dividend",
        "market", "trading", "equity", "fund", "etf", "ira", "401k",
        "capital", "roi", "yield", "interest", "mortgage",
    ],
}


class BiasListSelector:
    """Select and rank dictionary entries for ASR bias lists.

    The selector combines multiple signals to rank entries:
    - Tier priority (critical/standard/low)
    - Boost weight (manual importance)
    - Context relevance
    - Recency (recently used entries rank higher)

    Scoring formula:
        score = boost_weight * tier_weight * recency_decay * context_match

    Where:
        - tier_weight: A=1.0 (critical), B-D=0.7 (standard), E-H=0.4 (low)
        - recency_decay: 1.0 if seen in 60 days, 0.5 otherwise
        - context_match: 1.5 if entry has matching context tag, 1.0 otherwise
    """

    def __init__(self):
        """Initialize the selector.

        Uses SQLite database directly for entry storage.
        Context profiles are still loaded from JSON config files.
        """
        _ensure_dirs()

    # =========================================================================
    # Context Profile Management
    # =========================================================================

    def load_context_profile(self, name: str) -> ContextProfile | None:
        """Load a context profile by name.

        Args:
            name: Profile name (without .json extension)

        Returns:
            ContextProfile if found, None otherwise
        """
        if not name or len(name) > 50:
            return None

        # Validate name for safety
        if not all(c.isalnum() or c in "_-" for c in name):
            return None

        profile_path = CONTEXTS_DIR / f"{name}.json"
        if not profile_path.exists():
            return None

        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            return ContextProfile(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Failed to load context profile '{name}': {e}")
            return None

    def save_context_profile(self, profile: ContextProfile) -> None:
        """Save a context profile.

        Args:
            profile: ContextProfile to save
        """
        _ensure_dirs()

        # Validate name for safety
        if not all(c.isalnum() or c in "_-" for c in profile.name):
            raise ValueError(f"Invalid profile name: {profile.name}")

        profile_path = CONTEXTS_DIR / f"{profile.name}.json"

        with _file_lock(f"context-{profile.name}"):
            profile_path.write_text(
                json.dumps(profile.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            profile_path.chmod(0o600)

        logger.info(f"Saved context profile: {profile.name}")

    def list_context_profiles(self) -> list[str]:
        """List all available context profile names.

        Returns:
            List of profile names
        """
        if not CONTEXTS_DIR.exists():
            return []

        return sorted([
            p.stem for p in CONTEXTS_DIR.glob("*.json")
            if p.is_file()
        ])

    def delete_context_profile(self, name: str) -> bool:
        """Delete a context profile.

        Args:
            name: Profile name to delete

        Returns:
            True if deleted, False if not found
        """
        if not name or not all(c.isalnum() or c in "_-" for c in name):
            return False

        profile_path = CONTEXTS_DIR / f"{name}.json"
        if profile_path.exists():
            profile_path.unlink()
            logger.info(f"Deleted context profile: {name}")
            return True
        return False

    # =========================================================================
    # Bias List Selection
    # =========================================================================

    def select_bias_list(
        self,
        context: str | None = None,
        max_entries: int = 60,
        min_tier: TierLevel = "H",
        include_contexts: list[str] | None = None,
        exclude_contexts: list[str] | None = None,
    ) -> list[EntryWithRelations]:
        """Select entries for a bias list based on scoring.

        Scoring formula:
            score = boost_weight * tier_weight * recency_decay * context_match

        Where:
            - tier_weight: A=1.0 (critical), B-D=0.7 (standard), E-H=0.4 (low)
            - recency_decay: 1.0 if seen in 60 days, 0.5 otherwise
            - context_match: 1.5 if entry has matching context tag, 1.0 otherwise

        Args:
            context: Context for scoring (entries with matching context get 1.5x boost)
            max_entries: Maximum number of entries to return (default 150)
            min_tier: Minimum tier to include (e.g., "C" includes A, B, C only)
            include_contexts: Only include entries with these contexts
            exclude_contexts: Exclude entries with these contexts

        Returns:
            List of entries sorted by score (highest first)
        """
        # Get entries from SQLite database
        all_entries = get_all_entries()

        # Handle empty database
        if not all_entries:
            logger.warning("No entries found in dictionary database")
            return []

        # Score and filter entries
        scored: list[tuple[float, EntryWithRelations]] = []

        min_tier_idx = TIER_ORDER.index(min_tier)

        for entry in all_entries:
            # Filter by tier (but always include tier A)
            if entry.tier != "A":
                entry_tier_idx = TIER_ORDER.index(entry.tier)
                if entry_tier_idx > min_tier_idx:
                    continue

            # Filter by include_contexts (tier A always passes)
            if include_contexts:
                has_required = any(
                    entry.has_context(ctx) for ctx in include_contexts
                ) or entry.tier == "A"
                if not has_required:
                    continue

            # Filter by exclude_contexts (tier A can still be excluded)
            if exclude_contexts:
                if any(entry.has_context(ctx) for ctx in exclude_contexts):
                    continue

            # Calculate score: boost_weight * tier_weight * recency_decay * context_match
            score = self._calculate_score(entry, context=context)

            scored.append((score, entry))

        # Sort by score (descending), then by canonical (ascending) for stability
        scored.sort(key=lambda x: (-x[0], x[1].canonical.lower()))

        # Return top entries
        return [entry for _, entry in scored[:max_entries]]

    # Backward-compatible alias for select_bias_list
    def select(
        self,
        context: str | None = None,
        max_entries: int = 60,
        min_tier: TierLevel = "H",
        include_aliases: bool = True,
        recency_boost_days: int = 30,
    ) -> list[EntryWithRelations]:
        """Select entries for a bias list, ranked by relevance.

        This is an alias for select_bias_list for backward compatibility.

        Args:
            context: Context name to filter by (e.g., "running", "work").
            max_entries: Maximum number of entries to return
            min_tier: Minimum tier to include (entries below this tier are excluded)
            include_aliases: Whether to include alias forms in scoring (unused)
            recency_boost_days: Days within which recency boost applies (unused - uses default)

        Returns:
            List of entries sorted by relevance score (highest first)
        """
        return self.select_bias_list(
            context=context,
            max_entries=max_entries,
            min_tier=min_tier,
        )

    def select_with_profile(
        self,
        profile: ContextProfile,
        include_aliases: bool = True,
    ) -> list[EntryWithRelations]:
        """Select entries using a context profile configuration.

        Args:
            profile: ContextProfile with selection settings
            include_aliases: Whether to include alias forms

        Returns:
            List of entries matching profile criteria
        """
        # Get entries from SQLite database
        all_entries = get_all_entries()

        # Handle empty database
        if not all_entries:
            logger.warning("No entries found in dictionary database")
            return []

        scored: list[tuple[float, EntryWithRelations]] = []

        # Get min tier from profile (use include_tiers if set)
        min_tier: TierLevel = "H"
        if profile.include_tiers:
            # Use the lowest (worst) tier from include_tiers as min
            tier_indices = [TIER_ORDER.index(t) for t in profile.include_tiers]
            min_tier = TIER_ORDER[max(tier_indices)]

        min_tier_idx = TIER_ORDER.index(min_tier)

        for entry in all_entries:
            # Check tier requirement (but always include tier A)
            if entry.tier != "A":
                entry_tier_idx = TIER_ORDER.index(entry.tier)
                if entry_tier_idx > min_tier_idx:
                    continue

                # If include_tiers is set, check exact membership
                if profile.include_tiers and entry.tier not in profile.include_tiers:
                    continue

            # Check required contexts
            if profile.include_contexts:
                has_required = any(
                    entry.has_context(ctx) for ctx in profile.include_contexts
                ) or entry.tier == "A"  # Tier A always passes
                if not has_required:
                    continue

            # Calculate base score
            score = self._calculate_score(
                entry,
                context=profile.name if profile.include_contexts else None,
            )

            # Apply profile boost multiplier
            score *= profile.boost_multiplier

            scored.append((score, entry))

        # Sort and limit
        scored.sort(key=lambda x: -x[0])
        return [entry for _, entry in scored[: profile.max_entries]]

    def _calculate_score(
        self,
        entry: EntryWithRelations,
        context: str | None = None,
    ) -> float:
        """Calculate relevance score for an entry.

        Score = boost_weight * tier_weight * recency_decay * context_match

        Args:
            entry: Entry to score
            context: Active context name

        Returns:
            Relevance score (higher is better)
        """
        # Tier weight (fallback to lowest tier weight if tier is invalid)
        tier_weight = TIER_WEIGHTS.get(entry.tier, 0.4)

        # Boost weight (already 0.0-3.0)
        # Special case: if boost is 0.0, use small non-zero value to preserve tier ordering
        boost = max(entry.boost_weight, 0.01)

        # Context relevance
        context_weight = 1.0
        if context:
            if entry.has_context(context):
                context_weight = 1.5  # Boost for matching context
            elif entry.tier == "A":
                context_weight = 1.0  # Tier A always relevant
            else:
                context_weight = 1.0  # No penalty for non-matching

        # Recency decay: binary (recent = 60 days, otherwise old)
        recency_weight = 0.5  # Default for never seen or old
        if entry.last_seen_at:
            # Protect against future timestamps (clock skew or data corruption)
            days_since = (datetime.now() - entry.last_seen_at).days
            if days_since < 0:
                logger.warning(
                    f"Entry '{entry.canonical}' has future last_seen_at: "
                    f"{entry.last_seen_at}"
                )
                recency_weight = 0.5  # Treat as old/unseen
            else:
                recency_weight = 1.0 if days_since <= 60 else 0.5

        return tier_weight * boost * context_weight * recency_weight

    # =========================================================================
    # Context Detection
    # =========================================================================

    def detect_context(self, text: str) -> str | None:
        """Auto-detect context from keywords in text.

        Scans the provided text for keywords that indicate a specific context.
        Returns the context with the most keyword matches.

        Keyword mappings:
            - "strava", "running", "marathon" -> "running"
            - "veeva", "vault", "gxp", "capa" -> "work"
            - "whisper", "mlx", "claude", "transcription" -> "asr_dev"
            - etc.

        Args:
            text: Text to analyze for context keywords

        Returns:
            Detected context name, or None if no context detected
        """
        if not text:
            return None

        text_lower = text.lower()

        # Count keyword matches for each context
        context_scores: dict[str, int] = {}

        for context, keywords in CONTEXT_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                # Check for keyword as whole word or phrase
                if keyword in text_lower:
                    # Give more weight to longer keywords (more specific)
                    score += len(keyword.split())
            if score > 0:
                context_scores[context] = score

        if not context_scores:
            return None

        # Return context with highest score
        best_context = max(context_scores.items(), key=lambda x: x[1])
        return best_context[0]

    def get_context_keywords(self) -> dict[str, list[str]]:
        """Get the keyword mappings used for context detection.

        Returns:
            Dictionary mapping context names to keyword lists
        """
        return CONTEXT_KEYWORDS.copy()

    def add_context_keywords(self, context: str, keywords: list[str]) -> None:
        """Add custom keywords for a context.

        Note: This only updates the in-memory mappings for this session.
        For persistent custom keywords, save them in a context profile.

        Args:
            context: Context name
            keywords: Keywords to add
        """
        if context not in CONTEXT_KEYWORDS:
            CONTEXT_KEYWORDS[context] = []

        for keyword in keywords:
            keyword = keyword.lower().strip()
            if keyword and keyword not in CONTEXT_KEYWORDS[context]:
                CONTEXT_KEYWORDS[context].append(keyword)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def get_forms_for_whisper(
        self,
        entries: list[EntryWithRelations],
        max_forms: int = 200,
    ) -> list[str]:
        """Extract text forms from entries for Whisper prompt.

        Returns canonical forms and key aliases, prioritized by entry score.
        Limits total forms to stay within Whisper prompt limits.

        Args:
            entries: Entries to extract forms from (should be pre-sorted by relevance)
            max_forms: Maximum total forms to return

        Returns:
            List of unique text forms (canonical + aliases)
        """
        forms: list[str] = []
        seen: set[str] = set()

        for entry in entries:
            # Add canonical
            if entry.canonical.lower() not in seen:
                forms.append(entry.canonical)
                seen.add(entry.canonical.lower())

            # Add display form if different
            if entry.display:
                display_lower = entry.display.lower()
                if display_lower not in seen:
                    forms.append(entry.display)
                    seen.add(display_lower)

            # Add top aliases (non-misspelling ones first)
            for alias in sorted(entry.aliases, key=lambda a: a.is_common_misspelling):
                if len(forms) >= max_forms:
                    break
                alias_lower = alias.alias.lower()
                if alias_lower not in seen and not alias.is_common_misspelling:
                    forms.append(alias.alias)
                    seen.add(alias_lower)

            if len(forms) >= max_forms:
                break

        return forms

    def get_entries_for_context(
        self,
        context: str,
        max_entries: int = 60,
    ) -> list[EntryWithRelations]:
        """Get entries specifically tagged with a context.

        Args:
            context: Context tag to filter by
            max_entries: Maximum entries to return

        Returns:
            List of entries with the specified context
        """
        # Get entries from SQLite database by context
        entries = get_entries_by_context(context, limit=max_entries)

        # Sort by tier then by canonical
        entries.sort(key=lambda e: (TIER_ORDER.index(e.tier), e.canonical.lower()))

        return entries[:max_entries]

    def get_recently_used_entries(
        self,
        days: int = 30,
        max_entries: int = 50,
    ) -> list[EntryWithRelations]:
        """Get entries that have been used recently.

        Args:
            days: Number of days to look back
            max_entries: Maximum entries to return

        Returns:
            List of recently used entries, sorted by last_seen (most recent first)
        """
        # Get recent entries directly from SQLite database
        return get_recent_entries(days=days, limit=max_entries)

    def get_top_entries_by_usage(
        self,
        max_entries: int = 50,
    ) -> list[EntryWithRelations]:
        """Get entries with the highest usage counts.

        Args:
            max_entries: Maximum entries to return

        Returns:
            List of most-used entries, sorted by occurrence_count (highest first)
        """
        # Get all entries from SQLite database
        all_entries = get_all_entries(limit=max_entries * 2)

        # Sort by occurrence_count descending
        all_entries.sort(key=lambda e: e.occurrence_count, reverse=True)

        return all_entries[:max_entries]

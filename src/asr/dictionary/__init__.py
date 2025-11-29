"""ASR dictionary system for bias list management and context-aware selection.

This module provides:
- DictionaryManager: High-level CRUD for dictionary entries
- BiasListSelector: Context-aware entry selection and ranking
- CandidateMatcher: Fuzzy matching of text to dictionary entries
- Helper functions for generating Whisper prompts and correction blocks
- SQLite database operations for persistent storage

Auto-initialization:
On first import, the database directory is created if it doesn't exist.
The actual database is lazily initialized when accessed.
"""

from pathlib import Path

from asr.dictionary.db import (
    bulk_create_entries,
    create_entry,
    delete_entry,
    entry_exists,
    get_all_contexts,
    get_all_entries,
    get_connection,
    get_entries_by_context,
    get_entries_by_tier,
    get_entry,
    get_entry_by_canonical,
    get_recent_entries,
    get_stats,
    increment_occurrence,
    init_db,
    record_occurrence,
    search_entries,
    update_entry,
    update_last_seen,
)
from asr.dictionary.manager import DictionaryManager
from asr.dictionary.matcher import CandidateMatcher, MatchCandidate, SpanMatch, create_matcher
from asr.dictionary.migrations import (
    check_integrity,
    export_to_json,
    import_from_json,
    migrate_from_legacy,
)
from asr.dictionary.models import (
    Alias,
    ContextProfile,
    DictionaryEntry,
    DictionaryStats,
    Entry,
    EntryAlias,
    EntryContext,
    EntryPronunciation,
    EntryTier,
    EntryType,
    EntryWithRelations,
    Pronunciation,
    SearchResult,
    TierLevel,
    TIER_WEIGHTS,
)
from asr.dictionary.prompt_generator import (
    estimate_tokens,
    format_entries_for_context,
    generate_combined_prompt,
    generate_correction_block,
    generate_whisper_prompt,
)
from asr.dictionary.selector import BiasListSelector

# NER functions (optional - requires: pip install asr[ner])
_NER_AVAILABLE = False
try:
    # Check if gliner is actually installed (not just ner.py)
    import gliner as _gliner_check  # noqa: F401
    del _gliner_check

    from asr.dictionary.ner import (
        ExtractedEntity,
        extract_proper_nouns,
        extract_proper_nouns_batch,
        filter_entities_by_type,
        deduplicate_entities,
        ENTITY_TYPES,
    )
    _NER_AVAILABLE = True
except ImportError:
    # GLiNER not installed - NER features unavailable
    # Still export the types for type hints
    from asr.dictionary.ner import ExtractedEntity, ENTITY_TYPES
    extract_proper_nouns = None
    extract_proper_nouns_batch = None
    filter_entities_by_type = None
    deduplicate_entities = None

# Discovery pipeline (NER → Dictionary)
from asr.dictionary.discovery import (  # noqa: E402
    DiscoveredNoun,
    DiscoveryResult,
    add_discovered_to_pending,
    approve_pending_noun,
    discover_proper_nouns,
    get_pending_nouns,
    reject_pending_noun,
)

# Auto-initialize dictionary directory on import
_DICT_DIR = Path.home() / ".asr" / "dictionaries"
_DICT_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    # Manager (high-level JSON-based storage)
    "DictionaryManager",
    # Selector
    "BiasListSelector",
    # Matcher
    "CandidateMatcher",
    "MatchCandidate",
    "SpanMatch",
    "create_matcher",
    # Prompt Generator
    "generate_whisper_prompt",
    "generate_correction_block",
    "generate_combined_prompt",
    "estimate_tokens",
    "format_entries_for_context",
    # Core Models
    "Entry",
    "Alias",
    "Pronunciation",
    "EntryWithRelations",
    "ContextProfile",
    "SearchResult",
    "DictionaryStats",
    "EntryType",
    "TierLevel",
    "TIER_WEIGHTS",
    # Backward-compatible model aliases
    "DictionaryEntry",
    "EntryAlias",
    "EntryPronunciation",
    "EntryTier",
    "EntryContext",
    # SQLite Database Operations
    "init_db",
    "get_connection",
    "create_entry",
    "get_entry",
    "get_entry_by_canonical",
    "update_entry",
    "delete_entry",
    "search_entries",
    "bulk_create_entries",
    "get_entries_by_context",
    "get_entries_by_tier",
    "get_all_contexts",
    "get_all_entries",
    "get_recent_entries",
    "get_stats",
    "entry_exists",
    "increment_occurrence",
    "update_last_seen",
    "record_occurrence",
    # Migration and import/export
    "migrate_from_legacy",
    "import_from_json",
    "export_to_json",
    "check_integrity",
    # NER (optional - requires pip install asr[ner])
    "_NER_AVAILABLE",
    "ExtractedEntity",
    "extract_proper_nouns",
    "extract_proper_nouns_batch",
    "filter_entities_by_type",
    "deduplicate_entities",
    "ENTITY_TYPES",
    # Discovery (NER → Dictionary pipeline)
    "discover_proper_nouns",
    "add_discovered_to_pending",
    "get_pending_nouns",
    "approve_pending_noun",
    "reject_pending_noun",
    "DiscoveredNoun",
    "DiscoveryResult",
]

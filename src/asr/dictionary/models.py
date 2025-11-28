"""Pydantic v2 models for ASR dictionary system.

This module defines the data models for a sophisticated ASR vocabulary dictionary
that supports proper nouns, domain-specific terms, aliases, and pronunciations.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# Type definitions
EntryType = Literal["person", "org", "product", "event", "location", "jargon", "misc"]
TierLevel = Literal["A", "B", "C", "D", "E", "F", "G", "H"]


# Tier weight mapping for scoring
TIER_WEIGHTS: dict[str, float] = {
    "A": 1.0,
    "B": 0.9,
    "C": 0.8,
    "D": 0.7,
    "E": 0.6,
    "F": 0.5,
    "G": 0.4,
    "H": 0.3,
}


class Entry(BaseModel):
    """A dictionary entry representing a term to be recognized by ASR.

    Entries are the core unit of the dictionary system. Each entry represents
    a canonical form of a term (e.g., proper noun, technical term) that should
    be correctly recognized during transcription.

    Attributes:
        id: Unique identifier (UUID string)
        canonical: The correct/canonical spelling of the term
        display: Optional display form if different from canonical
        type: Category of the entry (person, org, product, etc.)
        tier: Priority tier A-H (A = highest priority, H = lowest)
        boost_weight: Multiplier for scoring (0.0-3.0, default 1.0)
        language: ISO language code (default "en")
        occurrence_count: How many times this term has been seen in transcripts
        last_seen_at: When this term was last encountered
        source: Origin of the entry (manual, learned, imported, etc.)
        notes: Optional notes about the entry
        created_at: When the entry was created
        updated_at: When the entry was last modified
    """

    id: str = Field(..., description="UUID string identifier")
    canonical: str = Field(..., min_length=1, description="Canonical spelling of the term")
    display: str | None = Field(default=None, description="Display form if different from canonical")
    type: EntryType = Field(..., description="Category of the entry")
    tier: TierLevel = Field(default="D", description="Priority tier A-H")
    boost_weight: float = Field(default=1.0, ge=0.0, le=3.0, description="Scoring multiplier")
    language: str = Field(default="en", description="ISO language code")
    occurrence_count: int = Field(default=0, ge=0, description="Number of times seen in transcripts")
    last_seen_at: datetime | None = Field(default=None, description="When last encountered")
    source: str | None = Field(default=None, description="Origin of the entry")
    notes: str | None = Field(default=None, description="Optional notes")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class Alias(BaseModel):
    """An alternative spelling or variation of a dictionary entry.

    Aliases help match variations of a term to its canonical form.
    This includes common misspellings, abbreviations, and alternative spellings.

    Attributes:
        id: Database row ID (None for new entries)
        entry_id: UUID of the parent Entry
        alias: The alternative spelling/variation
        is_common_misspelling: Whether this is a known misspelling vs. valid alternative
    """

    id: int | None = Field(default=None, description="Database row ID")
    entry_id: str = Field(..., description="UUID of parent entry")
    alias: str = Field(..., min_length=1, description="Alternative spelling/variation")
    is_common_misspelling: bool = Field(default=False, description="Is this a misspelling?")


class Pronunciation(BaseModel):
    """Phonetic pronunciation information for a dictionary entry.

    Supports both IPA (International Phonetic Alphabet) and phoneme sequences
    for integration with speech recognition systems.

    Attributes:
        id: Database row ID (None for new entries)
        entry_id: UUID of the parent Entry
        ipa: IPA transcription (e.g., "/njutn/")
        phoneme_sequence: Space-separated phoneme codes
        language: ISO language code for this pronunciation
        variant: Pronunciation variant (e.g., "US", "UK", "formal")
    """

    id: int | None = Field(default=None, description="Database row ID")
    entry_id: str = Field(..., description="UUID of parent entry")
    ipa: str | None = Field(default=None, description="IPA transcription")
    phoneme_sequence: str | None = Field(default=None, description="Space-separated phonemes")
    language: str = Field(default="en", description="ISO language code")
    variant: str | None = Field(default=None, description="Pronunciation variant (US, UK, etc.)")


class EntryWithRelations(BaseModel):
    """A dictionary entry with all its related data.

    This is the full representation of an entry including its aliases,
    pronunciations, and context associations. Used for creating and
    retrieving complete entries.

    Attributes:
        All fields from Entry, plus:
        aliases: List of alternative spellings
        pronunciations: List of pronunciation variants
        contexts: List of context names this entry belongs to
    """

    # Entry fields
    id: str = Field(..., description="UUID string identifier")
    canonical: str = Field(..., min_length=1, description="Canonical spelling of the term")
    display: str | None = Field(default=None, description="Display form if different from canonical")
    type: EntryType = Field(..., description="Category of the entry")
    tier: TierLevel = Field(default="D", description="Priority tier A-H")
    boost_weight: float = Field(default=1.0, ge=0.0, le=3.0, description="Scoring multiplier")
    language: str = Field(default="en", description="ISO language code")
    occurrence_count: int = Field(default=0, ge=0, description="Number of times seen in transcripts")
    last_seen_at: datetime | None = Field(default=None, description="When last encountered")
    source: str | None = Field(default=None, description="Origin of the entry")
    notes: str | None = Field(default=None, description="Optional notes")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    # Relations
    aliases: list[Alias] = Field(default_factory=list, description="Alternative spellings")
    pronunciations: list[Pronunciation] = Field(default_factory=list, description="Pronunciation variants")
    contexts: list[str] = Field(default_factory=list, description="Associated context names")

    def to_entry(self) -> Entry:
        """Extract just the Entry portion without relations."""
        return Entry(
            id=self.id,
            canonical=self.canonical,
            display=self.display,
            type=self.type,
            tier=self.tier,
            boost_weight=self.boost_weight,
            language=self.language,
            occurrence_count=self.occurrence_count,
            last_seen_at=self.last_seen_at,
            source=self.source,
            notes=self.notes,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_entry(
        cls,
        entry: Entry,
        aliases: list[Alias] | None = None,
        pronunciations: list[Pronunciation] | None = None,
        contexts: list[str] | None = None,
    ) -> "EntryWithRelations":
        """Create an EntryWithRelations from an Entry and optional relations."""
        return cls(
            id=entry.id,
            canonical=entry.canonical,
            display=entry.display,
            type=entry.type,
            tier=entry.tier,
            boost_weight=entry.boost_weight,
            language=entry.language,
            occurrence_count=entry.occurrence_count,
            last_seen_at=entry.last_seen_at,
            source=entry.source,
            notes=entry.notes,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            aliases=aliases or [],
            pronunciations=pronunciations or [],
            contexts=contexts or [],
        )

    def get_all_forms(self) -> list[str]:
        """Get all text forms of this entry (canonical + aliases)."""
        forms = [self.canonical]
        forms.extend(alias.alias for alias in self.aliases)
        return forms

    def has_context(self, context: str) -> bool:
        """Check if entry has a specific context tag."""
        return any(c.lower() == context.lower() for c in self.contexts)


class ContextProfile(BaseModel):
    """A context profile for filtering dictionary entries.

    Context profiles define which entries should be loaded for a specific
    transcription context. For example, a "medical" profile might include
    only tiers A-C and medical-related contexts.

    Attributes:
        name: Unique name for this profile
        description: Human-readable description
        include_tiers: Which tiers to include (e.g., ["A", "B", "C"])
        include_contexts: Which contexts to include (e.g., ["medical", "anatomy"])
        max_entries: Maximum number of entries to load (default 150)
        boost_multiplier: Global boost multiplier for this profile (default 1.0)
    """

    name: str = Field(..., min_length=1, description="Unique profile name")
    description: str = Field(default="", description="Human-readable description")
    include_tiers: list[TierLevel] = Field(default_factory=list, description="Tiers to include")
    include_contexts: list[str] = Field(default_factory=list, description="Contexts to include")
    max_entries: int = Field(default=150, ge=1, description="Maximum entries to load")
    boost_multiplier: float = Field(default=1.0, ge=0.0, le=10.0, description="Global boost multiplier")


# Search result model
class SearchResult(BaseModel):
    """A search result with relevance score.

    Returned by search operations to provide ranked results with
    match quality information.
    """

    entry: EntryWithRelations
    score: float = Field(ge=0.0, description="Relevance score")
    matched_on: str = Field(default="canonical", description="Which field matched (canonical, alias)")


# Dictionary statistics model
class DictionaryStats(BaseModel):
    """Statistics about the dictionary."""

    total_entries: int = 0
    entries_by_tier: dict[str, int] = Field(default_factory=dict)
    entries_by_type: dict[str, int] = Field(default_factory=dict)
    entries_by_context: dict[str, int] = Field(default_factory=dict)
    total_aliases: int = 0
    total_pronunciations: int = 0
    last_updated: datetime | None = None


# Backward-compatible aliases
DictionaryEntry = Entry
EntryAlias = Alias
EntryPronunciation = Pronunciation
EntryTier = TierLevel
# Note: EntryContext is not a separate model in this design - contexts are stored as list[str]
# on EntryWithRelations. Adding a dummy for import compatibility.
EntryContext = str

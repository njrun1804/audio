"""Proper noun discovery and validation for ASR dictionary.

This module connects NER extraction to the dictionary system with
anti-hallucination safeguards.

Flow:
1. NER extracts candidate proper nouns from transcript
2. Validation checks each candidate appeared in original ASR output
3. Validated candidates go to pending queue (not directly to dictionary)
4. Manual or auto-approval moves them to active dictionary

Requires: pip install asr[ner]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asr.models.transcript import Segment

logger = logging.getLogger(__name__)

# Common English words to filter out (not proper nouns)
# Top 200 most common words + some common verbs/adjectives
COMMON_WORDS = frozenset([
    # Articles, pronouns, prepositions
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "us", "them", "who", "what", "which", "whom", "whose",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "out", "off", "over", "under", "again", "further",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "only", "also", "just", "even", "still", "already", "ever",
    # Common verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "can", "shall", "need", "get", "got", "go", "went", "gone",
    "come", "came", "take", "took", "taken", "make", "made", "see", "saw",
    "seen", "know", "knew", "known", "think", "thought", "say", "said",
    "tell", "told", "give", "gave", "given", "find", "found", "want", "use",
    # Common adjectives/adverbs
    "good", "bad", "new", "old", "great", "high", "small", "big", "large",
    "long", "little", "own", "other", "same", "right", "left", "first",
    "last", "next", "few", "many", "much", "more", "most", "less", "least",
    "very", "really", "quite", "rather", "too", "enough", "almost", "well",
    "now", "then", "here", "there", "where", "when", "how", "why", "all",
    "each", "every", "both", "any", "some", "no", "not", "none", "yes",
    # Common nouns (non-proper)
    "time", "year", "people", "way", "day", "man", "woman", "child", "world",
    "life", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact",
    "month", "lot", "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend", "father",
    "power", "hour", "game", "line", "end", "member", "law", "car", "city",
    "name", "president", "team", "minute", "idea", "body", "information",
    "back", "face", "others", "level", "office", "door", "health", "person",
    "art", "war", "history", "party", "result", "change", "morning", "reason",
])

# Minimum word length for proper nouns
MIN_WORD_LENGTH = 4

# Maximum file size for pending nouns JSON (10MB)
MAX_PENDING_FILE_SIZE = 10 * 1024 * 1024

# Map NER entity types to dictionary entry types
NER_TYPE_MAP = {
    "person": "person",
    "organization": "org",
    "product": "product",
    "location": "location",
    "event": "event",
    "company": "org",
    "brand": "product",
}


def _is_weird_enough(
    text: str, session_counts: dict[str, int] | None = None
) -> tuple[bool, str | None]:
    """Check if a term passes "weirdness" filters for proper noun candidacy.

    Filters:
    1. Not a common English word
    2. Minimum length (4+ characters)
    3. Optional: 2+ occurrences in session

    Args:
        text: The candidate term
        session_counts: Optional dict of term -> count in current session

    Returns:
        Tuple of (passes_filter, rejection_reason)
    """
    text_lower = text.lower().strip()
    words = text_lower.split()

    # Filter 1: Minimum length (at least one word must be 4+ chars)
    has_long_word = any(len(w) >= MIN_WORD_LENGTH for w in words)
    if not has_long_word:
        return False, f"Too short: '{text}' (need 4+ char word)"

    # Filter 2: Not entirely common words
    # At least one word must NOT be in common words list
    has_uncommon_word = any(w not in COMMON_WORDS for w in words)
    if not has_uncommon_word:
        return False, f"Common word(s): '{text}'"

    # Filter 3: Session frequency (if tracking)
    if session_counts is not None:
        count = session_counts.get(text_lower, 0)
        if count < 2:
            return False, f"Low frequency: '{text}' (seen {count} time(s), need 2+)"

    return True, None


@dataclass
class DiscoveredNoun:
    """A proper noun discovered via NER and validated against ASR output."""

    text: str
    entity_type: str
    confidence: float
    source_file: str
    validation_method: str  # "exact", "partial", "phonetic"
    matched_asr_text: str | None = None  # What it matched in ASR output
    snippet: str | None = None  # Sentence context where term was found


@dataclass
class DiscoveryResult:
    """Results from proper noun discovery."""

    discovered: list[DiscoveredNoun] = field(default_factory=list)
    rejected_hallucinations: list[str] = field(default_factory=list)
    rejected_weirdness: list[str] = field(default_factory=list)  # Failed weirdness filters
    already_in_dictionary: list[str] = field(default_factory=list)
    added_to_pending: int = 0


def discover_proper_nouns(
    segments: list[Segment],
    source_file: str | Path,
    min_confidence: float = 0.7,
    context: str | None = None,
    require_session_frequency: bool = True,
) -> DiscoveryResult:
    """Discover proper nouns from transcript segments using NER.

    Validates each discovered entity against the original ASR output
    to prevent learning hallucinated terms from corrections.

    Applies "weirdness" filters:
    - Not a common English word
    - Minimum length (4+ characters)
    - Appears 2+ times in session (if require_session_frequency=True)

    Args:
        segments: Transcript segments (with raw_text for validation)
        source_file: Source audio file path (for tracking)
        min_confidence: Minimum NER confidence to consider
        context: Dictionary context for categorization
        require_session_frequency: Require 2+ occurrences in session

    Returns:
        DiscoveryResult with validated nouns and rejection stats
    """
    from asr.dictionary import _NER_AVAILABLE

    if not _NER_AVAILABLE:
        raise ImportError(
            "NER is required for discovery. Install with: pip install asr[ner]"
        )

    from asr.dictionary.ner import extract_proper_nouns, deduplicate_entities
    from asr.dictionary.db import get_entry_by_canonical, search_entries

    result = DiscoveryResult()
    if isinstance(source_file, (str, Path)):
        source_name = Path(source_file).name
    else:
        source_name = str(source_file)

    # Collect all text for NER
    corrected_text = " ".join(seg.text for seg in segments)

    # Collect original ASR words for validation (anti-hallucination)
    original_words: set[str] = set()
    for seg in segments:
        # Use raw_text if available (before correction), else use text
        raw = getattr(seg, "raw_text", None) or seg.text
        original_words.update(w.lower() for w in raw.split())

    # Run NER on corrected text
    entities = extract_proper_nouns(corrected_text)
    entities = deduplicate_entities(entities)

    # Filter by confidence
    entities = [e for e in entities if e.confidence >= min_confidence]

    # Count session frequency for weirdness filter
    session_counts: dict[str, int] = {}
    for entity in entities:
        key = entity.text.lower()
        session_counts[key] = session_counts.get(key, 0) + 1

    for entity in entities:
        # WEIRDNESS FILTER: Check common word, length, session frequency
        is_weird, reject_reason = _is_weird_enough(
            entity.text,
            session_counts=session_counts if require_session_frequency else None,
        )
        if not is_weird:
            result.rejected_weirdness.append(f"{entity.text}: {reject_reason}")
            logger.debug(f"Rejected weirdness: {reject_reason}")
            continue

        # Skip if already in dictionary
        existing = get_entry_by_canonical(entity.text)
        if existing:
            result.already_in_dictionary.append(entity.text)
            continue

        # Also check via search (catches aliases)
        search_results = search_entries(entity.text, limit=1)
        if search_results and search_results[0].canonical.lower() == entity.text.lower():
            result.already_in_dictionary.append(entity.text)
            continue

        # VALIDATION: Check entity appeared in original ASR output
        validation = _validate_against_asr(entity.text, original_words)

        if validation is None:
            # Failed validation - likely hallucinated by correction
            result.rejected_hallucinations.append(entity.text)
            logger.debug(f"Rejected hallucination: {entity.text}")
            continue

        # Extract snippet context (sentence containing the term)
        snippet = _extract_snippet(corrected_text, entity.text)

        # Passed validation
        discovered = DiscoveredNoun(
            text=entity.text,
            entity_type=NER_TYPE_MAP.get(entity.entity_type, "misc"),
            confidence=entity.confidence,
            source_file=source_name,
            validation_method=validation[0],
            matched_asr_text=validation[1],
            snippet=snippet,
        )
        result.discovered.append(discovered)

    return result


def _extract_snippet(text: str, term: str, max_length: int = 150) -> str | None:
    """Extract a sentence snippet containing the term.

    Args:
        text: Full text to search
        term: Term to find context for
        max_length: Maximum snippet length

    Returns:
        Sentence containing the term, or None if not found
    """
    # Find term position (case insensitive)
    term_lower = term.lower()
    text_lower = text.lower()
    pos = text_lower.find(term_lower)
    if pos == -1:
        return None

    # Find sentence boundaries
    # Look for sentence start (period, newline, or start of text)
    sentence_start = max(0, pos - 75)
    for i in range(pos - 1, sentence_start - 1, -1):
        if i >= 0 and text[i] in ".!?\n":
            sentence_start = i + 1
            break

    # Look for sentence end
    sentence_end = min(len(text), pos + len(term) + 75)
    for i in range(pos + len(term), sentence_end):
        if text[i] in ".!?\n":
            sentence_end = i + 1
            break

    snippet = text[sentence_start:sentence_end].strip()

    # Truncate if too long
    if len(snippet) > max_length:
        # Center around the term
        term_pos_in_snippet = snippet.lower().find(term_lower)
        start = max(0, term_pos_in_snippet - max_length // 2)
        end = min(len(snippet), start + max_length)
        snippet = "..." + snippet[start:end].strip() + "..."

    return snippet


def _validate_against_asr(
    entity_text: str,
    original_words: set[str],
) -> tuple[str, str] | None:
    """Validate an entity appeared in original ASR output.

    Returns (validation_method, matched_text) if valid, None if hallucination.
    """
    entity_lower = entity_text.lower()
    entity_words = set(entity_lower.split())

    # Method 1: Exact match (all words present)
    if entity_words.issubset(original_words):
        return ("exact", entity_text)

    # Method 2: Partial match (at least half the words)
    overlap = entity_words & original_words
    if len(overlap) >= len(entity_words) / 2:
        return ("partial", " ".join(overlap))

    # Method 3: Phonetic match (for misspellings)
    try:
        from asr.dictionary.matcher import CandidateMatcher
        from asr.dictionary.models import EntryWithRelations

        # Create a temporary entry for matching
        temp_entry = EntryWithRelations(
            id="temp",
            canonical=entity_text,
            type="misc",
            tier="C",
            boost_weight=1.0,
        )
        matcher = CandidateMatcher([temp_entry])

        # Check if any original word phonetically matches
        for word in original_words:
            if len(word) < 3:
                continue
            candidates = matcher.find_candidates(word)
            if candidates and candidates[0].confidence >= 0.6:
                return ("phonetic", word)
    except Exception:
        pass  # Phonetic matching optional

    return None  # Failed all validation


def add_discovered_to_pending(
    discovered: list[DiscoveredNoun],
    context: str = "discovered",
    auto_approve_threshold: int = 3,
) -> tuple[int, int]:
    """Add discovered nouns to the dictionary pending queue.

    Args:
        discovered: List of validated discovered nouns
        context: Dictionary context to assign
        auto_approve_threshold: Auto-approve if seen this many times

    Returns:
        (added_to_pending, auto_approved) counts
    """
    from asr.dictionary.db import create_entry
    from asr.dictionary.models import EntryWithRelations

    # Track pending in dictionary's own system
    PENDING_FILE = Path.home() / ".asr" / "dictionaries" / "pending_nouns.json"

    import json
    import os
    pending: dict = {}
    if PENDING_FILE.exists():
        try:
            # SECURITY: Check file size before reading to prevent memory exhaustion
            file_size = os.path.getsize(PENDING_FILE)
            if file_size > MAX_PENDING_FILE_SIZE:
                logger.error(
                    f"Pending file too large: {file_size} bytes (max {MAX_PENDING_FILE_SIZE})"
                )
                return 0, 0

            pending = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Corrupted pending file, starting fresh: {e}")
            pending = {}
        except OSError as e:
            logger.error(f"Failed to read pending file: {e}")
            return 0, 0

    added = 0
    auto_approved = 0

    for noun in discovered:
        key = noun.text.lower()

        if key in pending:
            # Increment occurrence count
            pending[key]["occurrences"] += 1
            if noun.source_file not in pending[key]["sources"]:
                pending[key]["sources"].append(noun.source_file)

            # Check for auto-approval
            if pending[key]["occurrences"] >= auto_approve_threshold:
                # Auto-approve: add to dictionary
                entry = EntryWithRelations(
                    id=f"discovered-{key.replace(' ', '-')}",
                    canonical=noun.text,
                    type=noun.entity_type,
                    tier="C",  # Discovered nouns start at tier C
                    boost_weight=2.0,
                    contexts=[context],
                    source=f"discovered:{pending[key]['sources'][0]}",
                )
                try:
                    create_entry(entry)
                    auto_approved += 1
                    occ = pending[key]['occurrences']
                    logger.info(f"Auto-approved: {noun.text} ({occ} occurrences)")
                    del pending[key]
                except Exception as e:
                    logger.warning(f"Failed to auto-approve {noun.text}: {e}")
        else:
            # New pending entry with snippet context for review
            pending[key] = {
                "text": noun.text,
                "type": noun.entity_type,
                "confidence": noun.confidence,
                "context": context,
                "occurrences": 1,
                "sources": [noun.source_file],
                "validation": noun.validation_method,
                "snippet": noun.snippet,  # Sentence context for review
            }
            added += 1

    # Save pending
    try:
        PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
        PENDING_FILE.write_text(json.dumps(pending, indent=2), encoding="utf-8")
        PENDING_FILE.chmod(0o600)
    except OSError as e:
        logger.error(f"Failed to save pending nouns: {e}")
        return 0, 0

    return added, auto_approved


def get_pending_nouns() -> list[dict]:
    """Get all pending discovered nouns awaiting approval."""
    PENDING_FILE = Path.home() / ".asr" / "dictionaries" / "pending_nouns.json"

    import json
    import os
    if not PENDING_FILE.exists():
        return []

    try:
        # SECURITY: Check file size before reading to prevent memory exhaustion
        file_size = os.path.getsize(PENDING_FILE)
        if file_size > MAX_PENDING_FILE_SIZE:
            logger.error(
                f"Pending file too large: {file_size} bytes (max {MAX_PENDING_FILE_SIZE})"
            )
            return []

        pending = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
        return [
            {**info, "key": key}
            for key, info in pending.items()
        ]
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Corrupted pending file: {e}")
        return []
    except OSError as e:
        logger.error(f"Failed to read pending file: {e}")
        return []


def approve_pending_noun(text: str, context: str | None = None) -> bool:
    """Approve a pending noun and add to dictionary.

    Args:
        text: The noun text to approve
        context: Override context (uses pending context if None)

    Returns:
        True if approved, False if not found
    """
    from asr.dictionary.db import create_entry
    from asr.dictionary.models import EntryWithRelations

    PENDING_FILE = Path.home() / ".asr" / "dictionaries" / "pending_nouns.json"

    import json
    import os
    if not PENDING_FILE.exists():
        return False

    try:
        # SECURITY: Check file size before reading to prevent memory exhaustion
        file_size = os.path.getsize(PENDING_FILE)
        if file_size > MAX_PENDING_FILE_SIZE:
            logger.error(
                f"Pending file too large: {file_size} bytes (max {MAX_PENDING_FILE_SIZE})"
            )
            return False

        pending = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Corrupted pending file: {e}")
        return False
    except OSError as e:
        logger.error(f"Failed to read pending file: {e}")
        return False

    key = text.lower()
    if key not in pending:
        return False

    info = pending[key]
    entry = EntryWithRelations(
        id=f"approved-{key.replace(' ', '-')}",
        canonical=info["text"],
        type=info["type"],
        tier="C",
        boost_weight=2.0,
        contexts=[context or info.get("context", "discovered")],
        source=f"approved:{info['sources'][0]}",
    )

    try:
        create_entry(entry)
        del pending[key]
        PENDING_FILE.write_text(json.dumps(pending, indent=2), encoding="utf-8")
        PENDING_FILE.chmod(0o600)
        return True
    except OSError as e:
        logger.error(f"Failed to save pending file after approval: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to approve {text}: {e}")
        return False


def reject_pending_noun(text: str) -> bool:
    """Reject a pending noun (remove without adding to dictionary)."""
    PENDING_FILE = Path.home() / ".asr" / "dictionaries" / "pending_nouns.json"

    import json
    import os
    if not PENDING_FILE.exists():
        return False

    try:
        # SECURITY: Check file size before reading to prevent memory exhaustion
        file_size = os.path.getsize(PENDING_FILE)
        if file_size > MAX_PENDING_FILE_SIZE:
            logger.error(
                f"Pending file too large: {file_size} bytes (max {MAX_PENDING_FILE_SIZE})"
            )
            return False

        pending = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Corrupted pending file: {e}")
        return False
    except OSError as e:
        logger.error(f"Failed to read pending file: {e}")
        return False

    key = text.lower()
    if key not in pending:
        return False

    try:
        del pending[key]
        PENDING_FILE.write_text(json.dumps(pending, indent=2), encoding="utf-8")
        PENDING_FILE.chmod(0o600)
        return True
    except OSError as e:
        logger.error(f"Failed to save pending file after rejection: {e}")
        return False

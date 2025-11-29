"""Fuzzy and phonetic matching for dictionary entry correction candidates.

This module provides the CandidateMatcher class for finding dictionary entries
that might match ASR output text, using multiple matching strategies:
- Exact matching (canonical and aliases)
- Case-insensitive matching
- Phonetic matching (Double Metaphone)
- Edit distance matching (Levenshtein via rapidfuzz)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from metaphone import doublemetaphone
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

if TYPE_CHECKING:
    from asr.dictionary.models import EntryWithRelations


@dataclass
class MatchCandidate:
    """A potential match between input text and a dictionary entry."""

    entry: EntryWithRelations
    confidence: float
    matched_form: str  # Which form matched (canonical or alias)
    match_type: str  # "exact", "case_insensitive", "soundex", "metaphone", "edit_distance"


@dataclass
class SpanMatch:
    """A potential proper noun span in transcript text with candidates."""

    span: str
    start: int
    end: int
    candidates: list[MatchCandidate]


class CandidateMatcher:
    """Find dictionary entries that match ASR output text.

    Uses multiple matching strategies to identify potential corrections
    for proper nouns in transcription output.

    Matching strategy priority:
    1. Exact match (canonical or alias) -> confidence 1.0
    2. Case-insensitive exact match -> confidence 0.95
    3. Metaphone match -> confidence 0.8 * similarity
    4. Edit distance < 3 -> confidence 0.6 * (1 - distance/max_len)
    """

    def __init__(self, entries: list[EntryWithRelations]) -> None:
        """Initialize the matcher with dictionary entries.

        Args:
            entries: List of dictionary entries to match against.
        """
        # Validate entries input
        if entries is None:
            entries = []
        self.entries = entries
        self._canonical_lower: dict[str, EntryWithRelations] = {}
        self._alias_lower: dict[str, EntryWithRelations] = {}
        # Double Metaphone indexes: primary and alternate codes
        self._metaphone_primary: dict[str, list[tuple[str, EntryWithRelations]]] = {}
        self._metaphone_alternate: dict[str, list[tuple[str, EntryWithRelations]]] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build lookup tables for efficient matching.

        Creates indexes for:
        - canonical_lower -> entry
        - alias_lower -> entry
        - metaphone_primary -> list[(form, entry)] (primary Double Metaphone code)
        - metaphone_alternate -> list[(form, entry)] (alternate Double Metaphone code)
        """
        for entry in self.entries:
            # Skip entries with invalid/missing canonical forms
            if not entry.canonical or not entry.canonical.strip():
                continue

            # Index canonical form
            canonical = entry.canonical
            canonical_lower = canonical.lower()

            # Warn about duplicate canonical entries (last one wins)
            # In production, might want to log or raise an error
            self._canonical_lower[canonical_lower] = entry

            # Add to phonetic indexes
            self._add_to_phonetic_index(canonical, entry)

            # Index aliases
            for alias in entry.aliases:
                # Skip invalid aliases
                if not alias or not hasattr(alias, 'alias'):
                    continue
                if not alias.alias or not alias.alias.strip():
                    continue

                alias_text = alias.alias
                alias_lower = alias_text.lower()

                # Warn about duplicate alias entries (last one wins)
                self._alias_lower[alias_lower] = entry

                # Add aliases to phonetic indexes too
                self._add_to_phonetic_index(alias_text, entry)

    def _add_to_phonetic_index(self, text: str, entry: EntryWithRelations) -> None:
        """Add a text form to phonetic indexes.

        Args:
            text: The text form to index.
            entry: The entry this form belongs to.
        """
        # Only index words with alphabetic characters
        if not any(c.isalpha() for c in text):
            return

        # For multi-word phrases, index each word
        words = text.split()
        for word in words:
            # Clean the word for phonetic matching
            clean_word = re.sub(r"[^a-zA-Z]", "", word)
            if len(clean_word) < 2:
                continue

            # Double Metaphone index - stores both primary and alternate codes
            try:
                primary, alternate = doublemetaphone(clean_word)
                if primary:
                    if primary not in self._metaphone_primary:
                        self._metaphone_primary[primary] = []
                    self._metaphone_primary[primary].append((text, entry))
                if alternate:
                    if alternate not in self._metaphone_alternate:
                        self._metaphone_alternate[alternate] = []
                    self._metaphone_alternate[alternate].append((text, entry))
            except Exception:
                pass  # Skip if metaphone fails

    def find_candidates(
        self,
        text: str,
        max_candidates: int = 5,
    ) -> list[MatchCandidate]:
        """Find dictionary entries that might match the given text.

        Uses multiple matching strategies to identify potential candidates,
        then returns the top matches sorted by confidence.

        Args:
            text: The text to find matches for.
            max_candidates: Maximum number of candidates to return.

        Returns:
            List of (entry, confidence) tuples, sorted by confidence descending.
        """
        if not text or not text.strip():
            return []

        # Validate max_candidates to prevent resource exhaustion
        if max_candidates <= 0:
            return []
        # Cap at reasonable limit to prevent memory issues
        max_candidates = min(max_candidates, 100)

        text = text.strip()
        text_lower = text.lower()
        candidates: dict[str, MatchCandidate] = {}  # entry_id -> best match

        # 1. Exact match on canonical
        if text_lower in self._canonical_lower:
            entry = self._canonical_lower[text_lower]
            candidates[entry.id] = MatchCandidate(
                entry=entry,
                confidence=1.0,
                matched_form=entry.canonical,
                match_type="exact",
            )

        # 2. Exact match on alias
        if text_lower in self._alias_lower:
            entry = self._alias_lower[text_lower]
            if entry.id not in candidates:
                candidates[entry.id] = MatchCandidate(
                    entry=entry,
                    confidence=0.98,  # Slightly lower for alias match
                    matched_form=text,
                    match_type="exact_alias",
                )

        # 3. Phonetic matching (Metaphone)
        metaphone_matches = self._find_metaphone_matches(text)
        for form, entry, base_confidence in metaphone_matches:
            if entry.id not in candidates or candidates[entry.id].confidence < base_confidence:
                candidates[entry.id] = MatchCandidate(
                    entry=entry,
                    confidence=base_confidence,
                    matched_form=form,
                    match_type="metaphone",
                )

        # 4. Edit distance matching
        edit_matches = self._find_edit_distance_matches(text)
        for form, entry, base_confidence in edit_matches:
            if entry.id not in candidates or candidates[entry.id].confidence < base_confidence:
                candidates[entry.id] = MatchCandidate(
                    entry=entry,
                    confidence=base_confidence,
                    matched_form=form,
                    match_type="edit_distance",
                )

        # Sort by confidence and return top candidates
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: c.confidence,
            reverse=True,
        )
        return sorted_candidates[:max_candidates]

    def _find_metaphone_matches(
        self,
        text: str,
    ) -> list[tuple[str, EntryWithRelations, float]]:
        """Find matches using Double Metaphone phonetic algorithm.

        Matches against both primary and alternate metaphone codes with
        different confidence scores:
        - primary-primary match: base confidence 0.8
        - primary-alternate match: base confidence 0.7
        - alternate-alternate match: base confidence 0.6

        Args:
            text: Text to match.

        Returns:
            List of (matched_form, entry, confidence) tuples.
        """
        matches: list[tuple[str, EntryWithRelations, float]] = []
        seen: set[tuple[str, str]] = set()  # (form, entry_id) to avoid duplicates

        # Handle empty string edge case
        if not text or not text.strip():
            return []

        # Get metaphone for each word in text
        words = text.split()
        for word in words:
            clean_word = re.sub(r"[^a-zA-Z]", "", word)
            if len(clean_word) < 2:
                continue

            try:
                input_primary, input_alternate = doublemetaphone(clean_word)
            except Exception:
                continue

            # Skip if metaphone returned empty strings (can happen with special input)
            if not input_primary and not input_alternate:
                continue

            # Match input primary against indexed primary (highest confidence: 0.8)
            if input_primary and input_primary in self._metaphone_primary:
                for form, entry in self._metaphone_primary[input_primary]:
                    key = (form, entry.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    # Calculate similarity using rapidfuzz with empty string protection
                    if not text or not form:
                        continue
                    similarity = fuzz.ratio(text.lower(), form.lower()) / 100.0
                    confidence = 0.8 * similarity
                    if confidence > 0.35:
                        matches.append((form, entry, confidence))

            # Match input primary against indexed alternate (medium confidence: 0.7)
            if input_primary and input_primary in self._metaphone_alternate:
                for form, entry in self._metaphone_alternate[input_primary]:
                    key = (form, entry.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    if not text or not form:
                        continue
                    similarity = fuzz.ratio(text.lower(), form.lower()) / 100.0
                    confidence = 0.7 * similarity
                    if confidence > 0.35:
                        matches.append((form, entry, confidence))

            # Match input alternate against indexed primary (medium confidence: 0.7)
            if input_alternate and input_alternate in self._metaphone_primary:
                for form, entry in self._metaphone_primary[input_alternate]:
                    key = (form, entry.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    if not text or not form:
                        continue
                    similarity = fuzz.ratio(text.lower(), form.lower()) / 100.0
                    confidence = 0.7 * similarity
                    if confidence > 0.35:
                        matches.append((form, entry, confidence))

            # Match input alternate against indexed alternate (lower confidence: 0.6)
            if input_alternate and input_alternate in self._metaphone_alternate:
                for form, entry in self._metaphone_alternate[input_alternate]:
                    key = (form, entry.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    if not text or not form:
                        continue
                    similarity = fuzz.ratio(text.lower(), form.lower()) / 100.0
                    confidence = 0.6 * similarity
                    if confidence > 0.3:
                        matches.append((form, entry, confidence))

        return matches

    def _find_edit_distance_matches(
        self,
        text: str,
        max_distance: int = 3,
    ) -> list[tuple[str, EntryWithRelations, float]]:
        """Find matches using Levenshtein edit distance via rapidfuzz.

        Only considers matches with edit distance <= max_distance.

        Args:
            text: Text to match.
            max_distance: Maximum edit distance to consider.

        Returns:
            List of (matched_form, entry, confidence) tuples.
        """
        matches: list[tuple[str, EntryWithRelations, float]] = []

        # Handle empty string edge case
        if not text or not text.strip():
            return []

        text_lower = text.lower()

        # Validate max_distance to prevent excessive computation
        if max_distance < 0:
            max_distance = 0
        # Cap at reasonable limit to prevent performance issues
        max_distance = min(max_distance, 10)

        # Check all canonical forms using rapidfuzz Levenshtein
        for canonical_lower, entry in self._canonical_lower.items():
            # Skip empty canonical forms
            if not canonical_lower:
                continue
            distance = Levenshtein.distance(text_lower, canonical_lower)
            if distance <= max_distance and distance > 0:  # Skip exact matches (handled elsewhere)
                max_len = max(len(text), len(canonical_lower))
                # Defensive check: max_len should never be 0 at this point, but guard anyway
                if max_len > 0:
                    confidence = 0.6 * (1 - distance / max_len)
                    if confidence > 0.3:
                        matches.append((entry.canonical, entry, confidence))

        # Check all alias forms using rapidfuzz Levenshtein
        for alias_lower, entry in self._alias_lower.items():
            # Skip empty alias forms
            if not alias_lower:
                continue
            distance = Levenshtein.distance(text_lower, alias_lower)
            if distance <= max_distance and distance > 0:
                max_len = max(len(text), len(alias_lower))
                # Defensive check: max_len should never be 0 at this point, but guard anyway
                if max_len > 0:
                    confidence = 0.55 * (1 - distance / max_len)  # Slightly lower for aliases
                    if confidence > 0.3:
                        # Find the original alias text
                        for alias in entry.aliases:
                            if alias.alias.lower() == alias_lower:
                                matches.append((alias.alias, entry, confidence))
                                break

        return matches

    def find_best_match(
        self,
        text: str,
        threshold: float = 0.6,
    ) -> EntryWithRelations | None:
        """Return best matching entry if confidence >= threshold.

        Args:
            text: Text to match.
            threshold: Minimum confidence threshold.

        Returns:
            Best matching entry if confidence meets threshold, None otherwise.
        """
        # Validate threshold to prevent logic errors
        if threshold < 0.0:
            threshold = 0.0
        elif threshold > 1.0:
            threshold = 1.0

        candidates = self.find_candidates(text, max_candidates=1)
        if candidates and candidates[0].confidence >= threshold:
            return candidates[0].entry
        return None

    def match_spans(self, transcript_text: str) -> list[SpanMatch]:
        """Find all potential proper noun spans in transcript and match to dictionary.

        Uses heuristics to identify likely proper noun spans:
        - Capitalized words (not at sentence start)
        - Sequences of capitalized words
        - Words matching phonetic codes of entries

        Args:
            transcript_text: Full transcript text to search.

        Returns:
            List of SpanMatch objects containing span info and candidate matches.
        """
        if not transcript_text:
            return []

        matches: list[SpanMatch] = []
        seen_spans: set[tuple[int, int]] = set()

        # Strategy 1: Find capitalized words/phrases (not sentence start)
        cap_spans = self._find_capitalized_spans(transcript_text)
        for span_text, start, end in cap_spans:
            if (start, end) in seen_spans:
                continue
            candidates = self.find_candidates(span_text)
            if candidates:
                matches.append(SpanMatch(
                    span=span_text,
                    start=start,
                    end=end,
                    candidates=candidates,
                ))
                seen_spans.add((start, end))

        # Strategy 2: Find words matching phonetic codes of any entry
        phonetic_spans = self._find_phonetic_matches_in_text(transcript_text)
        for span_text, start, end, candidates in phonetic_spans:
            if (start, end) in seen_spans:
                continue
            if candidates:
                matches.append(SpanMatch(
                    span=span_text,
                    start=start,
                    end=end,
                    candidates=candidates,
                ))
                seen_spans.add((start, end))

        # Sort by position in text
        matches.sort(key=lambda m: m.start)
        return matches

    def _find_capitalized_spans(
        self,
        text: str,
    ) -> list[tuple[str, int, int]]:
        """Find capitalized words/phrases that might be proper nouns.

        Args:
            text: Text to search.

        Returns:
            List of (span_text, start_pos, end_pos) tuples.
        """
        spans: list[tuple[str, int, int]] = []

        # Pattern for capitalized words
        # Matches sequences like "Ron Chernow" or single capitalized words
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"

        for match in re.finditer(pattern, text):
            span_text = match.group(1)
            start = match.start(1)
            end = match.end(1)

            # Skip if at very start of text (likely sentence start)
            if start == 0:
                continue

            # Skip if preceded by sentence-ending punctuation
            if start > 0:
                before_char_idx = start - 1
                # Skip whitespace to find actual preceding char
                while before_char_idx > 0 and text[before_char_idx] in " \t\n":
                    before_char_idx -= 1
                if before_char_idx >= 0 and text[before_char_idx] in ".!?":
                    continue

            spans.append((span_text, start, end))

        return spans

    def _find_phonetic_matches_in_text(
        self,
        text: str,
    ) -> list[tuple[str, int, int, list[MatchCandidate]]]:
        """Find words in text that phonetically match dictionary entries.

        Args:
            text: Text to search.

        Returns:
            List of (span_text, start_pos, end_pos, candidates) tuples.
        """
        matches: list[tuple[str, int, int, list[MatchCandidate]]] = []

        # Handle empty string edge case
        if not text:
            return []

        # Find all word boundaries
        word_pattern = r"\b([A-Za-z]+)\b"

        for match in re.finditer(word_pattern, text):
            word = match.group(1)
            start = match.start(1)
            end = match.end(1)

            # Skip very short words
            if len(word) < 3:
                continue

            # Check if this word's phonetic codes match any entry
            try:
                word_primary, word_alternate = doublemetaphone(word)
            except Exception:
                continue

            # Skip if metaphone returned empty strings
            if not word_primary and not word_alternate:
                continue

            has_phonetic_match = (
                (word_primary and word_primary in self._metaphone_primary)
                or (word_primary and word_primary in self._metaphone_alternate)
                or (word_alternate and word_alternate in self._metaphone_primary)
                or (word_alternate and word_alternate in self._metaphone_alternate)
            )

            if has_phonetic_match:
                candidates = self.find_candidates(word)
                if candidates and candidates[0].confidence >= 0.5:
                    matches.append((word, start, end, candidates))

        return matches


def create_matcher(entries: list[EntryWithRelations]) -> CandidateMatcher:
    """Factory function to create a CandidateMatcher.

    Args:
        entries: Dictionary entries to match against.

    Returns:
        Configured CandidateMatcher instance.
    """
    return CandidateMatcher(entries)

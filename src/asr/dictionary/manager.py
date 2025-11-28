"""High-level dictionary operations for ASR bias list management.

This module provides the DictionaryManager class for CRUD operations on
dictionary entries, including bulk import/export and usage tracking.

Storage: Entries are stored in ~/.asr/dictionaries/entries.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from asr.config import CONFIG_DIR, _file_lock
from asr.dictionary.models import (
    Alias,
    DictionaryStats,
    EntryType,
    EntryWithRelations,
    Pronunciation,
    TierLevel,
)

logger = logging.getLogger(__name__)

# Storage paths
DICTIONARIES_DIR = CONFIG_DIR / "dictionaries"
ENTRIES_FILE = DICTIONARIES_DIR / "entries.json"
CONTEXTS_DIR = DICTIONARIES_DIR / "contexts"


def _ensure_dirs() -> None:
    """Ensure dictionary storage directories exist."""
    DICTIONARIES_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)


def _validate_id_or_canonical(value: str) -> bool:
    """Validate an ID or canonical string for safety.

    Prevents path traversal and injection attacks.
    """
    if not value or len(value) > 200:
        return False
    # Block obvious path traversal
    if ".." in value or "/" in value or "\\" in value:
        return False
    return True


class DictionaryManager:
    """High-level manager for ASR dictionary operations.

    Provides CRUD operations for dictionary entries with persistence
    to JSON storage. Thread-safe via file locking.

    Usage:
        manager = DictionaryManager()
        entry = manager.add_entry(
            canonical="Ron Chernow",
            type="person",
            tier="B",
            boost_weight=1.2,
            contexts=["biography"],
        )
        found = manager.search("chernow")
    """

    def __init__(self) -> None:
        """Initialize the dictionary manager."""
        _ensure_dirs()
        self._entries: dict[str, EntryWithRelations] = {}
        self._canonical_index: dict[str, str] = {}  # canonical.lower() -> id
        self._alias_index: dict[str, str] = {}  # alias.lower() -> id
        self._load()

    def _load(self) -> None:
        """Load entries from storage."""
        if not ENTRIES_FILE.exists():
            self._entries = {}
            self._canonical_index = {}
            self._alias_index = {}
            return

        try:
            with _file_lock("dictionary-entries"):
                data = json.loads(ENTRIES_FILE.read_text(encoding="utf-8"))
                self._entries = {}
                self._canonical_index = {}
                self._alias_index = {}

                for entry_data in data.get("entries", []):
                    try:
                        entry_with_rel = EntryWithRelations(**entry_data)
                        self._entries[entry_with_rel.id] = entry_with_rel
                        self._canonical_index[entry_with_rel.canonical.lower()] = entry_with_rel.id
                        for alias in entry_with_rel.aliases:
                            self._alias_index[alias.alias.lower()] = entry_with_rel.id
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid entry: {e}")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to load dictionary: {e}")
            self._entries = {}
            self._canonical_index = {}
            self._alias_index = {}

    def _save(self) -> None:
        """Save entries to storage."""
        _ensure_dirs()

        with _file_lock("dictionary-entries"):
            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "entries": [
                    entry.model_dump(mode="json")
                    for entry in self._entries.values()
                ],
            }
            ENTRIES_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Restrictive permissions
            ENTRIES_FILE.chmod(0o600)

    def add_entry(
        self,
        canonical: str,
        type: EntryType,
        tier: TierLevel = "D",
        boost_weight: float = 1.0,
        aliases: list[str] | None = None,
        pronunciations: list[str] | None = None,
        contexts: list[str] | None = None,
        display: str | None = None,
        source: str | None = None,
    ) -> EntryWithRelations:
        """Add a new dictionary entry.

        Args:
            canonical: The canonical (correct) form of the term
            type: Entry type (person, org, product, event, location, jargon, misc)
            tier: Priority tier (A-H, default D)
            boost_weight: Additional weight multiplier (0.0-3.0)
            aliases: Alternative forms or spellings
            pronunciations: Phonetic pronunciations (IPA or simplified)
            contexts: Context tags (e.g., "work", "running", "asr_dev")
            display: Optional display form
            source: Source of this entry

        Returns:
            The created EntryWithRelations

        Raises:
            ValueError: If canonical already exists or validation fails
        """
        # Normalize inputs
        canonical = canonical.strip()
        if not canonical:
            raise ValueError("Canonical form cannot be empty")

        # Check for duplicates
        canonical_lower = canonical.lower()
        if canonical_lower in self._canonical_index:
            raise ValueError(f"Entry with canonical '{canonical}' already exists")

        # Generate ID
        entry_id = str(uuid4())
        now = datetime.now()

        # Build aliases
        alias_objs: list[Alias] = []
        if aliases:
            for alias in aliases:
                alias = alias.strip()
                if alias and alias.lower() != canonical_lower:
                    alias_objs.append(Alias(
                        entry_id=entry_id,
                        alias=alias,
                        is_common_misspelling=False,
                    ))
                    self._alias_index[alias.lower()] = entry_id

        # Build pronunciations
        pron_objs: list[Pronunciation] = []
        if pronunciations:
            for pron in pronunciations:
                pron = pron.strip()
                if pron:
                    pron_objs.append(Pronunciation(
                        entry_id=entry_id,
                        ipa=pron,
                    ))

        # Build contexts (just strings in the new model)
        ctx_list: list[str] = []
        if contexts:
            for ctx in contexts:
                ctx = ctx.strip()
                if ctx:
                    ctx_list.append(ctx)

        entry_with_rel = EntryWithRelations(
            id=entry_id,
            canonical=canonical,
            display=display,
            type=type,
            tier=tier,
            boost_weight=boost_weight,
            source=source,
            created_at=now,
            updated_at=now,
            aliases=alias_objs,
            pronunciations=pron_objs,
            contexts=ctx_list,
        )

        # Store
        self._entries[entry_id] = entry_with_rel
        self._canonical_index[canonical_lower] = entry_id

        self._save()
        logger.info(f"Added dictionary entry: {canonical} ({entry_id})")

        return entry_with_rel

    def get_entry(self, id_or_canonical: str) -> EntryWithRelations | None:
        """Get an entry by ID or canonical form.

        Args:
            id_or_canonical: Entry ID or canonical form

        Returns:
            EntryWithRelations if found, None otherwise
        """
        if not _validate_id_or_canonical(id_or_canonical):
            return None

        # Try direct ID lookup
        if id_or_canonical in self._entries:
            return self._entries[id_or_canonical]

        # Try canonical index
        lower = id_or_canonical.lower()
        if lower in self._canonical_index:
            entry_id = self._canonical_index[lower]
            return self._entries.get(entry_id)

        # Try alias index
        if lower in self._alias_index:
            entry_id = self._alias_index[lower]
            return self._entries.get(entry_id)

        return None

    def update_entry(self, id: str, **kwargs: Any) -> EntryWithRelations:
        """Update an existing entry.

        Args:
            id: Entry ID
            **kwargs: Fields to update (canonical, type, tier, boost_weight,
                     display, source, aliases, pronunciations, contexts)

        Returns:
            Updated EntryWithRelations

        Raises:
            ValueError: If entry not found or validation fails
        """
        if not _validate_id_or_canonical(id):
            raise ValueError(f"Invalid entry ID: {id}")

        if id not in self._entries:
            raise ValueError(f"Entry not found: {id}")

        entry = self._entries[id]
        now = datetime.now()

        # Build update dict
        update_data = entry.model_dump()

        # Handle canonical change (need to update index)
        if "canonical" in kwargs:
            new_canonical = kwargs["canonical"].strip()
            old_lower = entry.canonical.lower()
            new_lower = new_canonical.lower()

            if new_lower != old_lower:
                if new_lower in self._canonical_index:
                    raise ValueError(f"Entry with canonical '{new_canonical}' already exists")
                del self._canonical_index[old_lower]
                self._canonical_index[new_lower] = id

            update_data["canonical"] = new_canonical

        # Update simple fields
        for field in ["type", "tier", "boost_weight", "display", "source", "notes", "language"]:
            if field in kwargs:
                update_data[field] = kwargs[field]

        # Handle aliases replacement
        if "aliases" in kwargs:
            # Remove old alias index entries
            for alias in entry.aliases:
                self._alias_index.pop(alias.alias.lower(), None)

            # Add new aliases
            canonical_lower = update_data["canonical"].lower()
            new_aliases = []
            for alias in kwargs["aliases"]:
                if isinstance(alias, str):
                    alias = alias.strip()
                    if alias and alias.lower() != canonical_lower:
                        new_aliases.append(Alias(
                            entry_id=id,
                            alias=alias,
                            is_common_misspelling=False,
                        ).model_dump())
                        self._alias_index[alias.lower()] = id
                elif isinstance(alias, Alias):
                    new_aliases.append(alias.model_dump())
                    self._alias_index[alias.alias.lower()] = id
                elif isinstance(alias, dict):
                    new_aliases.append(alias)
                    self._alias_index[alias["alias"].lower()] = id

            update_data["aliases"] = new_aliases

        # Handle pronunciations replacement
        if "pronunciations" in kwargs:
            new_prons = []
            for pron in kwargs["pronunciations"]:
                if isinstance(pron, str):
                    pron = pron.strip()
                    if pron:
                        new_prons.append(Pronunciation(
                            entry_id=id,
                            ipa=pron,
                        ).model_dump())
                elif isinstance(pron, Pronunciation):
                    new_prons.append(pron.model_dump())
                elif isinstance(pron, dict):
                    new_prons.append(pron)
            update_data["pronunciations"] = new_prons

        # Handle contexts replacement
        if "contexts" in kwargs:
            new_ctxs = []
            for ctx in kwargs["contexts"]:
                if isinstance(ctx, str):
                    ctx = ctx.strip()
                    if ctx:
                        new_ctxs.append(ctx)
            update_data["contexts"] = new_ctxs

        update_data["updated_at"] = now

        # Rebuild entry
        updated_entry = EntryWithRelations(**update_data)
        self._entries[id] = updated_entry
        self._save()
        logger.info(f"Updated dictionary entry: {updated_entry.canonical} ({id})")

        return updated_entry

    def remove_entry(self, id_or_canonical: str) -> bool:
        """Remove an entry from the dictionary.

        Args:
            id_or_canonical: Entry ID or canonical form

        Returns:
            True if removed, False if not found
        """
        entry = self.get_entry(id_or_canonical)
        if not entry:
            return False

        # Remove from indices
        self._canonical_index.pop(entry.canonical.lower(), None)
        for alias in entry.aliases:
            self._alias_index.pop(alias.alias.lower(), None)

        # Remove entry
        del self._entries[entry.id]
        self._save()
        logger.info(f"Removed dictionary entry: {entry.canonical} ({entry.id})")

        return True

    def search(self, query: str, limit: int = 20) -> list[EntryWithRelations]:
        """Search for entries matching a query.

        Searches canonical forms, aliases, and display forms.
        Results are sorted by relevance (exact matches first).

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of matching entries
        """
        if not query or len(query) > 200:
            return []

        query_lower = query.lower().strip()
        if not query_lower:
            return []

        results: list[tuple[int, EntryWithRelations]] = []  # (score, entry)

        for entry in self._entries.values():
            score = 0

            # Exact canonical match (highest score)
            if entry.canonical.lower() == query_lower:
                score = 100
            # Canonical starts with query
            elif entry.canonical.lower().startswith(query_lower):
                score = 80
            # Canonical contains query
            elif query_lower in entry.canonical.lower():
                score = 60
            # Exact alias match
            elif any(a.alias.lower() == query_lower for a in entry.aliases):
                score = 90
            # Alias starts with query
            elif any(a.alias.lower().startswith(query_lower) for a in entry.aliases):
                score = 70
            # Alias contains query
            elif any(query_lower in a.alias.lower() for a in entry.aliases):
                score = 50
            # Display contains query
            elif entry.display and query_lower in entry.display.lower():
                score = 40

            if score > 0:
                results.append((score, entry))

        # Sort by score (descending), then by canonical (ascending)
        results.sort(key=lambda x: (-x[0], x[1].canonical.lower()))

        return [entry for _, entry in results[:limit]]

    def add_alias(
        self,
        entry_id: str,
        alias: str,
        is_misspelling: bool = False,
    ) -> EntryWithRelations:
        """Add an alias to an existing entry.

        Args:
            entry_id: Entry ID
            alias: Alias to add
            is_misspelling: Whether this alias is a known misspelling

        Returns:
            Updated EntryWithRelations

        Raises:
            ValueError: If entry not found or alias already exists
        """
        entry = self.get_entry(entry_id)
        if not entry:
            raise ValueError(f"Entry not found: {entry_id}")

        alias = alias.strip()
        if not alias:
            raise ValueError("Alias cannot be empty")

        alias_lower = alias.lower()

        # Check if alias already exists for this or another entry
        if alias_lower in self._alias_index:
            existing_id = self._alias_index[alias_lower]
            if existing_id == entry_id:
                # Already exists for this entry, just return
                return entry
            else:
                raise ValueError(f"Alias '{alias}' already exists for another entry")

        if alias_lower == entry.canonical.lower():
            raise ValueError("Alias cannot be the same as canonical form")

        # Add new alias - need to rebuild entry with new alias
        new_aliases = list(entry.aliases)
        new_aliases.append(Alias(
            entry_id=entry_id,
            alias=alias,
            is_common_misspelling=is_misspelling,
        ))

        return self.update_entry(entry_id, aliases=new_aliases)

    def maybe_add_alias(
        self,
        entry_id: str,
        alias: str,
        min_occurrences: int = 3,
    ) -> bool:
        """Conditionally add an alias only if seen enough times.

        Tracks occurrences internally and only promotes to a real alias
        after reaching the threshold. Useful for learning from transcriptions.

        Note: This simplified implementation tracks via a pending file since
        the Alias model doesn't have an occurrence_count field.

        Args:
            entry_id: Entry ID
            alias: Candidate alias
            min_occurrences: Minimum times seen before adding

        Returns:
            True if alias was added (threshold reached), False otherwise
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False

        alias = alias.strip()
        if not alias:
            return False

        alias_lower = alias.lower()

        # Check if already an alias
        if alias_lower in self._alias_index:
            return False

        if alias_lower == entry.canonical.lower():
            return False

        # Load/update pending aliases tracker
        pending_file = DICTIONARIES_DIR / "pending_aliases.json"
        pending: dict[str, dict[str, int]] = {}

        if pending_file.exists():
            try:
                pending = json.loads(pending_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                pending = {}

        # Key is entry_id:alias_lower
        key = f"{entry_id}:{alias_lower}"
        count = pending.get(key, 0) + 1
        pending[key] = count

        # Save pending
        pending_file.write_text(json.dumps(pending, indent=2), encoding="utf-8")
        pending_file.chmod(0o600)

        # Check if threshold reached
        if count >= min_occurrences:
            # Promote to real alias
            try:
                self.add_alias(entry_id, alias, is_misspelling=True)
                # Remove from pending
                del pending[key]
                pending_file.write_text(json.dumps(pending, indent=2), encoding="utf-8")
                return True
            except ValueError:
                return False

        return False

    def record_usage(self, entry_id: str) -> None:
        """Record usage of an entry (increment occurrence, update last_seen).

        Args:
            entry_id: Entry ID
        """
        if entry_id not in self._entries:
            return

        entry = self._entries[entry_id]
        now = datetime.now()

        # Update entry with new counts
        self.update_entry(
            entry_id,
            occurrence_count=entry.occurrence_count + 1,
        )

        # Also update last_seen_at directly
        entry = self._entries[entry_id]
        update_data = entry.model_dump()
        update_data["last_seen_at"] = now
        update_data["updated_at"] = now
        self._entries[entry_id] = EntryWithRelations(**update_data)
        self._save()

    def import_from_json(self, path: Path) -> int:
        """Bulk import entries from a JSON file.

        Expected format:
        {
            "entries": [
                {
                    "canonical": "...",
                    "type": "person",
                    "tier": "B",
                    "boost_weight": 1.0,
                    "aliases": ["..."],
                    "pronunciations": ["..."],
                    "contexts": ["..."],
                    "display": "...",
                    "source": "..."
                },
                ...
            ]
        }

        Args:
            path: Path to JSON file

        Returns:
            Number of entries imported

        Raises:
            ValueError: If file is invalid
        """
        if not path.exists():
            raise ValueError(f"File not found: {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        entries_data = data.get("entries", [])
        if not isinstance(entries_data, list):
            raise ValueError("Expected 'entries' to be a list")

        count = 0
        for entry_data in entries_data:
            try:
                canonical = entry_data.get("canonical", "").strip()
                if not canonical:
                    logger.warning("Skipping entry with empty canonical")
                    continue

                # Skip if already exists
                if canonical.lower() in self._canonical_index:
                    logger.debug(f"Skipping duplicate: {canonical}")
                    continue

                self.add_entry(
                    canonical=canonical,
                    type=entry_data.get("type", "misc"),
                    tier=entry_data.get("tier", "E"),
                    boost_weight=entry_data.get("boost_weight", 1.0),
                    aliases=entry_data.get("aliases", []),
                    pronunciations=entry_data.get("pronunciations", []),
                    contexts=entry_data.get("contexts", []),
                    display=entry_data.get("display"),
                    source=entry_data.get("source"),
                )
                count += 1

            except (ValueError, ValidationError) as e:
                logger.warning(f"Skipping invalid entry '{entry_data.get('canonical', '?')}': {e}")

        logger.info(f"Imported {count} entries from {path}")
        return count

    def export_to_json(
        self,
        path: Path,
        tier: str | None = None,
        context: str | None = None,
    ) -> int:
        """Export entries to a JSON file.

        Args:
            path: Output path
            tier: Filter by tier (e.g., "A", "B") - exports this tier and above
            context: Filter by context tag

        Returns:
            Number of entries exported
        """
        entries_to_export = []

        # Determine tier cutoff
        tier_order = ["A", "B", "C", "D", "E", "F", "G", "H"]
        tier_cutoff_idx = len(tier_order) - 1
        if tier:
            try:
                tier_cutoff_idx = tier_order.index(tier)
            except ValueError:
                pass

        for entry in self._entries.values():
            # Filter by tier
            if tier:
                entry_tier_idx = tier_order.index(entry.tier)
                if entry_tier_idx > tier_cutoff_idx:
                    continue

            # Filter by context
            if context and not entry.has_context(context):
                continue

            # Build export format
            export_data = {
                "canonical": entry.canonical,
                "type": entry.type,
                "tier": entry.tier,
                "boost_weight": entry.boost_weight,
                "aliases": [a.alias for a in entry.aliases],
                "pronunciations": [p.ipa for p in entry.pronunciations if p.ipa],
                "contexts": entry.contexts,
            }
            if entry.display:
                export_data["display"] = entry.display
            if entry.source:
                export_data["source"] = entry.source

            entries_to_export.append(export_data)

        # Sort by tier, then canonical
        entries_to_export.sort(key=lambda x: (x["tier"], x["canonical"].lower()))

        output = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "filters": {"tier": tier, "context": context},
            "count": len(entries_to_export),
            "entries": entries_to_export,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        path.chmod(0o600)

        logger.info(f"Exported {len(entries_to_export)} entries to {path}")
        return len(entries_to_export)

    def get_stats(self) -> dict[str, Any]:
        """Get dictionary statistics.

        Returns:
            Dictionary with stats: total_entries, entries_by_tier,
            entries_by_type, entries_by_context, total_aliases, etc.
        """
        stats = DictionaryStats()

        for entry in self._entries.values():
            stats.total_entries += 1

            # Count by tier
            stats.entries_by_tier[entry.tier] = stats.entries_by_tier.get(entry.tier, 0) + 1

            # Count by type
            stats.entries_by_type[entry.type] = stats.entries_by_type.get(entry.type, 0) + 1

            # Count by context
            for ctx in entry.contexts:
                stats.entries_by_context[ctx] = (
                    stats.entries_by_context.get(ctx, 0) + 1
                )

            # Count aliases and pronunciations
            stats.total_aliases += len(entry.aliases)
            stats.total_pronunciations += len(entry.pronunciations)

            # Track last update
            if stats.last_updated is None or entry.updated_at > stats.last_updated:
                stats.last_updated = entry.updated_at

        return stats.model_dump()

    def get_all_entries(self) -> list[EntryWithRelations]:
        """Get all dictionary entries.

        Returns:
            List of all entries
        """
        return list(self._entries.values())

    def get_entries_by_context(self, context: str) -> list[EntryWithRelations]:
        """Get all entries with a specific context tag.

        Args:
            context: Context tag to filter by

        Returns:
            List of matching entries
        """
        return [
            entry for entry in self._entries.values()
            if entry.has_context(context)
        ]

    def get_entries_by_tier(self, tier: TierLevel) -> list[EntryWithRelations]:
        """Get all entries with a specific tier.

        Args:
            tier: Tier to filter by

        Returns:
            List of matching entries
        """
        return [
            entry for entry in self._entries.values()
            if entry.tier == tier
        ]

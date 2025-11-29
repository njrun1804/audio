"""SQLite database operations for ASR dictionary.

This module provides all database operations for the ASR dictionary system,
including CRUD operations, search, and bulk operations.

Uses WAL mode for concurrent reads and file locking for write safety.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from asr.config import _file_lock
from asr.dictionary.models import (
    Alias,
    DictionaryStats,
    Entry,
    EntryType,
    EntryWithRelations,
    Pronunciation,
    SearchResult,
    TierLevel,
    TIER_WEIGHTS,
)


# Database path configuration
DB_DIR = Path.home() / ".asr" / "dictionaries"
DB_PATH = DB_DIR / "dictionary.db"

# Track whether DB has been initialized this session
_db_initialized = False


def _ensure_db_dir() -> None:
    """Ensure the database directory exists with proper permissions."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    # Set restrictive permissions on the directory
    os.chmod(DB_DIR, 0o700)


def get_connection() -> sqlite3.Connection:
    """Get a database connection with proper configuration.

    Enables WAL mode for concurrent reads and configures foreign keys.
    Auto-initializes the database on first access.

    Returns:
        sqlite3.Connection configured for the dictionary database
    """
    global _db_initialized

    _ensure_db_dir()

    # Initialize DB if needed (first access this session)
    if not _db_initialized:
        init_db()
        _db_initialized = True

    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row

    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL mode is persistent, only needs to be set once during init
    # But checking/setting it is idempotent and fast, so we do it for safety
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")  # Good balance of safety/performance
    conn.execute("PRAGMA cache_size = -64000")  # 64MB cache

    return conn


@contextmanager
def db_connection() -> Iterator[sqlite3.Connection]:
    """Context manager for database connections.

    Yields:
        sqlite3.Connection that will be properly closed on exit
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def db_transaction() -> Iterator[sqlite3.Connection]:
    """Context manager for database transactions with automatic commit/rollback.

    Yields:
        sqlite3.Connection within a transaction
    """
    with db_connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def init_db() -> None:
    """Initialize the database schema.

    Creates all required tables and indexes if they don't exist.
    Safe to call multiple times - uses IF NOT EXISTS.
    """
    from asr.dictionary.migrations import create_schema

    _ensure_db_dir()

    with _file_lock("dictionary-db"):
        conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
        try:
            create_schema(conn)
            conn.commit()
        finally:
            conn.close()

    # Set restrictive permissions on the database file
    if DB_PATH.exists():
        os.chmod(DB_PATH, 0o600)


# =============================================================================
# Helper Functions
# =============================================================================


def _row_to_entry(row: sqlite3.Row) -> Entry:
    """Convert a database row to an Entry model."""
    return Entry(
        id=row["id"],
        canonical=row["canonical"],
        display=row["display"],
        type=row["type"],
        tier=row["tier"],
        boost_weight=row["boost_weight"],
        language=row["language"],
        occurrence_count=row["occurrence_count"],
        last_seen_at=datetime.fromisoformat(row["last_seen_at"]) if row["last_seen_at"] else None,
        source=row["source"],
        notes=row["notes"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_alias(row: sqlite3.Row) -> Alias:
    """Convert a database row to an Alias model."""
    return Alias(
        id=row["id"],
        entry_id=row["entry_id"],
        alias=row["alias"],
        is_common_misspelling=bool(row["is_common_misspelling"]),
    )


def _row_to_pronunciation(row: sqlite3.Row) -> Pronunciation:
    """Convert a database row to a Pronunciation model."""
    return Pronunciation(
        id=row["id"],
        entry_id=row["entry_id"],
        ipa=row["ipa"],
        phoneme_sequence=row["phoneme_sequence"],
        language=row["language"],
        variant=row["variant"],
    )


def _load_entry_relations(
    conn: sqlite3.Connection, entry_id: str
) -> tuple[list[Alias], list[Pronunciation], list[str]]:
    """Load all relations for an entry."""
    # Load aliases
    cursor = conn.execute(
        "SELECT id, entry_id, alias, is_common_misspelling FROM aliases WHERE entry_id = ?",
        (entry_id,)
    )
    aliases = [_row_to_alias(row) for row in cursor.fetchall()]

    # Load pronunciations
    cursor = conn.execute(
        """SELECT id, entry_id, ipa, phoneme_sequence, language, variant
           FROM pronunciations WHERE entry_id = ?""",
        (entry_id,)
    )
    pronunciations = [_row_to_pronunciation(row) for row in cursor.fetchall()]

    # Load contexts
    cursor = conn.execute(
        "SELECT context FROM entry_contexts WHERE entry_id = ?",
        (entry_id,)
    )
    contexts = [row["context"] for row in cursor.fetchall()]

    return aliases, pronunciations, contexts


def _entry_to_entry_with_relations(conn: sqlite3.Connection, entry: Entry) -> EntryWithRelations:
    """Load relations and create EntryWithRelations."""
    aliases, pronunciations, contexts = _load_entry_relations(conn, entry.id)
    return EntryWithRelations.from_entry(entry, aliases, pronunciations, contexts)


# =============================================================================
# CRUD Operations
# =============================================================================


def create_entry(entry: EntryWithRelations) -> EntryWithRelations:
    """Create a new dictionary entry with all relations.

    Args:
        entry: The entry to create (id will be generated if empty)

    Returns:
        The created entry with assigned ID
    """
    # Generate ID if not provided
    if not entry.id:
        entry = entry.model_copy(update={"id": str(uuid4())})

    now = datetime.now()

    with _file_lock("dictionary-db"):
        with db_transaction() as conn:
            # Insert the main entry
            conn.execute(
                """
                INSERT INTO entries (
                    id, canonical, display, type, tier, boost_weight,
                    language, occurrence_count, last_seen_at, source, notes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.canonical,
                    entry.display,
                    entry.type,
                    entry.tier,
                    entry.boost_weight,
                    entry.language,
                    entry.occurrence_count,
                    entry.last_seen_at.isoformat() if entry.last_seen_at else None,
                    entry.source,
                    entry.notes,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )

            # Insert aliases
            for alias in entry.aliases:
                conn.execute(
                    "INSERT INTO aliases (entry_id, alias, is_common_misspelling) VALUES (?, ?, ?)",
                    (entry.id, alias.alias, alias.is_common_misspelling),
                )

            # Insert pronunciations
            for pron in entry.pronunciations:
                conn.execute(
                    """
                    INSERT INTO pronunciations (entry_id, ipa, phoneme_sequence, language, variant)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (entry.id, pron.ipa, pron.phoneme_sequence, pron.language, pron.variant),
                )

            # Insert contexts
            for context in entry.contexts:
                conn.execute(
                    "INSERT INTO entry_contexts (entry_id, context) VALUES (?, ?)",
                    (entry.id, context),
                )

    # Return updated entry with timestamps
    return entry.model_copy(update={"created_at": now, "updated_at": now})


def get_entry(entry_id: str) -> EntryWithRelations | None:
    """Get an entry by ID with all relations.

    Args:
        entry_id: The UUID of the entry

    Returns:
        EntryWithRelations if found, None otherwise
    """
    with db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, canonical, display, type, tier, boost_weight,
                   language, occurrence_count, last_seen_at, source, notes,
                   created_at, updated_at
            FROM entries WHERE id = ?
            """,
            (entry_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        entry = _row_to_entry(row)
        return _entry_to_entry_with_relations(conn, entry)


def get_entry_by_canonical(canonical: str, language: str = "en") -> EntryWithRelations | None:
    """Get an entry by its canonical name.

    Args:
        canonical: The canonical form of the entry
        language: Language code (default "en")

    Returns:
        EntryWithRelations if found, None otherwise
    """
    with db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, canonical, display, type, tier, boost_weight,
                   language, occurrence_count, last_seen_at, source, notes,
                   created_at, updated_at
            FROM entries WHERE canonical = ? AND language = ?
            """,
            (canonical, language),
        )
        row = cursor.fetchone()
        if not row:
            return None

        entry = _row_to_entry(row)
        return _entry_to_entry_with_relations(conn, entry)


def update_entry(entry: EntryWithRelations) -> EntryWithRelations:
    """Update an existing entry and all its relations.

    Args:
        entry: The entry with updated data

    Returns:
        The updated entry

    Raises:
        ValueError: If entry does not exist
    """
    now = datetime.now()

    with _file_lock("dictionary-db"):
        with db_transaction() as conn:
            # Check entry exists
            cursor = conn.execute("SELECT id FROM entries WHERE id = ?", (entry.id,))
            if not cursor.fetchone():
                raise ValueError(f"Entry not found: {entry.id}")

            # Update main entry
            conn.execute(
                """
                UPDATE entries SET
                    canonical = ?, display = ?, type = ?, tier = ?, boost_weight = ?,
                    language = ?, occurrence_count = ?, last_seen_at = ?,
                    source = ?, notes = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    entry.canonical,
                    entry.display,
                    entry.type,
                    entry.tier,
                    entry.boost_weight,
                    entry.language,
                    entry.occurrence_count,
                    entry.last_seen_at.isoformat() if entry.last_seen_at else None,
                    entry.source,
                    entry.notes,
                    now.isoformat(),
                    entry.id,
                ),
            )

            # Replace aliases (delete and re-insert)
            conn.execute("DELETE FROM aliases WHERE entry_id = ?", (entry.id,))
            for alias in entry.aliases:
                conn.execute(
                    "INSERT INTO aliases (entry_id, alias, is_common_misspelling) VALUES (?, ?, ?)",
                    (entry.id, alias.alias, alias.is_common_misspelling),
                )

            # Replace pronunciations
            conn.execute("DELETE FROM pronunciations WHERE entry_id = ?", (entry.id,))
            for pron in entry.pronunciations:
                conn.execute(
                    """
                    INSERT INTO pronunciations (entry_id, ipa, phoneme_sequence, language, variant)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (entry.id, pron.ipa, pron.phoneme_sequence, pron.language, pron.variant),
                )

            # Replace contexts
            conn.execute("DELETE FROM entry_contexts WHERE entry_id = ?", (entry.id,))
            for context in entry.contexts:
                conn.execute(
                    "INSERT INTO entry_contexts (entry_id, context) VALUES (?, ?)",
                    (entry.id, context),
                )

    return entry.model_copy(update={"updated_at": now})


def delete_entry(entry_id: str) -> bool:
    """Delete an entry and all its relations.

    Args:
        entry_id: The UUID of the entry to delete

    Returns:
        True if deleted, False if not found
    """
    with _file_lock("dictionary-db"):
        with db_transaction() as conn:
            # Check entry exists
            cursor = conn.execute("SELECT id FROM entries WHERE id = ?", (entry_id,))
            if not cursor.fetchone():
                return False

            # Delete relations (foreign key cascade should handle this, but be explicit)
            conn.execute("DELETE FROM aliases WHERE entry_id = ?", (entry_id,))
            conn.execute("DELETE FROM pronunciations WHERE entry_id = ?", (entry_id,))
            conn.execute("DELETE FROM entry_contexts WHERE entry_id = ?", (entry_id,))

            # Delete main entry
            conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))

    return True


# =============================================================================
# Search Operations
# =============================================================================


def search_entries(
    query: str,
    limit: int = 20,
    entry_type: EntryType | None = None,
    tier: TierLevel | None = None,
    language: str = "en",
) -> list[SearchResult]:
    """Search entries by canonical name or alias (fuzzy search).

    Uses SQLite's LIKE for fuzzy matching on canonical and alias fields.
    Results are ranked by match quality and tier weight.

    Args:
        query: Search query string
        limit: Maximum number of results
        entry_type: Filter by entry type
        tier: Filter by tier
        language: Filter by language

    Returns:
        List of SearchResult ordered by relevance
    """
    if not query or len(query) < 1:
        return []

    # Normalize query for search
    query_lower = query.lower().strip()
    query_pattern = f"%{query_lower}%"

    results: list[SearchResult] = []

    with db_connection() as conn:
        # Validate limit to prevent SQL injection
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"Invalid limit value: {limit}")

        # Build query with optional filters
        where_clauses = ["e.language = ?"]
        params: list = [language]

        if entry_type:
            where_clauses.append("e.type = ?")
            params.append(entry_type)

        if tier:
            where_clauses.append("e.tier = ?")
            params.append(tier)

        where_sql = " AND ".join(where_clauses)

        # Search canonical names
        # Build query with string formatting for WHERE clause only (safe - trusted sources)
        # All user input goes through parameterized queries
        params_canonical = params + [query_pattern, query_lower, f"{query_lower}%", limit]
        cursor = conn.execute(
            f"""
            SELECT id, canonical, display, type, tier, boost_weight,
                   language, occurrence_count, last_seen_at, source, notes,
                   created_at, updated_at
            FROM entries e
            WHERE {where_sql} AND LOWER(canonical) LIKE ?
            ORDER BY
                CASE
                    WHEN LOWER(canonical) = ? THEN 0
                    WHEN LOWER(canonical) LIKE ? THEN 1
                    ELSE 2
                END,
                tier ASC,
                occurrence_count DESC
            LIMIT ?
            """,
            params_canonical,
        )

        seen_ids = set()
        for row in cursor.fetchall():
            entry = _row_to_entry(row)
            entry_with_relations = _entry_to_entry_with_relations(conn, entry)

            # Calculate score
            if entry.canonical.lower() == query_lower:
                score = 1.0
            elif entry.canonical.lower().startswith(query_lower):
                score = 0.9
            else:
                score = 0.7

            score *= TIER_WEIGHTS.get(entry.tier, 0.5)
            score *= entry.boost_weight

            results.append(SearchResult(
                entry=entry_with_relations,
                score=score,
                matched_on="canonical",
            ))
            seen_ids.add(entry.id)

        # Search aliases
        params_alias = params + [query_pattern, query_lower, f"{query_lower}%", limit]
        cursor = conn.execute(
            f"""
            SELECT e.id, e.canonical, e.display, e.type, e.tier, e.boost_weight,
                   e.language, e.occurrence_count, e.last_seen_at, e.source, e.notes,
                   e.created_at, e.updated_at, a.alias
            FROM entries e
            JOIN aliases a ON e.id = a.entry_id
            WHERE {where_sql} AND LOWER(a.alias) LIKE ?
            ORDER BY
                CASE
                    WHEN LOWER(a.alias) = ? THEN 0
                    WHEN LOWER(a.alias) LIKE ? THEN 1
                    ELSE 2
                END,
                e.tier ASC
            LIMIT ?
            """,
            params_alias,
        )

        for row in cursor.fetchall():
            if row["id"] in seen_ids:
                continue

            entry = _row_to_entry(row)
            entry_with_relations = _entry_to_entry_with_relations(conn, entry)

            # Calculate score (slightly lower for alias matches)
            alias_value = row["alias"].lower()
            if alias_value == query_lower:
                score = 0.95
            elif alias_value.startswith(query_lower):
                score = 0.85
            else:
                score = 0.65

            score *= TIER_WEIGHTS.get(entry.tier, 0.5)
            score *= entry.boost_weight

            results.append(SearchResult(
                entry=entry_with_relations,
                score=score,
                matched_on="alias",
            ))
            seen_ids.add(row["id"])

    # Sort by score and limit
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


# =============================================================================
# Bulk Operations
# =============================================================================


def bulk_create_entries(entries: list[EntryWithRelations]) -> int:
    """Bulk create multiple entries efficiently.

    Uses a single transaction for all entries to improve performance.

    Args:
        entries: List of entries to create

    Returns:
        Number of entries created
    """
    if not entries:
        return 0

    now = datetime.now()
    created = 0

    with _file_lock("dictionary-db"):
        with db_transaction() as conn:
            for entry in entries:
                # Generate ID if not provided
                entry_id = entry.id if entry.id else str(uuid4())

                try:
                    # Insert the main entry
                    conn.execute(
                        """
                        INSERT INTO entries (
                            id, canonical, display, type, tier, boost_weight,
                            language, occurrence_count, last_seen_at, source, notes,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            entry_id,
                            entry.canonical,
                            entry.display,
                            entry.type,
                            entry.tier,
                            entry.boost_weight,
                            entry.language,
                            entry.occurrence_count,
                            entry.last_seen_at.isoformat() if entry.last_seen_at else None,
                            entry.source,
                            entry.notes,
                            now.isoformat(),
                            now.isoformat(),
                        ),
                    )

                    # Insert aliases
                    for alias in entry.aliases:
                        conn.execute(
                            """INSERT INTO aliases
                               (entry_id, alias, is_common_misspelling) VALUES (?, ?, ?)""",
                            (entry_id, alias.alias, alias.is_common_misspelling),
                        )

                    # Insert pronunciations
                    for pron in entry.pronunciations:
                        conn.execute(
                            """INSERT INTO pronunciations
                               (entry_id, ipa, phoneme_sequence, language, variant)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                entry_id,
                                pron.ipa,
                                pron.phoneme_sequence,
                                pron.language,
                                pron.variant,
                            ),
                        )

                    # Insert contexts
                    for context in entry.contexts:
                        conn.execute(
                            "INSERT INTO entry_contexts (entry_id, context) VALUES (?, ?)",
                            (entry_id, context),
                        )

                    created += 1
                except sqlite3.IntegrityError:
                    # Skip duplicates (e.g., same canonical already exists)
                    continue

    return created


# =============================================================================
# Context Operations
# =============================================================================


def get_entries_by_context(
    context: str,
    tier: TierLevel | None = None,
    limit: int | None = None,
    language: str = "en",
) -> list[EntryWithRelations]:
    """Get entries associated with a specific context.

    Args:
        context: The context name to filter by
        tier: Optional tier filter (entries at this tier or higher)
        limit: Maximum number of entries to return
        language: Language filter

    Returns:
        List of entries matching the context
    """
    with db_connection() as conn:
        # Build query
        where_clauses = ["ec.context = ?", "e.language = ?"]
        params: list = [context, language]

        if tier:
            # Include entries at this tier or higher (A > B > C > ...)
            tier_order = ["A", "B", "C", "D", "E", "F", "G", "H"]
            tier_index = tier_order.index(tier)
            allowed_tiers = tier_order[: tier_index + 1]
            placeholders = ",".join("?" * len(allowed_tiers))
            where_clauses.append(f"e.tier IN ({placeholders})")
            params.extend(allowed_tiers)

        where_sql = " AND ".join(where_clauses)

        query = f"""
            SELECT DISTINCT e.id, e.canonical, e.display, e.type, e.tier, e.boost_weight,
                   e.language, e.occurrence_count, e.last_seen_at, e.source, e.notes,
                   e.created_at, e.updated_at
            FROM entries e
            JOIN entry_contexts ec ON e.id = ec.entry_id
            WHERE {where_sql}
            ORDER BY e.tier ASC, e.boost_weight DESC, e.occurrence_count DESC
        """

        if limit:
            # Validate limit is a positive integer to prevent SQL injection
            if not isinstance(limit, int) or limit < 1:
                raise ValueError(f"Invalid limit value: {limit}")
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)

        results = []
        for row in cursor.fetchall():
            entry = _row_to_entry(row)
            entry_with_relations = _entry_to_entry_with_relations(conn, entry)
            results.append(entry_with_relations)

        return results


def get_all_contexts() -> list[str]:
    """Get all unique context names in the database.

    Returns:
        List of context names sorted alphabetically
    """
    with db_connection() as conn:
        cursor = conn.execute(
            "SELECT DISTINCT context FROM entry_contexts ORDER BY context"
        )
        return [row["context"] for row in cursor.fetchall()]


# =============================================================================
# Statistics Operations
# =============================================================================


def increment_occurrence(entry_id: str) -> bool:
    """Increment the occurrence count for an entry.

    Args:
        entry_id: The UUID of the entry

    Returns:
        True if updated, False if entry not found
    """
    with _file_lock("dictionary-db"):
        with db_transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE entries
                SET occurrence_count = occurrence_count + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(), entry_id),
            )
            return cursor.rowcount > 0


def update_last_seen(entry_id: str, seen_at: datetime | None = None) -> bool:
    """Update the last_seen_at timestamp for an entry.

    Args:
        entry_id: The UUID of the entry
        seen_at: The timestamp (defaults to now)

    Returns:
        True if updated, False if entry not found
    """
    if seen_at is None:
        seen_at = datetime.now()

    with _file_lock("dictionary-db"):
        with db_transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE entries
                SET last_seen_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (seen_at.isoformat(), datetime.now().isoformat(), entry_id),
            )
            return cursor.rowcount > 0


def record_occurrence(entry_id: str) -> bool:
    """Record an occurrence of an entry (increment count and update last_seen).

    Convenience function that combines increment_occurrence and update_last_seen.

    Args:
        entry_id: The UUID of the entry

    Returns:
        True if updated, False if entry not found
    """
    now = datetime.now()

    with _file_lock("dictionary-db"):
        with db_transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE entries
                SET occurrence_count = occurrence_count + 1,
                    last_seen_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now.isoformat(), now.isoformat(), entry_id),
            )
            return cursor.rowcount > 0


def get_stats() -> DictionaryStats:
    """Get statistics about the dictionary.

    Returns:
        DictionaryStats with counts and breakdowns
    """
    with db_connection() as conn:
        # Total entries
        cursor = conn.execute("SELECT COUNT(*) as count FROM entries")
        total_entries = cursor.fetchone()["count"]

        # Entries by tier
        cursor = conn.execute(
            "SELECT tier, COUNT(*) as count FROM entries GROUP BY tier"
        )
        entries_by_tier = {row["tier"]: row["count"] for row in cursor.fetchall()}

        # Entries by type
        cursor = conn.execute(
            "SELECT type, COUNT(*) as count FROM entries GROUP BY type"
        )
        entries_by_type = {row["type"]: row["count"] for row in cursor.fetchall()}

        # Entries by context
        cursor = conn.execute(
            "SELECT context, COUNT(*) as count FROM entry_contexts GROUP BY context"
        )
        entries_by_context = {row["context"]: row["count"] for row in cursor.fetchall()}

        # Total aliases
        cursor = conn.execute("SELECT COUNT(*) as count FROM aliases")
        total_aliases = cursor.fetchone()["count"]

        # Total pronunciations
        cursor = conn.execute("SELECT COUNT(*) as count FROM pronunciations")
        total_pronunciations = cursor.fetchone()["count"]

        # Last updated
        cursor = conn.execute("SELECT MAX(updated_at) as last_updated FROM entries")
        row = cursor.fetchone()
        last_updated = datetime.fromisoformat(row["last_updated"]) if row["last_updated"] else None

        return DictionaryStats(
            total_entries=total_entries,
            entries_by_tier=entries_by_tier,
            entries_by_type=entries_by_type,
            entries_by_context=entries_by_context,
            total_aliases=total_aliases,
            total_pronunciations=total_pronunciations,
            last_updated=last_updated,
        )


# =============================================================================
# Utility Operations
# =============================================================================


def entry_exists(canonical: str, language: str = "en") -> bool:
    """Check if an entry with the given canonical form exists.

    Args:
        canonical: The canonical form to check
        language: Language code

    Returns:
        True if exists, False otherwise
    """
    with db_connection() as conn:
        cursor = conn.execute(
            "SELECT 1 FROM entries WHERE canonical = ? AND language = ? LIMIT 1",
            (canonical, language),
        )
        return cursor.fetchone() is not None


def get_entries_by_tier(
    tier: TierLevel,
    language: str = "en",
    limit: int | None = None,
) -> list[EntryWithRelations]:
    """Get all entries at a specific tier.

    Args:
        tier: The tier to filter by
        language: Language filter
        limit: Maximum number of entries

    Returns:
        List of entries at the specified tier
    """
    with db_connection() as conn:
        query = """
            SELECT id, canonical, display, type, tier, boost_weight,
                   language, occurrence_count, last_seen_at, source, notes,
                   created_at, updated_at
            FROM entries
            WHERE tier = ? AND language = ?
            ORDER BY boost_weight DESC, occurrence_count DESC
        """
        params: list = [tier, language]

        if limit:
            # Validate limit is a positive integer to prevent SQL injection
            if not isinstance(limit, int) or limit < 1:
                raise ValueError(f"Invalid limit value: {limit}")
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)

        results = []
        for row in cursor.fetchall():
            entry = _row_to_entry(row)
            entry_with_relations = _entry_to_entry_with_relations(conn, entry)
            results.append(entry_with_relations)

        return results


def get_all_entries(
    language: str = "en",
    limit: int | None = None,
) -> list[EntryWithRelations]:
    """Get all dictionary entries.

    Args:
        language: Language filter (default "en")
        limit: Maximum number of entries to return

    Returns:
        List of all entries with relations
    """
    with db_connection() as conn:
        query = """
            SELECT id, canonical, display, type, tier, boost_weight,
                   language, occurrence_count, last_seen_at, source, notes,
                   created_at, updated_at
            FROM entries
            WHERE language = ?
            ORDER BY tier ASC, boost_weight DESC, occurrence_count DESC
        """
        params: list = [language]

        if limit:
            # Validate limit is a positive integer to prevent SQL injection
            if not isinstance(limit, int) or limit < 1:
                raise ValueError(f"Invalid limit value: {limit}")
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)

        results = []
        for row in cursor.fetchall():
            entry = _row_to_entry(row)
            entry_with_relations = _entry_to_entry_with_relations(conn, entry)
            results.append(entry_with_relations)

        return results


def get_recent_entries(
    days: int = 30,
    limit: int = 50,
    language: str = "en",
) -> list[EntryWithRelations]:
    """Get recently seen entries.

    Args:
        days: Number of days to look back
        limit: Maximum number of entries
        language: Language filter

    Returns:
        List of entries seen within the specified period
    """
    cutoff = datetime.now().isoformat()

    with db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, canonical, display, type, tier, boost_weight,
                   language, occurrence_count, last_seen_at, source, notes,
                   created_at, updated_at
            FROM entries
            WHERE language = ?
              AND last_seen_at IS NOT NULL
              AND datetime(last_seen_at) >= datetime(?, '-' || ? || ' days')
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            (language, cutoff, days, limit),
        )

        results = []
        for row in cursor.fetchall():
            entry = _row_to_entry(row)
            entry_with_relations = _entry_to_entry_with_relations(conn, entry)
            results.append(entry_with_relations)

        return results

"""Schema management and migrations for ASR dictionary database.

This module handles database schema creation, migrations, and data import
from legacy vocabulary files.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from asr.config import VOCAB_DIR

logger = logging.getLogger(__name__)

# Schema version for tracking migrations
SCHEMA_VERSION = 1


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the database schema with all tables and indexes.

    This function is idempotent - safe to call multiple times.
    Uses IF NOT EXISTS for all CREATE statements.

    Args:
        conn: SQLite connection (caller manages transaction)
    """
    # Create entries table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY,
            canonical TEXT NOT NULL,
            display TEXT,
            type TEXT NOT NULL CHECK(type IN (
                'person', 'org', 'product', 'event', 'location', 'jargon', 'misc'
            )),
            tier TEXT NOT NULL DEFAULT 'D' CHECK(tier IN (
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'
            )),
            boost_weight REAL NOT NULL DEFAULT 1.0 CHECK(
                boost_weight >= 0.0 AND boost_weight <= 3.0
            ),
            language TEXT NOT NULL DEFAULT 'en',
            occurrence_count INTEGER NOT NULL DEFAULT 0,
            last_seen_at TEXT,
            source TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Create aliases table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            is_common_misspelling INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (entry_id) REFERENCES entries(id) ON DELETE CASCADE,
            UNIQUE(entry_id, alias)
        )
    """)

    # Create pronunciations table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pronunciations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT NOT NULL,
            ipa TEXT,
            phoneme_sequence TEXT,
            language TEXT NOT NULL DEFAULT 'en',
            variant TEXT,
            FOREIGN KEY (entry_id) REFERENCES entries(id) ON DELETE CASCADE
        )
    """)

    # Create entry_contexts table (many-to-many relationship)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entry_contexts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_id TEXT NOT NULL,
            context TEXT NOT NULL,
            FOREIGN KEY (entry_id) REFERENCES entries(id) ON DELETE CASCADE,
            UNIQUE(entry_id, context)
        )
    """)

    # Create schema_version table for migration tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
    """)

    # Create all indexes
    _create_indexes(conn)

    # Record schema version
    conn.execute("""
        INSERT OR IGNORE INTO schema_version (version, applied_at)
        VALUES (?, ?)
    """, (SCHEMA_VERSION, datetime.now().isoformat()))


def _create_indexes(conn: sqlite3.Connection) -> None:
    """Create all database indexes.

    Indexes are designed for common query patterns:
    - Looking up entries by canonical name
    - Filtering by tier, type, language
    - Searching aliases
    - Context-based queries
    - Recent entries queries
    """
    # Primary lookup indexes for entries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_canonical
        ON entries(canonical)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_canonical_lower
        ON entries(LOWER(canonical))
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_canonical_language
        ON entries(canonical, language)
    """)

    # Filter indexes
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_tier
        ON entries(tier)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_type
        ON entries(type)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_language
        ON entries(language)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_tier_language
        ON entries(tier, language)
    """)

    # Recency and usage indexes
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_last_seen
        ON entries(last_seen_at)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_occurrence_count
        ON entries(occurrence_count DESC)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_updated_at
        ON entries(updated_at)
    """)

    # Compound index for common query pattern (tier + boost + occurrence)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entries_ranking
        ON entries(tier ASC, boost_weight DESC, occurrence_count DESC)
    """)

    # Alias indexes
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_aliases_entry_id
        ON aliases(entry_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_aliases_alias
        ON aliases(alias)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_aliases_alias_lower
        ON aliases(LOWER(alias))
    """)

    # Pronunciation indexes
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pronunciations_entry_id
        ON pronunciations(entry_id)
    """)

    # Context indexes
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entry_contexts_entry_id
        ON entry_contexts(entry_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entry_contexts_context
        ON entry_contexts(context)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_entry_contexts_context_entry
        ON entry_contexts(context, entry_id)
    """)


def get_schema_version(conn: sqlite3.Connection) -> int | None:
    """Get the current schema version.

    Args:
        conn: SQLite connection

    Returns:
        Current schema version or None if not initialized
    """
    try:
        cursor = conn.execute(
            "SELECT MAX(version) as version FROM schema_version"
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else None
    except sqlite3.OperationalError:
        # Table doesn't exist
        return None


def needs_migration(conn: sqlite3.Connection) -> bool:
    """Check if the database needs migration.

    Args:
        conn: SQLite connection

    Returns:
        True if migration is needed
    """
    current_version = get_schema_version(conn)
    return current_version is None or current_version < SCHEMA_VERSION


# =============================================================================
# Legacy Migration
# =============================================================================


def migrate_from_legacy() -> dict[str, int]:
    """Import entries from legacy vocabulary files.

    Reads vocabulary files from ~/.asr/vocabularies/*.txt and imports
    them as dictionary entries.

    Legacy format:
    - One term per line
    - Lines starting with # are comments
    - File name (without .txt) becomes the context

    Returns:
        Dict mapping domain/context names to number of entries imported
    """
    from asr.dictionary.db import bulk_create_entries, entry_exists
    from asr.dictionary.models import EntryWithRelations

    if not VOCAB_DIR.exists():
        return {}

    results: dict[str, int] = {}

    for vocab_file in VOCAB_DIR.glob("*.txt"):
        domain = vocab_file.stem
        terms = _parse_legacy_vocab_file(vocab_file)

        if not terms:
            continue

        # Convert terms to entries
        entries: list[EntryWithRelations] = []
        for term in terms:
            # Skip if already exists
            if entry_exists(term):
                continue

            # Determine entry type based on simple heuristics
            entry_type = _infer_entry_type(term, domain)

            # Create entry
            entry = EntryWithRelations(
                id=str(uuid4()),
                canonical=term,
                display=None,
                type=entry_type,
                tier="D",  # Default tier for imported entries
                boost_weight=1.0,
                language="en",
                occurrence_count=0,
                last_seen_at=None,
                source=f"legacy:{domain}",
                notes=f"Imported from legacy vocabulary file: {vocab_file.name}",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                aliases=[],
                pronunciations=[],
                contexts=[domain],
            )
            entries.append(entry)

        # Bulk create
        if entries:
            created = bulk_create_entries(entries)
            results[domain] = created

    return results


def _parse_legacy_vocab_file(path: Path) -> list[str]:
    """Parse a legacy vocabulary file.

    Args:
        path: Path to the vocabulary file

    Returns:
        List of terms from the file
    """
    if not path.exists():
        return []

    terms = []
    try:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            terms.append(line)
    except Exception as e:
        # Silently skip files that can't be read
        logger.debug(f"Failed to read legacy vocabulary file {path}: {e}")

    return terms


def _infer_entry_type(term: str, domain: str) -> str:
    """Infer entry type from term and domain.

    Uses simple heuristics to guess the type:
    - Names with common name patterns -> person
    - Domain hints (tech, medical, etc.)
    - Default to misc

    Args:
        term: The vocabulary term
        domain: The domain/context name

    Returns:
        Entry type string
    """
    domain_lower = domain.lower()

    # Domain-based inference
    if domain_lower in ("biography", "names", "people", "speakers"):
        return "person"
    if domain_lower in ("tech", "technical", "programming", "software"):
        return "jargon"
    if domain_lower in ("companies", "organizations", "orgs"):
        return "org"
    if domain_lower in ("products", "brands"):
        return "product"
    if domain_lower in ("places", "locations", "geography"):
        return "location"
    if domain_lower in ("events", "conferences"):
        return "event"

    # Term-based inference (simple heuristics)
    # Check if it looks like a person's name (2-3 capitalized words)
    words = term.split()
    if len(words) >= 2 and len(words) <= 4:
        if all(w[0].isupper() for w in words if w):
            # Likely a person's name
            return "person"

    # Check for common company suffixes
    company_suffixes = ("Inc", "LLC", "Corp", "Ltd", "Co", "Company", "Technologies", "Labs")
    if any(term.endswith(suffix) for suffix in company_suffixes):
        return "org"

    # Default
    return "misc"


def import_from_json(json_path: Path) -> int:
    """Import entries from a JSON file.

    Expected format:
    [
        {
            "canonical": "Term",
            "type": "person",
            "tier": "B",
            "aliases": ["Alt1", "Alt2"],
            "contexts": ["domain1"]
        },
        ...
    ]

    Args:
        json_path: Path to the JSON file

    Returns:
        Number of entries imported
    """
    import json

    from asr.dictionary.db import bulk_create_entries, entry_exists
    from asr.dictionary.models import Alias, EntryWithRelations

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of entries")

    entries: list[EntryWithRelations] = []
    now = datetime.now()

    for item in data:
        canonical = item.get("canonical", "").strip()
        if not canonical:
            continue

        # Skip if already exists
        if entry_exists(canonical):
            continue

        # Parse entry
        entry_type = item.get("type", "misc")
        if entry_type not in ("person", "org", "product", "event", "location", "jargon", "misc"):
            entry_type = "misc"

        tier = item.get("tier", "D")
        if tier not in ("A", "B", "C", "D", "E", "F", "G", "H"):
            tier = "D"

        # Build aliases
        alias_strings = item.get("aliases", [])
        aliases = [
            Alias(entry_id="", alias=a, is_common_misspelling=False)
            for a in alias_strings if a and a != canonical
        ]

        # Build entry
        entry = EntryWithRelations(
            id=str(uuid4()),
            canonical=canonical,
            display=item.get("display"),
            type=entry_type,
            tier=tier,
            boost_weight=float(item.get("boost_weight", 1.0)),
            language=item.get("language", "en"),
            occurrence_count=0,
            last_seen_at=None,
            source=f"json:{json_path.name}",
            notes=item.get("notes"),
            created_at=now,
            updated_at=now,
            aliases=aliases,
            pronunciations=[],
            contexts=item.get("contexts", []),
        )
        entries.append(entry)

    return bulk_create_entries(entries)


def export_to_json(output_path: Path, context: str | None = None) -> int:
    """Export dictionary entries to a JSON file.

    Args:
        output_path: Path for the output JSON file
        context: Optional context to filter by

    Returns:
        Number of entries exported
    """
    import json

    from asr.dictionary.db import db_connection, get_entries_by_context

    entries: list[dict] = []

    if context:
        # Export entries for specific context
        entry_list = get_entries_by_context(context)
    else:
        # Export all entries
        from asr.dictionary.db import _row_to_entry, _entry_to_entry_with_relations

        with db_connection() as conn:
            cursor = conn.execute("""
                SELECT id, canonical, display, type, tier, boost_weight,
                       language, occurrence_count, last_seen_at, source, notes,
                       created_at, updated_at
                FROM entries
                ORDER BY tier ASC, canonical ASC
            """)

            entry_list = []
            for row in cursor.fetchall():
                entry = _row_to_entry(row)
                entry_with_relations = _entry_to_entry_with_relations(conn, entry)
                entry_list.append(entry_with_relations)

    # Convert to export format
    for entry in entry_list:
        export_item = {
            "canonical": entry.canonical,
            "type": entry.type,
            "tier": entry.tier,
            "boost_weight": entry.boost_weight,
            "language": entry.language,
        }

        if entry.display:
            export_item["display"] = entry.display
        if entry.aliases:
            export_item["aliases"] = [a.alias for a in entry.aliases]
        if entry.contexts:
            export_item["contexts"] = entry.contexts
        if entry.notes:
            export_item["notes"] = entry.notes

        entries.append(export_item)

    # Write JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    return len(entries)


# =============================================================================
# Database Maintenance
# =============================================================================


def vacuum_database() -> None:
    """Vacuum the database to reclaim space and optimize.

    Should be run periodically after many deletions.
    """
    from asr.dictionary.db import db_connection

    with db_connection() as conn:
        # VACUUM cannot run inside a transaction
        conn.isolation_level = None
        conn.execute("VACUUM")


def analyze_database() -> None:
    """Run ANALYZE to update query planner statistics.

    Should be run after significant data changes.
    """
    from asr.dictionary.db import db_connection

    with db_connection() as conn:
        # ANALYZE cannot run inside a transaction
        conn.isolation_level = None
        conn.execute("ANALYZE")


def rebuild_indexes() -> None:
    """Drop and recreate all indexes.

    Useful if indexes become fragmented or corrupted.
    """
    from asr.config import _file_lock
    from asr.dictionary.db import db_transaction

    with _file_lock("dictionary-db"):
        with db_transaction() as conn:
            # Get list of all indexes (except sqlite internal ones)
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type = 'index'
                  AND name NOT LIKE 'sqlite_%'
                  AND name NOT LIKE 'idx_sqlite_%'
            """)
            indexes = [row[0] for row in cursor.fetchall()]

            # Drop all indexes
            for idx_name in indexes:
                conn.execute(f"DROP INDEX IF EXISTS {idx_name}")

            # Recreate indexes
            _create_indexes(conn)


def check_integrity() -> tuple[bool, list[str]]:
    """Check database integrity.

    Returns:
        Tuple of (is_ok, list of error messages)
    """
    from asr.dictionary.db import db_connection

    errors: list[str] = []

    with db_connection() as conn:
        # Run integrity check
        cursor = conn.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        if result[0] != "ok":
            errors.append(f"Integrity check failed: {result[0]}")

        # Check foreign keys
        cursor = conn.execute("PRAGMA foreign_key_check")
        fk_errors = cursor.fetchall()
        for fk_error in fk_errors:
            errors.append(f"Foreign key violation: {fk_error}")

    return len(errors) == 0, errors

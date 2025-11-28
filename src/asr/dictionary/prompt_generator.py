"""Generate prompts for Whisper and Claude from dictionary entries.

This module provides functions to format dictionary entries for:
1. Whisper initial_prompt - Bias the ASR model toward known terms
2. Claude correction block - Guide LLM correction with known proper nouns
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asr.dictionary.models import EntryWithRelations


def estimate_tokens(text: str) -> int:
    """Rough token estimate for English text.

    Uses the common heuristic of ~4 characters per token for English.
    This is approximate but sufficient for prompt budget estimation.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    # Average ~4 chars per token for English text
    # Add small buffer for safety
    return max(1, len(text) // 4)


def generate_whisper_prompt(
    entries: list[EntryWithRelations],
    max_tokens: int = 200,
) -> str:
    """Generate Whisper initial_prompt from dictionary entries.

    Creates a formatted prompt that biases Whisper toward recognizing
    specific proper nouns and terms. Groups entries by type for readability
    and prioritizes by boost_weight.

    Format:
        "In this conversation, you may hear: [People: X, Y] [Products: A, B] [Terms: C, D]..."

    Args:
        entries: List of dictionary entries to include in the prompt.
        max_tokens: Maximum token budget for the prompt (Whisper limit ~224).
            Default is 200 to leave headroom.

    Returns:
        Formatted prompt string suitable for Whisper's initial_prompt parameter.
    """
    if not entries:
        return ""

    # Sort by boost_weight (descending) to prioritize important entries
    sorted_entries = sorted(entries, key=lambda e: e.boost_weight, reverse=True)

    # Group entries by type
    type_groups: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)

    for entry in sorted_entries:
        # Get canonical name and up to 2 key aliases
        canonical = entry.canonical
        key_aliases = [a.alias for a in entry.aliases[:2] if not a.is_common_misspelling]

        # Map entry types to display labels
        # EntryWithRelations has type directly, not via entry.type
        type_label = _get_type_label(entry.type)
        type_groups[type_label].append((canonical, key_aliases))

    # Build the prompt, respecting token budget
    parts: list[str] = []
    current_tokens = estimate_tokens("In this conversation, you may hear: ")

    # Priority order for type groups
    type_priority = [
        "People",
        "Places",
        "Organizations",
        "Products",
        "Events",
        "Terms",
        "Technical",
        "Medical",
        "Legal",
        "Other",
    ]

    for type_label in type_priority:
        if type_label not in type_groups:
            continue

        group_items = type_groups[type_label]
        if not group_items:
            continue

        # Build group string
        terms: list[str] = []
        for canonical, aliases in group_items:
            # Include canonical
            terms.append(canonical)
            # Include key aliases if space permits
            for alias in aliases:
                if alias.lower() != canonical.lower():
                    terms.append(alias)

        group_str = f"[{type_label}: {', '.join(terms)}]"
        group_tokens = estimate_tokens(group_str) + 1  # +1 for space

        # Check if we have budget
        if current_tokens + group_tokens <= max_tokens:
            parts.append(group_str)
            current_tokens += group_tokens
        else:
            # Try to fit a truncated version
            truncated_terms: list[str] = []
            truncated_tokens = estimate_tokens(f"[{type_label}: ]") + 1

            for term in terms:
                term_tokens = estimate_tokens(term + ", ")
                if current_tokens + truncated_tokens + term_tokens <= max_tokens:
                    truncated_terms.append(term)
                    truncated_tokens += term_tokens
                else:
                    break

            if truncated_terms:
                truncated_str = f"[{type_label}: {', '.join(truncated_terms)}]"
                parts.append(truncated_str)
                current_tokens += estimate_tokens(truncated_str) + 1

            # Stop adding more groups if we're near the limit
            if current_tokens >= max_tokens * 0.9:
                break

    if not parts:
        return ""

    return f"In this conversation, you may hear: {' '.join(parts)}"


def _get_type_label(entry_type: str) -> str:
    """Map entry type to display label for prompt grouping.

    Args:
        entry_type: The entry type value.

    Returns:
        Human-readable label for the type group.
    """
    type_map = {
        "person": "People",
        "place": "Places",
        "location": "Places",  # Alias for place
        "org": "Organizations",
        "organization": "Organizations",
        "product": "Products",
        "technical": "Technical",
        "event": "Events",
        "jargon": "Terms",
        "medical": "Medical",
        "legal": "Legal",
        "misc": "Other",
        "other": "Other",
    }
    return type_map.get(entry_type.lower(), "Other")


def generate_correction_block(
    entries: list[EntryWithRelations],
    max_entries: int = 100,
) -> str:
    """Generate dictionary block for Claude correction prompt.

    Creates a markdown table of known proper nouns with their canonical
    spellings, aliases, and notes to guide the LLM correction process.

    Format:
        ## Known Proper Nouns (use these exact spellings when correcting)

        | Term | Type | Aliases | Notes |
        |------|------|---------|-------|
        | Ron Chernow | person | | Historian |
        | Navesink Challenge | event | Navesink, Navesink 12K | Running race |

        When you see text that sounds like one of these terms, use the canonical spelling.
        Only substitute if confident - do not force-fit.

    Args:
        entries: List of dictionary entries to include.
        max_entries: Maximum number of entries to include (default 100 for cost control).

    Returns:
        Markdown-formatted dictionary block for inclusion in correction prompts.
    """
    if not entries:
        return ""

    # Sort by boost_weight (descending) and take top entries
    sorted_entries = sorted(entries, key=lambda e: e.boost_weight, reverse=True)
    selected_entries = sorted_entries[:max_entries]

    # Build markdown table
    lines: list[str] = [
        "## Known Proper Nouns (use these exact spellings when correcting)",
        "",
        "| Term | Type | Aliases | Notes |",
        "|------|------|---------|-------|",
    ]

    for entry in selected_entries:
        canonical = _escape_markdown_cell(entry.canonical)
        entry_type = entry.type  # Direct attribute on EntryWithRelations

        # Get non-misspelling aliases (up to 3)
        aliases = [
            _escape_markdown_cell(a.alias)
            for a in entry.aliases[:3]
            if not a.is_common_misspelling
        ]
        aliases_str = ", ".join(aliases) if aliases else ""

        # Build notes from display name or source
        notes_parts: list[str] = []
        if entry.display and entry.display != entry.canonical:
            notes_parts.append(entry.display)
        if entry.source:
            notes_parts.append(f"Source: {entry.source}")
        notes_str = _escape_markdown_cell("; ".join(notes_parts)) if notes_parts else ""

        lines.append(f"| {canonical} | {entry_type} | {aliases_str} | {notes_str} |")

    # Add guidance footer
    lines.extend([
        "",
        "When you see text that sounds like one of these terms, use the canonical spelling.",
        "Only substitute if confident - do not force-fit.",
    ])

    return "\n".join(lines)


def _escape_markdown_cell(text: str) -> str:
    """Escape special characters for markdown table cells.

    Args:
        text: Text to escape.

    Returns:
        Escaped text safe for markdown table cells.
    """
    if not text:
        return ""
    # Escape pipe characters and newlines
    return text.replace("|", "\\|").replace("\n", " ").strip()


def generate_combined_prompt(
    entries: list[EntryWithRelations],
    whisper_max_tokens: int = 200,
    correction_max_entries: int = 100,
) -> tuple[str, str]:
    """Generate both Whisper prompt and correction block.

    Convenience function to generate both prompt types from the same
    entry list with appropriate limits for each.

    Args:
        entries: List of dictionary entries to use.
        whisper_max_tokens: Token budget for Whisper prompt.
        correction_max_entries: Maximum entries for correction block.

    Returns:
        Tuple of (whisper_prompt, correction_block).
    """
    whisper_prompt = generate_whisper_prompt(entries, max_tokens=whisper_max_tokens)
    correction_block = generate_correction_block(entries, max_entries=correction_max_entries)
    return whisper_prompt, correction_block


def format_entries_for_context(
    entries: list[EntryWithRelations],
    context: str | None = None,
) -> list[EntryWithRelations]:
    """Filter and sort entries for a specific context.

    Filters entries to those matching the context (if provided) and
    sorts by relevance (boost_weight).

    Args:
        entries: List of all available entries.
        context: Optional context to filter by. If None, returns all entries.

    Returns:
        Filtered and sorted list of entries.
    """
    if not entries:
        return []

    if context is None:
        # Return all entries sorted by boost weight
        return sorted(entries, key=lambda e: e.boost_weight, reverse=True)

    # Filter to entries matching context and calculate effective weight
    weighted_entries: list[tuple[float, EntryWithRelations]] = []
    for entry in entries:
        if entry.has_context(context):
            # Context match - full weight
            weighted_entries.append((entry.boost_weight, entry))
        elif not entry.contexts:
            # Include entries without any context tags (universal entries)
            # but with slightly reduced weight
            weighted_entries.append((entry.boost_weight * 0.8, entry))

    # Sort by effective weight descending
    weighted_entries.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in weighted_entries]

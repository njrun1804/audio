"""Named Entity Recognition for proper noun discovery.

This module provides utilities for extracting proper nouns from text using
GLiNER (zero-shot NER) and optionally spaCy for production workloads.

Requires: pip install asr[ner]
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
# Supported entity types for proper noun extraction
ENTITY_TYPES = ["person", "organization", "product", "location", "event"]


@dataclass
class ExtractedEntity:
    """A proper noun entity extracted from text."""

    text: str
    entity_type: str
    confidence: float
    start: int | None = None
    end: int | None = None


@lru_cache(maxsize=1)
def _get_gliner_model(model_name: str = "urchade/gliner_small-v2.1"):
    """Load and cache the GLiNER model.

    Uses small model by default for faster inference.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded GLiNER model instance.

    Raises:
        ImportError: If gliner is not installed.
    """
    try:
        from gliner import GLiNER
    except ImportError as e:
        raise ImportError(
            "GLiNER is required for NER. Install with: pip install asr[ner]"
        ) from e

    return GLiNER.from_pretrained(model_name)


def extract_proper_nouns(
    text: str,
    labels: list[str] | None = None,
    model_name: str = "urchade/gliner_small-v2.1",
    min_confidence: float = 0.5,
) -> list[ExtractedEntity]:
    """Extract proper nouns from text using GLiNER zero-shot NER.

    GLiNER is a lightweight model that can recognize arbitrary entity types
    without fine-tuning, making it ideal for discovering proper nouns in
    transcription text.

    Args:
        text: The text to extract entities from.
        labels: Entity types to extract. Defaults to ENTITY_TYPES.
        model_name: GLiNER model to use. Defaults to small model for speed.
        min_confidence: Minimum confidence threshold (0-1). Defaults to 0.5.

    Returns:
        List of ExtractedEntity objects with text, type, confidence, and positions.

    Raises:
        ImportError: If gliner is not installed.

    Example:
        >>> entities = extract_proper_nouns("Ron Chernow wrote about Alexander Hamilton")
        >>> for e in entities:
        ...     print(f"{e.text}: {e.entity_type} ({e.confidence:.2f})")
        Ron Chernow: person (0.95)
        Alexander Hamilton: person (0.97)
    """
    if not text or not text.strip():
        return []

    if labels is None:
        labels = ENTITY_TYPES

    model = _get_gliner_model(model_name)

    try:
        entities = model.predict_entities(text, labels=labels)
    except Exception:
        # Return empty list on model errors rather than crashing
        return []

    results = []
    for entity in entities:
        score = entity.get("score", 0.0)
        if score >= min_confidence:
            results.append(
                ExtractedEntity(
                    text=entity["text"],
                    entity_type=entity["label"],
                    confidence=score,
                    start=entity.get("start"),
                    end=entity.get("end"),
                )
            )

    # Sort by confidence descending
    results.sort(key=lambda e: e.confidence, reverse=True)
    return results


def extract_proper_nouns_batch(
    texts: list[str],
    labels: list[str] | None = None,
    model_name: str = "urchade/gliner_small-v2.1",
    min_confidence: float = 0.5,
) -> list[list[ExtractedEntity]]:
    """Extract proper nouns from multiple texts efficiently.

    Processes texts in a batch for better throughput when handling
    multiple transcript segments.

    Args:
        texts: List of text strings to process.
        labels: Entity types to extract. Defaults to ENTITY_TYPES.
        model_name: GLiNER model to use.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of entity lists, one per input text.
    """
    if not texts:
        return []

    # GLiNER doesn't have native batch support, so process sequentially
    # but benefit from cached model loading
    return [
        extract_proper_nouns(text, labels, model_name, min_confidence)
        for text in texts
    ]


def filter_entities_by_type(
    entities: list[ExtractedEntity],
    entity_types: list[str],
) -> list[ExtractedEntity]:
    """Filter entities to only include specified types.

    Args:
        entities: List of extracted entities.
        entity_types: Types to keep (e.g., ["person", "organization"]).

    Returns:
        Filtered list of entities.
    """
    type_set = set(entity_types)
    return [e for e in entities if e.entity_type in type_set]


def deduplicate_entities(
    entities: list[ExtractedEntity],
    case_sensitive: bool = False,
) -> list[ExtractedEntity]:
    """Remove duplicate entities, keeping highest confidence.

    Args:
        entities: List of extracted entities.
        case_sensitive: Whether to treat case differences as distinct.

    Returns:
        Deduplicated list of entities.
    """
    seen: dict[str, ExtractedEntity] = {}

    for entity in entities:
        key = entity.text if case_sensitive else entity.text.lower()
        if key not in seen or entity.confidence > seen[key].confidence:
            seen[key] = entity

    return list(seen.values())

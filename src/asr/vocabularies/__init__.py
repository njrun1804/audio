"""Built-in vocabulary lists for ASR enhancement.

Provides lightweight proper noun dictionaries for common names, places,
companies, and products to improve transcription accuracy.

These are bundled with the package (< 100KB total) and can be used
as vocabulary hints for CrisperWhisper's initial_prompt or Claude correction.
"""

from asr.vocabularies.common_names import (
    get_common_first_names,
    get_common_last_names,
    get_historian_names,
)
from asr.vocabularies.tech_terms import (
    get_tech_companies,
    get_tech_products,
    get_programming_terms,
)

__all__ = [
    "get_common_first_names",
    "get_common_last_names",
    "get_historian_names",
    "get_tech_companies",
    "get_tech_products",
    "get_programming_terms",
    "get_domain_vocabulary",
]


def get_domain_vocabulary(domain: str) -> list[str]:
    """Get vocabulary terms for a specific domain.

    Args:
        domain: One of: biography, technical, medical, legal, conversational

    Returns:
        List of proper nouns and domain-specific terms
    """
    if domain == "biography":
        return (
            get_common_first_names()[:200] +  # Top 200 names
            get_common_last_names()[:200] +
            get_historian_names()
        )
    elif domain == "technical":
        return (
            get_tech_companies() +
            get_tech_products() +
            get_programming_terms()
        )
    elif domain == "medical":
        # Medical vocabulary would need careful curation
        # For now, return common medical terms
        return []
    elif domain == "legal":
        # Legal vocabulary would need careful curation
        return []
    else:
        # General/conversational - just common names
        return get_common_first_names()[:100] + get_common_last_names()[:100]

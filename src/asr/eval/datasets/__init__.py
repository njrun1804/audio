"""ASR Evaluation Datasets.

Provides access to public domain test datasets for ASR evaluation.
"""

from asr.eval.datasets.lovecraft import (
    LOVECRAFT_STORIES,
    download_lovecraft_dataset,
    get_lovecraft_reference,
)

__all__ = [
    "LOVECRAFT_STORIES",
    "download_lovecraft_dataset",
    "get_lovecraft_reference",
]

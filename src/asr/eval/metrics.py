"""WER and other ASR evaluation metrics.

Implements standard ASR evaluation metrics following the conventions
used by academic benchmarks (LibriSpeech, Common Voice, etc.)
"""

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class WERResult:
    """Word Error Rate calculation result."""

    wer: float  # Word Error Rate (0.0 = perfect, 1.0 = 100% errors)
    substitutions: int
    insertions: int
    deletions: int
    reference_words: int
    hypothesis_words: int

    @property
    def correct(self) -> int:
        """Number of correctly matched words."""
        return self.reference_words - self.substitutions - self.deletions

    @property
    def accuracy(self) -> float:
        """Word accuracy (1.0 - WER, clamped to [0, 1])."""
        return max(0.0, 1.0 - self.wer)


def normalize_text(text: str, keep_case: bool = False) -> str:
    """Normalize text for WER calculation.

    Follows standard ASR normalization:
    1. Unicode normalize (NFKC)
    2. Lowercase (unless keep_case=True)
    3. Remove punctuation
    4. Collapse whitespace
    5. Strip leading/trailing whitespace

    Args:
        text: Input text to normalize
        keep_case: If True, preserve case (default: lowercase)

    Returns:
        Normalized text suitable for WER comparison
    """
    if not text:
        return ""

    # Unicode normalization (handles accents, ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    if not keep_case:
        text = text.lower()

    # Remove punctuation (keep apostrophes in contractions)
    # Replace punctuation with space to avoid word merging
    text = re.sub(r"[^\w\s']", " ", text)

    # Remove standalone apostrophes (keep contractions like "don't")
    text = re.sub(r"\s'\s", " ", text)
    text = re.sub(r"^'\s", " ", text)
    text = re.sub(r"\s'$", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip
    text = text.strip()

    return text


def _levenshtein_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Calculate Levenshtein distance with edit operation counts.

    Uses dynamic programming to find minimum edit distance and
    track substitutions, insertions, and deletions.

    Args:
        ref: Reference word list
        hyp: Hypothesis word list

    Returns:
        Tuple of (distance, substitutions, insertions, deletions)
    """
    n = len(ref)
    m = len(hyp)

    # DP table: dp[i][j] = (distance, subs, ins, dels)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    # Initialize base cases
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)  # All deletions
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)  # All insertions

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                # Match - no edit needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Consider three operations
                sub_cost = dp[i - 1][j - 1]
                ins_cost = dp[i][j - 1]
                del_cost = dp[i - 1][j]

                # Choose minimum
                if sub_cost[0] <= ins_cost[0] and sub_cost[0] <= del_cost[0]:
                    # Substitution
                    dp[i][j] = (sub_cost[0] + 1, sub_cost[1] + 1, sub_cost[2], sub_cost[3])
                elif ins_cost[0] <= del_cost[0]:
                    # Insertion
                    dp[i][j] = (ins_cost[0] + 1, ins_cost[1], ins_cost[2] + 1, ins_cost[3])
                else:
                    # Deletion
                    dp[i][j] = (del_cost[0] + 1, del_cost[1], del_cost[2], del_cost[3] + 1)

    return dp[n][m]


def calculate_wer(reference: str, hypothesis: str, normalize: bool = True) -> WERResult:
    """Calculate Word Error Rate between reference and hypothesis.

    WER = (S + I + D) / N

    Where:
    - S = Substitutions (wrong word)
    - I = Insertions (extra word in hypothesis)
    - D = Deletions (missing word in hypothesis)
    - N = Number of words in reference

    Args:
        reference: Ground truth text
        hypothesis: ASR transcription output
        normalize: Whether to normalize texts before comparison

    Returns:
        WERResult with WER and edit operation counts
    """
    # Normalize if requested
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    # Tokenize into words
    ref_words = reference.split() if reference else []
    hyp_words = hypothesis.split() if hypothesis else []

    # Handle edge cases
    if len(ref_words) == 0:
        if len(hyp_words) == 0:
            return WERResult(
                wer=0.0,
                substitutions=0,
                insertions=0,
                deletions=0,
                reference_words=0,
                hypothesis_words=0,
            )
        else:
            return WERResult(
                wer=float('inf'),  # Can't calculate WER with empty reference
                substitutions=0,
                insertions=len(hyp_words),
                deletions=0,
                reference_words=0,
                hypothesis_words=len(hyp_words),
            )

    # Calculate Levenshtein distance
    distance, subs, ins, dels = _levenshtein_distance(ref_words, hyp_words)

    # WER = (S + I + D) / N
    wer = distance / len(ref_words)

    return WERResult(
        wer=wer,
        substitutions=subs,
        insertions=ins,
        deletions=dels,
        reference_words=len(ref_words),
        hypothesis_words=len(hyp_words),
    )


def calculate_cer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """Calculate Character Error Rate.

    Similar to WER but operates on characters instead of words.
    Useful for languages without clear word boundaries.

    Args:
        reference: Ground truth text
        hypothesis: ASR transcription output
        normalize: Whether to normalize texts before comparison

    Returns:
        CER as a float (0.0 = perfect)
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)

    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float('inf')

    distance, _, _, _ = _levenshtein_distance(ref_chars, hyp_chars)
    return distance / len(ref_chars)

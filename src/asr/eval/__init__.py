"""ASR Evaluation Module.

Provides tools for evaluating transcription accuracy using
Word Error Rate (WER) and other metrics against reference texts.
"""

from asr.eval.metrics import calculate_wer, normalize_text
from asr.eval.benchmark import run_benchmark, BenchmarkResult

__all__ = [
    "calculate_wer",
    "normalize_text",
    "run_benchmark",
    "BenchmarkResult",
]

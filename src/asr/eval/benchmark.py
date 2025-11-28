"""ASR Benchmark Runner.

Runs transcription benchmarks on test datasets and reports
WER, RTF (Real-Time Factor), and other metrics.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from asr.eval.metrics import WERResult, calculate_wer


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    audio_path: str
    duration_seconds: float

    # Timing
    transcription_time: float
    correction_time: float = 0.0

    # Quality metrics
    wer_raw: WERResult | None = None  # WER before correction
    wer_corrected: WERResult | None = None  # WER after correction

    # Transcription output
    raw_text: str = ""
    corrected_text: str = ""

    @property
    def rtf(self) -> float:
        """Real-Time Factor (< 1.0 means faster than real-time)."""
        if self.duration_seconds == 0:
            return 0.0
        return self.transcription_time / self.duration_seconds

    @property
    def total_time(self) -> float:
        """Total processing time (transcription + correction)."""
        return self.transcription_time + self.correction_time

    @property
    def wer_improvement(self) -> float | None:
        """WER improvement from correction (positive = better)."""
        if self.wer_raw is None or self.wer_corrected is None:
            return None
        return self.wer_raw.wer - self.wer_corrected.wer

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "audio_path": self.audio_path,
            "duration_seconds": self.duration_seconds,
            "transcription_time": self.transcription_time,
            "correction_time": self.correction_time,
            "rtf": self.rtf,
            "wer_raw": self.wer_raw.wer if self.wer_raw else None,
            "wer_corrected": self.wer_corrected.wer if self.wer_corrected else None,
            "wer_improvement": self.wer_improvement,
            "accuracy_raw": self.wer_raw.accuracy if self.wer_raw else None,
            "accuracy_corrected": self.wer_corrected.accuracy if self.wer_corrected else None,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    name: str
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def avg_wer_raw(self) -> float | None:
        """Average WER before correction."""
        wers = [r.wer_raw.wer for r in self.results if r.wer_raw]
        return sum(wers) / len(wers) if wers else None

    @property
    def avg_wer_corrected(self) -> float | None:
        """Average WER after correction."""
        wers = [r.wer_corrected.wer for r in self.results if r.wer_corrected]
        return sum(wers) / len(wers) if wers else None

    @property
    def avg_rtf(self) -> float:
        """Average Real-Time Factor."""
        rtfs = [r.rtf for r in self.results if r.rtf > 0]
        return sum(rtfs) / len(rtfs) if rtfs else 0.0

    @property
    def total_duration(self) -> float:
        """Total audio duration in seconds."""
        return sum(r.duration_seconds for r in self.results)

    @property
    def total_time(self) -> float:
        """Total processing time in seconds."""
        return sum(r.total_time for r in self.results)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "total_files": len(self.results),
            "total_duration_seconds": self.total_duration,
            "total_processing_time": self.total_time,
            "avg_rtf": self.avg_rtf,
            "avg_wer_raw": self.avg_wer_raw,
            "avg_wer_corrected": self.avg_wer_corrected,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, path: Path) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Benchmark: {self.name}",
            f"Files: {len(self.results)}",
            f"Total audio: {self.total_duration / 60:.1f} min",
            f"Processing time: {self.total_time / 60:.1f} min",
            f"Average RTF: {self.avg_rtf:.2f}x",
            "",
        ]

        if self.avg_wer_raw is not None:
            lines.append(f"WER (raw): {self.avg_wer_raw * 100:.1f}%")

        if self.avg_wer_corrected is not None:
            lines.append(f"WER (corrected): {self.avg_wer_corrected * 100:.1f}%")

            if self.avg_wer_raw is not None:
                improvement = (self.avg_wer_raw - self.avg_wer_corrected) * 100
                lines.append(f"WER improvement: {improvement:+.1f}%")

        return "\n".join(lines)


def run_benchmark(
    audio_path: Path,
    reference_text: str,
    transcribe_fn: Callable[[Path], tuple[str, float]],
    correct_fn: Callable[[str], tuple[str, float]] | None = None,
    name: str | None = None,
) -> BenchmarkResult:
    """Run a single benchmark.

    Args:
        audio_path: Path to audio file
        reference_text: Ground truth text for WER calculation
        transcribe_fn: Function that takes audio path, returns (text, duration)
        correct_fn: Optional function that takes text, returns (corrected_text, time)
        name: Optional name for this benchmark

    Returns:
        BenchmarkResult with metrics
    """
    name = name or audio_path.stem

    # Get audio duration
    from asr.audio.ingest import get_audio_duration
    duration = get_audio_duration(audio_path)

    # Transcribe
    start = time.time()
    raw_text, _ = transcribe_fn(audio_path)
    transcription_time = time.time() - start

    # Calculate raw WER
    wer_raw = calculate_wer(reference_text, raw_text)

    # Correct if function provided
    corrected_text = raw_text
    correction_time = 0.0
    wer_corrected = None

    if correct_fn:
        start = time.time()
        corrected_text, _ = correct_fn(raw_text)
        correction_time = time.time() - start
        wer_corrected = calculate_wer(reference_text, corrected_text)

    return BenchmarkResult(
        name=name,
        audio_path=str(audio_path),
        duration_seconds=duration,
        transcription_time=transcription_time,
        correction_time=correction_time,
        wer_raw=wer_raw,
        wer_corrected=wer_corrected,
        raw_text=raw_text,
        corrected_text=corrected_text,
    )

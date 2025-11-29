"""Structured logging for ASR pipeline analysis.

Logs transcription sessions to ~/.asr/logs/ in JSON-lines format for
analysis of accuracy, correction effectiveness, and performance.

Example usage:
    from asr.logging import TranscriptLogger

    logger = TranscriptLogger("input.m4a")
    logger.log_audio_prepared(duration=120.5, chunks=3)
    logger.log_chunk_transcribed(chunk_id=0, words=words)
    logger.log_correction_applied(segment_id=5, changes=[...])
    logger.log_correction_rejected(segment_id=7, reason="diff_gating", violations=["word"])
    logger.finalize()
"""

import atexit
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from asr.config import CONFIG_DIR

LOGS_DIR = CONFIG_DIR / "logs"


@dataclass
class SessionMetrics:
    """Aggregated metrics for a transcription session."""

    total_words: int = 0
    low_conf_words: int = 0
    total_segments: int = 0
    corrections_applied: int = 0
    corrections_rejected: int = 0
    diff_gating_rejections: int = 0
    chunks_processed: int = 0
    confidence_sum: float = 0.0
    confidence_count: int = 0


@dataclass
class TranscriptLogger:
    """Session-based logger for transcription pipeline events."""

    audio_file: str
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_file: Path = field(init=False)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    _started: datetime = field(default_factory=datetime.now)
    _config: dict[str, Any] = field(default_factory=dict)
    # PERFORMANCE: Buffer log events to reduce file I/O
    _log_buffer: list[dict[str, Any]] = field(default_factory=list)
    _BUFFER_SIZE: int = field(default=10, repr=False)
    # Thread safety for buffer operations
    _buffer_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = LOGS_DIR / f"session_{self.session_id}.jsonl"
        self._write_event("session_start", {
            "audio_file": self.audio_file,
            "timestamp": self._started.isoformat(),
        })

    def set_config(self, config: dict) -> None:
        """Record configuration for this session."""
        self._config = config
        self._write_event("config", config)

    def log_audio_prepared(
        self,
        duration_seconds: float,
        chunks: int,
        enhancement: dict | None = None,
    ) -> None:
        """Log audio preparation completion."""
        self.metrics.chunks_processed = chunks
        self._write_event("audio_prepared", {
            "duration_seconds": round(duration_seconds, 2),
            "chunks": chunks,
            "enhancement": enhancement or {},
        })

    def log_chunk_transcribed(
        self,
        chunk_id: int,
        words: list[dict],
        start_time: float,
        duration: float,
    ) -> None:
        """Log chunk transcription with confidence stats.

        Args:
            chunk_id: Chunk index
            words: List of word dicts with 'word' and 'probability' keys
            start_time: Chunk start time in audio
            duration: Chunk duration
        """
        word_count = len(words)
        low_conf_count = 0
        conf_sum = 0.0

        for w in words:
            prob = w.get("probability", 1.0)
            conf_sum += prob
            if prob < 0.7:  # Low confidence threshold
                low_conf_count += 1

        # Update metrics
        self.metrics.total_words += word_count
        self.metrics.low_conf_words += low_conf_count
        self.metrics.confidence_sum += conf_sum
        self.metrics.confidence_count += word_count

        low_conf_pct = (low_conf_count / word_count * 100) if word_count > 0 else 0
        avg_conf = (conf_sum / word_count) if word_count > 0 else 0

        # Find lowest confidence words for debugging
        low_conf_words = sorted(
            [(w["word"], w.get("probability", 1.0)) for w in words],
            key=lambda x: x[1]
        )[:5]

        self._write_event("chunk_transcribed", {
            "chunk_id": chunk_id,
            "start_time": round(start_time, 2),
            "duration": round(duration, 2),
            "word_count": word_count,
            "low_conf_count": low_conf_count,
            "low_conf_pct": round(low_conf_pct, 1),
            "avg_confidence": round(avg_conf, 3),
            "lowest_conf_words": low_conf_words,
        })

    def log_chunk_timing(
        self,
        chunk_index: int,
        total_chunks: int,
        chunk_duration_seconds: float,
        transcription_seconds: float,
        word_count: int,
        avg_confidence: float,
        memory_gb: Optional[float] = None,
    ) -> None:
        """Log per-chunk transcription metrics for performance analysis."""
        if chunk_duration_seconds > 0:
            chunk_rtf = transcription_seconds / chunk_duration_seconds
        else:
            chunk_rtf = 0
        self._write_event("chunk_timing", {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_duration_seconds": round(chunk_duration_seconds, 3),
            "transcription_seconds": round(transcription_seconds, 3),
            "chunk_rtf": round(chunk_rtf, 4),
            "word_count": word_count,
            "avg_confidence": round(avg_confidence, 4) if avg_confidence else None,
            "memory_gb": round(memory_gb, 2) if memory_gb else None,
        })

    def log_segment_created(self, segment_id: int, text: str, word_count: int) -> None:
        """Log segment creation."""
        self.metrics.total_segments += 1
        self._write_event("segment_created", {
            "segment_id": segment_id,
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "word_count": word_count,
        })

    def log_correction_attempted(
        self,
        segment_id: int,
        original_text: str,
        low_conf_regions: int,
    ) -> None:
        """Log correction attempt for a segment."""
        orig_preview = (
            original_text[:100] + "..." if len(original_text) > 100 else original_text
        )
        self._write_event("correction_attempted", {
            "segment_id": segment_id,
            "original_preview": orig_preview,
            "low_conf_regions": low_conf_regions,
        })

    def log_correction_applied(
        self,
        segment_id: int,
        changes: list[dict],
        corrected_text: str,
    ) -> None:
        """Log successful correction application."""
        self.metrics.corrections_applied += len(changes)
        corr_preview = (
            corrected_text[:100] + "..." if len(corrected_text) > 100 else corrected_text
        )
        self._write_event("correction_applied", {
            "segment_id": segment_id,
            "change_count": len(changes),
            "changes": changes,
            "corrected_preview": corr_preview,
        })

    def log_correction_rejected(
        self,
        segment_id: int,
        reason: str,
        violations: list[str] | None = None,
        details: dict | None = None,
    ) -> None:
        """Log rejected correction (diff gating, etc.)."""
        self.metrics.corrections_rejected += 1
        if reason == "diff_gating":
            self.metrics.diff_gating_rejections += 1

        self._write_event("correction_rejected", {
            "segment_id": segment_id,
            "reason": reason,
            "violations": violations[:10] if violations else [],
            "details": details or {},
        })

    def log_vocabulary_used(
        self,
        domain: str,
        term_count: int,
        terms_matched: list[str] | None = None,
    ) -> None:
        """Log vocabulary hints usage."""
        self._write_event("vocabulary_used", {
            "domain": domain,
            "term_count": term_count,
            "terms_matched": terms_matched or [],
        })

    def log_learning_candidates(
        self,
        candidates: list[str],
        accepted: list[str],
        rejected_hallucinated: list[str],
    ) -> None:
        """Log vocabulary learning results."""
        self._write_event("learning_candidates", {
            "total_candidates": len(candidates),
            "accepted": accepted,
            "rejected_hallucinated": rejected_hallucinated,
        })

    def log_pass_complete(
        self,
        pass_number: int,
        model: str,
        segments_modified: int,
        api_tokens: int | None = None,
    ) -> None:
        """Log correction pass completion."""
        self._write_event("pass_complete", {
            "pass_number": pass_number,
            "model": model,
            "segments_modified": segments_modified,
            "api_tokens": api_tokens,
        })

    def log_error(self, error_type: str, message: str, details: dict | None = None) -> None:
        """Log an error event."""
        self._write_event("error", {
            "error_type": error_type,
            "message": message,
            "details": details or {},
        })

    def finalize(self) -> dict:
        """Finalize session and write summary.

        Returns:
            Summary metrics dict
        """
        elapsed = (datetime.now() - self._started).total_seconds()
        avg_conf = (
            self.metrics.confidence_sum / self.metrics.confidence_count
            if self.metrics.confidence_count > 0 else 0
        )
        low_conf_pct = (
            self.metrics.low_conf_words / self.metrics.total_words * 100
            if self.metrics.total_words > 0 else 0
        )

        summary = {
            "duration_seconds": round(elapsed, 2),
            "total_words": self.metrics.total_words,
            "total_segments": self.metrics.total_segments,
            "chunks_processed": self.metrics.chunks_processed,
            "avg_confidence": round(avg_conf, 3),
            "low_conf_words": self.metrics.low_conf_words,
            "low_conf_pct": round(low_conf_pct, 1),
            "corrections_applied": self.metrics.corrections_applied,
            "corrections_rejected": self.metrics.corrections_rejected,
            "diff_gating_rejections": self.metrics.diff_gating_rejections,
            "correction_acceptance_rate": self._calc_acceptance_rate(),
        }

        self._write_event("session_complete", summary)
        # Ensure all buffered events are written before returning
        self._flush_logs()
        return summary

    def _calc_acceptance_rate(self) -> float | None:
        """Calculate correction acceptance rate."""
        total = self.metrics.corrections_applied + self.metrics.corrections_rejected
        if total > 0:
            return round(self.metrics.corrections_applied / total * 100, 1)
        return None

    def _write_event(self, event_type: str, data: dict) -> None:
        """Buffer a JSON event and flush when buffer is full.

        PERFORMANCE: Reduces file I/O by batching writes.
        Thread-safe: Uses _buffer_lock to prevent concurrent buffer modifications.
        """
        event = {
            "event": event_type,
            "ts": datetime.now().isoformat(),
            **data,
        }

        with self._buffer_lock:
            self._log_buffer.append(event)

            # Flush if buffer is full or if this is a critical event
            # chunk_timing included for real-time visibility during long transcriptions
            critical_events = {"session_start", "session_complete", "error", "chunk_timing"}
            buffer_full = len(self._log_buffer) >= self._BUFFER_SIZE
            should_flush = buffer_full or event_type in critical_events

        # Flush outside the lock to avoid holding lock during I/O
        if should_flush:
            self._flush_logs()

    def _flush_logs(self) -> None:
        """Flush buffered log events to disk.

        Thread-safe: Acquires _buffer_lock to safely copy and clear buffer.
        """
        with self._buffer_lock:
            if not self._log_buffer:
                return

            # Copy buffer before attempting write to prevent data loss on failure
            events_to_write = self._log_buffer.copy()

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for event in events_to_write:
                    try:
                        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                    except (TypeError, ValueError) as e:
                        # If serialization fails, write error placeholder
                        f.write(json.dumps({
                            "event": "serialization_error",
                            "ts": datetime.now().isoformat(),
                            "error": str(e),
                            "original_event_type": event.get("event", "unknown")
                        }) + "\n")

            # Only clear buffer after successful write
            with self._buffer_lock:
                # Remove only the events we successfully wrote
                # (in case new events were added during write)
                self._log_buffer = [e for e in self._log_buffer if e not in events_to_write]

        except IOError as e:
            # On I/O error, keep buffer intact for retry
            # Log to stderr as fallback (best effort)
            import sys
            print(f"Warning: Failed to flush logs to {self.log_file}: {e}", file=sys.stderr)


# Global logger instance for current session (thread-safe)
_current_logger: TranscriptLogger | None = None
_logger_lock = threading.Lock()


def get_logger() -> TranscriptLogger | None:
    """Get the current session logger (thread-safe)."""
    with _logger_lock:
        return _current_logger


def set_logger(logger: TranscriptLogger | None) -> None:
    """Set the current session logger (thread-safe)."""
    global _current_logger
    with _logger_lock:
        _current_logger = logger


def log_event(event_type: str, data: dict) -> None:
    """Convenience function to log to current session if active (thread-safe)."""
    with _logger_lock:
        if _current_logger:
            # Call _write_event while holding the lock to prevent race conditions
            # where logger could be set to None between check and write
            _current_logger._write_event(event_type, data)


def _flush_on_exit():
    """Flush any pending log events on process exit to prevent data loss."""
    with _logger_lock:
        if _current_logger and _current_logger._log_buffer:
            try:
                _current_logger._flush_logs()
            except Exception:
                pass  # Best-effort flush, ignore errors on exit


# Register atexit handler to prevent log buffer loss on crash
atexit.register(_flush_on_exit)


def analyze_logs(limit: int = 10) -> dict[str, Any]:
    """Analyze recent log sessions for patterns.

    Returns aggregated insights across sessions.
    """
    if not LOGS_DIR.exists():
        return {"error": "No logs directory found"}

    log_files = sorted(LOGS_DIR.glob("session_*.jsonl"), reverse=True)[:limit]

    if not log_files:
        return {"error": "No log files found"}

    sessions = []
    all_low_conf_words = []

    for log_file in log_files:
        session_data = {"file": log_file.name, "events": []}
        try:
            file_content = log_file.read_text(encoding="utf-8")
        except (IOError, OSError) as e:
            # Skip corrupted or unreadable files
            session_data["events"].append({
                "event": "file_read_error",
                "error": str(e)
            })
            sessions.append(session_data)
            continue

        for line in file_content.splitlines():
            if not line.strip():
                continue

            try:
                event = json.loads(line)
                session_data["events"].append(event)

                # Collect low-confidence words
                if event.get("event") == "chunk_transcribed":
                    for word, conf in event.get("lowest_conf_words", []):
                        all_low_conf_words.append((word.lower(), conf))

                # Capture session summary
                if event.get("event") == "session_complete":
                    session_data["summary"] = event

            except json.JSONDecodeError:
                # Skip malformed JSON lines
                continue

        sessions.append(session_data)

    # Aggregate stats
    total_words = sum(
        s.get("summary", {}).get("total_words", 0) for s in sessions
    )
    total_corrections = sum(
        s.get("summary", {}).get("corrections_applied", 0) for s in sessions
    )
    total_rejections = sum(
        s.get("summary", {}).get("corrections_rejected", 0) for s in sessions
    )

    # Safe average calculation
    sessions_with_summary = [s for s in sessions if "summary" in s]
    if sessions_with_summary:
        avg_confidence = sum(
            s.get("summary", {}).get("avg_confidence", 0) for s in sessions_with_summary
        ) / len(sessions_with_summary)
    else:
        avg_confidence = 0

    # Find most common low-confidence words
    word_counts: dict[str, int] = {}
    for word, _ in all_low_conf_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    common_low_conf = sorted(word_counts.items(), key=lambda x: -x[1])[:20]

    # Safe acceptance rate calculation
    total_correction_attempts = total_corrections + total_rejections
    acceptance_rate = None
    if total_correction_attempts > 0:
        acceptance_rate = round(total_corrections / total_correction_attempts * 100, 1)

    return {
        "sessions_analyzed": len(sessions),
        "total_words_transcribed": total_words,
        "avg_confidence": round(avg_confidence, 3),
        "total_corrections": total_corrections,
        "total_rejections": total_rejections,
        "acceptance_rate": acceptance_rate,
        "common_low_confidence_words": common_low_conf,
    }

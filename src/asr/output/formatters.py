"""Output formatters for transcripts."""

import html
import math
import re

from asr.models.transcript import Transcript


# =============================================================================
# Timestamp Formatting (DRY refactor - shared parser)
# =============================================================================

def _parse_time_components(seconds: float) -> tuple[int, int, int, int]:
    """Parse seconds into (hours, minutes, secs, millis) with validation.

    Shared helper to avoid duplicated calculation logic.
    """
    # Handle invalid values (negative, NaN, infinity, wrong type)
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = round((seconds % 1) * 1000)

    # Handle millisecond overflow (e.g., 999.9999 -> 1000ms)
    if millis >= 1000:
        millis = 0
        secs += 1
        if secs >= 60:
            secs = 0
            minutes += 1
            if minutes >= 60:
                minutes = 0
                hours += 1

    return hours, minutes, secs, millis


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    hours, minutes, secs, _ = _parse_time_components(seconds)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours, minutes, secs, millis = _parse_time_components(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_time(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours, minutes, secs, millis = _parse_time_components(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _escape_markdown(text: str) -> str:
    """Escape markdown special characters to prevent formatting issues.

    Escapes: \\ ` * _ { } [ ] ( ) # + - . ! |
    """
    # Characters that need escaping in markdown
    special_chars = r'\\`*_{}[\]()#+\-.!|'
    return re.sub(f'([{re.escape(special_chars)}])', r'\\\1', text)


def to_json(transcript: Transcript, indent: int = 2) -> str:
    """Export transcript as JSON with proper Unicode handling."""
    # ensure_ascii=False to preserve Unicode characters (é, ñ, 中文, etc.)
    # instead of escaping them as \uXXXX
    return transcript.model_dump_json(indent=indent, ensure_ascii=False)


def to_srt(transcript: Transcript) -> str:
    """Export transcript as SRT subtitle format."""
    if not transcript.segments:
        return ""

    lines = []
    entry_num = 1

    for seg in transcript.segments:
        start = format_srt_time(seg.start)
        end = format_srt_time(seg.end)
        # Replace newlines with spaces to avoid breaking SRT format
        # Handle None text safely
        text = (seg.text or "").strip().replace("\n", " ").replace("\r", "")

        # Skip empty segments
        if not text:
            continue

        lines.append(f"{entry_num}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # Blank line between entries
        entry_num += 1

    return "\n".join(lines) + "\n" if lines else ""


def to_vtt(transcript: Transcript) -> str:
    """Export transcript as WebVTT subtitle format."""
    lines = ["WEBVTT", ""]

    if not transcript.segments:
        return "\n".join(lines) + "\n"

    for seg in transcript.segments:
        start = format_vtt_time(seg.start)
        end = format_vtt_time(seg.end)
        # Replace newlines with spaces to avoid breaking VTT format
        # Handle None text safely
        text = (seg.text or "").strip().replace("\n", " ").replace("\r", "")

        # Skip empty segments
        if not text:
            continue

        # Escape HTML entities to prevent VTT parsing issues
        # WebVTT uses HTML-like markup, so <, >, & need escaping
        text = html.escape(text, quote=False)

        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines) + "\n"


def to_markdown(transcript: Transcript) -> str:
    """Export transcript as readable markdown with timestamps."""
    lines = []

    # Header
    lines.append("# Transcript")
    lines.append("")
    # Escape audio path since it goes in a code block (backticks in path would break it)
    lines.append(f"**Source:** `{_escape_markdown(transcript.audio_path)}`")
    lines.append(f"**Duration:** {format_timestamp(transcript.duration_seconds)}")
    lines.append(f"**Model:** {transcript.config.model} ({transcript.config.quantization})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Segments
    for seg in transcript.segments:
        # Handle None text safely and skip empty segments
        text = (seg.text or "").strip()
        if not text:
            continue

        ts = format_timestamp(seg.start)
        # Escape speaker name and text to prevent markdown formatting issues
        speaker_prefix = f"**{_escape_markdown(seg.speaker)}:** " if seg.speaker else ""
        lines.append(f"[{ts}] {speaker_prefix}{_escape_markdown(text)}")
        lines.append("")

    # Metadata if present
    if transcript.metadata.summary:
        lines.append("---")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(_escape_markdown(transcript.metadata.summary))
        lines.append("")

    if transcript.metadata.tags:
        lines.append("## Tags")
        lines.append("")
        # Tags in backticks need escaping - filter out None/empty tags
        valid_tags = [tag for tag in transcript.metadata.tags if tag and str(tag).strip()]
        if valid_tags:
            lines.append(", ".join(f"`{_escape_markdown(str(tag))}`" for tag in valid_tags))
            lines.append("")

    if transcript.metadata.tasks:
        lines.append("## Tasks")
        lines.append("")
        # Filter out None/empty tasks
        for task in transcript.metadata.tasks:
            if task and str(task).strip():
                lines.append(f"- [ ] {_escape_markdown(str(task))}")
        lines.append("")

    if transcript.metadata.decisions:
        lines.append("## Decisions")
        lines.append("")
        # Filter out None/empty decisions
        for decision in transcript.metadata.decisions:
            if decision and str(decision).strip():
                lines.append(f"- {_escape_markdown(str(decision))}")
        lines.append("")

    return "\n".join(lines)


def to_text(transcript: Transcript) -> str:
    """Export transcript as plain text."""
    # Handle None text safely
    text = "\n".join(
        (seg.text or "").strip()
        for seg in transcript.segments
        if seg.text and seg.text.strip()
    )
    return text + "\n" if text else ""


def format_transcript(transcript: Transcript, output_format: str) -> str:
    """Format transcript based on file extension or format name."""
    # Validate inputs
    if transcript is None:
        raise ValueError("Transcript cannot be None")
    if not output_format or not output_format.strip():
        raise ValueError("Output format cannot be empty")

    formatters = {
        "json": to_json,
        "srt": to_srt,
        "vtt": to_vtt,
        "md": to_markdown,
        "markdown": to_markdown,
        "txt": to_text,
        "text": to_text,
    }

    formatter = formatters.get(output_format.lower().lstrip("."))
    if not formatter:
        raise ValueError(
            f"Unknown format: {output_format}. "
            f"Supported: {', '.join(formatters.keys())}"
        )

    return formatter(transcript)

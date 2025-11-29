"""CLI for ASR transcription tool."""

import json
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from asr import __version__
from asr.audio.ingest import AudioChunk, prepare_audio
from asr.backends.base import BaseEngine
from asr.backends.crisperwhisper_engine import get_crisperwhisper_engine, DEFAULT_MODEL_PATH
from asr.config import apply_profile, load_config, save_config, PROFILES
from asr.logging import TranscriptLogger, set_logger, analyze_logs
from asr.models.transcript import Segment, Transcript, TranscriptConfig, TranscriptMetadata
from asr.output.formatters import format_transcript, format_timestamp

app = typer.Typer(
    name="asr",
    help="Local Whisper transcription optimized for Apple Silicon.",
    no_args_is_help=True,
)
console = Console()


# =============================================================================
# CLI Helpers (DRY refactor)
# =============================================================================

def _cli_error(message: str, detail: str | None = None) -> None:
    """Print formatted error message to console."""
    if detail:
        console.print(f"[red]Error:[/red] {message}: {detail}")
    else:
        console.print(f"[red]Error:[/red] {message}")


def _parse_vocabulary_terms(
    text: str | None,
    extract_after_colon: bool = False,
    validate: bool = False,
) -> list[str]:
    """Parse comma-separated terms with optional colon extraction and validation.

    Args:
        text: Comma-separated string of terms
        extract_after_colon: If True, extract text after ":" in each term
        validate: If True, validate each term with validate_vocabulary_term

    Returns:
        List of parsed and optionally validated terms
    """
    if not text:
        return []

    from asr.config import validate_vocabulary_term

    terms = []
    for part in text.split(","):
        part = part.strip()
        if extract_after_colon and ":" in part:
            part = part.split(":", 1)[1].strip()
        if not part:
            continue
        if validate and not validate_vocabulary_term(part):
            continue
        terms.append(part)
    return terms


def _safe_write_file(path: Path, content: str, description: str = "file") -> bool:
    """Write file with error handling and atomic write. Returns True on success."""
    temp_path = None
    try:
        # Atomic write: write to temp file first, then rename
        temp_path = path.with_suffix(path.suffix + '.tmp')
        temp_path.write_text(content, encoding='utf-8')
        temp_path.replace(path)  # Atomic on POSIX systems
        return True
    except PermissionError as e:
        _cli_error(f"Permission denied writing {description}", str(e))
        return False
    except OSError as e:
        _cli_error(f"Failed to write {description}", str(e))
        return False
    except UnicodeEncodeError as e:
        _cli_error(f"Unicode encoding error writing {description}", str(e))
        return False
    finally:
        # Clean up temp file if it exists and write failed
        if temp_path and temp_path.exists() and not path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass  # Best effort cleanup


def _transcribe_chunk(
    chunk: AudioChunk,
    engine: BaseEngine,
    word_timestamps: bool,
    language: str | None,
    prompt: str | None,
) -> tuple[float, list[Segment]]:
    """Worker function for parallel chunk transcription.

    Returns (start_time, segments) for ordering results.
    """
    segments = engine.transcribe(
        audio=chunk,
        word_timestamps=word_timestamps,
        language=language,
        initial_prompt=prompt,
    )
    return (chunk.start_time, segments)


def merge_segments(
    all_segments: list[list[Segment]],
    chunk_start_times: list[float] | None = None,
) -> list[Segment]:
    """Merge segments from multiple chunks, handling overlaps.

    Args:
        all_segments: List of segment lists, one per chunk
        chunk_start_times: Optional list of chunk start times for chunk_id assignment

    Returns new Segment objects with updated IDs and chunk_id (does not mutate originals).
    """
    if not all_segments:
        return []

    # Flatten with chunk_id tracking
    flat: list[tuple[Segment, int]] = []
    for chunk_idx, chunk_segs in enumerate(all_segments):
        for seg in chunk_segs:
            flat.append((seg, chunk_idx))

    if not flat:
        return []

    # Sort by start time
    flat.sort(key=lambda x: x[0].start)

    # Dedupe overlapping segments (simple approach: keep first occurrence)
    merged: list[tuple[Segment, int]] = []
    for seg, chunk_idx in flat:
        if not merged:
            merged.append((seg, chunk_idx))
            continue

        # Skip if this segment overlaps significantly with the last one
        last_seg, _ = merged[-1]
        seg_duration = seg.end - seg.start
        # Handle zero-duration segments (keep them to avoid division by zero)
        if seg_duration <= 0:
            merged.append((seg, chunk_idx))
            continue

        overlap = max(0, last_seg.end - seg.start)
        if overlap > seg_duration * 0.5:
            continue  # Skip duplicate

        merged.append((seg, chunk_idx))

    # Re-number segment IDs and assign chunk_id
    renumbered = []
    for i, (seg, chunk_idx) in enumerate(merged):
        renumbered.append(seg.model_copy(update={
            "id": i,
            "chunk_id": chunk_idx,
        }))

    # Assert monotonicity (segments should be sorted by start time)
    for i in range(len(renumbered) - 1):
        if renumbered[i].start > renumbered[i + 1].start:
            print(
                f"Warning: Non-monotonic segment timestamps at index {i}: "
                f"{renumbered[i].start:.2f}s > {renumbered[i + 1].start:.2f}s",
                file=sys.stderr,
            )

    return renumbered


@app.command()
def transcribe(
    file: Annotated[Path, typer.Argument(help="Audio file to transcribe")],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output file (.json, .srt, .md)")
    ] = None,
    profile: Annotated[
        Optional[str],
        typer.Option("--profile", help="Resource profile: interactive, balanced (default), batch")
    ] = None,
    language: Annotated[
        Optional[str],
        typer.Option("-l", "--lang", help="Language code (e.g., 'en')")
    ] = None,
    correct: Annotated[
        bool,
        typer.Option("--correct", help="Apply ASR error correction via Claude")
    ] = False,
    domain: Annotated[
        Optional[str],
        typer.Option("--domain", "-d", help="Domain context: biography, tech, medical, legal, etc.")
    ] = None,
    passes: Annotated[
        int,
        typer.Option("--passes", help="Number of correction passes (1 or 2)")
    ] = 2,
    no_vad: Annotated[
        bool,
        typer.Option("--no-vad", help="Disable voice activity detection")
    ] = False,
    no_words: Annotated[
        bool,
        typer.Option("--no-words", help="Disable word-level timestamps")
    ] = False,
    prompt: Annotated[
        Optional[str],
        typer.Option("--prompt", "-p", help="Context prompt: names, vocabulary, domain hints")
    ] = None,
    no_enhance: Annotated[
        bool,
        typer.Option("--no-enhance", help="Disable audio pre-processing (loudness norm, highpass)")
    ] = False,
    noise_reduction: Annotated[
        Optional[str],
        typer.Option("--noise", help="Noise reduction: off, light, moderate (default: off)")
    ] = None,
    learn: Annotated[
        bool,
        typer.Option("--learn", help="Learn vocabulary from corrections (adds to pending)")
    ] = False,
    warm_up: Annotated[
        bool,
        typer.Option("--warm-up", help="Pre-compile Metal kernels (reduces first-chunk latency)")
    ] = False,
    context: Annotated[
        Optional[str],
        typer.Option("--context", "-c", help="Dictionary context for bias list (e.g., 'running')")
    ] = None,
    no_auto_context: Annotated[
        bool,
        typer.Option("--no-auto-context", help="Disable auto-detection of dictionary context")
    ] = False,
):
    """Transcribe an audio file using CrisperWhisper."""
    # Validate input
    if not file.exists():
        _cli_error("File not found", str(file))
        raise typer.Exit(1)

    # Get CrisperWhisper engine
    if warm_up:
        console.print("[dim]Warming up model...[/dim]")
    engine = get_crisperwhisper_engine(warm_up=warm_up)
    if not engine.is_available():
        _cli_error(
            "CrisperWhisper model not found",
            f"Run: huggingface-cli download --local-dir {DEFAULT_MODEL_PATH} "
            "kyr0/crisperwhisper-unsloth-mlx"
        )
        raise typer.Exit(1)

    # Initialize session logger
    logger = TranscriptLogger(audio_file=str(file.name))
    set_logger(logger)

    # Load and configure
    config = load_config()

    # Apply resource profile if specified
    profile_word_timestamps = None
    if profile:
        if profile not in PROFILES:
            _cli_error("Invalid profile", f"'{profile}' (expected: {', '.join(PROFILES.keys())})")
            raise typer.Exit(1)
        config, profile_word_timestamps = apply_profile(config, profile)
        console.print(f"[dim]Profile: {profile} - {PROFILES[profile].description}[/dim]")

    if language:
        config.language = language
    if no_vad:
        config.use_vad = False

    # Audio enhancement options
    if no_enhance:
        config.audio_enhancement.loudness_enabled = False
        config.audio_enhancement.highpass_enabled = False
    if noise_reduction:
        if noise_reduction not in ("off", "light", "moderate"):
            _cli_error(
                "Invalid noise reduction",
                f"'{noise_reduction}' (expected: off, light, moderate)"
            )
            raise typer.Exit(1)
        config.audio_enhancement.noise_reduction = noise_reduction

    # Word timestamps: --no-words > profile > default(True)
    if no_words:
        word_timestamps = False
    elif profile_word_timestamps is not None:
        word_timestamps = profile_word_timestamps
    else:
        word_timestamps = True

    # Log configuration
    logger.set_config({
        "model": "crisperwhisper",
        "language": config.language,
        "use_vad": config.use_vad,
        "word_timestamps": word_timestamps,
        "correct": correct,
        "domain": domain,
        "passes": passes,
        "audio_enhancement": config.audio_enhancement.model_dump(),
    })

    # Merge prompt (CLI wins over config)
    effective_prompt = prompt or config.prompt

    # Apply dictionary context to override config if specified via CLI
    if context:
        config.dictionary_context = context
    if no_auto_context:
        config.dictionary_auto_detect = False

    # Generate dictionary-based prompt if context is set or auto-detect is enabled
    dictionary_prompt = None
    if config.dictionary_context or config.dictionary_auto_detect:
        try:
            from asr.dictionary import (
                BiasListSelector,
                generate_whisper_prompt,
            )

            # BiasListSelector uses SQLite directly
            selector = BiasListSelector()

            # Use explicit context or domain as context
            effective_context = config.dictionary_context or domain

            if effective_context or config.dictionary_auto_detect:
                # Select entries for bias list (uses selector's default max_entries=60)
                entries = selector.select_bias_list(context=effective_context)

                if entries:
                    # Pass context for meta-prompting
                    dictionary_prompt = generate_whisper_prompt(
                        entries, max_tokens=200, context=effective_context
                    )
                    if dictionary_prompt:
                        ctx_name = effective_context or 'default'
                        console.print(
                            f"[dim]Dictionary: {len(entries)} entries for '{ctx_name}'[/dim]"
                        )

        except Exception as e:
            # Dictionary system is optional - continue without it
            console.print(f"[dim]Dictionary unavailable: {e}[/dim]")

    # Merge dictionary prompt with user prompt (user prompt takes priority)
    if dictionary_prompt:
        if effective_prompt:
            effective_prompt = f"{dictionary_prompt}\n\n{effective_prompt}"
        else:
            effective_prompt = dictionary_prompt

    # Prepare audio
    console.print(f"[bold]Transcribing:[/bold] {file.name}")
    console.print(f"[dim]Model: CrisperWhisper FP16 | Word timestamps: {word_timestamps}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Prepare audio with VAD/chunking
        task = progress.add_task("Preparing audio...", total=None)
        chunks, duration = prepare_audio(file, config)
        progress.update(task, completed=True, description=f"Audio prepared ({len(chunks)} chunks)")

        # Log audio preparation
        logger.log_audio_prepared(
            duration_seconds=duration,
            chunks=len(chunks),
            enhancement={
                "loudness_enabled": config.audio_enhancement.loudness_enabled,
                "highpass_enabled": config.audio_enhancement.highpass_enabled,
                "noise_reduction": config.audio_enhancement.noise_reduction,
            },
        )

        # Transcribe chunks
        task = progress.add_task("Transcribing...", total=len(chunks))
        all_segments = []

        # Start timing for RTF calculation
        transcription_start = time.time()

        # Use parallel transcription for multiple chunks
        max_workers = min(config.max_parallel_chunks, len(chunks))

        if max_workers > 1 and len(chunks) > 1:
            # Parallel transcription with error handling
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _transcribe_chunk,
                        chunk,
                        engine,
                        word_timestamps,
                        config.language,
                        effective_prompt,
                    ): chunk
                    for chunk in chunks
                }

                results = []
                failed_chunks = []
                for future in as_completed(futures):
                    chunk = futures[future]
                    try:
                        start_time, segments = future.result(timeout=600)
                        results.append((start_time, segments))
                    except Exception as e:
                        # Log error but continue with other chunks
                        failed_chunks.append((chunk.start_time, str(e)))
                        logger.log_error(
                            "chunk_transcription_failed",
                            f"Chunk at {chunk.start_time:.1f}s failed: {e}",
                        )
                    progress.advance(task)

                if failed_chunks:
                    console.print(
                        f"[yellow]Warning:[/yellow] {len(failed_chunks)} chunk(s) failed"
                    )
                    for start_time, error in failed_chunks[:3]:
                        console.print(f"  [dim]Chunk at {start_time:.1f}s: {error[:100]}[/dim]")

                # Sort by start time to maintain correct order
                results.sort(key=lambda x: x[0])
                all_segments = [segs for _, segs in results]
        else:
            # Sequential transcription (single chunk or parallel disabled)
            # Use cross-chunk context: rolling buffer of recent text improves accuracy
            context_buffer = ""
            MAX_CONTEXT_WORDS = 100  # ~20-30 seconds of speech

            HARD_CHUNK_THRESHOLD = 0.4  # Skip word timestamps for difficult audio

            for chunk_idx, chunk in enumerate(chunks):
                # Build prompt: vocab/bias FIRST, context LAST (gets trimmed)
                bias_block = effective_prompt or ""
                ctx_block = f'\nPrevious transcript: "{context_buffer}"' if context_buffer else ""

                # Whisper initial_prompt is limited; trim context to fit, keep vocab intact
                available = 500 - len(bias_block)
                chunk_prompt = bias_block + ctx_block[:available] if available > 0 else bias_block

                # Adaptive word timestamps: skip for hard chunks (20-30% faster)
                chunk_word_timestamps = word_timestamps
                if word_timestamps and chunk.difficulty_score > HARD_CHUNK_THRESHOLD:
                    chunk_word_timestamps = False

                chunk_start = time.time()
                segments = engine.transcribe(
                    audio=chunk,
                    word_timestamps=chunk_word_timestamps,
                    language=config.language,
                    initial_prompt=chunk_prompt if chunk_prompt.strip() else None,
                )
                chunk_transcription_time = time.time() - chunk_start
                all_segments.append(segments)

                # Log per-chunk timing metrics
                chunk_word_count = sum(len(seg.words) if seg.words else 0 for seg in segments)
                chunk_avg_conf = (
                    sum(seg.confidence for seg in segments) / len(segments)
                    if segments else 0
                )
                logger.log_chunk_timing(
                    chunk_index=chunk_idx,
                    total_chunks=len(chunks),
                    chunk_duration_seconds=chunk.duration,
                    transcription_seconds=chunk_transcription_time,
                    word_count=chunk_word_count,
                    avg_confidence=chunk_avg_conf,
                )

                # Log per-chunk word confidence details
                all_words = []
                for seg in segments:
                    if seg.words:
                        all_words.extend(
                            {"word": w.word, "probability": w.confidence}
                            for w in seg.words
                        )
                if all_words:
                    logger.log_chunk_transcribed(
                        chunk_id=chunk_idx,
                        words=all_words,
                        start_time=chunk.start_time,
                        duration=chunk.duration,
                    )

                # Update rolling context buffer with this chunk's output
                for seg in segments:
                    context_buffer += " " + seg.text
                # Trim to last N words
                words = context_buffer.split()
                context_buffer = " ".join(words[-MAX_CONTEXT_WORDS:])

                progress.advance(task)

        progress.update(task, description="Transcription complete")

        # Calculate and log RTF
        transcription_elapsed = time.time() - transcription_start
        rtf = duration / transcription_elapsed if transcription_elapsed > 0 else 0

        # Log RTF and check for pathologies
        console.print(
            f"[dim]Transcription: {transcription_elapsed:.1f}s | RTF: {rtf:.1f}x[/dim]"
        )

        # Pathology warnings
        if len(chunks) > 40:
            console.print(
                f"[yellow]Warning:[/yellow] High chunk count ({len(chunks)}) - "
                "VAD may be too aggressive"
            )
        if rtf < 1.0 and duration > 60:  # Only warn for files > 1 min
            console.print(
                f"[yellow]Warning:[/yellow] RTF below realtime ({rtf:.2f}x)"
            )

    # Merge segments (pass chunk start times for chunk_id assignment)
    chunk_start_times = [c.start_time for c in chunks]
    segments = merge_segments(all_segments, chunk_start_times)

    # Log transcription performance
    from asr.logging import log_event
    log_event("transcription_complete", {
        "elapsed_seconds": round(transcription_elapsed, 2),
        "rtf": round(rtf, 2),
        "chunks": len(chunks),
        "segments": len(segments) if segments else 0,
    })

    # Apply correction if requested
    if correct:
        from asr.nlp.corrector import Corrector
        from asr.nlp.models import CorrectionConfig
        from asr.config import load_domain_vocabulary

        # Parse vocabulary from prompt (using DRY helper)
        vocabulary = _parse_vocabulary_terms(effective_prompt, extract_after_colon=True)

        # Load domain vocabulary (as hints, not forced)
        if domain:
            domain_vocab = load_domain_vocabulary(domain)
            if domain_vocab:
                console.print(f"[dim]Loaded {len(domain_vocab)} {domain} vocab terms[/dim]")
                vocabulary.extend(domain_vocab)

            # Also load built-in vocabulary for the domain
            try:
                from asr.vocabularies import get_domain_vocabulary
                builtin_vocab = get_domain_vocabulary(domain)
                if builtin_vocab:
                    vocabulary.extend(builtin_vocab)
            except ImportError:
                pass  # Built-in vocabularies not available

        correction_config = CorrectionConfig(
            passes=min(2, max(1, passes)),
            vocabulary=vocabulary,
            domain=domain,
        )

        console.print(
            f"[dim]Applying {correction_config.passes}-pass correction (Sonnet 4.5)...[/dim]"
        )

        # Store original segments for vocabulary learning
        original_segments = segments if learn else None

        # Use context manager to ensure Anthropic client is properly closed
        # Pass dictionary context for enhanced correction
        effective_context = config.dictionary_context or domain
        with Corrector(provider="claude", config=correction_config) as corrector:
            segments = corrector.correct(
                segments,
                vocabulary=vocabulary,
                domain=domain,
                dictionary_context=effective_context,
                dictionary_entries_limit=config.correction.dictionary_entries_for_correction,
            )

        # Extract and store learnable terms if requested
        if learn and domain and original_segments:
            from asr.config import add_pending_terms

            # Collect original words for safety gating
            original_words = set()
            for seg in original_segments:
                if seg.words:
                    for word in seg.words:
                        if word.word:
                            original_words.add(word.word)
                if seg.raw_text:
                    for word in seg.raw_text.split():
                        stripped = word.strip(".,!?;:\"'")
                        if stripped:
                            original_words.add(stripped)

            # Extract candidate terms from correction changes
            candidate_terms = []
            for seg in segments:
                if seg.corrections.applied:
                    for change_str in seg.corrections.changes:
                        # Parse "original -> corrected" format
                        if " -> " in change_str:
                            _, corrected = change_str.split(" -> ", 1)
                            corrected = corrected.strip()
                            # Only consider proper nouns (capitalized)
                            if corrected and corrected[0].isupper():
                                candidate_terms.append(corrected)

            if candidate_terms:
                added = add_pending_terms(
                    candidate_terms,
                    domain=domain,
                    source=file.name,
                    original_words=original_words,  # Safety gating
                )
                if added > 0:
                    console.print(
                        f"[dim]Added {added} terms to pending (review: 'asr vocab pending')[/dim]"
                    )

            # Run NER discovery if available
            try:
                from asr.dictionary import (
                    _NER_AVAILABLE,
                    discover_proper_nouns,
                    add_discovered_to_pending,
                )
                if _NER_AVAILABLE:
                    discovery_result = discover_proper_nouns(
                        segments=segments,
                        source_file=file,
                        min_confidence=0.7,
                        context=effective_context or domain,
                    )
                    if discovery_result.discovered:
                        pending_added, auto_approved = add_discovered_to_pending(
                            discovery_result.discovered,
                            context=effective_context or domain or "discovered",
                        )
                        if pending_added > 0 or auto_approved > 0:
                            discovered_count = len(discovery_result.discovered)
                            console.print(
                                f"[dim]NER discovered {discovered_count} proper nouns: "
                                f"{pending_added} pending, {auto_approved} auto-approved[/dim]"
                            )
                        if discovery_result.rejected_hallucinations:
                            rejected_count = len(discovery_result.rejected_hallucinations)
                            console.print(f"[dim]Rejected {rejected_count} hallucinations[/dim]")
            except ImportError:
                pass  # NER not installed

    # Build transcript
    transcript = Transcript(
        audio_path=str(file.absolute()),
        duration_seconds=duration,
        created_at=datetime.now(),
        config=TranscriptConfig(
            model="crisperwhisper",
            backend="mlx",
            quantization="fp16",
            language=config.language,
        ),
        segments=segments,
        metadata=TranscriptMetadata(),
    )

    # Determine output format with error handling
    if output:
        fmt = output.suffix.lstrip(".")
        content = format_transcript(transcript, fmt)
        if not _safe_write_file(output, content, f"transcript to {output}"):
            raise typer.Exit(1)
        console.print(f"[green]Saved:[/green] {output}")
    else:
        # Default: print to terminal and save JSON
        console.print()
        for seg in segments:
            ts = format_timestamp(seg.start)
            console.print(f"[dim][{ts}][/dim] {seg.text}")

        # Save JSON alongside input with error handling
        json_path = file.with_suffix(".json")
        json_content = format_transcript(transcript, "json")
        if not _safe_write_file(json_path, json_content, f"JSON to {json_path}"):
            raise typer.Exit(1)
        console.print()
        console.print(f"[green]Saved:[/green] {json_path}")

    # Finalize session logging
    summary = logger.finalize()
    set_logger(None)
    console.print()
    console.print(f"[dim]Session logged: {logger.log_file.name}[/dim]")
    if summary.get("corrections_applied") or summary.get("corrections_rejected"):
        console.print(
            f"[dim]Corrections: {summary.get('corrections_applied', 0)} applied, "
            f"{summary.get('corrections_rejected', 0)} rejected "
            f"(avg confidence: {summary.get('avg_confidence', 0):.1%})[/dim]"
        )


@app.command(name="config")
def config_cmd(
    show: Annotated[
        bool,
        typer.Option("--show", help="Show current configuration")
    ] = False,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Reset to default configuration")
    ] = False,
):
    """Manage ASR configuration."""
    from asr.config import ASRConfig, CONFIG_FILE

    if reset:
        config = ASRConfig()
        save_config(config)
        console.print("[green]Configuration reset to defaults.[/green]")
        show = True

    if show or reset:
        config = load_config()
        console.print(f"[bold]Configuration:[/bold] {CONFIG_FILE}")
        console.print()
        for key, value in config.model_dump().items():
            if key == "correction":
                console.print(f"  [dim]correction.enabled:[/dim] {value['enabled']}")
                console.print(f"  [dim]correction.passes:[/dim] {value['passes']}")
                console.print(f"  [dim]correction.quality:[/dim] {value['quality']}")
            elif key == "audio_enhancement":
                console.print(f"  [dim]ae.loudness_enabled:[/dim] {value['loudness_enabled']}")
                lufs = value['loudness_target_lufs']
                console.print(f"  [dim]ae.loudness_target_lufs:[/dim] {lufs}")
                console.print(f"  [dim]ae.highpass_enabled:[/dim] {value['highpass_enabled']}")
                console.print(f"  [dim]ae.highpass_freq:[/dim] {value['highpass_freq']}")
                console.print(f"  [dim]ae.noise_reduction:[/dim] {value['noise_reduction']}")
            else:
                console.print(f"  [dim]{key}:[/dim] {value}")


@app.command()
def clear_cache():
    """Clear the audio cache directory."""
    import shutil
    config = load_config()
    if config.cache_dir.exists():
        # Calculate size with error handling for concurrent file changes
        size = 0
        for f in config.cache_dir.rglob("*"):
            try:
                if f.is_file():
                    size += f.stat().st_size
            except (FileNotFoundError, PermissionError, OSError):
                continue  # File deleted or inaccessible during enumeration

        try:
            shutil.rmtree(config.cache_dir)
            config.cache_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]Cleared cache:[/green] {size / 1024 / 1024:.1f} MB freed")
        except PermissionError as e:
            console.print(f"[red]Error:[/red] Permission denied clearing cache: {e}")
            raise typer.Exit(1)
        except OSError as e:
            console.print(f"[red]Error:[/red] Failed to clear cache: {e}")
            raise typer.Exit(1)
    else:
        console.print("[dim]Cache directory is empty.[/dim]")


@app.command()
def correct_transcript(
    file: Annotated[Path, typer.Argument(help="JSON transcript file to correct")],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output file (default: overwrite input)")
    ] = None,
    prompt: Annotated[
        Optional[str],
        typer.Option("--prompt", "-p", help="Vocabulary hints: names, terms")
    ] = None,
    domain: Annotated[
        Optional[str],
        typer.Option("--domain", "-d", help="Domain context")
    ] = None,
    passes: Annotated[
        int,
        typer.Option("--passes", help="Number of passes (1 or 2)")
    ] = 2,
):
    """Post-process an existing transcript with Claude Sonnet correction."""
    import json
    from asr.nlp.corrector import Corrector
    from asr.nlp.models import CorrectionConfig

    if not file.exists():
        _cli_error("File not found", str(file))
        raise typer.Exit(1)

    if file.suffix != ".json":
        _cli_error("Expected JSON file", f"got {file.suffix}")
        raise typer.Exit(1)

    # Load transcript
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        transcript = Transcript.model_validate(data)
    except json.JSONDecodeError as e:
        _cli_error("Invalid JSON file", str(e))
        raise typer.Exit(1)
    except Exception as e:
        _cli_error("Failed to load transcript", str(e))
        raise typer.Exit(1)

    # Parse vocabulary (using DRY helper)
    vocabulary = _parse_vocabulary_terms(prompt, extract_after_colon=True)

    correction_config = CorrectionConfig(
        passes=min(2, max(1, passes)),
        vocabulary=vocabulary,
        domain=domain,
    )

    console.print(f"[bold]Correcting:[/bold] {file.name}")
    vocab_str = vocabulary or 'none'
    console.print(f"[dim]{correction_config.passes}-pass | Vocab: {vocab_str}[/dim]")

    # Use context manager to ensure Anthropic client is properly closed
    # Pass domain as dictionary context for enhanced correction
    with Corrector(provider="claude", config=correction_config) as corrector:
        corrected_segments = corrector.correct(
            transcript.segments,
            vocabulary=vocabulary,
            domain=domain,
            dictionary_context=domain,  # Use domain as dictionary context
        )

    # Update transcript
    transcript.segments = corrected_segments

    # Save
    out_path = output or file
    json_content = format_transcript(transcript, "json")
    if not _safe_write_file(out_path, json_content, f"corrected transcript to {out_path}"):
        raise typer.Exit(1)
    console.print(f"[green]Saved:[/green] {out_path}")


@app.command()
def bench(
    file: Annotated[Path, typer.Argument(help="Audio file to benchmark")],
    iterations: Annotated[
        int,
        typer.Option("-n", "--iterations", help="Number of iterations per config")
    ] = 2,
    warm_up: Annotated[
        bool,
        typer.Option("--warm-up", help="Pre-compile Metal kernels before benchmark")
    ] = True,
    json_output: Annotated[
        Optional[Path],
        typer.Option("--json", help="Export benchmark results to JSON file")
    ] = None,
):
    """Benchmark transcription with different settings.

    Runs transcription with various quantization levels and reports:
    - Wall-clock time per iteration
    - Real-Time Factor (RTF) - how many seconds of audio processed per second
    - Peak memory usage
    - Word count and confidence statistics

    Example:
        asr bench recording.m4a -n 3
    """
    import time
    import psutil
    import os

    if not file.exists():
        _cli_error("File not found", str(file))
        raise typer.Exit(1)

    # Get engine with optional warm-up
    if warm_up:
        console.print("[dim]Warming up model...[/dim]")
    engine = get_crisperwhisper_engine(warm_up=warm_up)
    if not engine.is_available():
        _cli_error("CrisperWhisper model not found")
        raise typer.Exit(1)

    config = load_config()

    console.print(f"[bold]Benchmarking:[/bold] {file.name}")
    console.print(f"[dim]Iterations: {iterations} | Warm-up: {warm_up}[/dim]")
    console.print()

    # Prepare audio once (shared across all runs)
    console.print("[dim]Preparing audio...[/dim]")
    chunks, duration = prepare_audio(file, config)
    console.print(f"[dim]Audio: {duration:.1f}s, {len(chunks)} chunks[/dim]")
    console.print()

    process = psutil.Process(os.getpid())

    console.print("[bold]CrisperWhisper FP16[/bold]")
    run_results = []

    for i in range(iterations):
        # Transcribe
        start = time.time()
        all_segments = []
        for chunk in chunks:
            segments = engine.transcribe(
                audio=chunk,
                word_timestamps=True,
                language=config.language,
                initial_prompt=None,
            )
            all_segments.extend(segments)
        elapsed = time.time() - start

        # Memory after
        mem_after = process.memory_info().rss / 1024**3

        # Calculate metrics
        rtf = duration / elapsed if elapsed > 0 else 0
        word_count = sum(len(seg.words) if seg.words else 0 for seg in all_segments)
        avg_conf = 0.0
        if all_segments:
            confs = [seg.confidence for seg in all_segments if seg.confidence]
            avg_conf = sum(confs) / len(confs) if confs else 0.0

        run_results.append({
            "elapsed": elapsed,
            "rtf": rtf,
            "memory_gb": mem_after,
            "words": word_count,
            "confidence": avg_conf,
        })

        console.print(
            f"  Run {i+1}: {elapsed:.2f}s | RTF: {rtf:.1f}x | "
            f"Mem: {mem_after:.1f}GB | Words: {word_count} | Conf: {avg_conf:.1%}"
        )

    # Average
    avg_elapsed = sum(r["elapsed"] for r in run_results) / len(run_results)
    avg_rtf = sum(r["rtf"] for r in run_results) / len(run_results)
    avg_mem = sum(r["memory_gb"] for r in run_results) / len(run_results)
    avg_words = sum(r["words"] for r in run_results) / len(run_results)
    avg_conf = sum(r["confidence"] for r in run_results) / len(run_results)

    console.print()
    console.print("[bold]Summary[/bold]")
    console.print(f"  Audio duration: {duration:.1f}s ({len(chunks)} chunks)")
    console.print(f"  Avg time:       {avg_elapsed:.2f}s")
    console.print(f"  Avg RTF:        {avg_rtf:.1f}x realtime")
    console.print(f"  Avg memory:     {avg_mem:.1f}GB")
    console.print(f"  Avg words:      {int(avg_words)}")
    console.print(f"  Avg confidence: {avg_conf:.1%}")

    # Export to JSON if requested
    if json_output:
        # Calculate standard deviation
        rtf_values = [r["rtf"] for r in run_results]
        std_rtf = statistics.stdev(rtf_values) if len(rtf_values) > 1 else 0.0

        # Build JSON structure
        json_data = {
            "audio_file": str(file.resolve()),
            "audio_duration_seconds": duration,
            "iterations": iterations,
            "warm_up": warm_up,
            "results": [
                {
                    "iteration": i + 1,
                    "elapsed_seconds": r["elapsed"],
                    "rtf": r["rtf"],
                    "peak_memory_gb": r["memory_gb"],
                    "word_count": r["words"],
                    "avg_confidence": r["confidence"],
                }
                for i, r in enumerate(run_results)
            ],
            "summary": {
                "avg_rtf": avg_rtf,
                "min_rtf": min(rtf_values),
                "max_rtf": max(rtf_values),
                "std_rtf": std_rtf,
                "avg_memory_gb": avg_mem,
                "avg_confidence": avg_conf,
            },
            "system_info": {
                "platform": platform.system(),
                "python_version": sys.version.split()[0],
                "model": "crisperwhisper",
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Write JSON file with pretty printing
        json_content = json.dumps(json_data, indent=2)
        if not _safe_write_file(json_output, json_content, "benchmark results"):
            raise typer.Exit(1)
        console.print()
        console.print(f"[green]Benchmark results saved:[/green] {json_output}")


@app.command()
def vocab(
    action: Annotated[
        str,
        typer.Argument(help="Action: add, list, show, remove, pending, approve, reject")
    ],
    domain: Annotated[
        Optional[str],
        typer.Option("-d", "--domain", help="Domain name")
    ] = None,
    terms: Annotated[
        Optional[str],
        typer.Option("-t", "--terms", help="Comma-separated terms to add/approve/reject")
    ] = None,
    file: Annotated[
        Optional[Path],
        typer.Option("-f", "--file", help="Load terms from file (one per line)")
    ] = None,
    auto: Annotated[
        bool,
        typer.Option("--auto", help="Auto-approve terms with 3+ occurrences")
    ] = False,
):
    """Manage domain-specific vocabularies.

    Vocabularies are stored in ~/.asr/vocabularies/{domain}.txt
    and used as hints for transcription and correction.

    SAFETY: Vocabulary learning uses a pending queue to prevent hallucinations:
    - Terms are first added to pending (requires manual approval)
    - Terms with 3+ occurrences can be auto-approved with --auto
    - Use 'asr vocab pending' to review, 'approve' to accept, 'reject' to discard

    Examples:
        asr vocab list                          # List all domains
        asr vocab show -d biography             # Show terms in biography domain
        asr vocab add -d biography -t "Ron Chernow, MacBook Air"
        asr vocab pending                       # Show pending terms
        asr vocab approve -t "Ron Chernow"      # Approve specific term
        asr vocab approve --auto                # Auto-approve high-confidence terms
        asr vocab reject -t "wrong term"        # Reject hallucinated term
    """
    from asr.config import (
        approve_pending_terms,
        get_auto_approve_candidates,
        list_domains,
        load_domain_vocabulary,
        load_pending_vocabulary,
        reject_pending_terms,
        remove_domain_vocabulary,
        save_domain_vocabulary,
        validate_vocabulary_term,
        VOCAB_DIR,
    )

    if action == "list":
        domains = list_domains()
        if not domains:
            console.print("[dim]No vocabularies found.[/dim]")
            console.print("[dim]Create: asr vocab add -d <domain> -t \"term1, term2\"[/dim]")
            return

        console.print(f"[bold]Vocabularies:[/bold] {VOCAB_DIR}")
        for d in domains:
            terms = load_domain_vocabulary(d)
            console.print(f"  {d}: {len(terms)} terms")

    elif action == "show":
        if not domain:
            _cli_error("--domain required for show")
            raise typer.Exit(1)

        vocab_terms = load_domain_vocabulary(domain)
        if not vocab_terms:
            console.print(f"[dim]No vocabulary found for domain: {domain}[/dim]")
            return

        console.print(f"[bold]{domain}[/bold] ({len(vocab_terms)} terms)")
        for term in vocab_terms:
            console.print(f"  {term}")

    elif action == "add":
        if not domain:
            _cli_error("--domain required for add")
            raise typer.Exit(1)

        new_terms = []

        # From --terms option
        if terms:
            for t in terms.split(","):
                t = t.strip()
                if t and validate_vocabulary_term(t):
                    new_terms.append(t)
                elif t:
                    console.print(f"[yellow]Skipped invalid term:[/yellow] {t}")

        # From file
        if file:
            if not file.exists():
                _cli_error("File not found", str(file))
                raise typer.Exit(1)
            for line in file.read_text(encoding="utf-8").splitlines():
                t = line.strip()
                if t and not t.startswith("#") and validate_vocabulary_term(t):
                    new_terms.append(t)

        if not new_terms:
            console.print("[yellow]No valid terms to add.[/yellow]")
            return

        added = save_domain_vocabulary(domain, new_terms)
        console.print(f"[green]Added {added} new terms to {domain}[/green]")
        total = len(load_domain_vocabulary(domain))
        console.print(f"[dim]Total: {total} terms[/dim]")

    elif action == "remove":
        if not domain:
            _cli_error("--domain required for remove")
            raise typer.Exit(1)

        if remove_domain_vocabulary(domain):
            console.print(f"[green]Removed vocabulary: {domain}[/green]")
        else:
            console.print(f"[yellow]No vocabulary found: {domain}[/yellow]")

    elif action == "pending":
        pending = load_pending_vocabulary()
        if not pending:
            console.print("[dim]No pending vocabulary terms.[/dim]")
            return

        console.print(f"[bold]Pending Terms:[/bold] ({len(pending)} awaiting approval)")
        console.print()

        for term, info in sorted(pending.items()):
            occ = info["occurrences"]
            occ_str = f"[green]{occ}x[/green]" if occ >= 3 else f"[yellow]{occ}x[/yellow]"
            console.print(f"  {occ_str} [bold]{term}[/bold] ({info['domain']})")
            if len(info["sources"]) <= 3:
                for src in info["sources"]:
                    console.print(f"      [dim]from: {src}[/dim]")

        # Show auto-approve hint
        auto_candidates = get_auto_approve_candidates()
        if auto_candidates:
            console.print()
            count = len(auto_candidates)
            console.print(f"[dim]{count} terms 3+ occurrences (auto-approvable)[/dim]")
            console.print("[dim]Run: asr vocab approve --auto[/dim]")

    elif action == "approve":
        if auto:
            # Auto-approve terms with 3+ occurrences
            candidates = get_auto_approve_candidates()
            if not candidates:
                console.print("[dim]No terms qualify for auto-approval (need 3+ occurrences)[/dim]")
                return

            terms_to_approve = [term for term, _ in candidates]
            approved = approve_pending_terms(terms_to_approve)
            console.print(f"[green]Auto-approved {approved} terms[/green]")
            for term, info in candidates:
                console.print(f"  {term} -> {info['domain']}")

        elif terms:
            # Approve specific terms
            term_list = [t.strip() for t in terms.split(",")]
            approved = approve_pending_terms(term_list, domain=domain)
            console.print(f"[green]Approved {approved} terms[/green]")

        else:
            _cli_error("Specify --terms or --auto")
            raise typer.Exit(1)

    elif action == "reject":
        if not terms:
            _cli_error("--terms required for reject")
            raise typer.Exit(1)

        term_list = [t.strip() for t in terms.split(",")]
        rejected = reject_pending_terms(term_list)
        console.print(f"[green]Rejected {rejected} terms[/green]")

    else:
        valid = "list, show, add, remove, pending, approve, reject"
        _cli_error("Unknown action", f"'{action}' (valid: {valid})")
        raise typer.Exit(1)


@app.command(name="dict")
def dictionary(
    action: Annotated[
        str,
        typer.Argument(help="Action: stats, pending, approve, reject, search, load, learn")
    ],
    term: Annotated[
        Optional[str],
        typer.Option("-t", "--term", help="Term to approve/reject/search")
    ] = None,
    context: Annotated[
        Optional[str],
        typer.Option("-c", "--context", help="Context for filtering/adding")
    ] = None,
    seeds_dir: Annotated[
        Optional[Path],
        typer.Option("--seeds", help="Directory with seed JSON files to load")
    ] = None,
):
    """Manage the proper noun dictionary system.

    The dictionary stores proper nouns with metadata (type, tier, aliases)
    for context-aware transcription bias and correction.

    NER discovery adds validated nouns to a pending queue.
    Use 'pending', 'approve', 'reject' to manage them.

    Examples:
        asr dict stats                          # Show dictionary statistics
        asr dict pending                        # Show pending discovered nouns
        asr dict approve -t "Ron Chernow"       # Approve a pending noun
        asr dict reject -t "hallucinated"       # Reject a pending noun
        asr dict search -t "chernow"            # Search dictionary
        asr dict load --seeds seeds/            # Load seed files
        asr dict learn                          # Batch learn from session logs
        asr dict learn -c biography             # Learn with context
    """
    from asr.dictionary import (
        init_db,
        get_pending_nouns,
        approve_pending_noun,
        reject_pending_noun,
        search_entries,
        import_from_json,
        BiasListSelector,
    )
    from asr.dictionary.db import get_stats

    init_db()

    if action == "stats":
        stats = get_stats()
        console.print("[bold]Dictionary Statistics[/bold]")
        console.print(f"  Total entries: {stats.total_entries}")
        console.print("  By tier:")
        for tier, count in sorted(stats.entries_by_tier.items()):
            console.print(f"    {tier}: {count}")
        if stats.entries_by_type:
            console.print("  By type:")
            for t, count in sorted(stats.entries_by_type.items()):
                console.print(f"    {t}: {count}")

        # Show pending count
        pending = get_pending_nouns()
        if pending:
            console.print(f"\n  [yellow]Pending approval: {len(pending)}[/yellow]")
            console.print("  [dim]Review with: asr dict pending[/dim]")

    elif action == "pending":
        pending = get_pending_nouns()
        if not pending:
            console.print("[dim]No pending discovered nouns.[/dim]")
            console.print("[dim]Run 'asr transcribe --correct --learn' to discover.[/dim]")
            return

        console.print(f"[bold]Pending Discovered Nouns[/bold] ({len(pending)} total)")
        for item in sorted(pending, key=lambda x: -x.get("occurrences", 1)):
            occurrences = item.get("occurrences", 1)
            sources = item.get("sources", [])
            validation = item.get("validation", "unknown")
            console.print(f"  {item['text']} ({item['type']}) - {occurrences}x [{validation}]")
            if sources:
                console.print(f"    [dim]Sources: {', '.join(sources[:3])}[/dim]")

        console.print()
        console.print("[dim]Approve: asr dict approve -t \"term\"[/dim]")
        console.print("[dim]Reject: asr dict reject -t \"term\"[/dim]")

    elif action == "approve":
        if not term:
            _cli_error("--term required for approve")
            raise typer.Exit(1)

        if approve_pending_noun(term, context=context):
            console.print(f"[green]Approved:[/green] {term}")
        else:
            console.print(f"[yellow]Not found in pending:[/yellow] {term}")

    elif action == "reject":
        if not term:
            _cli_error("--term required for reject")
            raise typer.Exit(1)

        if reject_pending_noun(term):
            console.print(f"[green]Rejected:[/green] {term}")
        else:
            console.print(f"[yellow]Not found in pending:[/yellow] {term}")

    elif action == "search":
        if not term:
            _cli_error("--term required for search")
            raise typer.Exit(1)

        results = search_entries(term, limit=10)
        if not results:
            console.print(f"[dim]No matches for:[/dim] {term}")
            return

        console.print(f"[bold]Search Results for '{term}'[/bold]")
        for result in results:
            entry = result.entry
            aliases = ", ".join(a.alias for a in entry.aliases[:3]) if entry.aliases else ""
            console.print(f"  {entry.canonical} ({entry.type}, tier {entry.tier})")
            if aliases:
                console.print(f"    [dim]Aliases: {aliases}[/dim]")

    elif action == "load":
        seeds_path = seeds_dir or Path("seeds")
        if not seeds_path.exists():
            _cli_error("Seeds directory not found", str(seeds_path))
            raise typer.Exit(1)

        total = 0
        for seed_file in sorted(seeds_path.glob("tier_*.json")):
            count = import_from_json(seed_file)
            console.print(f"  Loaded {seed_file.name}: {count} entries")
            total += count

        # Load context profiles
        contexts_dir = seeds_path / "contexts"
        if contexts_dir.exists():
            import json
            from asr.dictionary.models import ContextProfile
            selector = BiasListSelector()
            for ctx_file in contexts_dir.glob("*.json"):
                with open(ctx_file) as f:
                    profile = ContextProfile(**json.load(f))
                selector.save_context_profile(profile)
                console.print(f"  Loaded context: {profile.name}")

        stats = get_stats()
        console.print(f"\n[green]Total:[/green] {stats.total_entries} entries in dictionary")

    elif action == "learn":
        # Batch learning: process session logs to discover proper nouns
        from asr.logging import LOGS_DIR
        from asr.dictionary import _NER_AVAILABLE, discover_proper_nouns, add_discovered_to_pending

        if not _NER_AVAILABLE:
            _cli_error("NER required for learning", "Install with: pip install asr[ner]")
            raise typer.Exit(1)

        if not LOGS_DIR.exists():
            console.print("[dim]No logs found. Run some transcriptions first.[/dim]")
            return

        # Process recent log files
        import json as json_mod
        from asr.models.transcript import Segment

        log_files = sorted(LOGS_DIR.glob("session_*.jsonl"), reverse=True)[:20]  # Last 20 sessions
        console.print(f"[bold]Processing {len(log_files)} session logs for proper nouns...[/bold]")

        total_discovered = 0
        total_added = 0
        total_auto_approved = 0

        for log_file in log_files:
            # Parse log file to extract corrected segments
            segments = []
            original_words: set[str] = set()

            with open(log_file) as f:
                for line in f:
                    try:
                        event = json_mod.loads(line)
                    except json_mod.JSONDecodeError:
                        continue

                    # Extract corrected text from correction_applied events
                    if event.get("event") == "correction_applied":
                        seg_data = event.get("data", {})
                        if "corrected_text" in seg_data:
                            # Create a minimal Segment for NER
                            seg = Segment(
                                id=seg_data.get("segment_id", 0),
                                start=0.0,
                                end=0.0,
                                text=seg_data["corrected_text"],
                                raw_text=seg_data.get("original_text", seg_data["corrected_text"]),
                                confidence=0.9,
                            )
                            segments.append(seg)

                    # Extract original words from chunk_transcribed events
                    if event.get("event") == "chunk_transcribed":
                        words_data = event.get("data", {}).get("words", [])
                        for w in words_data:
                            word = w.get("word", "")
                            if word:
                                original_words.add(word.lower().strip())

            if not segments:
                continue

            # Run discovery (skip session frequency check for batch mode)
            try:
                result = discover_proper_nouns(
                    segments=segments,
                    source_file=log_file.name,
                    min_confidence=0.6,  # Lower threshold for batch
                    context=context or "discovered",
                    require_session_frequency=False,  # Don't require 2+ in single session
                )

                total_discovered += len(result.discovered)

                if result.discovered:
                    added, auto_approved = add_discovered_to_pending(
                        result.discovered,
                        context=context or "discovered",
                    )
                    total_added += added
                    total_auto_approved += auto_approved

                    for noun in result.discovered:
                        console.print(f"  [green]+[/green] {noun.text} ({noun.entity_type})")
                        if noun.snippet:
                            console.print(f"      [dim]\"{noun.snippet[:80]}...\"[/dim]")

            except Exception as e:
                console.print(f"[dim]Error processing {log_file.name}: {e}[/dim]")

        console.print()
        console.print("[bold]Batch Learning Complete[/bold]")
        console.print(f"  Discovered: {total_discovered} proper nouns")
        console.print(f"  Added to pending: {total_added}")
        console.print(f"  Auto-approved: {total_auto_approved}")
        if total_added > 0:
            console.print("\n[dim]Review pending: asr dict pending[/dim]")

    else:
        valid = "stats, pending, approve, reject, search, load, learn"
        _cli_error("Unknown action", f"'{action}' (valid: {valid})")
        raise typer.Exit(1)


@app.command()
def logs(
    sessions: Annotated[
        int,
        typer.Option("-n", "--sessions", help="Number of recent sessions to analyze")
    ] = 10,
    detail: Annotated[
        bool,
        typer.Option("--detail", "-d", help="Show detailed per-session info")
    ] = False,
):
    """Analyze transcription session logs for accuracy insights.

    Shows aggregated statistics across recent sessions including:
    - Average confidence scores
    - Correction acceptance rates
    - Common low-confidence words (candidates for vocabulary)

    Examples:
        asr logs                    # Analyze last 10 sessions
        asr logs -n 50              # Analyze last 50 sessions
        asr logs --detail           # Show per-session details
    """
    from asr.logging import LOGS_DIR
    import json

    if not LOGS_DIR.exists():
        console.print("[dim]No logs found yet. Run a transcription first.[/dim]")
        return

    analysis = analyze_logs(limit=sessions)

    if "error" in analysis:
        console.print(f"[yellow]{analysis['error']}[/yellow]")
        return

    console.print(f"[bold]Log Analysis[/bold] ({analysis['sessions_analyzed']} sessions)")
    console.print()

    # Summary stats
    console.print(f"  [dim]Total words transcribed:[/dim] {analysis['total_words_transcribed']:,}")
    console.print(f"  [dim]Average confidence:[/dim] {analysis['avg_confidence']:.1%}")

    if analysis.get('total_corrections') or analysis.get('total_rejections'):
        console.print(f"  [dim]Corrections applied:[/dim] {analysis['total_corrections']}")
        console.print(f"  [dim]Corrections rejected:[/dim] {analysis['total_rejections']}")
        if analysis.get('acceptance_rate') is not None:
            console.print(f"  [dim]Acceptance rate:[/dim] {analysis['acceptance_rate']:.1f}%")

    # Common low-confidence words
    if analysis.get('common_low_confidence_words'):
        console.print()
        console.print("[bold]Common Low-Confidence Words[/bold] (vocabulary candidates)")
        for word, count in analysis['common_low_confidence_words'][:15]:
            console.print(f"  {count:3}x  {word}")

    # Per-session details if requested
    if detail:
        console.print()
        console.print("[bold]Recent Sessions[/bold]")
        log_files = sorted(LOGS_DIR.glob("session_*.jsonl"), reverse=True)[:sessions]

        for log_file in log_files:
            # Find session_complete event
            for line in log_file.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                    if event.get("event") == "session_complete":
                        console.print(
                            f"  {log_file.stem}: "
                            f"{event.get('total_words', 0)} words, "
                            f"{event.get('avg_confidence', 0):.1%} conf, "
                            f"{event.get('corrections_applied', 0)} corrections"
                        )
                        break
                except json.JSONDecodeError:
                    continue


@app.command()
def evaluate(
    dataset: Annotated[
        str,
        typer.Argument(help="Dataset to evaluate: lovecraft, or path to audio file")
    ],
    reference: Annotated[
        Optional[Path],
        typer.Option("-r", "--reference", help="Path to reference text file")
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output JSON file for results")
    ] = None,
    correct: Annotated[
        bool,
        typer.Option("--correct", help="Apply Claude correction and measure WER improvement")
    ] = False,
    download_only: Annotated[
        bool,
        typer.Option("--download-only", help="Only download dataset, don't run evaluation")
    ] = False,
    no_words: Annotated[
        bool,
        typer.Option("--no-words", help="Skip word-level timestamps (~20% faster)")
    ] = False,
    warm_up: Annotated[
        bool,
        typer.Option("--warm-up", help="Pre-compile Metal kernels before evaluation")
    ] = False,
    stories: Annotated[
        Optional[str],
        typer.Option("--stories", "-s", help="Comma-separated story slugs (e.g., 'dagon')")
    ] = None,
):
    """Evaluate ASR accuracy against reference texts.

    Examples:
        asr evaluate lovecraft                    # Run Lovecraft benchmark
        asr evaluate lovecraft --stories dagon   # Run just Dagon
        asr evaluate lovecraft --download-only   # Just download test files
        asr evaluate lovecraft --correct         # Include correction in WER comparison
        asr evaluate lovecraft --no-words        # Faster evaluation without word timestamps
        asr evaluate audio.mp3 -r reference.txt  # Custom audio with reference
    """
    from asr.eval.metrics import calculate_wer
    from asr.eval.datasets.lovecraft import (
        LOVECRAFT_STORIES,
        LOVECRAFT_VOCABULARY,
        download_lovecraft_dataset,
        get_lovecraft_reference,
    )
    from asr.dictionary import (
        BiasListSelector,
        generate_whisper_prompt,
        init_db,
    )
    from asr.dictionary.db import get_stats

    # Handle built-in datasets
    if dataset.lower() == "lovecraft":
        # Download dataset
        data_dir = Path.home() / ".asr" / "eval" / "lovecraft"
        audio_files = download_lovecraft_dataset(data_dir)

        if download_only:
            console.print(f"[green]Downloaded {len(audio_files)} files to {data_dir}[/green]")
            return

        # Run evaluation on each story
        results = []
        total_ref_words = 0
        total_errors = 0

        # Load config and engine ONCE, outside the loop
        config = load_config()
        word_timestamps = not no_words

        if warm_up:
            console.print("[dim]Warming up Metal kernels...[/dim]")
        engine = get_crisperwhisper_engine(warm_up=warm_up)

        # Initialize dictionary once
        init_db()
        stats = get_stats()

        import time  # Import once, outside loop
        from concurrent.futures import ThreadPoolExecutor
        from asr.logging import TranscriptLogger

        # Build vocab prompt once (same for all stories)
        if stats.total_entries > 0:
            selector = BiasListSelector()
            dict_entries = selector.select_bias_list(context="lovecraft", max_entries=50)
            if dict_entries:
                vocab_prompt = generate_whisper_prompt(dict_entries, context="lovecraft")
                console.print(f"[dim]Using dictionary: {len(dict_entries)} entries[/dim]")
            else:
                vocab_prompt = "Names: " + ", ".join(LOVECRAFT_VOCABULARY[:20])
                dict_entries = []
        else:
            vocab_prompt = "Names: " + ", ".join(LOVECRAFT_VOCABULARY[:20])
            dict_entries = []

        # Parse --stories filter
        story_filter = set(s.strip().lower() for s in stories.split(",")) if stories else None

        # Filter to valid stories upfront
        valid_stories = []
        for story in LOVECRAFT_STORIES:
            # Apply --stories filter
            if story_filter and story.slug.lower() not in story_filter:
                continue

            ext = ".ogg" if story.audio_url.endswith(".ogg") else ".mp3"
            audio_path = data_dir / f"{story.slug}{ext}"
            if audio_path.exists():
                valid_stories.append((story, audio_path))
            else:
                console.print(f"[yellow]Skipping {story.name} (not downloaded)[/yellow]")

        # Pipeline: prepare next file's audio while current file transcribes
        def prepare_story(story_tuple):
            """Prepare audio and fetch reference in background."""
            story, audio_path = story_tuple
            ref_text = get_lovecraft_reference(story.slug, cache_dir=data_dir / "references")
            chunks, duration = prepare_audio(audio_path, config)
            return story, audio_path, ref_text, chunks, duration

        if not valid_stories:
            console.print("[yellow]No valid stories to evaluate[/yellow]")
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            # Start preparing first story
            pending_future = executor.submit(prepare_story, valid_stories[0])

            for i, (story, audio_path) in enumerate(valid_stories):
                console.print(f"\n[bold]Evaluating: {story.name}[/bold]")
                dur = story.duration_minutes
                diff = story.difficulty
                console.print(f"[dim]Duration: {dur:.1f} min | Difficulty: {diff}[/dim]")

                # Get prepared data (blocks if not ready)
                story, audio_path, ref_text, chunks, duration = pending_future.result()

                # Start preparing NEXT story while we transcribe this one
                if i + 1 < len(valid_stories):
                    pending_future = executor.submit(prepare_story, valid_stories[i + 1])

                if not ref_text:
                    console.print("[yellow]  Could not fetch reference text[/yellow]")
                    continue

                # Initialize session logger for this story
                session_logger = TranscriptLogger(audio_path)
                session_logger.set_config({
                    "model": "crisperwhisper",
                    "language": config.language,
                    "use_vad": True,
                    "word_timestamps": word_timestamps,
                    "correct": correct,
                    "domain": "lovecraft",
                })
                session_logger.log_audio_prepared(duration, len(chunks))

                # Transcribe (GPU-bound, can't parallelize)
                console.print(f"  Transcribing {len(chunks)} chunks...")
                start = time.time()

                # Cross-chunk context buffer for entity continuity
                context_buffer = ""
                MAX_CONTEXT_WORDS = 100

                all_segments = []
                HARD_CHUNK_THRESHOLD = 0.4  # Skip word timestamps for difficult audio

                for chunk_idx, chunk in enumerate(chunks):
                    chunk_start = time.time()

                    # Build prompt: vocab + rolling context from previous chunks
                    if context_buffer:
                        # Truncate context to fit within prompt limits
                        ctx_snippet = context_buffer[-400:]  # ~100 words
                        chunk_prompt = f"{vocab_prompt}\n\nPrevious transcript: \"{ctx_snippet}\""
                    else:
                        chunk_prompt = vocab_prompt

                    # Adaptive word timestamps: skip for hard chunks (20-30% faster)
                    chunk_word_timestamps = word_timestamps
                    if word_timestamps and chunk.difficulty_score > HARD_CHUNK_THRESHOLD:
                        chunk_word_timestamps = False
                        console.print(f"    [dim]Chunk {chunk_idx}: hard audio (d={chunk.difficulty_score:.2f}), skipping word timestamps[/dim]")

                    segments = engine.transcribe(
                        audio=chunk,
                        word_timestamps=chunk_word_timestamps,
                        language=config.language,
                        initial_prompt=chunk_prompt,
                    )
                    all_segments.extend(segments)

                    # Update rolling context buffer
                    chunk_text = " ".join(seg.text for seg in segments)
                    context_buffer = " ".join((context_buffer + " " + chunk_text).split()[-MAX_CONTEXT_WORDS:])

                    chunk_time = time.time() - chunk_start

                    # Log chunk timing for analysis
                    word_count = sum(len(seg.text.split()) for seg in segments)
                    avg_conf = sum(seg.confidence for seg in segments) / len(segments) if segments else 0

                    # Get memory usage
                    try:
                        import psutil
                        memory_gb = psutil.Process().memory_info().rss / (1024**3)
                    except Exception:
                        memory_gb = None

                    session_logger.log_chunk_timing(
                        chunk_index=chunk_idx,
                        total_chunks=len(chunks),
                        chunk_duration_seconds=chunk.duration,
                        transcription_seconds=chunk_time,
                        word_count=word_count,
                        avg_confidence=avg_conf,
                        memory_gb=memory_gb,
                    )

                    console.print(f"    Chunk {chunk_idx + 1}/{len(chunks)}: {chunk_time:.1f}s", end="\r")  # noqa: E501
                console.print()  # Clear the \r line

                raw_text = " ".join(seg.text for seg in all_segments)
                transcription_time = time.time() - start

                # Calculate raw WER
                wer_result = calculate_wer(ref_text, raw_text)
                total_ref_words += wer_result.reference_words
                total_errors += (
                    wer_result.substitutions + wer_result.insertions + wer_result.deletions
                )

                rtf_val = duration / transcription_time if transcription_time > 0 else 0
                console.print(f"  Time: {transcription_time:.1f}s | RTF: {rtf_val:.1f}x")
                wer_pct = wer_result.wer * 100
                ref_words = wer_result.reference_words
                console.print(f"  [bold]WER (raw): {wer_pct:.1f}%[/bold] ({ref_words} words)")

                result = {
                    "name": story.name,
                    "slug": story.slug,
                    "duration": duration,
                    "transcription_time": transcription_time,
                    "wer_raw": wer_result.wer,
                    "ref_words": wer_result.reference_words,
                }

                # Apply correction if requested
                if correct:
                    from asr.nlp.corrector import Corrector
                    from asr.nlp.models import CorrectionConfig

                    console.print("  Applying correction...")
                    start = time.time()

                    # Use dictionary entries for correction if available
                    correction_vocab = LOVECRAFT_VOCABULARY
                    if dict_entries:
                        dict_names = [e.canonical for e in dict_entries]
                        correction_vocab = dict_names + LOVECRAFT_VOCABULARY
                        # Dedupe preserving order
                        correction_vocab = list(dict.fromkeys(correction_vocab))

                    correction_config = CorrectionConfig(
                        passes=2,
                        vocabulary=correction_vocab,
                        domain="biography",
                    )

                    with Corrector(provider="claude", config=correction_config) as corrector:
                        corrected_segments = corrector.correct(
                            merge_segments(all_segments),
                            vocabulary=correction_vocab,
                            domain="biography",
                            dictionary_context="lovecraft",  # Pass dictionary context
                        )

                    corrected_text = " ".join(seg.text for seg in corrected_segments)
                    correction_time = time.time() - start

                    wer_corrected = calculate_wer(ref_text, corrected_text)
                    improvement = (wer_result.wer - wer_corrected.wer) * 100
                    wer_corr_pct = wer_corrected.wer * 100

                    console.print(
                        f"  [bold]WER (corrected): {wer_corr_pct:.1f}%[/bold] "
                        f"({improvement:+.1f}% improvement)"
                    )
                    result["wer_corrected"] = wer_corrected.wer
                    result["correction_time"] = correction_time
                    result["wer_improvement"] = improvement

                results.append(result)

                # Finalize session logger for this story
                session_logger.metrics.total_words = sum(len(seg.text.split()) for seg in all_segments)
                session_logger.metrics.total_segments = len(all_segments)
                session_logger.finalize()

        # Summary
        if results:
            avg_wer = total_errors / total_ref_words if total_ref_words > 0 else 0
            console.print("\n[bold]Summary[/bold]")
            console.print(f"  Stories evaluated: {len(results)}")
            console.print(f"  Total words: {total_ref_words}")
            console.print(f"  [bold]Overall WER: {avg_wer*100:.1f}%[/bold]")

            if output:
                import json
                json_content = json.dumps(results, indent=2)
                if not _safe_write_file(output, json_content, "evaluation results"):
                    raise typer.Exit(1)
                console.print(f"  Results saved: {output}")

    else:
        # Custom audio file evaluation
        audio_path = Path(dataset)
        if not audio_path.exists():
            _cli_error("File not found", str(audio_path))
            raise typer.Exit(1)

        if not reference:
            _cli_error("Reference text required", "Use -r/--reference to provide ground truth text")
            raise typer.Exit(1)

        ref_text = reference.read_text(encoding="utf-8")
        console.print(f"[bold]Evaluating: {audio_path.name}[/bold]")

        # Transcribe
        config = load_config()
        engine = get_crisperwhisper_engine()
        chunks, duration = prepare_audio(audio_path, config)

        all_segments = []
        for chunk in chunks:
            segments = engine.transcribe(
                audio=chunk,
                word_timestamps=True,
                language=config.language,
            )
            all_segments.extend(segments)

        raw_text = " ".join(seg.text for seg in merge_segments(all_segments))

        wer_result = calculate_wer(ref_text, raw_text)
        console.print(f"  Duration: {duration:.1f}s")
        console.print(f"  [bold]WER: {wer_result.wer*100:.1f}%[/bold]")
        console.print(f"  Reference words: {wer_result.reference_words}")
        console.print(f"  Hypothesis words: {wer_result.hypothesis_words}")
        console.print(f"  Substitutions: {wer_result.substitutions}")
        console.print(f"  Insertions: {wer_result.insertions}")
        console.print(f"  Deletions: {wer_result.deletions}")


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show version")
    ] = False,
):
    """ASR - Local Whisper transcription for Apple Silicon."""
    if version:
        console.print(f"asr {__version__}")
        raise typer.Exit(0)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()

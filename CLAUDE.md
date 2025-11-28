# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e .
brew install ffmpeg  # Required system dependency

# Download CrisperWhisper model (first-time setup)
pip install huggingface_hub[hf_xet]
huggingface-cli download --local-dir ~/.asr/models/crisperwhisper kyr0/crisperwhisper-unsloth-mlx

# Environment setup for correction pipeline
# Create .env file in project root:
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# Run transcription
asr transcribe file.m4a                    # Transcribe with CrisperWhisper
asr transcribe file.m4a --prompt "Names: Ron Chernow, MacBook Air"
asr transcribe file.m4a --profile batch    # Max accuracy for offline jobs
asr transcribe file.m4a --profile interactive  # Stay responsive while transcribing

# Output formats: .json, .srt, .vtt, .md, .txt
asr transcribe file.m4a -o output.vtt

# Post-processing correction (requires ANTHROPIC_API_KEY)
asr transcribe file.m4a --correct --domain biography      # With domain-specific prompts
asr transcribe file.m4a --correct --quality max           # Opus final pass
asr transcribe file.m4a --correct --learn --domain tech   # Learn vocabulary

# Audio enhancement options
asr transcribe file.m4a --no-enhance       # Disable audio pre-processing
asr transcribe file.m4a --noise moderate   # Enable noise reduction

# Vocabulary management
asr vocab list                             # List all domain vocabularies
asr vocab add -d biography -t "Ron Chernow, MacBook Air"
asr vocab show -d biography                # Show terms
asr vocab pending                          # Review pending learned terms
asr vocab approve --auto                   # Auto-approve high-confidence terms

# Other commands
asr config --show     # View configuration
asr config --reset    # Reset to defaults
asr clear-cache       # Free disk space

# Post-process existing transcripts
asr correct-transcript transcript.json --domain biography

# Benchmark performance
asr bench recording.m4a -n 3    # Run 3 iterations, measure RTF

# Lint
ruff check src/
```

## Architecture

This is a local ASR (Automatic Speech Recognition) tool using CrisperWhisper via MLX on Apple Silicon.

**Model**: CrisperWhisper (kyr0/crisperwhisper-unsloth-mlx) - a Whisper Large v3 fine-tuned for word-by-word precision. Optimized for verbatim transcription rather than grammatically polished output.

**Hardware**: Tuned for MacBook Air M4 with 24GB unified memory.

**Data flow:** Audio file → ffmpeg enhancement → Silero VAD chunking → CrisperWhisper transcription → optional Claude correction → formatted output

### Key Modules

| Module | Purpose |
|--------|---------|
| `cli.py` | Typer CLI, orchestrates the pipeline |
| `config.py` | TOML config, vocabulary management, profile settings |
| `audio/ingest.py` | ffmpeg wrapper with audio enhancement, Silero VAD, chunking |
| `backends/crisperwhisper_engine.py` | CrisperWhisper MLX integration |
| `nlp/corrector.py` | Two-pass Claude correction with safety constraints |
| `nlp/prompts.py` | Domain-specific prompts with safety rules, confidence marking |
| `nlp/models.py` | CorrectionConfig, CorrectionResult Pydantic models |
| `output/formatters.py` | JSON, SRT, VTT, Markdown, plain text formatters |
| `models/transcript.py` | Pydantic v2 models defining the JSON schema |
| `logging.py` | Session logging for accuracy analysis |
| `dictionary/` | Proper noun dictionary with context-aware bias lists |

### Configuration

CLI args override `~/.asr/config.toml` defaults. Model is always CrisperWhisper - configuration controls profiles, VAD, and audio enhancement only.

### Environment Variables

The correction pipeline requires an Anthropic API key. Create a `.env` file in the project root:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

The `.env` file is automatically loaded when the package is imported (via `python-dotenv` in `src/asr/__init__.py`).

## Audio Enhancement

Audio pre-processing improves ASR accuracy by normalizing input quality. Enabled by default.

### Filters Applied

| Filter | Purpose | Default |
|--------|---------|---------|
| High-pass (80Hz) | Remove low-frequency rumble | ON |
| Loudness norm (-16 LUFS) | Consistent input levels | ON |
| Noise reduction | Reduce background noise | OFF |

### Options

| Option | Description |
|--------|-------------|
| `--no-enhance` | Disable all audio pre-processing |
| `--noise light` | Light noise reduction |
| `--noise moderate` | Moderate noise reduction (may affect clean audio) |

Configure defaults in `~/.asr/config.toml`:
```toml
[audio_enhancement]
loudness_enabled = true
loudness_target_lufs = -16
highpass_enabled = true
highpass_freq = 80
noise_reduction = "off"
```

## VAD (Voice Activity Detection) & Chunking

The VAD system detects speech segments and splits audio into chunks for transcription. Tuned parameters improve accuracy by preventing mid-word cuts.

### VAD Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `threshold` | 0.35 | VAD confidence threshold (lower = more sensitive) |
| `min_speech_duration` | 0.25s | Minimum speech segment to keep |
| `min_silence_duration` | 0.3s | Minimum silence to split on |
| `boundary_pad` | 0.2s | Padding added to segment boundaries |
| `inter_segment_overlap` | 0.25s | Overlap between consecutive segments |

### How It Works

1. **Silero VAD** detects speech vs. silence
2. **Boundary padding** (200ms) catches word onsets/offsets
3. **Inter-segment overlap** (250ms) prevents chopped words at VAD boundaries
4. **Long segment splitting** (>60s) uses 2s overlap for context continuity

Configure in `~/.asr/config.toml`:
```toml
[vad]
threshold = 0.35
min_speech_duration = 0.25
min_silence_duration = 0.3
boundary_pad = 0.2
inter_segment_overlap = 0.25
```

## Resource Profiles

Profiles optimize system resource usage for different use cases on M4 Air.

```bash
asr transcribe file.m4a --profile interactive  # Low resources, responsive system
asr transcribe file.m4a --profile balanced     # Default: good accuracy
asr transcribe file.m4a --profile batch        # Max accuracy, uses more resources
```

| Profile | Word Timestamps | Use Case |
|---------|-----------------|----------|
| `interactive` | Off | Quick notes while multitasking |
| `balanced` | On | Default: good accuracy |
| `batch` | On | Overnight/offline processing |

### M4 Air Memory Tuning

The CrisperWhisper engine is configured for optimal M4 performance:
- **MLX memory limit**: 21GB (leaves 3GB for OS + apps)
- **MLX cache limit**: 3GB for intermediate tensors
- **Chunk size**: 60 seconds max (fits comfortably in memory)

**Note:** CrisperWhisper is FP16 only. The quantization parameter is accepted for future compatibility but currently has no effect (runtime quantization is not supported by mlx_whisper).

## Correction Pipeline

Two-pass Claude-powered ASR error correction with safety constraints to prevent hallucinations.

### Safety Features

1. **Confidence-Guided Correction**: Low-confidence words are marked with `<low_conf>` tags. Claude can only rewrite text inside these tags.

2. **Diff Gating**: After correction, validates that high-confidence text wasn't changed. Rejects corrections that modify protected words.

3. **Domain-Specific Prompts**: Each domain has strict "don't guess" rules:
   - `medical`: Never guess drug names or dosages
   - `legal`: Preserve exact legal wording
   - `technical`: Don't change code/identifiers
   - `biography`: Never replace unknown names

4. **Cross-Segment Constraint**: "Edit each segment independently. Do not reorder or merge sentences."

### How It Works

| Pass | Model | Purpose |
|------|-------|---------|
| Pass 1 | Sonnet 4.5 | Fix errors inside `<low_conf>` regions only |
| Pass 2 | Sonnet 4.5 (or Opus 4.5) | Ensure entity consistency |

### Options

| Option | Description |
|--------|-------------|
| `--correct` | Enable correction pipeline |
| `--domain <name>` | Domain context: biography, technical, medical, legal, conversational |
| `--quality max` | Use Opus 4.5 for final pass |
| `--passes 1` | Single pass only (faster, cheaper) |
| `--learn` | Learn vocabulary from corrections (adds to pending) |

### Cost Estimate

| Quality | Model | Cost per 20-min transcript |
|---------|-------|---------------------------|
| standard | Sonnet 4.5 (both passes) | ~$0.07 |
| max | Sonnet + Opus final | ~$0.20 |

## Vocabulary Management

Vocabularies improve accuracy for domain-specific terms and proper nouns.

### Storage

Vocabularies are stored in `~/.asr/vocabularies/{domain}.txt` as simple text files (one term per line).

### Commands

```bash
asr vocab list                          # List all domains
asr vocab show -d biography             # Show terms in domain
asr vocab add -d biography -t "Ron Chernow, MacBook Air"
asr vocab add -d tech -f terms.txt      # Load from file
asr vocab remove -d old_domain          # Remove domain vocabulary
```

### Vocabulary Learning (Gated)

Vocabulary learning prevents hallucinations through a pending queue:

1. **Extraction**: `--learn` flag extracts proper nouns from corrections
2. **Safety Gating**: Terms must appear in original Whisper output (not invented by Claude)
3. **Manual Review**: New terms go to pending queue, not directly to vocabulary
4. **Approval**: Review with `asr vocab pending`, approve with `approve`

```bash
# Transcribe with learning enabled
asr transcribe interview.m4a --correct --domain biography --learn

# Review pending terms
asr vocab pending

# Approve specific terms
asr vocab approve -t "Ron Chernow"

# Auto-approve terms with 3+ occurrences (high confidence)
asr vocab approve --auto

# Reject hallucinated terms
asr vocab reject -t "wrong term"
```

### How Vocabulary Is Used

1. **Whisper Prompt**: Loaded as `initial_prompt` hints for transcription
2. **Correction Prompt**: Passed as trusted spellings to Claude
3. **Not Forced**: Vocabulary biases corrections but doesn't force replacements on ambiguous audio

## Dictionary System (Advanced)

The dictionary system provides context-aware bias lists for improving proper noun recognition. It's more powerful than simple vocabularies - entries have types, tiers, boost weights, aliases, and pronunciations.

### Storage

- **Database**: `~/.asr/dictionaries/dictionary.db` (SQLite with WAL mode, 64MB cache)
- **Context profiles**: `~/.asr/dictionaries/contexts/{name}.json`

### Tier System

| Tier | Boost | Use Case |
|------|-------|----------|
| A | 3.0 | Always-on core (personal names, daily tech) |
| B | 2.5 | Domain-specific high priority (running, work) |
| C-D | 2.0-2.5 | Domain-specific standard |
| E-H | 1.0-2.0 | Lower priority, situational |

### Usage

```bash
# Transcribe with dictionary context (selects entries by context + tier)
asr transcribe file.m4a --context running

# Context is auto-detected from audio content (disable with --no-auto-context)
asr transcribe file.m4a --no-auto-context

# Combined with correction (dictionary entries passed to Claude)
asr transcribe file.m4a --context work --correct --domain technical
```

### Loading Seed Data

```bash
# Load all tier seed files
python scripts/load_seeds.py

# Test the full dictionary flow
python scripts/test_dictionary.py
```

Seed files are in `seeds/tier_*.json`. Context profiles in `seeds/contexts/`.

### Entry Format

```json
{
  "canonical": "Ron Chernow",
  "display": "Ron Chernow",
  "type": "person",
  "tier": "B",
  "boost_weight": 2.5,
  "aliases": ["Chernow"],
  "contexts": ["biography"],
  "pronunciations": [{"ipa": "/rɒn ˈtʃɜːrnoʊ/"}]
}
```

### How It Works

1. **BiasListSelector** scores entries: `score = boost_weight × tier_weight × recency × context_match`
2. **Top entries** (default 150) become Whisper `initial_prompt`
3. **Correction block** (top 100) passed to Claude as known spellings
4. **CandidateMatcher** uses RapidFuzz + Double Metaphone for fuzzy matching

### Optional Dependencies

```bash
# NER for auto-discovering proper nouns (GLiNER + spaCy with Apple Silicon optimization)
pip install -e ".[ner]"

# Grapheme-to-phoneme for generating pronunciations
pip install -e ".[g2p]"
```

When NER is installed, you can discover proper nouns from transcripts:
```python
from asr.dictionary import extract_proper_nouns, _NER_AVAILABLE
if _NER_AVAILABLE:
    entities = extract_proper_nouns("Ron Chernow wrote about Hamilton")
```

## Safety Design Principles

This tool implements multiple safeguards against ASR correction hallucinations:

1. **Constrained Editing**: Claude can only modify low-confidence regions
2. **Diff Validation**: High-confidence text is protected
3. **Domain Constraints**: Domain-specific "don't guess" rules
4. **Vocabulary Gating**: Learned terms must appear in original ASR output
5. **Manual Approval**: New vocabulary requires review before use
6. **Configurable Aggressiveness**: Conservative/moderate/aggressive correction levels

## Internal Architecture

### DRY Helper Functions

The codebase uses internal helpers to reduce code duplication:

| Module | Helper | Purpose |
|--------|--------|---------|
| `config.py` | `validate_domain_name()` | Validates domain names for filesystem safety (prevents path traversal) |
| `config.py` | `_file_lock()` | Thread-safe file locking context manager for vocabulary operations |
| `ingest.py` | `_run_subprocess()` | Consistent subprocess execution with timeout and error handling |
| `cli.py` | `_cli_error()` | Formatted error output to console |
| `cli.py` | `_parse_vocabulary_terms()` | Parses comma-separated vocabulary terms |
| `cli.py` | `_safe_write_file()` | File writing with error handling |
| `formatters.py` | `_parse_time_components()` | Shared timestamp parsing logic |

### Security Hardening

- **File permissions**: Vocabulary files, cache directories, and audio chunks use restrictive permissions (0o600 for files, 0o700 for directories)
- **Domain validation**: Domain names are validated before filesystem operations to prevent path traversal
- **Parameter validation**: Numeric parameters (loudness, highpass) are range-checked before use
- **API key protection**: Error messages don't expose API key details

### Performance Optimizations

- **Log buffering**: Session logs are buffered (10 events) before disk writes to reduce I/O
- **Dict lookup**: Pass 2 correction uses O(1) dict lookup instead of O(n²) iteration
- **VAD caching**: Silero VAD model is cached globally to avoid reload overhead
- **Metal GPU lock**: Thread-safe serialization of MLX operations

## Session Logging

All transcription sessions are logged to `~/.asr/logs/` in JSON-lines format for accuracy analysis and debugging.

### What's Logged

| Event | Data |
|-------|------|
| `session_start` | Audio file, timestamp |
| `config` | Model, enhancement settings |
| `audio_prepared` | Duration, chunk count, enhancement filters |
| `chunk_transcribed` | Word count, confidence stats, low-confidence words |
| `correction_applied` | Segment ID, changes made, corrected text |
| `correction_rejected` | Segment ID, reason (diff_gating), violations |
| `pass_complete` | Pass number, model used, segments modified |
| `session_complete` | Summary metrics |

### Commands

```bash
asr logs                    # Analyze last 10 sessions
asr logs -n 50              # Analyze last 50 sessions
asr logs --detail           # Show per-session details
```

### Analysis Output

The `logs` command shows:
- **Average confidence**: Overall word-level confidence scores
- **Correction acceptance rate**: How many corrections passed diff gating
- **Common low-confidence words**: Vocabulary candidates (words Whisper frequently struggles with)

## First-Time Setup

```bash
# 1. Install dependencies
uv venv && source .venv/bin/activate
uv pip install -e .
brew install ffmpeg

# 2. Download CrisperWhisper model (~3GB) to user data directory
pip install huggingface_hub[hf_xet]
huggingface-cli download --local-dir ~/.asr/models/crisperwhisper kyr0/crisperwhisper-unsloth-mlx

# 3. (Optional) Set up Claude correction
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# 4. Test transcription
asr transcribe tests/fixtures/sample_short.m4a
```

### Model Location

The CrisperWhisper model is stored in `~/.asr/models/crisperwhisper` (user data directory). This keeps the 3GB model out of the project directory. The engine also checks `./models/crisperwhisper` as a fallback for local development.

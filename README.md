# ASR

Local CrisperWhisper transcription optimized for Apple Silicon M4.

CrisperWhisper is a Whisper Large v3 fine-tuned for word-by-word precision, ideal for transcription where exact words matter (interviews, dictation, voice automation).

## Installation

```bash
# Create virtual environment
uv venv && source .venv/bin/activate
uv pip install -e .

# Install system dependencies
brew install ffmpeg

# Download CrisperWhisper model (~3GB, one-time)
pip install huggingface_hub[hf_xet]
huggingface-cli download --local-dir ~/Code/audio/models/crisperwhisper kyr0/crisperwhisper-unsloth-mlx

# (Optional) Set up Claude correction
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

## Quick Start

```bash
# Basic transcription
asr transcribe recording.m4a

# With resource profile
asr transcribe recording.m4a --profile batch         # Max accuracy
asr transcribe recording.m4a --profile interactive   # Minimal resources

# With vocabulary hints
asr transcribe recording.m4a --prompt "Ron Chernow, MacBook Air"

# With Claude correction
asr transcribe recording.m4a --correct --domain biography
```

## Output Formats

```bash
asr transcribe file.m4a -o transcript.json   # Detailed JSON with metadata
asr transcribe file.m4a -o transcript.srt    # SRT subtitles
asr transcribe file.m4a -o transcript.vtt    # WebVTT subtitles
asr transcribe file.m4a -o transcript.md     # Markdown with timestamps
asr transcribe file.m4a -o transcript.txt    # Plain text
```

## Commands

| Command | Description |
|---------|-------------|
| `asr transcribe` | Transcribe audio files |
| `asr correct-transcript` | Post-process existing JSON transcripts |
| `asr vocab` | Manage domain vocabularies |
| `asr config` | View/reset configuration |
| `asr logs` | Analyze transcription sessions |
| `asr bench` | Benchmark performance |
| `asr clear-cache` | Free disk space |

## Configuration

Config stored at `~/.asr/config.toml`. View with:

```bash
asr config --show
```

## Documentation

See [CLAUDE.md](./CLAUDE.md) for complete documentation including:
- Audio enhancement options
- VAD configuration
- Correction pipeline details
- Vocabulary management
- Session logging

## License

Tool: MIT | Model: CC-BY-NC-4.0 (non-commercial)

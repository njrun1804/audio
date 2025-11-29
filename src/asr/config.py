"""Configuration management for ASR."""

import fcntl
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import toml
from pydantic import BaseModel, Field, ValidationError

CONFIG_DIR = Path.home() / ".asr"
CONFIG_FILE = CONFIG_DIR / "config.toml"
VOCAB_DIR = CONFIG_DIR / "vocabularies"
LOCK_DIR = CONFIG_DIR / "locks"


# =============================================================================
# Security: Domain Name Validation
# =============================================================================

def validate_domain_name(domain: str) -> bool:
    """Validate domain name is safe for filesystem use.

    Prevents path traversal attacks via domain names like "../etc/passwd".
    """
    if not domain:
        return False
    # Only allow alphanumeric, underscore, and hyphen
    if not all(c.isalnum() or c in "_-" for c in domain):
        return False
    # Reasonable length limit
    if len(domain) > 64:
        return False
    # Reserved names
    if domain in (".", "..", "pending_vocabulary", "config"):
        return False
    return True


# =============================================================================
# File Locking Context Manager (DRY refactor)
# =============================================================================

@contextmanager
def _file_lock(name: str):
    """Context manager for file-based locking.

    Usage:
        with _file_lock("vocab_biography"):
            # ... critical section ...
    """
    # Validate lock name to prevent path traversal
    if not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError(f"Invalid lock name: {name}")

    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = LOCK_DIR / f"{name}.lock"

    # Create lock file with restricted permissions
    fd = os.open(str(lock_file), os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        lock_fh = os.fdopen(fd, 'w', encoding='utf-8')
    except Exception:
        os.close(fd)
        raise

    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        yield lock_fh
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_fh.close()


class ProfileConfig(BaseModel):
    """System resource profile for different use cases."""

    max_parallel_chunks: int
    word_timestamps: bool
    description: str


# Resource profiles for M4 Air (24GB unified memory)
# Model is always CrisperWhisper FP16 with greedy decoding
PROFILES: dict[str, ProfileConfig] = {
    "interactive": ProfileConfig(
        max_parallel_chunks=1,
        word_timestamps=False,  # Faster without word-level alignment
        description="Low resource usage, keeps system responsive",
    ),
    "balanced": ProfileConfig(
        max_parallel_chunks=1,
        word_timestamps=True,
        description="Default: good accuracy with reasonable resources",
    ),
    "batch": ProfileConfig(
        max_parallel_chunks=2,  # Slight benefit for CPU post-processing overlap
        word_timestamps=True,
        description="Max accuracy for offline/overnight processing",
    ),
}


class HardwareConfig(BaseModel):
    """Hardware-specific configuration for M4 optimization."""

    # MLX memory limits (bytes) - tuned for M4 Air 24GB
    mlx_memory_limit: int = 21 * 1024**3  # 21GB for MLX (allows headroom for OS lazy mapping)
    mlx_cache_limit: int = 3 * 1024**3    # 3GB tensor cache (sufficient, reduces fragmentation)

    # CPU thread limits for parallel I/O
    cpu_threads: int = 4


class AudioEnhancementConfig(BaseModel):
    """Audio pre-processing settings for improved ASR accuracy."""

    # Loudness normalization (LUFS targeting)
    loudness_enabled: bool = True
    loudness_target_lufs: int = -16  # Broadcast standard, good for speech

    # High-pass filter (removes low-frequency rumble)
    highpass_enabled: bool = True
    highpass_freq: int = 80  # Hz, safe for all speech including male voices

    # Noise reduction (optional, can degrade clean audio)
    noise_reduction: Literal["off", "light", "moderate"] = "off"


class VADConfig(BaseModel):
    """Voice Activity Detection settings for better segmentation.

    TUNED FOR THROUGHPUT: These defaults prioritize fewer, larger chunks.
    CrisperWhisper handles 60s chunks well - excessive splitting kills performance
    due to MLX GPU setup overhead per chunk.

    For 15-min audio: expect ~15-25 chunks (not 100+).
    """

    # VAD confidence threshold (higher = less sensitive, fewer false splits)
    # 0.45 ignores brief noises while catching real speech
    threshold: float = 0.45  # Was 0.35 - too sensitive

    # Minimum speech duration to keep (filters noise spikes)
    min_speech_duration: float = 0.25  # 250ms

    # Minimum silence duration to split on (higher = more merging)
    # 500ms requires a real pause, not just breathing
    min_silence_duration: float = 0.5  # Was 0.3 - split too eagerly

    # Padding added to each segment boundary (catches word onsets/offsets)
    boundary_pad: float = 0.15  # Was 0.2 - reduced overhead

    # Overlap between consecutive VAD segments (prevents chopped words)
    inter_segment_overlap: float = 0.2  # Was 0.25 - reduced overhead

    # Post-VAD merging: merge small chunks up to this target
    # Reduces chunk count without exceeding max_chunk_seconds
    min_chunk_duration: float = 20.0  # Merge chunks under 20s with neighbors

    # Soft cap for merge policy: prefer to stay under this duration
    # Provides headroom before hitting hard cap (max_chunk_seconds)
    soft_cap_seconds: float = 15.0  # Prefer chunks under 15s

    # Gap limit: don't merge across silences longer than this (topic boundaries)
    max_gap_seconds: float = 2.0  # Silence > 2s suggests topic change


class CorrectionConfig(BaseModel):
    """ASR error correction settings (Sonnet 4.5 with thinking)."""

    enabled: bool = False
    passes: Literal[1, 2] = 2
    domain: str | None = None
    vocabulary: list[str] = Field(default_factory=list)
    # Max dictionary entries to include in correction prompt
    dictionary_entries_for_correction: int = 100


class ASRConfig(BaseModel):
    """Main configuration model.

    Note: Model is always CrisperWhisper (kyr0/crisperwhisper-unsloth-mlx).
    No model/quantization/backend selection - single-model architecture.
    """

    language: str = "en"
    word_timestamps: bool = True  # CrisperWhisper excels at word-level precision
    max_chunk_seconds: int = 60  # Optimized for 24GB
    chunk_overlap_seconds: float = 2.0
    use_vad: bool = True
    prompt: str | None = None  # Context prompt: names, vocabulary, domain
    cache_dir: Path = Field(default_factory=lambda: CONFIG_DIR / "cache")
    models_dir: Path = Field(default_factory=lambda: CONFIG_DIR / "models")
    correction: CorrectionConfig = Field(default_factory=CorrectionConfig)
    audio_enhancement: AudioEnhancementConfig = Field(default_factory=AudioEnhancementConfig)
    vad: VADConfig = Field(default_factory=VADConfig)

    # Dictionary system settings
    dictionary_context: str | None = None  # Active dictionary context (e.g., "biography", "tech")
    dictionary_auto_detect: bool = True  # Auto-detect context from audio/domain

    # Parallel processing settings
    # NOTE: MLX/Metal is not thread-safe for concurrent inference. With the required
    # GPU lock, parallel chunks just create waiting threads. Default to 1 for efficiency.
    # Higher values (2-3) only help if doing CPU-heavy post-processing between chunks.
    max_parallel_chunks: int = 1

    class Config:
        arbitrary_types_allowed = True


def ensure_config_dirs(config: ASRConfig) -> None:
    """Create config directories if they don't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> ASRConfig:
    """Load configuration from file, or create defaults."""
    if CONFIG_FILE.exists():
        try:
            data = toml.load(CONFIG_FILE)
            # Handle nested correction config
            if "correction" in data and isinstance(data["correction"], dict):
                data["correction"] = CorrectionConfig(**data["correction"])
            # Handle nested audio enhancement config
            if "audio_enhancement" in data and isinstance(data["audio_enhancement"], dict):
                data["audio_enhancement"] = AudioEnhancementConfig(**data["audio_enhancement"])
            # Handle nested VAD config
            if "vad" in data and isinstance(data["vad"], dict):
                data["vad"] = VADConfig(**data["vad"])
            # Convert path strings back to Path objects
            for key in ["cache_dir", "models_dir"]:
                if key in data and isinstance(data[key], str):
                    data[key] = Path(data[key]).expanduser()
                    # SECURITY: Validate paths are within expected directories
                    # Only allow paths under CONFIG_DIR or absolute paths the user owns
                    path_obj = data[key]
                    if not path_obj.is_absolute():
                        raise ValueError(f"Config path must be absolute: {key}={data[key]}")
            config = ASRConfig(**data)
        except (toml.TomlDecodeError, ValueError, TypeError) as e:
            print(f"Warning: Failed to load config ({e}), using defaults", file=sys.stderr)
            config = ASRConfig()
        except ValidationError as e:
            print(f"Warning: Config validation failed ({e}), using defaults", file=sys.stderr)
            config = ASRConfig()
        except Exception as e:
            err_type = type(e).__name__
            print(
                f"Warning: Unexpected error loading config ({err_type}: {e}), using defaults",
                file=sys.stderr,
            )
            config = ASRConfig()
    else:
        config = ASRConfig()
        save_config(config)

    ensure_config_dirs(config)
    return config


def save_config(config: ASRConfig) -> None:
    """Save configuration to file with atomic write."""
    import tempfile

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = config.model_dump()
    # Convert Path objects to strings for TOML
    data["cache_dir"] = str(config.cache_dir)
    data["models_dir"] = str(config.models_dir)
    # Convert nested models
    data["correction"] = config.correction.model_dump()
    data["audio_enhancement"] = config.audio_enhancement.model_dump()
    data["vad"] = config.vad.model_dump()

    # Write to temporary file first (atomic operation)
    temp_fd, temp_path = tempfile.mkstemp(dir=CONFIG_DIR, prefix=".config_", suffix=".toml.tmp")
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            toml.dump(data, f)

        # Set restrictive permissions before moving
        os.chmod(temp_path, 0o600)

        # Atomic rename
        os.replace(temp_path, CONFIG_FILE)
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        raise


def apply_profile(config: ASRConfig, profile: str) -> tuple[ASRConfig, bool]:
    """Apply a resource profile's settings to the config.

    Profiles control system resource usage for different use cases
    like interactive work vs batch processing.

    Note: Model is always CrisperWhisper FP16 with greedy decoding.

    Returns:
        Tuple of (updated config, word_timestamps)
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Available: {list(PROFILES.keys())}")

    profile_config = PROFILES[profile]
    updated = config.model_copy(
        update={
            "max_parallel_chunks": profile_config.max_parallel_chunks,
        }
    )
    return updated, profile_config.word_timestamps


# ============================================================================
# Vocabulary Management
# ============================================================================

def load_domain_vocabulary(domain: str | None) -> list[str]:
    """Load vocabulary terms for a specific domain.

    Vocabularies are stored as simple text files, one term per line.
    Lines starting with # are treated as comments.
    """
    if not domain:
        return []

    # SECURITY: Validate domain name to prevent path traversal
    if not validate_domain_name(domain):
        return []  # Return empty for invalid domains (don't raise to avoid DoS)

    vocab_file = VOCAB_DIR / f"{domain}.txt"
    if not vocab_file.exists():
        return []

    # Use file locking to prevent reading during concurrent writes
    with _file_lock(f"vocab_{domain}"):
        terms = []
        for line in vocab_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
        return terms


def save_domain_vocabulary(domain: str, terms: list[str], append: bool = True) -> int:
    """Save vocabulary terms to a domain file.

    Args:
        domain: Domain name (creates file {domain}.txt)
        terms: List of terms to save
        append: If True, add to existing; if False, replace

    Returns:
        Number of new unique terms added
    """
    # SECURITY: Validate domain name
    if not validate_domain_name(domain):
        raise ValueError(f"Invalid domain name: {domain}")

    VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    vocab_file = VOCAB_DIR / f"{domain}.txt"

    # Use file locking context manager (DRY refactor)
    with _file_lock(f"vocab_{domain}"):
        # Load existing terms if appending
        existing = set(load_domain_vocabulary(domain)) if append else set()
        new_terms = [t for t in terms if t not in existing]

        # Combine and sort
        combined = sorted(existing | set(terms))

        # Write with header comment (timestamp is current time)
        current_time = datetime.now().isoformat()
        lines = [
            f"# Vocabulary for domain: {domain}",
            f"# Updated: {current_time}",
            "",
        ]
        lines.extend(combined)
        vocab_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # SECURITY: Set restrictive file permissions
        vocab_file.chmod(0o600)

        return len(new_terms)


def list_domains() -> list[str]:
    """List all available vocabulary domains."""
    if not VOCAB_DIR.exists():
        return []
    return sorted([f.stem for f in VOCAB_DIR.glob("*.txt")])


def remove_domain_vocabulary(domain: str) -> bool:
    """Remove a domain vocabulary file."""
    # SECURITY: Validate domain name to prevent path traversal
    if not validate_domain_name(domain):
        return False

    vocab_file = VOCAB_DIR / f"{domain}.txt"
    if vocab_file.exists():
        vocab_file.unlink()
        return True
    return False


def validate_vocabulary_term(term: str) -> bool:
    """Check if a term is valid for vocabulary learning.

    Rejects:
    - Empty/very short terms
    - Purely numeric
    - Mostly punctuation
    """
    if len(term) < 2 or len(term) > 100:
        return False
    if term.isdigit():
        return False
    if not any(c.isalpha() for c in term):
        return False
    # Check alpha ratio
    alpha_count = sum(1 for c in term if c.isalpha())
    if alpha_count / len(term) < 0.5:
        return False
    return True


# ============================================================================
# Gated Vocabulary Learning
# ============================================================================

PENDING_VOCAB_FILE = CONFIG_DIR / "pending_vocabulary.json"


def load_pending_vocabulary() -> dict[str, dict]:
    """Load pending vocabulary terms awaiting approval.

    Returns dict of {term: {"domain": str, "occurrences": int, "sources": [str]}}
    """
    import json

    if not PENDING_VOCAB_FILE.exists():
        return {}

    try:
        return json.loads(PENDING_VOCAB_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {}


def save_pending_vocabulary(pending: dict[str, dict]) -> None:
    """Save pending vocabulary terms."""
    import json

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_VOCAB_FILE.write_text(json.dumps(pending, indent=2), encoding="utf-8")
    # SECURITY: Set restrictive file permissions
    PENDING_VOCAB_FILE.chmod(0o600)


def add_pending_terms(
    terms: list[str],
    domain: str,
    source: str,
    original_words: set[str] | None = None,
) -> int:
    """Add terms to pending vocabulary with safety gating.

    SAFETY: Only accepts terms that appear in original_words (if provided).
    This prevents learning hallucinated terms from Claude.

    Args:
        terms: Candidate terms to learn
        domain: Domain to associate with
        source: Source file identifier
        original_words: Set of words from original Whisper output (for gating)

    Returns:
        Number of new terms added to pending
    """
    # SECURITY: Validate domain name
    if not validate_domain_name(domain):
        raise ValueError(f"Invalid domain name: {domain}")

    # Use file locking context manager (DRY refactor)
    with _file_lock("pending-vocabulary"):
        pending = load_pending_vocabulary()
        added = 0

        for term in terms:
            # Skip invalid terms
            if not validate_vocabulary_term(term):
                continue

            # CRITICAL SAFETY: If original_words provided, verify term appeared in ASR output
            if original_words is not None:
                # Check if term (or its words) appeared in original
                term_words = set(w.lower() for w in term.split())
                original_lower = set(w.lower() for w in original_words)

                # At least one word from the term must appear in original
                if not term_words & original_lower:
                    continue  # Skip - this was likely hallucinated

            # Add or update pending entry
            if term in pending:
                pending[term]["occurrences"] += 1
                if source not in pending[term]["sources"]:
                    pending[term]["sources"].append(source)
            else:
                pending[term] = {
                    "domain": domain,
                    "occurrences": 1,
                    "sources": [source],
                }
                added += 1

        save_pending_vocabulary(pending)
        return added


def approve_pending_terms(
    terms: list[str],
    domain: str | None = None,
) -> int:
    """Approve pending terms and move to domain vocabulary.

    Args:
        terms: Terms to approve (or "all" approves all)
        domain: Override domain (uses pending domain if None)

    Returns:
        Number of terms approved
    """
    # SECURITY: Validate domain if provided
    if domain and not validate_domain_name(domain):
        raise ValueError(f"Invalid domain name: {domain}")

    # Use file locking context manager (DRY refactor)
    with _file_lock("pending-vocabulary"):
        pending = load_pending_vocabulary()
        approved = 0

        for term in terms:
            if term not in pending:
                continue

            entry = pending[term]
            target_domain = domain or entry["domain"]

            # Add to domain vocabulary
            save_domain_vocabulary(target_domain, [term])

            # Remove from pending
            del pending[term]
            approved += 1

        save_pending_vocabulary(pending)
        return approved


def reject_pending_terms(terms: list[str]) -> int:
    """Reject pending terms (remove without adding to vocabulary)."""
    # Use file locking context manager (DRY refactor)
    with _file_lock("pending-vocabulary"):
        pending = load_pending_vocabulary()
        rejected = 0

        for term in terms:
            if term in pending:
                del pending[term]
                rejected += 1

        save_pending_vocabulary(pending)
        return rejected


def get_auto_approve_candidates(min_occurrences: int = 3) -> list[tuple[str, dict]]:
    """Get pending terms that have enough occurrences for auto-approval.

    Terms with multiple independent occurrences are less likely to be
    hallucinations and can be auto-approved.
    """
    pending = load_pending_vocabulary()
    return [
        (term, info) for term, info in pending.items()
        if info["occurrences"] >= min_occurrences
    ]

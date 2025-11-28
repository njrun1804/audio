"""ASR - Local Whisper transcription optimized for Apple Silicon."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
_env_file = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_file if _env_file.exists() else None)

__version__ = "0.2.0"


def _configure_compute_threads():
    """Configure thread pools for M4 architecture (4P + 6E cores).

    Prevents thread oversubscription which causes contention with GPU operations.
    Uses setdefault to allow environment override.
    """
    # Limit BLAS threads to prevent oversubscription with MLX GPU
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

    # Configure PyTorch for M4 (if available)
    try:
        import torch
        torch.set_num_threads(4)           # Use P-cores for compute
        torch.set_num_interop_threads(2)   # Parallel op coordination
    except ImportError:
        pass


_configure_compute_threads()

"""Pytest fixtures for ASR tests."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_audio() -> Path:
    """Return path to sample audio file."""
    path = FIXTURES_DIR / "sample_short.m4a"
    if not path.exists():
        pytest.skip("Sample audio not found - run: cp test.m4a tests/fixtures/sample_short.m4a")
    return path


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return FIXTURES_DIR

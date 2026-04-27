from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def read_fixture(fixtures_dir: Path):
    def _read(rel_path: str) -> bytes:
        return (fixtures_dir / rel_path).read_bytes()

    return _read

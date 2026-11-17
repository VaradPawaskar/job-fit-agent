from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_cv() -> str:
    return (ROOT / "samples" / "sample_cv.txt").read_text()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # tests must never depend on the developer's real settings
    for key in ("API_KEY", "ALLOW_PRIVATE_URLS", "MODEL"):
        monkeypatch.delenv(key, raising=False)

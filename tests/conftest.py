import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lookroom import seed  # noqa: E402
from lookroom.models import Item, validate_context  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Every test gets its own temporary SQLite DB and image folder."""
    monkeypatch.setenv("LOOKROOM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LOOKROOM_DISABLE_LLM", "1")  # tests never call the paid API unless they opt in
    monkeypatch.setenv("LOOKROOM_LOCAL_LLM", "0")    # nor load the local LLM, except the tests that opt in
    return tmp_path


@pytest.fixture
def seeded(isolated_data):
    seed.seed()
    return isolated_data


@pytest.fixture
def ctx():
    return validate_context("work", 18, "smart")


def make_item(id, category, color="black", style="casual", warmth=3, formality=3):
    return Item(id=id, name=id, category=category, color=color, style=style, warmth=warmth, formality=formality)

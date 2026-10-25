import json
import os
from types import SimpleNamespace

import pytest

from lookroom import stylist
from lookroom.models import validate_context

DEFAULT = validate_context("work", 16, "smart")
GOOD_IDS = ["top-white-oxford", "bottom-navy-trousers", "shoes-black-formal", "outer-black-jacket"]


@pytest.mark.parametrize("msg,expect", [
    ("dinner date tonight, 8 degrees, classic", ("date", 8.0, "classic")),
    ("client meeting, about 22°C", ("work", 22.0, "smart")),
    ("lazy weekend, it's freezing", ("casual", -2.0, "smart")),
    ("streetwear look for the park", ("casual", 16.0, "streetwear")),
])
def test_parse_context(msg, expect):
    c = stylist.parse_context(msg, DEFAULT)
    assert (c.occasion, c.temperature, c.style) == expect


def test_offline_uses_rule_engine(seeded):
    r = stylist.chat("formal gala, 5 degrees", [], DEFAULT)
    assert r.source == "rules" and r.outfit is not None
    assert r.context.occasion == "formal" and "outerwear" in {i.category for i in r.outfit.items}


class FakeClient:
    """Stands in for anthropic.Anthropic so tests are free, offline and deterministic."""
    payload: dict = {}
    stop_reason = "end_turn"
    calls: list = []

    def __init__(self, **_):
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        FakeClient.calls.append(kwargs)
        return SimpleNamespace(stop_reason=FakeClient.stop_reason,
                               content=[SimpleNamespace(type="text", text=json.dumps(FakeClient.payload))],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=50))


@pytest.fixture
def fake_llm(seeded, monkeypatch):
    import anthropic
    monkeypatch.setenv("LOOKROOM_DISABLE_LLM", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "test-key"))
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    FakeClient.calls, FakeClient.stop_reason = [], "end_turn"
    FakeClient.payload = {"occasion": "date", "temperature": 8, "style": "classic", "item_ids": GOOD_IDS,
                          "reply": "Crisp white shirt, navy trousers, loafers and the camel coat."}
    return FakeClient


def test_claude_path_is_grounded_and_rescored(fake_llm):
    r = stylist.chat("dinner date at 8 degrees", [], DEFAULT)
    assert r.source == "claude" and sorted(r.outfit.key) == sorted(GOOD_IDS)
    assert 0 < r.outfit.score <= 100 and r.context.occasion == "date"
    req = fake_llm.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert '"wardrobe"' in req["messages"][-1]["content"]  # wardrobe metadata is sent as grounding


@pytest.mark.parametrize("bad_ids", [
    ["top-white-oxford", "bottom-navy-trousers", "shoes-invented-gucci"],    # hallucinated item
    ["top-white-oxford", "top-black-tee", "shoes-black-formal"],             # two tops, no bottom
])
def test_invalid_model_outfit_falls_back(fake_llm, bad_ids):
    fake_llm.payload["item_ids"] = bad_ids
    r = stylist.chat("dinner date at 8 degrees", [], DEFAULT)
    assert r.source == "rules" and "Claude unavailable" in r.note and r.outfit is not None


def test_daily_budget_cap_enforced(fake_llm, monkeypatch):
    monkeypatch.setenv("LOOKROOM_LLM_DAILY_CAP", "2")
    sources = [stylist.chat("dinner date at 8 degrees", [], DEFAULT).source for _ in range(3)]
    assert sources == ["claude", "claude", "rules"] and len(fake_llm.calls) == 2
    r = stylist.chat("dinner date", [], DEFAULT)
    assert "budget" in r.note


def test_refusal_falls_back(fake_llm):
    fake_llm.stop_reason = "refusal"
    assert stylist.chat("dinner date", [], DEFAULT).source == "rules"


@pytest.mark.skipif(os.environ.get("RUN_LIVE_LLM") != "1", reason="set RUN_LIVE_LLM=1 to call the real API")
def test_live_claude(seeded, monkeypatch):
    monkeypatch.setenv("LOOKROOM_DISABLE_LLM", "0")
    r = stylist.chat("Dinner date tonight, about 8 degrees, something classic", [], DEFAULT)
    assert r.source == "claude", r.note

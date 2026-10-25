"""AI stylist: Claude picks an outfit from the REAL wardrobe (structured output, validated),
with the deterministic engine as both grounding and fallback.

Routing: Claude (if ANTHROPIC_API_KEY is set and LOOKROOM_DISABLE_LLM != 1) -> rules on any failure.
Privacy: only item metadata (no images, no personal data) is sent to the API.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from . import db, i18n, recommender
from .models import OCCASIONS, REQUIRED_CATEGORIES, STYLES, TEMP_MAX, TEMP_MIN, Context, Item, validate_context

MODEL = "claude-opus-5-5"
MAX_HISTORY_TURNS = 6
ENGINE_HINTS = 5


def daily_cap() -> int:
    """Global spend guard for public hosting: max paid Claude calls per UTC day (env LOOKROOM_LLM_DAILY_CAP)."""
    try:
        return max(0, int(os.environ.get("LOOKROOM_LLM_DAILY_CAP", "150")))
    except ValueError:
        return 150

SYSTEM_PROMPT = """You are the stylist inside Wardrobe Intelligence Lab, a wardrobe app.
Pick ONE outfit for the user using only item ids from the wardrobe JSON you are given.
An outfit is exactly one top, one bottom and one pair of shoes, plus at most one outerwear piece when it is cool (below ~20°C) and required below 12°C.
Infer occasion, temperature (°C) and style from the user's message; when something is not stated, keep the current app context.
The engine's top-ranked outfits for the current context are provided with transparent scores. Prefer them unless the user's request clearly calls for something else; respect learned likes/dislikes.
Write `reply` in the language named in reply_language, as a warm, concise stylist note (2-4 sentences, plain text, no markdown) that names the pieces and says why they work. Never mention item ids or pieces that are not in the wardrobe."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "occasion": {"type": "string", "enum": list(OCCASIONS)},
        "temperature": {"type": "number"},
        "style": {"type": "string", "enum": list(STYLES)},
        "item_ids": {"type": "array", "items": {"type": "string"}},
        "reply": {"type": "string"},
    },
    "required": ["occasion", "temperature", "style", "item_ids", "reply"],
    "additionalProperties": False,
}


@dataclass
class StylistReply:
    text: str
    context: Context
    outfit: recommender.Outfit | None
    source: str                      # "claude" | "rules"
    latency_ms: int = 0
    note: str | None = None          # why we fell back, if we did
    usage: dict = field(default_factory=dict)


def llm_available() -> bool:
    if os.environ.get("LOOKROOM_DISABLE_LLM") == "1" or not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    import importlib.util
    return importlib.util.find_spec("anthropic") is not None  # cheap check; the SDK is imported on first use


# ---------- deterministic fallback: keyword context parsing + engine ----------

_OCCASION_WORDS = {
    "work": ("work", "office", "meeting", "interview", "presentation", "client"),
    "date": ("date", "dinner", "romantic", "drinks"),
    "formal": ("formal", "wedding", "gala", "ceremony", "black tie", "opera"),
    "casual": ("casual", "weekend", "park", "errands", "brunch", "coffee", "relax"),
}
_STYLE_WORDS = {s: (s,) for s in STYLES} | {"streetwear": ("streetwear", "street", "urban", "hoodie")}


def parse_context(message: str, default: Context) -> Context:
    m = message.lower()
    occasion = next((o for o, words in _OCCASION_WORDS.items() if any(w in m for w in words)), default.occasion)
    style = next((s for s, words in _STYLE_WORDS.items() if any(w in m for w in words)), default.style)
    temp = default.temperature
    if hit := re.search(r"(-?\d{1,2})\s*(?:°|degrees|deg|c\b)", m):
        temp = float(hit.group(1))
    elif any(w in m for w in ("freezing", "snow", "very cold")):
        temp = -2.0
    elif "cold" in m or "chilly" in m:
        temp = 6.0
    elif "hot" in m or "summer" in m:
        temp = 29.0
    elif "warm" in m or "sunny" in m:
        temp = 23.0
    temp = max(TEMP_MIN, min(TEMP_MAX, temp))
    return validate_context(occasion, temp, style)



LANGUAGE_NAMES = {"en": "English", "hy": "Armenian (Հայերեն)", "ru": "Russian (Русский)"}


def _rules_reply(message: str, default: Context, items: list[Item], prefs: dict, note: str | None,
                 lang: str = "en") -> StylistReply:
    t0 = time.perf_counter()
    ctx = parse_context(message, default)
    res = recommender.recommend(items, ctx, prefs, n=1, lang=lang)
    if not res.outfits:
        return StylistReply(res.message or "", ctx, None, "rules", note=note)
    o = res.outfits[0]
    text = i18n.t("rules_reply", lang, occ=i18n.label("occasion", ctx.occasion, lang), temp=f"{ctx.temperature:.0f}",
                  style=i18n.label("style", ctx.style, lang),
                  names=", ".join(i18n.item_name(i, lang) for i in o.items), why=o.explanation)
    return StylistReply(text, ctx, o, "rules", int((time.perf_counter() - t0) * 1000), note)


# ---------- Claude path ----------

def _validate_ids(ids: list[str], by_id: dict[str, Item]) -> tuple[Item, ...] | None:
    if len(ids) != len(set(ids)) or any(i not in by_id for i in ids):
        return None
    chosen = [by_id[i] for i in ids]
    cats = [c.category for c in chosen]
    if len(cats) != len(set(cats)) or not set(REQUIRED_CATEGORIES) <= set(cats):
        return None
    order = {c: n for n, c in enumerate((*REQUIRED_CATEGORIES, "outerwear"))}
    return tuple(sorted(chosen, key=lambda it: order[it.category]))


def _claude_reply(message: str, history: list[dict], default: Context, items: list[Item],
                  prefs: dict, lang: str = "en") -> StylistReply:
    import anthropic

    t0 = time.perf_counter()
    by_id = {i.id: i for i in items}
    hints = recommender.recommend(items, default, prefs, n=ENGINE_HINTS).outfits
    wardrobe = [{"id": i.id, "name": i18n.item_name(i, lang), "category": i.category, "color": i.color,
                 "style": i.style, "warmth": i.warmth, "formality": i.formality,
                 "learned_net_votes": prefs.get(i.id, 0)} for i in items]
    grounding = {
        "current_app_context": {"occasion": default.occasion, "temperature_c": default.temperature,
                                "style": default.style},
        "reply_language": LANGUAGE_NAMES[lang],
        "wardrobe": wardrobe,
        "engine_top_outfits": [{"item_ids": list(o.key), "score": o.score, **o.breakdown.as_dict()} for o in hints],
    }
    msgs = [{"role": h["role"], "content": h["content"]} for h in history[-MAX_HISTORY_TURNS * 2:]]
    msgs.append({"role": "user", "content": f"<data>{json.dumps(grounding, sort_keys=True, ensure_ascii=False)}</data>"
                                            f"\n\n{message}"})

    client = anthropic.Anthropic(timeout=45.0, max_retries=1)
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=msgs,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("model output truncated")
    raw = next((b.text for b in resp.content if b.type == "text"), "")
    data = json.loads(raw)
    ctx = validate_context(data["occasion"], max(TEMP_MIN, min(TEMP_MAX, float(data["temperature"]))), data["style"])
    chosen = _validate_ids(data["item_ids"], by_id)
    if chosen is None:
        raise ValueError(f"model proposed an invalid outfit: {data['item_ids']}")
    outfit = recommender.score_outfit(chosen, ctx, prefs, lang)
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return StylistReply(data["reply"].strip(), ctx, outfit, "claude", int((time.perf_counter() - t0) * 1000),
                        usage=usage)


def chat(message: str, history: list[dict] | None = None, default: Context | None = None,
         allow_llm: bool = True, user_id: str = db.DEFAULT_USER, lang: str = "en") -> StylistReply:
    """history: prior turns as [{"role": "user"|"assistant", "content": str}].
    allow_llm=False forces the rule engine (e.g. a per-session limit). lang: en | hy | ru."""
    lang = i18n.norm(lang)
    message = (message or "").strip()[:1000]
    default = default or validate_context("casual", 18, "casual")
    items = db.list_items(user_id)
    prefs = db.item_preferences(user_id)

    def rules(note: str | None) -> StylistReply:
        return _rules_reply(message, default, items, prefs, note, lang)

    if not message or recommender.missing_categories(items):
        return rules(None)
    if not llm_available():
        return rules("LLM offline: rule engine")
    if not allow_llm:
        return rules("session Claude limit reached: rule engine")
    if not db.try_consume_llm_call(daily_cap()):
        return rules("daily Claude budget reached: rule engine")
    try:
        return _claude_reply(message, history or [], default, items, prefs, lang)
    except Exception as exc:  # network, auth, rate limit, refusal, invalid JSON/outfit -> deterministic fallback
        return rules(f"Claude unavailable ({type(exc).__name__}); used the rule engine")

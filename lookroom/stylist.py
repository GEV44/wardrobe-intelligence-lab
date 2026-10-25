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
from .models import (DISPLAY_ORDER, OCCASIONS, STYLES, TEMP_MAX, TEMP_MIN, Context, Item, is_valid_outfit,
                     validate_context)

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
An outfit is either one top + one bottom + one pair of shoes, or one dress + one pair of shoes, plus at most one outerwear piece when it is cool (below ~20°C) and required below 12°C.
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

# Multilingual keyword rules (EN / RU / HY stems). Explicit matches always win over the LLM.
_OCCASION_WORDS = {
    "formal": ("formal", "wedding", "gala", "ceremony", "black tie", "opera", "theatre", "theater",
               "свадьб", "официальн", "торжеств", "театр", "опер", "церемон",
               "հարսանիք", "պաշտոնական", "թատրոն", "արարողություն"),
    "work": ("work", "office", "meeting", "interview", "presentation", "client", "conference",
             "работ", "офис", "встреч", "собеседован", "клиент", "презентац", "конференц",
             "աշխատանք", "գրասենյակ", "հանդիպում", "հարցազրույց", "հաճախորդ", "շնորհանդես"),
    "date": ("date", "dinner", "romantic", "drinks", "restaurant",
             "свидани", "ужин", "романт", "ресторан",
             "ժամադրություն", "ընթրիք", "ռեստորան", "ռոմանտիկ"),
    "casual": ("casual", "weekend", "park", "errands", "brunch", "coffee", "relax", "walk",
               "выходн", "парк", "кофе", "прогулк", "отдых", "повседнев",
               "հանգստյան", "այգի", "սուրճ", "զբոսանք", "ամենօրյա"),
}
_STYLE_WORDS = {
    "streetwear": ("streetwear", "street", "urban", "hoodie", "стритвир", "уличн", "худи", "սթրիթ", "փողոցային"),
    "classic": ("classic", "классик", "классическ", "դասական"),
    "minimal": ("minimal", "минимал", "մինիմալ"),
    "smart": ("smart", "business", "смарт", "делов", "սմարթ", "գործնական"),
    "casual": ("casual", "relaxed", "кэжуал", "расслаблен", "ազատ"),
}
_TEMP_WORDS = ((-2.0, ("freezing", "snow", "very cold", "мороз", "снег", "сառնամանիք", "ձյուն")),
               (6.0, ("cold", "chilly", "холодн", "прохладн", "ցուրտ", "սառը", "զով")),
               (29.0, ("hot", "summer", "жарк", "жара", "лето", "շոգ", "ամառ")),
               (23.0, ("warm", "sunny", "тепл", "солнечн", "տաք", "արևոտ")))
_DEGREES = re.compile(r"(-?\d{1,2})\s*(?:°|degrees|deg|c\b|градус|град|աստիճան)")


def parse_context_explicit(message: str) -> dict:
    """Only what the keyword rules can see: {"occasion"|"style"|"temperature": value} (missing keys = unknown)."""
    m = message.lower()
    found: dict = {}
    if occ := next((o for o, words in _OCCASION_WORDS.items() if any(w in m for w in words)), None):
        found["occasion"] = occ
    if sty := next((s for s, words in _STYLE_WORDS.items() if any(w in m for w in words)), None):
        found["style"] = sty
    if hit := _DEGREES.search(m):
        found["temperature"] = float(hit.group(1))
    elif t := next((t for t, words in _TEMP_WORDS if any(w in m for w in words)), None):
        found["temperature"] = t
    return found


def parse_context(message: str, default: Context, extra: dict | None = None) -> Context:
    """Keyword rules first, then optional LLM-extracted fields (`extra`), then the app's current context."""
    found = parse_context_explicit(message)
    extra = extra or {}
    occasion = found.get("occasion") or (extra.get("occasion") if extra.get("occasion") in OCCASIONS else None) \
        or default.occasion
    style = found.get("style") or (extra.get("style") if extra.get("style") in STYLES else None) or default.style
    temp = found.get("temperature")
    if temp is None and isinstance(extra.get("temperature_c"), (int, float)):
        temp = float(extra["temperature_c"])
    temp = max(TEMP_MIN, min(TEMP_MAX, default.temperature if temp is None else temp))
    return validate_context(occasion, temp, style)



LANGUAGE_NAMES = {"en": "English", "hy": "Armenian (Հայերեն)", "ru": "Russian (Русский)"}


def _rules_reply(message: str, default: Context, items: list[Item], prefs: dict, note: str | None,
                 lang: str = "en", use_local_llm: bool = False) -> StylistReply:
    """Engine-chosen outfit. With use_local_llm, a free local model (Qwen3) fills unknown context fields and
    phrases the reply; every LLM step falls back to the deterministic rule/template path."""
    from . import llm

    t0 = time.perf_counter()
    local = use_local_llm and llm.enabled()
    extra = None
    if local:
        found = parse_context_explicit(message)
        if len(found) < 3:  # only ask the model for what the keyword rules could not see
            extra = llm.extract_context(message)
    ctx = parse_context(message, default, extra)
    res = recommender.recommend(items, ctx, prefs, n=1, lang=lang)
    if not res.outfits:
        return StylistReply(res.message or "", ctx, None, "rules", note=note)
    o = res.outfits[0]
    names = [i18n.item_name(i, lang) for i in o.items]
    text = llm.stylist_reply(lang, i18n.label("occasion", ctx.occasion, lang), ctx.temperature,
                             i18n.label("style", ctx.style, lang), names, o.explanation) if local else None
    source = "local" if (local and (text or extra)) else "rules"
    if not text:
        text = i18n.t("rules_reply", lang, occ=i18n.label("occasion", ctx.occasion, lang),
                      temp=f"{ctx.temperature:.0f}", style=i18n.label("style", ctx.style, lang),
                      names=", ".join(names), why=o.explanation)
    return StylistReply(text, ctx, o, source, int((time.perf_counter() - t0) * 1000), note)


# ---------- Claude path ----------

def _validate_ids(ids: list[str], by_id: dict[str, Item]) -> tuple[Item, ...] | None:
    if len(ids) != len(set(ids)) or any(i not in by_id for i in ids):
        return None
    chosen = [by_id[i] for i in ids]
    if not is_valid_outfit([c.category for c in chosen]):
        return None
    return tuple(sorted(chosen, key=lambda it: DISPLAY_ORDER[it.category]))


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

    def rules(note: str | None, local: bool = True) -> StylistReply:
        return _rules_reply(message, default, items, prefs, note, lang, use_local_llm=local)

    if not message or recommender.missing_categories(items):
        return rules(None, local=False)
    if not llm_available():  # no paid key configured: the free local model (Qwen3) + engine is the default path
        return rules(None)
    if not allow_llm:
        return rules("session Claude limit reached: rule engine")
    if not db.try_consume_llm_call(daily_cap()):
        return rules("daily Claude budget reached: rule engine")
    try:
        return _claude_reply(message, history or [], default, items, prefs, lang)
    except Exception as exc:  # network, auth, rate limit, refusal, invalid JSON/outfit -> deterministic fallback
        return rules(f"Claude unavailable ({type(exc).__name__}); used the rule engine")

"""Pure, deterministic outfit recommendation engine.

No database or UI imports: every function takes plain data and returns plain data,
so the scoring is unit-testable and reusable from Streamlit and FastAPI.

total = 100 * sum(WEIGHTS[c] * component_c), each component normalized to [0, 1].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable, Mapping

from . import i18n
from .models import Context, Item

WEIGHTS = {"occasion": 0.30, "weather": 0.25, "color": 0.15, "style": 0.15, "preference": 0.15}

OCCASION_FORMALITY = {"casual": 1.5, "work": 3.5, "date": 3.0, "formal": 4.5}

OUTERWEAR_REQUIRED_BELOW = 12.0  # °C: candidates must include outerwear (if any is owned)
OUTERWEAR_ALLOWED_BELOW = 20.0   # °C: outerwear optional; at/above this it is excluded
OUTERWEAR_WARMTH_WEIGHT = 2      # the outer layer counts double in outfit warmth

NEUTRALS = frozenset({"white", "black", "grey", "cream", "beige", "camel", "brown", "navy"})
# Non-neutral pairs that follow a simple analogous / muted-contrast rule.
PALETTE_PAIRS = {
    frozenset({"blue", "olive"}), frozenset({"blue", "burgundy"}), frozenset({"olive", "burgundy"}),
    frozenset({"green", "burgundy"}), frozenset({"blue", "pink"}), frozenset({"olive", "yellow"}),
}
COLOR_SCORES = {"neutral_pair": 1.0, "neutral_anchor": 0.9, "monochrome": 0.75, "palette": 0.7, "clash": 0.4}

STYLE_NEIGHBORS = {
    "classic": {"smart", "minimal"},
    "smart": {"classic", "minimal"},
    "minimal": {"classic", "smart", "casual"},
    "casual": {"streetwear", "minimal"},
    "streetwear": {"casual"},
}
STYLE_SCORES = {"exact": 1.0, "neighbor": 0.6, "other": 0.25}

PREFERENCE_SCALE = 2.0      # signal = tanh(net_votes / scale): bounded in (-1, 1)
DIVERSITY_PENALTY = 4.0     # points subtracted per item shared with an already-selected outfit
TOP_REPEAT_PENALTY = 6.0    # extra points when the top repeats
TOP_N = 3


# ---------- component scores (all return 0..1) ----------

def target_warmth(temperature: float) -> float:
    """Linear map: 30 °C -> 1.5, -5 °C -> 5.0, clamped to the 1..5 warmth scale."""
    return max(1.0, min(5.0, 1.5 + (30.0 - temperature) * 0.1))


def outfit_warmth(items: Iterable[Item]) -> float:
    total = weight = 0
    for it in items:
        w = OUTERWEAR_WARMTH_WEIGHT if it.category == "outerwear" else 1
        total += it.warmth * w
        weight += w
    return total / weight if weight else 0.0


def weather_score(items: Iterable[Item], temperature: float) -> float:
    return max(0.0, 1.0 - abs(outfit_warmth(items) - target_warmth(temperature)) / 3.0)


def occasion_score(items: Iterable[Item], occasion: str) -> float:
    target = OCCASION_FORMALITY[occasion]
    fits = [max(0.0, 1.0 - abs(it.formality - target) / 4.0) for it in items]
    return sum(fits) / len(fits)


def color_rule(a: str, b: str) -> str:
    if a in NEUTRALS and b in NEUTRALS:
        return "neutral_pair"
    if a in NEUTRALS or b in NEUTRALS:
        return "neutral_anchor"
    if a == b:
        return "monochrome"
    if frozenset({a, b}) in PALETTE_PAIRS:
        return "palette"
    return "clash"


def color_score(items: Iterable[Item]) -> float:
    colors = [it.color for it in items]
    pairs = list(combinations(colors, 2))
    if not pairs:
        return 1.0
    return sum(COLOR_SCORES[color_rule(a, b)] for a, b in pairs) / len(pairs)


def style_score(items: Iterable[Item], style: str) -> float:
    def one(s: str) -> float:
        if s == style:
            return STYLE_SCORES["exact"]
        return STYLE_SCORES["neighbor"] if s in STYLE_NEIGHBORS.get(style, ()) else STYLE_SCORES["other"]
    vals = [one(it.style) for it in items]
    return sum(vals) / len(vals)


def preference_signal(net_votes: float) -> float:
    return math.tanh(net_votes / PREFERENCE_SCALE)


def preference_score(items: Iterable[Item], prefs: Mapping[str, float]) -> float:
    """0.5 = neutral (no feedback); bounded to (0, 1) however many votes accumulate."""
    sig = [preference_signal(prefs.get(it.id, 0)) for it in items]
    return 0.5 + 0.5 * (sum(sig) / len(sig))


# ---------- outfit objects ----------

@dataclass(frozen=True)
class ScoreBreakdown:
    occasion: float
    weather: float
    color: float
    style: float
    preference: float

    @property
    def total(self) -> float:
        return 100.0 * sum(WEIGHTS[k] * getattr(self, k) for k in WEIGHTS)

    def as_dict(self) -> dict[str, float]:
        return {k: round(getattr(self, k), 3) for k in WEIGHTS}


@dataclass(frozen=True)
class Outfit:
    items: tuple[Item, ...]
    breakdown: ScoreBreakdown
    explanation: str

    @property
    def key(self) -> tuple[str, ...]:
        return tuple(sorted(it.id for it in self.items))

    @property
    def score(self) -> float:
        return round(self.breakdown.total, 1)

    def by_category(self, category: str) -> Item | None:
        return next((it for it in self.items if it.category == category), None)


@dataclass
class RecommendationResult:
    outfits: list[Outfit]
    message: str | None = None
    candidates_considered: int = 0


# ---------- explanation ----------

def explain(items: tuple[Item, ...], b: ScoreBreakdown, ctx: Context, lang: str = "en") -> str:
    """Deterministic explanation derived from the actual score components (en / hy / ru)."""
    def tr(key: str, **kw) -> str:
        return i18n.t(key, lang, **kw)

    occ = i18n.label("occasion", ctx.occasion, lang)
    style = i18n.label("style", ctx.style, lang)
    parts = [tr("occ_good" if b.occasion >= 0.75 else "occ_bad" if b.occasion < 0.55 else "occ_ok", occ=occ)]
    outer = next((it for it in items if it.category == "outerwear"), None)
    temp = f"{ctx.temperature:.0f}°C"
    good = b.weather >= 0.75
    if outer:
        parts.append(tr("warm_good_outer" if good else "warm_bad_outer", temp=temp, outer=i18n.item_name(outer, lang)))
    else:
        parts.append(tr("warm_good" if good else "warm_bad", temp=temp))
    rules = {color_rule(a.color, c.color) for a, c in combinations(items, 2)}
    parts.append(tr("color_neutral" if rules == {"neutral_pair"} else "color_clash" if "clash" in rules
                    else "color_palette" if "palette" in rules else "color_anchor"))
    parts.append(tr("style_good" if b.style >= 0.75 else "style_mix", style=style))
    if b.preference > 0.55:
        parts.append(tr("pref_up"))
    elif b.preference < 0.45:
        parts.append(tr("pref_down"))
    return " ".join(parts)


# ---------- generation, scoring, selection ----------

def score_outfit(items: tuple[Item, ...], ctx: Context, prefs: Mapping[str, float] | None = None,
                 lang: str = "en") -> Outfit:
    prefs = prefs or {}
    b = ScoreBreakdown(
        occasion=occasion_score(items, ctx.occasion),
        weather=weather_score(items, ctx.temperature),
        color=color_score(items),
        style=style_score(items, ctx.style),
        preference=preference_score(items, prefs),
    )
    return Outfit(items=items, breakdown=b, explanation=explain(items, b, ctx, lang))


def missing_categories(items: Iterable[Item]) -> list[str]:
    """What blocks every outfit shape: shoes always; top+bottom unless a dress is available."""
    present = {it.category for it in items}
    missing = [] if "shoes" in present else ["shoes"]
    if "dress" not in present:
        missing = [c for c in ("top", "bottom") if c not in present] + missing
    return missing


def _outer_options(outer: list[Item], temperature: float) -> list[Item | None]:
    if temperature < OUTERWEAR_REQUIRED_BELOW and outer:
        return list(outer)
    return [None, *outer] if temperature < OUTERWEAR_ALLOWED_BELOW else [None]


def combos_from(by_cat: Mapping[str, list[Item]], temperature: float) -> list[tuple[Item, ...]]:
    """Both outfit shapes: top+bottom+shoes(+outer) and dress+shoes(+outer)."""
    outer_opts = _outer_options(list(by_cat.get("outerwear", [])), temperature)
    shoes = by_cat.get("shoes", [])
    combos = [(t, b, s) + ((o,) if o else ())
              for t, b, s, o in product(by_cat.get("top", []), by_cat.get("bottom", []), shoes, outer_opts)]
    combos += [(d, s) + ((o,) if o else ()) for d, s, o in product(by_cat.get("dress", []), shoes, outer_opts)]
    return combos


def generate_candidates(items: list[Item], temperature: float) -> list[tuple[Item, ...]]:
    by_cat: dict[str, list[Item]] = {}
    for it in sorted(items, key=lambda i: i.id):
        by_cat.setdefault(it.category, []).append(it)
    return combos_from(by_cat, temperature)


def _main_piece(o: Outfit) -> str:
    """The piece that defines the look: the top, or the dress for one-piece outfits."""
    main = o.by_category("top") or o.by_category("dress")
    return main.id if main else ""


def select_diverse(ranked: list[Outfit], n: int = TOP_N) -> list[Outfit]:
    """Greedy pick with a small penalty for items/main pieces already shown. Stable tie-break on ids."""
    chosen: list[Outfit] = []
    pool = list(ranked[:400])  # the top of the ranking is plenty to diversify over
    while pool and len(chosen) < n:
        used_items = {i for o in chosen for i in o.key}
        used_main = {_main_piece(o) for o in chosen}

        def adjusted(o: Outfit) -> float:
            shared = sum(1 for i in o.key if i in used_items)
            rep = TOP_REPEAT_PENALTY if _main_piece(o) in used_main else 0.0
            return o.breakdown.total - DIVERSITY_PENALTY * shared - rep

        best = min(pool, key=lambda o: (-adjusted(o), o.key))
        chosen.append(best)
        pool.remove(best)
    # Diversity decides *which* outfits are shown; display order is by true score.
    return sorted(chosen, key=lambda o: (-o.breakdown.total, o.key))


def _rank(combos: list[tuple[Item, ...]], ctx: Context, prefs: Mapping[str, float] | None) -> list[Outfit]:
    """Score every candidate (no explanation yet: it is generated only for outfits actually shown)."""
    prefs = prefs or {}
    scored = [Outfit(c, ScoreBreakdown(occasion_score(c, ctx.occasion), weather_score(c, ctx.temperature),
                                       color_score(c), style_score(c, ctx.style), preference_score(c, prefs)), "")
              for c in combos]
    return sorted(scored, key=lambda o: (-o.breakdown.total, o.key))


def _explained(o: Outfit, ctx: Context, lang: str) -> Outfit:
    return Outfit(o.items, o.breakdown, explain(o.items, o.breakdown, ctx, lang))


REPEAT_YESTERDAY_PENALTY = 8.0   # points per item also worn the previous day
REPEAT_WEEK_PENALTY = 2.0        # points per earlier use of the same item this week


def plan_week(items: list[Item], contexts: list[Context], prefs: Mapping[str, float] | None = None,
              lang: str = "en") -> list[Outfit | None]:
    """One outfit per day: best score minus penalties for repeating yesterday's items and overusing pieces.
    Greedy and deterministic (stable tie-break on item ids)."""
    if missing_categories(items):
        return [None] * len(contexts)
    plan: list[Outfit | None] = []
    uses: dict[str, int] = {}
    yesterday: set[str] = set()
    for ctx in contexts:
        scored = _rank(generate_candidates(items, ctx.temperature), ctx, prefs)

        def adjusted(o: Outfit) -> float:
            return (o.breakdown.total - REPEAT_YESTERDAY_PENALTY * sum(i in yesterday for i in o.key)
                    - REPEAT_WEEK_PENALTY * sum(uses.get(i, 0) for i in o.key))

        best = _explained(min(scored, key=lambda o: (-adjusted(o), o.key)), ctx, lang)
        plan.append(best)
        yesterday = set(best.key)
        for i in best.key:
            uses[i] = uses.get(i, 0) + 1
    return plan


def recommend(items: list[Item], ctx: Context, prefs: Mapping[str, float] | None = None,
              n: int = TOP_N, lang: str = "en", exclude: set[tuple[str, ...]] | None = None) -> RecommendationResult:
    """exclude: exact outfit keys the user rejected (a Dislike swaps the look immediately)."""
    missing = missing_categories(items)
    if missing:
        cats = ", ".join(i18n.label("category", c, lang) for c in missing)
        return RecommendationResult([], i18n.t("missing_cats", lang, cats=cats))
    candidates = generate_candidates(items, ctx.temperature)
    ranked = _rank(candidates, ctx, prefs)
    if exclude:
        kept = [o for o in ranked if o.key not in exclude]
        ranked = kept or ranked  # never show an empty page because of rejections
    message = None
    if ctx.temperature < OUTERWEAR_REQUIRED_BELOW and not any(i.category == "outerwear" for i in items):
        message = i18n.t("cold_no_outer", lang)
    shown = [_explained(o, ctx, lang) for o in select_diverse(ranked, n)]
    return RecommendationResult(shown, message, len(candidates))

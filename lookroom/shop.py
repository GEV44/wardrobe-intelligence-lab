"""'Shop the gap': which real catalog product would most improve this user's outfits?

For every product we compute the best outfit the user could make WITH it (product fixed, the rest from their
wardrobe) and compare against their current best. Gain is measured with the same transparent scoring engine,
so a suggestion is explainable ("+4.2 pts for Work at 8°C, pairs with your navy trousers").
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import product as cartesian
from pathlib import Path
from typing import Mapping

from . import recommender as rec
from .models import OCCASIONS, Context, Item

SHOP_DIR = Path(__file__).resolve().parent.parent / "data" / "shop"


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    category: str
    article: str
    color: str
    style: str
    warmth: int
    formality: int
    image: str

    def as_item(self) -> Item:
        return Item(id=self.id, name=self.name, category=self.category, color=self.color, style=self.style,
                    warmth=self.warmth, formality=self.formality)

    @property
    def image_path(self) -> Path:
        return SHOP_DIR / self.image


@lru_cache(maxsize=1)
def load_products() -> tuple[Product, ...]:
    path = SHOP_DIR / "products.json"
    if not path.is_file():
        return ()
    return tuple(Product(**{k: p[k] for k in Product.__dataclass_fields__}) for p in json.loads(path.read_text("utf-8")))


def _total(items: tuple[Item, ...], ctx: Context, prefs: Mapping[str, float]) -> float:
    b = rec.ScoreBreakdown(rec.occasion_score(items, ctx.occasion), rec.weather_score(items, ctx.temperature),
                           rec.color_score(items), rec.style_score(items, ctx.style), rec.preference_score(items, prefs))
    return b.total


def _outer_options(outer: list[Item], temperature: float) -> list[Item | None]:
    if temperature < rec.OUTERWEAR_REQUIRED_BELOW and outer:
        return list(outer)
    return [None, *outer] if temperature < rec.OUTERWEAR_ALLOWED_BELOW else [None]


def best_with(fixed: Item, wardrobe: list[Item], ctx: Context, prefs: Mapping[str, float]) -> tuple[float, tuple]:
    by_cat: dict[str, list[Item]] = {c: [] for c in ("top", "bottom", "shoes", "outerwear")}
    for it in wardrobe:
        by_cat[it.category].append(it)
    by_cat[fixed.category] = [fixed]
    if fixed.category == "outerwear" and ctx.temperature >= rec.OUTERWEAR_ALLOWED_BELOW:
        return 0.0, ()
    best, best_combo = 0.0, ()
    for top, bottom, shoes, o in cartesian(by_cat["top"], by_cat["bottom"], by_cat["shoes"],
                                           _outer_options(by_cat["outerwear"], ctx.temperature)):
        combo = (top, bottom, shoes) + ((o,) if o else ())
        if fixed not in combo:
            continue
        s = _total(combo, ctx, prefs)
        if s > best:
            best, best_combo = s, combo
    return best, best_combo


def current_best(wardrobe: list[Item], ctx: Context, prefs: Mapping[str, float]) -> float:
    cands = rec.generate_candidates(wardrobe, ctx.temperature)
    return max((_total(c, ctx, prefs) for c in cands), default=0.0)


def suggestions(wardrobe: list[Item], ctx: Context, prefs: Mapping[str, float] | None = None, n: int = 6,
                max_per_category: int = 2) -> list[dict]:
    """Top products by score gain in this context, plus how many occasions they improve (versatility)."""
    prefs = prefs or {}
    products = load_products()
    if not products or rec.missing_categories(wardrobe):
        return []
    base = current_best(wardrobe, ctx, prefs)
    other_ctx = [Context(o, ctx.temperature, ctx.style) for o in OCCASIONS]
    other_base = {c.occasion: current_best(wardrobe, c, prefs) for c in other_ctx}
    scored = []
    for p in products:
        item = p.as_item()
        s, combo = best_with(item, wardrobe, ctx, prefs)
        gain = s - base
        if gain <= 0.3 or not combo:
            continue
        scored.append((gain, s, p, combo))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2].id))
    out, per_cat = [], {}
    for gain, s, p, combo in scored:
        if per_cat.get(p.category, 0) >= max_per_category:
            continue
        per_cat[p.category] = per_cat.get(p.category, 0) + 1
        versatile = sum(1 for c in other_ctx if best_with(p.as_item(), wardrobe, c, prefs)[0] > other_base[c.occasion] + 0.3)
        out.append({"product": p, "gain": round(gain, 1), "score": round(s, 1), "base": round(base, 1),
                    "pairs_with": [it for it in combo if it.id != p.id], "occasions_improved": versatile})
        if len(out) >= n:
            break
    return out

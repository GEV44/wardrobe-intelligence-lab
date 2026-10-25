"""'Shop the gap': which real catalog product would most improve this user's outfits?

For every product we compute the best outfit the user could make WITH it (product fixed, the rest from their
wardrobe) and compare against their current best. Gain is measured with the same transparent scoring engine,
so a suggestion is explainable ("+4.2 pts for Work at 8°C, pairs with your navy trousers").
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
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
    gender: str = "men"

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
    return tuple(Product(**{k: p[k] for k in Product.__dataclass_fields__ if k in p})
                 for p in json.loads(path.read_text("utf-8")))


def _total(items: tuple[Item, ...], ctx: Context, prefs: Mapping[str, float]) -> float:
    b = rec.ScoreBreakdown(rec.occasion_score(items, ctx.occasion), rec.weather_score(items, ctx.temperature),
                           rec.color_score(items), rec.style_score(items, ctx.style), rec.preference_score(items, prefs))
    return b.total


NEAR_BEST = 5.0  # a look within this many points of the user's current best counts as a strong look


def best_with(fixed: Item, wardrobe: list[Item], ctx: Context, prefs: Mapping[str, float],
              with_count: bool = False, good_at: float = 75.0):
    """Best outfit that includes `fixed` (both outfit shapes), with the rest from the wardrobe.
    with_count=True also returns how many outfits with it score >= good_at (its versatility)."""
    by_cat: dict[str, list[Item]] = {}
    for it in wardrobe:
        by_cat.setdefault(it.category, []).append(it)
    by_cat[fixed.category] = [fixed]
    best, best_combo, good = 0.0, (), 0
    if not (fixed.category == "outerwear" and ctx.temperature >= rec.OUTERWEAR_ALLOWED_BELOW):
        for combo in rec.combos_from(by_cat, ctx.temperature):
            if fixed not in combo:
                continue
            s = _total(combo, ctx, prefs)
            good += s >= good_at
            if s > best:
                best, best_combo = s, combo
    return (best, best_combo, good) if with_count else (best, best_combo)


def current_best(wardrobe: list[Item], ctx: Context, prefs: Mapping[str, float]) -> float:
    cands = rec.generate_candidates(wardrobe, ctx.temperature)
    return max((_total(c, ctx, prefs) for c in cands), default=0.0)


TASTE_WEIGHT = 1.0   # usefulness multiplier: x2 at cosine similarity 1.0 to the user's liked-items taste vector
TODAY_WEIGHT = 2.0   # today's context counts double in the wardrobe-gap ranking
POOL_SIZE = 110      # representative products evaluated per request (one per attribute bucket first)
MIN_GAIN = 0.3


def situations(ctx: Context) -> list[Context]:
    """Today plus five representative situations: a gap analysis looks beyond a single day."""
    grid = [("formal", 4.0), ("work", 12.0), ("date", 18.0), ("casual", 24.0), ("casual", 6.0)]
    out = [ctx] + [Context(o, t, ctx.style) for o, t in grid]
    seen, uniq = set(), []
    for c in out:
        if (c.occasion, c.temperature) not in seen:
            seen.add((c.occasion, c.temperature))
            uniq.append(c)
    return uniq[:6]


def _pool(products: list[Product], sims: dict[str, float]) -> list[Product]:
    """Diverse, taste-aware subset: best-taste product per (category, color, style) bucket, then the rest."""
    ranked = sorted(products, key=lambda p: (-sims.get(p.id, 0.0), p.id))
    first, rest, seen = [], [], set()
    for p in ranked:
        key = (p.category, p.color, p.style)
        (rest if key in seen else first).append(p)
        seen.add(key)
    return (first + rest)[:POOL_SIZE]


def suggestions(wardrobe: list[Item], ctx: Context, prefs: Mapping[str, float] | None = None, n: int = 6,
                max_per_category: int = 2, taste=None, gender: str | None = None) -> list[dict]:
    """Wardrobe-gap analysis: products ranked by how much they raise the best possible outfit across today's
    context (double weight) and five representative situations, plus a visual taste bonus (image embeddings).
    `gender` filters the catalog (men / women)."""
    prefs = prefs or {}
    products = [p for p in load_products() if gender is None or p.gender == gender]
    if not products or rec.missing_categories(wardrobe):
        return []
    sims: dict[str, float] = {}
    if taste is not None:
        from . import similar
        sims = {p.id: s for p, s in similar.similar_products(taste, k=len(products))}
    ctxs = situations(ctx)
    bases = [current_best(wardrobe, c, prefs) for c in ctxs]
    scored = []
    for p in _pool(products, sims):
        item = p.as_item()
        results = [best_with(item, wardrobe, c, prefs, with_count=True, good_at=b - NEAR_BEST)
                   for c, b in zip(ctxs, bases)]
        gains = [max(0.0, s - b) for (s, _, _), b in zip(results, bases)]
        unlocks = results[0][2]  # new good looks it creates for today's context
        if sum(gains) <= MIN_GAIN and unlocks < 3:
            continue
        value = TODAY_WEIGHT * gains[0] + sum(gains[1:]) + 0.12 * min(unlocks, 40)
        rank = value * (1.0 + TASTE_WEIGHT * max(0.0, sims.get(p.id, 0.0)))  # taste re-ranks useful items only
        best_i = max(range(len(gains)), key=lambda i: (gains[i], i == 0))
        scored.append((rank, p, results, gains, best_i, unlocks))
    scored.sort(key=lambda x: (-x[0], x[1].id))
    out, per_cat = [], {}
    for _rank, p, results, gains, i, unlocks in scored:
        if per_cat.get(p.category, 0) >= max_per_category or not results[i][1]:
            continue
        per_cat[p.category] = per_cat.get(p.category, 0) + 1
        out.append({"product": p, "gain": round(gains[i], 1), "score": round(results[i][0], 1),
                    "base": round(bases[i], 1), "context": ctxs[i], "today_gain": round(gains[0], 1),
                    "unlocks": unlocks, "pairs_with": [it for it in results[i][1] if it.id != p.id],
                    "occasions_improved": sum(g > MIN_GAIN for g in gains), "situations": len(ctxs),
                    "taste": round(sims[p.id], 3) if p.id in sims else None})
        if len(out) >= n:
            break
    return out

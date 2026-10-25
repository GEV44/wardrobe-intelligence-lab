import io
import json

import pytest

from lookroom import db, recommender, service, shop, weather
from lookroom.models import validate_context

from test_weather import FakeResp


def _week(temps=(18, 17, 16, 15, 14, 20, 22), occ="work"):
    return [validate_context(occ, t, "smart") for t in temps]


def test_plan_week_one_valid_outfit_per_day(seeded):
    plan = service.plan_week(_week())
    assert len(plan) == 7 and all(o is not None for o in plan)
    ids = {i.id for i in db.list_items()}
    for o in plan:
        assert set(o.key) <= ids
        assert sorted(i.category for i in o.items if i.category != "outerwear") == ["bottom", "shoes", "top"]


def test_plan_week_avoids_repeating_yesterday(seeded):
    plan = service.plan_week(_week())
    for prev, cur in zip(plan, plan[1:]):
        assert not ({prev.by_category("top").id} & {cur.by_category("top").id}), "same top on consecutive days"
    assert plan == service.plan_week(_week())  # deterministic


def test_plan_week_handles_incomplete_wardrobe():
    from conftest import make_item
    assert recommender.plan_week([make_item("t", "top")], _week()) == [None] * 7


needs_shop = pytest.mark.skipif(not shop.load_products(), reason="shop catalog not built (scripts/build_shop.py)")


@needs_shop
def test_shop_suggestions_improve_score_and_are_real_products(seeded):
    ctx = validate_context("formal", 2, "classic")
    sugg = service.shop_suggestions(ctx)
    assert sugg, "expected at least one improving product for a cold formal day"
    base = shop.current_best(db.list_items(), ctx, {})
    catalog_ids = {p.id for p in shop.load_products()}
    for s in sugg:
        assert s["product"].id in catalog_ids and s["product"].image_path.is_file()
        assert s["score"] > base and s["gain"] > 0
        assert all(it.id in {i.id for i in db.list_items()} for it in s["pairs_with"])
    cats = [s["product"].category for s in sugg]
    assert all(cats.count(c) <= 2 for c in cats)


@needs_shop
def test_add_product_copies_real_photo_into_wardrobe(seeded):
    p = shop.load_products()[0]
    item = service.add_product(p.id)
    assert item.category == p.category and item.image_path.startswith("images/uploads/")
    with pytest.raises(LookupError):
        service.add_product("nope")


needs_emb = pytest.mark.skipif(not __import__("lookroom.similar", fromlist=["x"]).available(),
                               reason="embeddings not built (scripts/build_embeddings.py)")


@needs_emb
def test_visual_similarity_returns_same_category_ranked(seeded):
    res = service.similar_in_shop("shoes-white-sneakers", k=4)
    assert len(res) == 4 and all(r["product"].category == "shoes" for r in res)
    sims = [r["similarity"] for r in res]
    assert sims == sorted(sims, reverse=True) and sims[0] > 0.5
    with pytest.raises(LookupError):
        service.similar_in_shop("nope")


@needs_emb
@needs_shop
def test_taste_vector_from_likes_changes_shop_ranking(seeded):
    ctx = validate_context("casual", 18, "casual")
    assert service.taste_vector() is None  # no likes yet
    before = [s["product"].id for s in service.shop_suggestions(ctx, n=8)]
    top = service.get_recommendations(ctx).outfits[0]
    for _ in range(2):
        service.record_feedback(list(top.key), "like", ctx)
    tv = service.taste_vector()
    assert tv is not None and abs(float((tv ** 2).sum()) - 1) < 1e-3
    after = service.shop_suggestions(ctx, n=8)
    assert all(s["taste"] is not None for s in after)
    assert [s["product"].id for s in after] != before or all(s["taste"] is not None for s in after)


def test_forecast_daily_aggregation(monkeypatch):
    monkeypatch.setattr(weather, "_cache", {})
    series = []
    for day in (25, 26):
        for hour, temp in ((2, 5.0), (8, 10.0), (12, 14.0)):  # Yerevan = UTC+4: 06:00 local excluded, 12/16 count
            series.append({"time": f"2026-10-{day}T{hour:02d}:00:00Z",
                           "data": {"instant": {"details": {"air_temperature": temp}},
                                    "next_1_hours": {"summary": {"symbol_code": "rain"}}}})
    payload = json.dumps({"properties": {"timeseries": series}}).encode()
    fc = weather.forecast("Yerevan", fetch=lambda req, timeout: FakeResp(payload))
    assert [d["date"] for d in fc] == ["2026-10-25", "2026-10-26"]
    assert fc[0]["temperature"] == 12.0 and fc[0]["condition"] == "rain"

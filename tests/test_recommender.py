from lookroom import db, recommender, seed, service
from lookroom.models import CATEGORIES, validate_context

from conftest import make_item


def test_seed_creates_all_categories(seeded):
    items = db.list_items()
    assert len(items) == 14
    assert {i.category for i in items} == set(CATEGORIES)
    assert all(i.image_path and (seeded / i.image_path).is_file() for i in items)


def test_seed_is_idempotent(seeded):
    assert seed.seed() == 0
    assert len(db.list_items()) == 14


def test_recommendations_only_use_wardrobe_items(seeded, ctx):
    ids = {i.id for i in db.list_items()}
    res = service.get_recommendations(ctx)
    assert len(res.outfits) == 3
    for o in res.outfits:
        assert set(o.key) <= ids


def test_each_outfit_has_top_bottom_shoes(seeded):
    for temp in (-5, 15, 30):
        res = service.get_recommendations(validate_context("casual", temp, "casual"))
        for o in res.outfits:
            cats = [i.category for i in o.items]
            assert sorted(c for c in cats if c != "outerwear") == ["bottom", "shoes", "top"]
            assert cats.count("outerwear") <= 1
            if temp < recommender.OUTERWEAR_REQUIRED_BELOW:
                assert "outerwear" in cats
            if temp >= recommender.OUTERWEAR_ALLOWED_BELOW:
                assert "outerwear" not in cats


def test_top_recommendations_unique_and_diverse(seeded, ctx):
    res = service.get_recommendations(ctx)
    keys = [o.key for o in res.outfits]
    assert len(set(keys)) == len(keys)
    assert len({o.by_category("top").id for o in res.outfits}) == 3


def test_occasion_changes_occasion_score():
    items = (make_item("t", "top", formality=4), make_item("b", "bottom", formality=4),
             make_item("s", "shoes", formality=4))
    formal = recommender.score_outfit(items, validate_context("formal", 18, "casual"))
    casual = recommender.score_outfit(items, validate_context("casual", 18, "casual"))
    assert formal.breakdown.occasion > casual.breakdown.occasion
    assert formal.score > casual.score


def test_temperature_changes_weather_score():
    light = (make_item("t", "top", warmth=1), make_item("b", "bottom", warmth=2), make_item("s", "shoes", warmth=2))
    hot = recommender.score_outfit(light, validate_context("casual", 30, "casual"))
    cold = recommender.score_outfit(light, validate_context("casual", -5, "casual"))
    assert hot.breakdown.weather > cold.breakdown.weather


def test_scores_are_normalized(seeded, ctx):
    for o in service.get_recommendations(ctx).outfits:
        assert all(0.0 <= v <= 1.0 for v in o.breakdown.as_dict().values())
        assert 0 <= o.score <= 100
        assert o.explanation


def test_empty_and_incomplete_wardrobe_do_not_crash(ctx):
    empty = recommender.recommend([], ctx)
    assert empty.outfits == [] and "tops" in empty.message.lower()
    partial = recommender.recommend([make_item("t", "top"), make_item("b", "bottom")], ctx)
    assert partial.outfits == [] and "shoes" in partial.message.lower()
    assert "Обувь" in recommender.recommend(partial_items := [make_item("t", "top")], ctx, lang="ru").message
    assert partial_items
    db.init_db()
    assert service.get_recommendations(ctx).outfits == []


def test_cold_without_outerwear_still_recommends():
    items = [make_item("t", "top"), make_item("b", "bottom"), make_item("s", "shoes")]
    res = recommender.recommend(items, validate_context("casual", 0, "casual"))
    assert len(res.outfits) == 1 and "outerwear" in res.message


def test_recommendations_are_deterministic(seeded, ctx):
    a = service.get_recommendations(ctx)
    b = service.get_recommendations(ctx)
    assert [(o.key, o.score) for o in a.outfits] == [(o.key, o.score) for o in b.outfits]


def test_color_rules():
    assert recommender.color_rule("white", "navy") == "neutral_pair"
    assert recommender.color_rule("blue", "beige") == "neutral_anchor"
    assert recommender.color_rule("blue", "blue") == "monochrome"
    assert recommender.color_rule("blue", "olive") == "palette"
    assert recommender.color_rule("red", "green") == "clash"

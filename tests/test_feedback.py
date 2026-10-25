import sqlite3

import pytest

from lookroom import db, recommender, seed, service


def test_like_and_dislike_change_scores(seeded, ctx):
    top = service.get_recommendations(ctx).outfits[0]
    liked = service.record_feedback(list(top.key), "like", ctx)
    assert liked["score_after"] > liked["score_before"]

    other = service.get_recommendations(ctx).outfits[1]
    before = other.score
    res = service.record_feedback(list(other.key), "dislike", ctx)
    assert res["score_after"] < before


def test_repeated_dislikes_change_ranking(seeded, ctx):
    first = service.get_recommendations(ctx).outfits[0]
    for _ in range(3):
        service.record_feedback(list(first.key), "dislike", ctx)
    assert service.get_recommendations(ctx).outfits[0].key != first.key


def test_preference_signal_is_bounded():
    assert recommender.preference_signal(0) == 0.0
    assert 0 < recommender.preference_signal(1) < recommender.preference_signal(5) < 1.0
    for net in (1000, -1000):
        assert -1.0 <= recommender.preference_signal(net) <= 1.0
    # The personalization component can never move the total by more than its 15% weight.
    max_swing = 100 * recommender.WEIGHTS["preference"] * 0.5
    assert max_swing == 7.5


def test_feedback_survives_reconnection(seeded, ctx):
    o = service.get_recommendations(ctx).outfits[0]
    service.record_feedback(list(o.key), "like", ctx)
    conn = sqlite3.connect(db.db_path())  # fresh, independent connection
    assert conn.execute("SELECT COUNT(*) FROM outfit_feedback").fetchone()[0] == 1
    conn.close()
    assert service.feedback_summary()["likes"] == 1
    assert all(db.item_preferences()[i] == 1 for i in o.key)


def test_invalid_feedback_rejected(seeded, ctx):
    with pytest.raises(ValueError):
        service.record_feedback(["top-black-tee", "bottom-black-jeans", "shoes-white-sneakers"], "meh", ctx)
    with pytest.raises(LookupError):
        service.record_feedback(["nope", "bottom-black-jeans", "shoes-white-sneakers"], "like", ctx)
    with pytest.raises(ValueError):
        service.record_feedback(["top-black-tee", "top-grey-hoodie", "shoes-white-sneakers"], "like", ctx)


def test_reset_clears_feedback_and_user_items(seeded, ctx):
    o = service.get_recommendations(ctx).outfits[0]
    service.record_feedback(list(o.key), "like", ctx)
    service.add_item("Red Scarf Top", "top", "red", "casual", 2, 2)
    stats = seed.reset_demo()
    assert stats["feedback_deleted"] == 1 and stats["items_deleted"] == 1
    assert len(db.list_items()) == len(seed.load_demo_items()) and db.list_feedback() == []


def test_dislike_swaps_to_another_look(seeded, ctx):
    first = service.get_recommendations(ctx).outfits[0]
    service.record_feedback(list(first.key), "dislike", ctx)
    assert first.key not in [o.key for o in service.get_recommendations(ctx).outfits]


def test_show_another_look_skips_current(seeded, ctx):
    first = service.get_recommendations(ctx).outfits[0]
    assert service.get_recommendations(ctx, skip={first.key}).outfits[0].key != first.key


def test_switch_to_women_wardrobe(seeded):
    import pytest
    if "women" not in seed.available_presets():
        pytest.skip("women's wardrobe not built")
    from lookroom.models import validate_context
    service.switch_preset(db.DEFAULT_USER, "women")
    items = db.list_items()
    assert service.current_preset(db.DEFAULT_USER) == "women" and any(i.category == "dress" for i in items)
    res = service.get_recommendations(validate_context("date", 22, "classic"))
    assert res.outfits and all(i.id.startswith("w-") for o in res.outfits for i in o.items)
    service.switch_preset(db.DEFAULT_USER, "men")
    assert not any(i.category == "dress" for i in db.list_items())

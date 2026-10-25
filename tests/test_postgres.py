"""Same flows on PostgreSQL (the live backend). Runs only with TEST_DATABASE_URL set, e.g. the Supabase pooler URL."""
import os

import pytest

from lookroom import db, images, service
from lookroom.models import validate_context

PG = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not PG, reason="set TEST_DATABASE_URL to run against PostgreSQL")


@pytest.fixture
def pg(monkeypatch, isolated_data):
    monkeypatch.setenv("DATABASE_URL", PG)
    monkeypatch.delenv("LOOKROOM_NO_SECRETS", raising=False)
    users: list[str] = []
    yield users
    with db.connect() as conn:  # clean up only what this test created
        for uid in users:
            for table in ("wardrobe_items", "outfit_feedback", "user_profile", "accounts", "sessions", "blobs"):
                conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (uid,))


def test_account_wardrobe_feedback_and_photos_persist(pg):
    name = f"pgtest{os.getpid()}"
    uid = service.register(name, "secret123", "women")
    pg.append(uid)
    assert service.authenticate(name.upper(), "secret123") == uid
    with pytest.raises(service.AuthError):
        service.register(name, "secret123")
    token = service.start_session(uid)
    assert service.session_user(token) == uid

    assert db.count_items(uid) > 0
    item = service.add_item("Test Scarf", "outerwear", "red", "casual", 3, 2, user_id=uid)
    local = images.resolve_image(item.image_path)
    assert local is not None
    local.unlink()  # a fresh container has no local files ...
    assert images.resolve_image(item.image_path) is not None  # ... the photo comes back from the database

    ctx = validate_context("work", 12, "smart")
    hero = service.get_recommendations(ctx, uid).outfits[0]
    service.record_feedback(list(hero.key), "dislike", ctx, uid)
    assert tuple(sorted(hero.key)) in db.disliked_outfits(uid)
    assert service.get_recommendations(ctx, uid).outfits[0].key != hero.key

    service.end_session(token)
    assert service.session_user(token) is None

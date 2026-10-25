"""Idempotent sample-wardrobe seeding and per-user demo reset."""
from __future__ import annotations

import json
from dataclasses import replace

from . import db, images
from .models import validate_item

DEMO_ITEMS_PATH = db.PROJECT_ROOT / "data" / "demo_items.json"
CATALOG_DIR = db.PROJECT_ROOT / "data" / "catalog"   # real product photos (see data/catalog/CREDITS.md)


def load_demo_items():
    raw = json.loads(DEMO_ITEMS_PATH.read_text(encoding="utf-8"))
    return [validate_item({**r, "is_seed": True}) for r in raw]


def seed(restore: bool = False, user_id: str = db.DEFAULT_USER) -> int:
    """Give a user the sample wardrobe (existing rows untouched unless restore=True). Returns rows written."""
    db.init_db()
    written = 0
    for item in load_demo_items():
        item = replace(item, image_path=images.seed_image(item, CATALOG_DIR))
        written += db.upsert_item(item, replace=restore, user_id=user_id)
    return written


def ensure_user(user_id: str) -> None:
    """First visit: provision the sample wardrobe for this user."""
    if db.count_items(user_id) == 0:
        seed(user_id=user_id)


def reset_demo(user_id: str = db.DEFAULT_USER) -> dict[str, int]:
    """Restore this user's known-good state: original sample items, no feedback, no uploads."""
    db.init_db()
    paths = db.delete_non_seed_items(user_id)
    stats = {"feedback_deleted": db.clear_feedback(user_id), "items_deleted": len(paths),
             "uploads_deleted": images.delete_uploads(paths)}
    stats["seed_items"] = seed(restore=True, user_id=user_id)
    return stats

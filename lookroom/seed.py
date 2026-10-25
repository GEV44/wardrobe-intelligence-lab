"""Idempotent sample-wardrobe seeding (men's / women's presets) and per-user demo reset."""
from __future__ import annotations

import json
from dataclasses import replace

from . import db, images
from .models import validate_item

DATA = db.PROJECT_ROOT / "data"
CATALOG_DIR = DATA / "catalog"   # real product photos (see data/catalog/CREDITS.md)
PRESETS = {"men": DATA / "demo_items.json", "women": DATA / "demo_items_women.json"}
PRESET_GENDER = {"men": "men", "women": "women"}
DEFAULT_PRESET = "men"


def available_presets() -> list[str]:
    return [p for p, path in PRESETS.items() if path.is_file()]


def load_demo_items(preset: str = DEFAULT_PRESET):
    path = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [validate_item({**r, "is_seed": True}) for r in raw]


def seed(restore: bool = False, user_id: str = db.DEFAULT_USER, preset: str | None = None) -> int:
    """Give a user a sample wardrobe (existing rows untouched unless restore=True). Returns rows written."""
    db.init_db()
    preset = preset or db.get_preset(user_id) or DEFAULT_PRESET
    db.set_preset(user_id, preset)
    written = 0
    for item in load_demo_items(preset):
        item = replace(item, image_path=images.seed_image(item, CATALOG_DIR))
        written += db.upsert_item(item, replace=restore, user_id=user_id)
    return written


def ensure_user(user_id: str, preset: str | None = None) -> None:
    """First visit: provision the sample wardrobe for this user."""
    if db.count_items(user_id) == 0:
        seed(user_id=user_id, preset=preset)


def reset_demo(user_id: str = db.DEFAULT_USER, preset: str | None = None) -> dict[str, int]:
    """Restore a known-good state: sample items (optionally another preset), no feedback, no uploads."""
    db.init_db()
    switching = preset is not None and preset != (db.get_preset(user_id) or DEFAULT_PRESET)
    paths = db.delete_all_items(user_id) if switching else db.delete_non_seed_items(user_id)
    stats = {"feedback_deleted": db.clear_feedback(user_id), "items_deleted": len(paths),
             "uploads_deleted": images.delete_uploads(paths)}
    stats["seed_items"] = seed(restore=True, user_id=user_id, preset=preset)
    return stats

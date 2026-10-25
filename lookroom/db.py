"""SQLite persistence: schema + CRUD. Per-operation connections, parameterized SQL only.

Multi-tenant: every wardrobe item and feedback row belongs to a `user_id`, and every query is scoped by it.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .models import Context, Item

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_USER = "demo"
SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS wardrobe_items (
    user_id TEXT NOT NULL,
    id TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('top','bottom','dress','shoes','outerwear')),
    color TEXT NOT NULL,
    style TEXT NOT NULL,
    warmth INTEGER NOT NULL CHECK (warmth BETWEEN 1 AND 5),
    formality INTEGER NOT NULL CHECK (formality BETWEEN 1 AND 5),
    image_path TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    is_seed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, id)
);
CREATE TABLE IF NOT EXISTS outfit_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    item_ids TEXT NOT NULL,              -- JSON list, sorted
    vote INTEGER NOT NULL CHECK (vote IN (-1, 1)),
    occasion TEXT NOT NULL,
    temperature REAL NOT NULL,
    style TEXT NOT NULL,
    score_at_feedback REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON outfit_feedback(user_id, created_at);
CREATE TABLE IF NOT EXISTS user_profile (
    user_id TEXT PRIMARY KEY,
    preset TEXT NOT NULL DEFAULT 'men'    -- which sample wardrobe the user started from
);
CREATE TABLE IF NOT EXISTS llm_usage (
    day TEXT PRIMARY KEY,                -- UTC date, global spend guard
    calls INTEGER NOT NULL DEFAULT 0
);
"""


def data_dir() -> Path:
    d = Path(os.environ.get("LOOKROOM_DATA_DIR", PROJECT_ROOT / "runtime"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / "lookroom.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")  # concurrent readers while one session writes
        if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
            # Demo database: pre-tenancy layouts are rebuilt (documented in ARCHITECTURE.md).
            conn.executescript("DROP TABLE IF EXISTS wardrobe_items; DROP TABLE IF EXISTS outfit_feedback; "
                               "DROP TABLE IF EXISTS user_profile;")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.executescript(SCHEMA)


def _row_to_item(r: sqlite3.Row) -> Item:
    return Item(id=r["id"], name=r["name"], category=r["category"], color=r["color"], style=r["style"],
                warmth=r["warmth"], formality=r["formality"], image_path=r["image_path"],
                tags=tuple(json.loads(r["tags"] or "[]")), is_seed=bool(r["is_seed"]))


def upsert_item(item: Item, replace: bool = False, user_id: str = DEFAULT_USER) -> bool:
    """Insert an item. replace=False keeps an existing row (idempotent seed). Returns True if written."""
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    with connect() as conn:
        cur = conn.execute(
            f"{verb} INTO wardrobe_items (user_id, id, name, category, color, style, warmth, formality, image_path,"
            " tags, is_seed, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (user_id, item.id, item.name, item.category, item.color, item.style, item.warmth, item.formality,
             item.image_path, json.dumps(list(item.tags)), int(item.is_seed), _now()),
        )
        return cur.rowcount > 0


def list_items(user_id: str = DEFAULT_USER) -> list[Item]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY category, id",
                            (user_id,)).fetchall()
    return [_row_to_item(r) for r in rows]


def count_items(user_id: str = DEFAULT_USER) -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM wardrobe_items WHERE user_id = ?", (user_id,)).fetchone()[0]


def get_items(ids: list[str], user_id: str = DEFAULT_USER) -> dict[str, Item]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    with connect() as conn:
        rows = conn.execute(f"SELECT * FROM wardrobe_items WHERE user_id = ? AND id IN ({marks})",
                            [user_id, *ids]).fetchall()
    return {r["id"]: _row_to_item(r) for r in rows}


def delete_item(item_id: str, user_id: str = DEFAULT_USER) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM wardrobe_items WHERE user_id = ? AND id = ?",
                            (user_id, item_id)).rowcount > 0


def delete_non_seed_items(user_id: str = DEFAULT_USER) -> list[str]:
    """Delete a user's own items; returns their image paths so the caller can remove the files."""
    with connect() as conn:
        paths = [r[0] for r in conn.execute(
            "SELECT image_path FROM wardrobe_items WHERE user_id = ? AND is_seed = 0", (user_id,)) if r[0]]
        conn.execute("DELETE FROM wardrobe_items WHERE user_id = ? AND is_seed = 0", (user_id,))
    return paths


def add_feedback(item_ids: list[str], vote: int, ctx: Context, score: float | None,
                 user_id: str = DEFAULT_USER) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO outfit_feedback (user_id, item_ids, vote, occasion, temperature, style, score_at_feedback,"
            " created_at) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, json.dumps(sorted(item_ids)), vote, ctx.occasion, ctx.temperature, ctx.style, score, _now()),
        )
        return int(cur.lastrowid or 0)


def list_feedback(limit: int = 50, user_id: str = DEFAULT_USER) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM outfit_feedback WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                            (user_id, limit)).fetchall()
    return [{**dict(r), "item_ids": json.loads(r["item_ids"])} for r in rows]


def item_vote_counts(user_id: str = DEFAULT_USER) -> dict[str, dict[str, int]]:
    """{item_id: {"likes": n, "dislikes": m}} aggregated from this user's outfit-level feedback."""
    counts: dict[str, dict[str, int]] = {}
    with connect() as conn:
        rows = conn.execute("SELECT item_ids, vote FROM outfit_feedback WHERE user_id = ?", (user_id,)).fetchall()
    for r in rows:
        for iid in json.loads(r["item_ids"]):
            c = counts.setdefault(iid, {"likes": 0, "dislikes": 0})
            c["likes" if r["vote"] > 0 else "dislikes"] += 1
    return counts


def item_preferences(user_id: str = DEFAULT_USER) -> dict[str, int]:
    return {iid: c["likes"] - c["dislikes"] for iid, c in item_vote_counts(user_id).items()}


def clear_feedback(user_id: str = DEFAULT_USER) -> int:
    with connect() as conn:
        return conn.execute("DELETE FROM outfit_feedback WHERE user_id = ?", (user_id,)).rowcount


def disliked_outfits(user_id: str = DEFAULT_USER) -> set[tuple[str, ...]]:
    """Exact outfit combinations this user rejected (net votes < 0)."""
    with connect() as conn:
        rows = conn.execute("SELECT item_ids, SUM(vote) AS net FROM outfit_feedback WHERE user_id = ? "
                            "GROUP BY item_ids HAVING net < 0", (user_id,)).fetchall()
    return {tuple(json.loads(r["item_ids"])) for r in rows}


def get_preset(user_id: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT preset FROM user_profile WHERE user_id = ?", (user_id,)).fetchone()
    return row["preset"] if row else None


def set_preset(user_id: str, preset: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO user_profile (user_id, preset) VALUES (?, ?) "
                     "ON CONFLICT(user_id) DO UPDATE SET preset = excluded.preset", (user_id, preset))


def delete_all_items(user_id: str) -> list[str]:
    """Remove every wardrobe item of a user; returns uploaded image paths to clean up."""
    with connect() as conn:
        paths = [r[0] for r in conn.execute(
            "SELECT image_path FROM wardrobe_items WHERE user_id = ? AND is_seed = 0", (user_id,)) if r[0]]
        conn.execute("DELETE FROM wardrobe_items WHERE user_id = ?", (user_id,))
    return paths


def try_consume_llm_call(daily_cap: int) -> bool:
    """Atomically reserve one paid LLM call for today (UTC). False once the cap is reached."""
    day = datetime.now(timezone.utc).date().isoformat()
    with connect() as conn:
        conn.execute("INSERT OR IGNORE INTO llm_usage (day, calls) VALUES (?, 0)", (day,))
        cur = conn.execute("UPDATE llm_usage SET calls = calls + 1 WHERE day = ? AND calls < ?", (day, daily_cap))
        return cur.rowcount == 1


def llm_calls_today() -> int:
    day = datetime.now(timezone.utc).date().isoformat()
    with connect() as conn:
        row = conn.execute("SELECT calls FROM llm_usage WHERE day = ?", (day,)).fetchone()
    return int(row["calls"]) if row else 0

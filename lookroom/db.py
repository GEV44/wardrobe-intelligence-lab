"""Persistence: schema + CRUD. Parameterized SQL only, one short transaction per operation.

Two backends, same SQL (written in the common subset of SQLite and PostgreSQL):
- PostgreSQL when DATABASE_URL is set (the live app: Supabase, so data survives restarts and redeploys);
- SQLite under LOOKROOM_DATA_DIR otherwise (tests, local runs, offline fallback).
Multi-tenant: every row belongs to a `user_id`, and every query is scoped by it.
"""
from __future__ import annotations

import copy
import functools
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import Context, Item

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_USER = "demo"
SCHEMA_VERSION = 3

_TABLES = """
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
    id {serial},
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
CREATE TABLE IF NOT EXISTS accounts (
    user_id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,       -- lower-case
    pw_hash TEXT NOT NULL,               -- scrypt, see service.hash_password
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,         -- sha256 of the token kept in the visitor's URL
    user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blobs (       -- uploaded photos, so they survive a fresh container
    path TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    data {blob} NOT NULL,
    created_at TEXT NOT NULL
);
"""
SCHEMA = _TABLES.format(serial="INTEGER PRIMARY KEY AUTOINCREMENT", blob="BLOB")
PG_SCHEMA = _TABLES.format(serial="BIGSERIAL PRIMARY KEY", blob="BYTEA")

_ITEM_COLS = "user_id, id, name, category, color, style, warmth, formality, image_path, tags, is_seed, created_at"
_ready: set[str] = set()
_pools: dict[str, Any] = {}
_pool_lock = threading.Lock()


_CACHE_TTL_S = 60.0
_cache: dict[tuple, tuple[float, int, Any]] = {}
_versions: dict[str, int] = {}


def _touch(user_id: str) -> None:
    """A write for this user: drop every cached read of theirs."""
    _versions[user_id] = _versions.get(user_id, 0) + 1


def _cached(fn):
    """Remote database only: memoise per-user reads in-process (writes go through _touch)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not database_url():
            return fn(*args, **kwargs)
        user_id = kwargs.get("user_id", args[0] if args else DEFAULT_USER)
        key = (fn.__name__, args, tuple(sorted(kwargs.items())))
        version, hit = _versions.get(user_id, 0), _cache.get(key)
        if hit and hit[1] == version and time.monotonic() - hit[0] < _CACHE_TTL_S:
            return copy.deepcopy(hit[2])
        value = fn(*args, **kwargs)
        _cache[key] = (time.monotonic(), version, value)
        return copy.deepcopy(value)
    return wrapper


def database_url() -> str:
    if os.environ.get("LOOKROOM_NO_SECRETS") == "1":  # tests / recordings: Streamlit may have exported secrets
        return ""
    return os.environ.get("DATABASE_URL", "").strip()


def backend() -> str:
    return "postgres" if database_url() else "sqlite"


def data_dir() -> Path:
    d = Path(os.environ.get("LOOKROOM_DATA_DIR", PROJECT_ROOT / "runtime"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / "lookroom.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _Row(dict):
    """Like sqlite3.Row: r["col"] and r[0] both work."""

    def __init__(self, cols: list[str], values: tuple):
        super().__init__(zip(cols, values))
        self._values = values

    def __getitem__(self, key):
        return self._values[key] if isinstance(key, int) else super().__getitem__(key)


def _row_factory(cursor):
    cols = [d.name for d in cursor.description] if cursor.description else []
    return lambda values: _Row(cols, values)


class _PgConn:
    """Gives a psycopg connection the sqlite3 calling convention used below ("?" placeholders)."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params=()):
        cur = self._conn.cursor(row_factory=_row_factory)
        cur.execute(sql.replace("?", "%s"), params or None)
        return cur

    def executemany(self, sql: str, seq):
        cur = self._conn.cursor()
        cur.executemany(sql.replace("?", "%s"), seq)  # pipelined: one network round trip for the batch
        return cur


def _pool(url: str):
    with _pool_lock:
        if url not in _pools:
            from psycopg_pool import ConnectionPool

            _pools[url] = ConnectionPool(url, min_size=1, max_size=4, timeout=20, open=True,
                                         kwargs={"prepare_threshold": None, "connect_timeout": 15})
        return _pools[url]


@contextmanager
def connect() -> Iterator[Any]:
    url = database_url()
    if url:
        with _pool(url).connection() as conn:  # commits on success, rolls back on error
            yield _PgConn(conn)
        return
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
    key = database_url() or str(db_path())
    if key in _ready and (database_url() or db_path().exists()):
        return
    if database_url():
        with connect() as conn:
            conn.execute(PG_SCHEMA)
    else:
        with connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # concurrent readers while one session writes
            if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
                # Demo database: pre-tenancy layouts are rebuilt (documented in ARCHITECTURE.md).
                conn.executescript("DROP TABLE IF EXISTS wardrobe_items; DROP TABLE IF EXISTS outfit_feedback; "
                                   "DROP TABLE IF EXISTS user_profile;")
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.executescript(SCHEMA)
    _ready.add(key)


def _row_to_item(r) -> Item:
    return Item(id=r["id"], name=r["name"], category=r["category"], color=r["color"], style=r["style"],
                warmth=r["warmth"], formality=r["formality"], image_path=r["image_path"],
                tags=tuple(json.loads(r["tags"] or "[]")), is_seed=bool(r["is_seed"]))


def _upsert_sql(replace: bool) -> str:
    on_conflict = ("DO UPDATE SET name = excluded.name, category = excluded.category, color = excluded.color, "
                   "style = excluded.style, warmth = excluded.warmth, formality = excluded.formality, "
                   "image_path = excluded.image_path, tags = excluded.tags, is_seed = excluded.is_seed"
                   if replace else "DO NOTHING")
    return (f"INSERT INTO wardrobe_items ({_ITEM_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
            f"ON CONFLICT (user_id, id) {on_conflict}")


def _item_params(item: Item, user_id: str) -> tuple:
    return (user_id, item.id, item.name, item.category, item.color, item.style, item.warmth, item.formality,
            item.image_path, json.dumps(list(item.tags)), int(item.is_seed), _now())


def upsert_item(item: Item, replace: bool = False, user_id: str = DEFAULT_USER) -> bool:
    """Insert an item. replace=False keeps an existing row (idempotent seed). Returns True if written."""
    return upsert_items([item], user_id, replace) > 0


def upsert_items(items: list[Item], user_id: str = DEFAULT_USER, replace: bool = False) -> int:
    """Bulk insert in one transaction and one batched round trip. Returns rows written."""
    if not items:
        return 0
    with connect() as conn:
        cur = conn.executemany(_upsert_sql(replace), [_item_params(it, user_id) for it in items])
        written = max(cur.rowcount, 0)
    _touch(user_id)
    return written


@_cached
def list_items(user_id: str = DEFAULT_USER) -> list[Item]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM wardrobe_items WHERE user_id = ? ORDER BY category, id",
                            (user_id,)).fetchall()
    return [_row_to_item(r) for r in rows]


@_cached
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
        deleted = conn.execute("DELETE FROM wardrobe_items WHERE user_id = ? AND id = ?",
                               (user_id, item_id)).rowcount > 0
    _touch(user_id)
    return deleted


def delete_non_seed_items(user_id: str = DEFAULT_USER) -> list[str]:
    """Delete a user's own items; returns their image paths so the caller can remove the files."""
    with connect() as conn:
        paths = [r[0] for r in conn.execute(
            "SELECT image_path FROM wardrobe_items WHERE user_id = ? AND is_seed = 0", (user_id,)).fetchall() if r[0]]
        conn.execute("DELETE FROM wardrobe_items WHERE user_id = ? AND is_seed = 0", (user_id,))
    _result = paths
    _touch(user_id)
    return _result


def add_feedback(item_ids: list[str], vote: int, ctx: Context, score: float | None,
                 user_id: str = DEFAULT_USER) -> int:
    with connect() as conn:
        row = conn.execute(
            "INSERT INTO outfit_feedback (user_id, item_ids, vote, occasion, temperature, style, score_at_feedback,"
            " created_at) VALUES (?,?,?,?,?,?,?,?) RETURNING id",
            (user_id, json.dumps(sorted(item_ids)), vote, ctx.occasion, ctx.temperature, ctx.style, score, _now()),
        ).fetchone()
    _touch(user_id)
    return int(row[0])


def list_feedback(limit: int = 50, user_id: str = DEFAULT_USER) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM outfit_feedback WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                            (user_id, limit)).fetchall()
    return [{**dict(r), "item_ids": json.loads(r["item_ids"])} for r in rows]


@_cached
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
        n = conn.execute("DELETE FROM outfit_feedback WHERE user_id = ?", (user_id,)).rowcount
    _touch(user_id)
    return n


@_cached
def disliked_outfits(user_id: str = DEFAULT_USER) -> set[tuple[str, ...]]:
    """Exact outfit combinations this user rejected (net votes < 0)."""
    with connect() as conn:
        rows = conn.execute("SELECT item_ids FROM outfit_feedback WHERE user_id = ? "
                            "GROUP BY item_ids HAVING SUM(vote) < 0", (user_id,)).fetchall()
    return {tuple(json.loads(r["item_ids"])) for r in rows}


@_cached
def liked_outfits(user_id: str, limit: int = 24) -> list[dict]:
    """Outfits with net positive votes, most recently liked first (the user's saved looks)."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT item_ids, MAX(occasion) AS occasion, MAX(temperature) AS temperature, MAX(style) AS style, "
            "MAX(created_at) AS last_at FROM outfit_feedback WHERE user_id = ? GROUP BY item_ids "
            "HAVING SUM(vote) > 0 ORDER BY last_at DESC LIMIT ?", (user_id, limit)).fetchall()
    return [{**dict(r), "item_ids": json.loads(r["item_ids"])} for r in rows]


@_cached
def get_preset(user_id: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT preset FROM user_profile WHERE user_id = ?", (user_id,)).fetchone()
    return row["preset"] if row else None


def set_preset(user_id: str, preset: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO user_profile (user_id, preset) VALUES (?, ?) "
                     "ON CONFLICT(user_id) DO UPDATE SET preset = excluded.preset", (user_id, preset))
    _touch(user_id)


def delete_all_items(user_id: str) -> list[str]:
    """Remove every wardrobe item of a user; returns uploaded image paths to clean up."""
    with connect() as conn:
        paths = [r[0] for r in conn.execute(
            "SELECT image_path FROM wardrobe_items WHERE user_id = ? AND is_seed = 0", (user_id,)).fetchall() if r[0]]
        conn.execute("DELETE FROM wardrobe_items WHERE user_id = ?", (user_id,))
    _result = paths
    _touch(user_id)
    return _result


def try_consume_llm_call(daily_cap: int) -> bool:
    """Atomically reserve one paid LLM call for today (UTC). False once the cap is reached."""
    day = datetime.now(timezone.utc).date().isoformat()
    with connect() as conn:
        conn.execute("INSERT INTO llm_usage (day, calls) VALUES (?, 0) ON CONFLICT (day) DO NOTHING", (day,))
        cur = conn.execute("UPDATE llm_usage SET calls = calls + 1 WHERE day = ? AND calls < ?", (day, daily_cap))
        return cur.rowcount == 1


def llm_calls_today() -> int:
    day = datetime.now(timezone.utc).date().isoformat()
    with connect() as conn:
        row = conn.execute("SELECT calls FROM llm_usage WHERE day = ?", (day,)).fetchone()
    return int(row["calls"]) if row else 0


# ---------- accounts & sessions ----------

def create_account(user_id: str, username: str, pw_hash: str) -> bool:
    """False if the username is taken."""
    with connect() as conn:
        cur = conn.execute("INSERT INTO accounts (user_id, username, pw_hash, created_at) VALUES (?,?,?,?) "
                           "ON CONFLICT (username) DO NOTHING", (user_id, username, pw_hash, _now()))
        return cur.rowcount > 0


def get_account(username: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT user_id, username, pw_hash FROM accounts WHERE username = ?",
                           (username,)).fetchone()
    return dict(row) if row else None


@_cached
def account_name(user_id: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT username FROM accounts WHERE user_id = ?", (user_id,)).fetchone()
    return row["username"] if row else None


def add_session(token_hash: str, user_id: str, expires_at: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?,?,?)",
                     (token_hash, user_id, expires_at))


def session_user(token_hash: str, now_iso: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT user_id FROM sessions WHERE token_hash = ? AND expires_at > ?",
                           (token_hash, now_iso)).fetchone()
    return row["user_id"] if row else None


def delete_session(token_hash: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


# ---------- uploaded photos ----------

def put_blob(path: str, user_id: str, data: bytes) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO blobs (path, user_id, data, created_at) VALUES (?,?,?,?) "
                     "ON CONFLICT (path) DO UPDATE SET data = excluded.data", (path, user_id, data, _now()))


def get_blob(path: str) -> bytes | None:
    with connect() as conn:
        row = conn.execute("SELECT data FROM blobs WHERE path = ?", (path,)).fetchone()
    return bytes(row["data"]) if row else None


def delete_blobs(paths: list[str]) -> None:
    if not paths:
        return
    with connect() as conn:
        conn.execute(f"DELETE FROM blobs WHERE path IN ({','.join('?' * len(paths))})", list(paths))

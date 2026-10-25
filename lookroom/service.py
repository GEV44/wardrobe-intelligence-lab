"""Service layer shared by the Streamlit UI and the FastAPI app. Every call is scoped to a user_id."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from . import db, images, recommender, seed
from .models import DISPLAY_ORDER, Context, Item, is_valid_outfit, validate_item

VOTES = {"like": 1, "dislike": -1}
_USER_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def clean_user_id(user_id: str | None) -> str:
    """Untrusted id (URL/header) -> safe tenant key; anything malformed falls back to the shared demo user."""
    return user_id if user_id and _USER_RE.match(user_id) else db.DEFAULT_USER


def new_user_id() -> str:
    return uuid.uuid4().hex[:12]


USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")
MIN_PASSWORD = 6
SESSION_DAYS = 30


class AuthError(ValueError):
    """Message key (i18n) explaining why sign-up / sign-in failed."""


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt$16384$8$1${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(digest) // 2)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got.hex(), digest)


def register(username: str, password: str, preset: str | None = None) -> str:
    """Create an account with its own sample wardrobe. Returns the new user_id; raises AuthError."""
    username = (username or "").strip().lower()
    if not USERNAME_RE.match(username) or len(password or "") < MIN_PASSWORD:
        raise AuthError("auth_rules")
    db.init_db()
    user_id = new_user_id()
    if not db.create_account(user_id, username, hash_password(password)):
        raise AuthError("auth_taken")
    ensure_ready(user_id, preset)
    return user_id


def authenticate(username: str, password: str) -> str:
    """user_id for valid credentials; raises AuthError (same message for unknown user and wrong password)."""
    db.init_db()
    acc = db.get_account((username or "").strip().lower())
    if not acc or not verify_password(password or "", acc["pw_hash"]):
        raise AuthError("auth_bad")
    return acc["user_id"]


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(user_id: str) -> str:
    """Opaque token for the visitor's URL; only its hash is stored."""
    token = secrets.token_urlsafe(24)
    expires = (datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)).isoformat(timespec="seconds")
    db.add_session(_token_hash(token), user_id, expires)
    return token


def session_user(token: str | None) -> str | None:
    if not token or len(token) > 64:
        return None
    db.init_db()
    return db.session_user(_token_hash(token), datetime.now(timezone.utc).isoformat(timespec="seconds"))


def end_session(token: str | None) -> None:
    if token:
        db.delete_session(_token_hash(token))


def account_name(user_id: str) -> str | None:
    db.init_db()
    return db.account_name(user_id)


def saved_looks(user_id: str) -> list[list[Item]]:
    """Liked outfits whose pieces still exist, newest first."""
    owned = {it.id: it for it in db.list_items(user_id)}
    return [sorted((owned[i] for i in row["item_ids"]), key=lambda it: DISPLAY_ORDER[it.category])
            for row in db.liked_outfits(user_id) if all(i in owned for i in row["item_ids"])]


def ensure_ready(user_id: str = db.DEFAULT_USER, preset: str | None = None) -> None:
    db.init_db()
    seed.ensure_user(user_id, preset)


def current_preset(user_id: str) -> str:
    return db.get_preset(user_id) or seed.DEFAULT_PRESET


def switch_preset(user_id: str, preset: str) -> dict:
    """Replace the user's wardrobe with another sample wardrobe (clears feedback and uploads)."""
    return seed.reset_demo(user_id, preset=preset)


def get_recommendations(ctx: Context, user_id: str = db.DEFAULT_USER, lang: str = "en",
                        skip: set[tuple[str, ...]] | None = None,
                        avoid: list[tuple[str, ...]] | None = None) -> recommender.RecommendationResult:
    """Rejected outfits (disliked, or skipped with "Show another") never come back as-is; looks close to the
    recently rejected ones (`avoid`) are shown only after clearly different ones."""
    exclude = db.disliked_outfits(user_id) | (skip or set())
    return recommender.recommend(db.list_items(user_id), ctx, db.item_preferences(user_id), lang=lang,
                                 exclude=exclude, avoid=avoid)


def score_items(item_ids: list[str], ctx: Context, user_id: str = db.DEFAULT_USER,
                lang: str = "en") -> recommender.Outfit | None:
    """Re-score a specific outfit with current preferences (used for before/after comparisons)."""
    found = db.get_items(item_ids, user_id)
    if len(found) != len(set(item_ids)):
        return None
    items = tuple(sorted(found.values(), key=lambda it: DISPLAY_ORDER[it.category]))
    return recommender.score_outfit(items, ctx, db.item_preferences(user_id), lang)


def record_feedback(item_ids: list[str], vote: str, ctx: Context, user_id: str = db.DEFAULT_USER) -> dict:
    if vote not in VOTES:
        raise ValueError("vote must be 'like' or 'dislike'")
    ids = sorted(set(item_ids))
    if len(ids) != len(item_ids):
        raise ValueError("Duplicate item ids in outfit")
    found = db.get_items(ids, user_id)
    missing = [i for i in ids if i not in found]
    if missing:
        raise LookupError(f"Unknown item ids: {', '.join(missing)}")
    if not is_valid_outfit([found[i].category for i in ids]):
        raise ValueError("An outfit needs one top, bottom and shoes, or one dress and shoes (outerwear optional)")
    before = score_items(ids, ctx, user_id)
    fid = db.add_feedback(ids, VOTES[vote], ctx, before.score if before else None, user_id)
    after = score_items(ids, ctx, user_id)
    return {"feedback_id": fid, "score_before": before.score if before else 0.0,
            "score_after": after.score if after else 0.0}


def feedback_summary(user_id: str = db.DEFAULT_USER) -> dict:
    counts = db.item_vote_counts(user_id)
    items = {it.id: it for it in db.list_items(user_id)}
    rows = []
    for iid, c in counts.items():
        if iid not in items:
            continue
        net = c["likes"] - c["dislikes"]
        rows.append({"item_id": iid, "name": items[iid].name, "style": items[iid].style,
                     "likes": c["likes"], "dislikes": c["dislikes"], "net": net,
                     "signal": round(recommender.preference_signal(net), 3)})
    rows.sort(key=lambda r: (-r["net"], r["item_id"]))
    feedback = db.list_feedback(limit=1000, user_id=user_id)
    style_net: dict[str, int] = {}
    for r in rows:
        style_net[r["style"]] = style_net.get(r["style"], 0) + r["net"]
    return {
        "likes": sum(1 for f in feedback if f["vote"] > 0),
        "dislikes": sum(1 for f in feedback if f["vote"] < 0),
        "items": rows,
        "style_affinity": dict(sorted(style_net.items(), key=lambda kv: (-kv[1], kv[0]))),
        "recent": feedback[:10],
    }


def add_item(name: str, category: str, color: str, style: str, warmth: int, formality: int,
             image_bytes: bytes | None = None, user_id: str = db.DEFAULT_USER) -> Item:
    """Validate and store a user item. Upload is re-encoded; otherwise an illustration is generated."""
    item = validate_item({"id": f"user-{uuid.uuid4().hex[:10]}", "name": name, "category": category,
                          "color": color, "style": style, "warmth": warmth, "formality": formality})
    rel = images.save_upload(image_bytes) if image_bytes else images.ensure_illustration(item)
    item = validate_item({**item.__dict__, "image_path": rel})
    images.persist(rel, user_id)
    db.upsert_item(item, user_id=user_id)
    return item


def _item_vector(item: Item):
    from . import similar
    vec = similar.vector_for(item.id)  # precomputed for catalog / sample items
    if vec is None and item.image_path:
        p = images.resolve_image(item.image_path)
        vec = similar.vector_for(item.id, p.read_bytes()) if p else None  # uploads: embed with the live model
    return vec


def taste_vector(user_id: str = db.DEFAULT_USER):
    """Mean FashionSigLIP embedding of the items this user likes (net votes > 0), or None."""
    import numpy as np
    prefs = db.item_preferences(user_id)
    liked = [it for it in db.list_items(user_id) if prefs.get(it.id, 0) > 0]
    vecs = [v for v in (_item_vector(it) for it in liked) if v is not None]
    if not vecs:
        return None
    m = np.mean(vecs, axis=0)
    return m / np.linalg.norm(m)


def shop_suggestions(ctx: Context, user_id: str = db.DEFAULT_USER, n: int = 6) -> list[dict]:
    from . import shop
    return shop.suggestions(db.list_items(user_id), ctx, db.item_preferences(user_id), n=n,
                            taste=taste_vector(user_id), gender=seed.PRESET_GENDER.get(current_preset(user_id)))


def similar_in_shop(item_id: str, user_id: str = db.DEFAULT_USER, k: int = 4) -> list[dict]:
    """Visually similar real products (same category) for a wardrobe item, via image embeddings."""
    from . import similar
    item = db.get_items([item_id], user_id).get(item_id)
    if item is None:
        raise LookupError(f"Unknown item {item_id}")
    vec = _item_vector(item)
    return [{"product": p, "similarity": round(s, 3)}
            for p, s in similar.similar_products(vec, k=k, category=item.category)]


def add_product(product_id: str, user_id: str = db.DEFAULT_USER) -> Item:
    """'Buy' (demo): copy a real catalog product, with its photo, into the user's wardrobe."""
    from . import shop
    p = next((x for x in shop.load_products() if x.id == product_id), None)
    if p is None:
        raise LookupError(f"Unknown product {product_id}")
    return add_item(p.name[:60], p.category, p.color, p.style, p.warmth, p.formality,
                    p.image_path.read_bytes(), user_id=user_id)


def plan_week(contexts: list[Context], user_id: str = db.DEFAULT_USER, lang: str = "en"):
    return recommender.plan_week(db.list_items(user_id), contexts, db.item_preferences(user_id), lang)


def remove_item(item_id: str, user_id: str = db.DEFAULT_USER) -> bool:
    found = db.get_items([item_id], user_id).get(item_id)
    ok = db.delete_item(item_id, user_id)
    if ok and found and not found.is_seed and found.image_path:
        images.delete_uploads([found.image_path])
    return ok


def tag_image(data: bytes) -> dict:
    """Suggest item attributes for an upload. Route: FashionSigLIP embedding + fine-tuned linear-probe heads
    -> zero-shot vision model -> color heuristic.

    Suggestions only: the user confirms them in the form, so a model mistake never silently enters the wardrobe.
    """
    import time

    from . import probe, vision

    if len(data) > images.MAX_UPLOAD_BYTES:
        raise images.ImageValidationError("Image is larger than 5 MB.")
    tagger = vision.get_tagger()
    if tagger is not None and hasattr(tagger, "embed") and probe.available():
        try:
            t0 = time.perf_counter()
            r = probe.predict(tagger.embed(data))
            warmth, formality = probe.GARMENT_ATTRS.get(r["garment"], (3, 3))
            return {"source": "probe", "model": r["model"], "category": r["category"], "garment": r["garment"],
                    "color": r["color"], "style": r["style"], "warmth": warmth, "formality": formality,
                    "suggested_name": f"{r['color'].title()} {r['garment'].title()}",
                    "confidence": r["confidence"], "latency_ms": int((time.perf_counter() - t0) * 1000)}
        except Exception:
            pass  # fall through to zero-shot
    if tagger is not None:
        try:
            r = tagger.tag(data)
            return {"source": "fashionclip", "model": getattr(r, "model", vision.REPO_ID),
                    "category": r.category, "garment": r.garment, "color": r.color,
                    "style": r.style, "warmth": r.warmth, "formality": r.formality,
                    "suggested_name": f"{r.color.title()} {r.garment.title()}",
                    "confidence": r.confidence, "latency_ms": r.latency_ms}
        except Exception:
            pass  # corrupt image or runtime error -> heuristic below
    color = images.dominant_color(data)
    if color is None:
        raise images.ImageValidationError("File is not a valid image.")
    return {"source": "heuristic", "model": "pillow-color-quantization", "category": None, "garment": None,
            "color": color, "style": None, "warmth": None, "formality": None, "suggested_name": None,
            "confidence": {}, "latency_ms": 0}


def bgremove_online() -> bool:
    """BiRefNet costs ~45 s and ~4 GB RAM per image on 2 CPU threads: only on request paths when explicitly enabled."""
    return os.environ.get("LOOKROOM_BGREMOVE_ONLINE") == "1"


def prepare_upload(data: bytes) -> tuple[bytes, bool]:
    """Cut the garment out onto white. Routed by latency budget:
    fast classical cut-out (plain backdrops, ~50 ms) -> BiRefNet-lite (only if enabled online) -> original photo.
    Never raises for model problems."""
    if len(data) > images.MAX_UPLOAD_BYTES:
        raise images.ImageValidationError("Image is larger than 5 MB.")
    cleaned = images.flat_background_cutout(data)
    if cleaned is None and bgremove_online():
        try:
            from . import bgremove
            cleaned = bgremove.remove_background(data)
        except Exception:
            cleaned = None
    return (cleaned, True) if cleaned else (data, False)


def ai_status() -> dict:
    from . import stylist, vision
    try:
        from . import bgremove
        bg = bgremove.status()
    except Exception as exc:
        bg = {"available": False, "loaded": False, "detail": f"not installed ({type(exc).__name__})"}
    from . import llm
    return {"vision": vision.status(), "bgremove": bg, "llm": stylist.llm_available(), "local_llm": llm.status()}

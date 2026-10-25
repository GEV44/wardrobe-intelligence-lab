"""Visual similarity search: FashionSigLIP image embeddings + cosine nearest neighbours over the shop catalog.

Catalog and sample-wardrobe embeddings are precomputed offline (scripts/build_embeddings.py, ~0.9 MB float16);
user uploads are embedded on demand. Brute-force cosine over ~600 vectors takes microseconds; at catalog
scale this becomes a vector index (pgvector / FAISS), with the same interface.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import shop

EMB_PATH = shop.SHOP_DIR / "embeddings.npz"


@lru_cache(maxsize=1)
def _index() -> tuple[list[str], np.ndarray] | None:
    if not EMB_PATH.is_file():
        return None
    with np.load(EMB_PATH) as z:
        return [str(i) for i in z["ids"]], z["emb"].astype(np.float32)


def available() -> bool:
    return _index() is not None


def vector_for(item_id: str, image_bytes: bytes | None = None) -> np.ndarray | None:
    """Precomputed vector for catalog/sample items; otherwise embed the given image with the live model."""
    idx = _index()
    if idx and item_id in idx[0]:
        return idx[1][idx[0].index(item_id)]
    if image_bytes is None:
        return None
    from . import vision
    tagger = vision.get_tagger()
    if tagger is None or not hasattr(tagger, "embed"):
        return None
    try:
        return tagger.embed(image_bytes)
    except Exception:
        return None


def similar_products(vec: np.ndarray, k: int = 4, category: str | None = None,
                     exclude: set[str] | None = None) -> list[tuple[shop.Product, float]]:
    idx = _index()
    if idx is None or vec is None:
        return []
    ids, emb = idx
    products = {p.id: p for p in shop.load_products()}
    sims = emb @ vec
    out = []
    for j in np.argsort(-sims):
        pid = ids[j]
        p = products.get(pid)
        if p is None or (exclude and pid in exclude) or (category and p.category != category):
            continue
        out.append((p, float(sims[j])))
        if len(out) >= k:
            break
    return out

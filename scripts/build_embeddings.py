"""Precompute FashionSigLIP image embeddings for the shop catalog + sample wardrobe photos.

Output: data/shop/embeddings.npz  {ids: str[N], emb: float16[N, 768]} (L2-normalized).
Usage: python scripts/build_embeddings.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lookroom import seed, shop, vision  # noqa: E402

if __name__ == "__main__":
    vision.ensure_siglip()
    tagger = vision.get_tagger()
    assert tagger is not None and hasattr(tagger, "embed"), "FashionSigLIP is required (scripts/setup_models.py)"
    jobs = [(p.id, p.image_path) for p in shop.load_products()]
    jobs += [(f.stem, f) for f in sorted(seed.CATALOG_DIR.glob("*.jpg"))]
    ids, vecs, t0 = [], [], time.time()
    for n, (pid, path) in enumerate(jobs, 1):
        vecs.append(tagger.embed(path.read_bytes()).astype(np.float16))
        ids.append(pid)
        if n % 100 == 0:
            print(f"[..] {n}/{len(jobs)} ({time.time() - t0:.0f}s)")
    np.savez_compressed(shop.SHOP_DIR / "embeddings.npz", ids=np.array(ids), emb=np.stack(vecs))
    print(f"[ok] {len(ids)} embeddings -> {shop.SHOP_DIR / 'embeddings.npz'} in {time.time() - t0:.0f}s")

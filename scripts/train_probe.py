"""Train linear-probe heads (garment, color, style) on frozen Marqo FashionSigLIP image embeddings.

Data: Fashion Product Images (benitomartin/fashion-product-images-small-900x1200) shards 0-4 from the HF cache.
Compares against the current zero-shot tagger on the same held-out test split and writes
  data/models/probe/probe.npz      (W, b per head + class labels, float32)
  data/models/probe/metrics.json   (all measured numbers)
Embeddings are cached in runtime/probe_cache/ so reruns are instant.

Usage: LOOKROOM_VISION_BACKEND=siglip python scripts/train_probe.py [--shards 0,1,2,3,4] [--cap 250] [--seed 7]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOOKROOM_VISION_BACKEND", "siglip")

REPO = "benitomartin/fashion-product-images-small-900x1200"
CACHE = ROOT / "runtime" / "probe_cache"
OUT = ROOT / "data" / "models" / "probe"
EMB_MODEL = "Marqo/marqo-fashionSigLIP (int8 ONNX vision tower)"

# dataset articleType -> garment label (garment determines category)
ARTICLE_GARMENT = {
    "Tshirts": "t-shirt", "Shirts": "shirt", "Tops": "top", "Sweaters": "sweater", "Sweatshirts": "sweatshirt",
    "Jackets": "jacket", "Blazers": "blazer", "Trousers": "trousers", "Jeans": "jeans", "Track Pants": "track pants",
    "Shorts": "shorts", "Skirts": "skirt", "Dresses": "dress", "Jumpsuit": "dress", "Casual Shoes": "casual shoes",
    "Formal Shoes": "formal shoes", "Sports Shoes": "sports shoes", "Heels": "heels", "Flats": "flats",
    "Sandals": "sandals", "Flip Flops": "sandals",
}
GARMENT_CATEGORY = {
    "t-shirt": "top", "shirt": "top", "top": "top", "sweater": "top", "sweatshirt": "top",
    "jacket": "outerwear", "blazer": "outerwear",
    "trousers": "bottom", "jeans": "bottom", "track pants": "bottom", "shorts": "bottom", "skirt": "bottom",
    "dress": "dress",
    "casual shoes": "shoes", "formal shoes": "shoes", "sports shoes": "shoes", "heels": "shoes", "flats": "shoes",
    "sandals": "shoes",
}
# scripts/build_shop.py COLORS, extended with a few clearly-unambiguous extras; purple/orange/silver/gold/multi dropped
COLORS = {
    "White": "white", "Black": "black", "Grey": "grey", "Grey Melange": "grey", "Charcoal": "grey", "Steel": "grey",
    "Navy Blue": "navy", "Blue": "blue", "Teal": "blue", "Turquoise Blue": "blue",
    "Brown": "brown", "Coffee Brown": "brown", "Mushroom Brown": "brown", "Tan": "camel",
    "Beige": "beige", "Khaki": "beige", "Skin": "beige", "Nude": "beige", "Taupe": "beige",
    "Cream": "cream", "Off White": "cream", "Olive": "olive", "Green": "green", "Sea Green": "green",
    "Lime Green": "green", "Red": "red", "Maroon": "burgundy", "Burgundy": "burgundy",
    "Pink": "pink", "Rose": "pink", "Magenta": "pink", "Yellow": "yellow", "Mustard": "yellow",
}
USAGE_STYLE = {"Formal": "classic", "Smart Casual": "smart", "Sports": "streetwear", "Casual": "casual"}
# zero-shot garment label -> acceptable probe-space garments (lenient, documented in metrics.json)
ZS_GARMENT_OK = {
    "t-shirt": {"t-shirt"}, "polo shirt": {"t-shirt", "shirt"}, "button-down shirt": {"shirt"},
    "blouse": {"top", "shirt"}, "knit sweater": {"sweater"}, "hoodie": {"sweatshirt"}, "tank top": {"top"},
    "jeans": {"jeans"}, "tailored trousers": {"trousers"}, "chinos": {"trousers"}, "shorts": {"shorts"},
    "skirt": {"skirt"}, "sneakers": {"casual shoes", "sports shoes"}, "leather boots": {"casual shoes", "formal shoes"},
    "loafers": {"formal shoes", "casual shoes"}, "high heels": {"heels"}, "sandals": {"sandals"},
    "wool coat": {"jacket", "blazer"}, "denim jacket": {"jacket"}, "blazer": {"blazer", "jacket"},
    "puffer jacket": {"jacket"},
}
MIN_CLASS = 10  # garment classes with fewer samples in the cached shards are dropped (only 'blazer': 1 sample)


def load_meta(shards):
    """Yield metadata for usable rows (no image bytes); bytes are re-read lazily per shard."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rows = []
    for i in shards:
        path = hf_hub_download(REPO, f"data/train-{i:05d}-of-00014.parquet", repo_type="dataset")
        t = pq.read_table(path, columns=["id", "gender", "articleType", "baseColour", "usage", "productDisplayName"])
        for j, r in enumerate(t.to_pylist()):
            g = ARTICLE_GARMENT.get(r["articleType"])
            name = r["productDisplayName"] or ""
            if not g or r["gender"] in ("Boys", "Girls") or any(w in name for w in ("Kids", "Boys", "Girls", "Infant")):
                continue
            rows.append({"id": int(r["id"]), "shard": i, "row": j, "garment": g,
                         "color": COLORS.get(r["baseColour"] or ""), "style": USAGE_STYLE.get(r["usage"] or "")})
    return rows


def sample(rows, cap, seed):
    by = defaultdict(list)
    for r in rows:
        by[r["garment"]].append(r)
    counts = Counter({g: len(v) for g, v in by.items()})
    rng = random.Random(seed)
    picked, dropped = [], []
    for g, v in sorted(by.items()):
        if len(v) < MIN_CLASS:
            dropped.append(g)
            continue
        v = sorted(v, key=lambda r: r["id"])
        rng.shuffle(v)
        picked += v[:cap]
    return picked, counts, dropped


def split(picked, seed):
    """Stratified 80/20 by garment (per-class shuffle, fixed seed)."""
    rng = random.Random(seed + 1)
    by = defaultdict(list)
    for r in picked:
        by[r["garment"]].append(r)
    train, test = [], []
    for g, v in sorted(by.items()):
        v = sorted(v, key=lambda r: r["id"])
        rng.shuffle(v)
        k = max(1, round(0.2 * len(v)))
        test += v[:k]
        train += v[k:]
    return train, test


def fetch_bytes(items, shards_needed):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    want = defaultdict(set)
    for r in items:
        want[r["shard"]].add(r["row"])
    for s, rowset in sorted(want.items()):
        path = hf_hub_download(REPO, f"data/train-{s:05d}-of-00014.parquet", repo_type="dataset")
        tbl = pq.read_table(path, columns=["id", "image"])
        ids = tbl.column("id").to_pylist()
        imgs = tbl.column("image")
        for row in sorted(rowset):
            yield s, row, ids[row], imgs[row].as_py()["bytes"]


def embed_all(items, tagger, log_every=50):
    """Embeddings cached by product id in runtime/probe_cache/embeddings.npz."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "embeddings.npz"
    cache = {}
    if f.is_file():
        with np.load(f) as z:
            cache = dict(zip(z["ids"].tolist(), z["emb"]))
    todo = [r for r in items if r["id"] not in cache]
    print(f"[embed] {len(items)} wanted, {len(cache)} cached, {len(todo)} to compute", flush=True)
    t0, n = time.time(), 0
    for s, row, pid, data in fetch_bytes(todo, None):
        cache[int(pid)] = tagger.embed(data)
        n += 1
        if n % log_every == 0:
            rate = (time.time() - t0) / n
            print(f"[embed] {n}/{len(todo)}  {rate:.2f}s/img  eta {(len(todo) - n) * rate / 60:.1f} min", flush=True)
        if n % 200 == 0:
            _save_emb(f, cache)
    if todo:
        _save_emb(f, cache)
    return np.stack([cache[r["id"]] for r in items]).astype(np.float32)


def _save_emb(f, cache):
    ids = np.array(sorted(cache), dtype=np.int64)
    np.savez(f, ids=ids, emb=np.stack([cache[i] for i in ids]).astype(np.float32))


def zeroshot_all(items, tagger):
    """Run the current zero-shot tagger .tag() on the items; cached by id in zeroshot.json."""
    f = CACHE / "zeroshot.json"
    cache = json.loads(f.read_text()) if f.is_file() else {}
    todo = [r for r in items if str(r["id"]) not in cache]
    print(f"[zeroshot] {len(items)} wanted, {len(cache)} cached, {len(todo)} to compute", flush=True)
    n = 0
    for s, row, pid, data in fetch_bytes(todo, None):
        t = tagger.tag(data)
        cache[str(pid)] = {"category": t.category, "garment": t.garment, "color": t.color, "style": t.style}
        n += 1
        if n % 50 == 0:
            print(f"[zeroshot] {n}/{len(todo)}", flush=True)
    if todo:
        f.write_text(json.dumps(cache))
    return [cache[str(r["id"])] for r in items]


def fit_head(X, y, seed, metric="acc"):
    """Pick C / class_weight on a validation split carved from train, then refit on all train."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    strat = y if counts.min() >= 2 else None
    Xa, Xv, ya, yv = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
    best = (-1.0, None)
    for cw in (None, "balanced"):
        for C in (1, 3, 10, 30, 100, 300, 1000):
            m = LogisticRegression(C=C, class_weight=cw, max_iter=3000).fit(Xa, ya)
            p = m.predict(Xv)
            score = accuracy_score(yv, p) if metric == "acc" else f1_score(yv, p, average="macro")
            if score > best[0]:
                best = (score, (C, cw))
    C, cw = best[1]
    m = LogisticRegression(C=C, class_weight=cw, max_iter=3000).fit(X, y)
    return m, {"C": C, "class_weight": cw, "val_" + ("acc" if metric == "acc" else "macro_f1"): round(best[0], 4)}


def acc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((a == b).mean()) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="0,1,2,3,4")
    ap.add_argument("--cap", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-zeroshot", action="store_true")
    a = ap.parse_args()
    shards = [int(s) for s in a.shards.split(",")]

    from lookroom import vision
    from sklearn.metrics import f1_score

    rows = load_meta(shards)
    picked, avail, dropped = sample(rows, a.cap, a.seed)
    train, test = split(picked, a.seed)
    print(f"[data] usable {len(rows)}, sampled {len(picked)} (train {len(train)} / test {len(test)}), dropped classes {dropped}")
    print("[data] per-garment sampled:", dict(Counter(r["garment"] for r in picked)))

    assert vision.ensure_siglip(), "SigLIP model not available"
    tagger = vision.get_tagger()
    assert tagger is not None and hasattr(tagger, "embed"), "SigLIP tagger required (LOOKROOM_VISION_BACKEND=siglip)"

    Xtr, Xte = embed_all(train, tagger), embed_all(test, tagger)

    heads, info, preds = {}, {}, {}
    ytr_g = [r["garment"] for r in train]
    m, info["garment"] = fit_head(Xtr, ytr_g, a.seed, "acc")
    heads["garment"] = m
    for name, metric in (("color", "acc"), ("style", "f1")):
        idx = [i for i, r in enumerate(train) if r[name]]
        m, info[name] = fit_head(Xtr[idx], [train[i][name] for i in idx], a.seed, metric)
        info[name]["n_train_labeled"] = len(idx)
        heads[name] = m

    # ---- evaluate probe
    te_g = [r["garment"] for r in test]
    p_g = heads["garment"].predict(Xte)
    te_cat = [GARMENT_CATEGORY[g] for g in te_g]
    p_cat = [GARMENT_CATEGORY[g] for g in p_g]
    c_idx = [i for i, r in enumerate(test) if r["color"]]
    s_idx = [i for i, r in enumerate(test) if r["style"]]
    p_col = heads["color"].predict(Xte)
    p_sty = heads["style"].predict(Xte)

    probe = {
        "category": acc(te_cat, p_cat), "garment": acc(te_g, p_g),
        "garment_macro_f1": float(f1_score(te_g, p_g, average="macro")),
        "color": acc([test[i]["color"] for i in c_idx], p_col[c_idx]),
        "style": acc([test[i]["style"] for i in s_idx], p_sty[s_idx]),
        "style_macro_f1": float(f1_score([test[i]["style"] for i in s_idx], p_sty[s_idx], average="macro")),
    }
    cats = sorted(set(te_cat))
    probe_cat_pc = {c: acc([t for t in te_cat if t == c], [p for t, p in zip(te_cat, p_cat) if t == c]) for c in cats}

    # ---- zero-shot on the same images
    zs = None
    if not a.skip_zeroshot:
        z = zeroshot_all(test, tagger)
        z_g_ok = [z[i]["garment"] in ZS_GARMENT_OK and te_g[i] in ZS_GARMENT_OK[z[i]["garment"]] for i in range(len(test))]
        z_cat = [x["category"] for x in z]
        nd = [i for i, c in enumerate(te_cat) if c != "dress"]
        zs = {
            "category": acc(te_cat, z_cat),
            "category_excluding_dress": acc([te_cat[i] for i in nd], [z_cat[i] for i in nd]),
            "garment_lenient": float(np.mean(z_g_ok)),
            "color": acc([test[i]["color"] for i in c_idx], [z[i]["color"] for i in c_idx]),
            "style": acc([test[i]["style"] for i in s_idx], [z[i]["style"] for i in s_idx]),
        }
        zs["category_per_class"] = {c: acc([t for t in te_cat if t == c],
                                           [p for t, p in zip(te_cat, z_cat) if t == c]) for c in cats}
        probe["category_excluding_dress"] = acc([te_cat[i] for i in nd], [p_cat[i] for i in nd])
        probe["garment_lenient_same_rule_n/a"] = None
        probe.pop("garment_lenient_same_rule_n/a")

    probe["category_per_class"] = probe_cat_pc
    per_g = {g: acc([t for t in te_g if t == g], [p for t, p in zip(te_g, p_g) if t == g]) for g in sorted(set(te_g))}

    # ---- save weights + metrics
    OUT.mkdir(parents=True, exist_ok=True)
    arrs = {}
    for name, m in heads.items():
        arrs[f"{name}_W"] = m.coef_.astype(np.float32)
        arrs[f"{name}_b"] = m.intercept_.astype(np.float32)
        arrs[f"{name}_classes"] = np.array(m.classes_, dtype="U32")
    arrs["garment_category"] = np.array([GARMENT_CATEGORY[g] for g in heads["garment"].classes_], dtype="U16")
    arrs["n_train"] = np.array(len(train))
    np.savez_compressed(OUT / "probe.npz", **arrs)

    metrics = {
        "date": datetime.date.today().isoformat(), "embedding_model": EMB_MODEL, "seed": a.seed, "cap_per_garment": a.cap,
        "n_total": len(picked), "n_train": len(train), "n_test": len(test),
        "n_test_color_labeled": len(c_idx), "n_test_style_labeled": len(s_idx),
        "train_per_garment": dict(Counter(ytr_g)), "test_per_garment": dict(Counter(te_g)),
        "test_per_category": dict(Counter(te_cat)),
        "style_classes_trained": list(heads["style"].classes_),
        "style_note": "'minimal' has no dataset label; style head trained on 4 labeled classes only",
        "dropped_garment_classes": dropped, "available_per_garment_in_shards": dict(avail),
        "hyperparameters": {k: {kk: (vv if not isinstance(vv, np.generic) else vv.item()) for kk, vv in v.items()}
                            for k, v in info.items()},
        "probe_test": probe, "zeroshot_test": zs, "probe_garment_per_class_acc": per_g,
        "zeroshot_garment_rule": "lenient: zero-shot label counted correct if it maps into acceptable probe-space garments "
                                 "(ZS_GARMENT_OK); zero-shot vocabulary has no dress class and no formal/casual shoe split",
        "probe_npz_bytes": (OUT / "probe.npz").stat().st_size,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"probe": probe, "zeroshot": zs}, indent=2))
    print(f"[done] wrote {OUT} ({metrics['probe_npz_bytes']} bytes)")


if __name__ == "__main__":
    main()

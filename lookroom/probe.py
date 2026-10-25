"""Linear-probe heads on frozen FashionSigLIP embeddings (trained by scripts/train_probe.py).

Pure numpy at inference. Never raises on missing files: `available()` is simply False.
predict(emb) -> {category, garment, color, style, confidence{garment,color,style}, model}
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

PROBE_FILE = Path(__file__).resolve().parent.parent / "data" / "models" / "probe" / "probe.npz"
_HEADS = ("garment", "color", "style")
# Probe garment vocabulary -> (default warmth, default formality), same 1-5 scales as vision.GARMENTS.
GARMENT_ATTRS: dict[str, tuple[int, int]] = {
    "t-shirt": (2, 1), "shirt": (3, 4), "top": (2, 2), "sweater": (4, 3), "sweatshirt": (4, 1),
    "jeans": (2, 1), "trousers": (3, 4), "track pants": (2, 1), "shorts": (1, 1), "skirt": (2, 3),
    "dress": (2, 4), "casual shoes": (2, 2), "sports shoes": (2, 1), "formal shoes": (3, 5),
    "heels": (2, 5), "flats": (2, 3), "sandals": (1, 1), "jacket": (4, 3),
}
_lock = threading.Lock()
_state: dict = {}  # "data": loaded dict | None (load attempted)


def _load() -> dict | None:
    with _lock:
        if "data" not in _state:
            try:
                with np.load(PROBE_FILE, allow_pickle=False) as z:
                    d = {h: (z[f"{h}_W"].astype(np.float32), z[f"{h}_b"].astype(np.float32),
                             [str(c) for c in z[f"{h}_classes"]]) for h in _HEADS}
                    d["garment_category"] = [str(c) for c in z["garment_category"]]
                    d["n_train"] = int(z["n_train"])
                _state["data"] = d
            except Exception:
                _state["data"] = None
        return _state["data"]


def available() -> bool:
    return _load() is not None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(emb: np.ndarray) -> dict:
    """emb: 768-d L2-normalized FashionSigLIP image embedding (vision.SigLipTagger.embed)."""
    d = _load()
    if d is None:
        raise RuntimeError("probe not available; check probe.available() first")
    x = np.asarray(emb, dtype=np.float32).reshape(-1)
    out, conf = {}, {}
    for h in _HEADS:
        W, b, classes = d[h]
        p = _softmax(W @ x + b)
        i = int(p.argmax())
        out[h] = classes[i]
        conf[h] = round(float(p[i]), 3)
        if h == "garment":
            out["category"] = d["garment_category"][i]
    return {
        "category": out["category"], "garment": out["garment"], "color": out["color"], "style": out["style"],
        "confidence": conf,
        "model": f"FashionSigLIP + linear probe (fine-tuned on {d['n_train']} labeled products)",
    }

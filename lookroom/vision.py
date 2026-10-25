"""Optional zero-shot garment tagger (ONNX Runtime, CPU, int8-quantized). No torch, no remote code.

Backends, in `auto` order: Marqo/marqo-fashionSigLIP (Apache-2.0, int8 vision ONNX + text embeddings
precomputed once by setup in fp32, since int8 text lost accuracy) then patrickjohncyh/fashion-clip
(MIT, single int8 ONNX). Force one with
LOOKROOM_VISION_BACKEND=auto|siglip|fashionclip. Setup: python scripts/setup_models.py
Degrades gracefully: if no model or onnxruntime is missing, get_tagger() returns None and the
app falls back to the Pillow color heuristic + manual form fields.
"""
from __future__ import annotations

import io
import os
import string
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .db import PROJECT_ROOT

REPO_ID = "patrickjohncyh/fashion-clip"  # FashionCLIP (fallback backend)
REVISION = "main"
MODEL_DIR = Path(os.environ.get("LOOKROOM_MODEL_DIR", PROJECT_ROOT / "models" / "fashion-clip"))
MODEL_FILE = "model.int8.onnx"
SIGLIP_REPO_ID = "Marqo/marqo-fashionSigLIP"  # Apache-2.0
SIGLIP_DIR = Path(os.environ.get("LOOKROOM_SIGLIP_DIR", PROJECT_ROOT / "models" / "fashion-siglip"))
SIGLIP_FILES = ("vision_model_int8.onnx", "text_embeds.npz")  # npz: prompts + L2-normalized embeddings
SIGLIP_SCALE = 100.0  # cosine -> logits scale (model card) before the per-group softmax
BACKENDS = ("siglip", "fashionclip")  # auto preference order

# Garment vocabulary -> (category, default warmth, default formality). Zero-shot labels.
GARMENTS: dict[str, tuple[str, int, int]] = {
    "t-shirt": ("top", 2, 1), "polo shirt": ("top", 2, 2), "button-down shirt": ("top", 3, 4),
    "blouse": ("top", 2, 3), "knit sweater": ("top", 4, 3), "hoodie": ("top", 4, 1),
    "tank top": ("top", 1, 1), "jeans": ("bottom", 2, 1), "tailored trousers": ("bottom", 3, 4),
    "chinos": ("bottom", 3, 3), "shorts": ("bottom", 1, 1), "skirt": ("bottom", 2, 3),
    "sneakers": ("shoes", 2, 2), "leather boots": ("shoes", 4, 3), "loafers": ("shoes", 3, 4),
    "high heels": ("shoes", 2, 5), "sandals": ("shoes", 1, 1), "wool coat": ("outerwear", 5, 4),
    "denim jacket": ("outerwear", 3, 2), "blazer": ("outerwear", 3, 5), "puffer jacket": ("outerwear", 5, 1),
}
COLOR_NAMES = ("white", "black", "grey", "cream", "beige", "camel", "brown", "navy", "blue", "olive",
               "green", "red", "burgundy", "pink", "yellow")
STYLE_PROMPTS = {
    "casual": "a casual everyday outfit piece", "classic": "a classic timeless elegant piece",
    "minimal": "a minimalist clean simple piece", "smart": "a smart business piece",
    "streetwear": "an urban streetwear piece",
}
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SIGLIP_MEAN = SIGLIP_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIZE = 224
_PUNCT = str.maketrans("", "", string.punctuation)


@dataclass
class TagResult:
    category: str
    garment: str
    color: str
    style: str
    warmth: int
    formality: int
    confidence: dict[str, float] = field(default_factory=dict)
    latency_ms: int = 0
    model: str = REPO_ID


def preprocess(data: bytes, mean: np.ndarray = CLIP_MEAN, std: np.ndarray = CLIP_STD, squash: bool = False) -> np.ndarray:
    """Image -> NCHW float32. White-composites alpha. CLIP: resize shortest side + center crop; SigLIP: squash."""
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, "white")
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        w, h = img.size
        if squash:
            img = img.resize((SIZE, SIZE), Image.BICUBIC)
        else:
            s = SIZE / min(w, h)
            img = img.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
            w, h = img.size
            left, top = (w - SIZE) // 2, (h - SIZE) // 2
            img = img.crop((left, top, left + SIZE, top + SIZE))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - mean) / std
    return arr.transpose(2, 0, 1)[None, ...]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _prompts() -> tuple[dict[str, list[str]], list[str]]:
    """Label groups and the flat prompt list (garment, color, style) shared by every backend."""
    groups = {"garment": list(GARMENTS), "color": list(COLOR_NAMES), "style": list(STYLE_PROMPTS)}
    prompts = ([f"a photo of {g}" for g in GARMENTS] + [f"a photo of a {c} garment" for c in COLOR_NAMES]
               + [f"a photo of {p}" for p in STYLE_PROMPTS.values()])
    return groups, prompts


def siglip_prompts() -> list[str]:
    """Prompts as SigLIP saw them in training (open_clip canonicalize: lowercase, no punctuation)."""
    return [" ".join(p.translate(_PUNCT).lower().split()) for p in _prompts()[1]]


def _session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class _Tagger:
    """Shared zero-shot logic: subclasses set `model`, `groups` and implement `_logits(data)`."""

    model: str
    groups: dict[str, list[str]]

    def _logits(self, data: bytes) -> np.ndarray:
        raise NotImplementedError

    def tag(self, data: bytes) -> TagResult:
        t0 = time.perf_counter()
        logits = self._logits(data)
        res, offset = {}, 0
        for name, labels in self.groups.items():
            probs = _softmax(logits[offset: offset + len(labels)])
            offset += len(labels)
            i = int(probs.argmax())
            res[name] = (labels[i], float(probs[i]))
        garment = res["garment"][0]
        cat, warmth, formality = GARMENTS[garment]
        return TagResult(
            category=cat, garment=garment, color=res["color"][0], style=res["style"][0],
            warmth=warmth, formality=formality,
            confidence={k: round(v[1], 3) for k, v in res.items()},
            latency_ms=int((time.perf_counter() - t0) * 1000), model=self.model,
        )


class FashionClipTagger(_Tagger):
    """patrickjohncyh/fashion-clip: one joint ONNX graph (image + all prompts per call)."""

    model = REPO_ID

    def __init__(self, model_dir: Path | None = None):
        model_dir = model_dir or MODEL_DIR
        from tokenizers import Tokenizer

        self.session = _session(model_dir / MODEL_FILE)
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tok.enable_truncation(77)
        tok.enable_padding(pad_id=tok.token_to_id("<|endoftext|>") or 0, pad_token="<|endoftext|>")
        self.groups, prompts = _prompts()
        enc = tok.encode_batch(prompts)
        self.input_ids = np.array([e.ids for e in enc], dtype=np.int64)
        self.attention_mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _logits(self, data: bytes) -> np.ndarray:
        feeds = {"input_ids": self.input_ids, "pixel_values": preprocess(data), "attention_mask": self.attention_mask}
        return self.session.run(["logits_per_image"], {k: v for k, v in feeds.items() if k in self.input_names})[0][0]


class SigLipTagger(_Tagger):
    """Marqo/marqo-fashionSigLIP: int8 vision ONNX; text embeddings come precomputed (setup_models.py)."""

    model = SIGLIP_REPO_ID

    def __init__(self, model_dir: Path | None = None):
        model_dir = model_dir or SIGLIP_DIR
        self.vision = _session(model_dir / "vision_model_int8.onnx")
        self.groups = _prompts()[0]
        with np.load(model_dir / "text_embeds.npz") as z:
            if [str(p) for p in z["prompts"]] != siglip_prompts():
                raise ValueError("text_embeds.npz is stale (label vocabulary changed); rerun scripts/setup_models.py")
            self.text_emb = z["emb"].astype(np.float32)

    def embed(self, data: bytes) -> np.ndarray:
        """L2-normalized 768-d image embedding (for visual similarity search)."""
        x = preprocess(data, SIGLIP_MEAN, SIGLIP_STD, squash=True)
        img = self.vision.run(["image_embeds"], {"pixel_values": x})[0][0].astype(np.float32)
        return img / np.linalg.norm(img)

    def _logits(self, data: bytes) -> np.ndarray:
        return SIGLIP_SCALE * (self.text_emb @ self.embed(data))


_lock = threading.Lock()
_taggers: dict[str, _Tagger] = {}
_errors: dict[str, str] = {}


def _installed(backend: str) -> bool:
    if backend == "siglip":
        return all((SIGLIP_DIR / f).is_file() for f in SIGLIP_FILES)
    return (MODEL_DIR / MODEL_FILE).is_file() and (MODEL_DIR / "tokenizer.json").is_file()


def _candidates() -> list[str]:
    """Installed backends to try, in order. LOOKROOM_VISION_BACKEND=siglip|fashionclip forces one (no fallback)."""
    choice = os.environ.get("LOOKROOM_VISION_BACKEND", "auto").strip().lower()
    return [b for b in (BACKENDS if choice not in BACKENDS else (choice,)) if _installed(b)]


def model_installed() -> bool:
    return bool(_candidates())


BUNDLED_SIGLIP_EMBEDS = PROJECT_ROOT / "data" / "models" / "fashion-siglip" / "text_embeds.npz"


def ensure_siglip() -> bool:
    """Cloud bootstrap (no Docker build step): fetch the int8 vision ONNX from the Hub once and use the
    text embeddings bundled in the repo. Idempotent, thread-safe, never raises. LOOKROOM_AUTO_DOWNLOAD=0 disables."""
    if _installed("siglip"):
        return True
    if os.environ.get("LOOKROOM_AUTO_DOWNLOAD", "1") == "0":
        return False
    with _lock:
        try:
            import shutil

            from huggingface_hub import hf_hub_download

            SIGLIP_DIR.mkdir(parents=True, exist_ok=True)
            if not (SIGLIP_DIR / "text_embeds.npz").is_file() and BUNDLED_SIGLIP_EMBEDS.is_file():
                shutil.copyfile(BUNDLED_SIGLIP_EMBEDS, SIGLIP_DIR / "text_embeds.npz")
            dst = SIGLIP_DIR / "vision_model_int8.onnx"
            if not dst.is_file():
                tmp = dst.with_suffix(".part")
                shutil.copyfile(hf_hub_download(SIGLIP_REPO_ID, "onnx/vision_model_int8.onnx", revision=REVISION), tmp)
                tmp.replace(dst)  # atomic: a half-written file is never mistaken for a model
            _errors.pop("siglip", None)
        except Exception as exc:
            _errors.setdefault("siglip-download", f"{type(exc).__name__}: {exc}")
            return False
    return _installed("siglip")


def get_tagger() -> _Tagger | None:
    """Lazy, thread-safe singleton per backend. Returns None (never raises) when no model is usable."""
    if os.environ.get("LOOKROOM_DISABLE_VISION") == "1":
        return None
    with _lock:
        for b in _candidates():
            if b in _errors:
                continue
            if b not in _taggers:
                try:
                    _taggers[b] = SigLipTagger() if b == "siglip" else FashionClipTagger()
                except Exception as exc:  # missing onnxruntime, corrupt file, low memory...
                    _errors[b] = f"{type(exc).__name__}: {exc}"
                    continue
            return _taggers[b]
    return None


def status() -> dict:
    if os.environ.get("LOOKROOM_DISABLE_VISION") == "1":
        return {"available": False, "loaded": False, "detail": "disabled by LOOKROOM_DISABLE_VISION"}
    cands = _candidates()
    if not cands:
        detail = _errors.get("siglip-download") or "model not installed yet (downloads on first start)"
        return {"available": False, "loaded": False, "detail": detail}
    loaded = next((b for b in cands if b in _taggers), None)
    errs = "; ".join(f"{b}: {_errors[b]}" for b in cands if b in _errors)
    detail = _taggers[loaded].model if loaded else (errs or f"{cands[0]} int8 ONNX")
    return {"available": any(b not in _errors for b in cands), "loaded": loaded is not None, "detail": detail}

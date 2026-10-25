"""Optional background removal for garment photos (BiRefNet_lite, ONNX Runtime, CPU).

Model: ZhengPeng7/BiRefNet_lite (MIT). ONNX export: wonfy/BiRefNet_lite-ONNX-webgpu (MIT), a weights-unchanged,
bit-identical graph rewrite of onnx-community/BiRefNet_lite-ONNX (MIT). Setup: python scripts/setup_bgremove.py
The graph has a fixed 1024x1024 input (smaller sizes fail inside the decoder), ~245 MB fp32, several GB of peak RAM
and tens of seconds per image on 2 vCPUs: call it from a background/offline step, not a request handler.

Never raises to callers: remove_background() returns None on any failure so the original image is kept.
Env: LOOKROOM_DISABLE_BGREMOVE=1 disables; LOOKROOM_BGREMOVE_DIR overrides the model folder;
LOOKROOM_BGREMOVE_THREADS overrides intra-op threads (default min(4, cpu_count)).
"""
from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .db import PROJECT_ROOT

REPO_ID = "ZhengPeng7/BiRefNet_lite"
EXPORT_REPO_ID = "wonfy/BiRefNet_lite-ONNX-webgpu"
EXPORT_FILE = "onnx/model.onnx"
MODEL_FILE = "model.onnx"
INPUT_SIZE = 1024  # fixed by the exported graph
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_FG, MAX_FG = 0.02, 0.98  # mask foreground fraction outside this range = degenerate -> None
PAD = 0.06

_lock = threading.Lock()  # guards load
_run_lock = threading.Lock()  # one inference at a time (multi-GB activations)
_session = None
_error: str | None = None  # load error
_error_last: str | None = None  # last per-image error


def model_dir() -> Path:
    return Path(os.environ.get("LOOKROOM_BGREMOVE_DIR", PROJECT_ROOT / "models" / "birefnet-lite"))


def _disabled() -> bool:
    return os.environ.get("LOOKROOM_DISABLE_BGREMOVE", "").strip().lower() in ("1", "true", "yes")


def _threads() -> int:
    try:
        n = int(os.environ.get("LOOKROOM_BGREMOVE_THREADS", ""))
        if n > 0:
            return n
    except ValueError:
        pass
    return max(1, min(4, os.cpu_count() or 1))


def _get_session():
    """Lazy thread-safe singleton; stores the load error string instead of raising."""
    global _session, _error
    if _session is not None or _error is not None:
        return _session
    with _lock:
        if _session is not None or _error is not None:
            return _session
        try:
            import onnxruntime as ort

            path = model_dir() / MODEL_FILE
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run python scripts/setup_bgremove.py")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = _threads()
            opts.enable_cpu_mem_arena = False  # hand big activations back to the OS between images
            opts.enable_mem_pattern = False
            opts.log_severity_level = 3
            _session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            _error = f"{type(exc).__name__}: {exc}"
            _session = None
    return _session


def available() -> bool:
    if _disabled():
        return False
    if _session is not None:
        return True
    if _error is not None:
        return False
    try:
        import onnxruntime  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return (model_dir() / MODEL_FILE).exists()


def status() -> dict:
    if _disabled():
        detail = "disabled via LOOKROOM_DISABLE_BGREMOVE"
    elif _error is not None:
        detail = _error
    elif _session is not None:
        detail = f"last error: {_error_last}" if _error_last else f"loaded ({REPO_ID}, {_threads()} threads)"
    elif available():
        detail = "model present, not loaded yet"
    else:
        detail = f"model not installed in {model_dir()} (python scripts/setup_bgremove.py) or onnxruntime missing"
    return {"available": available(), "loaded": _session is not None, "detail": detail}


def _alpha_mask(img: Image.Image) -> np.ndarray | None:
    """Foreground probability in [0,1] at img.size (H, W), or None."""
    sess = _get_session()
    if sess is None:
        return None
    small = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    x = (np.asarray(small, dtype=np.float32) / 255.0 - MEAN) / STD
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...], dtype=np.float32)
    with _run_lock:
        logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    logits = np.clip(np.asarray(logits, dtype=np.float32).reshape(INPUT_SIZE, INPUT_SIZE), -30, 30)
    prob = 1.0 / (1.0 + np.exp(-logits))
    if not np.isfinite(prob).all():
        return None
    m = Image.fromarray((prob * 255.0 + 0.5).astype(np.uint8), "L").resize(img.size, Image.BILINEAR)
    return np.asarray(m, dtype=np.float32) / 255.0


def remove_background(data: bytes, max_side: int = 1024) -> bytes | None:
    """Garment photo bytes -> PNG bytes (RGB): cut-out on pure white, cropped to the subject with ~6% padding.

    Returns None on any failure or degenerate mask so callers keep the original image.
    """
    if _disabled() or not data:
        return None
    try:
        if not available():
            return None
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, "white")
                im = Image.alpha_composite(bg, im)
            img = im.convert("RGB")
        if max_side and max(img.size) > max_side:
            s = max_side / max(img.size)
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
        alpha = _alpha_mask(img)
        if alpha is None:
            return None
        binary = alpha > 0.5
        frac = float(binary.mean())
        if frac < MIN_FG or frac > MAX_FG:
            return None
        ys, xs = np.where(binary)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        pad = round(PAD * max(y1 - y0, x1 - x0))
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1, x1 = min(img.height, y1 + pad), min(img.width, x1 + pad)
        rgb = np.asarray(img, dtype=np.float32)[y0:y1, x0:x1]
        a = alpha[y0:y1, x0:x1, None]
        out = np.clip(rgb * a + 255.0 * (1.0 - a) + 0.5, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(out, "RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001 - never raise to callers
        global _error_last
        _error_last = f"{type(exc).__name__}: {exc}"
        return None

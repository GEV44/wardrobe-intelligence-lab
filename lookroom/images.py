"""Safe upload handling and deterministic sample clothing illustrations (Pillow only)."""
from __future__ import annotations

import io
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError

from . import db
from .models import COLOR_HEX, Item

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 20_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
THUMB_SIZE = (1024, 1024)
CANVAS = 320
BACKGROUND = "#F4F1EC"


class ImageValidationError(ValueError):
    pass


def images_dir(kind: str) -> Path:
    d = db.data_dir() / "images" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_image(rel_path: str | None) -> Path | None:
    """Map a stored relative path to a file inside the data dir; reject anything escaping it."""
    if not rel_path:
        return None
    root = (db.data_dir() / "images").resolve()
    p = (db.data_dir() / rel_path).resolve()
    if not p.is_relative_to(root):
        return None
    if not p.is_file():
        _restore(rel_path, p)
    return p if p.is_file() else None


def _restore(rel_path: str, dst: Path) -> None:
    """A fresh container has no local copies: bring back bundled catalog photos and stored uploads."""
    try:
        kind, name = rel_path.split("/")[1], Path(rel_path).name
        src = db.PROJECT_ROOT / "data" / "catalog" / name if kind == "catalog" else None
        data = src.read_bytes() if src is not None and src.is_file() else (
            db.get_blob(rel_path) if kind in ("uploads", "seed") else None)
        if data:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
    except Exception:  # a missing photo must never break a page
        pass


def persist(rel_path: str, user_id: str) -> None:
    """Keep a user's photo in the database (remote backend), so it survives container restarts."""
    if db.backend() == "postgres" and (p := resolve_image(rel_path)):
        db.put_blob(rel_path, user_id, p.read_bytes())


# ---------- uploads ----------

def save_upload(data: bytes) -> str:
    """Validate untrusted bytes, re-encode to PNG under a UUID name. Returns path relative to data dir."""
    if not data:
        raise ImageValidationError("Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("Image is larger than 5 MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            if fmt not in ALLOWED_FORMATS:
                raise ImageValidationError("Only JPEG, PNG or WebP images are accepted.")
            if probe.width * probe.height > MAX_PIXELS:
                raise ImageValidationError("Image dimensions are too large.")
            probe.verify()
        with Image.open(io.BytesIO(data)) as img:  # verify() invalidates the object; reopen to decode
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail(THUMB_SIZE)
            name = f"{uuid.uuid4().hex}.png"
            img.save(images_dir("uploads") / name, format="PNG")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("File is not a valid image.") from exc
    return f"images/uploads/{name}"


def dominant_color(data: bytes) -> str | None:
    """Lightweight CV heuristic: most frequent quantized color of the image center -> nearest palette name."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            w, h = img.size
            img = img.crop((w // 5, h // 5, w - w // 5, h - h // 5)).resize((64, 64))
            q = img.quantize(colors=4)
            pal = q.getpalette()
            count, idx = max(q.getcolors())
            rgb = tuple(pal[idx * 3: idx * 3 + 3])
    except Exception:
        return None
    return nearest_color_name(rgb)


def nearest_color_name(rgb: tuple[int, int, int]) -> str:
    def dist(hexv: str) -> int:
        c = tuple(int(hexv[i:i + 2], 16) for i in (1, 3, 5))
        return sum((a - b) ** 2 for a, b in zip(rgb, c))
    return min(COLOR_HEX, key=lambda name: dist(COLOR_HEX[name]))


# ---------- generated illustrations ----------

def _darker(hexv: str, f: float = 0.65) -> tuple[int, int, int]:
    return tuple(int(int(hexv[i:i + 2], 16) * f) for i in (1, 3, 5))


def render_illustration(item: Item) -> Image.Image:
    img = Image.new("RGB", (CANVAS, CANVAS), BACKGROUND)
    d = ImageDraw.Draw(img)
    fill = COLOR_HEX.get(item.color, "#999999")
    line = _darker(fill) if item.color != "black" else (90, 90, 90)
    kw = {"fill": fill, "outline": line, "width": 4}
    if item.category == "top":
        d.polygon([(110, 60), (210, 60), (280, 110), (255, 150), (225, 130), (225, 270),
                   (95, 270), (95, 130), (65, 150), (40, 110)], **kw)
        d.arc((135, 45, 185, 85), 0, 180, fill=line, width=4)
    elif item.category == "bottom":
        d.polygon([(100, 50), (220, 50), (240, 280), (175, 280), (160, 120), (145, 280), (80, 280)], **kw)
        d.line([(100, 72), (220, 72)], fill=line, width=3)
    elif item.category == "shoes":
        for dx in (0, 120):
            d.rounded_rectangle((40 + dx, 165, 160 + dx, 215), radius=18, **kw)
            d.polygon([(40 + dx, 190), (40 + dx, 125), (95 + dx, 125), (110 + dx, 170)], **kw)
            d.line([(40 + dx, 205), (160 + dx, 205)], fill=line, width=3)
    else:  # outerwear
        d.polygon([(105, 45), (215, 45), (290, 105), (260, 160), (230, 135), (230, 295),
                   (90, 295), (90, 135), (60, 160), (30, 105)], **kw)
        d.line([(160, 70), (160, 295)], fill=line, width=4)
        d.polygon([(105, 45), (160, 130), (130, 45)], fill=line)
        d.polygon([(215, 45), (160, 130), (190, 45)], fill=line)
        for y in (160, 200, 240):
            d.ellipse((168, y, 178, y + 10), fill=line)
    return img


def ensure_illustration(item: Item) -> str:
    rel = f"images/seed/{item.id}.png"
    path = db.data_dir() / rel
    if not path.is_file():
        images_dir("seed")
        render_illustration(item).save(path, format="PNG")
    return rel


def seed_image(item: Item, catalog_dir: Path) -> str:
    """Real catalog photo for a sample item if bundled (data/catalog/<id>.jpg|png), else a generated illustration."""
    for ext in ("jpg", "png", "webp"):
        src = catalog_dir / f"{item.id}.{ext}"
        if src.is_file():
            rel = f"images/catalog/{item.id}.{ext}"
            dst = db.data_dir() / rel
            if not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
                images_dir("catalog")
                dst.write_bytes(src.read_bytes())
            return rel
    return ensure_illustration(item)


def flat_background_cutout(data: bytes, max_side: int = 900, tol: float = 38.0) -> bytes | None:
    """Fast classical cut-out (~50 ms) for product-style photos on a plain backdrop.

    Estimates the backdrop colour from the image border, flood-fills everything border-connected that is close
    to it, and composites the remaining garment onto pure white, cropped with padding. Returns None when the
    border is not uniform (busy background) or the mask is degenerate, so callers can route to a heavier model.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    except Exception:
        return None
    arr = np.asarray(im, dtype=np.float32)
    h, w, _ = arr.shape
    b = max(2, int(min(h, w) * 0.03))
    border = np.concatenate([arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3),
                             arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    if (np.linalg.norm(border - bg, axis=1) < tol).mean() < 0.85:
        return None  # busy background: not safe for the classical path
    near = (np.linalg.norm(arr - bg, axis=2) < tol).astype(np.uint8) * 255
    mask_img = Image.fromarray(near, "L").copy()  # copy: fromarray shares a read-only buffer, floodfill would be lost
    step = max(8, min(h, w) // 40)
    seeds = [(x, y) for x in range(0, w, step) for y in (0, h - 1)] + \
            [(x, y) for y in range(0, h, step) for x in (0, w - 1)]
    for s in seeds:
        if mask_img.getpixel(s) == 255:
            ImageDraw.floodfill(mask_img, s, 128)
    fg = (np.asarray(mask_img) != 128)
    frac = fg.mean()
    if not 0.03 < frac < 0.95:
        return None
    alpha = Image.fromarray((fg * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1.2))
    white = Image.new("RGB", im.size, "white")
    out = Image.composite(im, white, alpha)
    ys, xs = np.nonzero(fg)
    pad = int(max(h, w) * 0.06)
    out = out.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + pad), min(h, ys.max() + pad)))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


BOARD_BG = (243, 239, 233)


def _trim(img: Image.Image, pad: int = 8) -> Image.Image:
    """Crop the near-white studio margins so the garment, not the background, fills its cell."""
    ys, xs = np.nonzero(np.asarray(img.convert("L")) < 243)
    if xs.size < 64:
        return img
    return img.crop((max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0),
                     min(int(xs.max()) + pad + 1, img.width), min(int(ys.max()) + pad + 1, img.height)))


def _fit(img: Image.Image, box: tuple[int, int]) -> Image.Image:
    img = _trim(img)
    scale = min(box[0] / img.width, box[1] / img.height, 2.0)  # upscale small photos, at most 2x
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS)


def compose_board(items: list[Item], width: int = 1280) -> str | None:
    """Flat-lay 'outfit board': each piece on a white card over a warm backdrop. Cached by content."""
    import hashlib

    from PIL import ImageFilter

    paths = [resolve_image(i.image_path) for i in items]
    if not items or any(p is None for p in paths):
        return None
    sig = "|".join(f"{p}:{p.stat().st_mtime_ns}" for p in paths)  # type: ignore[union-attr]
    name = hashlib.sha1(f"{sig}|{width}|v3".encode()).hexdigest()[:16] + ".jpg"  # bump vN when the layout changes
    out = images_dir("boards") / name
    if out.is_file():
        return f"images/boards/{name}"
    h = int(width * 0.78)
    pad, gap = int(width * 0.035), int(width * 0.025)
    canvas = Image.new("RGB", (width, h), BOARD_BG)
    by_cat = {i.category: p for i, p in zip(items, paths)}
    inner_w, inner_h = width - 2 * pad, h - 2 * pad
    if "dress" in by_cat:          # dress tall on the left; shoes (+ outerwear) on the right
        lw = int(inner_w * 0.5)
        rw = inner_w - lw - gap
        right = [c for c in ("outerwear", "shoes") if c in by_cat]
        rh = (inner_h - gap * (len(right) - 1)) // len(right)
        cells = {"dress": (pad, pad, lw, inner_h)}
        for k, cat in enumerate(right):
            cells[cat] = (pad + lw + gap, pad + k * (rh + gap), rw, rh)
    elif "outerwear" in by_cat:    # coat tall on the left, top/bottom/shoes stacked on the right
        lw = int(inner_w * 0.42)
        rw = inner_w - lw - gap
        rh = (inner_h - 2 * gap) // 3
        cells = {"outerwear": (pad, pad, lw, inner_h)}
        for k, cat in enumerate(("top", "bottom", "shoes")):
            cells[cat] = (pad + lw + gap, pad + k * (rh + gap), rw, rh)
    else:                          # top tall on the left, bottom + shoes on the right
        lw = int(inner_w * 0.5)
        rw = inner_w - lw - gap
        rh = (inner_h - gap) // 2
        cells = {"top": (pad, pad, lw, inner_h), "bottom": (pad + lw + gap, pad, rw, rh),
                 "shoes": (pad + lw + gap, pad + rh + gap, rw, rh)}
    shadow = Image.new("L", canvas.size, 0)
    sd = ImageDraw.Draw(shadow)
    for x, y, w, hh in cells.values():
        sd.rounded_rectangle((x + 4, y + 8, x + w + 4, y + hh + 8), radius=22, fill=70)
    canvas.paste((205, 198, 188), mask=shadow.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(canvas)
    for cat, (x, y, w, hh) in cells.items():
        d.rounded_rectangle((x, y, x + w, y + hh), radius=22, fill="white")
        with Image.open(by_cat[cat]) as src:
            img = _trim(ImageOps.exif_transpose(src).convert("RGB"))
        if hh < w and img.height > 1.5 * img.width:  # full-body model shot in a short cell: show the garment
            if cat == "top":
                img = img.crop((0, 0, img.width, int(img.height * 0.58)))
            elif cat == "bottom":
                img = img.crop((0, int(img.height * 0.40), img.width, img.height))
        img = _fit(img, (int(w * 0.86), int(hh * 0.86)))
        canvas.paste(img, (x + (w - img.width) // 2, y + (hh - img.height) // 2))
    canvas.save(out, format="JPEG", quality=92)
    return f"images/boards/{name}"


def shop_image(src: Path, product_id: str) -> str | None:
    """Expose a shop catalog photo inside the data dir so outfit boards can use it. Returns its relative path."""
    if not src.is_file():
        return None
    rel = f"images/shop/{product_id}.jpg"
    dst = db.data_dir() / rel
    if not dst.is_file():
        images_dir("shop")
        dst.write_bytes(src.read_bytes())
    return rel


def delete_uploads(rel_paths: list[str]) -> int:
    """Delete specific uploaded files (only inside images/uploads)."""
    n = 0
    root = images_dir("uploads").resolve()
    for rel in rel_paths:
        p = (db.data_dir() / rel).resolve()
        if p.is_relative_to(root) and p.is_file():
            p.unlink(missing_ok=True)
            n += 1
    if db.backend() == "postgres":
        db.delete_blobs([r for r in rel_paths if r.startswith(("images/uploads/", "images/seed/user-"))])
    return n

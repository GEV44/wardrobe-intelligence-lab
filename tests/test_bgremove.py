import io
from pathlib import Path

import pytest
from PIL import Image

from lookroom import bgremove

ROOT = Path(__file__).resolve().parent.parent
SNEAKERS = ROOT / "demo_assets" / "sneakers_public_domain.jpg"
import os

# BiRefNet takes ~45 s/image on CPU: run its model tests only when explicitly requested (RUN_SLOW=1).
needs_model = pytest.mark.skipif(not bgremove.available() or os.environ.get("RUN_SLOW") != "1",
                                 reason="BiRefNet not installed or RUN_SLOW!=1")


def test_garbage_bytes_return_none():
    assert bgremove.remove_background(b"definitely not an image") is None
    assert bgremove.remove_background(b"") is None


def test_disabled_via_env_returns_none(monkeypatch):
    monkeypatch.setenv("LOOKROOM_DISABLE_BGREMOVE", "1")
    assert bgremove.available() is False
    assert bgremove.status()["available"] is False
    assert bgremove.remove_background(SNEAKERS.read_bytes()) is None


def test_missing_model_dir_degrades(monkeypatch, tmp_path):
    monkeypatch.setenv("LOOKROOM_BGREMOVE_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(bgremove, "_session", None)  # other tests/modules may have loaded the singleton
    monkeypatch.setattr(bgremove, "_error", None)
    assert bgremove.available() is False
    assert bgremove.remove_background(SNEAKERS.read_bytes()) is None
    assert set(bgremove.status()) == {"available", "loaded", "detail"}


@needs_model
def test_sneakers_cutout_white_corners_and_cropped():
    data = SNEAKERS.read_bytes()
    out = bgremove.remove_background(data)
    assert out is not None
    img = Image.open(io.BytesIO(out))
    img.load()
    assert img.format == "PNG" and img.mode == "RGB"
    orig = Image.open(io.BytesIO(data))
    assert img.width <= orig.width and img.height <= orig.height
    w, h = img.size
    for xy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        assert min(img.getpixel(xy)) >= 235, (xy, img.getpixel(xy))


@needs_model
def test_uniform_image_does_not_raise():
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (200, 30, 30)).save(buf, format="PNG")
    out = bgremove.remove_background(buf.getvalue())
    assert out is None or Image.open(io.BytesIO(out)).mode == "RGB"

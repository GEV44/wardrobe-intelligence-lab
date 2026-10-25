import io
from pathlib import Path

import pytest
from PIL import Image

from lookroom import db, images, service, vision

SNEAKERS = Path(__file__).resolve().parent.parent / "demo_assets" / "sneakers_public_domain.jpg"
needs_model = pytest.mark.skipif(not vision.model_installed(), reason="run scripts/setup_models.py first")
needs_siglip = pytest.mark.skipif(not vision._installed("siglip"), reason="run scripts/setup_models.py first")
needs_clip = pytest.mark.skipif(not vision._installed("fashionclip"), reason="run scripts/setup_models.py first")


@pytest.fixture
def backend(monkeypatch):
    """Force a backend via LOOKROOM_VISION_BACKEND with a clean tagger cache; returns the tagger (or None)."""
    def force(name: str):
        monkeypatch.setenv("LOOKROOM_VISION_BACKEND", name)
        monkeypatch.setattr(vision, "_taggers", {})
        monkeypatch.setattr(vision, "_errors", {})
        return vision.get_tagger()
    return force


def test_preprocess_shape_and_alpha():
    buf = io.BytesIO()
    Image.new("RGBA", (300, 120), (0, 0, 0, 0)).save(buf, format="PNG")
    x = vision.preprocess(buf.getvalue())
    assert x.shape == (1, 3, 224, 224) and x.dtype.name == "float32"
    # fully transparent -> composited on white, so every pixel equals normalized white per channel
    white = (1.0 - vision.CLIP_MEAN) / vision.CLIP_STD
    assert abs(x[0] - white[:, None, None]).max() < 1e-2


@needs_clip
def test_fashionclip_tags_real_photo(backend):
    r = backend("fashionclip").tag(SNEAKERS.read_bytes())
    assert (r.category, r.color) == ("shoes", "white") and r.model == vision.REPO_ID
    assert set(r.confidence) == {"garment", "color", "style"} and r.latency_ms < 5000


@needs_clip
def test_fashionclip_categories_on_seed_illustrations(seeded, backend):
    tagger = backend("fashionclip")
    items = [i for i in db.list_items() if i.category != "outerwear"]  # flat drawings of coats read as shirts
    hits = sum(tagger.tag((seeded / i.image_path).read_bytes()).category == i.category for i in items)
    assert hits == len(items)


@needs_siglip
def test_siglip_tags_real_photo(backend):
    r = backend("siglip").tag(SNEAKERS.read_bytes())
    assert (r.category, r.garment, r.color) == ("shoes", "sneakers", "white") and r.model == vision.SIGLIP_REPO_ID
    assert set(r.confidence) == {"garment", "color", "style"} and all(0 < v <= 1 for v in r.confidence.values())


@needs_siglip
def test_siglip_categories_on_seed_illustrations(seeded, backend):
    tagger = backend("siglip")
    items = [i for i in db.list_items() if i.category != "outerwear"]
    hits = sum(tagger.tag((seeded / i.image_path).read_bytes()).category == i.category for i in items)
    assert hits == len(items)


@needs_siglip
def test_siglip_text_embeddings_cached_and_normalized(backend):
    t = backend("siglip")
    assert t.text_emb.shape == (len(vision.siglip_prompts()), 768)
    assert abs(((t.text_emb ** 2).sum(axis=1) - 1) < 1e-4).all()


def test_siglip_preprocess_squashes_without_crop():
    buf = io.BytesIO()
    img = Image.new("RGB", (400, 100), (255, 255, 255))
    img.paste((0, 0, 0), (0, 0, 20, 100))  # black strip at the far left edge would be cropped away by CLIP crop
    img.save(buf, format="PNG")
    x = vision.preprocess(buf.getvalue(), vision.SIGLIP_MEAN, vision.SIGLIP_STD, squash=True)
    assert x.shape == (1, 3, 224, 224) and x.min() == pytest.approx(-1.0) and x[0, 0, 0, 0] == pytest.approx(-1.0)


@needs_siglip
@needs_clip
def test_backend_env_selects_model(backend, monkeypatch):
    assert backend("siglip").model == vision.SIGLIP_REPO_ID
    assert backend("fashionclip").model == vision.REPO_ID
    assert backend("auto").model == vision.SIGLIP_REPO_ID  # primary in auto order
    assert backend("nonsense").model == vision.SIGLIP_REPO_ID  # unknown value behaves as auto
    assert vision.status()["detail"] == vision.SIGLIP_REPO_ID


def test_forced_missing_backend_has_no_fallback(backend, monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "SIGLIP_DIR", tmp_path)  # empty -> not installed
    assert backend("siglip") is None and not vision.model_installed()
    assert vision.status()["available"] is False


@needs_clip
def test_auto_falls_back_to_fashionclip_when_siglip_broken(backend, monkeypatch, tmp_path):
    (tmp_path / "vision_model_int8.onnx").write_bytes(b"not an onnx file")
    (tmp_path / "text_embeds.npz").write_bytes(b"")
    monkeypatch.setattr(vision, "SIGLIP_DIR", tmp_path)
    assert backend("auto").model == vision.REPO_ID
    assert "siglip" in vision._errors


def test_tag_image_falls_back_to_heuristic(monkeypatch):
    monkeypatch.setenv("LOOKROOM_DISABLE_VISION", "1")
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (74, 120, 181)).save(buf, format="PNG")
    t = service.tag_image(buf.getvalue())
    assert t["source"] == "heuristic" and t["color"] == "blue" and t["category"] is None


def test_tag_image_rejects_garbage():
    with pytest.raises(images.ImageValidationError):
        service.tag_image(b"definitely not an image")

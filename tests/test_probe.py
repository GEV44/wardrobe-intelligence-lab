from pathlib import Path

import pytest

from lookroom import probe, vision

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = {"top", "bottom", "shoes", "outerwear", "dress"}
COLORS = set(vision.COLOR_NAMES)
STYLES = {"casual", "classic", "smart", "streetwear", "minimal"}

pytestmark = pytest.mark.skipif(not probe.available(), reason="probe.npz not trained (scripts/train_probe.py)")


@pytest.fixture(scope="module")
def emb():
    import os

    os.environ["LOOKROOM_VISION_BACKEND"] = "siglip"
    if not vision.ensure_siglip():
        pytest.skip("SigLIP model not available")
    tagger = vision.get_tagger()
    if tagger is None or not hasattr(tagger, "embed"):
        pytest.skip("SigLIP tagger unavailable")
    return tagger.embed((ROOT / "demo_assets" / "sneakers_public_domain.jpg").read_bytes())


def test_output_shape_and_domains(emb):
    r = probe.predict(emb)
    assert {"category", "garment", "color", "style", "confidence", "model"} <= set(r)
    assert r["category"] in CATEGORIES
    assert r["color"] in COLORS
    assert r["style"] in STYLES
    assert set(r["confidence"]) == {"garment", "color", "style"}
    assert all(0.0 < v <= 1.0 for v in r["confidence"].values())
    assert r["model"].startswith("FashionSigLIP + linear probe")


def test_sneakers_are_shoes(emb):
    r = probe.predict(emb)
    assert r["category"] == "shoes", r


def test_upload_tagging_uses_probe_and_recognises_dresses():
    from lookroom import service, vision
    if not (probe.available() and vision._installed("siglip")):
        pytest.skip("needs FashionSigLIP (scripts/setup_models.py) and probe.npz")
    root = Path(__file__).resolve().parent.parent
    dress = service.tag_image((root / "data" / "catalog" / "w-dress-black.jpg").read_bytes())
    assert dress["source"] == "probe" and dress["category"] == "dress" and dress["warmth"] and dress["formality"]
    shoes = service.tag_image((root / "demo_assets" / "sneakers_public_domain.jpg").read_bytes())
    assert shoes["category"] == "shoes" and shoes["color"] == "white"

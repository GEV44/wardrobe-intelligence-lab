import io
import re

import pytest
from PIL import Image

from lookroom import images, service


def _img_bytes(fmt="PNG", size=(64, 64), color=(74, 120, 181)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_valid_upload_is_reencoded_with_uuid_name(isolated_data, fmt):
    rel = images.save_upload(_img_bytes(fmt))
    assert re.fullmatch(r"images/uploads/[0-9a-f]{32}\.png", rel)
    with Image.open(isolated_data / rel) as img:
        assert img.format == "PNG"


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50])
def test_invalid_bytes_rejected(isolated_data, data):
    with pytest.raises(images.ImageValidationError):
        images.save_upload(data)


def test_disallowed_format_rejected(isolated_data):
    with pytest.raises(images.ImageValidationError):
        images.save_upload(_img_bytes("GIF"))


def test_oversize_rejected(isolated_data):
    with pytest.raises(images.ImageValidationError, match="5 MB"):
        images.save_upload(b"0" * (images.MAX_UPLOAD_BYTES + 1))


def test_path_traversal_not_resolved(isolated_data):
    assert images.resolve_image("../../etc/passwd") is None
    assert images.resolve_image("images/../lookroom.db") is None


def test_flat_background_cutout_product_shot():
    from pathlib import Path
    data = (Path(__file__).resolve().parent.parent / "demo_assets" / "sneakers_public_domain.jpg").read_bytes()
    out = images.flat_background_cutout(data)
    assert out is not None
    with Image.open(io.BytesIO(out)) as im:
        im = im.convert("RGB")
        assert im.getpixel((0, 0)) == (255, 255, 255) and im.getpixel((im.width - 1, im.height - 1)) == (255, 255, 255)


def test_flat_background_cutout_rejects_busy_or_empty():
    import numpy as np
    noise = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(noise).save(buf, format="PNG")
    assert images.flat_background_cutout(buf.getvalue()) is None          # busy background -> route elsewhere
    assert images.flat_background_cutout(_img_bytes(color=(250, 250, 250))) is None  # nothing in the frame
    assert images.flat_background_cutout(b"garbage") is None


def test_board_composition(seeded):
    from lookroom import db
    items = [i for i in db.list_items() if i.category in ("top", "bottom", "shoes")][::5][:3]
    by_cat = {}
    for i in db.list_items():
        by_cat.setdefault(i.category, i)
    rel = images.compose_board([by_cat["top"], by_cat["bottom"], by_cat["shoes"], by_cat["outerwear"]])
    assert rel and images.resolve_image(rel)
    assert images.compose_board([by_cat["top"], by_cat["bottom"], by_cat["shoes"]]) != rel
    assert items is not None


def test_dominant_color_detects_blue():
    assert images.dominant_color(_img_bytes()) == "blue"


def test_add_item_with_upload(seeded):
    item = service.add_item("Blue Tee", "top", "blue", "casual", 2, 1, _img_bytes())
    assert item.image_path.startswith("images/uploads/") and images.resolve_image(item.image_path)

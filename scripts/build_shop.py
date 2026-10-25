"""Build the real-photo sample wardrobe + the shop catalog from the Fashion Product Images dataset.

Source: Kaggle "Fashion Product Images" by Param Aggarwal (44k Myntra catalog products, listed as MIT), via the
Hugging Face mirror benitomartin/fashion-product-images-small-900x1200 (parquet). Provenance caveat in RESEARCH.md.

Outputs
- data/catalog/<wardrobe_item_id>.jpg : photos for the 14 sample wardrobe items (FashionCLIP picks the best match)
- data/shop/products.json + data/shop/img/<id>.jpg : a balanced men's catalog for "shop the gap" suggestions
Usage: python scripts/build_shop.py [--shards 0,1] [--per-bucket 6]
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_catalog import Scorer  # noqa: E402

REPO = "benitomartin/fashion-product-images-small-900x1200"
SHOP = ROOT / "data" / "shop"
CATALOG = ROOT / "data" / "catalog"

# articleType -> (category, warmth, formality, default style)
ARTICLES = {
    "Shirts": ("top", 3, 4, "smart"), "Tshirts": ("top", 2, 1, "casual"), "Sweaters": ("top", 4, 3, "minimal"),
    "Sweatshirts": ("top", 4, 1, "streetwear"), "Jackets": ("outerwear", 4, 2, "casual"),
    "Blazers": ("outerwear", 3, 5, "classic"), "Trousers": ("bottom", 3, 4, "smart"), "Jeans": ("bottom", 2, 1, "casual"),
    "Track Pants": ("bottom", 2, 1, "streetwear"), "Shorts": ("bottom", 1, 1, "casual"),
    "Casual Shoes": ("shoes", 2, 2, "casual"), "Formal Shoes": ("shoes", 3, 4, "classic"),
    "Sports Shoes": ("shoes", 2, 1, "streetwear"), "Sandals": ("shoes", 1, 1, "casual"),
}
COLORS = {
    "White": "white", "Black": "black", "Grey": "grey", "Grey Melange": "grey", "Charcoal": "grey",
    "Navy Blue": "navy", "Blue": "blue", "Teal": "blue", "Brown": "brown", "Coffee Brown": "brown", "Tan": "camel",
    "Beige": "beige", "Khaki": "beige", "Cream": "cream", "Off White": "cream", "Olive": "olive", "Green": "green",
    "Red": "red", "Maroon": "burgundy", "Burgundy": "burgundy", "Pink": "pink", "Yellow": "yellow", "Mustard": "yellow",
}
USAGE_STYLE = {"Formal": "classic", "Smart Casual": "smart", "Casual": None, "Sports": "streetwear"}

# Sample wardrobe slots -> (articleTypes, colors, usage preference, FashionCLIP target prompt)
SLOTS = {  # slot id -> (articleTypes, colors, preferred usage, CLIP prompt, optional name keyword)
    "top-white-oxford": (["Shirts"], ["white"], "Formal", "a white formal button-down shirt", None),
    "top-black-tee": (["Tshirts"], ["black"], "Casual", "a black polo t-shirt", None),
    "top-charcoal-turtleneck": (["Sweaters"], ["grey"], None, "a charcoal turtleneck sweater", None),
    "top-blue-buttondown": (["Shirts"], ["blue"], None, "a light blue button-down shirt", None),
    "top-grey-hoodie": (["Sweatshirts"], ["grey"], None, "a grey hooded sweatshirt", "Hood"),
    "bottom-navy-trousers": (["Trousers"], ["navy"], "Formal", "navy formal trousers", None),
    "bottom-black-jeans": (["Jeans"], ["black"], None, "black jeans", None),
    "bottom-beige-chinos": (["Trousers"], ["beige", "cream"], None, "beige chino trousers", None),
    "bottom-lightblue-jeans": (["Jeans"], ["blue"], None, "light blue denim jeans", None),
    "shoes-white-sneakers": (["Casual Shoes", "Sports Shoes"], ["white"], None, "white sneakers", None),
    "shoes-black-formal": (["Formal Shoes"], ["black"], None, "black leather formal shoes", None),
    "shoes-brown-boots": (["Casual Shoes", "Formal Shoes"], ["brown", "camel"], None, "brown leather ankle boots", None),
    "outer-black-jacket": (["Blazers", "Jackets"], ["black"], "Casual", "a plain black jacket", None),
    "outer-olive-jacket": (["Jackets"], ["olive", "green"], "Casual", "an olive casual jacket", None),
}


def load_rows(shards: list[int]) -> list[dict]:
    rows = []
    for i in shards:
        path = hf_hub_download(REPO, f"data/train-{i:05d}-of-00014.parquet", repo_type="dataset")
        table = pq.read_table(path, columns=["id", "gender", "articleType", "baseColour", "usage",
                                             "productDisplayName", "image"])
        for r in table.to_pylist():
            if r["gender"] not in ("Men", "Unisex") or r["articleType"] not in ARTICLES:
                continue
            color = COLORS.get(r["baseColour"] or "")
            if not color or not r.get("image") or not r["image"].get("bytes"):
                continue
            cat, warmth, formality, style = ARTICLES[r["articleType"]]
            style = USAGE_STYLE.get(r["usage"] or "") or style
            if r["usage"] == "Formal":
                formality = max(formality, 4)
            rows.append({"id": f"p{r['id']}", "name": (r["productDisplayName"] or "").strip(), "category": cat,
                         "article": r["articleType"], "color": color, "style": style, "warmth": warmth,
                         "formality": formality, "usage": r["usage"], "bytes": r["image"]["bytes"]})
        print(f"[ok] shard {i}: {len(rows)} usable men's products so far")
    return rows


def save_jpg(data: bytes, dst: Path, width: int = 480) -> None:
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((width, int(width * 4 / 3)), Image.Resampling.LANCZOS)
        im.save(dst, format="JPEG", quality=86, optimize=True)


def main(shards: list[int], per_bucket: int) -> None:
    rows = load_rows(shards)
    scorer = Scorer()
    CATALOG.mkdir(parents=True, exist_ok=True)
    (SHOP / "img").mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    picks = {}
    for slot, (articles, colors, usage, prompt, keyword) in SLOTS.items():
        cands = [r for r in rows if r["article"] in articles and r["color"] in colors and r["id"] not in used
                 and (keyword is None or keyword.lower() in r["name"].lower())]
        if usage:
            pref = [r for r in cands if r["usage"] == usage]
            cands = pref or cands
        cands = cands[:40]
        if not cands:
            print(f"[skip] {slot}: no candidates")
            continue
        others = ["a t-shirt", "a shirt", "a sweater", "trousers", "jeans", "shoes", "a jacket", "a hoodie"]
        prompts = [f"a product photo of {prompt}", *[f"a product photo of {o}" for o in others]]
        best = max(cands, key=lambda r: float(scorer.probs(r["bytes"], prompts)[0]))
        used.add(best["id"])
        save_jpg(best["bytes"], CATALOG / f"{slot}.jpg", width=600)
        picks[slot] = {"source_id": best["id"], "title": best["name"], "article": best["article"],
                       "color": best["color"]}
        print(f"[ok] {slot}: {best['name']} ({best['article']}, {best['color']})")
    # balanced shop catalog: up to `per_bucket` products per (category, color, style) bucket
    buckets: dict[tuple, list] = {}
    for r in rows:
        if r["id"] in used:
            continue
        key = (r["category"], r["color"], r["style"])
        if len(buckets.setdefault(key, [])) < per_bucket:
            buckets[key].append(r)
    products = []
    for items in buckets.values():
        for r in items:
            save_jpg(r["bytes"], SHOP / "img" / f"{r['id']}.jpg")
            products.append({k: r[k] for k in ("id", "name", "category", "article", "color", "style", "warmth",
                                               "formality")} | {"image": f"img/{r['id']}.jpg"})
    products.sort(key=lambda p: (p["category"], p["color"], p["id"]))
    (SHOP / "products.json").write_text(json.dumps(products, indent=0, ensure_ascii=False), encoding="utf-8")
    (CATALOG / "credits.json").write_text(json.dumps(picks, indent=1, ensure_ascii=False), encoding="utf-8")
    (CATALOG / "CREDITS.md").write_text(
        "# Sample wardrobe and shop photo credits\n\n"
        "Real catalog photos from the *Fashion Product Images* dataset by Param Aggarwal (Kaggle; listed as MIT; "
        "images originate from the Myntra catalog), via the Hugging Face mirror "
        "`benitomartin/fashion-product-images-small-900x1200`. Used for a non-commercial demo; a production system "
        "would use the retailer's own licensed catalog.\n", encoding="utf-8")
    print(f"[ok] shop: {len(products)} products in {len(buckets)} buckets")


if __name__ == "__main__":
    a = sys.argv[1:]
    shards = [int(x) for x in a[a.index("--shards") + 1].split(",")] if "--shards" in a else [0, 1]
    per = int(a[a.index("--per-bucket") + 1]) if "--per-bucket" in a else 6
    main(shards, per)

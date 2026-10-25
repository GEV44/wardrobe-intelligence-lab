"""Build the real-photo sample wardrobes (men's + women's) and the shop catalog.

Source: Kaggle "Fashion Product Images" by Param Aggarwal (44k Myntra catalog products, listed as MIT), via the
Hugging Face mirror benitomartin/fashion-product-images-small-900x1200 (parquet). Provenance caveat in RESEARCH.md.

Outputs
- data/catalog/<item_id>.jpg                    photos for sample wardrobe items (FashionCLIP picks per slot)
- data/demo_items.json, data/demo_items_women.json   sample wardrobes (names in EN/HY/RU)
- data/shop/products.json + data/shop/img/*.jpg  balanced men's + women's catalog for "shop the gap"
Usage: python scripts/build_shop.py [--shards 0,1,2,3,4] [--per-bucket 4]
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
GENDERS = {"Men": "men", "Women": "women", "Unisex": None}

# articleType -> (category, warmth, formality, default style, garment key)
ARTICLES = {
    "Shirts": ("top", 3, 4, "smart", "shirt"), "Tshirts": ("top", 2, 1, "casual", "tshirt"),
    "Tops": ("top", 2, 2, "casual", "top"), "Sweaters": ("top", 4, 3, "minimal", "sweater"),
    "Sweatshirts": ("top", 4, 1, "streetwear", "sweatshirt"), "Jackets": ("outerwear", 4, 2, "casual", "jacket"),
    "Blazers": ("outerwear", 3, 5, "classic", "blazer"), "Trousers": ("bottom", 3, 4, "smart", "trousers"),
    "Jeans": ("bottom", 2, 1, "casual", "jeans"), "Track Pants": ("bottom", 2, 1, "streetwear", "trackpants"),
    "Shorts": ("bottom", 1, 1, "casual", "shorts"), "Skirts": ("bottom", 2, 3, "classic", "skirt"),
    "Dresses": ("dress", 2, 3, "classic", "dress"),
    "Casual Shoes": ("shoes", 2, 2, "casual", "shoes"), "Formal Shoes": ("shoes", 3, 4, "classic", "formalshoes"),
    "Sports Shoes": ("shoes", 2, 1, "streetwear", "sneakers"), "Sandals": ("shoes", 1, 1, "casual", "sandals"),
    "Heels": ("shoes", 2, 4, "classic", "heels"), "Flats": ("shoes", 2, 2, "minimal", "flats"),
}
COLORS = {
    "White": "white", "Black": "black", "Grey": "grey", "Grey Melange": "grey", "Charcoal": "grey",
    "Navy Blue": "navy", "Blue": "blue", "Teal": "blue", "Brown": "brown", "Coffee Brown": "brown", "Tan": "camel",
    "Beige": "beige", "Khaki": "beige", "Cream": "cream", "Off White": "cream", "Olive": "olive", "Green": "green",
    "Red": "red", "Maroon": "burgundy", "Burgundy": "burgundy", "Pink": "pink", "Yellow": "yellow", "Mustard": "yellow",
}
USAGE_STYLE = {"Formal": "classic", "Smart Casual": "smart", "Casual": None, "Sports": "streetwear"}

# ---- localized names: garment nouns (EN, HY, RU, RU grammatical gender) + colour adjectives ----
GARMENTS = {
    "shirt": ("Shirt", "վերնաշապիկ", "рубашка", "f"), "tshirt": ("T-Shirt", "շապիկ", "футболка", "f"),
    "top": ("Top", "թոփ", "топ", "m"), "sweater": ("Sweater", "սվիտեր", "свитер", "m"),
    "sweatshirt": ("Sweatshirt", "սվիտշոթ", "свитшот", "m"), "jacket": ("Jacket", "բաճկոն", "куртка", "f"),
    "blazer": ("Blazer", "բլեյզեր", "блейзер", "m"), "trousers": ("Trousers", "տաբատ", "брюки", "p"),
    "jeans": ("Jeans", "ջինս", "джинсы", "p"), "trackpants": ("Track Pants", "սպորտային տաբատ", "спортивные брюки", "p"),
    "shorts": ("Shorts", "շորտեր", "шорты", "p"), "skirt": ("Skirt", "կիսաշրջազգեստ", "юбка", "f"),
    "dress": ("Dress", "զգեստ", "платье", "n"), "shoes": ("Shoes", "կոշիկներ", "туфли", "p"),
    "formalshoes": ("Formal Shoes", "դասական կոշիկներ", "классические туфли", "p"),
    "sneakers": ("Sneakers", "սնիքերներ", "кроссовки", "p"), "sandals": ("Sandals", "սանդալներ", "сандалии", "p"),
    "heels": ("Heels", "բարձրակրունկ կոշիկներ", "туфли на каблуке", "p"), "flats": ("Flats", "բալետկաներ", "балетки", "p"),
}
COLOR_EN = {c: c.title() for c in ("white", "black", "grey", "cream", "beige", "camel", "brown", "navy", "blue",
                                   "olive", "green", "red", "burgundy", "pink", "yellow")}
COLOR_HY = {"white": "Սպիտակ", "black": "Սև", "grey": "Մոխրագույն", "cream": "Կրեմագույն", "beige": "Բեժ",
            "camel": "Կամել", "brown": "Շագանակագույն", "navy": "Մուգ կապույտ", "blue": "Կապույտ",
            "olive": "Ձիթագույն", "green": "Կանաչ", "red": "Կարմիր", "burgundy": "Բորդո", "pink": "Վարդագույն",
            "yellow": "Դեղին"}
RU_STEMS = {"white": ("бел", "hard"), "black": ("чёрн", "hard"), "grey": ("сер", "hard"), "cream": ("кремов", "hard"),
            "beige": ("бежев", "hard"), "brown": ("коричнев", "hard"), "navy": ("тёмно-син", "soft"),
            "blue": ("син", "soft"), "olive": ("оливков", "hard"), "green": ("зелён", "hard"), "red": ("красн", "hard"),
            "burgundy": ("бордов", "hard"), "pink": ("розов", "hard"), "yellow": ("жёлт", "hard")}
RU_ENDINGS = {"hard": {"m": "ый", "f": "ая", "n": "ое", "p": "ые"}, "soft": {"m": "ий", "f": "яя", "n": "ее", "p": "ие"}}


def localized_names(color: str, garment: str) -> tuple[str, dict]:
    en, hy, ru, g = GARMENTS[garment]
    if color == "camel":
        ru_name = f"{ru[0].upper()}{ru[1:]} цвета кэмел"
    else:
        stem, kind = RU_STEMS[color]
        adj = stem + RU_ENDINGS[kind][g]
        ru_name = f"{adj[0].upper()}{adj[1:]} {ru}"
    return f"{COLOR_EN[color]} {en}", {"hy": f"{COLOR_HY[color]} {hy}", "ru": ru_name}


# ---- sample wardrobe slots: id -> (articleTypes, colors, preferred usage, CLIP prompt, name keyword) ----
MEN_EXTRA = {   # appended to the existing curated 14-item men's wardrobe
    "top-white-tee": (["Tshirts"], ["white"], "Casual", "a plain white t-shirt", None),
    "top-navy-polo": (["Tshirts"], ["navy"], "Casual", "a navy polo t-shirt", "Polo"),
    "top-black-shirt": (["Shirts"], ["black"], None, "a black shirt", None),
    "top-checked-shirt": (["Shirts"], ["burgundy", "red"], None, "a red checked shirt", None),
    "top-green-tee": (["Tshirts"], ["green", "olive"], "Casual", "a green t-shirt", None),
    "bottom-grey-trousers": (["Trousers"], ["grey"], "Formal", "grey formal trousers", None),
    "bottom-black-trousers": (["Trousers"], ["black"], "Formal", "black formal trousers", None),
    "bottom-beige-shorts": (["Shorts"], ["beige", "olive", "navy"], None, "casual shorts", None),
    "shoes-brown-formal": (["Formal Shoes"], ["brown", "camel"], None, "brown leather formal shoes", None),
    "shoes-grey-sports": (["Sports Shoes"], ["grey", "black", "blue"], None, "running sports shoes", None),
    "shoes-brown-sandals": (["Sandals"], ["brown", "black"], None, "leather sandals", None),
    "outer-grey-jacket": (["Jackets"], ["grey", "navy"], "Casual", "a grey casual jacket", None),
}
WOMEN = {
    "w-top-white-shirt": (["Shirts", "Tops"], ["white"], None, "a white blouse", None),
    "w-top-black-top": (["Tops"], ["black"], None, "a black top", None),
    "w-top-pink-top": (["Tops"], ["pink"], None, "a pink top", None),
    "w-top-red-top": (["Tops"], ["red", "burgundy"], None, "a red top", None),
    "w-top-navy-tee": (["Tshirts"], ["navy", "blue"], None, "a navy t-shirt", None),
    "w-top-cream-sweater": (["Sweaters", "Sweatshirts"], ["cream", "white", "beige", "grey"], None, "a knit sweater", None),
    "w-bottom-blue-jeans": (["Jeans"], ["blue"], None, "blue jeans", None),
    "w-bottom-black-trousers": (["Trousers", "Jeans"], ["black"], None, "black trousers", None),
    "w-bottom-beige-trousers": (["Trousers"], ["beige", "cream", "brown"], None, "beige trousers", None),
    "w-bottom-black-skirt": (["Skirts"], ["black", "navy"], None, "a black skirt", None),
    "w-bottom-blue-skirt": (["Skirts"], ["blue", "pink", "red", "white"], None, "a skirt", None),
    "w-dress-black": (["Dresses"], ["black"], None, "a black dress", None),
    "w-dress-red": (["Dresses"], ["red", "burgundy"], None, "a red dress", None),
    "w-dress-blue": (["Dresses"], ["blue", "navy"], None, "a blue dress", None),
    "w-dress-white": (["Dresses"], ["white", "cream"], None, "a white dress", None),
    "w-dress-pink": (["Dresses"], ["pink"], None, "a pink dress", None),
    "w-dress-green": (["Dresses"], ["green", "olive", "yellow"], None, "a green dress", None),
    "w-shoes-white-sneakers": (["Casual Shoes", "Sports Shoes"], ["white"], None, "white sneakers", None),
    "w-shoes-black-heels": (["Heels"], ["black"], None, "black high heels", None),
    "w-shoes-beige-heels": (["Heels"], ["beige", "pink", "cream", "camel"], None, "nude heels", None),
    "w-shoes-brown-flats": (["Flats"], ["brown", "camel", "black", "beige"], None, "ballet flats", None),
    "w-shoes-sandals": (["Sandals", "Flats"], ["brown", "camel", "black"], None, "flat sandals", None),
    "w-outer-black-jacket": (["Jackets"], ["black"], None, "a black jacket", None),
    "w-outer-blue-jacket": (["Jackets"], ["blue", "navy", "grey", "pink"], None, "a casual jacket", None),
}


def load_rows(shards: list[int]) -> list[dict]:
    rows = []
    for i in shards:
        path = hf_hub_download(REPO, f"data/train-{i:05d}-of-00014.parquet", repo_type="dataset")
        table = pq.read_table(path, columns=["id", "gender", "articleType", "baseColour", "usage",
                                             "productDisplayName", "image"])
        for r in table.to_pylist():
            name = (r["productDisplayName"] or "").strip()
            if r["gender"] not in GENDERS or r["articleType"] not in ARTICLES:
                continue
            if any(w in name for w in ("Kids", "Boys", "Girls", "Infant")):
                continue  # some kids' items carry an adult gender label in the source data
            color = COLORS.get(r["baseColour"] or "")
            if not color or not r.get("image") or not r["image"].get("bytes"):
                continue
            cat, warmth, formality, style, garment = ARTICLES[r["articleType"]]
            style = USAGE_STYLE.get(r["usage"] or "") or style
            if r["usage"] == "Formal":
                formality = max(formality, 4)
            for gender in ([GENDERS[r["gender"]]] if GENDERS[r["gender"]] else ["men", "women"]):
                rows.append({"id": f"p{r['id']}", "name": name, "category": cat, "article": r["articleType"],
                             "garment": garment, "color": color, "style": style, "warmth": warmth,
                             "formality": formality, "usage": r["usage"], "gender": gender,
                             "bytes": r["image"]["bytes"]})
        print(f"[ok] shard {i}: {len(rows)} usable rows so far")
    return rows


def save_jpg(data: bytes, dst: Path, width: int = 480) -> None:
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((width, int(width * 4 / 3)), Image.Resampling.LANCZOS)
        im.save(dst, format="JPEG", quality=86, optimize=True)


def pick(rows: list[dict], slots: dict, gender: str, scorer: Scorer, used: set) -> list[dict]:
    items = []
    others = ["a t-shirt", "a shirt", "a sweater", "trousers", "jeans", "a dress", "a skirt", "shoes", "a jacket"]
    for slot, (articles, colors, usage, prompt, keyword) in slots.items():
        cands = [r for r in rows if r["gender"] == gender and r["article"] in articles and r["color"] in colors
                 and r["id"] not in used and (keyword is None or keyword.lower() in r["name"].lower())]
        if usage:
            cands = [r for r in cands if r["usage"] == usage] or cands
        cands = cands[:30]
        if not cands:
            print(f"[skip] {slot}: no candidates")
            continue
        prompts = [f"a product photo of {prompt}", *[f"a product photo of {o}" for o in others]]
        best = max(cands, key=lambda r: float(scorer.probs(r["bytes"], prompts)[0]))
        used.add(best["id"])
        save_jpg(best["bytes"], CATALOG / f"{slot}.jpg", width=600)
        name_en, names = localized_names(best["color"], best["garment"])
        items.append({"id": slot, "name": name_en, "category": best["category"], "color": best["color"],
                      "style": best["style"], "warmth": best["warmth"], "formality": best["formality"],
                      "tags": [best["article"].lower()], "names": names, "source_product": best["name"]})
        print(f"[ok] {slot}: {best['name']} -> {name_en}")
    return items


def write_items(path: Path, items: list[dict], keep_existing: bool) -> None:
    existing = json.loads(path.read_text(encoding="utf-8")) if keep_existing and path.exists() else []
    ids = {e["id"] for e in existing}
    merged = existing + [i for i in items if i["id"] not in ids]
    lines = ",\n".join("  " + json.dumps(i, ensure_ascii=False) for i in merged)
    path.write_text(f"[\n{lines}\n]\n", encoding="utf-8")
    print(f"[ok] {path.name}: {len(merged)} items")


def main(shards: list[int], per_bucket: int) -> None:
    rows = load_rows(shards)
    scorer = Scorer()
    CATALOG.mkdir(parents=True, exist_ok=True)
    (SHOP / "img").mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    write_items(ROOT / "data" / "demo_items.json", pick(rows, MEN_EXTRA, "men", scorer, used), keep_existing=True)
    write_items(ROOT / "data" / "demo_items_women.json", pick(rows, WOMEN, "women", scorer, used), keep_existing=False)
    buckets: dict[tuple, list] = {}
    for r in rows:
        if r["id"] in used:
            continue
        key = (r["gender"], r["category"], r["color"], r["style"])
        if len(buckets.setdefault(key, [])) < per_bucket:
            buckets[key].append(r)
    products, seen = [], set()
    for items in buckets.values():
        for r in items:
            if (r["id"], r["gender"]) in seen:
                continue
            seen.add((r["id"], r["gender"]))
            img = SHOP / "img" / f"{r['id']}.jpg"
            if not img.exists():
                save_jpg(r["bytes"], img)
            products.append({k: r[k] for k in ("id", "name", "category", "article", "color", "style", "warmth",
                                               "formality", "gender")} | {"image": f"img/{r['id']}.jpg"})
    products.sort(key=lambda p: (p["gender"], p["category"], p["color"], p["id"]))
    (SHOP / "products.json").write_text(json.dumps(products, indent=0, ensure_ascii=False), encoding="utf-8")
    keep = {p["image"].split("/")[-1] for p in products}
    for f in (SHOP / "img").glob("*.jpg"):
        if f.name not in keep:
            f.unlink()
    (CATALOG / "CREDITS.md").write_text(
        "# Sample wardrobe and shop photo credits\n\n"
        "Real catalog photos from the *Fashion Product Images* dataset by Param Aggarwal (Kaggle; listed as MIT; "
        "images originate from the Myntra catalog), via the Hugging Face mirror "
        "`benitomartin/fashion-product-images-small-900x1200`. Used for a non-commercial demo; a production system "
        "would use the retailer's own licensed catalog.\n", encoding="utf-8")
    print(f"[ok] shop: {len(products)} products ({sum(p['gender'] == 'women' for p in products)} women's)")


if __name__ == "__main__":
    a = sys.argv[1:]
    shards = [int(x) for x in a[a.index("--shards") + 1].split(",")] if "--shards" in a else [0, 1, 2, 3, 4]
    per = int(a[a.index("--per-bucket") + 1]) if "--per-bucket" in a else 4
    main(shards, per)

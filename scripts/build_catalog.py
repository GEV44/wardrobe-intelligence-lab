"""Curate real product photos for the sample wardrobe: an offline, reproducible ML pipeline.

1. Search Openverse for openly licensed photos (CC0 / Public Domain Mark / CC BY only; no NC, no ND, no SA).
2. Rank candidates with FashionCLIP zero-shot (target garment vs other garments, people, drawings, text).
   Photos where a person is likely are rejected (privacy rule: no real people).
3. Cut the best candidates out onto white with BiRefNet-lite and re-score the cut-out.
4. Write data/catalog/<item_id>.jpg + data/catalog/credits.json (+ CREDITS.md) and a review contact sheet.

Usage: python scripts/build_catalog.py [item_id ...]
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lookroom import bgremove, images, vision  # noqa: E402

USE_BIREFNET = "--birefnet" in sys.argv
from lookroom.seed import load_demo_items  # noqa: E402

OUT = ROOT / "data" / "catalog"
CACHE = ROOT / "runtime" / "catalog_cache"
UA = {"User-Agent": "WardrobeIntelligenceLab/0.2 (catalog curation; +https://huggingface.co/spaces/GevLLM/wardrobe-intelligence-lab)"}
LICENSES = "cc0,pdm,by"
PER_ITEM = 24

QUERIES = {
    "top-white-oxford": ["white oxford shirt", "white button down shirt", "white dress shirt"],
    "top-black-tee": ["black t-shirt", "black tshirt plain"],
    "top-cream-knit": ["cream knit sweater", "white wool sweater", "knitted jumper"],
    "top-blue-buttondown": ["blue button down shirt", "blue dress shirt", "light blue shirt"],
    "top-olive-hoodie": ["olive hoodie", "green hoodie", "hooded sweatshirt"],
    "bottom-navy-trousers": ["navy trousers", "dress pants navy", "suit trousers"],
    "bottom-black-jeans": ["black jeans", "black denim pants"],
    "bottom-beige-chinos": ["beige chinos", "khaki pants", "khaki trousers"],
    "bottom-lightblue-jeans": ["blue jeans", "denim jeans", "light blue jeans"],
    "shoes-white-sneakers": ["white sneakers", "white canvas shoes", "white trainers"],
    "shoes-black-loafers": ["black loafers", "black leather shoes", "penny loafers"],
    "shoes-brown-boots": ["brown leather boots", "chelsea boots", "brown boots"],
    "outer-camel-coat": ["camel coat", "wool overcoat", "beige coat"],
    "outer-denim-jacket": ["denim jacket", "jean jacket", "blue denim jacket"],
}
GARMENT_WORD = {
    "top-white-oxford": "white button-down shirt", "top-black-tee": "black t-shirt",
    "top-cream-knit": "cream knit sweater", "top-blue-buttondown": "light blue button-down shirt",
    "top-olive-hoodie": "olive green hoodie", "bottom-navy-trousers": "navy tailored trousers",
    "bottom-black-jeans": "black jeans", "bottom-beige-chinos": "beige chino trousers",
    "bottom-lightblue-jeans": "light blue denim jeans", "shoes-white-sneakers": "white sneakers",
    "shoes-black-loafers": "black leather loafers", "shoes-brown-boots": "brown leather boots",
    "outer-camel-coat": "camel wool coat", "outer-denim-jacket": "blue denim jacket",
}
PERSON_PROMPTS = ["a photo of a person wearing {g}", "a model posing in {g}", "a person's hands and torso",
                  "a photo of a man or a woman", "a portrait of a face", "a person's legs and feet"]
NEGATIVES = ["a drawing or an illustration", "a screenshot with text", "a shop window or a store",
             "a pile of many clothes", "an animal", "a landscape"]
SEARCH_SUFFIXES = ["isolated", "flat lay", "on hanger", "product"]
PERSON_MAX = 0.08  # reject when the summed person probability exceeds this (privacy rule: no real people)


class Scorer:
    """FashionCLIP (combined text+image ONNX) with arbitrary prompts: offline curation only."""

    def __init__(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = vision.MODEL_DIR
        self.session = ort.InferenceSession(str(d / vision.MODEL_FILE), providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(77)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("<|endoftext|>") or 0, pad_token="<|endoftext|>")
        self.names = {i.name for i in self.session.get_inputs()}

    def probs(self, data: bytes, prompts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(prompts)
        feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                 "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                 "pixel_values": vision.preprocess(data)}
        logits = self.session.run(["logits_per_image"], {k: v for k, v in feeds.items() if k in self.names})[0][0]
        e = np.exp(logits - logits.max())
        return e / e.sum()


def get_json(url: str) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            raise
    raise RuntimeError("rate limited")


def fetch(url: str, max_bytes: int = 4_000_000) -> bytes | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
            data = r.read(max_bytes + 1)
        if len(data) > max_bytes:
            return None
        with Image.open(io.BytesIO(data)) as im:
            if min(im.size) < 300:
                return None
        return data
    except Exception:
        return None


def candidates(item_id: str) -> list[dict]:
    seen, out = set(), []
    base = QUERIES[item_id]
    for q in [f"{base[0]} {s}" for s in SEARCH_SUFFIXES] + base:
        qs = urllib.parse.urlencode({"q": q, "license": LICENSES, "category": "photograph", "page_size": 20,
                                     "mature": "false"})
        for r in get_json(f"https://api.openverse.org/v1/images/?{qs}").get("results", []):
            if r["id"] not in seen and r.get("url"):
                seen.add(r["id"])
                out.append(r)
        time.sleep(1.5)
    return out[:PER_ITEM * 2]


def to_square_jpg(data: bytes, size: int = 720) -> bytes:
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def curate(item, scorer: Scorer, others: list[str]) -> tuple[dict | None, list[tuple[float, bytes]]]:
    g = GARMENT_WORD[item.id]
    target = f"a product photo of {g}"
    persons = [p.format(g=g) for p in PERSON_PROMPTS]
    prompts = [target, *[f"a product photo of {o}" for o in others], *persons, *NEGATIVES]
    person_slice = slice(1 + len(others), 1 + len(others) + len(persons))
    ranked = []
    for r in candidates(item.id):
        data = (r.get("thumbnail") and fetch(r["thumbnail"])) or fetch(r["url"])
        if data is None:
            continue
        p = scorer.probs(data, prompts)
        if p[person_slice].sum() > PERSON_MAX:
            continue  # privacy: never use photos that likely show a person
        ranked.append((float(p[0]), r, data))
        if len(ranked) >= PER_ITEM:
            break
    ranked.sort(key=lambda x: -x[0])
    best, sheet = None, []
    for rank, (score, r, data) in enumerate(ranked[:6]):
        # Fast classical cut-out first (also proves a plain studio backdrop); BiRefNet only if asked.
        cut = images.flat_background_cutout(data)
        if cut is None and rank == 0 and USE_BIREFNET:
            cut = bgremove.remove_background(data)
        final = cut or data
        p = scorer.probs(final, prompts)
        if p[person_slice].sum() > PERSON_MAX:
            continue
        s2 = float(p[0]) + (0.25 if cut is not None else 0.0)  # prefer verified plain-backdrop product shots
        sheet.append((s2, to_square_jpg(final)))
        if best is None or s2 > best["score"]:
            best = {"score": s2, "pre_score": score, "data": final, "cutout": cut is not None, "meta": r}
    return best, sheet


def main(only: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    scorer = Scorer()
    items = [i for i in load_demo_items() if not only or i.id in only]
    words = list(GARMENT_WORD.values())
    credits_path = OUT / "credits.json"
    credits = json.loads(credits_path.read_text(encoding="utf-8")) if credits_path.exists() else {}
    sheet_rows = []
    for item in items:
        others = [w for k, w in GARMENT_WORD.items() if k != item.id and k.split("-")[0] != item.id.split("-")[0]][:6]
        best, sheet = curate(item, scorer, others)
        if not best or best["score"] < 0.35:
            print(f"[skip] {item.id}: no confident candidate ({best and round(best['score'], 2)})")
            continue
        (OUT / f"{item.id}.jpg").write_bytes(to_square_jpg(best["data"]))
        m = best["meta"]
        credits[item.id] = {"title": m.get("title"), "creator": m.get("creator"), "license": m.get("license"),
                            "license_version": m.get("license_version"), "license_url": m.get("license_url"),
                            "source": m.get("source"), "landing_url": m.get("foreign_landing_url"),
                            "modified": "background removed, cropped, resized" if best["cutout"] else "cropped, resized",
                            "clip_score": round(best["score"], 3)}
        sheet_rows.append((item.id, sheet[:4]))
        print(f"[ok] {item.id}: score {best['score']:.2f} ({m.get('license')}) {m.get('foreign_landing_url')}")
    credits_path.write_text(json.dumps(credits, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# Sample wardrobe photo credits", "",
          "Openly licensed photos from Openverse, adapted (background removed, cropped). No endorsement implied.", ""]
    for k, c in sorted(credits.items()):
        md.append(f"- `{k}`: \"{c['title']}\" by {c['creator'] or 'unknown'}, {c['license'].upper()} "
                  f"{c.get('license_version') or ''} ({c['landing_url']}). Modified: {c['modified']}.")
    (OUT / "CREDITS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if sheet_rows:  # review contact sheet: chosen first, then alternates
        cell = 200
        img = Image.new("RGB", (cell * 4 + 260, cell * len(sheet_rows)), "white")
        d = ImageDraw.Draw(img)
        for row, (iid, cells) in enumerate(sheet_rows):
            d.text((8, row * cell + 90), iid, fill="black")
            for col, (s, data) in enumerate(sorted(cells, key=lambda c: -c[0])):
                with Image.open(io.BytesIO(data)) as im:
                    img.paste(im.resize((cell - 6, cell - 6)), (260 + col * cell, row * cell + 3))
                d.text((262 + col * cell, row * cell + 4), f"{s:.2f}", fill="red")
        img.save(CACHE / "contact_sheet.jpg", quality=80)
        print(f"[ok] review sheet: {CACHE / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")])

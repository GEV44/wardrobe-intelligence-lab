"""One-time setup for the optional zero-shot garment taggers (ONNX only, no remote code executed).

- models/fashion-siglip/: Marqo/marqo-fashionSigLIP (Apache-2.0): the repo's int8 vision ONNX (~94 MB) plus
  text embeddings for our label prompts, computed once here with the fp32 text ONNX (int8 text lost accuracy;
  the 441 MB fp32 text model stays only in the Hugging Face cache, not in models/).
- models/fashion-clip/:   patrickjohncyh/fashion-clip (MIT) official ONNX export, quantized here to int8
  (~4x smaller); the fp32 copy is deleted afterwards.
Idempotent: files already present are skipped. Which one is used at runtime: see lookroom/vision.py.

Usage: python scripts/setup_models.py [--only siglip|fashionclip] [--keep-fp32]
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lookroom.vision import (  # noqa: E402
    MODEL_DIR, MODEL_FILE, REPO_ID, REVISION, SIGLIP_DIR, SIGLIP_FILES, SIGLIP_REPO_ID, siglip_prompts,
)

CLIP_FILES = ["onnx/model.onnx", "onnx/tokenizer.json", "onnx/preprocessor_config.json"]
# repo path -> local name (the repo ships a ready-made int8 vision ONNX, so no local quantization needed)
SIGLIP_REMOTE = {"onnx/vision_model_int8.onnx": "vision_model_int8.onnx", "tokenizer.json": "tokenizer.json",
                 "preprocessor_config.json": "preprocessor_config.json", "open_clip_config.json": "open_clip_config.json"}


def _fetch(repo: str, remote: str, dst: Path) -> None:
    from huggingface_hub import hf_hub_download

    if not dst.exists():
        shutil.copyfile(hf_hub_download(repo, remote, revision=REVISION), dst)
    print(f"[ok] {repo}:{remote} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def setup_siglip() -> None:
    SIGLIP_DIR.mkdir(parents=True, exist_ok=True)
    for remote, local in SIGLIP_REMOTE.items():
        _fetch(SIGLIP_REPO_ID, remote, SIGLIP_DIR / local)
    _siglip_text_embeds(SIGLIP_DIR / "text_embeds.npz")
    assert all((SIGLIP_DIR / f).is_file() for f in SIGLIP_FILES)


def _siglip_text_embeds(dst: Path) -> None:
    """Encode all label prompts with the fp32 text tower once; skip if dst already matches the current prompts."""
    import numpy as np
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    prompts = siglip_prompts()
    if dst.exists():
        with np.load(dst) as z:
            if [str(p) for p in z["prompts"]] == prompts:
                print(f"[ok] {dst} up to date ({len(prompts)} prompts)")
                return
    from lookroom.vision import _session

    tok = Tokenizer.from_file(str(SIGLIP_DIR / "tokenizer.json"))  # pads to 64 with </s>, appends </s>
    ids = np.array([e.ids for e in tok.encode_batch(prompts)], dtype=np.int64)
    sess = _session(Path(hf_hub_download(SIGLIP_REPO_ID, "onnx/text_model.onnx", revision=REVISION)))
    emb = sess.run(["text_embeds"], {"input_ids": ids})[0].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.savez(dst, prompts=np.array(prompts), emb=emb)
    print(f"[ok] {dst} ({len(prompts)} prompts, {dst.stat().st_size / 1e3:.0f} KB)")


def setup_fashionclip(keep_fp32: bool = False) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fp32, int8 = MODEL_DIR / "model.onnx", MODEL_DIR / MODEL_FILE
    for f in CLIP_FILES:
        if f == "onnx/model.onnx" and int8.exists() and not fp32.exists():
            continue  # already quantized; skip the large fp32 download
        _fetch(REPO_ID, f, MODEL_DIR / Path(f).name)
    if not int8.exists():
        print("[..] quantizing to int8 (one-time, ~1 min)")
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)
    print(f"[ok] {int8} ({int8.stat().st_size / 1e6:.1f} MB)")
    if not keep_fp32 and fp32.exists():
        fp32.unlink()
        print("[ok] removed fp32 model to save disk (HF cache still holds a copy)")


def main(only: str | None = None, keep_fp32: bool = False) -> None:
    if only in (None, "siglip"):
        setup_siglip()
    if only in (None, "fashionclip"):
        setup_fashionclip(keep_fp32)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(only=args[args.index("--only") + 1] if "--only" in args else None, keep_fp32="--keep-fp32" in args)

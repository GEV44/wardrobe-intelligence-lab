"""One-time setup for optional background removal (BiRefNet_lite, MIT).

Downloads the fp32 ONNX export (wonfy/BiRefNet_lite-ONNX-webgpu, MIT; weights-unchanged rewrite of
onnx-community/BiRefNet_lite-ONNX) into models/birefnet-lite. ONNX is a data format: no remote code runs.
Idempotent. Usage: python scripts/setup_bgremove.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lookroom.bgremove import EXPORT_FILE, EXPORT_REPO_ID, MODEL_FILE, model_dir  # noqa: E402


def main() -> None:
    from huggingface_hub import hf_hub_download

    dst_dir = model_dir()
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / MODEL_FILE
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"[skip] {dst} already present ({dst.stat().st_size / 1e6:.1f} MB)")
    else:
        src = Path(hf_hub_download(EXPORT_REPO_ID, EXPORT_FILE))
        shutil.copyfile(src, dst)
        print(f"[ok] {EXPORT_FILE} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    print("[note] fixed 1024x1024 input; expect several GB peak RAM and tens of seconds per image on 2 vCPUs.")


if __name__ == "__main__":
    main()

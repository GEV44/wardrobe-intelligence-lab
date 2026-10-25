"""Deploy to a public Hugging Face Space (Docker SDK). Runs 24/7 on HF hardware, independent of this laptop.

Usage: python scripts/deploy_space.py [--no-llm] [--space NAME]
- Requires `huggingface-cli login` (or HF_TOKEN). Your ANTHROPIC_API_KEY (if set and not --no-llm) is stored as an
  encrypted Space *secret*; it is never written to any file or printed.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ["app.py", "api.py", "Dockerfile", "requirements-deploy.txt", "ARCHITECTURE.md", "RESEARCH.md",
           ".streamlit/config.toml", "lookroom", "data", "demo_assets",
           "scripts/setup_models.py", "scripts/setup_bgremove.py", "scripts/reset_demo.py", "scripts/smoke_test.py"]
VARIABLES = {"LOOKROOM_LIVE_WEATHER_DEFAULT": "1", "LOOKROOM_LLM_DAILY_CAP": "150", "LOOKROOM_SESSION_LLM_LIMIT": "20"}
FRONT_MATTER = """---
title: Wardrobe Intelligence Lab
emoji: 🧥
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Context-aware outfit AI with feedback learning (concept demo)
---

"""


def stage(dst: Path) -> None:
    for rel in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            print(f"[skip] {rel} (missing)")
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, target)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    (dst / "README.md").write_text(FRONT_MATTER + readme, encoding="utf-8")


def main() -> None:
    args = sys.argv[1:]
    name = args[args.index("--space") + 1] if "--space" in args else "wardrobe-intelligence-lab"
    api = HfApi()
    user = api.whoami()["name"]
    repo = f"{user}/{name}"
    api.create_repo(repo, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key and "--no-llm" not in args:
        api.add_space_secret(repo, "ANTHROPIC_API_KEY", key)
        print("[ok] ANTHROPIC_API_KEY stored as an encrypted Space secret")
    else:
        if "--no-llm" in args:
            try:
                api.delete_space_secret(repo, "ANTHROPIC_API_KEY")
            except Exception:
                pass
        print("[ok] no LLM key on the server: stylist uses the rule engine")
    for k, v in VARIABLES.items():
        api.add_space_variable(repo, k, v)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=repo, repo_type="space",
                          commit_message="Deploy Wardrobe Intelligence Lab")
    print(f"[ok] deployed: https://huggingface.co/spaces/{repo}")
    print(f"     app URL:  https://{user.lower()}-{name}.hf.space")


if __name__ == "__main__":
    main()

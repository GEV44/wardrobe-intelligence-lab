# CLAUDE.md

Interview MVP: Wardrobe Intelligence Lab (an independent LookRoom-inspired concept).

- Keep `lookroom/recommender.py` pure, with no db or UI imports. Weights are constants at the top of the file.
- `app.py` and `api.py` must call `lookroom/service.py`. Do not duplicate logic between them.
- Use sqlite3 with per-operation `db.connect()` and parameterized SQL only.
- Tests are isolated through the autouse `LOOKROOM_DATA_DIR` tmp fixture. Run them with `python -m pytest -q`.
- Use English and keep things concise. No paid APIs, no heavy ML dependencies, no network calls at runtime.
- Every DB and service call takes a `user_id`; the UI keeps it in `?u=`. All UI text goes through `lookroom/i18n.py` (en/hy/ru).
- Sample wardrobes (men's `data/demo_items.json`, women's `data/demo_items_women.json`) and the shop come from `scripts/build_shop.py` (Fashion Product Images catalog photos). Rebuild embeddings with `scripts/build_embeddings.py`. Deploy by pushing to GitHub (Streamlit Community Cloud).
- The AI layer is `lookroom/vision.py` (Marqo FashionSigLIP int8 ONNX with FashionCLIP fallback, set up with `scripts/setup_models.py`), `lookroom/llm.py` (local Qwen3 GGUF via llama.cpp) and `lookroom/stylist.py` (optional Claude `claude-opus-5-5` when a key is set). All must keep their fallbacks. Tests set `LOOKROOM_DISABLE_LLM=1` and fake the client.
- Videos: `python scripts/record_demo.py` writes to `docs/videos/`.
- Port 8000 is taken on this machine, so use `uvicorn api:app --port 8010`.
- Outfit shapes: top + bottom + shoes, or dress + shoes, with optional outerwear (`models.is_valid_outfit`).
- Do not git push or deploy without approval.

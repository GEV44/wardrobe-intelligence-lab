# CLAUDE.md

Interview MVP: Wardrobe Intelligence Lab (an independent LookRoom-inspired concept).

- Keep `lookroom/recommender.py` pure, with no db or UI imports. Weights are constants at the top of the file.
- `app.py` and `api.py` must call `lookroom/service.py`. Do not duplicate logic between them.
- Use sqlite3 with per-operation `db.connect()` and parameterized SQL only.
- Tests are isolated through the autouse `LOOKROOM_DATA_DIR` tmp fixture. Run them with `python -m pytest -q`.
- Use English and keep things concise. No paid APIs, no heavy ML dependencies, no network calls at runtime.
- Every DB and service call takes a `user_id`; the UI keeps it in `?u=`. All UI text goes through `lookroom/i18n.py` (en/hy/ru).
- Sample photos come from `scripts/build_catalog.py` (CC0/PD/CC BY, no people). Deploy with `scripts/deploy_space.py`.
- The AI layer is `lookroom/vision.py` (FashionCLIP int8 ONNX, set up with `scripts/setup_models.py`) and `lookroom/stylist.py` (Claude `claude-opus-5-5`). Both must keep their fallbacks. Tests set `LOOKROOM_DISABLE_LLM=1` and fake the client.
- Videos: `python scripts/record_demo.py` writes to `docs/videos/`.
- Port 8000 is taken on this machine, so use `uvicorn api:app --port 8010`.
- Do not git push or deploy without approval.

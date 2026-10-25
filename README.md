# Wardrobe Intelligence Lab
*A LookRoom-inspired AI stylist MVP. Independent concept demo, not affiliated with LookRoom.ai.*

It implements the full loop: **context → wardrobe → AI decision → action → feedback → deeper personalization**, with real garment photos, pretrained models and three languages (English · Հայերեն · Русский).

## What it does
| Feature | How |
|---|---|
| Today's look + 2 alternatives | Transparent engine scores every top × bottom × shoes (+ outerwear) combination: occasion 30%, weather 25%, color 15%, style 15%, learned preference 15%. Picks are diverse and deterministic, shown as flat-lay boards. |
| Live weather | MET Norway API (no key, commercial use allowed, CC BY 4.0), cached and falling back to a manual slider. |
| Learns from feedback | Like/Dislike is stored per user. A bounded `tanh` item signal shifts scores by at most ±7.5 pts, and *How it learns* shows the before/after ranking. |
| AI Stylist chat | **Claude Opus 5.5** with a JSON-schema structured output. It may only pick existing item ids; picks are validated, then re-scored by the engine. It replies in the UI language, and falls back to the rule engine on any failure, budget cap or missing key. |
| Photo upload → auto-tagging | **Marqo FashionSigLIP** (Apache-2.0, int8 ONNX, CPU) zero-shot garment, color and style recognition, with **FashionCLIP** (MIT) as fallback and a color heuristic as the last resort. Results are suggestions the user confirms. |
| Product cut-outs | A fast classical backdrop removal (~50 ms) on the request path. **BiRefNet-lite** (MIT) runs offline for catalog curation, because it takes ~45 s/image on a 2-vCPU server. |
| Real sample wardrobe | `scripts/build_catalog.py` searches Openverse for CC0, PD or CC BY photos only. FashionCLIP ranks them and rejects any with people, the backdrop is cut out, and credits go to `data/catalog/CREDITS.md`. |
| Private workspace per visitor | Every row is scoped by `user_id`, which is kept in the URL (`?u=`). Each new visitor gets their own copy of the sample wardrobe. |
| Robustness | Each UI section is isolated so one failure can't take down the page. Models are baked into the Docker image, every AI call has a fallback, and the Claude budget guard has a daily cap plus a per-session limit. |
| API | FastAPI with typed schemas: `/health`, `/wardrobe`, `/recommendations`, `/feedback`, `/feedback-summary`, `/vision/tag`, `/stylist/chat`. Swagger is at `/docs`. Optional `X-User-Id` header and `lang` parameter. |

## Run locally (Windows, Python 3.11, CPU only)
```powershell
cd lookroom-night-demo
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/setup_models.py            # FashionSigLIP + FashionCLIP ONNX (~250 MB)
$env:ANTHROPIC_API_KEY = "sk-ant-..."      # optional: enables the Claude stylist
streamlit run app.py                       # http://localhost:8501
uvicorn api:app --port 8010                # http://127.0.0.1:8010/docs
python -m pytest -q                        # offline; Claude is faked, BiRefNet tests need RUN_SLOW=1
```
Optional tools: `python scripts/build_catalog.py` rebuilds the photo catalog, `python scripts/record_demo.py` re-records the demo videos, and `python scripts/reset_demo.py` resets the demo user.

## Deploy (always on, independent of this laptop)
**Streamlit Community Cloud (free):** at share.streamlit.io choose New app, then this repo, then `app.py`, with Python 3.11. In *Secrets*, add:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
LOOKROOM_LIVE_WEATHER_DEFAULT = "1"
LOOKROOM_LLM_DAILY_CAP = "150"
```
On first start the app downloads the FashionSigLIP int8 vision model (~94 MB) in the background. The text embeddings are bundled in the repo, and until the download finishes every feature works through its fallback.

**Docker** (any container host, or Hugging Face Spaces with PRO): the pinned `Dockerfile` bakes the models into the image, and `python scripts/deploy_space.py` deploys it to a Space with the key stored as an encrypted secret.

## Architecture
```
Streamlit (app.py) ─┐                       ┌> recommender.py  pure scoring (source of truth), i18n explanations
                    ├─> lookroom/service.py ─┼> db.py           SQLite (WAL), every query scoped by user_id
FastAPI  (api.py) ──┘                       ├> vision.py       FashionSigLIP ─fallback→ FashionCLIP ─fallback→ heuristic
                                            ├> images.py       safe uploads, cut-outs, outfit boards
                                            ├> stylist.py      Claude structured output ─fallback→ rule engine
                                            └> weather.py      MET Norway ─fallback→ manual slider
```
See [ARCHITECTURE.md](ARCHITECTURE.md) and [RESEARCH.md](RESEARCH.md).

## Demo videos
`docs/videos/demo_walkthrough.mp4` and `docs/videos/api_demo.mp4`, recorded unedited from the running app with Playwright.

## Honest limitations
- The scoring is hand-tuned rules plus bounded feedback adaptation. It is not a trained ranker, and there has been no real-user evaluation yet.
- Zero-shot tagging works best on product-style photos. I measured it on a small sample only, so I don't claim an accuracy figure.
- Identity is a URL token, not authentication. On the free hosting tier, data resets when the server restarts.
- The Claude stylist sends item metadata only (no images, no personal data) and costs about $0.01–0.02 per message, with a budget cap.

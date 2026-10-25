# Wardrobe Intelligence Lab
*A LookRoom-inspired AI stylist MVP. Independent concept demo, not affiliated with LookRoom.ai.*

**Live demo:** https://wardrobe-intelligence-lab.streamlit.app (free hosting; it wakes up in about 1 minute if it has been idle).

It implements the full loop: **context → wardrobe → AI decision → action → feedback → deeper personalization**, with real garment photos, pretrained models and three languages (English · Հայերեն · Русский).

## What it does
| Feature | How |
|---|---|
| Today's look + 2 alternatives | Transparent engine scores every top × bottom × shoes **or** dress × shoes (+ outerwear) combination: occasion 30%, weather 25%, color 15%, style 15%, learned preference 15%. Picks are diverse and deterministic, shown as flat-lay boards. **Show another look** and **Dislike** both swap in the next-best outfit from the wardrobe. |
| Live weather | MET Norway API (no key, commercial use allowed, CC BY 4.0), cached and falling back to a manual slider. |
| Learns from feedback | Like/Dislike is stored per user. A bounded `tanh` item signal shifts scores by at most ±7.5 pts, and *How it learns* shows the before/after ranking. |
| AI Stylist chat (free, no API key) | **Qwen3-1.7B** (Apache-2.0, GGUF Q4_K_M) running locally on the CPU via llama.cpp. Multilingual EN/RU/HY keyword rules extract the context first; the LLM fills unknown fields via schema-constrained JSON and writes the reply in English or Russian (Armenian uses translated templates). The engine always picks the outfit, so the model can't invent clothes. Optional: if `ANTHROPIC_API_KEY` is set, Claude is used instead, with a budget cap. |
| Photo upload → auto-tagging | **Marqo FashionSigLIP** (Apache-2.0, int8 ONNX, CPU) zero-shot garment, color and style recognition, with **FashionCLIP** (MIT) as fallback and a color heuristic as the last resort. Results are suggestions the user confirms. |
| Plan my week | A 7-day plan from the MET Norway daily forecast and an occasion per day. Repeat penalties stop any item appearing on consecutive days. Deterministic. |
| Shop the gap | 984 real catalog products (men's and women's). A wardrobe-gap analysis scores each candidate with the user's wardrobe by the same engine across today's context (double weight) plus 5 representative situations, counts the near-best looks it unlocks, then re-ranks by visual taste match. The page shows the biggest gap, its best look and smart picks. "Add to my wardrobe" copies the real product in. |
| Visual similarity | FashionSigLIP image embeddings (1,034 precomputed) with cosine nearest neighbours. Every wardrobe item has "Similar in shop", and a taste vector (mean embedding of liked items) personalizes shop ranking. |
| Product cut-outs | A fast classical backdrop removal (~50 ms) on the request path. **BiRefNet-lite** (MIT) runs offline for catalog curation, because it takes ~45 s/image on a 2-vCPU server. |
| Real sample wardrobe | Real catalog photos from the *Fashion Product Images* dataset (44k Myntra products, listed as MIT), picked per slot by FashionCLIP zero-shot ranking (`scripts/build_shop.py`). Provenance is in `data/catalog/CREDITS.md`. |
| Men's / Women's wardrobes | Two real sample wardrobes (26 men's items; 24 women's items including dresses, heels and flats), switchable in the header. The shop is filtered to match. |
| Private workspace per visitor | Every row is scoped by `user_id`, which is kept in the URL (`?u=`). Each new visitor gets their own copy of the sample wardrobe. |
| Robustness | Each UI section is isolated so one failure can't take down the page. Models are baked into the Docker image, every AI call has a fallback, and the Claude budget guard has a daily cap plus a per-session limit. |
| API | FastAPI with typed schemas: `/health`, `/wardrobe`, `/recommendations`, `/feedback`, `/feedback-summary`, `/vision/tag`, `/stylist/chat`, `/plan/week`, `/shop/suggestions`, `/shop/similar/{item_id}`. Swagger is at `/docs`. Optional `X-User-Id` header and `lang` parameter. |

## Run locally (Windows, Python 3.11, CPU only)
```powershell
cd lookroom-night-demo
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/setup_models.py            # FashionSigLIP + FashionCLIP ONNX (~250 MB)
# no API keys needed: the stylist LLM (Qwen3-1.7B GGUF, ~1.1 GB) downloads on first start
streamlit run app.py                       # http://localhost:8501
uvicorn api:app --port 8010                # http://127.0.0.1:8010/docs
python -m pytest -q                        # offline; paid APIs are faked, BiRefNet tests need RUN_SLOW=1
```
Optional tools: `python scripts/build_catalog.py` rebuilds the photo catalog, `python scripts/record_demo.py` re-records the demo videos, and `python scripts/reset_demo.py` resets the demo user.

## Deploy (always on, independent of this laptop)
**Streamlit Community Cloud (free):** at share.streamlit.io choose New app, then this repo, then `app.py`, with Python 3.11. No keys are needed. Optionally, in *Secrets* add `LOOKROOM_LIVE_WEATHER_DEFAULT = "1"`. If the host is short on memory, switch to the smaller model with `LOOKROOM_LLM_REPO = "Qwen/Qwen3-0.6B-GGUF"` and `LOOKROOM_LLM_FILE = "Qwen3-0.6B-Q8_0.gguf"`.
On first start the app downloads the FashionSigLIP int8 vision model (~94 MB) and Qwen3-1.7B (~1.1 GB) in the background. The text embeddings are bundled in the repo, and until the download finishes every feature works through its fallback.

**Docker** (any container host, or Hugging Face Spaces with PRO): the pinned `Dockerfile` bakes the models into the image, and `python scripts/deploy_space.py` deploys it to a Space with the key stored as an encrypted secret.

## Architecture
```
Streamlit (app.py) ─┐                       ┌> recommender.py  pure scoring (source of truth), i18n explanations
                    ├─> lookroom/service.py ─┼> db.py           SQLite (WAL), every query scoped by user_id
FastAPI  (api.py) ──┘                       ├> vision.py       FashionSigLIP ─fallback→ FashionCLIP ─fallback→ heuristic
                                            ├> images.py       safe uploads, cut-outs, outfit boards
                                            ├> stylist.py      rules + local Qwen3 (llm.py) ─fallback→ templates
                                            │                  (optional Claude if a key is configured)
                                            └> weather.py      MET Norway ─fallback→ manual slider
```
See [ARCHITECTURE.md](ARCHITECTURE.md) and [RESEARCH.md](RESEARCH.md).

## Demo videos
`docs/videos/demo_walkthrough.mp4` and `docs/videos/api_demo.mp4`, recorded unedited from the running app with Playwright.

## Honest limitations
- The scoring is hand-tuned rules plus bounded feedback adaptation. It is not a trained ranker, and there has been no real-user evaluation yet.
- Zero-shot tagging works best on product-style photos. I measured it on a small sample only, so I don't claim an accuracy figure.
- Identity is a URL token, not authentication. On the free hosting tier, data resets when the server restarts.
- The local 1.7B model is small: its phrasing is good in English, acceptable in Russian, and replaced by templates in Armenian. Replies take about 5–10 s on 2 CPU cores.
- Everything is free: no paid API or key is required, and nothing leaves the server except the MET Norway weather request.

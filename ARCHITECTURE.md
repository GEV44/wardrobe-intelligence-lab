# Architecture

## Components and data flow
```
User context (occasion, °C, style)
   │
Streamlit app.py ──┐                 ┌─ recommender.py  pure functions: generate → score → rank → diversify → explain
FastAPI  api.py ───┴─> service.py ───┼─ db.py           sqlite3, per-operation connections, parameterized SQL
                                     └─ images.py       upload validation, re-encoding, generated illustrations
```
Both front doors call the same `service` functions, so no recommendation logic is duplicated. The UI calls the service directly, which means the demo needs only one process.

## AI layer: model routing and fallback (implemented)
| Capability | Primary | Fallback | Guardrail |
|---|---|---|---|
| Upload tagging | Marqo FashionSigLIP int8 ONNX (precomputed text embeddings), then FashionCLIP int8 ONNX | Pillow dominant-color heuristic | Results are suggestions pre-filled into a form. The user confirms them, so a model error never writes silently. |
| Product cut-out | Classical border flood-fill (~50 ms) on plain backdrops | Original photo; BiRefNet-lite only if `LOOKROOM_BGREMOVE_ONLINE=1` | Routed by **latency budget**: BiRefNet measured ~45 s and ~4 GB per image on 2 threads, so it runs offline (catalog curation) |
| Live context | MET Norway weather, cached until `Expires` | Manual temperature slider | 4 s timeout, identifying User-Agent, ≤4-decimal coordinates per their terms |
| Catalog curation (offline) | Openverse CC0/PD/CC BY → FashionCLIP ranking → cut-out | Generated illustration if no confident photo | Any photo likely showing a person is rejected; credits are recorded |
| Natural-language styling | Claude Opus 5.5 with structured JSON output | Keyword context parser + rule engine | Item ids must exist, with exactly one top, bottom and shoes plus at most one outerwear. Refusals, `max_tokens`, bad JSON, timeouts and auth errors all fall back. |
| Ranking and scores | Deterministic engine | (none needed) | It is the single source of truth: LLM picks are re-scored by it. |

Design rule: **the LLM chooses, the engine verifies.** The model never sees images or personal data, only item metadata, and it cannot introduce an item that doesn't exist. Tests replace the Anthropic client with a fake, so CI is free and deterministic, and one opt-in live test (`RUN_LIVE_LLM=1`) checks the real API.

## Multi-tenancy, i18n and robustness (implemented)
- **Per-visitor isolation.** `wardrobe_items` has primary key `(user_id, id)`, and `outfit_feedback.user_id` is indexed. Every service and DB call takes `user_id`. The UI keeps a random id in the URL (`?u=`) and the API reads `X-User-Id`. Both are validated against `^[A-Za-z0-9_-]{1,40}$`. New users get the sample wardrobe on first visit. *Demo identity, not authentication.*
- **Languages.** `lookroom/i18n.py` holds the en/hy/ru strings, labels, explanation templates and localized sample-item names. The recommender's explanations are generated in the requested language, and Claude replies in it.
- **Failure isolation.** Each UI section runs inside a guard, so an exception shows a localized message and the rest of the page keeps working. Startup failure gives a clear message instead of a stack trace. A warm-up thread loads models once per process.
- **Spend guard.** An atomic SQLite counter caps paid Claude calls per UTC day (`LOOKROOM_LLM_DAILY_CAP`, default 150), and each browser session has its own limit (`LOOKROOM_SESSION_LLM_LIMIT`, default 20).
- **Deployment.** A pinned `Dockerfile` (python:3.11-slim, non-root user 1000) bakes the ONNX models into the image and has a healthcheck. It runs on Hugging Face Spaces (free CPU tier) with secrets in the Space settings.

## SQLite schema
- `wardrobe_items(id PK, name, category CHECK, color, style, warmth 1–5, formality 1–5, image_path, tags JSON, is_seed, created_at)`
- `outfit_feedback(id PK, item_ids JSON sorted, vote ±1, occasion, temperature, style, score_at_feedback, created_at)`

Item preferences are **derived** from the feedback log when a request comes in, not stored. The log is the source of truth, so it can be replayed or re-weighted later.

## Recommendation formula
`total = 100 × Σ wᵢ·cᵢ`, with weights occasion .30, weather .25, color .15, style .15, preference .15.

| Component | Rule (0–1) |
|---|---|
| occasion | mean over items of `1 − |formality − target|/4`; targets: casual 1.5, work 3.5, date 3.0, formal 4.5 |
| weather | target warmth `clamp(1.5 + (30 − T)·0.1, 1, 5)`; outfit warmth = weighted mean (outerwear ×2); `1 − |diff|/3` |
| color | mean pairwise: neutral+neutral 1.0, neutral+color 0.9, same color 0.75, palette pair 0.7, clash 0.4 |
| style | per item: exact 1.0, neighbor style 0.6, other 0.25 |
| preference | `0.5 + 0.5·mean(tanh(net_votes/2))`, where 0.5 means neutral |

Candidates are every top × bottom × shoes combination. Outerwear is required below 12 °C, optional from 12 to 20 °C, and excluded at 20 °C and above.

Selection is greedy over the score-sorted list. Each item shared with an already-picked outfit costs −4 pts and a repeated top costs another −6. Ties break on the sorted item ids, so the output is deterministic.

Edge cases:
- A missing required category returns a helpful message instead of outfits.
- Cold weather with no outerwear still recommends outfits and adds a warning.

## Feedback adaptation
This is *feedback-driven preference adaptation*, not model training. A vote gives ±1 to every item in the outfit. The `tanh` bound keeps any single item's signal inside (−1, 1), so personalization can move a total by at most ±7.5 points. Repeated votes saturate instead of exploding, and resetting the demo removes the effect completely.

## Privacy and security boundaries (demo)
- Parameterized SQL only.
- Uploads are checked for size (5 MB), pixel count (20 MP) and format (JPEG/PNG/WebP), verified with Pillow, EXIF-transposed and re-encoded to PNG under a `uuid4` name. The original filename is never used.
- Image paths are resolved against the images root, and anything outside it is rejected.
- No network calls, no secrets, no auth. **This is not production-secure.**

## Production path (not implemented)
- **1,000 users:** add `user_id` to every table with composite indexes. Run the API statelessly behind a load balancer and keep images in object storage (S3/GCS) with signed URLs. Candidate generation is O(T·B·S·O) per request; pre-filter by category and context, then cache top-N per (user, context bucket).
- **PostgreSQL migration:** the schema maps 1:1. Use Alembic migrations, `JSONB` for tags and item_ids (or an `outfit_feedback_items` join table), and row-level security per user.
- **Auth and isolation:** use an OIDC provider with short-lived JWTs and scope every query by `user_id` from the token, never from the request body. Encrypt photos at rest and give users data-export and delete endpoints.
- **Async AI inference:** an upload creates the item immediately, then a queue (SQS, Redis/RQ or Celery) runs vision tagging (for example FashionCLIP via ONNX). Results write back with `source=model` and a confidence score, and users confirm or correct them, which also produces labelled data.
- **Model routing and fallback:** use cheap deterministic rules first, a small model for tagging, and an LLM only for the chat or explanation layer, with structured outputs validated against the wardrobe (never invent items). Wrap calls in timeouts and circuit breakers and fall back to the rule engine.
- **Monitoring, evaluation and cost:** log every recommendation with context and component scores. Track like rate, top-1 acceptance and score drift. Offline-replay the feedback log against new weights and run A/B tests on weights. Track per-request model cost with budgets, cache LLM explanations and rate-limit per user.

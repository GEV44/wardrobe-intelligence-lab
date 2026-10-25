# Research notes (verified 2026-10-24)

## Verified facts
| Topic | Finding | Source |
|---|---|---|
| LookRoom product | Daily outfit suggestions (style, wardrobe, weather, schedule); AI avatar try-on; digital wardrobe auto-organized by category/season/color; AI stylist chat; trends; shopping suggestions; sale notifications. | https://lookroom.ai/ |
| Company | Yerevan, Armenia; B2C + B2B (fashion e-commerce partners); ~3 employees on LinkedIn. | https://linkedin.com/company/lookroom-ai |
| Role | Senior Backend Engineer – AI Systems & Platform. Flow: User → Personal Context → Digital Wardrobe → AI Intelligence → Decision → Action → Feedback → Deeper Personalization. Responsibilities: context-aware pipelines, model selection/routing, cost optimisation, failure fallbacks, external integrations (weather, calendar, retailers). Requirements: APIs, SQL/NoSQL, auth, cloud, Docker, queues/async, CI/CD, observability, privacy; "know the difference between MVP architecture and overengineering". Pre-MVP, equity-based, hybrid. | https://eraldt.com/jobs/backend-engineer-in-ai-systems-lookroom-yerevan-1e55944987c5419e , https://us.jobiglo.com/job-senior-backend-engineer-ai-systems-platform-j-6aa18d203b4a011c65065567 |
| FashionCLIP | `patrickjohncyh/fashion-clip`, **MIT** (HF card + GitHub). CLIP ViT-B/32 fine-tuned on 800K Farfetch products (FashionCLIP 2.0 from LAION checkpoint). Ships safetensors, PyTorch and **ONNX** weights. Card warns it was "not developed for model deployment" and is biased toward white-background product shots. | https://huggingface.co/patrickjohncyh/fashion-clip |
| Florence-2-base | `microsoft/Florence-2-base`, **MIT**, 0.23B params, captioning/OD/grounding via prompts. Requires `trust_remote_code=True` (executes repo Python), plus torch + transformers. | https://huggingface.co/microsoft/Florence-2-base |
| Reference repo | AMit090912/AI-Fashion-Outfit-Recommendation-System: FashionCLIP + FAISS retrieval + optional Gemini agent with offline fallback. No license reviewed, **no code copied**. | https://github.com/AMit090912/AI-Fashion-Outfit-Recommendation-System |
| Versions (PyPI) | fastapi 0.162.0, streamlit 1.66.1 (local 1.60.0 used), pillow 13.0.0 (local 12.2), uvicorn 0.54, pytest 9.1.1. | pypi.org |

Tooling note: Firecrawl ran out of credits, so I used Exa fetch plus the Hugging Face and PyPI JSON APIs.

## Inferences (not verified)
- Approximate CPU memory: FashionCLIP (~150M params) is about 0.6 GB fp32 for weights, about 1–1.5 GB with torch at runtime. Florence-2-base would need about 1–2 GB with torch. Both are feasible on a 16 GB CPU laptop but slow to load. I have not benchmarked either.
- The LookRoom stylist chat presumably uses hosted LLMs. Its internal architecture is unknown, and this demo makes no claims about it.

## Models integrated (measured on this laptop: Iris Xe, CPU only, about 0.6 GB RAM free)
| Model | How | Measured |
|---|---|---|
| FashionCLIP (`patrickjohncyh/fashion-clip`, MIT) | Official `onnx/model.onnx` (606 MB), dynamically quantized to int8 with `onnxruntime.quantization` (153 MB). Zero-shot over 21 garment types, 15 colors and 5 styles, run in a single `onnxruntime` call. No torch and no remote code. | Load 1.1 s · RSS ~250 MB · 140–210 ms/image. Public-domain sneaker photo → shoes / white (88% / 90%). Seed drawings: 12/14 categories correct, and both misses were coats drawn flat that read as shirts. |
| Claude Opus 5.5 (`claude-opus-5-5`, Anthropic API) | Single call with structured output (`json_schema`), `effort: low`, and server-side refusal fallback (`fallbacks: "default"`). Grounded on wardrobe JSON plus the engine's top 5. Output is validated and re-scored, with a rule-engine fallback. | ~5 s/message, ~2.3k input and ~0.2k output tokens (~$0.01–0.02). Extracted "dinner date, 8 degrees, classic" → date / 8 °C / classic with a valid 4-piece outfit. |

## Round 2: strongest permissive models per task (measured 2026-10-25, same laptop)
| Task | Model (license) | Result |
|---|---|---|
| Garment tagging (primary) | `Marqo/marqo-fashionSigLIP` (Apache-2.0), repo's int8 vision ONNX. Text prompts are pre-encoded once with the fp32 text tower, because the int8 text tower cost 1 color hit. | Same accuracy as FashionCLIP on my 15-image check (13/15 category, 12/15 color) but ~1.7× faster (~460 vs ~800 ms), loads in 0.7–1.2 s vs ~3 s, and uses 193 vs 264 MB RSS. Marqo reports +57% text-to-image recall vs FashionCLIP 2.0 (their benchmark). |
| Background removal | `ZhengPeng7/BiRefNet_lite` (MIT), ONNX export `wonfy/BiRefNet_lite-ONNX-webgpu` (MIT) | Clean cut-outs, but ~45 s/image and ~4.2 GB peak at 2 threads (input fixed at 1024²). fp16 and int8 gave no speedup. **Offline only.** The request path uses a classical border flood-fill (~50 ms) that only fires on plain backdrops. |
| Rejected as non-commercial | `briaai/RMBG-1.4`, `briaai/RMBG-2.0` (CC BY-NC), `onnx-community/ISNet-ONNX` (AGPL) | Not used |
| Weather | MET Norway locationforecast 2.0 (free, commercial use OK, CC BY 4.0) | Chosen over Open-Meteo, whose free tier is non-commercial only |
| Sample photos | Openverse API, filtered to `cc0,pdm,by` (no NC/ND/SA) | Curated by FashionCLIP with people rejected. See `data/catalog/CREDITS.md`. |

## Evaluated, not integrated
| Model | Why not | Integration path |
|---|---|---|
| Florence-2-base | `trust_remote_code` runs third-party code, and it needs torch | Pin a commit hash and sandbox it in a worker process for captioning or attribute extraction |

## Licensing caveats
MIT model licenses do not cover the *training data*: FashionCLIP was trained on Farfetch data. Commercial use needs a legal review. Open weights also do not mean free hosted inference.

## Deferred until after the interview
Weather and calendar APIs, multi-user auth, Postgres, image embeddings for visual similarity and duplicate detection, a learned ranker trained on the feedback log, and an evaluation set for the stylist.

## Demo asset licensing
The only real photo is a public-domain sneaker image from Wikimedia Commons (see `demo_assets/CREDITS.md`). I rejected a candidate t-shirt photo because it showed a real person.

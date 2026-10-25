# Demo script

## 30-second pitch
"LookRoom's promise is fewer daily decisions. I built a working version of that intelligence loop. Context and wardrobe go into an explainable scoring engine, which returns three outfits, and every Like or Dislike measurably changes future rankings. On top of that sit pretrained models, each routed by latency budget with a real fallback. Marqo FashionSigLIP runs locally as int8 ONNX on a CPU and recognises uploaded garments. BiRefNet curated the real product-photo catalog offline. A free local Qwen3 LLM is the conversational stylist: it understands the request and phrases the advice, while the engine chooses only from the clothes you actually own. It needs no API key and costs nothing. It also uses live weather, works in English, Armenian and Russian, gives each visitor a private workspace, and runs 24/7 on Hugging Face from a pinned Docker image."

## 2-minute walkthrough
1. **Reset**: tick *Confirm* in the sidebar and click **Reset Demo** to start from a known state.
2. **Context**: set Work, 16 °C, Smart. Show the three looks, each score's breakdown bars, and the "Why" explanation.
3. **Context sensitivity**: drag the temperature to 2 °C. Outerwear appears and the weather fit changes. Switch Occasion to Casual and the formality scores shift.
4. **Personalization**: click **Dislike** on Look 1. The toast shows the score drop, for example 84.9 → 81.5, and the ranking reorders.
5. **How It Learns**: show the counts, the before/after ranking, the item signal table and the bounded formula (±7.5 pts maximum).
5b. **Plan my week**: seven looks from the MET Norway daily forecast, an occasion per day, and no item repeated on consecutive days. Explain that this is the same engine with a repeat penalty: deterministic, greedy and explainable.
5c. **Shop the gap**: 576 real catalog products, each tested against the wardrobe by the same engine. It shows the score gain, how many occasions improve, and what it pairs with. "Add to my wardrobe" closes the loop, which is LookRoom's "what to buy" made measurable.
6. **AI Stylist (free, local)**: click "Dinner date, 8°C, classic". The keyword rules plus the local Qwen3-1.7B extract date / 8 °C / classic, the engine picks real items, and Qwen phrases the reply. No API key and no cost. Show the "engine score" badge and Like the result, which closes the loop.
7. **Wardrobe**: upload `demo_assets/sneakers_public_domain.jpg`. FashionCLIP suggests White Sneakers · shoes · white in about 0.2 s. Review the suggestion, add it, and show it in the grid.
8. **API**: open `http://127.0.0.1:8010/docs`, run `GET /recommendations` and `POST /stylist/chat`.

Backup: if Wi-Fi or the API key fails during the interview, the stylist still answers through the rule engine, and its badge says so. That itself demonstrates the fallback design. The recorded videos are in `docs/videos/`.

## Likely questions
1. **Why not a neural network?** There is no training data at cold start, and the rules are debuggable and explainable. The feedback log I store is exactly the dataset a learned ranker would need later, for example logistic regression or LightGBM over the component scores plus item embeddings.
2. **How does feedback avoid runaway effects?** Each item's net votes pass through `tanh`, and the preference component is capped at a 15% weight, so the maximum swing is ±7.5 points and it is reversible.
3a. **Why a 1.7B local model and not GPT/Claude?** Zero cost, no key, and data stays on the server. I narrowed the LLM's job to what small models do well: slot filling with a JSON grammar, and short phrasing. The engine does the decision, and explicit keyword matches override the model. A paid model is a config switch, not a rewrite.
3. **How do you stop the LLM from hallucinating clothes?** In the default free path it never chooses items: the engine does. In the optional Claude path, Claude gets the wardrobe as JSON plus the engine's top 5, and answers with a strict JSON schema. I validate every id and the category structure, then re-score the outfit with the engine. Anything invalid, a refusal, a timeout or a missing key falls back to the rule engine. Tests fake the client to cover each of these failure modes.
3b. **Why FashionCLIP over Florence-2?** It is MIT licensed, ships an official ONNX export (so no torch and no `trust_remote_code`), and is domain-tuned for fashion. Int8 quantization took it from 606 MB to 153 MB, with ~250 MB RAM and ~0.2 s per image on this laptop.
4. **How does it scale to many users?** Add per-user tenancy, Postgres with row-level security, object storage for images, async workers for vision, and caching of top-N per context bucket. Details are in ARCHITECTURE.md.
5. **How do you know recommendations are good?** I don't claim accuracy yet. I would log impressions and votes, track like rate and top-1 acceptance, replay the log offline for weight changes, and A/B test.

## Where this stands vs. LookRoom's public product (as of 2026-10-25)
LookRoom announced an internal pre-MVP release on 2026-10-01 and has a waitlist landing page; I found no public app-store listing. Its site claims daily outfits (style, wardrobe, weather, schedule), a digital wardrobe, AI stylist chat, a 3D avatar with virtual try-on, trends, shopping and sales alerts.

| Claimed capability | This demo |
|---|---|
| Daily outfit suggestions (weather, schedule) | Implemented: Today's look, live MET Norway weather, a 7-day plan with per-day occasions |
| Digital wardrobe, auto-organized | Implemented: upload → cut-out → FashionSigLIP tagging, with category filters |
| AI stylist chat | Implemented: free local Qwen3 + engine, multilingual |
| Shopping suggestions | Implemented: "Shop the gap" with measurable score gain, plus visual-similarity search |
| Learns preferences | Implemented and visible: bounded feedback adaptation and an embedding taste vector |
| Avatar / virtual try-on, trends, sales alerts | **Not implemented.** Try-on needs GPU diffusion models; it's a roadmap item |

How to say it in the interview: *"I built the decision engine and data loop behind the product claims. Try-on is the part I deliberately left out, because doing it properly needs GPU infrastructure."* Don't claim it's "better than LookRoom". Show that it works.

## Honest limitations
1. Rules are hand-tuned and there is no real-user evaluation.
2. Feedback credits every item in an outfit equally, so it misses pairwise effects ("this shirt with these trousers").
3. Single user, no auth. FashionCLIP is zero-shot and works best on product-style photos. I measured accuracy only on a small sample, so I don't claim an accuracy figure.

## Roadmap
Move FashionCLIP into an async worker queue → FashionCLIP image embeddings for visual similarity → weather and calendar context → a learned ranker trained on the feedback log → an eval set and prompt caching for the stylist → multi-user Postgres, auth and observability.

## Fallback if the live UI fails
1. Run `python scripts/smoke_test.py`, which prints three outfits and a score change from feedback.
2. Run `uvicorn api:app --port 8010` and demo from `/docs`.
3. Run `python -m pytest -q` (53 tests), or play `docs/videos/demo_walkthrough.mp4`.
4. Walk through `lookroom/recommender.py` directly.

"""FastAPI interface over the same service layer used by the Streamlit app.

Run: uvicorn api:app --reload   ->  http://127.0.0.1:8000/docs
Demo only: no authentication, single local user.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from lookroom import __version__, db, images, service, stylist, vision
from lookroom.models import TEMP_MAX, TEMP_MIN, validate_context
from lookroom.recommender import WEIGHTS, Outfit

Occasion = Literal["casual", "work", "date", "formal"]
Style = Literal["casual", "classic", "minimal", "smart", "streetwear"]
Lang = Literal["en", "hy", "ru"]


def current_user(x_user_id: str | None = Header(None, description="Optional tenant id; omitted = shared demo user")) -> str:
    """Per-user isolation. Demo-grade identity (no auth); production derives this from a verified token."""
    uid = service.clean_user_id(x_user_id)
    service.ensure_ready(uid)
    return uid


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.ensure_ready()
    yield


app = FastAPI(title="Wardrobe Intelligence Lab API", version=__version__, lifespan=lifespan,
              description="Independent LookRoom-inspired concept demo. Local, unauthenticated, not production.")


class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str
    items: int
    vision_available: bool
    bgremove_available: bool
    llm_available: bool
    llm_calls_today: int


class ItemOut(BaseModel):
    id: str
    name: str
    category: str
    color: str
    style: str
    warmth: int
    formality: int
    tags: list[str]
    is_seed: bool


class BreakdownOut(BaseModel):
    occasion: float = Field(ge=0, le=1)
    weather: float = Field(ge=0, le=1)
    color: float = Field(ge=0, le=1)
    style: float = Field(ge=0, le=1)
    preference: float = Field(ge=0, le=1)


class OutfitOut(BaseModel):
    rank: int
    item_ids: list[str]
    items: list[ItemOut]
    score: float = Field(ge=0, le=100)
    breakdown: BreakdownOut
    explanation: str


class RecommendationsOut(BaseModel):
    occasion: Occasion
    temperature: float
    style: Style
    weights: dict[str, float]
    outfits: list[OutfitOut]
    message: str | None
    candidates_considered: int


class FeedbackIn(BaseModel):
    item_ids: list[str] = Field(min_length=3, max_length=4)
    vote: Literal["like", "dislike"]
    occasion: Occasion
    temperature: float = Field(ge=TEMP_MIN, le=TEMP_MAX)
    style: Style


class FeedbackOut(BaseModel):
    feedback_id: int
    score_before: float
    score_after: float


class ItemPreferenceOut(BaseModel):
    item_id: str
    name: str
    style: str
    likes: int
    dislikes: int
    net: int
    signal: float


class FeedbackSummaryOut(BaseModel):
    likes: int
    dislikes: int
    items: list[ItemPreferenceOut]
    style_affinity: dict[str, int]


def _item_out(it) -> ItemOut:
    return ItemOut(**{k: getattr(it, k) for k in ItemOut.model_fields if k != "tags"}, tags=list(it.tags))


def _outfit_out(rank: int, o: Outfit) -> OutfitOut:
    return OutfitOut(rank=rank, item_ids=list(o.key), items=[_item_out(i) for i in o.items], score=o.score,
                     breakdown=BreakdownOut(**o.breakdown.as_dict()), explanation=o.explanation)


@app.get("/health", response_model=HealthOut)
def health():
    ai = service.ai_status()
    return HealthOut(status="ok", version=__version__, items=len(db.list_items()),
                     vision_available=ai["vision"]["available"], bgremove_available=ai["bgremove"]["available"],
                     llm_available=ai["llm"], llm_calls_today=db.llm_calls_today())


@app.get("/wardrobe", response_model=list[ItemOut])
def wardrobe(user: str = Depends(current_user)):
    return [_item_out(i) for i in db.list_items(user)]


@app.get("/recommendations", response_model=RecommendationsOut)
def recommendations(occasion: Occasion = "casual",
                    temperature: float = Query(18, ge=TEMP_MIN, le=TEMP_MAX),
                    style: Style = "casual", lang: Lang = "en", user: str = Depends(current_user)):
    ctx = validate_context(occasion, temperature, style)
    res = service.get_recommendations(ctx, user, lang)
    return RecommendationsOut(occasion=occasion, temperature=temperature, style=style, weights=WEIGHTS,
                              outfits=[_outfit_out(i + 1, o) for i, o in enumerate(res.outfits)],
                              message=res.message, candidates_considered=res.candidates_considered)


@app.post("/feedback", response_model=FeedbackOut, status_code=201)
def feedback(body: FeedbackIn, user: str = Depends(current_user)):
    ctx = validate_context(body.occasion, body.temperature, body.style)
    try:
        return FeedbackOut(**service.record_feedback(body.item_ids, body.vote, ctx, user))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/feedback-summary", response_model=FeedbackSummaryOut)
def feedback_summary(user: str = Depends(current_user)):
    return FeedbackSummaryOut(**service.feedback_summary(user))


class ShopOut(BaseModel):
    product_id: str
    name: str
    category: str
    color: str
    style: str
    gain: float
    score_with_product: float
    current_best: float
    occasions_improved: int = Field(ge=0, le=4)
    pairs_with: list[str]


@app.get("/shop/suggestions", response_model=list[ShopOut])
def shop_suggestions(occasion: Occasion = "casual", temperature: float = Query(18, ge=TEMP_MIN, le=TEMP_MAX),
                     style: Style = "casual", n: int = Query(6, ge=1, le=20), user: str = Depends(current_user)):
    """Real catalog products ranked by how much they would raise this user's best outfit score."""
    out = service.shop_suggestions(validate_context(occasion, temperature, style), user, n)
    return [ShopOut(product_id=s["product"].id, name=s["product"].name, category=s["product"].category,
                    color=s["product"].color, style=s["product"].style, gain=s["gain"], score_with_product=s["score"],
                    current_best=s["base"], occasions_improved=s["occasions_improved"],
                    pairs_with=[i.id for i in s["pairs_with"]]) for s in out]


class DayIn(BaseModel):
    occasion: Occasion
    temperature: float = Field(ge=TEMP_MIN, le=TEMP_MAX)


class WeekIn(BaseModel):
    days: list[DayIn] = Field(min_length=1, max_length=14)
    style: Style = "casual"
    lang: Lang = "en"


@app.post("/plan/week", response_model=list[OutfitOut | None])
def plan_week(body: WeekIn, user: str = Depends(current_user)):
    """One outfit per day, avoiding repeats on consecutive days."""
    plan = service.plan_week([validate_context(d.occasion, d.temperature, body.style) for d in body.days], user, body.lang)
    return [_outfit_out(i + 1, o) if o else None for i, o in enumerate(plan)]


class TagOut(BaseModel):
    source: Literal["fashionclip", "heuristic"]
    model: str
    background_removed: bool
    category: str | None
    garment: str | None
    color: str
    style: str | None
    warmth: int | None
    formality: int | None
    suggested_name: str | None
    confidence: dict[str, float]
    latency_ms: int


@app.post("/vision/tag", response_model=TagOut)
def vision_tag(file: UploadFile = File(..., description="JPEG, PNG or WebP, max 5 MB")):
    """Suggest attributes for a garment photo (FashionCLIP zero-shot if installed, else color heuristic)."""
    data = file.file.read(images.MAX_UPLOAD_BYTES + 1)
    try:
        clean, removed = service.prepare_upload(data)
        return TagOut(**service.tag_image(clean), background_removed=removed)
    except images.ImageValidationError as exc:
        raise HTTPException(422, str(exc)) from exc


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    occasion: Occasion = "casual"
    temperature: float = Field(18, ge=TEMP_MIN, le=TEMP_MAX)
    style: Style = "casual"
    lang: Lang = "en"


class ChatOut(BaseModel):
    reply: str
    source: Literal["claude", "rules"]
    occasion: Occasion
    temperature: float
    style: Style
    outfit: OutfitOut | None
    latency_ms: int
    note: str | None


@app.post("/stylist/chat", response_model=ChatOut)
def stylist_chat(body: ChatIn, user: str = Depends(current_user)):
    """Natural-language styling. Claude chooses from the real wardrobe; the engine validates and scores it."""
    r = stylist.chat(body.message, [t.model_dump() for t in body.history],
                     validate_context(body.occasion, body.temperature, body.style), user_id=user, lang=body.lang)
    return ChatOut(reply=r.text, source=r.source, occasion=r.context.occasion, temperature=r.context.temperature,
                   style=r.context.style, outfit=_outfit_out(1, r.outfit) if r.outfit else None,
                   latency_ms=r.latency_ms, note=r.note)

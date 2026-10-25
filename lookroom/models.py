"""Domain models and validation helpers."""
from __future__ import annotations

from dataclasses import dataclass, field

CATEGORIES = ("top", "bottom", "dress", "shoes", "outerwear")
REQUIRED_CATEGORIES = ("top", "bottom", "shoes")   # the two-piece outfit shape
DRESS_OUTFIT = ("dress", "shoes")                  # the one-piece outfit shape
DISPLAY_ORDER = {c: i for i, c in enumerate(("top", "dress", "bottom", "shoes", "outerwear"))}


def is_valid_outfit(categories: list[str]) -> bool:
    """Exactly one of each piece and either top+bottom+shoes or dress+shoes; optional single outerwear."""
    if len(categories) != len(set(categories)):
        return False
    core = set(categories) - {"outerwear"}
    return core in ({"top", "bottom", "shoes"}, {"dress", "shoes"})
OCCASIONS = ("casual", "work", "date", "formal")
STYLES = ("casual", "classic", "minimal", "smart", "streetwear")

COLOR_HEX = {
    "white": "#F5F5F2", "black": "#262626", "grey": "#8C8C8C", "cream": "#EFE6D2",
    "beige": "#D8C3A0", "camel": "#C19A6B", "brown": "#7B5236", "navy": "#22304F",
    "blue": "#4A78B5", "olive": "#6B7043", "green": "#3E7D4F", "red": "#B23A3A",
    "burgundy": "#6E1F2E", "pink": "#E3A6B5", "yellow": "#E8C547",
}
COLORS = tuple(COLOR_HEX)

TEMP_MIN, TEMP_MAX = -30.0, 45.0
NAME_MAX_LEN = 60


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    category: str
    color: str
    style: str
    warmth: int
    formality: int
    image_path: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    is_seed: bool = False


@dataclass(frozen=True)
class Context:
    occasion: str
    temperature: float
    style: str


def _check_choice(value: str, allowed: tuple[str, ...], label: str) -> str:
    v = str(value).strip().lower()
    if v not in allowed:
        raise ValueError(f"Invalid {label} '{value}'. Allowed: {', '.join(allowed)}")
    return v


def _check_level(value, label: str) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be an integer 1-5") from None
    if not 1 <= v <= 5:
        raise ValueError(f"{label} must be between 1 and 5")
    return v


def validate_item(data: dict) -> Item:
    """Build a validated Item from untrusted input (form, JSON, API)."""
    item_id = str(data.get("id", "")).strip()
    if not item_id or len(item_id) > 64 or not all(c.isalnum() or c in "-_" for c in item_id):
        raise ValueError("Item id must be 1-64 chars: letters, digits, '-' or '_'")
    name = str(data.get("name", "")).strip()
    if not name or len(name) > NAME_MAX_LEN:
        raise ValueError(f"Name is required (max {NAME_MAX_LEN} chars)")
    tags = tuple(str(t).strip().lower() for t in data.get("tags", ()) if str(t).strip())
    return Item(
        id=item_id,
        name=name,
        category=_check_choice(data.get("category", ""), CATEGORIES, "category"),
        color=_check_choice(data.get("color", ""), COLORS, "color"),
        style=_check_choice(data.get("style", ""), STYLES, "style"),
        warmth=_check_level(data.get("warmth"), "warmth"),
        formality=_check_level(data.get("formality"), "formality"),
        image_path=data.get("image_path") or None,
        tags=tags,
        is_seed=bool(data.get("is_seed", False)),
    )


def validate_context(occasion: str, temperature: float, style: str) -> Context:
    t = float(temperature)
    if not TEMP_MIN <= t <= TEMP_MAX:
        raise ValueError(f"Temperature must be between {TEMP_MIN} and {TEMP_MAX} °C")
    return Context(
        occasion=_check_choice(occasion, OCCASIONS, "occasion"),
        temperature=t,
        style=_check_choice(style, STYLES, "style"),
    )

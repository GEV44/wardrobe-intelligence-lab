"""Streamlit UI for Wardrobe Intelligence Lab. Run: streamlit run app.py"""
from __future__ import annotations

import html
import logging
import os
from contextlib import contextmanager

import streamlit as st

from lookroom import db, i18n, images, seed, service, stylist, weather
from lookroom.models import CATEGORIES, COLORS, DISPLAY_ORDER, OCCASIONS, STYLES, validate_context
from lookroom.recommender import PREFERENCE_SCALE, WEIGHTS

st.set_page_config(page_title="Wardrobe Intelligence Lab", page_icon=":material/checkroom:", layout="wide",
                   initial_sidebar_state="expanded")
log = logging.getLogger("wil")
SESSION_LLM_LIMIT = int(os.environ.get("LOOKROOM_SESSION_LLM_LIMIT", "20"))
LIVE_WEATHER_DEFAULT = os.environ.get("LOOKROOM_LIVE_WEATHER_DEFAULT") == "1"
ss = st.session_state

# ---------- identity (private workspace per visitor) & language, both kept in the URL ----------
_raw_uid = st.query_params.get("u")
if _raw_uid and service.clean_user_id(_raw_uid) == _raw_uid:
    UID = _raw_uid
else:
    UID = service.new_user_id()
    st.query_params["u"] = UID
LANG = i18n.norm(st.query_params.get("lang"))


def T(key: str, **kw) -> str:
    return i18n.t(key, LANG, **kw)


def L(kind: str, value: str) -> str:
    return i18n.label(kind, value, LANG)


def name_of(item) -> str:
    return i18n.item_name(item, LANG)


st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@500;600&family=Noto+Sans+Armenian:wght@400;500;600&family=Noto+Serif+Armenian:wght@500;600&display=swap');
:root {--ink:#141414; --muted:#6b665f; --line:#ebe5dc; --accent:#9a7650; --accent-soft:#f4ede4; --bg:#f8f6f2;
       --card:#ffffff; --radius:20px; --good:#3f7d58; --bad:#a5483d;
       --shadow:0 1px 2px rgba(20,20,20,.04), 0 8px 28px rgba(20,20,20,.06);
       --shadow-hover:0 2px 6px rgba(20,20,20,.06), 0 18px 40px rgba(20,20,20,.10);}
html, body, [class*="css"], .stMarkdown, .stButton button, input, textarea, label {
  font-family: 'Inter', 'Noto Sans Armenian', system-ui, sans-serif;}
.stApp {background: radial-gradient(1200px 500px at 85% -10%, #efe6da 0%, rgba(239,230,218,0) 60%), var(--bg);}
h1, h2, h3, .serif {font-family: 'Playfair Display', 'Noto Serif Armenian', Georgia, serif !important;
  font-weight: 600 !important; letter-spacing: -.015em; color: var(--ink);}
h3 {font-size: 1.45rem !important; margin-top: 1.4rem !important;}
.block-container {padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1380px;}
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer,
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none !important;}
/* --- cards: every keyed container (st-key-card*, ctxbar) --- */
[class*="st-key-card"], .st-key-ctxbar {background: var(--card); border: 1px solid var(--line) !important;
  border-radius: var(--radius) !important; box-shadow: var(--shadow); transition: box-shadow .25s ease, transform .25s ease;}
[class*="st-key-card"]:hover {box-shadow: var(--shadow-hover); transform: translateY(-2px);}
.st-key-card-hero:hover {transform: none;}
.st-key-ctxbar {padding: .35rem .6rem; margin-bottom: .2rem;}
/* --- imagery --- */
[data-testid="stImage"] img {border-radius: 14px;}
/* --- pill navigation (tabs) --- */
[data-testid="stTabs"] [role="tablist"] {gap: .35rem; background: rgba(255,255,255,.7); padding: .35rem;
  border: 1px solid var(--line); border-radius: 999px; width: fit-content; max-width: 100%; overflow-x: auto;
  backdrop-filter: blur(6px);}
[data-testid="stTab"] {border-radius: 999px !important; padding: .45rem 1rem !important; height: auto !important;
  border: none !important; transition: background .2s ease;}
[data-testid="stTab"] p {font-size: .92rem !important; font-weight: 500 !important; color: #3d3a35;}
[data-testid="stTab"][aria-selected="true"] {background: var(--ink) !important;}
[data-testid="stTab"][aria-selected="true"] p, [data-testid="stTab"][aria-selected="true"] span {color: #fff !important;}
[data-testid="stTab"]:hover {background: var(--accent-soft);}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {display: none !important;}
[data-testid="stTabPanel"] {padding-top: 1.1rem;}
/* --- buttons: black pills --- */
.stButton button, [data-testid="stPopover"] button, .stFormSubmitButton button {border-radius: 999px !important;
  font-weight: 500; transition: all .2s ease;}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {background: var(--ink) !important;
  border-color: var(--ink) !important; color: #fff !important;}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {background: #000 !important;
  box-shadow: 0 6px 18px rgba(0,0,0,.18);}
[data-testid="stBaseButton-secondary"] {background: #fff !important; border-color: var(--line) !important;}
[data-testid="stBaseButton-secondary"]:hover {border-color: var(--ink) !important; color: var(--ink) !important;}
[data-testid="stButtonGroup"] button {border-radius: 999px !important; margin: 0 .2rem .3rem 0;}
button[aria-checked="true"] {background: var(--ink) !important; color: #fff !important;
  border-color: var(--ink) !important;}
button[aria-checked="true"] p, button[aria-checked="true"] span {color: #fff !important;}
button[aria-checked="false"] {background: #fff !important; border-color: var(--line) !important;}
button[aria-checked="false"]:hover {border-color: var(--ink) !important;}
[data-testid="stTab"]::after, [data-testid="stTab"]::before {display: none !important;}
[data-testid="stTab"] {border-bottom: none !important; box-shadow: none !important;}
[data-testid="stTabs"] [role="tablist"] > :not([data-testid="stTab"]) {display: none !important;}
[data-testid="stWidgetLabel"] p {font-size: .74rem !important; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted) !important; font-weight: 600 !important;}
[data-testid="stExpander"] details {border-radius: 14px !important; border-color: var(--line) !important;}
[data-testid="stChatMessage"] {background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 1rem;}
.wil-statusline {margin: .1rem 0 .6rem .2rem;}
.st-key-mainnav [data-testid="stButtonGroup"] {background: rgba(255,255,255,.75); border: 1px solid var(--line);
  border-radius: 999px; padding: .35rem; width: fit-content; max-width: 100%; backdrop-filter: blur(6px);}
.st-key-mainnav [data-testid="stButtonGroup"] button {border: none !important; padding: .45rem 1.05rem !important;
  margin: 0 !important;}
.st-key-mainnav [data-testid="stButtonGroup"] button[aria-checked="false"] {background: transparent !important;}
.st-key-mainnav [data-testid="stButtonGroup"] button[aria-checked="false"]:hover {background: var(--accent-soft) !important;}
.st-key-mainnav {margin-bottom: 1rem;}
.wil-lead {font-size: 1.02rem; color: #3d3a35; max-width: 860px; margin: .2rem 0 .2rem 0; line-height: 1.55;}
.wil-title {font: 600 1.45rem/1.2 'Playfair Display', 'Noto Serif Armenian', serif; color: var(--ink); margin: .25rem 0 .55rem 0;}
.wil-gain {font: 600 3.1rem/1 'Playfair Display', serif; color: var(--ink); margin: .7rem 0 .25rem 0;}
.wil-gain span {font: 500 .95rem 'Inter', sans-serif; color: var(--good); margin-left: .45rem;}
.wil-row {display: flex; align-items: center; gap: .6rem; margin: .5rem 0;}
.wil-dots {display: inline-flex; gap: 5px;}
.wil-dot {width: 9px; height: 9px; border-radius: 50%; background: #e6dfd4;}
.wil-dot.on {background: var(--accent);}
.wil-minis {display: flex; flex-wrap: wrap; gap: .25rem; margin: .2rem 0 .4rem 0;}
.wil-mini {font-size: .72rem; padding: .12rem .45rem; border-radius: 6px; background: #f4f1ec; color: #4a4640;}
.wil-item-name {display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;}
.wil-step {background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); padding: 1.1rem 1.2rem;
  box-shadow: var(--shadow); height: 100%;}
.wil-step-n {font: 600 .8rem 'Inter', sans-serif; color: var(--accent); letter-spacing: .12em;}
.wil-step-t {font: 600 1.15rem 'Playfair Display', 'Noto Serif Armenian', serif; color: var(--ink); margin: .3rem 0;}
.wil-step-d {font-size: .88rem; color: var(--muted); line-height: 1.5;}
.bd-fill.neg {background: linear-gradient(90deg, #a9a093, #cfc7bb);}
.bd-track.div {position: relative;}
.wil-gain span.down {color: #b0503c;}
.bd-track.div .bd-fill {position: absolute; top: 0;}
.bd-track.div::after {content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px; width: 2px; background: var(--ink); opacity: .25;}
.wil-swatches {display: flex; flex-wrap: wrap; gap: .9rem 1.2rem; align-items: flex-end; margin-top: .4rem;}
.wil-swatch {display: flex; flex-direction: column; align-items: center; gap: .3rem;}
.wil-swatch span {border-radius: 50%; border: 1px solid rgba(0,0,0,.12); box-shadow: inset 0 0 0 3px rgba(255,255,255,.6);}
.wil-swatch em {font-style: normal; font-size: .74rem; color: var(--muted);}
.wil-rank {display: grid; grid-template-columns: 1.3rem 1fr auto; gap: .5rem; align-items: baseline;
  font-size: .84rem; color: #4a4640; padding: .3rem 0; border-bottom: 1px dashed var(--line);}
.wil-rank b {color: var(--accent);}
.wil-rank i {font-style: normal; font-weight: 600; color: var(--ink);}
.wil-rank.now b {color: var(--ink);}
.wil-timeline {display: flex; flex-direction: column; gap: .55rem;}
.wil-tl {display: flex; gap: .8rem; align-items: center; background: var(--card); border: 1px solid var(--line);
  border-radius: 14px; padding: .65rem .9rem;}
.wil-tl-dot {width: 30px; height: 30px; flex: 0 0 30px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-size: .85rem; color: #fff;}
.wil-tl-dot.up {background: var(--ink);}
.wil-tl-dot.down {background: #b9b2a8;}
.wil-tagline {font-size: .86rem; color: var(--muted); font-family: 'Playfair Display', 'Noto Serif Armenian', serif;
  font-style: italic;}
.wil-brand {display:flex; align-items:center; gap:.8rem;}
.wil-mark {width:42px; height:42px; border-radius:12px; background:var(--ink); color:#fff; display:flex;
  align-items:center; justify-content:center; font:600 16px 'Playfair Display', serif; letter-spacing:.04em;
  box-shadow: 0 6px 16px rgba(0,0,0,.18);}
.wil-name {font:600 1.08rem 'Inter', sans-serif; color:var(--ink); line-height:1.15;}
.wil-eyebrow {font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:var(--accent);}
.wil-hero h1 {font-size: clamp(2rem, 3.6vw, 3.1rem); margin: .2rem 0 .3rem 0; line-height:1.08;}
.wil-sub {color:var(--muted); font-size:1.04rem; max-width: 760px; margin-bottom: .4rem;}
.wil-tiles {display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:.75rem; margin:.9rem 0 .4rem 0;}
.wil-tile {background:#fff; border:1px solid var(--line); border-radius:14px; padding:.8rem 1rem;}
.wil-tile .k {font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em;}
.wil-tile .v {font-size:1.35rem; font-weight:600; color:var(--ink); margin-top:.15rem;}
.wil-chip {display:inline-block; padding:.18rem .6rem; border-radius:999px; background:var(--accent-soft);
  color:#5e4733; font-size:.78rem; margin:0 .3rem .3rem 0; border:1px solid #e6d9c9;}
.wil-chip.dark {background:var(--ink); color:#fff; border-color:var(--ink);}
.wil-score {display:flex; align-items:center; gap:.9rem; margin:.2rem 0 .6rem 0;}
.wil-score .lbl {font-size:.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em;}
.wil-why {font-size:.98rem; line-height:1.55; color:#34312d; margin:.3rem 0 .8rem 0;}
.bd-row {display:flex; justify-content:space-between; font-size:.82rem; color:#4a4640; margin-top:.45rem;}
.bd-track {height:7px; background:#efeae3; border-radius:99px; overflow:hidden; margin-top:.22rem;}
.bd-fill {height:100%; background:linear-gradient(90deg, #b08e6d, #8a6a4f); border-radius:99px;}
.wil-item-name {font-weight:600; font-size:.92rem; color:var(--ink); margin-top:.3rem; line-height:1.25;}
.wil-item-meta {font-size:.78rem; color:var(--muted);}
.wil-footer {margin-top:2.2rem; padding-top:1rem; border-top:1px solid var(--line); font-size:.78rem; color:var(--muted);}
.wil-note {font-size:.8rem; color:var(--muted);}
div[data-testid="stTabs"] button p {font-size: 1rem; font-weight: 500;}
@media (max-width: 760px) {.wil-tiles {grid-template-columns:repeat(2, minmax(0,1fr));}}
</style>""", unsafe_allow_html=True)

if LANG == "hy":  # Armenian capitals are hard to scan in small eyebrow labels: keep natural case
    st.markdown("<style>.wil-eyebrow,.wil-tile .k,.wil-score .lbl,[data-testid='stWidgetLabel'] p"
                "{text-transform:none !important;letter-spacing:.01em !important;font-size:.82rem !important;}</style>",
                unsafe_allow_html=True)

try:
    service.ensure_ready(UID)
except Exception:
    log.exception("startup failed")
    st.error("The demo database could not be initialised. Please reload the page.")
    st.stop()

if flash := ss.pop("flash", None):
    st.toast(flash)


@st.cache_resource(show_spinner=False)
def _warmup() -> bool:
    """Once per server process: load ML models and the API SDK in the background so first use is fast."""
    import threading

    def run() -> None:
        for step in (lambda: __import__("lookroom.vision", fromlist=["x"]).ensure_siglip(),
                     lambda: __import__("lookroom.vision", fromlist=["x"]).get_tagger(),
                     lambda: __import__("lookroom.llm", fromlist=["x"]).warmup()):
            try:
                step()
            except Exception:
                log.warning("warmup step failed", exc_info=True)

    threading.Thread(target=run, name="wil-warmup", daemon=True).start()
    return True


_warmup()


@contextmanager
def safe(section: str):
    """Isolate failures: one broken section never takes down the page."""
    try:
        yield
    except Exception as exc:
        if type(exc).__name__ in ("RerunException", "StopException"):  # Streamlit control flow
            raise
        log.exception("section %s failed", section)
        st.error(T("section_error") + f" ({type(exc).__name__})")


@st.cache_data(ttl=900, show_spinner=False)
def live_weather(city: str) -> dict | None:
    return weather.current(city)


@st.cache_data(max_entries=16, show_spinner=False)
def process_upload(data: bytes) -> tuple[bytes, bool, dict]:
    """Background removal (if available) then garment recognition on the clean cut-out. Cached per image."""
    clean, removed = service.prepare_upload(data)
    return clean, removed, service.tag_image(clean)


def board_path(items) -> str | None:
    rel = images.compose_board(list(items))
    p = images.resolve_image(rel) if rel else None
    return str(p) if p else None


def score_ring(score: float, size: int = 86) -> str:
    pct = max(0.0, min(100.0, score))
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 36 36" aria-label="score {score:.1f}">'
            '<path d="M18 2.5a15.5 15.5 0 1 1 0 31a15.5 15.5 0 1 1 0-31" fill="none" stroke="#ece6de" stroke-width="3.2"/>'
            f'<path d="M18 2.5a15.5 15.5 0 1 1 0 31a15.5 15.5 0 1 1 0-31" fill="none" stroke="#8a6a4f" stroke-width="3.2" '
            f'stroke-linecap="round" stroke-dasharray="{pct * 0.974:.1f},100"/>'
            f'<text x="18" y="20.6" text-anchor="middle" font-size="8.6" font-weight="600" fill="#1c1b19" '
            f'font-family="Inter, sans-serif">{score:.0f}</text></svg>')


def breakdown_html(outfit) -> str:
    rows = []
    for k, w in WEIGHTS.items():
        v = getattr(outfit.breakdown, k)
        rows.append(f'<div class="bd-row"><span>{html.escape(L("component", k))}</span>'
                    f'<span>{100 * w * v:.1f} / {100 * w:.0f}</span></div>'
                    f'<div class="bd-track"><div class="bd-fill" style="width:{v * 100:.0f}%"></div></div>')
    return "".join(rows)


def chips(values: list[str], dark_first: bool = False) -> str:
    return "".join(f'<span class="wil-chip{" dark" if dark_first and i == 0 else ""}">{html.escape(v)}</span>'
                   for i, v in enumerate(values))


def ranking(result) -> list[tuple]:
    return [(i + 1, " + ".join(name_of(it) for it in o.items), o.score) for i, o in enumerate(result.outfits)]


# ---------- callbacks (run before the rerun, so the next render shows updated scores) ----------

def on_feedback(item_ids: list[str], vote: str, ctx_tuple: tuple, snapshot: list[tuple]) -> None:
    ctx_ = validate_context(*ctx_tuple)
    res = service.record_feedback(item_ids, vote, ctx_, UID)
    names = [name_of(i) for i in db.get_items(item_ids, UID).values()]
    ss["last_feedback"] = {**res, "vote": vote, "names": names, "ctx": ctx_tuple, "ranking_before": snapshot,
                           "item_ids": item_ids}
    ss["flash"] = T("liked_toast" if vote == "like" else "disliked_toast",
                    a=f"{res['score_before']:.1f}", b=f"{res['score_after']:.1f}")


def on_another(outfit_key: tuple) -> None:
    ss.setdefault("skip", set()).add(tuple(outfit_key))
    ss["flash"] = T("swapped_toast")


def on_preset() -> None:
    preset = ss.get("preset_sel") or "men"
    service.switch_preset(UID, preset)
    for k in ("last_feedback", "chat", "skip"):
        ss.pop(k, None)
    ss["flash"] = T("preset_switched", p=T(f"preset_{preset}"))


def on_delete(item_id: str) -> None:
    service.remove_item(item_id, UID)
    ss["flash"] = T("removed")


def on_lang() -> None:
    st.query_params["lang"] = ss["lang_sel"] or "en"


def queue_prompt(text: str) -> None:
    ss["pending_prompt"] = text


def feedback_buttons(outfit, ctx_tuple, snapshot, key: str, primary: bool = True) -> None:
    b1, b2 = st.columns(2)
    b1.button(T("like"), key=f"like-{key}", width="stretch", type="primary" if primary else "secondary",
              icon=":material/thumb_up:", on_click=on_feedback, args=(list(outfit.key), "like", ctx_tuple, snapshot))
    b2.button(T("dislike"), key=f"dislike-{key}", width="stretch", icon=":material/thumb_down:",
              on_click=on_feedback, args=(list(outfit.key), "dislike", ctx_tuple, snapshot))


# ---------- app header: brand · language · settings ----------

h1, h_p, h2, h3 = st.columns([5, 1.8, 1.7, 0.55], vertical_alignment="center")
h1.markdown('<div class="wil-brand"><div class="wil-mark">WI</div><div><div class="wil-name">'
            f'{T("brand")}</div><div class="wil-tagline">{html.escape(T("headline"))}</div></div></div>',
            unsafe_allow_html=True)
presets = seed.available_presets()
if len(presets) > 1:
    h_p.segmented_control("Wardrobe", presets, format_func=lambda p: T(f"preset_{p}"),
                          default=service.current_preset(UID), key="preset_sel", on_change=on_preset,
                          label_visibility="collapsed")
h2.segmented_control("Language", list(i18n.LANGS), format_func=lambda k: i18n.LANG_SHORT[k], default=LANG,
                     key="lang_sel", on_change=on_lang, label_visibility="collapsed")
with h3.popover("", icon=":material/tune:", help=T("settings"), width="stretch"):
    st.markdown(f"**{T('ai_services')}**")
    ai = service.ai_status()
    for label_key, on, detail in (("svc_tagger", ai["vision"]["available"], ai["vision"].get("detail", "")),
                                  ("svc_bg", True, "BiRefNet-lite" if service.bgremove_online()
                                   and ai["bgremove"]["available"] else "fast cut-out · BiRefNet offline"),
                                  ("svc_llm", ai["llm"] or ai["local_llm"]["available"],
                                   stylist.MODEL if ai["llm"] else ai["local_llm"]["detail"])):
        state = T("svc_on") if on else T("svc_off")
        st.caption(f"{'●' if on else '○'} {T(label_key)}: {state}" + (f" · {detail}" if on and detail else ""))
    st.divider()
    st.markdown(f"**{T('demo_controls')}**")
    confirm = st.checkbox(T("reset_confirm"))
    if st.button(T("reset"), disabled=not confirm, width="stretch", icon=":material/restart_alt:"):
        seed.reset_demo(UID)
        for k in ("last_feedback", "chat"):
            ss.pop(k, None)
        ss["flash"] = T("reset_done")
        st.rerun()
    st.caption(T("privacy_note"))

# ---------- context bar: the inputs that drive every recommendation ----------

with st.container(border=True, key="ctxbar"):
    c1, c2, c3, c4 = st.columns([2.6, 3.4, 2.0, 1.7], gap="medium")
    occasion = c1.segmented_control(T("occasion"), OCCASIONS, default="work", key="occ",
                                    format_func=lambda v: L("occasion", v)) or "work"
    style = c2.segmented_control(T("style_pref"), STYLES, default="smart", key="sty",
                                 format_func=lambda v: L("style", v)) or "smart"
    with c4:
        use_live = st.toggle(T("live_weather"), value=LIVE_WEATHER_DEFAULT)
        live, city = None, None
        if use_live:
            city = st.selectbox(T("city"), list(weather.CITIES), index=0, label_visibility="collapsed")
            live = live_weather(city)
            st.caption(T("weather_now", city=city, temp=f"{live['temperature']:g}", cond=live["condition"])
                       if live else T("weather_off"))
    default_temp = int(round(live["temperature"])) if live else 16
    temperature = c3.slider(T("temperature"), -10, 35, max(-10, min(35, default_temp)))

ctx = validate_context(occasion, temperature, style)
CTX_T = (occasion, temperature, style)
result = service.get_recommendations(ctx, UID, LANG, skip=ss.get("skip"))
items = db.list_items(UID)
summary = service.feedback_summary(UID)
st.markdown('<div class="wil-statusline">' + chips([
    f"{len(items)} {T('m_items').lower()}", f"{result.candidates_considered} {T('m_scored').lower()}",
    f"♥ {summary['likes']} · ✕ {summary['dislikes']}"]) + "</div>", unsafe_allow_html=True)

# Pill navigation: only the selected page is computed (Streamlit tabs would run every page on each rerun).
NAV = {"today": ("wb_sunny", "tab_today"), "week": ("calendar_month", "tab_week"),
       "stylist": ("forum", "tab_stylist"), "wardrobe": ("checkroom", "tab_wardrobe"),
       "shop": ("shopping_bag", "tab_shop"), "learn": ("insights", "tab_insights")}
with st.container(key="mainnav"):
    PAGE = st.segmented_control("Navigation", list(NAV), default="today", key="nav",
                                format_func=lambda k: f":material/{NAV[k][0]}: {T(NAV[k][1])}",
                                label_visibility="collapsed") or "today"
WARDROBE_SIG = tuple(sorted(i.id for i in items))
PREFS_SIG = tuple(sorted(db.item_preferences(UID).items()))

# ---------- Today ----------

if PAGE == "today":
    with safe("today"):
        if result.message:
            st.info(result.message, icon=":material/info:")
        if not result.outfits:
            st.warning(T("no_outfits"))
        else:
            snapshot = ranking(result)
            hero = result.outfits[0]
            with st.container(border=True, key="card-hero"):
                c1, c2 = st.columns([1.25, 1], gap="large")
                with c1:
                    if p := board_path(hero.items):
                        st.image(p, width="stretch")
                with c2:
                    st.markdown(f'<div class="wil-eyebrow">{T("todays_look")}</div>', unsafe_allow_html=True)
                    st.markdown(chips([L("occasion", occasion), f"{temperature}°C", L("style", style)], True),
                                unsafe_allow_html=True)
                    st.markdown(f'<div class="wil-score">{score_ring(hero.score)}<div><div class="lbl">'
                                f'{T("score_label")}</div><div class="serif" style="font-size:1.25rem">'
                                f'{hero.score:.1f} / 100</div></div></div>', unsafe_allow_html=True)
                    st.markdown(chips([name_of(i) for i in hero.items]), unsafe_allow_html=True)
                    st.markdown(f'<div class="wil-eyebrow" style="margin-top:.6rem">{T("why")}</div>'
                                f'<div class="wil-why">{html.escape(hero.explanation)}</div>', unsafe_allow_html=True)
                    st.markdown(breakdown_html(hero), unsafe_allow_html=True)
                    st.write("")
                    feedback_buttons(hero, CTX_T, snapshot, "hero-" + "|".join(hero.key))
                    st.button(T("another"), key="another-" + "|".join(hero.key), width="stretch",
                              icon=":material/autorenew:", on_click=on_another, args=(hero.key,))
            if len(result.outfits) > 1:
                st.markdown(f"### {T('alternatives')}")
                for n, (col, o) in enumerate(zip(st.columns(2, gap="large"), result.outfits[1:]), start=2):
                    with col, st.container(border=True, key=f"card-alt{n}"):
                        if p := board_path(o.items):
                            st.image(p, width="stretch")
                        st.markdown(f'<div class="wil-score">{score_ring(o.score, 64)}<div><div class="lbl">'
                                    f'{T("look_n", n=n)}</div><div style="font-size:.9rem;color:#4a4640">'
                                    f'{html.escape(" · ".join(name_of(i) for i in o.items))}</div></div></div>',
                                    unsafe_allow_html=True)
                        st.markdown(f'<div class="wil-why">{html.escape(o.explanation)}</div>', unsafe_allow_html=True)
                        with st.expander(T("breakdown")):
                            st.markdown(breakdown_html(o), unsafe_allow_html=True)
                        feedback_buttons(o, CTX_T, snapshot, f"alt{n}-" + "|".join(o.key), primary=False)

# ---------- Plan my week ----------

@st.cache_data(ttl=900, show_spinner=False)
def live_forecast(city_: str) -> list[dict] | None:
    return weather.forecast(city_)


@st.cache_data(max_entries=64, show_spinner=False)
def cached_week(uid: str, ctxs: tuple, wardrobe_sig: tuple, prefs_sig: tuple, lang: str):
    return service.plan_week([validate_context(*c) for c in ctxs], uid, lang)


@st.cache_data(max_entries=64, show_spinner=False)
def cached_shop(uid: str, ctx_t: tuple, wardrobe_sig: tuple, prefs_sig: tuple):
    return service.shop_suggestions(validate_context(*ctx_t), uid, n=8)


@st.cache_data(max_entries=256, show_spinner=False)
def cached_similar(uid: str, item_id: str) -> list[tuple]:
    try:
        return [(s["product"], s["similarity"]) for s in service.similar_in_shop(item_id, uid, k=3)]
    except Exception:
        return []


def on_buy(product_id: str, name: str) -> None:
    service.add_product(product_id, UID)
    ss["flash"] = T("added", name=name)


if PAGE == "week":
    with safe("week"):
        import datetime as dt

        st.caption(T("week_intro"))
        fc = live_forecast(city) if use_live and city else None
        st.caption(T("week_source_live", city=city) if fc else T("week_source_manual", temp=temperature))
        start = dt.date.today()
        day_names = T("days").split(",")
        occs, temps, cards = [], [], []
        for i, col in enumerate(st.columns(7, gap="small")):
            d = start + dt.timedelta(days=i)
            f = fc[i] if fc and i < len(fc) else None
            t_day = f["temperature"] if f else float(temperature)
            default = "work" if d.weekday() < 5 else ("casual" if d.weekday() == 5 else "date")
            card = col.container(border=True, key=f"card-day{i}")
            with card:
                st.markdown(f'<div class="wil-eyebrow">{day_names[d.weekday()]} · {d.day:02d}.{d.month:02d}</div>'
                            f'<div class="serif" style="font-size:1.35rem">{t_day:.0f}°C</div>'
                            f'<div class="wil-item-meta">{html.escape(f["condition"]) if f else "&nbsp;"}</div>',
                            unsafe_allow_html=True)
                occs.append(st.selectbox(T("plan_for"), OCCASIONS, index=OCCASIONS.index(default), key=f"wk-occ-{i}",
                                         format_func=lambda v: L("occasion", v), label_visibility="collapsed"))
            temps.append(max(-30.0, min(45.0, t_day)))
            cards.append(card)
        plan = cached_week(UID, tuple((o, t, style) for o, t in zip(occs, temps)), WARDROBE_SIG, PREFS_SIG, LANG)
        for card, o in zip(cards, plan):  # fill each day card after the whole week is planned
            with card:
                if o is None:
                    st.caption("—")
                    continue
                if p := board_path(o.items):
                    st.image(p, width="stretch")
                st.markdown(f'<div class="wil-chip dark">{o.score:.0f}/100</div>' +
                            "".join(f'<div class="wil-item-meta">{html.escape(name_of(it))}</div>' for it in o.items),
                            unsafe_allow_html=True)
        used_pieces = {i for o in plan if o for i in o.key}
        st.caption(T("week_variety", n=len(used_pieces)))

# ---------- Shop the gap ----------

def dots(n: int, total: int = 4) -> str:
    return '<div class="wil-dots">' + "".join(
        f'<span class="wil-dot{" on" if i < n else ""}"></span>' for i in range(total)) + "</div>"


def mini_thumbs(pieces) -> str:
    """Tiny inline thumbnails (data URIs are avoided: images are served via st.image elsewhere)."""
    return "".join(f'<span class="wil-mini">{html.escape(name_of(i))}</span>' for i in pieces)


if PAGE == "shop":
    with safe("shop"):
        from dataclasses import replace as _replace

        from lookroom import shop as shop_mod
        from lookroom.models import DISPLAY_ORDER

        preset_gender = seed.PRESET_GENDER.get(service.current_preset(UID))
        n_catalog = sum(1 for p in shop_mod.load_products() if p.gender == preset_gender)
        st.markdown(f'<div class="wil-lead">{html.escape(T("shop_intro"))}</div>'
                    f'<div class="wil-note">{html.escape(T("shop_catalog_note", n=n_catalog))}</div>',
                    unsafe_allow_html=True)
        sugg = cached_shop(UID, CTX_T, WARDROBE_SIG, PREFS_SIG)
        if not sugg:
            st.info(T("shop_none"))
        else:
            top = sugg[0]
            p = top["product"]
            p_item = _replace(p.as_item(), image_path=images.shop_image(p.image_path, p.id))
            look = sorted([*top["pairs_with"], p_item], key=lambda it: DISPLAY_ORDER[it.category])
            gctx = top["context"]
            gain_txt = T("now_vs", ctx=f"{L('occasion', gctx.occasion)} · {gctx.temperature:.0f}°C",
                         a=f"{top['base']:.1f}", b=f"{top['score']:.1f}")
            taste_pct = 100 * top["taste"] if top.get("taste") else 0.0
            with st.container(border=True, key="card-gap"):
                g1, g2, g3 = st.columns([1.0, 1.25, 1.55], gap="large")
                if p.image_path.is_file():
                    g1.image(str(p.image_path), width="stretch")
                with g2:
                    st.markdown(f'<div class="wil-eyebrow">{T("biggest_gap")}</div>'
                                f'<div class="wil-title">{html.escape(p.name)}</div>'
                                + chips([L("category", p.category), L("color", p.color), L("style", p.style)])
                                + (f'<div class="wil-gain">+{top["gain"]:.1f}<span>pts</span></div>' if top["gain"] >= 0.5
                                   else f'<div class="wil-gain">{top["unlocks"]}<span>{T("looks_word")}</span></div>')
                                + (chips([T("unlocks", n=top["unlocks"])]) if top["unlocks"] else "")
                                + f'<div class="wil-note">{html.escape(gain_txt)}</div>'
                                f'<div class="wil-row">{dots(top["occasions_improved"], top["situations"])}'
                                f'<span class="wil-note">{T("shop_versatile", n=top["occasions_improved"], m=top["situations"])}'
                                '</span></div>',
                                unsafe_allow_html=True)
                    if taste_pct > 0:
                        taste_label = T("taste", p=f"{taste_pct:.0f}")
                        st.markdown(f'<div class="bd-row"><span>{taste_label}</span></div>'
                                    f'<div class="bd-track"><div class="bd-fill" style="width:{taste_pct:.0f}%">'
                                    '</div></div>', unsafe_allow_html=True)
                    st.write("")
                    st.button(T("shop_add"), key=f"buy-{p.id}", width="stretch", type="primary",
                              icon=":material/add_shopping_cart:", on_click=on_buy, args=(p.id, p.name))
                with g3:
                    st.markdown(f'<div class="wil-eyebrow">{T("with_it")}</div>', unsafe_allow_html=True)
                    if bp := board_path(look):
                        st.image(bp, width="stretch")
            if len(sugg) > 1:
                st.markdown(f"### {T('more_picks')}")
                rest = sugg[1:]
                for row_start in range(0, len(rest), 4):
                    for col, s in zip(st.columns(4, gap="medium"), rest[row_start:row_start + 4]):
                        q = s["product"]
                        with col, st.container(border=True, key=f"card-shop-{q.id}"):
                            if q.image_path.is_file():
                                st.image(str(q.image_path), width="stretch")
                            badges = ([f"+{s['gain']:.1f} pts"] if s["gain"] >= 0.5 else []) + (
                                [T("unlocks", n=s["unlocks"])] if s["unlocks"] else [])
                            if s.get("taste") is not None and s["taste"] > 0:
                                badges.append(T("taste", p=f"{100 * s['taste']:.0f}"))  # noqa: E501
                            st.markdown(f'<div class="wil-item-name">{html.escape(q.name)}</div>'
                                        + chips(badges, True) + dots(s["occasions_improved"], s["situations"])
                                        + f'<div class="wil-item-meta" style="margin-top:.35rem">{T("shop_pairs")}</div>'
                                        + f'<div class="wil-minis">{mini_thumbs(s["pairs_with"])}</div>',
                                        unsafe_allow_html=True)
                            st.button(T("shop_add"), key=f"buy-{q.id}", width="stretch",
                                      icon=":material/add_shopping_cart:", on_click=on_buy, args=(q.id, q.name))

# ---------- AI Stylist ----------

if PAGE == "stylist":
    with safe("stylist"):
        chat = ss.setdefault("chat", [])
        st.caption(T("stylist_intro"))
        q = st.columns(4)
        for col, k in zip(q[:3], ("q1", "q2", "q3")):
            col.button(T(k), on_click=queue_prompt, args=(T(k + "_text"),), width="stretch", icon=":material/bolt:")
        q[3].button(T("clear_chat"), on_click=lambda: ss.update(chat=[]), width="stretch", icon=":material/delete:")
        for idx, msg in enumerate(chat):
            with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/checkroom:"):
                st.write(msg["content"])
                if msg.get("item_ids"):
                    found = db.get_items(msg["item_ids"], UID)
                    if len(found) == len(msg["item_ids"]):
                        order = {c: n for n, c in enumerate(CATEGORIES)}
                        outfit_items = sorted(found.values(), key=lambda it: order[it.category])
                        if p := board_path(outfit_items):
                            st.image(p, width=520)
                    c_occ, c_temp, c_style = msg["ctx"]
                    src = {"claude": T("src_claude") + f" ({stylist.MODEL})",
                           "local": "Qwen3-1.7B · " + T("src_local")}.get(msg["source"], T("src_rules"))
                    st.caption(f"{src} · {T('engine_score')} {msg['score']:.1f}/100 · {L('occasion', c_occ)} · "
                               f"{c_temp:g}°C · {L('style', c_style)} · {msg['latency_ms']} ms"
                               + (f" · {msg['note']}" if msg.get("note") else ""))
                    snap = ranking(service.get_recommendations(validate_context(*msg["ctx"]), UID, LANG))
                    f1, f2, _ = st.columns([1, 1, 3])
                    f1.button(T("like"), key=f"chat-like-{idx}", type="primary", width="stretch",
                              icon=":material/thumb_up:", on_click=on_feedback,
                              args=(msg["item_ids"], "like", msg["ctx"], snap))
                    f2.button(T("dislike"), key=f"chat-dislike-{idx}", width="stretch", icon=":material/thumb_down:",
                              on_click=on_feedback, args=(msg["item_ids"], "dislike", msg["ctx"], snap))
        prompt = st.chat_input(T("chat_placeholder"))
        prompt = prompt or ss.pop("pending_prompt", None)
        if prompt:
            history = [{"role": m["role"], "content": m["content"]} for m in chat]
            chat.append({"role": "user", "content": prompt})
            with st.spinner(T("styling")):
                used = sum(1 for m in chat if m.get("source") == "claude")
                r = stylist.chat(prompt, history, ctx, allow_llm=used < SESSION_LLM_LIMIT, user_id=UID, lang=LANG)
            chat.append({"role": "assistant", "content": r.text, "source": r.source, "latency_ms": r.latency_ms,
                         "note": r.note, "item_ids": list(r.outfit.key) if r.outfit else [],
                         "score": r.outfit.score if r.outfit else 0.0,
                         "ctx": (r.context.occasion, r.context.temperature, r.context.style)})
            st.rerun()

# ---------- Wardrobe ----------

if PAGE == "wardrobe":
    with safe("wardrobe"):
        counts = {c: sum(1 for i in items if i.category == c) for c in CATEGORIES}
        options = ["all", *CATEGORIES]
        pick = st.pills("filter", options, default="all", label_visibility="collapsed",
                        format_func=lambda c: f"{T('all')} · {len(items)}" if c == "all" else f"{L('category', c)} · {counts[c]}")
        shown = [i for i in items if pick in (None, "all") or i.category == pick]
        if not items:
            st.info(T("empty_wardrobe"))
        grid = st.columns(5)
        for idx, it in enumerate(shown):
            with grid[idx % 5], st.container(border=True, key=f"card-item-{it.id}"):
                if p := images.resolve_image(it.image_path):
                    st.image(str(p), width="stretch")
                st.markdown(f'<div class="wil-item-name">{html.escape(name_of(it))}</div>'
                            f'<div class="wil-item-meta">{html.escape(L("color", it.color))} · '
                            f'{html.escape(L("style", it.style))}<br>{T("warmth")} {it.warmth}/5 · '
                            f'{T("formality")} {it.formality}/5</div>', unsafe_allow_html=True)
                with st.popover(T("similar"), icon=":material/image_search:", width="stretch"):
                    sims = cached_similar(UID, it.id)
                    st.caption(T("similar_note"))
                    for sc, (prod, sim) in zip(st.columns(max(1, len(sims))), sims):
                        if prod.image_path.is_file():
                            sc.image(str(prod.image_path), width="stretch")
                        sc.caption(f"{prod.name[:42]} · {sim:.2f}")
                if not it.is_seed:
                    st.button(T("remove"), key=f"del-{it.id}", on_click=on_delete, args=(it.id,), width="stretch",
                              icon=":material/close:")

        credits_md = seed.CATALOG_DIR / "CREDITS.md"
        if credits_md.is_file():
            with st.expander(T("photo_credits")):
                st.markdown(credits_md.read_text(encoding="utf-8").split("\n", 2)[-1])

        st.markdown(f"### {T('add_item')}")
        st.caption(T("add_hint"))
        with st.container(border=True, key="card-add"):
            ss.setdefault("upload_nonce", 0)
            up = st.file_uploader(T("photo"), type=["jpg", "jpeg", "png", "webp"], key=f"upload-{ss['upload_nonce']}")
            raw = up.getvalue() if up else None
            data, removed, tags = None, False, {}
            if raw:
                try:
                    with st.spinner(T("analysing")):
                        data, removed, tags = process_upload(raw)
                except ValueError as exc:
                    st.error(str(exc))
            if data:
                pc0, pc1, pc2 = st.columns([1, 1, 3])
                pc0.image(raw, width=140, caption=T("original"))
                pc1.image(data, width=140, caption=T("cutout") if removed else T("as_is"))
                if tags.get("source") == "fashionclip":
                    conf = tags["confidence"]
                    pc2.markdown(f"**{T('ai_suggests')}:** {html.escape(tags['suggested_name'] or '')}")
                    pc2.markdown(chips([L("category", tags["category"]), L("color", tags["color"]),
                                        L("style", tags["style"])]), unsafe_allow_html=True)
                    pc2.caption(f"{tags['model']} · {T('confidence')}: {conf.get('garment', 0):.0%} / "
                                f"{conf.get('color', 0):.0%} / {conf.get('style', 0):.0%} · {tags['latency_ms']} ms · "
                                f"{T('review_note')}")
                else:
                    pc2.caption(T("heuristic_note", c=L("color", tags.get("color", "white"))))

            def _idx(opts, value):
                return opts.index(value) if value in opts else 0

            with st.form("add_item", clear_on_submit=True, border=False):
                f1, f2, f3, f4 = st.columns(4)
                name = f1.text_input(T("name"), value=tags.get("suggested_name") or "", max_chars=60)
                category = f2.selectbox(T("category"), CATEGORIES, index=_idx(CATEGORIES, tags.get("category")),
                                        format_func=lambda v: L("category", v))
                color = f3.selectbox(T("color"), COLORS, index=_idx(COLORS, tags.get("color")),
                                     format_func=lambda v: L("color", v))
                item_style = f4.selectbox(T("style"), STYLES, index=_idx(STYLES, tags.get("style")),
                                          format_func=lambda v: L("style", v))
                s1, s2 = st.columns(2)
                warmth = s1.slider(T("warmth"), 1, 5, tags.get("warmth") or 3)
                formality = s2.slider(T("formality"), 1, 5, tags.get("formality") or 3)
                submitted = st.form_submit_button(T("save_item"), type="primary", icon=":material/add:")
            if submitted:
                try:
                    new = service.add_item(name, category, color, item_style, warmth, formality, data, user_id=UID)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    ss["flash"] = T("added", name=new.name)
                    ss["upload_nonce"] += 1
                    st.rerun()

# ---------- How it learns ----------

def dna_bars(values: dict[str, int], label_kind: str) -> str:
    """Diverging bars around a centre line: liked grows right (camel), avoided grows left (grey)."""
    if not values:
        return ""
    peak = max(abs(v) for v in values.values()) or 1
    rows = []
    for k, v in sorted(values.items(), key=lambda kv: -kv[1]):
        w = 50 * abs(v) / peak
        side = f"left:50%;width:{w:.0f}%" if v >= 0 else f"right:50%;width:{w:.0f}%"
        rows.append(f'<div class="bd-row"><span>{html.escape(L(label_kind, k))}</span><span>{v:+d}</span></div>'
                    f'<div class="bd-track div"><div class="bd-fill{"" if v >= 0 else " neg"}" style="{side}"></div></div>')
    return "".join(rows)


def piece_cards(rows: list[dict], by_id: dict, sign: str) -> None:
    cols = st.columns(5, gap="small")
    for col, r in zip(cols, rows[:5]):
        it = by_id.get(r["item_id"])
        if not it:
            continue
        with col, st.container(border=True, key=f"card-dna-{sign}-{it.id}"):
            if p := images.resolve_image(it.image_path):
                st.image(str(p), width="stretch")
            st.markdown(f'<div class="wil-item-name">{html.escape(name_of(it))}</div>'
                        f'<span class="wil-chip{" dark" if r["net"] > 0 else ""}">{"♥" if r["net"] > 0 else "✕"} {r["net"]:+d}</span>',
                        unsafe_allow_html=True)


if PAGE == "learn":
    with safe("insights"):
        from lookroom.models import COLOR_HEX

        cap = f"{100 * WEIGHTS['preference'] * 0.5:.1f}"
        steps = [("01", T("how_step1"), T("how_step1_d")),
                 ("02", T("how_step2"), T("how_step2_d", scale=f"{PREFERENCE_SCALE:g}", cap=cap)),
                 ("03", T("how_step3"), T("how_step3_d"))]
        for col, (n, title, desc) in zip(st.columns(3, gap="medium"), steps):
            col.markdown(f'<div class="wil-step"><div class="wil-step-n">{n}</div><div class="wil-step-t">'
                         f'{html.escape(title)}</div><div class="wil-step-d">{html.escape(desc)}</div></div>',
                         unsafe_allow_html=True)
        n_learned = sum(1 for r in summary["items"] if r["net"] != 0)
        st.markdown(
            '<div class="wil-tiles">'
            f'<div class="wil-tile"><div class="k">{T("likes")}</div><div class="v">{summary["likes"]}</div></div>'
            f'<div class="wil-tile"><div class="k">{T("dislikes")}</div><div class="v">{summary["dislikes"]}</div></div>'
            f'<div class="wil-tile"><div class="k">{T("learned_items")}</div><div class="v">{n_learned}</div></div>'
            f'<div class="wil-tile"><div class="k">±{cap} pts</div><div class="v">tanh</div></div>'
            '</div>', unsafe_allow_html=True)
        if not summary["items"]:
            st.info(T("learn_empty"), icon=":material/school:")
        else:
            by_id = {i.id: i for i in items}
            colors: dict[str, int] = {}
            for r in summary["items"]:
                if r["item_id"] in by_id:
                    c = by_id[r["item_id"]].color
                    colors[c] = colors.get(c, 0) + r["net"]
            left, right = st.columns([1, 1.15], gap="large")
            with left, st.container(border=True, key="card-taste"):
                st.markdown(f'<div class="wil-eyebrow">{T("taste_profile")}</div>'
                            f'<div class="wil-title">{T("style_dna")}</div>'
                            + dna_bars(summary["style_affinity"], "style")
                            + f'<div class="wil-title" style="margin-top:1.1rem">{T("color_dna")}</div>'
                            + '<div class="wil-swatches">' + "".join(
                                f'<div class="wil-swatch"><span style="background:{COLOR_HEX.get(c, "#999")};'
                                f'width:{28 + 6 * min(abs(v), 5)}px;height:{28 + 6 * min(abs(v), 5)}px;'
                                f'opacity:{1 if v > 0 else .35}"></span><em>{html.escape(L("color", c))} {v:+d}</em></div>'
                                for c, v in sorted(colors.items(), key=lambda kv: -kv[1])) + "</div>",
                            unsafe_allow_html=True)
            with right, st.container(border=True, key="card-ba"):
                st.markdown(f'<div class="wil-eyebrow">{T("before_after")}</div>', unsafe_allow_html=True)
                lf = ss.get("last_feedback")
                if not lf:
                    st.caption(T("before_after_empty"))
                else:
                    lctx = validate_context(*lf["ctx"])
                    cur = service.score_items(lf["item_ids"], lctx, UID, LANG)
                    now = cur.score if cur else lf["score_after"]
                    b1, b2 = st.columns([1.2, 1], gap="medium")
                    with b1:
                        rated = db.get_items(lf["item_ids"], UID)
                        if rated and (bp := board_path(sorted(rated.values(),
                                                              key=lambda it: DISPLAY_ORDER[it.category]))):
                            st.image(bp, width="stretch")
                    with b2:
                        st.markdown(f'<span class="wil-chip dark">{T("like") if lf["vote"] == "like" else T("dislike")}'
                                    f'</span><div class="wil-gain">{now:.1f}<span class="{"down" if now < lf["score_before"] else ""}">'
                                    f'{now - lf["score_before"]:+.1f}</span>'
                                    '</div>', unsafe_allow_html=True)
                        now_rank = ranking(service.get_recommendations(lctx, UID, LANG, skip=ss.get("skip")))
                        st.markdown(f'<div class="wil-note">{T("ranking_before")}</div>' + "".join(
                            f'<div class="wil-rank"><b>{r}</b> {html.escape(nm)} <i>{sc:.1f}</i></div>'
                            for r, nm, sc in lf["ranking_before"])
                            + f'<div class="wil-note" style="margin-top:.6rem">{T("ranking_now")}</div>' + "".join(
                            f'<div class="wil-rank now"><b>{r}</b> {html.escape(nm)} <i>{sc:.1f}</i></div>'
                            for r, nm, sc in now_rank), unsafe_allow_html=True)
            loved = [r for r in summary["items"] if r["net"] > 0]
            avoided = sorted([r for r in summary["items"] if r["net"] < 0], key=lambda r: r["net"])
            if loved:
                st.markdown(f"### {T('loved')}")
                piece_cards(loved, by_id, "loved")
            if avoided:
                st.markdown(f"### {T('avoided')}")
                piece_cards(avoided, by_id, "avoid")
            if summary["recent"]:
                st.markdown(f"### {T('history')}")
                names = {i.id: name_of(i) for i in items}
                st.markdown('<div class="wil-timeline">' + "".join(
                    f'<div class="wil-tl"><span class="wil-tl-dot {"up" if f["vote"] > 0 else "down"}">'
                    f'{"♥" if f["vote"] > 0 else "✕"}</span><div><div class="wil-item-name">'
                    f'{html.escape(" + ".join(names.get(i, i) for i in f["item_ids"]))}</div><div class="wil-item-meta">'
                    f'{html.escape(L("occasion", f["occasion"]))} · {f["temperature"]:g}°C · '
                    f'{html.escape(L("style", f["style"]))} · {f["score_at_feedback"] or 0:.1f}/100</div></div></div>'
                    for f in summary["recent"][:8]) + "</div>", unsafe_allow_html=True)

st.markdown(f'<div class="wil-footer">{html.escape(T("footer"))}</div>', unsafe_allow_html=True)

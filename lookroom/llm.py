"""Free, local LLM for the AI stylist: Qwen3-1.7B (Apache-2.0), GGUF Q4_K_M via llama.cpp on CPU. No API key.

Role is deliberately narrow: (1) fill context fields the multilingual keyword rules could not, via
schema-constrained JSON, and (2) phrase the stylist reply. The outfit itself is always chosen by the
deterministic engine, so a small model can never invent clothes or break scoring.
Measured on a 2-thread CPU: load ~2 s, JSON extraction 2.5-5 s, 2-3 sentence reply 5-7 s, ~1.3 GB RAM.
Env: LOOKROOM_LOCAL_LLM=0 disables; LOOKROOM_LLM_REPO / LOOKROOM_LLM_FILE override the model.
"""
from __future__ import annotations

import json
import os
import threading
import time

REPO = os.environ.get("LOOKROOM_LLM_REPO", "unsloth/Qwen3-1.7B-GGUF")
FILE = os.environ.get("LOOKROOM_LLM_FILE", "Qwen3-1.7B-Q4_K_M.gguf")
LABEL = "Qwen3-1.7B (local, Apache-2.0)"

_lock = threading.Lock()        # llama.cpp contexts are not thread-safe: one generation at a time
_llm = None
_error: str | None = None

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "occasion": {"type": ["string", "null"], "enum": ["casual", "work", "date", "formal", None]},
        "temperature_c": {"type": ["number", "null"]},
        "style": {"type": ["string", "null"], "enum": ["casual", "classic", "minimal", "smart", "streetwear", None]},
    },
    "required": ["occasion", "temperature_c", "style"],
}
EXTRACT_SYSTEM = ("Extract the outfit context from the user's message (any language). "
                  "occasion: casual|work|date|formal, or null if not stated. temperature_c: Celsius number if stated "
                  "or clearly implied (cold~5, freezing~-3, hot~30), else null. style: casual|classic|minimal|smart|"
                  "streetwear if implied, else null. Reply with JSON only.")
LANG_NAMES = {"en": "English", "ru": "Russian"}


def enabled() -> bool:
    if os.environ.get("LOOKROOM_LOCAL_LLM", "1") == "0":
        return False
    import importlib.util
    return importlib.util.find_spec("llama_cpp") is not None


def _load():
    global _llm, _error
    if _llm is not None or _error is not None:
        return _llm
    try:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        path = hf_hub_download(REPO, FILE)
        _llm = Llama(model_path=path, n_ctx=2048, n_threads=max(1, min(4, os.cpu_count() or 2)), verbose=False)
    except Exception as exc:  # no wheel, no network for the first download, low memory...
        _error = f"{type(exc).__name__}: {exc}"
    return _llm


def warmup() -> bool:
    """Download + load once (call from a background thread at startup)."""
    if not enabled():
        return False
    with _lock:
        return _load() is not None


def status() -> dict:
    if not enabled():
        return {"available": False, "loaded": False, "detail": "disabled or llama-cpp-python not installed"}
    return {"available": _error is None, "loaded": _llm is not None, "detail": _error or LABEL}


def _chat(messages: list[dict], **kw) -> str | None:
    with _lock:
        llm = _load()
        if llm is None:
            return None
        out = llm.create_chat_completion(messages=messages, **kw)
    text = out["choices"][0]["message"]["content"] or ""
    return text.replace("<think>", "").replace("</think>", "").strip()


def extract_context(message: str) -> dict | None:
    """{"occasion", "temperature_c", "style"} (values may be None) or None if the model is unavailable."""
    if not enabled():
        return None
    try:
        raw = _chat([{"role": "system", "content": EXTRACT_SYSTEM},
                     {"role": "user", "content": message[:600] + " /no_think"}],
                    response_format={"type": "json_object", "schema": EXTRACT_SCHEMA}, temperature=0.0, max_tokens=80)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def stylist_reply(lang: str, occasion: str, temperature: float, style: str, pieces: list[str], why: str) -> str | None:
    """2-3 sentence stylist note in English or Russian; None (-> deterministic template) otherwise."""
    if not enabled() or lang not in LANG_NAMES:
        return None
    t0 = time.perf_counter()
    try:
        text = _chat([
            {"role": "system", "content": f"You are a warm, concise fashion stylist. Write 2-3 sentences in "
                                          f"{LANG_NAMES[lang]}. Mention only the given pieces, by their given names. "
                                          "No lists, no markdown, no emojis."},
            {"role": "user", "content": f"Occasion: {occasion}, {temperature:.0f}°C, style {style}. "
                                        f"Outfit: {', '.join(pieces)}. Why it works: {why} /no_think"}],
            temperature=0.5, max_tokens=150)
    except Exception:
        return None
    if not text or len(text) < 20 or time.perf_counter() - t0 > 60:
        return None
    return text

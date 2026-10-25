"""Live temperature from MET Norway (api.met.no, no key, commercial use allowed, CC BY 4.0).

Terms honoured: identifying User-Agent, coordinates with <= 4 decimals, response cached until its
Expires header (min 10 min). Any failure returns None so the UI falls back to the manual slider.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from email.utils import parsedate_to_datetime

API = "https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
USER_AGENT = "WardrobeIntelligenceLab/0.2 (+https://huggingface.co/spaces/GevLLM/wardrobe-intelligence-lab)"
ATTRIBUTION = "Weather data: MET Norway (CC BY 4.0)"
TIMEOUT_S = 4.0
MIN_CACHE_S = 600

CITIES: dict[str, tuple[float, float]] = {
    "Yerevan": (40.1792, 44.4991), "London": (51.5072, -0.1276), "New York": (40.7128, -74.006),
    "Paris": (48.8566, 2.3522), "Berlin": (52.52, 13.405), "Dubai": (25.2048, 55.2708),
    "Tokyo": (35.6762, 139.6503), "Moscow": (55.7558, 37.6173), "Los Angeles": (34.0522, -118.2437),
    "Sydney": (-33.8688, 151.2093),
}
UTC_OFFSET = {"Yerevan": 4, "London": 0, "New York": -5, "Paris": 1, "Berlin": 1, "Dubai": 4, "Tokyo": 9,
              "Moscow": 3, "Los Angeles": -8, "Sydney": 11}  # approximate, for "daytime" averaging only

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _describe(symbol: str) -> str:
    s = symbol.split("_")[0]
    for key, text in (("thunder", "thunder"), ("sleet", "sleet"), ("snow", "snow"), ("rain", "rain"),
                      ("fog", "fog"), ("cloudy", "cloudy"), ("fair", "fair"), ("clearsky", "clear")):
        if key in s:
            return f"light {text}" if s.startswith("light") and text in ("rain", "snow", "sleet") else text
    return s or "n/a"


def _get(city: str, fetch) -> dict | None:
    """Raw MET compact payload, cached per city until its Expires header (min 10 min)."""
    now = time.time()
    key = f"raw:{city}"
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    lat, lon = CITIES[city]
    req = urllib.request.Request(API.format(lat=f"{lat:.4f}", lon=f"{lon:.4f}"), headers={"User-Agent": USER_AGENT})
    try:
        with fetch(req, timeout=TIMEOUT_S) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            expires = resp.headers.get("Expires") if hasattr(resp, "headers") else None
    except Exception:
        return None
    ttl = MIN_CACHE_S
    if expires:
        try:
            ttl = max(MIN_CACHE_S, parsedate_to_datetime(expires).timestamp() - now)
        except (TypeError, ValueError):
            pass
    with _lock:
        _cache[key] = (now + ttl, body)
    return body


def forecast(city: str, days: int = 7, fetch=urllib.request.urlopen) -> list[dict] | None:
    """Daily daytime (09-18 local) mean temperature + dominant condition for the next `days` days, or None."""
    from datetime import datetime, timedelta, timezone

    if city not in CITIES:
        return None
    body = _get(city, fetch)
    if not body:
        return None
    off = timedelta(hours=UTC_OFFSET.get(city, 0))
    per_day: dict[str, dict] = {}
    try:
        for step in body["properties"]["timeseries"]:
            t = datetime.fromisoformat(step["time"].replace("Z", "+00:00")).astimezone(timezone.utc) + off
            if not 9 <= t.hour <= 18:
                continue
            d = per_day.setdefault(t.date().isoformat(), {"temps": [], "symbols": []})
            d["temps"].append(float(step["data"]["instant"]["details"]["air_temperature"]))
            nxt = step["data"].get("next_1_hours") or step["data"].get("next_6_hours") or {}
            if sym := nxt.get("summary", {}).get("symbol_code"):
                d["symbols"].append(sym)
    except (KeyError, TypeError, ValueError):
        return None
    out = []
    for day, d in sorted(per_day.items())[:days]:
        if not d["temps"]:
            continue
        sym = max(set(d["symbols"]), key=d["symbols"].count) if d["symbols"] else ""
        out.append({"date": day, "temperature": round(sum(d["temps"]) / len(d["temps"]), 1),
                    "condition": _describe(sym), "source": ATTRIBUTION})
    return out or None


def current(city: str, fetch=urllib.request.urlopen) -> dict | None:
    """{"city", "temperature", "condition", "precipitation", "source"} or None. `fetch` is injectable for tests."""
    if city not in CITIES:
        return None
    body = _get(city, fetch)
    if not body:
        return None
    try:
        step = body["properties"]["timeseries"][0]["data"]
        temp = float(step["instant"]["details"]["air_temperature"])
        nxt = step.get("next_1_hours") or step.get("next_6_hours") or {}
        symbol = nxt.get("summary", {}).get("symbol_code", "")
        precip = float(nxt.get("details", {}).get("precipitation_amount", 0.0))
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return {"city": city, "temperature": round(temp, 1), "condition": _describe(symbol),
            "precipitation": precip, "source": ATTRIBUTION}

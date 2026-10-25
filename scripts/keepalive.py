"""Keep the Streamlit Community Cloud app awake and fail loudly if it is broken.

Usage: python scripts/keepalive.py [URL]   (run every 2 h by .github/workflows/keepalive.yml)
Opens the app in a real browser (a plain HTTP GET does not count as a visit), clicks the
"get this app back up" button if the app is asleep, and exits 1 unless "Today's look" renders.
"""
from __future__ import annotations

import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = (sys.argv[1] if len(sys.argv) > 1 else "https://wardrobe-intelligence-lab.streamlit.app").rstrip("/")
DEADLINE_S = 420
ERRORS = ("This app has encountered an error", "Traceback", "Oh no.")


def frame_texts(page) -> list[str]:
    out = []
    for f in page.frames:
        try:
            out.append(f.locator("body").inner_text(timeout=3000))
        except Exception:
            pass
    return out


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(f"{URL}/?lang=en&u=keepalive01", timeout=120000)  # a guest workspace: skips the sign-in page
        end = time.time() + DEADLINE_S
        while time.time() < end:
            wake = page.get_by_role("button", name=re.compile("get this app back up", re.I))
            if wake.count():
                print("app was asleep: waking it up")
                wake.first.click()
            texts = frame_texts(page)
            if any("today's look" in t.lower() for t in texts):  # CSS uppercases it
                page.wait_for_timeout(5000)  # stay connected a moment so the visit counts
                print(f"OK: app is up ({URL})")
                return 0
            bad = next((t for t in texts if any(e in t for e in ERRORS)), None)
            if bad:
                print("FAIL: app shows an error:\n" + bad[:1500])
                return 1
            page.wait_for_timeout(5000)
        print("FAIL: app did not render 'Today's look' in time:\n" + "\n---\n".join(t[:500] for t in frame_texts(page)))
        page.screenshot(path="keepalive_fail.png")
        return 1


if __name__ == "__main__":
    sys.exit(main())

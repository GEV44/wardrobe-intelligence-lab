"""Record real, unedited demo videos of the running app with Playwright, then convert to MP4 with ffmpeg.

Usage: python scripts/record_demo.py [walkthrough|api|all]
Output: docs/videos/*.mp4 (+ a README GIF). Uses a throwaway data dir, so your demo DB is untouched.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "videos"
SNEAKERS = ROOT / "demo_assets" / "sneakers_public_domain.jpg"
UI_PORT, API_PORT = 8599, 8011
VIEW = {"width": 1440, "height": 900}

CAPTION_JS = """(text) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div'); el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:999999;' +
      'background:rgba(20,20,20,.86);color:#fff;font:500 19px/1.4 system-ui,Segoe UI,sans-serif;' +
      'padding:12px 22px;border-radius:10px;max-width:80%;text-align:center;box-shadow:0 6px 24px rgba(0,0,0,.25)';
    document.body.appendChild(el);
  }
  el.textContent = text; el.style.display = text ? 'block' : 'none';
}"""


def wait_http(url: str, timeout: float = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(url)


def caption(page: Page, text: str, hold: float = 0) -> None:
    page.evaluate(CAPTION_JS, text)
    if hold:
        page.wait_for_timeout(int(hold * 1000))


def smooth_scroll(page: Page, dy: int, steps: int = 12) -> None:
    for _ in range(steps):
        page.mouse.wheel(0, dy / steps)
        page.wait_for_timeout(60)


def tab(page: Page, name: str) -> None:
    loc = page.locator('.st-key-mainnav button').filter(has_text=name).first
    loc.scroll_into_view_if_needed()
    loc.click(force=True)  # Streamlit's sticky header can overlap after scrolling
    page.wait_for_timeout(900)


def walkthrough(page: Page) -> None:
    page.goto(f"http://localhost:{UI_PORT}/?lang=en")
    page.get_by_text("Today's look").first.wait_for(timeout=90000)
    page.wait_for_timeout(2500)  # let the outfit boards load
    page.mouse.move(900, 450)
    caption(page, "Wardrobe Intelligence Lab: an independent LookRoom-inspired AI stylist MVP", 3.5)
    caption(page, "Real wardrobe photos → context-aware 'Today's look' with a transparent 0-100 score", 4)
    smooth_scroll(page, 380)
    caption(page, "Every score is explainable: 5 weighted components + a plain-English reason", 4.5)
    smooth_scroll(page, 520)
    caption(page, "Two diverse alternatives, never duplicates, always from the user's real items", 3.5)
    smooth_scroll(page, -900)

    caption(page, "Live weather (MET Norway) or manual: it gets cold, 16°C → 2°C ...", 1.5)
    page.get_by_role("slider").first.focus()
    for _ in range(14):
        page.keyboard.press("ArrowLeft")
        page.wait_for_timeout(110)
    page.wait_for_timeout(2500)
    caption(page, "... a coat is layered in automatically and weather-fit scores update", 4.5)

    caption(page, "Feedback loop: Dislike today's look ...", 1.5)
    page.locator('button:has-text("Dislike")').first.click(force=True)
    page.wait_for_timeout(2500)
    caption(page, "... a different look from the same wardrobe replaces it; the signal is bounded (max ±7.5 pts)", 4.5)

    tab(page, "How it learns")
    caption(page, "How it learns: before/after ranking, learned item signals, the exact formula", 3.5)
    smooth_scroll(page, 600, 16)
    page.wait_for_timeout(2500)
    smooth_scroll(page, -600, 10)

    tab(page, "Plan my week")
    caption(page, "Plan my week: 7 looks from the daily forecast, no piece repeated on consecutive days", 4.5)
    smooth_scroll(page, 450)
    page.wait_for_timeout(2000)
    smooth_scroll(page, -450)

    tab(page, "Shop the gap")
    caption(page, "Shop the gap: 984 real products tested against the wardrobe across 6 situations", 4.5)
    smooth_scroll(page, 500)
    caption(page, "Biggest gap first: its best look, the score gain and how many strong looks it unlocks", 3.5)
    smooth_scroll(page, -500)

    tab(page, "AI Stylist")
    caption(page, "AI Stylist: free local LLM (Qwen3-1.7B, CPU, no API key) + the engine picks real items", 2.5)
    page.get_by_text("Dinner date, 8°C, classic", exact=True).click(force=True)
    page.locator('[data-testid="stChatMessage"]').nth(1).wait_for(timeout=120000)
    page.wait_for_timeout(2500)
    caption(page, "Context extracted (date · 8°C · classic); the pick is validated and re-scored by the engine", 3.5)
    smooth_scroll(page, 450)
    caption(page, "Grounded + fallback: invalid or failed LLM output → deterministic rule engine", 4.5)
    smooth_scroll(page, -450)

    tab(page, "Wardrobe")
    caption(page, "The wardrobe: real garment photos, filterable by category", 3.5)
    smooth_scroll(page, 1300, 20)
    caption(page, "Upload a photo: cut-out + Marqo FashionSigLIP recognition (ONNX, CPU) pre-fill the form", 2)
    page.locator('input[type="file"]').first.set_input_files(str(SNEAKERS))
    page.get_by_text("AI suggests").first.wait_for(timeout=120000)
    page.wait_for_timeout(800)
    smooth_scroll(page, 500, 10)
    caption(page, "Shoes · white · warmth/formality suggested; the user reviews, then saves", 4.5)
    page.locator('button:has-text("Add to wardrobe")').click(force=True)
    page.wait_for_timeout(3000)
    caption(page, "Validated, re-encoded, stored under a UUID name in the user's private workspace", 3.5)
    smooth_scroll(page, -3000, 12)

    caption(page, "Switch to the women's wardrobe: dresses become a full outfit shape", 1.5)
    page.locator('.st-key-preset_sel button').filter(has_text="Women").first.click(force=True)
    page.wait_for_timeout(6000)
    tab(page, "Today")
    page.wait_for_timeout(3500)
    smooth_scroll(page, 400)
    page.wait_for_timeout(1500)
    smooth_scroll(page, -400)

    caption(page, "Fully localized: English · Armenian · Russian", 1.5)
    for lang, today in (("hy", "Այսօրվա կերպարը"), ("ru", "Образ дня")):
        url = page.url.split("?")[0] + "?" + "&".join(
            p for p in page.url.split("?")[1].split("&") if not p.startswith("lang=")) + f"&lang={lang}"
        page.goto(url)
        page.get_by_text(today).first.wait_for(timeout=90000)
        page.wait_for_timeout(3500)
        smooth_scroll(page, 400)
        page.wait_for_timeout(1500)
    caption(page, "100% free: FashionSigLIP + FashionCLIP ONNX · Qwen3 GGUF · MET Norway · Streamlit · FastAPI", 4.5)


def api_demo(page: Page) -> None:
    page.goto(f"http://127.0.0.1:{API_PORT}/docs")
    page.locator(".opblock").first.wait_for(timeout=30000)
    caption(page, "Same service layer behind a typed FastAPI (OpenAPI / Swagger)", 3.5)

    def run(path_text: str, body: str | None = None, method: str = "get") -> None:
        block = page.locator(f".opblock-{method}:has(.opblock-summary-path[data-path='{path_text}'])").first
        block.locator(".opblock-summary").click()
        page.wait_for_timeout(500)
        block.locator("button.try-out__btn").click()
        if body is not None:
            block.locator("textarea.body-param__text").fill(body)
        block.locator("button.execute").click()
        block.locator(".responses-table .response").first.wait_for(timeout=90000)
        page.wait_for_timeout(600)
        block.locator(".live-responses-table").scroll_into_view_if_needed()

    caption(page, "GET /health: vision + LLM availability for routing decisions")
    run("/health")
    page.wait_for_timeout(3000)
    caption(page, "GET /recommendations: validated query params → 3 outfits with score breakdowns")
    run("/recommendations")
    page.wait_for_timeout(3500)
    smooth_scroll(page, 700)
    page.wait_for_timeout(1500)
    caption(page, "POST /stylist/chat: local-LLM reply + engine-chosen outfit + score")
    run("/stylist/chat", '{"message": "Client meeting tomorrow, about 10 degrees, smart", "style": "smart"}', "post")
    page.wait_for_timeout(4000)
    smooth_scroll(page, 600)
    page.wait_for_timeout(2500)


def to_mp4(webm: Path, mp4: Path, gif: Path | None = None) -> None:
    ff = shutil.which("ffmpeg") or "ffmpeg"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", "-movflags", "+faststart", str(mp4)], check=True)
    if gif:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                        "fps=6,scale=760:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse",
                        "-t", "36", str(gif)], check=True)


def record(name: str, fn, url: str) -> Path:
    tmp = Path(tempfile.mkdtemp())
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport=VIEW, record_video_dir=str(tmp), record_video_size=VIEW)
        page = ctx.new_page()
        fn(page)
        video = page.video
        ctx.close()
        browser.close()
        webm = Path(video.path())
    mp4 = OUT / f"{name}.mp4"
    to_mp4(webm, mp4, OUT / f"{name}.gif" if name == "demo_walkthrough" else None)
    print(f"[ok] {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
    return mp4


def main(which: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = tempfile.mkdtemp(prefix="wil-demo-")
    env = {**os.environ, "LOOKROOM_DATA_DIR": data, "LOOKROOM_DISABLE_LLM": "1"}  # show the free, key-less path
    procs = [
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                          "--server.port", str(UI_PORT)], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--port", str(API_PORT)], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    ]
    try:
        wait_http(f"http://localhost:{UI_PORT}")
        wait_http(f"http://127.0.0.1:{API_PORT}/health")
        if which in ("walkthrough", "all"):
            record("demo_walkthrough", walkthrough, f"http://localhost:{UI_PORT}")
        if which in ("api", "all"):
            record("api_demo", api_demo, f"http://127.0.0.1:{API_PORT}/docs")
    finally:
        for pr in procs:
            pr.terminate()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")

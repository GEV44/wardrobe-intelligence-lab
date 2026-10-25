"""Record real, unedited demo videos of the running app with Playwright, then convert to MP4 with ffmpeg.

Usage: python scripts/record_demo.py [walkthrough|api|all]
Output: docs/videos/*.mp4 (+ a README GIF). Uses a throwaway data dir, so your demo DB is untouched.
"""
from __future__ import annotations

import os
import re
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


def idle(page: Page, ms: int = 900) -> None:
    """Wait until Streamlit has finished rerunning."""
    page.wait_for_timeout(400)
    for _ in range(200):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0 and page.locator('[data-stale="true"]').count() == 0:
            break
        page.wait_for_timeout(250)
    page.wait_for_timeout(ms)


def hero_names(page: Page) -> list[str]:
    return page.locator('.st-key-card-hero .wil-chip').all_inner_texts()[3:]


def walkthrough(page: Page) -> None:
    page.goto(f"http://localhost:{UI_PORT}/?lang=en")
    page.get_by_text("Create account").first.wait_for(timeout=90000)
    idle(page, 1500)
    page.mouse.move(900, 450)
    caption(page, "Wardrobe Intelligence Lab: a personal AI stylist (independent LookRoom-inspired concept)", 3.5)
    caption(page, "Create an account, or look around as a guest", 1.2)
    page.get_by_role("tab", name="Create account").click()
    idle(page, 400)
    page.locator('.st-key-su_user input').press_sequentially("anna.style", delay=70)
    page.locator('.st-key-su_pw input').press_sequentially("secret123", delay=60)
    page.wait_for_timeout(500)
    page.locator('.st-key-su_go button').click()
    page.get_by_text("Today's look").first.wait_for(timeout=90000)
    idle(page, 1500)
    caption(page, "Today's look: picked from YOUR clothes, today's weather and your plans", 4)
    smooth_scroll(page, 380)
    caption(page, "Every pick is explained: a 0-100 match built from 5 clear factors", 4)
    smooth_scroll(page, -380)

    caption(page, "Colder outside? Drop the temperature ...", 1.2)
    page.get_by_role("slider").first.focus()
    for _ in range(10):
        page.keyboard.press("ArrowLeft")
        page.wait_for_timeout(90)
    idle(page, 1200)
    caption(page, "... and a coat is layered in automatically", 3.5)

    for n in range(2):
        before = hero_names(page)
        caption(page, "Not your style? Dislike ..." if n == 0 else "Dislike again ...", 1.2)
        page.locator('button:has-text("Dislike")').first.click(force=True)
        idle(page, 1000)
        changed = len(set(hero_names(page)) - set(before))
        caption(page, f"... a clearly different look ({changed} pieces changed), until you love one", 3)
    caption(page, "Like it: it's saved to your looks and the stylist learns", 1.5)
    page.locator('button:has-text("Like")').first.click(force=True)
    idle(page, 1500)
    tab(page, "Saved looks")
    idle(page, 800)
    caption(page, "Saved looks: everything you liked, kept in your account", 3.5)

    tab(page, "Wardrobe")
    caption(page, "Your wardrobe: real photos, filter by category", 2.5)
    smooth_scroll(page, 900, 14)
    caption(page, "The menu and the category filter stay pinned while you scroll", 3)
    page.locator('.st-key-catbar button').filter(has_text="Shoes").first.click(force=True)
    idle(page, 1200)
    smooth_scroll(page, 1600, 16)
    caption(page, "Add a new piece: just upload a photo", 1.5)
    page.locator('input[type="file"]').first.set_input_files(str(SNEAKERS))
    page.get_by_text("Looks like").first.wait_for(timeout=120000)
    idle(page, 600)
    smooth_scroll(page, 450, 10)
    caption(page, "Background cleaned, item recognised, details pre-filled: you just check and save", 4)
    page.locator('button:has-text("Add to wardrobe")').click(force=True)
    idle(page, 2000)
    smooth_scroll(page, -4000, 12)

    tab(page, "AI Stylist")
    caption(page, "Chat with your stylist in your own words", 1.5)
    page.get_by_text("Dinner date, 8°C, classic", exact=True).click(force=True)
    page.locator('[data-testid="stChatMessage"]').nth(1).wait_for(timeout=120000)
    idle(page, 1200)
    caption(page, "It only suggests clothes you actually own", 3.5)
    smooth_scroll(page, 450)
    page.wait_for_timeout(1500)
    smooth_scroll(page, -450)

    tab(page, "Plan my week")
    caption(page, "Plan my week: 7 looks from the forecast, no piece repeated two days in a row", 4)
    smooth_scroll(page, 450)
    page.wait_for_timeout(1500)
    smooth_scroll(page, -450)

    tab(page, "Shop the gap")
    caption(page, "Shop the gap: the one piece that would unlock the most new looks for you", 4)
    smooth_scroll(page, 500)
    page.wait_for_timeout(1500)
    smooth_scroll(page, -500)

    tab(page, "How it learns")
    caption(page, "How it learns: your taste profile and how each like changed the ranking", 4)
    smooth_scroll(page, 650, 16)
    page.wait_for_timeout(1500)
    smooth_scroll(page, -650, 10)

    caption(page, "For engineers: one switch shows what runs under the hood", 1.5)
    page.locator('[data-testid="stPopover"] button').first.click(force=True)
    idle(page, 700)
    page.locator('.st-key-tech label').click()
    idle(page, 1200)
    page.locator('[data-testid="stPopover"] button').first.click(force=True)
    idle(page, 600)
    caption(page, "Free, no API keys: FashionSigLIP vision, local Qwen3 LLM, PostgreSQL, transparent scoring", 4.5)
    page.keyboard.press("Escape")
    idle(page, 400)

    caption(page, "English · Armenian · Russian", 1.2)
    for lang in ("hy", "ru", "en"):
        url = re.sub(r"lang=\w+", f"lang={lang}", page.url)
        page.goto(url)
        page.locator(".st-key-card-hero").wait_for(timeout=90000)
        idle(page, 1800 if lang != "en" else 600)
    caption(page, "Sign out and back in: your wardrobe, uploads and saved looks are still there", 1.5)
    page.locator('[data-testid="stPopover"] button').first.click(force=True)
    idle(page, 600)
    page.locator('.st-key-signout button').click(force=True)
    idle(page, 1000)
    page.locator('.st-key-si_user input').press_sequentially("anna.style", delay=60)
    page.locator('.st-key-si_pw input').press_sequentially("secret123", delay=50)
    page.locator('.st-key-si_go button').click()
    page.get_by_text("Today's look").first.wait_for(timeout=90000)
    idle(page, 800)
    tab(page, "Saved looks")
    caption(page, "Wardrobe Intelligence Lab: your wardrobe, fewer decisions", 4)


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
    env = {**os.environ, "LOOKROOM_DATA_DIR": data, "LOOKROOM_DISABLE_LLM": "1",  # show the free, key-less path
           "LOOKROOM_LOCAL_LLM": os.environ.get("LOOKROOM_LOCAL_LLM", "1"),
           "LOOKROOM_NO_SECRETS": "1"}  # never record against the live database
    env.pop("DATABASE_URL", None)
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

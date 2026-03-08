from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from random import Random
from time import sleep

from demo_scenario import DEFAULT_SCENARIO_PATH, DemoScenario, ResumeChoiceAction, SendNowAction, load_demo_scenario
from dotenv import load_dotenv
from playwright.sync_api import BrowserContext, Locator, Page, Playwright, sync_playwright
from playwright.sync_api import Error as PlaywrightError

# Mobile viewport — iPhone 12 Pro dimensions
MOBILE_WIDTH = 390
MOBILE_HEIGHT = 844
MOBILE_SCALE = 2.0

DETERMINISTIC_SEED = 20260308
CHAT_URL_TEMPLATE = "https://web.telegram.org/k/#@{username}"
DEFAULT_PAUSE_SECONDS = 2.5
DEFAULT_SERVER_WAIT = 2.5
MIN_TYPO_LENGTH = 24
TYPO_PROB = 0.45

MISSING_BOT_USERNAME = "Set TELEGRAM_DEMO_BOT_USERNAME or TELEGRAM_BOT_USERNAME in env"
MISSING_COMPOSER = "Telegram message composer not visible. Is the bot chat open?"
LOGIN_TIMEOUT = "Timed out waiting for Telegram Web login to complete"
SERVER_EXITED_EARLY = "Demo bot server exited before demo started"
INVALID_SERVER_WAIT = "--server-wait must be a positive number"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="telegram-web-demo",
        description=(
            "Run a scripted Telegram demo. Use --mode login once to save your session, "
            "then run without --mode (or --mode demo) to play the demo and optionally record it."
        ),
    )
    parser.add_argument("--mode", choices=("login", "demo"), default="demo")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path(".cache/telegram-web-profile"),
        help="Persistent Chromium profile directory (stores Telegram login session).",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("artifacts/demo-videos"),
        help="Directory where recorded videos are saved.",
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record video in demo mode (default: true).",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO_PATH,
        help="Path to demo story JSON.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=DEFAULT_PAUSE_SECONDS,
        help="Seconds to pause between story steps (default: 2.5).",
    )
    parser.add_argument("--headless", action="store_true", help="Run browser headless.")
    parser.add_argument(
        "--start-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-start demo bot server in demo mode (default: true).",
    )
    parser.add_argument(
        "--server-wait",
        type=float,
        default=DEFAULT_SERVER_WAIT,
        help="Seconds to wait after starting demo server before beginning the demo.",
    )
    parser.add_argument(
        "--server-command",
        default=f"{shlex.quote(sys.executable)} scripts/demo/run_demo_bot.py",
        help="Command used to start the demo bot server.",
    )
    return parser.parse_args()


def _bot_username() -> str:
    value = os.getenv("TELEGRAM_DEMO_BOT_USERNAME") or os.getenv("TELEGRAM_BOT_USERNAME")
    if not value or not value.strip():
        raise SystemExit(MISSING_BOT_USERNAME)
    return value.lstrip("@").strip()


def _launch_context(
    playwright: Playwright,
    *,
    profile_dir: Path,
    video_dir: Path | None,
    headless: bool,
) -> BrowserContext:
    profile_dir.mkdir(parents=True, exist_ok=True)
    video_kwargs: dict[str, object] = {}
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        video_kwargs = {
            "record_video_dir": str(video_dir),
            "record_video_size": {"width": MOBILE_WIDTH, "height": MOBILE_HEIGHT},
        }
    return playwright.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        headless=headless,
        viewport={"width": MOBILE_WIDTH, "height": MOBILE_HEIGHT},
        screen={"width": MOBILE_WIDTH, "height": MOBILE_HEIGHT},
        is_mobile=True,
        has_touch=True,
        device_scale_factor=MOBILE_SCALE,
        **video_kwargs,  # type: ignore[arg-type]
    )


def _show_tap_marker(page: Page, *, x: float, y: float) -> None:
    """Render a brief tap-ripple animation at the given viewport coordinates."""
    page.evaluate(
        """
        ({x, y}) => {
          const m = document.createElement("div");
          m.style.cssText = [
            `position:fixed`, `left:${x - 22}px`, `top:${y - 22}px`,
            "width:44px", "height:44px", "border-radius:999px",
            "border:3px solid rgba(37,99,235,0.92)",
            "background:rgba(37,99,235,0.18)", "box-sizing:border-box",
            "pointer-events:none", "z-index:2147483647",
            "transform:scale(0.4)", "opacity:0.95",
            "transition:transform 160ms ease-out,opacity 200ms ease-out",
          ].join(";");
          document.body.appendChild(m);
          requestAnimationFrame(() => {
            m.style.transform = "scale(1.1)";
            m.style.opacity = "0.05";
          });
          setTimeout(() => m.remove(), 280);
        }
        """,
        {"x": x, "y": y},
    )


def _click_with_tap_marker(page: Page, locator: Locator, *, timeout_ms: int) -> None:
    """Wait for *locator* to be visible, show a tap animation, then click it."""
    locator.wait_for(state="visible", timeout=timeout_ms)
    box = locator.bounding_box()
    if box is not None:
        _show_tap_marker(page, x=box["x"] + box["width"] / 2, y=box["y"] + box["height"] / 2)
        page.wait_for_timeout(180)
    locator.click()


def _find_composer(page: Page) -> Locator:
    locator = page.locator("div.input-message-input[contenteditable='true']").first
    try:
        locator.wait_for(state="visible", timeout=5_000)
    except PlaywrightError as exc:
        raise RuntimeError(MISSING_COMPOSER) from exc
    return locator


def _typo_char(char: str) -> str:
    if char.isalpha():
        replacement = "e" if char.lower() != "e" else "r"
        return replacement.upper() if char.isupper() else replacement
    if char.isdigit():
        return "1" if char != "1" else "2"
    return char


def _jitter_ms(base: int, rng: Random) -> int:
    return max(15, base + rng.randint(-7, 9))


def _type_with_jitter(page: Page, text: str, *, delay_ms: int, rng: Random) -> None:
    """Type *text* with per-character delay jitter and an occasional realistic typo."""
    typo_at: int | None = None
    if len(text) >= MIN_TYPO_LENGTH and " " in text and rng.random() < TYPO_PROB:
        candidates = [i for i, c in enumerate(text[:-1]) if c.isalnum()]
        if candidates:
            typo_at = candidates[rng.randint(0, len(candidates) - 1)]
    for i, char in enumerate(text):
        if typo_at is not None and i == typo_at:
            wrong = _typo_char(char)
            if wrong != char:
                page.keyboard.type(wrong, delay=_jitter_ms(delay_ms, rng))
                page.keyboard.press("Backspace")
        page.keyboard.type(char, delay=_jitter_ms(delay_ms, rng))


def _send_message(page: Page, text: str, *, delay_ms: int, rng: Random) -> None:
    composer = _find_composer(page)
    composer.click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.press("Backspace")
    _type_with_jitter(page, text, delay_ms=delay_ms, rng=rng)
    page.keyboard.press("Enter")


def _wait_for_new_text(page: Page, pattern: str, *, timeout_seconds: float) -> None:
    """Wait until *pattern* is visible anywhere on the page; return immediately if already present."""
    compiled = re.compile(pattern, re.IGNORECASE)
    locator = page.get_by_text(compiled)
    if locator.count() > 0 and locator.last.is_visible():
        return  # already on screen — proceed without waiting
    try:
        locator.first.wait_for(state="visible", timeout=int(timeout_seconds * 1000))
    except PlaywrightError:
        print(f"Warning: '{pattern}' not detected within {timeout_seconds:.0f}s — continuing.")


def _click_send_now(page: Page, *, timeout_seconds: float = 30.0) -> None:
    btn = page.get_by_role("button", name=re.compile(r"send now", re.IGNORECASE)).first
    try:
        _click_with_tap_marker(page, btn, timeout_ms=int(timeout_seconds * 1000))
    except PlaywrightError:
        print("Warning: 'Send now' button not found within timeout — continuing.")


def _click_resume_choice(page: Page, *, index: int, timeout_seconds: float = 30.0) -> None:
    btn = page.get_by_role("button", name=re.compile(rf"^{index}\.", re.IGNORECASE)).first
    try:
        _click_with_tap_marker(page, btn, timeout_ms=int(timeout_seconds * 1000))
    except PlaywrightError:
        print(f"Warning: resume choice '{index}.' not found within timeout — continuing.")


def _run_story(page: Page, scenario: DemoScenario, *, pause_seconds: float) -> None:
    """Execute story steps sequentially, driven entirely by demo_story.json."""
    rng = Random(DETERMINISTIC_SEED)
    for step in scenario.user_steps:
        if step.wait_for_text is not None:
            _wait_for_new_text(
                page,
                step.wait_for_text.pattern,
                timeout_seconds=step.wait_for_text.timeout_seconds,
            )
            if step.wait_for_text.after_ms > 0:
                page.wait_for_timeout(step.wait_for_text.after_ms)
        _send_message(page, step.text, delay_ms=scenario.runtime.typing_delay_ms, rng=rng)
        for action in step.actions:
            if isinstance(action, SendNowAction):
                _click_send_now(page)
            elif isinstance(action, ResumeChoiceAction):
                _click_resume_choice(page, index=action.index)
        sleep(pause_seconds)


def _try_start_button(page: Page) -> None:
    """Click the bot Start/Open button if it is present (new chat or cleared history)."""
    btn = page.get_by_role("button", name=re.compile(r"^(start|open|iniciar|abrir)$", re.IGNORECASE)).first
    if btn.count() and btn.is_visible():
        btn.click()
        page.wait_for_timeout(600)


def _open_chat(page: Page, *, username: str) -> None:
    composer = page.locator("div.input-message-input[contenteditable='true']").first
    # Primary: Ctrl+K global search — most reliable navigation inside the Telegram Web K SPA.
    page.keyboard.press("ControlOrMeta+K")
    page.wait_for_timeout(500)
    page.keyboard.type(f"@{username}")
    page.wait_for_timeout(1_000)
    result = page.get_by_text(re.compile(re.escape(username), re.IGNORECASE)).first
    if result.count() and result.is_visible():
        result.click()
        page.wait_for_timeout(600)
    _try_start_button(page)
    try:
        composer.wait_for(state="visible", timeout=10_000)
    except PlaywrightError:
        pass
    else:
        return  # chat ready
    # Fallback: direct URL navigation.
    page.goto(CHAT_URL_TEMPLATE.format(username=username), wait_until="domcontentloaded")
    _try_start_button(page)
    try:
        composer.wait_for(state="visible", timeout=20_000)
    except PlaywrightError as exc:
        raise SystemExit(MISSING_COMPOSER) from exc


def _ensure_logged_in(page: Page) -> None:
    page.goto("https://web.telegram.org/k/", wait_until="domcontentloaded")
    page.wait_for_timeout(1_200)
    qr = page.get_by_text(re.compile(r"qr|scan", re.IGNORECASE)).first
    if not (qr.count() and qr.is_visible()):
        return  # already logged in
    print("Telegram session not found. Scan the QR code shown in the browser.")
    print("Waiting for login to complete (up to 4 minutes)...")
    try:
        page.locator("div.chatlist, div.chatlist-chat").first.wait_for(state="visible", timeout=240_000)
    except PlaywrightError as exc:
        raise SystemExit(LOGIN_TIMEOUT) from exc


def _start_server(command: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(shlex.split(command))


def _shutdown_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=4)


def run() -> int:
    load_dotenv(override=False)
    args = parse_args()
    if args.server_wait <= 0:
        raise SystemExit(INVALID_SERVER_WAIT)
    scenario = load_demo_scenario(args.scenario)
    username = _bot_username()

    video_dir: Path | None = None
    if args.mode == "demo" and args.record:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        video_dir = args.video_dir / run_id
        print(f"Recording to: {video_dir}")

    server: subprocess.Popen[bytes] | None = None
    try:
        if args.mode == "demo" and args.start_server:
            server = _start_server(args.server_command)
            sleep(args.server_wait)
            if server.poll() is not None:
                raise SystemExit(SERVER_EXITED_EARLY)

        with sync_playwright() as pw:
            ctx = _launch_context(pw, profile_dir=args.profile_dir, video_dir=video_dir, headless=args.headless)
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            _ensure_logged_in(page)

            if args.mode == "login":
                ctx.close()
                print(f"Login successful. Profile saved to: {args.profile_dir}")
                return 0

            _open_chat(page, username=username)
            _run_story(page, scenario, pause_seconds=args.pause)
            sleep(max(3.0, scenario.runtime.final_pause_seconds))
            ctx.close()
    finally:
        _shutdown_server(server)

    if video_dir is not None:
        videos = sorted(video_dir.rglob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True)
        if videos:
            print(f"Recorded video: {videos[0]}")
        else:
            print(f"No video found in {video_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

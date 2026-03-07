# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "playwright>=1.55.0",
#   "python-dotenv>=1.2.1",
# ]
# ///

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from time import sleep

from dotenv import load_dotenv
from playwright.sync_api import (
    BrowserContext,
    Locator,
    Page,
    Playwright,
    sync_playwright,
)
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
)

MISSING_BOT_USERNAME_MESSAGE = "Missing bot username. Set TELEGRAM_DEMO_BOT_USERNAME (or TELEGRAM_BOT_USERNAME) in .env"
MISSING_COMPOSER_MESSAGE = "Could not find Telegram message composer. Is the bot chat open?"
CHAT_URL_TEMPLATE = "https://web.telegram.org/k/#@{bot_username}"

PRIMARY_TASK_PROMPT = (
    "If there are no objections on the review, merge the latest PR and prepare a new patch release. "
    "Please do the full flow (gh + git + tests + release notes), and keep me posted."
)
QUEUE_PROMPT = (
    "When you have a moment take a webcam photo and send it to me over MCP, "
    "but keep working on the release in parallel."
)
IMAGE_PROMPT = "Now take the webcam photo with ffmpeg and send it via MCP as an image."
ATTACHMENT_PROMPT = "Also generate a short diagnostics report file and send it via MCP as an attachment."


class DemoConfig(argparse.Namespace):
    mode: str
    profile_dir: Path
    video_dir: Path
    headless: bool
    reply_timeout: float
    manual_open_chat: bool
    open_first_chat: bool


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(
        prog="record-telegram-web-demo",
        description="Record a Telegram Web demo using a persistent logged-in profile.",
    )
    parser.add_argument("--mode", choices=("login", "record"), default="record")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path(".cache/telegram-web-profile"),
        help="Directory used by Chromium persistent profile (keeps Telegram login session).",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("artifacts/demo-videos"),
        help="Directory where recorded videos are stored.",
    )
    parser.add_argument(
        "--reply-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for expected UI cues before continuing.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser headless. For demo recording this should usually stay disabled.",
    )
    parser.add_argument(
        "--manual-open-chat",
        action="store_true",
        help="Wait for manual chat opening before running the scripted story.",
    )
    parser.add_argument(
        "--open-first-chat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open first chat in sidebar instead of resolving bot username (default: true).",
    )
    return parser.parse_args(namespace=DemoConfig())


def _resolve_bot_username() -> str:
    value = os.getenv("TELEGRAM_DEMO_BOT_USERNAME") or os.getenv("TELEGRAM_BOT_USERNAME")
    if value is None or not value.strip():
        raise SystemExit(MISSING_BOT_USERNAME_MESSAGE)
    return value.lstrip("@").strip()


def _open_first_chat(page: Page) -> None:
    page.goto("https://web.telegram.org/k/", wait_until="domcontentloaded")
    page.wait_for_timeout(1200)
    selectors = (
        "div.chatlist div.chatlist-chat",
        "div.chatlist [data-peer-id]",
        "ul.chatlist li",
        "a[href*='/k/#']",
    )
    for selector in selectors:
        candidate = page.locator(selector).first
        if candidate.count() and candidate.is_visible():
            candidate.click()
            page.wait_for_timeout(800)
            return


def _launch_context(playwright: Playwright, config: DemoConfig) -> BrowserContext:
    config.profile_dir.mkdir(parents=True, exist_ok=True)
    config.video_dir.mkdir(parents=True, exist_ok=True)

    if config.mode == "record":
        return playwright.chromium.launch_persistent_context(
            user_data_dir=str(config.profile_dir),
            headless=config.headless,
            viewport={"width": 390, "height": 844},
            screen={"width": 390, "height": 844},
            record_video_dir=str(config.video_dir),
            record_video_size={"width": 390, "height": 844},
        )

    return playwright.chromium.launch_persistent_context(
        user_data_dir=str(config.profile_dir),
        headless=config.headless,
        viewport={"width": 390, "height": 844},
        screen={"width": 390, "height": 844},
    )


def _get_page(context: BrowserContext) -> Page:
    if context.pages:
        return context.pages[0]
    return context.new_page()


def _open_chat(page: Page, *, bot_username: str) -> None:
    page.goto(CHAT_URL_TEMPLATE.format(bot_username=bot_username), wait_until="domcontentloaded")
    page.wait_for_timeout(1200)
    if _composer_is_visible(page) and _is_target_chat_selected(page, bot_username=bot_username):
        return

    _open_chat_via_search(page, bot_username=bot_username)
    _wait_for_composer(page, timeout_seconds=15.0)


def _is_target_chat_selected(page: Page, *, bot_username: str) -> bool:
    header = page.locator("header, div.chat-info-wrapper, div.chat-info-container").first
    if not header.count() or not header.is_visible():
        return False
    text = header.inner_text().strip().lower()
    needle = bot_username.lower()
    return needle in text or f"@{needle}" in text


def _find_composer(page: Page) -> Locator:
    selectors = (
        "div.input-message-container div.input-message-input[contenteditable='true']",
        "div.input-message-input[contenteditable='true']",
        "div.input-message-container div[contenteditable='true'][dir='auto']",
        "footer div.input-message-input[contenteditable='true']",
    )
    for selector in selectors:
        locator = page.locator(selector).first
        if locator.count() and locator.is_visible():
            return locator
    raise RuntimeError(MISSING_COMPOSER_MESSAGE)


def _debug_composer_candidates(page: Page) -> None:
    selectors = (
        "div.input-message-container div.input-message-input[contenteditable='true']",
        "div.input-message-input[contenteditable='true']",
        "div.input-message-container div[contenteditable='true'][dir='auto']",
        "footer div.input-message-input[contenteditable='true']",
    )
    print("Composer selector debug:")
    for selector in selectors:
        locator = page.locator(selector)
        count = locator.count()
        visible = locator.first.is_visible() if count else False
        print(f"  - {selector}: count={count} visible_first={visible}")


def _composer_is_visible(page: Page) -> bool:
    try:
        _find_composer(page)
    except RuntimeError:
        return False
    return True


def _wait_for_composer(page: Page, *, timeout_seconds: float) -> None:
    deadline_ms = int(timeout_seconds * 1000)
    slice_ms = 300
    elapsed = 0
    while elapsed < deadline_ms:
        if _composer_is_visible(page):
            return
        page.wait_for_timeout(slice_ms)
        elapsed += slice_ms
    raise RuntimeError(MISSING_COMPOSER_MESSAGE)


def _open_chat_via_search(page: Page, *, bot_username: str) -> None:
    page.goto("https://web.telegram.org/k/", wait_until="domcontentloaded")
    page.wait_for_timeout(700)

    # Telegram Web global search shortcut.
    page.keyboard.press("ControlOrMeta+K")
    page.wait_for_timeout(300)
    page.keyboard.type(f"@{bot_username}")
    page.wait_for_timeout(800)

    patterns = (
        re.compile(rf"@{re.escape(bot_username)}", re.IGNORECASE),
        re.compile(rf"\b{re.escape(bot_username)}\b", re.IGNORECASE),
    )
    for pattern in patterns:
        candidate = page.get_by_text(pattern).first
        if candidate.count() and candidate.is_visible():
            candidate.click()
            page.wait_for_timeout(800)
            if _is_target_chat_selected(page, bot_username=bot_username):
                return


def _ensure_logged_in(page: Page) -> None:
    page.goto("https://web.telegram.org/k/", wait_until="domcontentloaded")
    page.wait_for_timeout(1500)
    qr_hint = page.get_by_text(re.compile(r"qr|scan", re.IGNORECASE))
    if qr_hint.count() and qr_hint.first.is_visible():
        print("Telegram Web session is not logged in yet.")
        print("Scan QR (and complete 2FA if prompted), then press Enter here to continue...")
        input("Press Enter to continue... ")


def _try_click_start(page: Page) -> None:
    start_button = page.get_by_role("button", name=re.compile(r"^(start|iniciar)$", re.IGNORECASE))
    if start_button.count():
        start_button.first.click()
        page.wait_for_timeout(500)


def _send_message(page: Page, text: str) -> None:
    composer = _find_composer(page)
    composer.click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.press("Backspace")
    page.keyboard.type(text)
    page.keyboard.press("Enter")


def _wait_for_activity_labels(page: Page, *, timeout_seconds: float) -> None:
    labels = (
        re.compile(r"Thinking", re.IGNORECASE),
        re.compile(r"Running", re.IGNORECASE),
        re.compile(r"Tool call", re.IGNORECASE),
        re.compile(r"Searching", re.IGNORECASE),
    )
    deadline_ms = int(timeout_seconds * 1000)
    slice_ms = 2500
    elapsed = 0
    while elapsed < deadline_ms:
        for pattern in labels:
            locator = page.get_by_text(pattern)
            if locator.count() and locator.last.is_visible():
                return
        page.wait_for_timeout(slice_ms)
        elapsed += slice_ms
    print("Warning: activity labels were not detected in time.")


def _click_send_now(page: Page, *, timeout_seconds: float) -> None:
    busy_notice = page.get_by_text(re.compile(r"Agent is busy", re.IGNORECASE))
    send_now_button = page.get_by_role("button", name=re.compile(r"send now", re.IGNORECASE)).first
    try:
        busy_notice.first.wait_for(timeout=int(timeout_seconds * 1000))
    except PlaywrightTimeoutError:
        print("Warning: busy notice not detected; continuing without Send now click.")
        return

    try:
        send_now_button.wait_for(timeout=2500)
        send_now_button.click()
        page.wait_for_timeout(500)
    except PlaywrightTimeoutError:
        print("Warning: Send now button not detected in time.")


def _run_story(page: Page, *, timeout_seconds: float) -> None:
    _send_message(page, "/clear")
    sleep(0.8)

    _send_message(page, "/new")
    sleep(1.0)

    _send_message(page, PRIMARY_TASK_PROMPT)
    sleep(1.0)

    _send_message(page, QUEUE_PROMPT)
    _click_send_now(page, timeout_seconds=timeout_seconds)

    _wait_for_activity_labels(page, timeout_seconds=timeout_seconds)

    _send_message(page, IMAGE_PROMPT)
    sleep(1.0)

    _send_message(page, ATTACHMENT_PROMPT)
    sleep(1.0)

    _send_message(page, "/resume")
    sleep(2.0)


def _latest_video_path(video_dir: Path) -> Path | None:
    candidates = sorted(video_dir.rglob("*.webm"), key=lambda item: item.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if candidate.stat().st_size > 0:
            return candidate
    return None


def _prepare_video_dir(config: DemoConfig) -> None:
    if config.mode != "record":
        return
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    config.video_dir = config.video_dir / run_id
    config.video_dir.mkdir(parents=True, exist_ok=True)


def run() -> int:
    load_dotenv(override=False)
    config = parse_args()
    bot_username = _resolve_bot_username() if not config.open_first_chat else ""
    _prepare_video_dir(config)
    if config.mode == "record":
        print(f"Recording output directory: {config.video_dir}")

    with sync_playwright() as playwright:
        context = _launch_context(playwright, config)
        page = _get_page(context)

        if config.mode == "login":
            page.goto("https://web.telegram.org/k/", wait_until="domcontentloaded")
            print("Complete Telegram login in the opened browser (QR + optional password).")
            print("Press Enter here when you see your chats to save session and exit...")
            input("Press Enter to continue... ")
            context.close()
            print(f"Saved session profile in: {config.profile_dir}")
            return 0

        _ensure_logged_in(page)
        if config.manual_open_chat:
            page.goto("https://web.telegram.org/k/", wait_until="domcontentloaded")
            print("Open the bot chat manually in the browser, then press Enter here.")
            input("Press Enter to continue... ")
            _debug_composer_candidates(page)
            _wait_for_composer(page, timeout_seconds=30.0)
        elif config.open_first_chat:
            _open_first_chat(page)
            _wait_for_composer(page, timeout_seconds=15.0)
        else:
            _open_chat(page, bot_username=bot_username)
        _try_click_start(page)
        _run_story(page, timeout_seconds=config.reply_timeout)

        sleep(1.0)
        if page.video is not None:
            try:
                print(f"Current run video path: {page.video.path()}")
            except PlaywrightTimeoutError:
                print("Warning: could not resolve current run video path before closing context.")
        context.close()

    video = _latest_video_path(config.video_dir)
    if video is not None:
        print(f"Recorded video: {video}")
    else:
        print(f"No video file found in {config.video_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

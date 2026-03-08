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

from demo_scenario import (
    DEFAULT_SCENARIO_PATH,
    DemoScenario,
    ResumeChoiceAction,
    SendNowAction,
    load_demo_scenario,
)
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
LOGIN_TIMEOUT_MESSAGE = "Timed out waiting for Telegram Web login to complete."
INVALID_COORDS_FORMAT_MESSAGE = "Expected coordinates as 'x,y'."
INVALID_COORDS_NUMERIC_MESSAGE = "Coordinates must be numeric values like '95,212'."
OPEN_FIRST_CHAT_FAILED_MESSAGE = (
    "Could not open first chat automatically. Use --manual-open-chat or --no-open-first-chat."
)
CHAT_URL_TEMPLATE = "https://web.telegram.org/k/#@{bot_username}"
MOBILE_VIEWPORT_WIDTH = 390
MOBILE_VIEWPORT_HEIGHT = 780
DEFAULT_FIRST_CHAT_CLICK_COORDS = (168.0, 749.0)
EXPECTED_COORD_PARTS = 2
DEFAULT_SERVER_WAIT_SECONDS = 2.5
SERVER_START_FAILED_MESSAGE = "Demo bot server exited before recording started."
INVALID_WAIT_SECONDS_MESSAGE = "--server-wait-seconds must be a positive number"
DETERMINISTIC_TYPING_SEED = 20260308


class DemoConfig(argparse.Namespace):
    mode: str
    profile_dir: Path
    video_dir: Path
    headless: bool
    reply_timeout: float
    manual_open_chat: bool
    open_first_chat: bool
    first_chat_click_coords: tuple[float, float] | None
    capture_click_coords: bool
    device_scale_factor: float
    scenario: Path
    manual_story_actions: bool
    start_step: str | None
    start_server: bool
    server_wait_seconds: float
    server_command: str


def _parse_click_coords(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",", maxsplit=1)]
    if len(parts) != EXPECTED_COORD_PARTS:
        raise argparse.ArgumentTypeError(INVALID_COORDS_FORMAT_MESSAGE)
    try:
        x = float(parts[0])
        y = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(INVALID_COORDS_NUMERIC_MESSAGE) from exc
    return (x, y)


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(
        prog="telegram-web-demo",
        description="Run Telegram Web demo flow. In record mode it can auto-start the demo bot server.",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for manual chat opening before running the scripted story (default: true).",
    )
    parser.add_argument(
        "--open-first-chat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open first chat in sidebar instead of resolving bot username (default: false).",
    )
    parser.add_argument(
        "--first-chat-click-coords",
        type=_parse_click_coords,
        default=DEFAULT_FIRST_CHAT_CLICK_COORDS,
        metavar="X,Y",
        help="Click coordinates used to open chat from list (default: 168,749).",
    )
    parser.add_argument(
        "--capture-click-coords",
        action="store_true",
        help="Print click coordinates in browser page to help tune --first-chat-click-coords.",
    )
    parser.add_argument(
        "--device-scale-factor",
        type=float,
        default=1.0,
        help="Emulated device scale factor (default: 1.0).",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO_PATH,
        help="Path to demo story JSON used by recorder and fake ACP agent.",
    )
    parser.add_argument(
        "--manual-story-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pause for manual clicks on story actions like Send now and resume choice (default: false).",
    )
    parser.add_argument(
        "--start-step",
        default=None,
        help="Start story from a specific step id (e.g. 'resume') or 1-based index (e.g. '4').",
    )
    parser.add_argument(
        "--start-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-start demo bot server in record mode (default: true).",
    )
    parser.add_argument(
        "--server-wait-seconds",
        type=float,
        default=DEFAULT_SERVER_WAIT_SECONDS,
        help="Seconds to wait after starting server before recording begins.",
    )
    parser.add_argument(
        "--server-command",
        default=f"{shlex.quote(sys.executable)} scripts/demo/run_demo_bot.py",
        help="Command used to run demo bot server.",
    )
    return parser.parse_args(namespace=DemoConfig())


def _start_server(command: str) -> subprocess.Popen[bytes]:
    parts = shlex.split(command)
    return subprocess.Popen(parts)


def _shutdown_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    # Prefer Ctrl+C-style shutdown so asyncio tasks can drain before loop closes.
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


def _resolve_bot_username() -> str:
    value = os.getenv("TELEGRAM_DEMO_BOT_USERNAME") or os.getenv("TELEGRAM_BOT_USERNAME")
    if value is None or not value.strip():
        raise SystemExit(MISSING_BOT_USERNAME_MESSAGE)
    return value.lstrip("@").strip()


def _click_and_wait_for_composer(page: Page, *, x: float, y: float, timeout_ms: int = 1800) -> bool:
    page.mouse.click(x, y)
    elapsed = 0
    tick_ms = 150
    while elapsed < timeout_ms:
        if _composer_is_visible(page):
            return True
        page.wait_for_timeout(tick_ms)
        elapsed += tick_ms
    return False


def _open_first_chat(page: Page, *, click_coords: tuple[float, float] | None = None) -> bool:
    page.goto("https://web.telegram.org/k/", wait_until="domcontentloaded")
    page.wait_for_timeout(1200)
    if click_coords is not None:
        x, y = click_coords
        if _click_and_wait_for_composer(page, x=x, y=y):
            return True
        print("Warning: explicit first chat coordinates did not open a chat; trying automatic selection.")

    selectors = (
        "div.chatlist div.chatlist-chat",
        "div.chatlist [data-peer-id]",
        "ul.chatlist li",
        "a[href*='/k/#']",
    )
    for selector in selectors:
        locator = page.locator(selector)
        count = locator.count()
        if not count:
            continue
        for index in range(count):
            candidate = locator.nth(index)
            if not candidate.is_visible():
                continue
            box = candidate.bounding_box()
            if box is None:
                continue
            # Use an actual mouse click to mimic a manual chat selection.
            click_x = box["x"] + (box["width"] / 2)
            click_y = box["y"] + (box["height"] / 2)
            if _click_and_wait_for_composer(page, x=click_x, y=click_y):
                return True
    return False


def _install_click_coordinates_logger(page: Page) -> None:
    def _on_console_message(message: object) -> None:
        text_attr = getattr(message, "text", "")
        text = str(text_attr() if callable(text_attr) else text_attr)
        if text.startswith("DEMO_CLICK "):
            print(text)

    page.on("console", _on_console_message)
    page.evaluate(
        """
        () => {
          if (window.__demoClickLoggerInstalled) return;
          window.__demoClickLoggerInstalled = true;
          document.addEventListener(
            "click",
            (event) => {
              console.log(`DEMO_CLICK ${event.clientX},${event.clientY}`);
            },
            true
          );
        }
        """
    )


def _launch_context(playwright: Playwright, config: DemoConfig) -> BrowserContext:
    config.profile_dir.mkdir(parents=True, exist_ok=True)
    config.video_dir.mkdir(parents=True, exist_ok=True)

    if config.mode == "record":
        return playwright.chromium.launch_persistent_context(
            user_data_dir=str(config.profile_dir),
            headless=config.headless,
            viewport={"width": MOBILE_VIEWPORT_WIDTH, "height": MOBILE_VIEWPORT_HEIGHT},
            screen={"width": MOBILE_VIEWPORT_WIDTH, "height": MOBILE_VIEWPORT_HEIGHT},
            is_mobile=True,
            has_touch=True,
            device_scale_factor=config.device_scale_factor,
            record_video_dir=str(config.video_dir),
            record_video_size={
                "width": MOBILE_VIEWPORT_WIDTH,
                "height": MOBILE_VIEWPORT_HEIGHT,
            },
        )

    return playwright.chromium.launch_persistent_context(
        user_data_dir=str(config.profile_dir),
        headless=config.headless,
        viewport={"width": MOBILE_VIEWPORT_WIDTH, "height": MOBILE_VIEWPORT_HEIGHT},
        screen={"width": MOBILE_VIEWPORT_WIDTH, "height": MOBILE_VIEWPORT_HEIGHT},
        is_mobile=True,
        has_touch=True,
        device_scale_factor=config.device_scale_factor,
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


def _wait_for_manual_chat_open(page: Page, *, timeout_seconds: float) -> None:
    print("Manual step: click the bot chat in the browser. Recording will continue automatically.")
    _wait_for_composer(page, timeout_seconds=timeout_seconds)


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


def _chat_list_visible(page: Page) -> bool:
    selectors = (
        "div.chatlist",
        "div.chatlist-chat",
        "ul.chatlist",
    )
    for selector in selectors:
        locator = page.locator(selector).first
        if locator.count() and locator.is_visible():
            return True
    return False


def _wait_for_login_completion(page: Page, *, timeout_seconds: float) -> None:
    deadline_ms = int(timeout_seconds * 1000)
    tick_ms = 500
    elapsed = 0
    while elapsed < deadline_ms:
        qr_hint = page.get_by_text(re.compile(r"qr|scan", re.IGNORECASE))
        qr_visible = qr_hint.count() and qr_hint.first.is_visible()
        if not qr_visible and _chat_list_visible(page):
            return
        page.wait_for_timeout(tick_ms)
        elapsed += tick_ms
    raise RuntimeError(LOGIN_TIMEOUT_MESSAGE)


def _ensure_logged_in(page: Page) -> None:
    page.goto("https://web.telegram.org/k/", wait_until="domcontentloaded")
    page.wait_for_timeout(1500)
    qr_hint = page.get_by_text(re.compile(r"qr|scan", re.IGNORECASE))
    if qr_hint.count() and qr_hint.first.is_visible():
        print("Telegram Web session is not logged in yet.")
        print("Scan QR (and complete 2FA if prompted). Waiting for login completion...")
        _wait_for_login_completion(page, timeout_seconds=240.0)


def _try_click_start(page: Page) -> None:
    start_button = page.get_by_role("button", name=re.compile(r"^(start|iniciar)$", re.IGNORECASE))
    if start_button.count():
        start_button.first.click()
        page.wait_for_timeout(500)


def _typing_delay_ms(*, base: int, rng: Random) -> int:
    return max(15, base + rng.randint(-7, 9))


def _typo_replacement(char: str) -> str:
    if char.isalpha():
        replacement = "e" if char.lower() != "e" else "r"
        return replacement.upper() if char.isupper() else replacement
    if char.isdigit():
        return "1" if char != "1" else "2"
    return char


def _pick_typo_index(text: str, *, rng: Random) -> int | None:
    if len(text) < 24 or " " not in text:
        return None
    if rng.random() >= 0.45:
        return None
    candidates = [index for index, char in enumerate(text[:-1]) if char.isalnum()]
    if not candidates:
        return None
    return candidates[rng.randint(0, len(candidates) - 1)]


def _type_message_with_jitter(page: Page, text: str, *, typing_delay_ms: int, rng: Random) -> None:
    typo_index = _pick_typo_index(text, rng=rng)
    for index, char in enumerate(text):
        if typo_index is not None and index == typo_index:
            wrong = _typo_replacement(char)
            if wrong != char:
                page.keyboard.type(wrong, delay=_typing_delay_ms(base=typing_delay_ms, rng=rng))
                page.keyboard.press("Backspace")
        page.keyboard.type(char, delay=_typing_delay_ms(base=typing_delay_ms, rng=rng))


def _send_message(page: Page, text: str, *, typing_delay_ms: int, rng: Random) -> None:
    composer = _find_composer(page)
    composer.click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.press("Backspace")
    _type_message_with_jitter(page, text, typing_delay_ms=typing_delay_ms, rng=rng)
    page.keyboard.press("Enter")


def _human_pause(seconds: float) -> None:
    sleep(seconds)


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


def _wait_for_text(
    page: Page,
    pattern: re.Pattern[str],
    *,
    timeout_seconds: float,
    min_count: int = 1,
) -> bool:
    deadline_ms = int(timeout_seconds * 1000)
    elapsed = 0
    tick_ms = 300
    while elapsed < deadline_ms:
        locator = page.get_by_text(pattern)
        count = locator.count()
        if count >= min_count:
            for index in range(count - 1, -1, -1):
                candidate = locator.nth(index)
                if candidate.is_visible():
                    return True
        page.wait_for_timeout(tick_ms)
        elapsed += tick_ms
    return False


def _click_send_now(page: Page, *, timeout_seconds: float, show_tap_marker: bool) -> bool:
    busy_notice = page.get_by_text(re.compile(r"Agent is busy", re.IGNORECASE))
    try:
        busy_notice.first.wait_for(timeout=int(timeout_seconds * 1000))
    except PlaywrightTimeoutError:
        print("Warning: busy notice not detected; continuing without Send now click.")
        return False

    deadline_ms = int(timeout_seconds * 1000)
    elapsed = 0
    tick_ms = 250
    while elapsed < deadline_ms:
        candidates = (
            page.get_by_role("button", name=re.compile(r"send now", re.IGNORECASE)).first,
            page.get_by_text(re.compile(r"send now", re.IGNORECASE)).first,
            page.locator("button:has-text('Send now')").first,
            page.locator("div:has-text('Send now')").first,
        )
        for candidate in candidates:
            if not candidate.count() or not candidate.is_visible():
                continue
            box = candidate.bounding_box()
            if box is not None:
                click_x = box["x"] + (box["width"] / 2)
                click_y = box["y"] + (box["height"] / 2)
                if show_tap_marker:
                    _show_tap_marker(page, x=click_x, y=click_y)
                    page.wait_for_timeout(170)
                page.touchscreen.tap(click_x, click_y)
                print(f"Auto-clicked Send now at ({int(click_x)},{int(click_y)}).")
                page.wait_for_timeout(500)
                return True
            candidate.click()
            print("Auto-clicked Send now (no bounding box available).")
            page.wait_for_timeout(500)
            return True
        page.wait_for_timeout(tick_ms)
        elapsed += tick_ms

    print("Warning: Send now button not detected in time.")
    return False


def _show_tap_marker(page: Page, *, x: float, y: float) -> None:
    page.evaluate(
        """
        ({x, y}) => {
          const marker = document.createElement("div");
          marker.style.position = "fixed";
          marker.style.left = `${x - 22}px`;
          marker.style.top = `${y - 22}px`;
          marker.style.width = "44px";
          marker.style.height = "44px";
          marker.style.borderRadius = "999px";
          marker.style.border = "3px solid rgba(37, 99, 235, 0.92)";
          marker.style.background = "rgba(37, 99, 235, 0.20)";
          marker.style.boxSizing = "border-box";
          marker.style.pointerEvents = "none";
          marker.style.zIndex = "2147483647";
          marker.style.transform = "scale(0.42)";
          marker.style.opacity = "0.95";
          marker.style.transition = "transform 180ms ease-out, opacity 220ms ease-out";
          document.body.appendChild(marker);
          requestAnimationFrame(() => {
            marker.style.transform = "scale(1.08)";
            marker.style.opacity = "0.05";
          });
          setTimeout(() => marker.remove(), 260);
        }
        """,
        {"x": x, "y": y},
    )


def _click_resume_choice(page: Page, *, choice_index: int, timeout_seconds: float) -> bool:
    pattern = re.compile(rf"{choice_index}\\.", re.IGNORECASE)
    resumed_pattern = re.compile(r"(Resumed session|Session resumed)", re.IGNORECASE)
    deadline_ms = int(timeout_seconds * 1000)
    elapsed = 0
    tick_ms = 250
    while elapsed < deadline_ms:
        # If manual click already resumed a session, continue immediately.
        if _wait_for_text(page, resumed_pattern, timeout_seconds=0.25, min_count=1):
            return True

        by_role = page.get_by_role("button", name=pattern)
        if by_role.count():
            for index in range(by_role.count()):
                candidate = by_role.nth(index)
                if not candidate.is_visible():
                    continue
                box = candidate.bounding_box()
                if box is None:
                    continue
                page.touchscreen.tap(box["x"] + (box["width"] / 2), box["y"] + (box["height"] / 2))
                page.wait_for_timeout(220)
                return True

        by_text = page.get_by_text(pattern)
        if by_text.count():
            for index in range(by_text.count()):
                candidate = by_text.nth(index)
                if not candidate.is_visible():
                    continue
                box = candidate.bounding_box()
                if box is None:
                    continue
                page.touchscreen.tap(box["x"] + (box["width"] / 2), box["y"] + (box["height"] / 2))
                page.wait_for_timeout(220)
                return True

        page.wait_for_timeout(tick_ms)
        elapsed += tick_ms

    print(f"Warning: could not click resume choice index {choice_index}.")
    return False


def _wait_for_manual_send_now_click(page: Page, *, timeout_seconds: float) -> bool:
    sent_now_text = page.get_by_text(re.compile(r"sent now", re.IGNORECASE)).last
    send_now_button = page.get_by_role("button", name=re.compile(r"send now", re.IGNORECASE)).first
    tick_ms = 300
    elapsed = 0
    deadline_ms = int(timeout_seconds * 1000)
    while elapsed < deadline_ms:
        if sent_now_text.count() and sent_now_text.is_visible():
            return True
        if send_now_button.count() and not send_now_button.is_visible():
            return True
        page.wait_for_timeout(tick_ms)
        elapsed += tick_ms
    return False


def _wait_for_manual_resume_click(page: Page, *, timeout_seconds: float) -> bool:
    return _wait_for_text(
        page,
        re.compile(r"(Resumed session|Session resumed)", re.IGNORECASE),
        timeout_seconds=timeout_seconds,
        min_count=1,
    )


def _run_story_action(
    page: Page,
    *,
    action: SendNowAction | ResumeChoiceAction,
    timeout_seconds: float,
    manual_story_actions: bool,
) -> None:
    if isinstance(action, SendNowAction):
        if manual_story_actions:
            print("Manual step: click 'Send now' in Telegram Web. Waiting for the click...")
            clicked = _wait_for_manual_send_now_click(page, timeout_seconds=timeout_seconds)
            if not clicked:
                print("Warning: manual Send now click was not detected in time.")
            return
        clicked = _click_send_now(page, timeout_seconds=timeout_seconds, show_tap_marker=action.tap_marker)
        if clicked:
            return
        print("Manual fallback: click 'Send now' in Telegram Web. Waiting for the click...")
        _wait_for_manual_send_now_click(page, timeout_seconds=timeout_seconds)
        return

    if manual_story_actions:
        print(f"Manual step: click resume option '{action.index}.' in Telegram Web. Waiting for selection...")
        clicked = _wait_for_manual_resume_click(page, timeout_seconds=timeout_seconds)
        if not clicked:
            print("Warning: manual resume selection was not detected in time.")
        return
    # Keep auto-attempt short; if it misses, manual click should not be delayed.
    auto_timeout = min(timeout_seconds, 3.0)
    clicked = _click_resume_choice(page, choice_index=action.index, timeout_seconds=auto_timeout)
    if clicked:
        return
    print(f"Warning: resume option '{action.index}.' was not auto-clicked.")
    _wait_for_manual_resume_click(page, timeout_seconds=timeout_seconds)


def _run_story(page: Page, *, timeout_seconds: float, scenario: DemoScenario, manual_story_actions: bool) -> None:
    pause_seconds = (scenario.runtime.pause_min_seconds + scenario.runtime.pause_max_seconds) / 2
    pdf_followup_pause_seconds = 0.75
    typing_rng = Random(DETERMINISTIC_TYPING_SEED)
    for step in scenario.user_steps:
        if step.id == "recap":
            waited_resume = _wait_for_text(
                page,
                re.compile(r"(Resumed session|Session resumed)", re.IGNORECASE),
                timeout_seconds=step.wait_for_text.timeout_seconds if step.wait_for_text else 12.0,
                min_count=1,
            )
            if not waited_resume:
                print("Warning: recap skipped because resumed session confirmation was not detected.")
                continue
            _send_message(page, step.text, typing_delay_ms=scenario.runtime.typing_delay_ms, rng=typing_rng)
            _human_pause(pause_seconds)
            continue

        if step.wait_for_text is not None:
            wait_pattern = re.compile(step.wait_for_text.pattern, re.IGNORECASE)
            seen_before = page.get_by_text(wait_pattern).count()
            min_count = seen_before + 1
            # After /resume selection, the confirmation may already be visible.
            # In that case we should not wait for a second identical message.
            if "resumed session:" in step.wait_for_text.pattern.lower():
                min_count = 1
            if "diagnostics pdf is ready and attached" in step.wait_for_text.pattern.lower():
                min_count = 1
            waited = _wait_for_text(
                page,
                wait_pattern,
                timeout_seconds=step.wait_for_text.timeout_seconds,
                min_count=min_count,
            )
            if not waited:
                print(f"Warning: did not detect `{step.wait_for_text.pattern}` before step `{step.id}`.")
            if step.wait_for_text.after_ms > 0:
                page.wait_for_timeout(step.wait_for_text.after_ms)
        _send_message(page, step.text, typing_delay_ms=scenario.runtime.typing_delay_ms, rng=typing_rng)
        for action in step.actions:
            _run_story_action(
                page,
                action=action,
                timeout_seconds=timeout_seconds,
                manual_story_actions=manual_story_actions,
            )
        if step.id == "new":
            started = _wait_for_text(
                page,
                re.compile(r"Session started:", re.IGNORECASE),
                timeout_seconds=min(timeout_seconds, 12.0),
            )
            if not started:
                print("Warning: session start confirmation did not appear quickly after /new.")
            page.wait_for_timeout(250)
            continue
        if step.id == "primary":
            page.wait_for_timeout(250)
            continue
        if step.id == "resume":
            continue
        if step.id == "attachment":
            _human_pause(pdf_followup_pause_seconds)
            continue
        _human_pause(pause_seconds)

    image_seen = _wait_for_text(
        page,
        re.compile(scenario.runtime.image_reply_pattern, re.IGNORECASE),
        timeout_seconds=timeout_seconds * 1.4,
    )
    _ = image_seen


def _resolve_start_index(scenario: DemoScenario, start_step: str | None) -> int:
    if start_step is None or not start_step.strip():
        return 0
    value = start_step.strip()
    if value.isdigit():
        index = int(value) - 1
        if 0 <= index < len(scenario.user_steps):
            return index
        raise SystemExit(f"--start-step index out of range: {value}")
    for index, step in enumerate(scenario.user_steps):
        if step.id == value:
            return index
    raise SystemExit(f"--start-step id not found: {value}")


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
    if config.server_wait_seconds <= 0:
        raise SystemExit(INVALID_WAIT_SECONDS_MESSAGE)
    scenario = load_demo_scenario(config.scenario)
    start_index = _resolve_start_index(scenario, config.start_step)
    if start_index > 0:
        scenario = DemoScenario(
            runtime=scenario.runtime,
            assets=scenario.assets,
            user_steps=scenario.user_steps[start_index:],
            agent_routes=scenario.agent_routes,
        )
    bot_username = _resolve_bot_username() if (not config.manual_open_chat and not config.open_first_chat) else ""
    _prepare_video_dir(config)
    if config.mode == "record":
        print(f"Recording output directory: {config.video_dir}")
    should_start_server = config.mode == "record" and config.start_server
    server_process: subprocess.Popen[bytes] | None = None

    try:
        if should_start_server:
            server_process = _start_server(config.server_command)
            sleep(config.server_wait_seconds)
            if server_process.poll() is not None:
                raise SystemExit(SERVER_START_FAILED_MESSAGE)

        with sync_playwright() as playwright:
            context = _launch_context(playwright, config)
            page = _get_page(context)

            if config.mode == "login":
                page.goto("https://web.telegram.org/k/", wait_until="domcontentloaded")
                print("Complete Telegram login in the opened browser (QR + optional password).")
                print("Waiting until your chat list is visible...")
                _wait_for_login_completion(page, timeout_seconds=240.0)
                context.close()
                print(f"Saved session profile in: {config.profile_dir}")
                return 0

            _ensure_logged_in(page)
            if config.manual_open_chat:
                page.goto("https://web.telegram.org/k/", wait_until="domcontentloaded")
                if config.capture_click_coords:
                    _install_click_coordinates_logger(page)
                    print("Click target chat in browser to print coordinates as: DEMO_CLICK X,Y")
                _wait_for_manual_chat_open(page, timeout_seconds=60.0)
                _debug_composer_candidates(page)
                _wait_for_composer(page, timeout_seconds=30.0)
            elif config.open_first_chat:
                if not _open_first_chat(page, click_coords=config.first_chat_click_coords):
                    print(f"Warning: {OPEN_FIRST_CHAT_FAILED_MESSAGE}")
                    _wait_for_manual_chat_open(page, timeout_seconds=60.0)
                _wait_for_composer(page, timeout_seconds=15.0)
            else:
                _open_chat(page, bot_username=bot_username)
            _try_click_start(page)
            _run_story(
                page,
                timeout_seconds=config.reply_timeout,
                scenario=scenario,
                manual_story_actions=config.manual_story_actions,
            )

            sleep(max(3.0, scenario.runtime.final_pause_seconds))
            context.close()
    finally:
        _shutdown_server(server_process)

    video = _latest_video_path(config.video_dir)
    if video is not None:
        print(f"Recorded video: {video}")
    else:
        print(f"No video file found in {config.video_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

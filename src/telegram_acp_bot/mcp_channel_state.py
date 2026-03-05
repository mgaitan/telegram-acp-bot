"""Shared state helpers for Telegram MCP channel tools."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import gettempdir

STATE_FILE_ENV = "ACP_TELEGRAM_CHANNEL_STATE_FILE"
TOKEN_ENV = "ACP_TELEGRAM_BOT_TOKEN"


def default_state_file(*, pid: int) -> Path:
    """Return the default state file for a bot process."""

    return Path(gettempdir()) / f"telegram-acp-bot-mcp-state-{pid}.json"


def load_session_chat_map(path: Path) -> dict[str, int]:
    """Load `session_id -> chat_id` mapping from disk."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    sessions = raw.get("sessions")
    if not isinstance(sessions, dict):
        return {}

    return {key: value for key, value in sessions.items() if isinstance(key, str) and isinstance(value, int)}


def save_session_chat_map(path: Path, mapping: dict[str, int]) -> None:
    """Persist `session_id -> chat_id` mapping atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_raw_state(path)
    payload = {"sessions": mapping}
    if isinstance(existing.get("last_session_id"), str):
        payload["last_session_id"] = existing["last_session_id"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_last_session_id(path: Path) -> str | None:
    """Load the most recent session id used for prompt-side effects."""

    value = _load_raw_state(path).get("last_session_id")
    return value if isinstance(value, str) else None


def save_last_session_id(path: Path, session_id: str | None) -> None:
    """Persist the last active session id while keeping session mappings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_raw_state(path)
    payload: dict[str, object] = {"sessions": existing.get("sessions", {})}
    if session_id is not None:
        payload["last_session_id"] = session_id
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _load_raw_state(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}

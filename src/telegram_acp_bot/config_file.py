"""Config file loading and validation for telegram-acp-bot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_VALID_PERMISSION_MODES = frozenset({"ask", "approve", "deny"})
_VALID_PERMISSION_EVENT_OUTPUTS = frozenset({"stdout", "off"})
_VALID_LOG_FORMATS = frozenset({"text", "json"})
_VALID_ACTIVITY_MODES = frozenset({"compact", "normal", "verbose"})


class ConfigFileError(ValueError):
    """Raised when the config file is missing, unreadable, or contains invalid values."""


def load_config_file(path: Path) -> dict[str, Any]:
    """Load, parse, and validate a JSON config file.

    Returns the parsed config dict.
    Raises :exc:`ConfigFileError` if the file cannot be read, is not valid
    JSON, or fails schema validation.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigFileError(f"Config file not found: {path}") from None  # noqa: TRY003
    except OSError as exc:
        raise ConfigFileError(f"Cannot read config file {path}: {exc}") from exc  # noqa: TRY003

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigFileError(f"Invalid JSON in {path}: {exc}") from exc  # noqa: TRY003

    if not isinstance(data, dict):
        raise ConfigFileError(  # noqa: TRY003
            f"Config file {path} must contain a JSON object at the top level, got {type(data).__name__}"
        )

    _validate_config(data, path)
    return data


def _err(path: Path, msg: str) -> ConfigFileError:
    """Build a ConfigFileError with a path prefix."""
    return ConfigFileError(f"{path}: {msg}")


def _validate_config(data: dict[str, Any], path: Path) -> None:
    if "telegram" in data:
        _validate_telegram_section(data["telegram"], path)
    if "acp" in data:
        _validate_acp_section(data["acp"], path)


def _validate_telegram_section(tg: dict[str, Any], path: Path) -> None:
    if not isinstance(tg, dict):
        raise _err(path, "'telegram' must be a JSON object")
    if "bot_token" in tg and not isinstance(tg["bot_token"], str):
        raise _err(path, "'telegram.bot_token' must be a string")
    if "allowed_user_ids" in tg:
        ids = tg["allowed_user_ids"]
        if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
            raise _err(path, "'telegram.allowed_user_ids' must be a list of integers")
    if "allowed_usernames" in tg:
        names = tg["allowed_usernames"]
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise _err(path, "'telegram.allowed_usernames' must be a list of strings")


def _validate_acp_section(acp: dict[str, Any], path: Path) -> None:
    if not isinstance(acp, dict):
        raise _err(path, "'acp' must be a JSON object")
    for key in ("agent_command", "restart_command", "log_level", "scheduled_tasks_db", "workspace"):
        if key in acp and not isinstance(acp[key], str):
            raise _err(path, f"'acp.{key}' must be a string")
    if "permission_mode" in acp and acp["permission_mode"] not in _VALID_PERMISSION_MODES:
        raise _err(
            path,
            f"'acp.permission_mode' must be one of {sorted(_VALID_PERMISSION_MODES)}, got {acp['permission_mode']!r}",
        )
    if "permission_event_output" in acp and acp["permission_event_output"] not in _VALID_PERMISSION_EVENT_OUTPUTS:
        raise _err(
            path,
            f"'acp.permission_event_output' must be one of"
            f" {sorted(_VALID_PERMISSION_EVENT_OUTPUTS)}, got {acp['permission_event_output']!r}",
        )
    if "log_format" in acp and acp["log_format"] not in _VALID_LOG_FORMATS:
        raise _err(
            path,
            f"'acp.log_format' must be one of {sorted(_VALID_LOG_FORMATS)}, got {acp['log_format']!r}",
        )
    if "activity_mode" in acp and acp["activity_mode"] not in _VALID_ACTIVITY_MODES:
        raise _err(
            path,
            f"'acp.activity_mode' must be one of {sorted(_VALID_ACTIVITY_MODES)}, got {acp['activity_mode']!r}",
        )
    if "stdio_limit" in acp and not isinstance(acp["stdio_limit"], int):
        raise _err(path, "'acp.stdio_limit' must be an integer")
    if "connect_timeout" in acp and not isinstance(acp["connect_timeout"], (int, float)):
        raise _err(path, "'acp.connect_timeout' must be a number")

"""Config file loading and validation for telegram-acp-bot."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_VALID_PERMISSION_MODES = frozenset({"ask", "approve", "deny"})
_VALID_PERMISSION_EVENT_OUTPUTS = frozenset({"stdout", "off"})
_VALID_LOG_FORMATS = frozenset({"text", "json"})
_VALID_ACTIVITY_MODES = frozenset({"compact", "normal", "verbose"})
_MCP_SERVER_NAME_RE = re.compile(r"^[a-z0-9-_]+$")


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
    if "mcp_servers" in data:
        _validate_mcp_servers_section(data["mcp_servers"], path)


def _validate_telegram_section(tg: dict[str, Any], path: Path) -> None:
    if not isinstance(tg, dict):
        raise _err(path, "'telegram' must be a JSON object")
    if "bot_token" in tg and not isinstance(tg["bot_token"], str):
        raise _err(path, "'telegram.bot_token' must be a string")
    if "allowed_user_ids" in tg:
        ids = tg["allowed_user_ids"]
        if not isinstance(ids, list) or not all(type(x) is int for x in ids):
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
    if "stdio_limit" in acp and (isinstance(acp["stdio_limit"], bool) or not isinstance(acp["stdio_limit"], int)):
        raise _err(path, "'acp.stdio_limit' must be an integer")
    if "connect_timeout" in acp and (
        isinstance(acp["connect_timeout"], bool) or not isinstance(acp["connect_timeout"], (int, float))
    ):
        raise _err(path, "'acp.connect_timeout' must be a number")


def _validate_mcp_servers_section(servers: dict[str, Any], path: Path) -> None:
    if not isinstance(servers, dict):
        raise _err(path, "'mcp_servers' must be a JSON object")
    for name, server in servers.items():
        _validate_mcp_server_entry(name, server, path)


def _validate_mcp_server_entry(name: str, server: dict[str, Any], path: Path) -> None:
    if not _MCP_SERVER_NAME_RE.fullmatch(name):
        raise _err(path, f"MCP server name {name!r} must match ^[a-z0-9-_]+$ (lowercase, no spaces)")
    if not isinstance(server, dict):
        raise _err(path, f"MCP server {name!r} must be a JSON object")
    has_command = "command" in server
    has_url = "url" in server
    if not has_command and not has_url:
        raise _err(path, f"MCP server {name!r} must define 'command' (stdio) or 'url' (remote)")
    if has_command and has_url:
        raise _err(path, f"MCP server {name!r} cannot define both 'command' and 'url'")
    if has_command:
        _validate_stdio_server(name, server, path)
    if has_url:
        _validate_http_server(name, server, path)


def _validate_stdio_server(name: str, server: dict[str, Any], path: Path) -> None:
    if not isinstance(server["command"], str):
        raise _err(path, f"MCP server {name!r} 'command' must be a string")
    if "args" in server:
        args = server["args"]
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise _err(path, f"MCP server {name!r} 'args' must be a list of strings")
    if "env" in server:
        env = server["env"]
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise _err(path, f"MCP server {name!r} 'env' must be an object of string → string")


def _validate_http_server(name: str, server: dict[str, Any], path: Path) -> None:
    if not isinstance(server["url"], str):
        raise _err(path, f"MCP server {name!r} 'url' must be a string")
    if "headers" in server:
        headers = server["headers"]
        if not isinstance(headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            raise _err(path, f"MCP server {name!r} 'headers' must be an object of string → string")

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

_CONTEXT_FIELDS = ("chat_id", "session_id", "prompt_cycle_id")
_MISSING = "-"
_log_context: ContextVar[dict[str, str] | None] = ContextVar("telegram_acp_log_context", default=None)


def configure_logging(*, level: int, log_format: str = "text", replace_handlers: bool = True) -> None:
    """Configure root logging with request/session context enrichment."""

    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(_JsonLogFormatter())
    else:
        handler.setFormatter(_TextLogFormatter())
    handler.addFilter(_ContextLogFilter())

    root_logger = logging.getLogger()
    if replace_handlers:
        root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


@contextmanager
def bind_log_context(**fields: object) -> Iterator[None]:
    """Temporarily bind contextual fields to log records in the current task."""

    current = _log_context.get() or {}
    merged = dict(current)
    for key, raw_value in fields.items():
        if key not in _CONTEXT_FIELDS or raw_value is None:
            continue
        merged[key] = str(raw_value)
    token: Token[dict[str, str] | None] = _log_context.set(merged)
    try:
        yield
    finally:
        _log_context.reset(token)


def get_log_context() -> dict[str, str]:
    """Return the current logging context values for this task."""

    return dict(_log_context.get() or {})


class _ContextLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = _log_context.get() or {}
        for key in _CONTEXT_FIELDS:
            setattr(record, key, context.get(key, _MISSING))
        return True


class _TextLogFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt=(
                "%(asctime)s %(levelname)s %(name)s "
                "chat_id=%(chat_id)s session_id=%(session_id)s prompt_cycle_id=%(prompt_cycle_id)s %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "chat_id": getattr(record, "chat_id", _MISSING),
            "session_id": getattr(record, "session_id", _MISSING),
            "prompt_cycle_id": getattr(record, "prompt_cycle_id", _MISSING),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)

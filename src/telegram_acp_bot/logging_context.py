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
_BASE_LOG_RECORD_FACTORY = logging.getLogRecordFactory()


def configure_logging(
    *,
    level: int,
    log_format: str = "text",
    replace_handlers: bool = True,
    close_replaced_handlers: bool = False,
) -> None:
    """Configure root logging with request/session context enrichment."""

    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(_JsonLogFormatter())
    else:
        handler.setFormatter(_TextLogFormatter())
    _install_log_record_factory()
    root_logger = logging.getLogger()
    if replace_handlers:
        for existing_handler in list(root_logger.handlers):
            root_logger.removeHandler(existing_handler)
            if close_replaced_handlers:
                existing_handler.close()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    _configure_third_party_logging()


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


def _install_log_record_factory() -> None:
    if logging.getLogRecordFactory() is _contextual_log_record_factory:
        return
    logging.setLogRecordFactory(_contextual_log_record_factory)


def _contextual_log_record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
    record = _BASE_LOG_RECORD_FACTORY(*args, **kwargs)
    context = _log_context.get() or {}
    for key in _CONTEXT_FIELDS:
        setattr(record, key, context.get(key, _MISSING))
    return record


def _configure_third_party_logging() -> None:
    """Reduce noisy transport/framework logs in default bot output."""

    for logger_name in ("httpx", "httpcore", "telegram.ext", "apscheduler"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


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

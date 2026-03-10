from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from rich.logging import RichHandler
from rich.markup import escape

_CONTEXT_FIELDS = ("chat_id", "session_id", "prompt_cycle_id")
_MISSING = "-"
LOG_TEXT_PREVIEW_MAX_CHARS = 160
_log_context: ContextVar[dict[str, str] | None] = ContextVar("telegram_acp_log_context", default=None)
_LOG_RECORD_FACTORY_STATE: dict[str, object] = {"delegate": logging.getLogRecordFactory()}


class _RecordFactory(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> logging.LogRecord: ...


def configure_logging(
    *,
    level: int,
    log_format: str = "text",
    replace_handlers: bool = True,
    close_replaced_handlers: bool = False,
) -> None:
    """Configure root logging with request/session context enrichment."""

    handler: logging.Handler
    if log_format == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonLogFormatter())
    else:
        handler = RichHandler(
            markup=True,
            rich_tracebacks=True,
            show_level=True,
            show_path=False,
            show_time=True,
            omit_repeated_times=False,
        )
        handler.setFormatter(_RichTextFormatter())
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
    current_factory = logging.getLogRecordFactory()
    if current_factory is _contextual_log_record_factory:
        return
    _LOG_RECORD_FACTORY_STATE["delegate"] = current_factory
    logging.setLogRecordFactory(_contextual_log_record_factory)


def _contextual_log_record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
    delegate = cast(_RecordFactory, _LOG_RECORD_FACTORY_STATE["delegate"])
    record = delegate(*args, **kwargs)
    context = _log_context.get() or {}
    for key in _CONTEXT_FIELDS:
        setattr(record, key, context.get(key, _MISSING))
    return record


def _configure_third_party_logging() -> None:
    """Reduce noisy transport/framework logs in default bot output."""

    for logger_name in ("httpx", "httpcore", "telegram.ext", "apscheduler"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def log_text_preview(text: str) -> str:
    """Return a compact single-line preview for prompt/reply logs."""

    compact = " ".join(text.split())
    if not compact:
        return "<empty>"
    if len(compact) <= LOG_TEXT_PREVIEW_MAX_CHARS:
        return compact
    return f"{compact[:LOG_TEXT_PREVIEW_MAX_CHARS]}..."


class _RichTextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        logger_label = _logger_label(record.name)
        parts = [f"[bold bright_cyan]{logger_label}[/]"]

        chat_id = getattr(record, "chat_id", _MISSING)
        if chat_id != _MISSING:
            parts.append(f"[cyan]chat[/]=[white]{escape(str(chat_id))}[/]")

        session_id = getattr(record, "session_id", _MISSING)
        if session_id != _MISSING:
            parts.append(f"[magenta]session[/]=[white]{escape(str(session_id))}[/]")

        prompt_cycle_id = getattr(record, "prompt_cycle_id", _MISSING)
        if prompt_cycle_id != _MISSING:
            parts.append(f"[yellow]cycle[/]=[white]{escape(str(prompt_cycle_id))}[/]")

        message = escape(record.getMessage())
        parts.append(message)
        return "  ".join(parts)


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


def _logger_label(logger_name: str) -> str:
    labels = {
        "telegram_acp_bot.telegram.bot": "telegram",
        "telegram_acp_bot.acp_app.acp_service": "acp",
    }
    return labels.get(logger_name, logger_name.rsplit(".", maxsplit=1)[-1])

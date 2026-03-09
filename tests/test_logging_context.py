from __future__ import annotations

import json
import logging
from typing import Any, cast

import pytest

from telegram_acp_bot.logging_context import bind_log_context, configure_logging, get_log_context


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_bind_log_context_is_scoped():
    assert get_log_context() == {}
    with bind_log_context(chat_id=123):
        assert get_log_context() == {"chat_id": "123"}
        with bind_log_context(session_id="abc"):
            assert get_log_context() == {"chat_id": "123", "session_id": "abc"}
        assert get_log_context() == {"chat_id": "123"}
    assert get_log_context() == {}


def test_configure_logging_text_includes_context_fields(caplog: pytest.LogCaptureFixture):
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    try:
        configure_logging(level=logging.INFO, log_format="text")
        root.addHandler(caplog.handler)
        root.setLevel(logging.INFO)
        logger = logging.getLogger("telegram_acp_bot.test")
        with bind_log_context(chat_id=42, session_id="s-1", prompt_cycle_id="c-1"):
            logger.info("hello")
    finally:
        root.handlers.clear()
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)

    assert len(caplog.records) == 1
    record = cast(Any, caplog.records[0])
    assert record.message == "hello"
    assert record.chat_id == "42"
    assert record.session_id == "s-1"
    assert record.prompt_cycle_id == "c-1"


def test_configure_logging_json_includes_context_fields(caplog: pytest.LogCaptureFixture):
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    formatter: logging.Formatter | None = None
    try:
        configure_logging(level=logging.INFO, log_format="json")
        formatter = root.handlers[0].formatter
        root.addHandler(caplog.handler)
        root.setLevel(logging.INFO)
        logger = logging.getLogger("telegram_acp_bot.test")
        with bind_log_context(chat_id=7, session_id="session-7", prompt_cycle_id="cycle-7"):
            logger.info("json hello")
    finally:
        root.handlers.clear()
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)

    assert len(caplog.records) == 1
    assert formatter is not None
    payload = json.loads(formatter.format(caplog.records[0]))
    assert payload["message"] == "json hello"
    assert payload["chat_id"] == "7"
    assert payload["session_id"] == "session-7"
    assert payload["prompt_cycle_id"] == "cycle-7"


def test_configure_logging_json_includes_exception_field(caplog: pytest.LogCaptureFixture):
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    formatter: logging.Formatter | None = None
    try:
        configure_logging(level=logging.INFO, log_format="json")
        formatter = root.handlers[0].formatter
        root.addHandler(caplog.handler)
        root.setLevel(logging.INFO)
        logger = logging.getLogger("telegram_acp_bot.test")
        try:
            _raise_runtime_error()
        except RuntimeError:
            logger.exception("failed")
    finally:
        root.handlers.clear()
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)

    assert len(caplog.records) == 1
    assert formatter is not None
    payload = json.loads(formatter.format(caplog.records[0]))
    assert payload["message"] == "failed"
    assert "exception" in payload

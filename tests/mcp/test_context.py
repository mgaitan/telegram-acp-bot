"""Tests for MCP request-context resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.mcp.support import STATE_FILE_ENV, TOKEN_ENV, mcp_channel, save_last_session_id, save_session_chat_map


def test_resolve_request_context_reports_missing_token_by_default():
    result = mcp_channel._resolve_request_context(session_id="s1")
    assert result == f"missing {TOKEN_ENV}"


def test_resolve_request_context_requires_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.setenv(STATE_FILE_ENV, str(tmp_path / "state.json"))
    result = mcp_channel._resolve_request_context(session_id="s1")
    assert result == f"missing {TOKEN_ENV}"


def test_resolve_request_context_requires_state_file_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.delenv(STATE_FILE_ENV, raising=False)
    result = mcp_channel._resolve_request_context(session_id="s1")
    assert result == f"missing {STATE_FILE_ENV}"


def test_resolve_request_context_requires_inferable_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = mcp_channel._resolve_request_context(session_id=None)

    assert result == "missing session_id and no active session could be inferred"


def test_resolve_request_context_requires_explicit_session_when_multiple_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123, "s2": 456})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = mcp_channel._resolve_request_context(session_id=None)

    assert (
        result == "missing session_id: multiple active sessions exist and no last active session could be inferred. "
        "Available session_ids: s1, s2"
    )


def test_resolve_request_context_uses_last_active_session_when_multiple_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123, "s2": 456})
    save_last_session_id(state_file, "s2")
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = mcp_channel._resolve_request_context(session_id=None)

    assert result == mcp_channel._RequestContext(token="TOKEN", chat_id=456, session_id="s2")


def test_resolve_request_context_ignores_stale_last_session_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123, "s2": 456})
    save_last_session_id(state_file, "stale")
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = mcp_channel._resolve_request_context(session_id=None)

    assert isinstance(result, str)
    assert result.startswith(
        "missing session_id: multiple active sessions exist and no last active session could be inferred."
    )

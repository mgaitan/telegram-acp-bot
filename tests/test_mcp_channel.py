"""Tests for the internal MCP channel server."""

from __future__ import annotations

import base64
import json
import stat
from pathlib import Path

import pytest

from telegram_acp_bot import mcp_channel
from telegram_acp_bot.mcp_channel_state import (
    STATE_FILE_ENV,
    TOKEN_ENV,
    save_last_session_id,
    save_session_chat_map,
)

STATE_FILE_PRIVATE_MODE = 0o600


def test_telegram_channel_info_reports_active_capabilities():
    payload = mcp_channel.telegram_channel_info()
    assert payload["status"] == "active"
    assert payload["supports_attachment_delivery"] is True
    assert payload["supports_followup_buttons"] is False


def test_main_runs_stdio_server(mocker):
    run = mocker.patch.object(mcp_channel.mcp, "run")
    mcp_channel.main()
    run.assert_called_once_with("stdio")


@pytest.mark.asyncio
async def test_send_attachment_from_path_as_photo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker):
    state_file = tmp_path / "state.json"
    image_file = tmp_path / "webcam.jpg"
    image_file.write_bytes(b"jpeg")
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(mcp_channel.ALLOW_PATH_ENV, "1")
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp_channel.Bot", return_value=bot)

    result = await mcp_channel.telegram_send_attachment(path=str(image_file))

    assert result["ok"] is True
    assert result["delivered_as"] == "photo"
    assert result["session_id"] == "s1"
    bot.send_photo.assert_awaited_once()
    bot.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_send_attachment_from_base64_as_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    save_last_session_id(state_file, "s1")
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp_channel.Bot", return_value=bot)

    result = await mcp_channel.telegram_send_attachment(
        session_id="s1",
        data_base64=base64.b64encode(b"payload").decode("ascii"),
        name="artifact.bin",
    )

    assert result["ok"] is True
    assert result["delivered_as"] == "document"
    bot.send_document.assert_awaited_once()
    bot.send_photo.assert_not_called()


@pytest.mark.asyncio
async def test_send_attachment_rejects_missing_session_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(mcp_channel.ALLOW_PATH_ENV, "1")

    result = await mcp_channel.telegram_send_attachment(session_id="unknown", path=str(tmp_path / "x.jpg"))
    assert result["ok"] is False
    assert "unknown session_id" in result["error"]


@pytest.mark.asyncio
async def test_send_attachment_rejects_invalid_base64_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = await mcp_channel.telegram_send_attachment(session_id="s1", data_base64="***not-base64***")

    assert result["ok"] is False
    assert result["error"] == "invalid base64 payload"


def test_load_attachment_bytes_rejects_missing_path(tmp_path: Path):
    result = mcp_channel._load_attachment_bytes(path=str(tmp_path / "missing.bin"), data_base64=None, name=None)
    assert isinstance(result, str)
    assert result.startswith("file not found:")


@pytest.mark.asyncio
async def test_send_attachment_rejects_path_when_not_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.delenv(mcp_channel.ALLOW_PATH_ENV, raising=False)

    result = await mcp_channel.telegram_send_attachment(session_id="s1", path=str(tmp_path / "x.jpg"))
    assert result["ok"] is False
    assert "`path` input is disabled by default" in result["error"]


@pytest.mark.asyncio
async def test_send_attachment_uses_overridden_name_for_mime_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
):
    state_file = tmp_path / "state.json"
    binary_file = tmp_path / "payload.bin"
    binary_file.write_bytes(b"jpeg")
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(mcp_channel.ALLOW_PATH_ENV, "1")
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp_channel.Bot", return_value=bot)

    result = await mcp_channel.telegram_send_attachment(path=str(binary_file), name="forced.jpg")

    assert result["ok"] is True
    assert result["delivered_as"] == "photo"
    bot.send_photo.assert_awaited_once()


def test_resolve_request_context_requires_exactly_one_payload_source():
    result = mcp_channel._resolve_request_context(session_id="s1", path=None, data_base64=None)
    assert result == "provide exactly one of `path` or `data_base64`"


def test_resolve_request_context_requires_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.setenv(STATE_FILE_ENV, str(tmp_path / "state.json"))
    result = mcp_channel._resolve_request_context(session_id="s1", path="file.bin", data_base64=None)
    assert result == f"missing {TOKEN_ENV}"


def test_resolve_request_context_requires_state_file_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.delenv(STATE_FILE_ENV, raising=False)
    result = mcp_channel._resolve_request_context(session_id="s1", path="file.bin", data_base64=None)
    assert result == f"missing {STATE_FILE_ENV}"


def test_resolve_request_context_requires_inferable_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = mcp_channel._resolve_request_context(session_id=None, path="file.bin", data_base64=None)

    assert result == "missing session_id and no active session could be inferred"


def test_load_session_chat_map_handles_invalid_json(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text("{invalid", encoding="utf-8")

    assert mcp_channel.load_session_chat_map(state_file) == {}


def test_load_session_chat_map_handles_non_dict_sessions(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text('{"sessions": []}', encoding="utf-8")

    assert mcp_channel.load_session_chat_map(state_file) == {}


def test_load_last_session_id_handles_invalid_json(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text("{invalid", encoding="utf-8")

    assert mcp_channel.load_last_session_id(state_file) is None


def test_save_last_session_id_coerces_non_dict_sessions(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"sessions": []}), encoding="utf-8")

    save_last_session_id(state_file, "s1")

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["sessions"] == {}
    assert payload["last_session_id"] == "s1"


def test_state_file_permissions_are_restricted(tmp_path: Path):
    state_file = tmp_path / "state.json"

    save_session_chat_map(state_file, {"s1": 123})
    save_last_session_id(state_file, "s1")

    mode = stat.S_IMODE(state_file.stat().st_mode)
    assert mode == STATE_FILE_PRIVATE_MODE

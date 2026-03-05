"""Tests for the internal MCP channel server."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from telegram_acp_bot import mcp_channel
from telegram_acp_bot.mcp_channel_state import STATE_FILE_ENV, TOKEN_ENV, save_session_chat_map


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
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp_channel.Bot", return_value=bot)

    result = await mcp_channel.telegram_send_attachment(session_id="s1", path=str(image_file))

    assert result["ok"] is True
    assert result["delivered_as"] == "photo"
    bot.send_photo.assert_awaited_once()
    bot.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_send_attachment_from_base64_as_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
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

    result = await mcp_channel.telegram_send_attachment(session_id="unknown", path=str(tmp_path / "x.jpg"))
    assert result["ok"] is False
    assert "unknown session_id" in result["error"]

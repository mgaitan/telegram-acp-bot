"""Tests for MCP attachment delivery tools."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import cast

import pytest

from tests.mcp.support import STATE_FILE_ENV, TOKEN_ENV, mcp_channel, save_last_session_id, save_session_chat_map


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
    mocker.patch("telegram_acp_bot.mcp.tools.attachments.Bot", return_value=bot)

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
    mocker.patch("telegram_acp_bot.mcp.tools.attachments.Bot", return_value=bot)

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
async def test_send_attachment_infers_last_active_session_when_multiple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
):
    expected_chat_id = 456
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123, "s2": expected_chat_id})
    save_last_session_id(state_file, "s2")
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp.tools.attachments.Bot", return_value=bot)

    result = await mcp_channel.telegram_send_attachment(
        data_base64=base64.b64encode(b"payload").decode("ascii"),
        name="artifact.bin",
    )

    assert result["ok"] is True
    assert result["session_id"] == "s2"
    assert result["chat_id"] == expected_chat_id
    bot.send_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_attachment_rejects_missing_session_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(mcp_channel.ALLOW_PATH_ENV, "1")

    result = await mcp_channel.telegram_send_attachment(session_id="unknown", path=str(tmp_path / "x.jpg"))
    assert result["ok"] is False
    assert "unknown session_id" in cast(str, result["error"])


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


def test_load_attachment_bytes_requires_exactly_one_source():
    assert (
        mcp_channel._load_attachment_bytes(path=None, data_base64=None, name=None)
        == "provide exactly one of `path` or `data_base64`"
    )
    assert (
        mcp_channel._load_attachment_bytes(path="artifact.bin", data_base64="cGF5bG9hZA==", name=None)
        == "provide exactly one of `path` or `data_base64`"
    )


@pytest.mark.asyncio
async def test_send_attachment_rejects_path_when_not_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.delenv(mcp_channel.ALLOW_PATH_ENV, raising=False)

    result = await mcp_channel.telegram_send_attachment(session_id="s1", path=str(tmp_path / "x.jpg"))
    assert result["ok"] is False
    assert "`path` input is disabled by default" in cast(str, result["error"])


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
    mocker.patch("telegram_acp_bot.mcp.tools.attachments.Bot", return_value=bot)

    result = await mcp_channel.telegram_send_attachment(path=str(binary_file), name="forced.jpg")

    assert result["ok"] is True
    assert result["delivered_as"] == "photo"
    bot.send_photo.assert_awaited_once()

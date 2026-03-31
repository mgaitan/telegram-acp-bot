"""Tests for MCP Telegram reaction tools."""

from __future__ import annotations

import pytest

from tests.mcp.support import (
    STATE_FILE_ENV,
    TOKEN_ENV,
    mcp_channel,
    save_prompt_message_id,
    save_session_chat_map,
)

INFERRED_MESSAGE_ID = 77


@pytest.mark.asyncio
async def test_set_message_reaction_with_explicit_message_id(tmp_path, monkeypatch: pytest.MonkeyPatch, mocker):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp.tools.reactions.Bot", return_value=bot)

    result = await mcp_channel.telegram_set_message_reaction(session_id="s1", message_id=42, emoji="👍", is_big=True)

    assert result == {
        "ok": True,
        "session_id": "s1",
        "chat_id": 123,
        "message_id": 42,
        "emoji": "👍",
        "is_big": True,
    }
    bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=42,
        reaction="👍",
        is_big=True,
    )


@pytest.mark.asyncio
async def test_set_message_reaction_normalizes_heart_variant(tmp_path, monkeypatch: pytest.MonkeyPatch, mocker):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp.tools.reactions.Bot", return_value=bot)

    result = await mcp_channel.telegram_set_message_reaction(session_id="s1", message_id=42, emoji="❤️")

    assert result["ok"] is True
    assert result["emoji"] == "❤"
    bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=42,
        reaction="❤",
        is_big=None,
    )


@pytest.mark.asyncio
async def test_set_message_reaction_uses_active_prompt_message_id(tmp_path, monkeypatch: pytest.MonkeyPatch, mocker):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    save_prompt_message_id(state_file, "s1", INFERRED_MESSAGE_ID)
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp.tools.reactions.Bot", return_value=bot)

    result = await mcp_channel.telegram_set_message_reaction(session_id="s1", emoji="🎉")

    assert result["ok"] is True
    assert result["message_id"] == INFERRED_MESSAGE_ID
    bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=INFERRED_MESSAGE_ID,
        reaction="🎉",
        is_big=None,
    )


@pytest.mark.asyncio
async def test_set_message_reaction_requires_message_id_when_none_can_be_inferred(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = await mcp_channel.telegram_set_message_reaction(session_id="s1", emoji="👍")

    assert result["ok"] is False
    assert result["error"] == "missing message_id and no active prompt message could be inferred"


@pytest.mark.asyncio
async def test_set_message_reaction_rejects_nonstandard_emoji(tmp_path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = await mcp_channel.telegram_set_message_reaction(session_id="s1", message_id=42, emoji="🫠")

    assert result["ok"] is False
    assert "unsupported reaction emoji" in str(result["error"])
    assert "Supported reactions:" in str(result["error"])


@pytest.mark.asyncio
async def test_set_message_reaction_reports_context_resolution_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.delenv(STATE_FILE_ENV, raising=False)

    result = await mcp_channel.telegram_set_message_reaction(session_id="s1", message_id=42, emoji="👍")

    assert result["ok"] is False
    assert result["error"] == f"missing {TOKEN_ENV}"

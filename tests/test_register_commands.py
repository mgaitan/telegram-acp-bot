"""Tests for the register-commands CLI sub-command."""

from __future__ import annotations

import argparse

import pytest
from telegram import (
    BotCommandScopeAllChatAdministrators,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeDefault,
)
from telegram.error import TelegramError

from telegram_acp_bot import main
from telegram_acp_bot.register_commands import (
    SCOPE_CHOICES,
    _call_api,
    get_register_commands_parser,
    register_commands_main,
)
from telegram_acp_bot.telegram.bot import BOT_COMMANDS

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch, mocker):
    """Remove real token env vars and stub load_dotenv for both dispatch paths.

    `main()` calls `telegram_acp_bot.load_dotenv`; `register_commands_main()`
    calls `telegram_acp_bot.register_commands.load_dotenv`.  Both are patched
    so neither path loads a real `.env` file during tests.
    """
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    mocker.patch("telegram_acp_bot.load_dotenv")
    return mocker.patch("telegram_acp_bot.register_commands.load_dotenv")


# ---------------------------------------------------------------------------
# Parser / constant sanity
# ---------------------------------------------------------------------------


def test_scope_choices_are_defined():
    """SCOPE_CHOICES exposes all supported scope names."""
    assert "default" in SCOPE_CHOICES
    assert "all_private_chats" in SCOPE_CHOICES
    assert "all_group_chats" in SCOPE_CHOICES
    assert "all_chat_administrators" in SCOPE_CHOICES


def test_bot_commands_non_empty():
    """BOT_COMMANDS is a non-empty sequence of (command, description) pairs."""
    assert len(BOT_COMMANDS) > 0
    for cmd, desc in BOT_COMMANDS:
        assert isinstance(cmd, str) and cmd
        assert isinstance(desc, str) and desc


def test_get_register_commands_parser_returns_parser():
    """get_register_commands_parser returns an ArgumentParser."""
    parser = get_register_commands_parser()
    assert isinstance(parser, argparse.ArgumentParser)


# ---------------------------------------------------------------------------
# register_commands_main - requires --telegram-token
# ---------------------------------------------------------------------------


def test_register_commands_main_requires_token():
    """Missing token exits with error."""
    with pytest.raises(SystemExit):
        register_commands_main([])


def test_register_commands_main_requires_token_via_main():
    """register-commands subcommand dispatched through main() also requires token."""
    with pytest.raises(SystemExit):
        main(["register-commands"])


# ---------------------------------------------------------------------------
# register_commands_main - dry-run paths
# ---------------------------------------------------------------------------


def test_dry_run_set(capsys: pytest.CaptureFixture):
    """--dry-run prints a summary of commands that would be registered."""
    rc = register_commands_main(["--telegram-token", "TOKEN", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert str(len(BOT_COMMANDS)) in out
    for cmd, desc in BOT_COMMANDS:
        assert f"/{cmd}" in out
        assert desc in out


def test_dry_run_delete(capsys: pytest.CaptureFixture):
    """--dry-run --delete prints a deletion summary without calling the API."""
    rc = register_commands_main(["--telegram-token", "TOKEN", "--dry-run", "--delete"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "delete" in out.lower()


def test_dry_run_with_scope_and_language(capsys: pytest.CaptureFixture):
    """--dry-run includes scope and language_code in its output."""
    rc = register_commands_main(
        ["--telegram-token", "TOKEN", "--dry-run", "--scope", "all_private_chats", "--language-code", "es"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "all_private_chats" in out
    assert "es" in out


def test_dry_run_no_language_code(capsys: pytest.CaptureFixture):
    """--dry-run with no language_code shows None in the output."""
    rc = register_commands_main(["--telegram-token", "TOKEN", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "None" in out


# ---------------------------------------------------------------------------
# register_commands_main - real API call paths (mocked _call_api)
# ---------------------------------------------------------------------------


def test_register_success(mocker, capsys: pytest.CaptureFixture):
    """Successful set_my_commands prints confirmation and returns 0."""
    mocker.patch("telegram_acp_bot.register_commands._call_api", new=mocker.AsyncMock())
    rc = register_commands_main(["--telegram-token", "TOKEN"])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(len(BOT_COMMANDS)) in out
    assert "Registered" in out


def test_delete_success(mocker, capsys: pytest.CaptureFixture):
    """Successful delete_my_commands prints confirmation and returns 0."""
    mocker.patch("telegram_acp_bot.register_commands._call_api", new=mocker.AsyncMock())
    rc = register_commands_main(["--telegram-token", "TOKEN", "--delete"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "deleted" in out.lower()


def test_register_with_language_code(mocker, capsys: pytest.CaptureFixture):
    """Successful registration with language_code shows it in output."""
    mocker.patch("telegram_acp_bot.register_commands._call_api", new=mocker.AsyncMock())
    rc = register_commands_main(["--telegram-token", "TOKEN", "--language-code", "en"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "'en'" in out


def test_register_telegram_error_returns_1(mocker, capsys: pytest.CaptureFixture):
    """TelegramError is caught, printed to stderr, and exits with code 1."""
    mocker.patch(
        "telegram_acp_bot.register_commands._call_api",
        new=mocker.AsyncMock(side_effect=TelegramError("Invalid token")),
    )
    rc = register_commands_main(["--telegram-token", "BADTOKEN"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Invalid token" in err


# ---------------------------------------------------------------------------
# register_commands_main - dispatched via main()
# ---------------------------------------------------------------------------


def test_main_dispatches_to_register_commands(mocker, capsys: pytest.CaptureFixture):
    """main(['register-commands', ...]) delegates to register_commands_main."""
    mocker.patch("telegram_acp_bot.register_commands._call_api", new=mocker.AsyncMock())
    rc = main(["register-commands", "--telegram-token", "TOKEN", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out


# ---------------------------------------------------------------------------
# _call_api - async unit tests for each scope and delete path
# ---------------------------------------------------------------------------


async def test_call_api_set_default_scope(mocker):
    """_call_api calls set_my_commands with default scope."""
    mock_bot = mocker.AsyncMock()
    mock_bot.__aenter__ = mocker.AsyncMock(return_value=mock_bot)
    mock_bot.__aexit__ = mocker.AsyncMock(return_value=False)
    mocker.patch("telegram_acp_bot.register_commands.Bot", return_value=mock_bot)

    await _call_api(token="TOKEN", scope_str="default", language_code=None, delete=False)

    mock_bot.set_my_commands.assert_awaited_once()
    call_kwargs = mock_bot.set_my_commands.call_args
    assert isinstance(call_kwargs.kwargs["scope"], BotCommandScopeDefault)
    assert call_kwargs.kwargs["language_code"] is None


async def test_call_api_set_all_private_chats(mocker):
    """_call_api calls set_my_commands with all_private_chats scope."""
    mock_bot = mocker.AsyncMock()
    mock_bot.__aenter__ = mocker.AsyncMock(return_value=mock_bot)
    mock_bot.__aexit__ = mocker.AsyncMock(return_value=False)
    mocker.patch("telegram_acp_bot.register_commands.Bot", return_value=mock_bot)

    await _call_api(token="TOKEN", scope_str="all_private_chats", language_code="en", delete=False)

    mock_bot.set_my_commands.assert_awaited_once()
    call_kwargs = mock_bot.set_my_commands.call_args
    assert isinstance(call_kwargs.kwargs["scope"], BotCommandScopeAllPrivateChats)
    assert call_kwargs.kwargs["language_code"] == "en"


async def test_call_api_set_all_group_chats(mocker):
    """_call_api uses all_group_chats scope."""
    mock_bot = mocker.AsyncMock()
    mock_bot.__aenter__ = mocker.AsyncMock(return_value=mock_bot)
    mock_bot.__aexit__ = mocker.AsyncMock(return_value=False)
    mocker.patch("telegram_acp_bot.register_commands.Bot", return_value=mock_bot)

    await _call_api(token="TOKEN", scope_str="all_group_chats", language_code=None, delete=False)

    call_kwargs = mock_bot.set_my_commands.call_args
    assert isinstance(call_kwargs.kwargs["scope"], BotCommandScopeAllGroupChats)


async def test_call_api_set_all_chat_administrators(mocker):
    """_call_api uses all_chat_administrators scope."""
    mock_bot = mocker.AsyncMock()
    mock_bot.__aenter__ = mocker.AsyncMock(return_value=mock_bot)
    mock_bot.__aexit__ = mocker.AsyncMock(return_value=False)
    mocker.patch("telegram_acp_bot.register_commands.Bot", return_value=mock_bot)

    await _call_api(token="TOKEN", scope_str="all_chat_administrators", language_code=None, delete=False)

    call_kwargs = mock_bot.set_my_commands.call_args
    assert isinstance(call_kwargs.kwargs["scope"], BotCommandScopeAllChatAdministrators)


async def test_call_api_delete(mocker):
    """_call_api calls delete_my_commands when delete=True."""
    mock_bot = mocker.AsyncMock()
    mock_bot.__aenter__ = mocker.AsyncMock(return_value=mock_bot)
    mock_bot.__aexit__ = mocker.AsyncMock(return_value=False)
    mocker.patch("telegram_acp_bot.register_commands.Bot", return_value=mock_bot)

    await _call_api(token="TOKEN", scope_str="default", language_code=None, delete=True)

    mock_bot.delete_my_commands.assert_awaited_once()
    mock_bot.set_my_commands.assert_not_awaited()
    call_kwargs = mock_bot.delete_my_commands.call_args
    assert isinstance(call_kwargs.kwargs["scope"], BotCommandScopeDefault)

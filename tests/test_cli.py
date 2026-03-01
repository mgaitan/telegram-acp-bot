"""Tests for the CLI."""

from __future__ import annotations

import runpy
import sys
from importlib import metadata

import pytest

from telegram_acp_bot import get_version, main

STDIO_LIMIT_TEST_VALUE = 2_097_152


@pytest.fixture(autouse=True)
def isolate_token_sources(monkeypatch: pytest.MonkeyPatch, mocker):
    """Prevent tests from loading real token values from environment or .env files."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    return mocker.patch("telegram_acp_bot.load_dotenv")


def test_main_loads_dotenv(isolate_token_sources, mocker) -> None:
    """CLI loads .env before parsing arguments."""
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    assert isolate_token_sources is not None
    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent"]) == 0
    isolate_token_sources.assert_called_once_with(override=False)


def test_main_requires_token() -> None:
    """Running the bot without token should fail fast."""
    with pytest.raises(SystemExit):
        main([])


def test_main_requires_agent_command() -> None:
    """Running the bot without ACP agent command should fail fast."""
    with pytest.raises(SystemExit):
        main(["--telegram-token", "TOKEN"])


def test_main_runs_bot(mocker) -> None:
    """Run path delegates to run_polling."""
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent --flag"]) == 0
    mock_run_polling.assert_called_once()


def test_main_passes_stdio_limit_to_service(mocker) -> None:
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    service_ctor = mocker.patch("telegram_acp_bot.AcpAgentService")

    assert (
        main(
            [
                "--telegram-token",
                "TOKEN",
                "--agent-command",
                "agent",
                "--acp-stdio-limit",
                str(STDIO_LIMIT_TEST_VALUE),
            ]
        )
        == 0
    )
    service_ctor.assert_called_once()
    assert service_ctor.call_args.kwargs["stdio_limit"] == STDIO_LIMIT_TEST_VALUE


def test_main_uses_env_token(mocker, monkeypatch) -> None:
    """Run path uses TELEGRAM_BOT_TOKEN when loaded from environment."""
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN")
    monkeypatch.setenv("ACP_AGENT_COMMAND", "agent")
    assert main([]) == 0
    mock_run_polling.assert_called_once()


def test_main_rejects_blank_agent_command(mocker) -> None:
    """Whitespace-only agent command should be rejected."""
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    with pytest.raises(SystemExit):
        main(["--telegram-token", "TOKEN", "--agent-command", "   "])


def test_main_rejects_non_positive_stdio_limit(mocker) -> None:
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    with pytest.raises(SystemExit):
        main(["--telegram-token", "TOKEN", "--agent-command", "agent", "--acp-stdio-limit", "0"])


def test_show_help(capsys: pytest.CaptureFixture) -> None:
    """Show help.

    Parameters:
        capsys: Pytest fixture to capture output.
    """
    with pytest.raises(SystemExit):
        main(["-h"])
    captured = capsys.readouterr()
    assert "acp-bot" in captured.out


def test_show_version(mocker, capsys: pytest.CaptureFixture) -> None:
    """Show version.

    Parameters:
        mocker: pytest-mock fixture to patch get_version.
        capsys: Pytest fixture to capture output.
    """
    mocker.patch("telegram_acp_bot.get_version", return_value="0.1.0")
    with pytest.raises(SystemExit):
        main(["-V"])
    captured = capsys.readouterr()
    assert "0.1.0" in captured.out


def test_main_module(mocker):
    """Test running the CLI via __main__ (python -m ...)."""
    module_name = "telegram_acp_bot.__main__"
    # Simulate: python -m acp-bot --version
    mocker.patch.object(sys, "argv", ["acp-bot", "-V"])
    with pytest.raises(SystemExit):
        runpy.run_module(module_name, run_name="__main__", alter_sys=False)


def test_get_version_package_not_found(mocker):
    """Test get_version returns 'unknown' if package is not found."""
    mocker.patch(
        "importlib.metadata.version",
        side_effect=metadata.PackageNotFoundError("not found"),
    )
    assert get_version() == "unknown"

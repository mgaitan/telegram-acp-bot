"""Tests for the CLI."""

from __future__ import annotations

import runpy
import sys
from importlib import metadata

import pytest

from telegram_acp_bot import get_version, main
from telegram_acp_bot.telegram.bot import RESTART_EXIT_CODE

CUSTOM_STDIO_LIMIT = 12_345
CUSTOM_CONNECT_TIMEOUT = 42.5


@pytest.fixture(autouse=True)
def isolate_token_sources(monkeypatch: pytest.MonkeyPatch, mocker):
    """Prevent tests from loading real token values from environment or .env files."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    return mocker.patch("telegram_acp_bot.load_dotenv")


def test_main_loads_dotenv(isolate_token_sources, mocker):
    """CLI loads .env before parsing arguments."""
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    assert isolate_token_sources is not None
    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent"]) == 0
    isolate_token_sources.assert_called_once_with(override=False)


def test_main_requires_token():
    """Running the bot without token should fail fast."""
    with pytest.raises(SystemExit):
        main([])


def test_main_requires_agent_command():
    """Running the bot without ACP agent command should fail fast."""
    with pytest.raises(SystemExit):
        main(["--telegram-token", "TOKEN"])


def test_main_runs_bot(mocker):
    """Run path delegates to run_polling."""
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent --flag"]) == 0
    mock_run_polling.assert_called_once()


def test_main_reexecs_process_on_restart_request(mocker):
    """When polling returns restart code, main re-execs current process."""
    mocker.patch("telegram_acp_bot.run_polling", return_value=RESTART_EXIT_CODE)
    mock_execvp = mocker.patch("telegram_acp_bot.os.execvp")
    mock_execv = mocker.patch("telegram_acp_bot.os.execv", side_effect=OSError("exec failed"))
    mocker.patch("telegram_acp_bot.sys.argv", ["acp-bot", "--telegram-token", "TOKEN", "--agent-command", "agent"])
    mocker.patch("telegram_acp_bot.sys.executable", "/usr/bin/python3")

    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent"]) == 1
    mock_execvp.assert_not_called()
    mock_execv.assert_called_once_with(
        "/usr/bin/python3", ["/usr/bin/python3", "acp-bot", "--telegram-token", "TOKEN", "--agent-command", "agent"]
    )


def test_main_reexecs_using_restart_command(mocker):
    """When restart command is configured, main re-execs using that command."""
    mocker.patch("telegram_acp_bot.run_polling", return_value=RESTART_EXIT_CODE)
    mock_execvp = mocker.patch("telegram_acp_bot.os.execvp", side_effect=OSError("execvp failed"))
    mock_execv = mocker.patch("telegram_acp_bot.os.execv")

    assert (
        main(
            [
                "--telegram-token",
                "TOKEN",
                "--agent-command",
                "agent",
                "--restart-command",
                "uv run acp-bot --telegram-token TOKEN --agent-command agent",
            ]
        )
        == 1
    )
    mock_execvp.assert_called_once_with(
        "uv",
        ["uv", "run", "acp-bot", "--telegram-token", "TOKEN", "--agent-command", "agent"],
    )
    mock_execv.assert_not_called()


def test_main_uses_env_token(mocker, monkeypatch):
    """Run path uses TELEGRAM_BOT_TOKEN when loaded from environment."""
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN")
    monkeypatch.setenv("ACP_AGENT_COMMAND", "agent")
    assert main([]) == 0
    mock_run_polling.assert_called_once()


def test_main_rejects_blank_agent_command(mocker):
    """Whitespace-only agent command should be rejected."""
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    with pytest.raises(SystemExit):
        main(["--telegram-token", "TOKEN", "--agent-command", "   "])


def test_main_rejects_non_positive_stdio_limit(mocker):
    """ACP stdio limit must be a positive integer."""
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    with pytest.raises(SystemExit):
        main(["--telegram-token", "TOKEN", "--agent-command", "agent", "--acp-stdio-limit", "0"])


def test_main_rejects_non_positive_connect_timeout(mocker):
    """ACP connect timeout must be a positive number."""
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    with pytest.raises(SystemExit):
        main(["--telegram-token", "TOKEN", "--agent-command", "agent", "--acp-connect-timeout", "0"])


def test_main_passes_stdio_limit_to_service(mocker):
    """CLI forwards acp stdio limit to service constructor."""
    mock_service = mocker.patch("telegram_acp_bot.AcpAgentService")
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)

    assert (
        main(
            [
                "--telegram-token",
                "TOKEN",
                "--agent-command",
                "agent",
                "--acp-stdio-limit",
                str(CUSTOM_STDIO_LIMIT),
            ]
        )
        == 0
    )
    assert mock_service.call_args is not None
    assert mock_service.call_args.kwargs["stdio_limit"] == CUSTOM_STDIO_LIMIT


def test_main_passes_connect_timeout_to_service(mocker):
    """CLI forwards ACP connect timeout to service constructor."""
    mock_service = mocker.patch("telegram_acp_bot.AcpAgentService")
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)

    assert (
        main(
            [
                "--telegram-token",
                "TOKEN",
                "--agent-command",
                "agent",
                "--acp-connect-timeout",
                str(CUSTOM_CONNECT_TIMEOUT),
            ]
        )
        == 0
    )
    assert mock_service.call_args is not None
    assert mock_service.call_args.kwargs["connect_timeout"] == CUSTOM_CONNECT_TIMEOUT


def test_show_help(capsys: pytest.CaptureFixture):
    """Show help.

    Parameters:
        capsys: Pytest fixture to capture output.
    """
    with pytest.raises(SystemExit):
        main(["-h"])
    captured = capsys.readouterr()
    assert "acp-bot" in captured.out


def test_show_version(mocker, capsys: pytest.CaptureFixture):
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

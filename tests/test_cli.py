"""Tests for the CLI."""

from __future__ import annotations

import json
import os
import runpy
import sys
from importlib import metadata
from pathlib import Path

import pytest

from telegram_acp_bot import get_version, main
from telegram_acp_bot.mcp.state import STATE_FILE_ENV, TOKEN_ENV
from telegram_acp_bot.telegram.bot import RESTART_EXIT_CODE

CUSTOM_STDIO_LIMIT = 12_345
CUSTOM_CONNECT_TIMEOUT = 42.5
CONFIG_FILE_STDIO_LIMIT = 1024
CONFIG_FILE_CONNECT_TIMEOUT = 5.0


@pytest.fixture(autouse=True)
def isolate_token_sources(monkeypatch: pytest.MonkeyPatch, mocker):
    """Prevent tests from loading real token values from environment or .env files."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERNAMES", raising=False)
    # Keep a default allowlist to satisfy secure-by-default startup validation.
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "1")
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


def test_main_requires_allowlist(mocker, monkeypatch):
    """Running the bot without explicit allowlist should fail fast."""
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERNAMES", raising=False)
    with pytest.raises(SystemExit):
        main(["--telegram-token", "TOKEN", "--agent-command", "agent"])


def test_main_accepts_allowed_usernames_from_env(mocker, monkeypatch):
    """Username allowlist from env should satisfy startup requirements."""
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERNAMES", "Alice, @Bob")

    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent"]) == 0
    mock_run_polling.assert_called_once()


def test_main_normalizes_allowed_username_cli(mocker, monkeypatch):
    """CLI usernames should be normalized (lowercase, no @)."""
    mocker.patch("telegram_acp_bot.AcpAgentService")
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)

    assert (
        main(
            [
                "--telegram-token",
                "TOKEN",
                "--agent-command",
                "agent",
                "--allowed-username",
                "@Alice",
            ]
        )
        == 0
    )
    assert mock_run_polling.call_args is not None
    config = mock_run_polling.call_args.args[0]
    assert config.allowed_usernames == {"alice"}


def test_main_rejects_invalid_allowed_user_ids_env(mocker, monkeypatch):
    """Invalid TELEGRAM_ALLOWED_USER_IDS should fail with parser error."""
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "1,abc")

    with pytest.raises(SystemExit):
        main(["--telegram-token", "TOKEN", "--agent-command", "agent"])


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
    mocker.patch(
        "telegram_acp_bot.sys.argv",
        ["telegram-acp-bot", "--telegram-token", "TOKEN", "--agent-command", "agent"],
    )
    mocker.patch("telegram_acp_bot.sys.executable", "/usr/bin/python3")

    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent"]) == 1
    mock_execvp.assert_not_called()
    mock_execv.assert_called_once_with(
        "/usr/bin/python3",
        [
            "/usr/bin/python3",
            "telegram-acp-bot",
            "--telegram-token",
            "TOKEN",
            "--agent-command",
            "agent",
        ],
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
                "uv run telegram-acp-bot --telegram-token TOKEN --agent-command agent",
            ]
        )
        == 1
    )
    mock_execvp.assert_called_once_with(
        "uv",
        ["uv", "run", "telegram-acp-bot", "--telegram-token", "TOKEN", "--agent-command", "agent"],
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


def test_main_passes_default_internal_mcp_server_to_service(mocker):
    """CLI always forwards the built-in MCP stdio server to service constructor."""
    mock_service = mocker.patch("telegram_acp_bot.AcpAgentService")
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)

    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent"]) == 0
    assert mock_service.call_args is not None
    mcp_servers = mock_service.call_args.kwargs["mcp_servers"]
    assert len(mcp_servers) == 1
    server = mcp_servers[0]
    assert server.name == "telegram-channel"
    assert server.command == sys.executable
    assert server.args == ["-m", "telegram_acp_bot.mcp.server"]
    env = {item.name: item.value for item in server.env}
    assert env[TOKEN_ENV] == "TOKEN"
    assert STATE_FILE_ENV in env


def test_show_help(capsys: pytest.CaptureFixture):
    """Show help.

    Parameters:
        capsys: Pytest fixture to capture output.
    """
    with pytest.raises(SystemExit):
        main(["-h"])
    captured = capsys.readouterr()
    assert "telegram-acp-bot" in captured.out
    assert "register-commands" in captured.out


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
    # Simulate: python -m telegram-acp-bot --version
    mocker.patch.object(sys, "argv", ["telegram-acp-bot", "-V"])
    with pytest.raises(SystemExit):
        runpy.run_module(module_name, run_name="__main__", alter_sys=False)


def test_get_version_package_not_found(mocker):
    """Test get_version returns 'unknown' if package is not found."""
    mocker.patch(
        "importlib.metadata.version",
        side_effect=metadata.PackageNotFoundError("not found"),
    )
    assert get_version() == "unknown"


def test_main_activity_mode_defaults_to_normal(mocker):
    """Default activity mode is normal."""
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent"]) == 0
    assert mock_run_polling.call_args is not None
    config = mock_run_polling.call_args.args[0]
    assert config.activity_mode == "normal"
    assert config.compact_activity is False


def test_main_compact_activity_mode_sets_flag(mocker):
    """--activity-mode compact sets compact_activity=True on the config."""
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent", "--activity-mode", "compact"]) == 0
    assert mock_run_polling.call_args is not None
    config = mock_run_polling.call_args.args[0]
    assert config.compact_activity is True


def test_main_activity_mode_env_var_compact(mocker, monkeypatch):
    """ACP_ACTIVITY_MODE=compact activates compact mode."""
    monkeypatch.setenv("ACP_ACTIVITY_MODE", "compact")
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent"]) == 0
    assert mock_run_polling.call_args is not None
    config = mock_run_polling.call_args.args[0]
    assert config.compact_activity is True


def test_main_verbose_activity_mode_sets_flag(mocker):
    """--activity-mode verbose keeps the verbose activity mode on the config."""
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent", "--activity-mode", "verbose"]) == 0
    assert mock_run_polling.call_args is not None
    config = mock_run_polling.call_args.args[0]
    assert config.activity_mode == "verbose"


def test_main_activity_mode_short_alias_sets_flag(mocker):
    """-m verbose is accepted as shorthand for --activity-mode verbose."""
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    assert main(["--telegram-token", "TOKEN", "--agent-command", "agent", "-m", "verbose"]) == 0
    assert mock_run_polling.call_args is not None
    config = mock_run_polling.call_args.args[0]
    assert config.activity_mode == "verbose"


# Config file tests


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return path


def test_main_config_file_provides_token_and_agent_command(mocker, monkeypatch, tmp_path):
    """Config file values are used when env vars and CLI flags are absent."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    config_path = _write_config(
        tmp_path, {"telegram": {"bot_token": "CFG_TOKEN"}, "acp": {"agent_command": "cfg_agent"}}
    )
    assert main(["--config", str(config_path)]) == 0


def test_main_config_file_provides_allowlist(mocker, monkeypatch, tmp_path):
    """Allowed user IDs from config file satisfy startup validation."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    monkeypatch.delenv("TELEGRAM_ALLOWED_USER_IDS", raising=False)
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    config_path = _write_config(
        tmp_path,
        {"telegram": {"bot_token": "T", "allowed_user_ids": [42]}, "acp": {"agent_command": "a"}},
    )
    assert main(["--config", str(config_path)]) == 0


def test_main_env_token_overrides_config_file(mocker, monkeypatch, tmp_path):
    """TELEGRAM_BOT_TOKEN env var takes precedence over config file."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "ENV_TOKEN")
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    config_path = _write_config(tmp_path, {"telegram": {"bot_token": "CFG_TOKEN"}, "acp": {"agent_command": "a"}})
    assert main(["--config", str(config_path)]) == 0
    config = mock_run_polling.call_args.args[0]
    assert config.token == "ENV_TOKEN"


def test_main_cli_token_overrides_config_file(mocker, monkeypatch, tmp_path):
    """--telegram-token CLI flag takes precedence over config file."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    config_path = _write_config(tmp_path, {"telegram": {"bot_token": "CFG_TOKEN"}, "acp": {"agent_command": "a"}})
    assert main(["--config", str(config_path), "--telegram-token", "CLI_TOKEN"]) == 0
    config = mock_run_polling.call_args.args[0]
    assert config.token == "CLI_TOKEN"


def test_main_config_file_activity_mode(mocker, monkeypatch, tmp_path):
    """activity_mode from config file is applied."""
    monkeypatch.delenv("ACP_ACTIVITY_MODE", raising=False)
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    config_path = _write_config(
        tmp_path,
        {"telegram": {"bot_token": "T"}, "acp": {"agent_command": "a", "activity_mode": "compact"}},
    )
    assert main(["--config", str(config_path)]) == 0
    config = mock_run_polling.call_args.args[0]
    assert config.compact_activity is True


def test_main_config_file_not_found_error(tmp_path):
    """--config pointing to a missing file causes a startup error."""
    with pytest.raises(SystemExit):
        main(["--config", str(tmp_path / "missing.json"), "--telegram-token", "T", "--agent-command", "a"])


def test_main_config_file_invalid_json_error(tmp_path):
    """--config pointing to a file with invalid JSON causes a startup error."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(SystemExit):
        main(["--config", str(bad), "--telegram-token", "T", "--agent-command", "a"])


def test_main_config_file_invalid_value_error(tmp_path):
    """--config with an invalid value (e.g. bad permission_mode) causes a startup error."""
    config_path = _write_config(tmp_path, {"acp": {"permission_mode": "not_valid"}})
    with pytest.raises(SystemExit):
        main(["--config", str(config_path), "--telegram-token", "T", "--agent-command", "a"])


def test_main_config_file_workspace(mocker, monkeypatch, tmp_path):
    """workspace from config file is passed to make_config."""
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    config_path = _write_config(
        tmp_path,
        {"telegram": {"bot_token": "T"}, "acp": {"agent_command": "a", "workspace": str(tmp_path)}},
    )
    assert main(["--config", str(config_path)]) == 0
    config = mock_run_polling.call_args.args[0]
    assert config.default_workspace == tmp_path


def test_main_config_file_external_stdio_mcp_server(mocker, monkeypatch, tmp_path):
    """Extra stdio MCP servers from config file are passed to the service."""
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    mock_service = mocker.patch("telegram_acp_bot.AcpAgentService")
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    config_path = _write_config(
        tmp_path,
        {
            "telegram": {"bot_token": "T"},
            "acp": {"agent_command": "a"},
            "mcp_servers": {"echo": {"command": "uv", "args": ["run", "echo.py"], "env": {"K": "v"}}},
        },
    )
    assert main(["--config", str(config_path)]) == 0
    mcp_servers = mock_service.call_args.kwargs["mcp_servers"]
    names = [s.name for s in mcp_servers]
    assert "telegram-channel" in names
    assert "echo" in names
    echo = next(s for s in mcp_servers if s.name == "echo")
    assert echo.command == "uv"
    assert echo.args == ["run", "echo.py"]
    assert any(e.name == "K" and e.value == "v" for e in echo.env)


def test_main_config_file_external_http_mcp_server(mocker, monkeypatch, tmp_path):
    """Extra remote HTTP MCP servers from config file are passed to the service."""
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    mock_service = mocker.patch("telegram_acp_bot.AcpAgentService")
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    config_path = _write_config(
        tmp_path,
        {
            "telegram": {"bot_token": "T"},
            "acp": {"agent_command": "a"},
            "mcp_servers": {"remote": {"url": "https://mcp.example.com/mcp", "headers": {"Auth": "Bearer tok"}}},
        },
    )
    assert main(["--config", str(config_path)]) == 0
    mcp_servers = mock_service.call_args.kwargs["mcp_servers"]
    remote = next(s for s in mcp_servers if s.name == "remote")
    assert remote.url == "https://mcp.example.com/mcp"
    assert any(h.name == "Auth" and h.value == "Bearer tok" for h in remote.headers)


def test_main_config_file_external_server_overrides_internal(mocker, monkeypatch, tmp_path):
    """A config server named 'telegram-channel' replaces the internal one."""
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    mock_service = mocker.patch("telegram_acp_bot.AcpAgentService")
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    config_path = _write_config(
        tmp_path,
        {
            "telegram": {"bot_token": "T"},
            "acp": {"agent_command": "a"},
            "mcp_servers": {"telegram-channel": {"command": "custom-server"}},
        },
    )
    assert main(["--config", str(config_path)]) == 0
    mcp_servers = mock_service.call_args.kwargs["mcp_servers"]
    channel = next(s for s in mcp_servers if s.name == "telegram-channel")
    assert channel.command == "custom-server"


def test_main_config_file_stdio_limit_and_timeout(mocker, monkeypatch, tmp_path):
    """stdio_limit and connect_timeout from config file are forwarded to the service."""
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    monkeypatch.delenv("ACP_STDIO_LIMIT", raising=False)
    monkeypatch.delenv("ACP_CONNECT_TIMEOUT", raising=False)
    mock_service = mocker.patch("telegram_acp_bot.AcpAgentService")
    mocker.patch("telegram_acp_bot.run_polling", return_value=0)
    config_path = _write_config(
        tmp_path,
        {
            "telegram": {"bot_token": "T"},
            "acp": {
                "agent_command": "a",
                "stdio_limit": CONFIG_FILE_STDIO_LIMIT,
                "connect_timeout": CONFIG_FILE_CONNECT_TIMEOUT,
            },
        },
    )
    assert main(["--config", str(config_path)]) == 0
    assert mock_service.call_args.kwargs["stdio_limit"] == CONFIG_FILE_STDIO_LIMIT
    assert mock_service.call_args.kwargs["connect_timeout"] == CONFIG_FILE_CONNECT_TIMEOUT


def test_register_commands_subcommand_uses_config_file(mocker, monkeypatch, tmp_path):
    """Subcommands keep config-file defaults after the full parse."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    mock_register = mocker.patch("telegram_acp_bot.register_commands._execute_register_commands", return_value=0)
    config_path = _write_config(tmp_path, {"telegram": {"bot_token": "CFG_TOKEN"}})

    assert main(["--config", str(config_path), "register-commands"]) == 0
    assert mock_register.call_count == 1
    assert mock_register.call_args is not None
    assert mock_register.call_args.args[0].telegram_token == "CFG_TOKEN"


def test_main_auto_discovers_config_file(mocker, monkeypatch, tmp_path):
    """Config file is automatically discovered from standard locations."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)

    # Create config in one of the standard locations
    config_dir = tmp_path / ".telegram_acp_bot"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text('{"telegram": {"bot_token": "AUTO_TOKEN"}, "acp": {"agent_command": "auto_agent"}}')

    # Change to the temp directory and run without --config
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert main([]) == 0
        config = mock_run_polling.call_args.args[0]
        assert config.token == "AUTO_TOKEN"
    finally:
        os.chdir(original_cwd)


def test_main_explicit_config_overrides_auto_discovery(mocker, monkeypatch, tmp_path):
    """Explicit --config path takes precedence over auto-discovered files."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ACP_AGENT_COMMAND", raising=False)
    mock_run_polling = mocker.patch("telegram_acp_bot.run_polling", return_value=0)

    # Create config in standard location
    config_dir = tmp_path / ".telegram_acp_bot"
    config_dir.mkdir()
    auto_config = config_dir / "config.json"
    auto_config.write_text('{"telegram": {"bot_token": "AUTO_TOKEN"}, "acp": {"agent_command": "auto_agent"}}')

    # Create explicit config with different values
    explicit_config = tmp_path / "explicit.json"
    explicit_config.write_text(
        '{"telegram": {"bot_token": "EXPLICIT_TOKEN"}, "acp": {"agent_command": "explicit_agent"}}'
    )

    # Change to the temp directory and run with explicit --config
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert main(["--config", str(explicit_config)]) == 0
        config = mock_run_polling.call_args.args[0]
        assert config.token == "EXPLICIT_TOKEN"
    finally:
        os.chdir(original_cwd)

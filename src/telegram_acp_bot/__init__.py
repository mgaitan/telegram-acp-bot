"""
Telegram ACP bot

A Telegram bot that implements Agent Client Protocol to interact with AI agents.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys
from importlib import metadata
from pathlib import Path
from typing import cast

from acp.schema import EnvVariable, McpServerStdio
from dotenv import load_dotenv

from telegram_acp_bot.acp_app.acp_service import AcpAgentService
from telegram_acp_bot.acp_app.models import PermissionEventOutput, PermissionMode
from telegram_acp_bot.core.session_registry import SessionRegistry
from telegram_acp_bot.logging_context import configure_logging
from telegram_acp_bot.mcp_channel_state import STATE_FILE_ENV, TOKEN_ENV, default_state_file
from telegram_acp_bot.register_commands import add_register_commands_subparser
from telegram_acp_bot.telegram.bot import RESTART_EXIT_CODE, BotConfig, TelegramBridge, make_config, run_polling

ALLOWED_USER_IDS_ENV = "TELEGRAM_ALLOWED_USER_IDS"
ALLOWED_USERNAMES_ENV = "TELEGRAM_ALLOWED_USERNAMES"


def get_version() -> str:
    try:
        return metadata.version("telegram-acp-bot")
    except metadata.PackageNotFoundError:
        return "unknown"


def get_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="telegram-acp-bot")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {get_version()}")

    subparsers = parser.add_subparsers(dest="subcommand", metavar="command")
    add_register_commands_subparser(subparsers)

    parser.add_argument("--telegram-token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""), help="Telegram bot token")
    parser.add_argument(
        "--agent-command",
        default=os.getenv("ACP_AGENT_COMMAND", ""),
        help="ACP agent command line, e.g. 'codex-acp' or 'uv run examples/echo_agent.py'.",
    )
    parser.add_argument(
        "--allowed-user-id",
        action="append",
        default=[],
        type=int,
        help="Allowed Telegram user ID. Can be repeated.",
    )
    parser.add_argument(
        "--allowed-username",
        action="append",
        default=[],
        help="Allowed Telegram username (with or without @). Can be repeated.",
    )
    parser.add_argument(
        "--workspace",
        default=os.getcwd(),
        help="Default workspace path for /new when path is not provided.",
    )
    parser.add_argument(
        "--permission-mode",
        default=os.getenv("ACP_PERMISSION_MODE", "ask"),
        choices=["ask", "approve", "deny"],
        help="Default ACP permission mode.",
    )
    parser.add_argument(
        "--permission-event-output",
        default=os.getenv("ACP_PERMISSION_EVENT_OUTPUT", "stdout"),
        choices=["stdout", "off"],
        help="Where ACP permission/tool event logs are emitted.",
    )
    parser.add_argument(
        "--acp-stdio-limit",
        default=int(os.getenv("ACP_STDIO_LIMIT", "8388608")),
        type=int,
        help="Asyncio StreamReader limit in bytes for ACP stdio transport.",
    )
    parser.add_argument(
        "--acp-connect-timeout",
        default=float(os.getenv("ACP_CONNECT_TIMEOUT", "30")),
        type=float,
        help="Timeout in seconds for ACP initialize/new_session handshake.",
    )
    parser.add_argument(
        "--restart-command",
        default=os.getenv("ACP_RESTART_COMMAND", ""),
        help="Optional command used by /restart to relaunch the bot (e.g. 'uv run telegram-acp-bot').",
    )
    parser.add_argument(
        "--log-format",
        default=os.getenv("ACP_LOG_FORMAT", "text"),
        choices=["text", "json"],
        help="Application log format.",
    )
    return parser


def _default_mcp_servers(*, telegram_token: str, state_file: Path) -> tuple[McpServerStdio, ...]:
    """Return MCP servers that should always be exposed to the ACP agent."""

    return (
        McpServerStdio(
            name="telegram-channel",
            command=sys.executable,
            args=["-m", "telegram_acp_bot.mcp_channel"],
            env=[
                EnvVariable(name=TOKEN_ENV, value=telegram_token),
                EnvVariable(name=STATE_FILE_ENV, value=str(state_file)),
            ],
        ),
    )


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_username(username: str) -> str:
    return username.lstrip("@").strip().lower()


def _resolve_allowed_users(*, parser: argparse.ArgumentParser, opts: argparse.Namespace) -> tuple[list[int], list[str]]:
    try:
        env_allowed_ids = [int(item) for item in _parse_csv(os.getenv(ALLOWED_USER_IDS_ENV, ""))]
    except ValueError:
        parser.error(f"{ALLOWED_USER_IDS_ENV} must be a comma-separated list of integers")
    env_allowed_usernames = [_normalize_username(item) for item in _parse_csv(os.getenv(ALLOWED_USERNAMES_ENV, ""))]
    cli_allowed_usernames = [_normalize_username(item) for item in opts.allowed_username]
    allowed_user_ids = [*env_allowed_ids, *opts.allowed_user_id]
    allowed_usernames = [*env_allowed_usernames, *cli_allowed_usernames]
    if not allowed_user_ids and not allowed_usernames:
        parser.error(
            "--allowed-user-id/--allowed-username (or TELEGRAM_ALLOWED_USER_IDS/TELEGRAM_ALLOWED_USERNAMES) "
            "must include at least one allowed user",
        )
    return allowed_user_ids, allowed_usernames


def _run_bot_loop(
    config: BotConfig,
    bridge: TelegramBridge,
    restart_command_parts: list[str] | None,
) -> int:
    """Poll until normal exit; re-exec the process on restart requests."""
    while True:
        exit_code = run_polling(config, bridge)
        if exit_code != RESTART_EXIT_CODE:
            return exit_code
        try:
            if restart_command_parts is not None:
                logging.info(
                    "Restart requested via Telegram command. Re-execing restart command: %s",
                    restart_command_parts,
                )
                os.execvp(restart_command_parts[0], restart_command_parts)
            # TODO: Drop manual re-exec when uv can natively watch local package code and
            # restart commands for this workflow. Tracking: astral-sh/uv#9652.
            argv = [sys.executable, *sys.argv]
            logging.info("Restart requested via Telegram command. Re-execing: %s", argv)
            os.execv(sys.executable, argv)
        except OSError:
            logging.exception("Failed to re-exec process after /restart")
            return 1


def main(args: list[str] | None = None) -> int:
    """Run the main program.

    Dispatches to the appropriate sub-command when one is given (e.g.
    `register-commands`), otherwise runs the Telegram bot polling loop.
    Sub-commands are registered via argparse subparsers and advertised in
    `--help`.
    """
    load_dotenv(override=False)
    parser = get_parser()
    argv = list(args) if args is not None else sys.argv[1:]
    opts = parser.parse_args(args=argv)

    # Sub-command dispatch: each sub-command sets opts.func via set_defaults.
    if hasattr(opts, "func"):
        return opts.func(opts)

    log_level = os.getenv("ACP_LOG_LEVEL", "INFO").upper()
    configure_logging(
        level=getattr(logging, log_level, logging.INFO),
        log_format=opts.log_format,
        close_replaced_handlers=True,
    )

    if not opts.telegram_token:
        parser.error("--telegram-token (or TELEGRAM_BOT_TOKEN) is required")
    if not opts.agent_command:
        parser.error("--agent-command (or ACP_AGENT_COMMAND) is required")
    if opts.acp_stdio_limit <= 0:
        parser.error("--acp-stdio-limit must be a positive integer")
    if opts.acp_connect_timeout <= 0:
        parser.error("--acp-connect-timeout must be a positive number")
    allowed_user_ids, allowed_usernames = _resolve_allowed_users(parser=parser, opts=opts)

    command_parts = shlex.split(opts.agent_command)
    if not command_parts:
        parser.error("--agent-command is empty after parsing")
    restart_command_parts = shlex.split(opts.restart_command) if opts.restart_command.strip() else None
    channel_state_file = default_state_file(pid=os.getpid())
    mcp_servers = _default_mcp_servers(telegram_token=opts.telegram_token, state_file=channel_state_file)

    config = make_config(
        token=opts.telegram_token,
        allowed_user_ids=allowed_user_ids,
        allowed_usernames=allowed_usernames,
        workspace=opts.workspace,
    )
    service = AcpAgentService(
        SessionRegistry(),
        program=command_parts[0],
        args=command_parts[1:],
        default_permission_mode=cast(PermissionMode, opts.permission_mode),
        permission_event_output=cast(PermissionEventOutput, opts.permission_event_output),
        mcp_servers=mcp_servers,
        channel_state_file=channel_state_file,
        stdio_limit=opts.acp_stdio_limit,
        connect_timeout=opts.acp_connect_timeout,
    )
    bridge = TelegramBridge(config=config, agent_service=service)
    return _run_bot_loop(config, bridge, restart_command_parts)


__all__: list[str] = ["get_parser", "main"]

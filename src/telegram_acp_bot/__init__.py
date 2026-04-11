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
from typing import Any, cast

from acp.schema import EnvVariable, McpServerStdio
from dotenv import load_dotenv

from telegram_acp_bot.acp.models import ActivityMode, PermissionEventOutput, PermissionMode
from telegram_acp_bot.acp.service import AcpAgentService
from telegram_acp_bot.config_file import ConfigFileError, load_config_file
from telegram_acp_bot.core.session_registry import SessionRegistry
from telegram_acp_bot.logging_context import configure_logging
from telegram_acp_bot.mcp.state import STATE_FILE_ENV, TOKEN_ENV, default_state_file
from telegram_acp_bot.register_commands import add_register_commands_subparser
from telegram_acp_bot.scheduled_tasks import (
    ACP_SCHEDULED_TASKS_DB_ENV,
    ScheduledTaskRunner,
    ScheduledTaskScheduler,
    ScheduledTaskStore,
    default_scheduled_tasks_db_path,
)
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

    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON config file. Values are overridden by environment variables"
            " and CLI flags. See docs for the full schema."
        ),
    )
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
    parser.add_argument(
        "-m",
        "--activity-mode",
        default=os.getenv("ACP_ACTIVITY_MODE", "normal"),
        choices=["compact", "normal", "verbose"],
        help=(
            "Activity display mode. "
            "'normal' (default) emits each activity event as its own message. "
            "'compact' collapses all intermediate events into a single in-place status message "
            "that is replaced by the final answer when the agent responds. "
            "'verbose' streams append-only updates for active reply and tool activity."
        ),
    )
    parser.add_argument(
        "--scheduled-tasks-db",
        default=os.getenv(ACP_SCHEDULED_TASKS_DB_ENV, str(default_scheduled_tasks_db_path())),
        help="SQLite database path used for deferred scheduled follow-ups.",
    )
    return parser


def _default_mcp_servers(
    *,
    telegram_token: str,
    state_file: Path,
    scheduled_tasks_db: Path,
) -> tuple[McpServerStdio, ...]:
    """Return MCP servers that should always be exposed to the ACP agent."""

    return (
        McpServerStdio(
            name="telegram-channel",
            command=sys.executable,
            args=["-m", "telegram_acp_bot.mcp.server"],
            env=[
                EnvVariable(name=TOKEN_ENV, value=telegram_token),
                EnvVariable(name=STATE_FILE_ENV, value=str(state_file)),
                EnvVariable(name=ACP_SCHEDULED_TASKS_DB_ENV, value=str(scheduled_tasks_db)),
            ],
        ),
    )


def _apply_config_file_defaults(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
) -> None:
    """Apply config file values as parser-level defaults.

    Only takes effect when the corresponding environment variable is not set.
    CLI flags always take precedence over these defaults.
    """
    tg = config.get("telegram", {})
    acp_cfg = config.get("acp", {})
    updates: dict[str, object] = {}

    # String settings: env var > config file > hardcoded default in add_argument
    _str_map: list[tuple[str, str, Any]] = [
        ("telegram_token", "TELEGRAM_BOT_TOKEN", tg.get("bot_token")),
        ("agent_command", "ACP_AGENT_COMMAND", acp_cfg.get("agent_command")),
        ("restart_command", "ACP_RESTART_COMMAND", acp_cfg.get("restart_command")),
        ("permission_mode", "ACP_PERMISSION_MODE", acp_cfg.get("permission_mode")),
        ("permission_event_output", "ACP_PERMISSION_EVENT_OUTPUT", acp_cfg.get("permission_event_output")),
        ("log_format", "ACP_LOG_FORMAT", acp_cfg.get("log_format")),
        ("activity_mode", "ACP_ACTIVITY_MODE", acp_cfg.get("activity_mode")),
        ("scheduled_tasks_db", ACP_SCHEDULED_TASKS_DB_ENV, acp_cfg.get("scheduled_tasks_db")),
    ]
    updates |= {
        dest: cfg_val for dest, env_name, cfg_val in _str_map if not os.getenv(env_name) and cfg_val is not None
    }

    # workspace has no env var
    if "workspace" in acp_cfg:
        updates["workspace"] = acp_cfg["workspace"]

    # Numeric settings
    if not os.getenv("ACP_STDIO_LIMIT") and "stdio_limit" in acp_cfg:
        updates["acp_stdio_limit"] = int(acp_cfg["stdio_limit"])
    if not os.getenv("ACP_CONNECT_TIMEOUT") and "connect_timeout" in acp_cfg:
        updates["acp_connect_timeout"] = float(acp_cfg["connect_timeout"])

    if updates:
        _set_parser_defaults(parser, updates)


def _set_parser_defaults(parser: argparse.ArgumentParser, updates: dict[str, object]) -> None:
    """Apply defaults to *parser* and any subparsers that expose the same dests."""
    parser_dests = {action.dest for action in parser._actions}
    parser_updates = {dest: value for dest, value in updates.items() if dest in parser_dests}
    if parser_updates:
        parser.set_defaults(**parser_updates)

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                _set_parser_defaults(subparser, updates)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_username(username: str) -> str:
    return username.lstrip("@").strip().lower()


def _resolve_allowed_users(
    *,
    parser: argparse.ArgumentParser,
    opts: argparse.Namespace,
    config_data: dict[str, Any] | None = None,
) -> tuple[list[int], list[str]]:
    try:
        env_allowed_ids = [int(item) for item in _parse_csv(os.getenv(ALLOWED_USER_IDS_ENV, ""))]
    except ValueError:
        parser.error(f"{ALLOWED_USER_IDS_ENV} must be a comma-separated list of integers")
    env_allowed_usernames = [_normalize_username(item) for item in _parse_csv(os.getenv(ALLOWED_USERNAMES_ENV, ""))]
    cli_allowed_usernames = [_normalize_username(item) for item in opts.allowed_username]
    allowed_user_ids = [*env_allowed_ids, *opts.allowed_user_id]
    allowed_usernames = [*env_allowed_usernames, *cli_allowed_usernames]

    # Config file: used as a fallback when neither env vars nor CLI provided any allowed users
    if not allowed_user_ids and not allowed_usernames and config_data:
        tg = config_data.get("telegram", {})
        allowed_user_ids = list(tg.get("allowed_user_ids", []))
        allowed_usernames = [_normalize_username(u) for u in tg.get("allowed_usernames", [])]

    if not allowed_user_ids and not allowed_usernames:
        parser.error(
            "--allowed-user-id/--allowed-username (or TELEGRAM_ALLOWED_USER_IDS/TELEGRAM_ALLOWED_USERNAMES) "
            "must include at least one allowed user",
        )
    return allowed_user_ids, allowed_usernames


def _run_bot_loop(
    config: BotConfig,
    bridge: TelegramBridge,
    scheduler: ScheduledTaskScheduler | None,
    restart_command_parts: list[str] | None,
) -> int:
    """Poll until normal exit; re-exec the process on restart requests."""
    while True:
        exit_code = run_polling(config, bridge, scheduler=scheduler)
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

    # Pre-parse to extract --config before full argument resolution
    pre_opts, _ = parser.parse_known_args(args=argv)

    config_data: dict[str, Any] = {}
    if pre_opts.config:
        config_path = Path(pre_opts.config).expanduser()
        try:
            config_data = load_config_file(config_path)
        except ConfigFileError as exc:
            parser.error(str(exc))
        _apply_config_file_defaults(parser, config_data)

    # Full parse: catches unknown/invalid flags for both the main command and subcommands.
    opts = parser.parse_args(args=argv)

    # Sub-command dispatch: each sub-command sets opts.func via set_defaults.
    if hasattr(opts, "func"):
        return opts.func(opts)

    log_level = os.getenv("ACP_LOG_LEVEL") or config_data.get("acp", {}).get("log_level", "INFO")
    configure_logging(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
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
    allowed_user_ids, allowed_usernames = _resolve_allowed_users(parser=parser, opts=opts, config_data=config_data)

    command_parts = shlex.split(opts.agent_command)
    if not command_parts:
        parser.error("--agent-command is empty after parsing")
    restart_command_parts = shlex.split(opts.restart_command) if opts.restart_command.strip() else None
    channel_state_file = default_state_file(pid=os.getpid())
    scheduled_tasks_db = Path(opts.scheduled_tasks_db).expanduser()
    mcp_servers = _default_mcp_servers(
        telegram_token=opts.telegram_token,
        state_file=channel_state_file,
        scheduled_tasks_db=scheduled_tasks_db,
    )

    config = make_config(
        token=opts.telegram_token,
        allowed_user_ids=allowed_user_ids,
        allowed_usernames=allowed_usernames,
        workspace=opts.workspace,
        activity_mode=cast(ActivityMode, opts.activity_mode),
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
    scheduled_task_store = ScheduledTaskStore(scheduled_tasks_db)
    bridge = TelegramBridge(config=config, agent_service=service, scheduled_task_store=scheduled_task_store)
    scheduler = ScheduledTaskScheduler(
        store=scheduled_task_store,
        runner=ScheduledTaskRunner(bridge.execute_scheduled_task),
    )
    return _run_bot_loop(config, bridge, scheduler, restart_command_parts)


__all__: list[str] = ["get_parser", "main"]

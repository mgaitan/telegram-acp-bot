"""
Telegram ACP bot

A Telegram bot that implements Agent Client Protocol to interact with AI agents.
"""

from __future__ import annotations

import argparse
import os
import shlex
from importlib import metadata

from dotenv import load_dotenv

from telegram_acp_bot.acp_app.acp_service import AcpAgentService
from telegram_acp_bot.core.session_registry import SessionRegistry
from telegram_acp_bot.telegram.bot import TelegramBridge, make_config, run_polling


def get_version() -> str:
    try:
        return metadata.version("telegram-acp-bot")
    except metadata.PackageNotFoundError:
        return "unknown"


def get_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="acp-bot")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {get_version()}")
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
        "--workspace",
        default=os.getcwd(),
        help="Default workspace path for /new when path is not provided.",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    """Run the main program."""
    load_dotenv(override=False)
    parser = get_parser()
    opts = parser.parse_args(args=args)

    if not opts.telegram_token:
        parser.error("--telegram-token (or TELEGRAM_BOT_TOKEN) is required")
    if not opts.agent_command:
        parser.error("--agent-command (or ACP_AGENT_COMMAND) is required")

    command_parts = shlex.split(opts.agent_command)
    if not command_parts:
        parser.error("--agent-command is empty after parsing")

    config = make_config(
        token=opts.telegram_token,
        allowed_user_ids=opts.allowed_user_id,
        workspace=opts.workspace,
    )
    service = AcpAgentService(
        SessionRegistry(),
        program=command_parts[0],
        args=command_parts[1:],
    )
    bridge = TelegramBridge(config=config, agent_service=service)
    return run_polling(config, bridge)


__all__: list[str] = ["get_parser", "main"]

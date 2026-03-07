from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from scripted_demo_service import ScriptedDemoAgentService

from telegram_acp_bot.core.session_registry import SessionRegistry
from telegram_acp_bot.telegram.bot import TelegramBridge, make_config, run_polling

TOKEN_REQUIRED_MESSAGE = "--telegram-token (or TELEGRAM_BOT_TOKEN) is required"
ALLOWLIST_REQUIRED_MESSAGE = "Provide at least one allowlist entry via --allowed-user-id or --allowed-username"
INVALID_ALLOWED_IDS_MESSAGE = "TELEGRAM_ALLOWED_USER_IDS must be a comma-separated list of integers"


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_allowed_user_ids(value: str) -> list[int]:
    try:
        return [int(item) for item in _parse_csv(value)]
    except ValueError as exc:
        raise SystemExit(INVALID_ALLOWED_IDS_MESSAGE) from exc


def _normalize_usernames(values: list[str]) -> list[str]:
    return [value.strip() for value in values if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run-echo-bot",
        description="Run telegram-acp-bot backed by ScriptedDemoAgentService for demo recording.",
    )
    parser.add_argument(
        "--telegram-token",
        default=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        help="Telegram bot token. Defaults to TELEGRAM_BOT_TOKEN.",
    )
    parser.add_argument(
        "--allowed-user-id",
        action="append",
        default=_parse_allowed_user_ids(os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")),
        type=int,
        help="Allowed Telegram user ID. Can be repeated.",
    )
    parser.add_argument(
        "--allowed-username",
        action="append",
        default=_parse_csv(os.getenv("TELEGRAM_ALLOWED_USERNAMES", "")),
        help="Allowed Telegram username (with or without @). Can be repeated.",
    )
    parser.add_argument(
        "--workspace",
        default=str(Path.cwd()),
        help="Default workspace path for /new when path is not provided.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("ACP_LOG_LEVEL", "INFO"),
        help="Log level for demo bot process.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(override=False)
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    token: str = args.telegram_token
    allowed_user_ids: list[int] = list(args.allowed_user_id)
    allowed_usernames: list[str] = _normalize_usernames(list(args.allowed_username))

    if not token:
        raise SystemExit(TOKEN_REQUIRED_MESSAGE)
    if not allowed_user_ids and not allowed_usernames:
        raise SystemExit(ALLOWLIST_REQUIRED_MESSAGE)

    config = make_config(
        token=token,
        allowed_user_ids=allowed_user_ids,
        allowed_usernames=allowed_usernames,
        workspace=args.workspace,
    )
    bridge = TelegramBridge(config=config, agent_service=ScriptedDemoAgentService(SessionRegistry()))
    return run_polling(config, bridge)


if __name__ == "__main__":
    raise SystemExit(main())

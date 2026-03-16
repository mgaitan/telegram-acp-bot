"""CLI sub-command to register Telegram slash commands via the Bot API.

Uses `setMyCommands` (or `deleteMyCommands` when `--delete` is given) to keep
the bot's registered commands in sync with {py:data}`telegram_acp_bot.telegram.bot.BOT_COMMANDS`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from telegram import (
    Bot,
    BotCommandScopeAllChatAdministrators,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeDefault,
)
from telegram.error import TelegramError

from telegram_acp_bot.telegram.bot import BOT_COMMANDS

_SCOPE_CLASSES = {
    "default": BotCommandScopeDefault,
    "all_private_chats": BotCommandScopeAllPrivateChats,
    "all_group_chats": BotCommandScopeAllGroupChats,
    "all_chat_administrators": BotCommandScopeAllChatAdministrators,
}

SCOPE_CHOICES: list[str] = list(_SCOPE_CLASSES)

_SUBCOMMAND_NAME = "register-commands"
_SUBCOMMAND_HELP = "Register Telegram slash commands via the Bot API (setMyCommands)."


def _populate_register_commands_args(parser: argparse.ArgumentParser) -> None:
    """Add `register-commands` arguments to *parser* (shared by standalone and subparser)."""
    parser.add_argument(
        "--telegram-token",
        default=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        help="Telegram bot token. Default: TELEGRAM_BOT_TOKEN env var.",
    )
    parser.add_argument(
        "--scope",
        default="default",
        choices=SCOPE_CHOICES,
        help="BotCommandScope to register commands for (default: default).",
    )
    parser.add_argument(
        "--language-code",
        default="",
        help="IETF language code for language-specific registration (e.g. 'en').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be registered without calling the API.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete registered commands instead of setting them (deleteMyCommands).",
    )


def add_register_commands_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Add `register-commands` as a subparser to an existing subparsers group.

    Sets `func` on the resulting namespace so callers can dispatch via
    `opts.func(opts)`.  See also `{py:func}`get_register_commands_parser`` for
    the standalone parser used by `{py:func}`register_commands_main``.
    """
    rc: argparse.ArgumentParser = subparsers.add_parser(
        _SUBCOMMAND_NAME,
        help=_SUBCOMMAND_HELP,
        description=_SUBCOMMAND_HELP,
    )
    _populate_register_commands_args(rc)
    rc.set_defaults(func=_execute_register_commands)
    return rc


def get_register_commands_parser() -> argparse.ArgumentParser:
    """Return a standalone argument parser for the `register-commands` sub-command."""
    parser = argparse.ArgumentParser(
        prog=f"telegram-acp-bot {_SUBCOMMAND_NAME}",
        description=_SUBCOMMAND_HELP,
    )
    _populate_register_commands_args(parser)
    return parser


async def _call_api(
    *,
    token: str,
    scope_str: str,
    language_code: str | None,
    delete: bool,
) -> None:
    """Invoke `setMyCommands` or `deleteMyCommands` on the Telegram Bot API."""
    bot_scope = _SCOPE_CLASSES[scope_str]()
    async with Bot(token=token) as bot:
        if delete:
            await bot.delete_my_commands(scope=bot_scope, language_code=language_code)
        else:
            await bot.set_my_commands(list(BOT_COMMANDS), scope=bot_scope, language_code=language_code)


def _execute_register_commands(opts: argparse.Namespace) -> int:
    """Execute the `register-commands` logic from a pre-parsed namespace.

    Validates the token, then either prints a dry-run summary or calls the
    Telegram Bot API.  Called both by `{py:func}`register_commands_main`` and
    by the argparse subparser dispatch in `{py:func}`telegram_acp_bot.main``.

    Returns:
        0 on success, 1 on API error, exits with code 2 on missing token.
    """
    if not opts.telegram_token:
        print(
            f"telegram-acp-bot {_SUBCOMMAND_NAME}: error: --telegram-token (or TELEGRAM_BOT_TOKEN) is required",
            file=sys.stderr,
        )
        raise SystemExit(2)

    language_code: str | None = opts.language_code.strip() or None

    if opts.dry_run:
        if opts.delete:
            print(f"[dry-run] Would delete commands: scope={opts.scope!r}, language_code={language_code!r}")
        else:
            print(
                f"[dry-run] Would register {len(BOT_COMMANDS)} command(s): "
                f"scope={opts.scope!r}, language_code={language_code!r}"
            )
            for cmd, desc in BOT_COMMANDS:
                print(f"  /{cmd} - {desc}")
        return 0

    try:
        asyncio.run(
            _call_api(
                token=opts.telegram_token,
                scope_str=opts.scope,
                language_code=language_code,
                delete=opts.delete,
            )
        )
    except TelegramError as exc:
        print(f"Telegram API error: {exc}", file=sys.stderr)
        return 1

    if opts.delete:
        print(f"Commands deleted: scope={opts.scope!r}, language_code={language_code!r}")
    else:
        print(f"Registered {len(BOT_COMMANDS)} command(s): scope={opts.scope!r}, language_code={language_code!r}")
    return 0


def register_commands_main(args: list[str] | None = None) -> int:
    """Run the `register-commands` sub-command standalone.

    Loads `.env`, parses *args* with `{py:func}`get_register_commands_parser``,
    and delegates to `{py:func}`_execute_register_commands``.

    Returns:
        0 on success, 1 on API error.
    """
    load_dotenv(override=False)
    parser = get_register_commands_parser()
    opts = parser.parse_args(args=args)
    return _execute_register_commands(opts)


__all__: list[str] = [
    "SCOPE_CHOICES",
    "add_register_commands_subparser",
    "get_register_commands_parser",
    "register_commands_main",
]

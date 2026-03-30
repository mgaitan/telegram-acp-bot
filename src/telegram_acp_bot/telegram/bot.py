"""Compatibility re-exports for `telegram_acp_bot.telegram.bot`.

The Telegram transport has been split into focused submodules.  This module
re-exports every public and semi-public name so that existing import paths
remain valid.

See also:
- `{py:mod}telegram_acp_bot.telegram.constants` — callback prefixes, labels, limits
- `{py:mod}telegram_acp_bot.telegram.config` — `BotConfig`, `make_config`
- `{py:mod}telegram_acp_bot.telegram.models` — internal data models and `AgentService` protocol
- `{py:mod}telegram_acp_bot.telegram.activity` — activity-mode strategy handlers
- `{py:mod}telegram_acp_bot.telegram.bridge` — `TelegramBridge`
- `{py:mod}telegram_acp_bot.telegram.app` — `build_application`, `run_polling`
"""

from __future__ import annotations

# Re-export third-party names that tests access via `bot_module.X`.
from telegram import Bot  # noqa: F401
from telegramify_markdown import MessageEntity as MarkdownMessageEntity  # noqa: F401
from telegramify_markdown import utf16_len  # noqa: F401

from telegram_acp_bot.telegram.activity import (  # noqa: F401
    VERBOSE_STREAM_TICK_SECONDS,
    _ActivityModeHandler,
    _CompactActivityModeHandler,
    _NormalActivityModeHandler,
    _VerboseActivityModeHandler,
)
from telegram_acp_bot.telegram.app import build_application, run_polling  # noqa: F401
from telegram_acp_bot.telegram.bridge import TelegramBridge  # noqa: F401
from telegram_acp_bot.telegram.config import BotConfig, make_config  # noqa: F401
from telegram_acp_bot.telegram.constants import (  # noqa: F401
    ACTIVITY_MODE_CHOICES,
    ACTIVITY_MODE_HELP,
    BOT_COMMANDS,
    BUSY_CALLBACK_PARTS,
    BUSY_CALLBACK_PREFIX,
    BUSY_QUEUED_TEXT,
    BUSY_SENT_TEXT,
    BUSY_STILL_QUEUED_TEXT,
    KIND_LABELS,
    MAX_RESTART_ARGS,
    MAX_RESUME_ARGS,
    PERMISSION_CALLBACK_PARTS,
    PERMISSION_CALLBACK_PREFIX,
    REPLY_LABEL,
    RESTART_EXIT_CODE,
    RESUME_CALLBACK_PARTS,
    RESUME_CALLBACK_PREFIX,
    RESUME_KEYBOARD_MAX_ROWS,
    SEARCH_LABEL_NEUTRAL,
    SEARCH_LABEL_WEB,
    TELEGRAM_MAX_UTF16_MESSAGE_LENGTH,
)
from telegram_acp_bot.telegram.models import (  # noqa: F401
    AgentService,
    ChatRequiredError,
    _PendingPrompt,
    _PromptInput,
    _QueuedVerboseBlock,
    _RestartArgs,
    _ResumeArgs,
    _VerboseActivityMessage,
)

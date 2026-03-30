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

# Standard-library re-exports (were at module scope in the original bot.py).
import asyncio  # noqa: F401
import base64  # noqa: F401
import logging  # noqa: F401
from collections.abc import Awaitable, Callable  # noqa: F401
from contextlib import suppress  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
from io import BytesIO  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Protocol, cast  # noqa: F401
from urllib.parse import urlparse  # noqa: F401
from uuid import uuid4  # noqa: F401

# Third-party re-exports.
from telegram import (  # noqa: F401
    Bot,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    Message,
    MessageEntity,
    Update,
)
from telegram.constants import ChatAction, ParseMode  # noqa: F401
from telegram.error import TelegramError  # noqa: F401
from telegram.ext import (  # noqa: F401
    AIORateLimiter,
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegramify_markdown import MessageEntity as MarkdownMessageEntity  # noqa: F401
from telegramify_markdown import convert, split_entities, utf16_len  # noqa: F401

# Internal re-exports from acp_app and logging_context.
from telegram_acp_bot.acp_app.models import (  # noqa: F401
    ActivityMode,
    AgentActivityBlock,
    AgentOutputLimitExceededError,
    AgentReply,
    FilePayload,
    ImagePayload,
    PermissionDecisionAction,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
    PromptFile,
    PromptImage,
    ResumableSession,
)
from telegram_acp_bot.logging_context import bind_log_context, log_text_preview  # noqa: F401
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

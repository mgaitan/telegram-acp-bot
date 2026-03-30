"""Application factory and polling entrypoint for the Telegram transport."""

from __future__ import annotations

from telegram import Update
from telegram.ext import AIORateLimiter, Application

from telegram_acp_bot.telegram.bridge import TelegramBridge
from telegram_acp_bot.telegram.config import BotConfig
from telegram_acp_bot.telegram.constants import RESTART_EXIT_CODE


def build_application(config: BotConfig, bridge: TelegramBridge) -> Application:
    # Permission prompts are awaited inside message handlers, so callback queries
    # must be processed concurrently to avoid deadlocking the update loop.
    app = Application.builder().token(config.token).rate_limiter(AIORateLimiter()).concurrent_updates(True).build()
    bridge.install(app)
    return app


def run_polling(config: BotConfig, bridge: TelegramBridge) -> int:
    app = build_application(config, bridge)
    app.run_polling(allowed_updates=Update.ALL_TYPES)
    if bridge._restart_requested:
        return RESTART_EXIT_CODE
    return 0

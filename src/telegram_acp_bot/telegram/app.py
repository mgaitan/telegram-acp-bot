"""Application factory and polling entrypoint for the Telegram transport."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from telegram import Update
from telegram.ext import AIORateLimiter, Application

from telegram_acp_bot.scheduled_tasks.scheduler import ScheduledTaskScheduler
from telegram_acp_bot.telegram.bridge import TelegramBridge
from telegram_acp_bot.telegram.config import BotConfig
from telegram_acp_bot.telegram.constants import RESTART_EXIT_CODE


def build_application(
    config: BotConfig,
    bridge: TelegramBridge,
    scheduler: ScheduledTaskScheduler | None = None,
) -> Application:
    # Permission prompts are awaited inside message handlers, so callback queries
    # must be processed concurrently to avoid deadlocking the update loop.
    builder = Application.builder().token(config.token).rate_limiter(AIORateLimiter()).concurrent_updates(True)
    if scheduler is not None:
        builder = builder.post_init(_post_init_factory(scheduler)).post_shutdown(_post_shutdown_factory(scheduler))
    app = builder.build()
    bridge.install(app)
    return app


def run_polling(
    config: BotConfig,
    bridge: TelegramBridge,
    scheduler: ScheduledTaskScheduler | None = None,
) -> int:
    if scheduler is None:
        app = build_application(config, bridge)
    else:
        app = build_application(config, bridge, scheduler=scheduler)
    app.run_polling(allowed_updates=Update.ALL_TYPES)
    if bridge._restart_requested:
        return RESTART_EXIT_CODE
    return 0


def _post_init_factory(scheduler: ScheduledTaskScheduler) -> Callable[[Application], Coroutine[Any, Any, None]]:
    async def _post_init(_: Application) -> None:
        await scheduler.start()

    return _post_init


def _post_shutdown_factory(scheduler: ScheduledTaskScheduler) -> Callable[[Application], Coroutine[Any, Any, None]]:
    async def _post_shutdown(_: Application) -> None:
        await scheduler.stop()

    return _post_shutdown

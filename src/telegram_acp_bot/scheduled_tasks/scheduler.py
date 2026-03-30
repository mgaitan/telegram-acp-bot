"""Async scheduler loop for deferred follow-up tasks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress

from telegram_acp_bot.scheduled_tasks.models import (
    ScheduledTask,
    ScheduledTaskDeferredError,
    ScheduledTaskExecutionError,
)
from telegram_acp_bot.scheduled_tasks.store import ScheduledTaskStore, utc_now

logger = logging.getLogger(__name__)

ScheduledTaskExecutor = Callable[[ScheduledTask], Awaitable[None]]


class ScheduledTaskRunner:
    """Bridge between claimed tasks and the Telegram execution layer."""

    def __init__(self, executor: ScheduledTaskExecutor) -> None:
        self._executor = executor

    async def run(self, task: ScheduledTask) -> None:
        """Execute one claimed task."""

        await self._executor(task)


class ScheduledTaskScheduler:
    """Poll the store for due tasks and dispatch them asynchronously."""

    def __init__(
        self,
        *,
        store: ScheduledTaskStore,
        runner: ScheduledTaskRunner,
        poll_interval_seconds: float = 15.0,
        claim_limit: int = 10,
    ) -> None:
        self._store = store
        self._runner = runner
        self._poll_interval_seconds = poll_interval_seconds
        self._claim_limit = claim_limit
        self._loop_task: asyncio.Task[None] | None = None
        self._active_runs: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        """Initialize storage and start the scheduler loop if needed."""

        await asyncio.to_thread(self._store.initialize)
        recovered = await asyncio.to_thread(self._store.recover_running_tasks)
        if recovered:
            logger.info("Recovered %s scheduled task(s) left in running state", recovered)
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the scheduler loop and cancel in-flight task wrappers."""

        loop_task = self._loop_task
        self._loop_task = None
        if loop_task is not None:
            loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await loop_task
        active_runs = tuple(self._active_runs)
        for task in active_runs:
            task.cancel()
        if active_runs:
            await asyncio.gather(*active_runs, return_exceptions=True)

    async def _run_loop(self) -> None:
        while True:
            claimed = await asyncio.to_thread(
                self._store.claim_due_tasks,
                now=utc_now(),
                limit=self._claim_limit,
            )
            for task in claimed:
                active = asyncio.create_task(self._run_claimed_task(task))
                self._active_runs.add(active)
                active.add_done_callback(self._active_runs.discard)
            await asyncio.sleep(self._poll_interval_seconds)

    async def _run_claimed_task(self, task: ScheduledTask) -> None:
        try:
            await self._runner.run(task)
        except ScheduledTaskDeferredError:
            await asyncio.to_thread(self._store.release_task, task.id)
        except ScheduledTaskExecutionError as exc:
            await asyncio.to_thread(self._store.mark_failed, task.id, error=str(exc))
        except Exception as exc:
            logger.exception("Unhandled error while running scheduled task %s", task.id)
            await asyncio.to_thread(self._store.mark_failed, task.id, error=str(exc))
        else:
            await asyncio.to_thread(self._store.mark_done, task.id)

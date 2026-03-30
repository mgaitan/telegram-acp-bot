"""Domain models for deferred Telegram follow-ups."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

ACP_SCHEDULED_TASKS_DB_ENV = "ACP_SCHEDULED_TASKS_DB"

ScheduledTaskMode = Literal["notify", "prompt_agent"]
ScheduledTaskStatus = Literal["pending", "running", "done", "failed", "cancelled"]


@dataclass(slots=True, frozen=True)
class ScheduledTask:
    """Persisted deferred task for one Telegram chat."""

    id: str
    chat_id: int
    anchor_message_id: int
    mode: ScheduledTaskMode
    prompt_text: str | None
    notify_text: str | None
    run_at: datetime
    status: ScheduledTaskStatus
    attempt_count: int
    last_error: str | None
    claimed_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ScheduledTaskDeferredError(RuntimeError):
    """Raised when a due task should stay pending and retry later."""


class ScheduledTaskExecutionError(RuntimeError):
    """Raised when a due task cannot be completed successfully."""

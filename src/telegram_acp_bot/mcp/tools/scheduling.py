"""Deferred follow-up scheduling tool for the internal Telegram MCP server."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from mcp.server.fastmcp import FastMCP

from telegram_acp_bot.mcp.context import resolve_request_context
from telegram_acp_bot.scheduled_tasks import ACP_SCHEDULED_TASKS_DB_ENV, ScheduledTaskMode, ScheduledTaskStore
from telegram_acp_bot.scheduled_tasks.store import parse_utc_timestamp

MISSING_SCHEDULED_DB_ERROR = f"missing {ACP_SCHEDULED_TASKS_DB_ENV}"
RUN_AT_OR_DELAY_ERROR = "provide run_at or at least one delay input"
MIXED_RUN_AT_AND_DELAY_ERROR = "provide either run_at or delay inputs, not both"
NEGATIVE_DELAY_ERROR = "delay inputs must be zero or positive"


def register_scheduling_tools(mcp: FastMCP) -> None:
    """Register scheduling tools on the provided MCP server."""

    mcp.tool(
        name="schedule_task",
        description=(
            "Schedule a one-shot deferred follow-up for the current Telegram chat. "
            "Prefer relative delay inputs for requests such as 'in 30 seconds' or 'in 10 minutes'."
        ),
    )(schedule_task)


async def schedule_task(  # noqa: PLR0911,PLR0913
    mode: str,
    run_at: str | None = None,
    delay_seconds: int | None = None,
    delay_minutes: int | None = None,
    delay_hours: int | None = None,
    prompt_text: str | None = None,
    notify_text: str | None = None,
    session_id: str | None = None,
) -> dict[str, object]:
    def fail(error: str) -> dict[str, object]:
        return {"ok": False, "error": error}

    context = resolve_request_context(session_id=session_id)
    if isinstance(context, str):
        return fail(context)

    if mode not in {"notify", "prompt_agent"}:
        return fail("mode must be one of: notify, prompt_agent")
    if mode == "notify" and not (notify_text or "").strip():
        return fail("notify mode requires notify_text")
    if mode == "prompt_agent" and not (prompt_text or "").strip():
        return fail("prompt_agent mode requires prompt_text")

    try:
        run_at_dt = resolve_run_at(
            run_at=run_at,
            delay_seconds=delay_seconds,
            delay_minutes=delay_minutes,
            delay_hours=delay_hours,
        )
    except ValueError as exc:
        return fail(str(exc))

    try:
        store = load_scheduled_task_store()
    except RuntimeError as exc:
        return fail(str(exc))
    task = store.create_task(
        chat_id=context.chat_id,
        session_id=context.session_id,
        mode=cast(ScheduledTaskMode, mode),
        prompt_text=prompt_text,
        notify_text=notify_text,
        run_at=run_at_dt,
    )
    return {
        "ok": True,
        "task_id": task.id,
        "chat_id": task.chat_id,
        "anchor_message_id": task.anchor_message_id,
        "run_at": task.run_at.isoformat(),
        "summary": format_scheduled_summary(run_at_dt),
    }


def load_scheduled_task_store() -> ScheduledTaskStore:
    """Load and initialize the scheduled-task store for MCP scheduling tools."""

    db_path_raw = os.getenv(ACP_SCHEDULED_TASKS_DB_ENV, "").strip()
    if not db_path_raw:
        raise RuntimeError(MISSING_SCHEDULED_DB_ERROR)
    store = ScheduledTaskStore(Path(db_path_raw).expanduser())
    store.initialize()
    return store


def format_scheduled_summary(run_at: datetime) -> str:
    """Render a short summary suitable for the agent's scheduling confirmation."""

    timestamp = run_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"Scheduled for {timestamp}."


def resolve_run_at(
    *,
    run_at: str | None,
    delay_seconds: int | None,
    delay_minutes: int | None,
    delay_hours: int | None,
) -> datetime:
    """Resolve an execution time from an absolute timestamp or relative delay inputs."""

    delays = [value for value in (delay_seconds, delay_minutes, delay_hours) if value is not None]
    if run_at is not None and delays:
        raise ValueError(MIXED_RUN_AT_AND_DELAY_ERROR)
    if run_at is not None:
        return parse_utc_timestamp(run_at)
    if not delays:
        raise ValueError(RUN_AT_OR_DELAY_ERROR)
    if any(value < 0 for value in delays):
        raise ValueError(NEGATIVE_DELAY_ERROR)
    delta = timedelta(
        seconds=0 if delay_seconds is None else delay_seconds,
        minutes=0 if delay_minutes is None else delay_minutes,
        hours=0 if delay_hours is None else delay_hours,
    )
    return datetime.now(UTC) + delta

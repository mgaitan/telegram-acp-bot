"""SQLite persistence for deferred follow-up tasks."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from telegram_acp_bot.scheduled_tasks.models import ScheduledTask, ScheduledTaskMode, ScheduledTaskStatus

SQLITE_FILE_MODE = 0o600
TIMESTAMP_TIMEZONE_REQUIRED_ERROR = "timestamp must include an explicit timezone offset"
TIMESTAMP_AWARE_REQUIRED_ERROR = "timestamp must be timezone-aware"


def default_scheduled_tasks_db_path() -> Path:
    """Return the default persistent database path for scheduled tasks."""

    xdg_state_home = os.getenv("XDG_STATE_HOME", "").strip()
    state_root = Path(xdg_state_home).expanduser() if xdg_state_home else Path.home() / ".local" / "state"
    return state_root / "telegram-acp-bot" / "scheduled-tasks.sqlite3"


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def parse_utc_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to UTC."""

    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(TIMESTAMP_TIMEZONE_REQUIRED_ERROR)
    return parsed.astimezone(UTC)


def format_utc_timestamp(value: datetime) -> str:
    """Serialize a timezone-aware timestamp to an ISO UTC string."""

    if value.tzinfo is None:
        raise ValueError(TIMESTAMP_AWARE_REQUIRED_ERROR)
    return value.astimezone(UTC).isoformat()


class ScheduledTaskStore:
    """SQLite-backed source of truth for deferred follow-up tasks."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Return the database path."""

        return self._path

    def initialize(self) -> None:
        """Create the database schema if it does not exist yet."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    id TEXT PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    session_id TEXT,
                    anchor_message_id INTEGER,
                    mode TEXT NOT NULL,
                    prompt_text TEXT,
                    notify_text TEXT,
                    run_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    claimed_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS scheduled_tasks_due_idx
                ON scheduled_tasks (status, run_at)
                """
            )
        if self._path.exists():
            os.chmod(self._path, SQLITE_FILE_MODE)

    def create_task(  # noqa: PLR0913
        self,
        *,
        chat_id: int,
        session_id: str | None = None,
        anchor_message_id: int | None = None,
        mode: ScheduledTaskMode,
        run_at: datetime,
        prompt_text: str | None = None,
        notify_text: str | None = None,
    ) -> ScheduledTask:
        """Insert a new pending deferred task."""

        created_at = utc_now()
        task = ScheduledTask(
            id=uuid4().hex,
            chat_id=chat_id,
            session_id=session_id,
            anchor_message_id=anchor_message_id,
            mode=mode,
            prompt_text=prompt_text,
            notify_text=notify_text,
            run_at=run_at.astimezone(UTC),
            status="pending",
            attempt_count=0,
            last_error=None,
            claimed_at=None,
            started_at=None,
            finished_at=None,
            created_at=created_at,
            updated_at=created_at,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO scheduled_tasks (
                    id,
                    chat_id,
                    session_id,
                    anchor_message_id,
                    mode,
                    prompt_text,
                    notify_text,
                    run_at,
                    status,
                    attempt_count,
                    last_error,
                    claimed_at,
                    started_at,
                    finished_at,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._serialize_task(task),
            )
        return task

    def get_task(self, task_id: str) -> ScheduledTask | None:
        """Return one task by id."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scheduled_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        return None if row is None else self._row_to_task(row)

    def list_tasks_for_chat(
        self,
        *,
        chat_id: int,
        statuses: tuple[ScheduledTaskStatus, ...] = ("pending", "running"),
        limit: int = 10,
    ) -> list[ScheduledTask]:
        """Return scheduled tasks for one chat, ordered for user-facing inspection."""

        if not statuses or limit <= 0:
            return []

        placeholders = ", ".join("?" for _ in statuses)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM scheduled_tasks
                WHERE chat_id = ? AND status IN ({placeholders})
                ORDER BY
                    CASE status
                        WHEN 'pending' THEN 0
                        WHEN 'running' THEN 1
                        ELSE 2
                    END,
                    run_at,
                    created_at
                LIMIT ?
                """,
                (chat_id, *statuses, limit),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def cancel_task(self, *, chat_id: int, task_id: str) -> bool:
        """Cancel one pending task for the given chat."""

        now = format_utc_timestamp(utc_now())
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE scheduled_tasks
                SET status = 'cancelled',
                    finished_at = ?,
                    updated_at = ?,
                    last_error = NULL
                WHERE id = ?
                  AND chat_id = ?
                  AND status = 'pending'
                """,
                (now, now, task_id, chat_id),
            )
        return cursor.rowcount > 0

    def cancel_pending_tasks_for_chat(self, *, chat_id: int) -> int:
        """Cancel all pending tasks for the given chat."""

        now = format_utc_timestamp(utc_now())
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE scheduled_tasks
                SET status = 'cancelled',
                    finished_at = ?,
                    updated_at = ?,
                    last_error = NULL
                WHERE chat_id = ?
                  AND status = 'pending'
                """,
                (now, now, chat_id),
            )
        return cursor.rowcount

    def claim_due_tasks(self, *, now: datetime, limit: int = 10) -> list[ScheduledTask]:
        """Atomically move due pending tasks into `running` and return them."""

        claimed_at = format_utc_timestamp(now)
        updated_rows: list[ScheduledTask] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            due_rows = connection.execute(
                """
                SELECT id
                FROM scheduled_tasks
                WHERE status = 'pending' AND anchor_message_id IS NOT NULL AND run_at <= ?
                ORDER BY run_at, created_at
                LIMIT ?
                """,
                (format_utc_timestamp(now), limit),
            ).fetchall()
            if not due_rows:
                connection.commit()
                return []
            task_ids = [str(row["id"]) for row in due_rows]
            placeholders = ", ".join("?" for _ in task_ids)
            connection.execute(
                f"""
                UPDATE scheduled_tasks
                SET status = 'running',
                    attempt_count = attempt_count + 1,
                    claimed_at = ?,
                    started_at = ?,
                    updated_at = ?,
                    last_error = NULL
                WHERE id IN ({placeholders})
                """,
                (claimed_at, claimed_at, claimed_at, *task_ids),
            )
            claimed_rows = connection.execute(
                f"""
                SELECT *
                FROM scheduled_tasks
                WHERE id IN ({placeholders})
                ORDER BY run_at, created_at
                """,
                task_ids,
            ).fetchall()
            connection.commit()
        updated_rows.extend(self._row_to_task(row) for row in claimed_rows)
        return updated_rows

    def release_task(self, task_id: str) -> None:
        """Return a claimed task back to `pending`."""

        now = format_utc_timestamp(utc_now())
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE scheduled_tasks
                SET status = 'pending',
                    claimed_at = NULL,
                    started_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, task_id),
            )

    def mark_done(self, task_id: str) -> None:
        """Mark a task as completed."""

        now = format_utc_timestamp(utc_now())
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE scheduled_tasks
                SET status = 'done',
                    finished_at = ?,
                    updated_at = ?,
                    last_error = NULL
                WHERE id = ?
                """,
                (now, now, task_id),
            )

    def mark_failed(self, task_id: str, *, error: str) -> None:
        """Mark a task as failed and persist the last error message."""

        now = format_utc_timestamp(utc_now())
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE scheduled_tasks
                SET status = 'failed',
                    finished_at = ?,
                    updated_at = ?,
                    last_error = ?
                WHERE id = ?
                """,
                (now, now, error, task_id),
            )

    def recover_running_tasks(self) -> int:
        """Move stale `running` tasks back to `pending` after restart."""

        now = format_utc_timestamp(utc_now())
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE scheduled_tasks
                SET status = 'pending',
                    claimed_at = NULL,
                    started_at = NULL,
                    updated_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
        return cursor.rowcount

    def bind_unanchored_tasks(
        self,
        *,
        chat_id: int,
        session_id: str | None,
        anchor_message_id: int,
    ) -> int:
        """Attach pending unanchored tasks for the current chat/session to one anchor message."""

        now = format_utc_timestamp(utc_now())
        with self._connect() as connection:
            if session_id is None:
                cursor = connection.execute(
                    """
                    UPDATE scheduled_tasks
                    SET anchor_message_id = ?, updated_at = ?
                    WHERE chat_id = ?
                      AND status = 'pending'
                      AND anchor_message_id IS NULL
                      AND session_id IS NULL
                    """,
                    (anchor_message_id, now, chat_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE scheduled_tasks
                    SET anchor_message_id = ?, updated_at = ?
                    WHERE chat_id = ?
                      AND status = 'pending'
                      AND anchor_message_id IS NULL
                      AND session_id = ?
                    """,
                    (anchor_message_id, now, chat_id, session_id),
                )
        return cursor.rowcount

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _serialize_task(task: ScheduledTask) -> tuple[object, ...]:
        return (
            task.id,
            task.chat_id,
            task.session_id,
            task.anchor_message_id,
            task.mode,
            task.prompt_text,
            task.notify_text,
            format_utc_timestamp(task.run_at),
            task.status,
            task.attempt_count,
            task.last_error,
            None if task.claimed_at is None else format_utc_timestamp(task.claimed_at),
            None if task.started_at is None else format_utc_timestamp(task.started_at),
            None if task.finished_at is None else format_utc_timestamp(task.finished_at),
            format_utc_timestamp(task.created_at),
            format_utc_timestamp(task.updated_at),
        )

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> ScheduledTask:
        def load_timestamp(name: str) -> datetime | None:
            value = row[name]
            return None if value is None else parse_utc_timestamp(str(value))

        return ScheduledTask(
            id=str(row["id"]),
            chat_id=int(row["chat_id"]),
            session_id=None if row["session_id"] is None else str(row["session_id"]),
            anchor_message_id=None if row["anchor_message_id"] is None else int(row["anchor_message_id"]),
            mode=cast(ScheduledTaskMode, str(row["mode"])),
            prompt_text=None if row["prompt_text"] is None else str(row["prompt_text"]),
            notify_text=None if row["notify_text"] is None else str(row["notify_text"]),
            run_at=parse_utc_timestamp(str(row["run_at"])),
            status=cast(ScheduledTaskStatus, str(row["status"])),
            attempt_count=int(row["attempt_count"]),
            last_error=None if row["last_error"] is None else str(row["last_error"]),
            claimed_at=load_timestamp("claimed_at"),
            started_at=load_timestamp("started_at"),
            finished_at=load_timestamp("finished_at"),
            created_at=parse_utc_timestamp(str(row["created_at"])),
            updated_at=parse_utc_timestamp(str(row["updated_at"])),
        )

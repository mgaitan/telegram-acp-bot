from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from telegram import Update

from telegram_acp_bot.acp.models import (
    AgentActivityBlock,
    AgentOutputLimitExceededError,
    AgentReply,
    FilePayload,
    ImagePayload,
)
from telegram_acp_bot.scheduled_tasks import (
    ScheduledTask,
    ScheduledTaskDeferredError,
    ScheduledTaskExecutionError,
    ScheduledTaskRunner,
    ScheduledTaskScheduler,
    ScheduledTaskStore,
)
from telegram_acp_bot.scheduled_tasks.models import ScheduledTaskMode
from telegram_acp_bot.scheduled_tasks.store import default_scheduled_tasks_db_path, format_utc_timestamp
from telegram_acp_bot.telegram import app as app_module
from telegram_acp_bot.telegram import bot as bot_module
from telegram_acp_bot.telegram.app import build_application
from telegram_acp_bot.telegram.bot import AgentService, Application
from telegram_acp_bot.telegram.bridge import TelegramBridge
from telegram_acp_bot.telegram.config import make_config
from tests.telegram.support import TEST_CHAT_ID, DummyBot
from tests.telegram.support import DummyMessage as DummyBotMessage

pytestmark = pytest.mark.asyncio
ANCHOR_MESSAGE_ID = 42
BOUND_REPLY_MESSAGE_ID = 11
UNEXPECTED_EXECUTION_ERROR = "unexpected boom"


class ScheduledPromptService:
    def __init__(self, *, active: bool = True, reply_text: str = "Scheduled reply") -> None:
        self._active = active
        self._reply_text = reply_text
        self.prompt_calls: list[tuple[int, str]] = []

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
        del images, files
        self.prompt_calls.append((chat_id, text))
        return SimpleNamespace(text=self._reply_text, images=(), files=())

    def get_active_session_context(self, *, chat_id: int):
        if not self._active:
            return None
        return ("session-1", Path("/tmp/workspace"))

    def get_workspace(self, *, chat_id: int):
        del chat_id
        return Path("/tmp/workspace")


class FailingScheduledService(ScheduledPromptService):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
        del chat_id, text, images, files
        raise self._exc


class EmptyReplyScheduledService(ScheduledPromptService):
    async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
        del chat_id, text, images, files


class FakeScheduler:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class SlowAcquireLock:
    def locked(self) -> bool:
        return False

    async def acquire(self) -> None:
        await asyncio.Future()

    def release(self) -> None:
        return None


async def wait_for_task_status(
    store: ScheduledTaskStore,
    task_id: str,
    *,
    expected_status: str,
    timeout_seconds: float = 0.5,
) -> ScheduledTask:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        stored = store.get_task(task_id)
        if stored is not None and stored.status == expected_status:
            return stored
        if asyncio.get_running_loop().time() >= deadline:
            assert stored is not None
            pytest.fail(f"timed out waiting for task {task_id} to reach status {expected_status!r}")
        await asyncio.sleep(0.01)


def make_task(  # noqa: PLR0913
    *,
    task_id: str = "task-1",
    mode: ScheduledTaskMode = "notify",
    session_id: str | None = "session-1",
    prompt_text: str | None = None,
    notify_text: str | None = "Ping now",
    anchor_message_id: int | None = ANCHOR_MESSAGE_ID,
) -> ScheduledTask:
    now = datetime.now(UTC)
    return ScheduledTask(
        id=task_id,
        chat_id=TEST_CHAT_ID,
        session_id=session_id,
        anchor_message_id=anchor_message_id,
        mode=mode,
        prompt_text=prompt_text,
        notify_text=notify_text,
        run_at=now,
        status="pending",
        attempt_count=0,
        last_error=None,
        claimed_at=None,
        started_at=None,
        finished_at=None,
        created_at=now,
        updated_at=now,
    )


async def test_store_claims_due_tasks_in_order(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    now = datetime.now(UTC)
    first = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="first",
        run_at=now - timedelta(minutes=2),
    )
    second = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=12,
        mode="notify",
        notify_text="second",
        run_at=now - timedelta(minutes=1),
    )
    store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=13,
        mode="notify",
        notify_text="later",
        run_at=now + timedelta(minutes=5),
    )

    claimed = store.claim_due_tasks(now=now)

    assert [task.id for task in claimed] == [first.id, second.id]
    assert all(task.status == "running" for task in claimed)
    assert all(task.attempt_count == 1 for task in claimed)


async def test_store_claim_due_tasks_returns_empty_when_nothing_is_due(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()

    claimed = store.claim_due_tasks(now=datetime.now(UTC))

    assert claimed == []


async def test_store_lists_tasks_for_chat_in_user_facing_order(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    now = datetime.now(UTC)
    pending = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="pending",
        run_at=now + timedelta(minutes=5),
    )
    running = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=12,
        mode="notify",
        notify_text="running",
        run_at=now - timedelta(minutes=1),
    )
    other_chat = store.create_task(
        chat_id=TEST_CHAT_ID + 1,
        anchor_message_id=13,
        mode="notify",
        notify_text="other",
        run_at=now + timedelta(minutes=1),
    )
    store.claim_due_tasks(now=now)

    listed = store.list_tasks_for_chat(chat_id=TEST_CHAT_ID)

    assert [task.id for task in listed] == [pending.id, running.id]
    assert other_chat.id not in {task.id for task in listed}


async def test_store_list_tasks_for_chat_handles_empty_statuses_or_limit(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()

    assert store.list_tasks_for_chat(chat_id=TEST_CHAT_ID, statuses=()) == []
    assert store.list_tasks_for_chat(chat_id=TEST_CHAT_ID, limit=0) == []


async def test_store_cancels_pending_tasks_only(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    now = datetime.now(UTC)
    pending = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="pending",
        run_at=now + timedelta(minutes=5),
    )
    running = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=12,
        mode="notify",
        notify_text="running",
        run_at=now - timedelta(minutes=1),
    )
    store.claim_due_tasks(now=now)

    assert store.cancel_task(chat_id=TEST_CHAT_ID, task_id=pending.id)
    assert not store.cancel_task(chat_id=TEST_CHAT_ID, task_id=running.id)

    pending_stored = store.get_task(pending.id)
    running_stored = store.get_task(running.id)
    assert pending_stored is not None
    assert running_stored is not None
    assert pending_stored.status == "cancelled"
    assert running_stored.status == "running"


async def test_store_cancels_all_pending_tasks_for_chat(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    now = datetime.now(UTC)
    first = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="first",
        run_at=now + timedelta(minutes=5),
    )
    second = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=12,
        mode="prompt_agent",
        prompt_text="second",
        run_at=now + timedelta(minutes=6),
    )
    other_chat = store.create_task(
        chat_id=TEST_CHAT_ID + 1,
        anchor_message_id=13,
        mode="notify",
        notify_text="other",
        run_at=now + timedelta(minutes=7),
    )

    cancelled = store.cancel_pending_tasks_for_chat(chat_id=TEST_CHAT_ID)

    first_stored = store.get_task(first.id)
    second_stored = store.get_task(second.id)
    other_stored = store.get_task(other_chat.id)
    assert cancelled == len((first, second))
    assert first_stored is not None
    assert second_stored is not None
    assert other_stored is not None
    assert first_stored.status == "cancelled"
    assert second_stored.status == "cancelled"
    assert other_stored.status == "pending"


async def test_store_recovers_running_tasks_after_restart(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    now = datetime.now(UTC)
    task = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="first",
        run_at=now - timedelta(minutes=2),
    )
    store.claim_due_tasks(now=now)

    recovered = store.recover_running_tasks()
    restored = store.get_task(task.id)

    assert recovered == 1
    assert restored is not None
    assert restored.status == "pending"
    assert restored.claimed_at is None
    assert restored.started_at is None


async def test_store_release_task_returns_running_task_to_pending(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    now = datetime.now(UTC)
    task = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="first",
        run_at=now - timedelta(minutes=2),
    )
    store.claim_due_tasks(now=now)

    store.release_task(task.id)

    released = store.get_task(task.id)
    assert released is not None
    assert released.status == "pending"
    assert released.claimed_at is None


async def test_store_binds_unanchored_tasks_for_chat_and_session(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    task = store.create_task(
        chat_id=TEST_CHAT_ID,
        session_id="session-1",
        mode="notify",
        notify_text="first",
        run_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    store.create_task(
        chat_id=TEST_CHAT_ID,
        session_id="session-2",
        mode="notify",
        notify_text="second",
        run_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    bound = store.bind_unanchored_tasks(
        chat_id=TEST_CHAT_ID,
        session_id="session-1",
        anchor_message_id=ANCHOR_MESSAGE_ID,
    )

    stored = store.get_task(task.id)
    assert bound == 1
    assert stored is not None
    assert stored.anchor_message_id == ANCHOR_MESSAGE_ID


async def test_store_binds_unanchored_tasks_without_session_id(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    task = store.create_task(
        chat_id=TEST_CHAT_ID,
        mode="notify",
        notify_text="first",
        run_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    bound = store.bind_unanchored_tasks(
        chat_id=TEST_CHAT_ID,
        session_id=None,
        anchor_message_id=ANCHOR_MESSAGE_ID,
    )

    stored = store.get_task(task.id)
    assert bound == 1
    assert stored is not None
    assert stored.anchor_message_id == ANCHOR_MESSAGE_ID


async def test_scheduler_marks_task_done_on_success(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    task = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="first",
        run_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    executed: list[str] = []

    async def executor(claimed: ScheduledTask) -> None:
        executed.append(claimed.id)

    scheduler = ScheduledTaskScheduler(
        store=store,
        runner=ScheduledTaskRunner(executor),
        poll_interval_seconds=0.01,
    )

    await scheduler.start()
    stored = await wait_for_task_status(store, task.id, expected_status="done")
    await scheduler.stop()

    assert executed == [task.id]
    assert stored.status == "done"


async def test_scheduler_marks_task_failed_on_execution_error(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    task = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="first",
        run_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    async def executor(_: ScheduledTask) -> None:
        raise ScheduledTaskExecutionError("boom")

    scheduler = ScheduledTaskScheduler(
        store=store,
        runner=ScheduledTaskRunner(executor),
        poll_interval_seconds=0.01,
    )

    await scheduler.start()
    stored = await wait_for_task_status(store, task.id, expected_status="failed")
    await scheduler.stop()

    assert stored.status == "failed"
    assert stored.last_error == "boom"


async def test_scheduler_releases_task_when_execution_is_deferred(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    task = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="first",
        run_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    async def executor(_: ScheduledTask) -> None:
        raise ScheduledTaskDeferredError("busy")

    scheduler = ScheduledTaskScheduler(
        store=store,
        runner=ScheduledTaskRunner(executor),
        poll_interval_seconds=0.01,
    )

    await scheduler.start()
    stored = await wait_for_task_status(store, task.id, expected_status="pending")
    await scheduler.stop()

    assert stored.status == "pending"


async def test_run_claimed_task_releases_deferred_task(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    task = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="first",
        run_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    claimed = store.claim_due_tasks(now=datetime.now(UTC))

    async def executor(_: ScheduledTask) -> None:
        raise ScheduledTaskDeferredError("busy")

    scheduler = ScheduledTaskScheduler(
        store=store,
        runner=ScheduledTaskRunner(executor),
    )

    await scheduler._run_claimed_task(claimed[0])

    stored = store.get_task(task.id)
    assert stored is not None
    assert stored.status == "pending"


async def test_scheduler_marks_task_failed_on_unexpected_error(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    task = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="first",
        run_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    async def executor(_: ScheduledTask) -> None:
        raise RuntimeError(UNEXPECTED_EXECUTION_ERROR)

    scheduler = ScheduledTaskScheduler(
        store=store,
        runner=ScheduledTaskRunner(executor),
        poll_interval_seconds=0.01,
    )

    await scheduler.start()
    stored = await wait_for_task_status(store, task.id, expected_status="failed")
    await scheduler.stop()

    assert stored.status == "failed"
    assert stored.last_error == UNEXPECTED_EXECUTION_ERROR


async def test_execute_scheduled_prompt_replies_to_anchor_message():
    service = ScheduledPromptService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    await bridge.execute_scheduled_task(
        make_task(mode="prompt_agent", prompt_text="check again", notify_text=None),
    )

    assert service.prompt_calls == [(TEST_CHAT_ID, "check again")]
    assert bot.actions == [(TEST_CHAT_ID, "typing")]
    assert bot.sent_messages[-1]["text"] == "Scheduled reply"
    assert bot.sent_messages[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID


async def test_dispatch_reply_binds_unanchored_scheduled_tasks(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    task = store.create_task(
        chat_id=TEST_CHAT_ID,
        session_id="session-1",
        mode="notify",
        notify_text="Ping later",
        run_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    service = ScheduledPromptService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
        scheduled_task_store=store,
    )
    update = cast(Update, SimpleNamespace(message=DummyBotMessage("schedule", message_id=10)))

    await bridge._dispatch_reply(chat_id=TEST_CHAT_ID, update=update, reply=AgentReply(text="Scheduled for later"))

    stored = store.get_task(task.id)
    assert stored is not None
    assert stored.anchor_message_id == BOUND_REPLY_MESSAGE_ID


async def test_bind_pending_scheduled_tasks_logs_and_continues_on_store_error(mocker):
    service = ScheduledPromptService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
        scheduled_task_store=cast(ScheduledTaskStore, mocker.Mock()),
    )
    mocker.patch.object(
        bridge._scheduled_task_store,
        "bind_unanchored_tasks",
        side_effect=RuntimeError("bind boom"),
    )
    log_exception = mocker.patch("telegram_acp_bot.telegram.bridge.logger.exception")

    await bridge._bind_pending_scheduled_tasks(chat_id=TEST_CHAT_ID, anchor_message_id=ANCHOR_MESSAGE_ID)

    log_exception.assert_called_once()


async def test_execute_scheduled_prompt_defers_while_chat_is_busy():
    service = ScheduledPromptService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    lock = bridge._chat_prompt_lock(TEST_CHAT_ID)
    await lock.acquire()
    try:
        with pytest.raises(ScheduledTaskDeferredError):
            await bridge.execute_scheduled_task(make_task(mode="prompt_agent", prompt_text="check again"))
    finally:
        lock.release()

    assert bot.sent_messages == []
    assert service.prompt_calls == []


async def test_execute_scheduled_prompt_defers_when_try_lock_times_out(mocker):
    service = ScheduledPromptService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    slow_lock = SlowAcquireLock()
    mocker.patch.object(bridge, "_chat_prompt_lock", return_value=cast(asyncio.Lock, slow_lock))

    with pytest.raises(ScheduledTaskDeferredError):
        await bridge.execute_scheduled_task(make_task(mode="prompt_agent", prompt_text="check again"))

    assert bot.sent_messages == []
    assert service.prompt_calls == []


async def test_execute_scheduled_notification_requires_running_app():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService()),
    )

    with pytest.raises(ScheduledTaskExecutionError, match="application is not running"):
        await bridge.execute_scheduled_task(make_task(mode="notify"))


async def test_execute_scheduled_notification_requires_anchor_message():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService()),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    with pytest.raises(ScheduledTaskExecutionError, match="anchor message"):
        await bridge.execute_scheduled_task(make_task(mode="notify", anchor_message_id=None))


async def test_execute_scheduled_notification_replies_to_anchor_message():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService()),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    await bridge.execute_scheduled_task(make_task(mode="notify", notify_text="Ping now"))

    assert bot.sent_messages[-1]["text"] == "Ping now"
    assert bot.sent_messages[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID


async def test_execute_scheduled_prompt_requires_running_app():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService()),
    )

    with pytest.raises(ScheduledTaskExecutionError, match="application is not running"):
        await bridge.execute_scheduled_task(make_task(mode="prompt_agent", prompt_text="check again"))


async def test_execute_scheduled_prompt_reports_missing_session():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService(active=False)),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    with pytest.raises(ScheduledTaskExecutionError, match="no active session"):
        await bridge.execute_scheduled_task(make_task(mode="prompt_agent", prompt_text="check again"))

    assert bot.sent_messages[-1]["text"] == "Could not run automatically: no active session."
    assert bot.sent_messages[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID


async def test_execute_scheduled_prompt_rejects_empty_prompt():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService()),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    with pytest.raises(ScheduledTaskExecutionError, match="scheduled prompt is empty"):
        await bridge.execute_scheduled_task(make_task(mode="prompt_agent", prompt_text="   "))

    assert bot.sent_messages[-1]["text"] == "Could not run automatically: scheduled prompt is empty."
    assert bot.sent_messages[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID


async def test_execute_scheduled_prompt_reports_output_limit_error():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, FailingScheduledService(AgentOutputLimitExceededError("too much"))),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    with pytest.raises(ScheduledTaskExecutionError, match="ACP stdio limit"):
        await bridge.execute_scheduled_task(make_task(mode="prompt_agent", prompt_text="check again"))

    assert bot.sent_messages[-1]["text"] == "Could not run automatically: agent output exceeded ACP stdio limit."
    assert bot.sent_messages[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID


async def test_execute_scheduled_prompt_reports_generic_error():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, FailingScheduledService(RuntimeError("agent boom"))),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    with pytest.raises(ScheduledTaskExecutionError, match="agent boom"):
        await bridge.execute_scheduled_task(make_task(mode="prompt_agent", prompt_text="check again"))

    assert bot.sent_messages[-1]["text"] == "Could not run automatically: agent boom"
    assert bot.sent_messages[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID


async def test_execute_scheduled_prompt_reports_empty_agent_reply():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, EmptyReplyScheduledService()),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    with pytest.raises(ScheduledTaskExecutionError, match="no active session"):
        await bridge.execute_scheduled_task(make_task(mode="prompt_agent", prompt_text="check again"))

    assert bot.sent_messages[-1]["text"] == "Could not run automatically: no active session."
    assert bot.sent_messages[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID


async def test_execute_scheduled_prompt_reports_delivery_failure(mocker):
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService()),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    mocker.patch.object(bridge, "_send_agent_reply_to_chat", side_effect=RuntimeError("send boom"))

    with pytest.raises(ScheduledTaskExecutionError, match="Could not deliver scheduled reply: send boom"):
        await bridge.execute_scheduled_task(make_task(mode="prompt_agent", prompt_text="check again"))

    assert bot.sent_messages[-1]["text"] == "Could not deliver scheduled reply: send boom"
    assert bot.sent_messages[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID


async def test_send_agent_reply_to_chat_requires_running_app():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService()),
    )

    with pytest.raises(ScheduledTaskExecutionError, match="application is not running"):
        await bridge._send_agent_reply_to_chat(
            chat_id=TEST_CHAT_ID,
            reply_to_message_id=ANCHOR_MESSAGE_ID,
            reply=AgentReply(text="hello"),
        )


async def test_send_scheduled_status_message_requires_running_app():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService()),
    )

    with pytest.raises(ScheduledTaskExecutionError, match="application is not running"):
        await bridge._send_scheduled_status_message(task=make_task(), text="Running now...")


async def test_send_agent_reply_to_chat_sends_attachments_and_text():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService()),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    reply = AgentReply(
        text="hello",
        images=(ImagePayload(data_base64=base64.b64encode(b"img").decode("ascii"), mime_type="image/jpeg"),),
        files=(FilePayload(name="note.txt", text_content="payload"),),
    )

    await bridge._send_agent_reply_to_chat(
        chat_id=TEST_CHAT_ID,
        reply_to_message_id=ANCHOR_MESSAGE_ID,
        reply=reply,
    )

    assert bot.sent_photos[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID
    assert bot.sent_documents[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID
    assert bot.sent_messages[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID


async def test_send_image_and_file_to_chat_without_reply_target():
    bot = DummyBot()
    image = ImagePayload(data_base64=base64.b64encode(b"img").decode("ascii"), mime_type="image/jpeg")
    file_payload = FilePayload(name="note.txt", data_base64=base64.b64encode(b"payload").decode("ascii"))

    await TelegramBridge._send_image_to_chat(
        bot=cast("bot_module.Bot", bot),
        chat_id=TEST_CHAT_ID,
        reply_to_message_id=None,
        payload=image,
    )
    await TelegramBridge._send_file_to_chat(
        bot=cast("bot_module.Bot", bot),
        chat_id=TEST_CHAT_ID,
        reply_to_message_id=None,
        payload=file_payload,
    )

    assert "reply_to_message_id" not in bot.sent_photos[-1]
    assert "reply_to_message_id" not in bot.sent_documents[-1]


async def test_send_file_to_chat_uses_empty_payload_when_no_content():
    bot = DummyBot()
    file_payload = FilePayload(name="empty.bin")

    await TelegramBridge._send_file_to_chat(
        bot=cast(bot_module.Bot, bot),
        chat_id=TEST_CHAT_ID,
        reply_to_message_id=None,
        payload=file_payload,
    )

    assert bot.sent_documents[-1]["chat_id"] == TEST_CHAT_ID


async def test_send_chat_message_handles_reply_target_with_entities():
    bot = DummyBot()

    await TelegramBridge._send_chat_message(
        bot=cast(bot_module.Bot, bot),
        chat_id=TEST_CHAT_ID,
        text="hello",
        entities=[],
        reply_to_message_id=ANCHOR_MESSAGE_ID,
    )

    assert bot.sent_messages[-1]["reply_to_message_id"] == ANCHOR_MESSAGE_ID
    assert bot.sent_messages[-1]["entities"] == []


async def test_default_scheduled_tasks_db_path_uses_xdg_state_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    assert default_scheduled_tasks_db_path() == tmp_path / "telegram-acp-bot" / "scheduled-tasks.sqlite3"


async def test_format_utc_timestamp_rejects_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        format_utc_timestamp(datetime.now())


async def test_store_exposes_database_path_property(tmp_path: Path):
    path = tmp_path / "scheduled.sqlite3"
    store = ScheduledTaskStore(path)
    assert store.path == path


async def test_scheduler_stop_cancels_active_wrapper_task(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()

    gate = asyncio.Event()

    async def executor(_: ScheduledTask) -> None:
        await gate.wait()

    scheduler = ScheduledTaskScheduler(store=store, runner=ScheduledTaskRunner(executor), poll_interval_seconds=1)
    active = asyncio.create_task(scheduler._run_claimed_task(make_task()))
    scheduler._active_runs.add(active)

    await scheduler.stop()

    assert active.cancelled()


async def test_scheduler_start_recovers_running_tasks_and_starts_loop(tmp_path: Path):
    store = ScheduledTaskStore(tmp_path / "scheduled.sqlite3")
    store.initialize()
    task = store.create_task(
        chat_id=TEST_CHAT_ID,
        anchor_message_id=11,
        mode="notify",
        notify_text="first",
        run_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    store.claim_due_tasks(now=datetime.now(UTC))

    scheduler = ScheduledTaskScheduler(store=store, runner=ScheduledTaskRunner(lambda _: asyncio.sleep(0)))

    await scheduler.start()
    await scheduler.stop()

    recovered = store.get_task(task.id)
    assert recovered is not None
    assert recovered.status == "pending"


async def test_run_polling_uses_scheduler_when_provided(monkeypatch: pytest.MonkeyPatch):
    calls: list[object] = []

    class DummyApp:
        def run_polling(self, *, allowed_updates):
            calls.append(allowed_updates)

    fake_scheduler = cast(app_module.ScheduledTaskScheduler, FakeScheduler())

    def fake_build_application(config, bridge, scheduler=None):
        del config, bridge
        assert scheduler is fake_scheduler
        return DummyApp()

    monkeypatch.setattr(app_module, "build_application", fake_build_application)

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(
        config=config,
        agent_service=cast(AgentService, ScheduledPromptService()),
    )
    assert app_module.run_polling(config, bridge, scheduler=fake_scheduler) == 0
    assert len(calls) == 1


async def test_build_application_registers_scheduler_hooks():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService()),
    )
    scheduler = FakeScheduler()
    app = build_application(
        make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        bridge,
        scheduler=cast(app_module.ScheduledTaskScheduler, scheduler),
    )

    await app_module._post_init_factory(cast(app_module.ScheduledTaskScheduler, scheduler))(app)
    await app_module._post_shutdown_factory(cast(app_module.ScheduledTaskScheduler, scheduler))(app)

    assert scheduler.started is True
    assert scheduler.stopped is True


async def test_on_activity_event_ignores_suppressed_chat():
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ScheduledPromptService()),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    bridge._suppressed_activity_chats.add(TEST_CHAT_ID)

    await bridge.on_activity_event(
        TEST_CHAT_ID,
        AgentActivityBlock(kind="execute", title="Run", status="in_progress"),
    )

    assert bot.sent_messages == []

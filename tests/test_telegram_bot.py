from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from telegram import InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import Application

from telegram_acp_bot.acp_app.echo_service import EchoAgentService
from telegram_acp_bot.acp_app.models import (
    AgentActivityBlock,
    AgentOutputLimitExceededError,
    AgentReply,
    FilePayload,
    ImagePayload,
    PermissionRequest,
    ResumableSession,
)
from telegram_acp_bot.core.session_registry import SessionRegistry
from telegram_acp_bot.telegram import bot as bot_module
from telegram_acp_bot.telegram.bot import (
    BUSY_CALLBACK_PREFIX,
    RESTART_EXIT_CODE,
    RESUME_KEYBOARD_MAX_ROWS,
    AgentService,
    ChatRequiredError,
    TelegramBridge,
    _PendingPrompt,
    _PromptInput,
    build_application,
    make_config,
    run_polling,
)

pytestmark = pytest.mark.asyncio

EXPECTED_OUTBOUND_DOCUMENTS = 2
TEST_CHAT_ID = 100
EXPECTED_ACTIVITY_MESSAGES = 3
ACP_STDIO_LIMIT_ERROR = "Separator is found, but chunk is longer than limit"
EXPECTED_TEXT_REPLIES_WITH_IMPLICIT_AND_EXPLICIT_SESSION = 3
EXPECTED_BUSY_NOTIFY_MESSAGES_AFTER_REPLACE = 2
QUEUED_MESSAGE_ID = 22


class MarkdownFailureError(TelegramError):
    """Raised by test doubles to emulate Telegram markdown parse failure."""

    def __init__(self) -> None:
        super().__init__("bad markdown")


class DummyLoadFailedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("load failed")


class DummyListBoomError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("list boom")


class DummyCancelBoomError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("cancel boom")


class DummyMessage:
    def __init__(
        self,
        text: str | None = None,
        *,
        message_id: int = 1,
        caption: str | None = None,
        photo: Sequence[object] | None = None,
        document: object | None = None,
    ) -> None:
        self.message_id = message_id
        self.text = text
        self.caption = caption
        self.photo = list(photo) if photo is not None else []
        self.document = document
        self.replies: list[str] = []
        self.reply_kwargs: list[dict[str, object]] = []
        self.fail_markdown = False
        self.fail_entities = False
        self.photos: list[object] = []
        self.documents: list[object] = []

    async def reply_text(self, text: str, **kwargs: object) -> None:
        if self.fail_markdown and kwargs.get("parse_mode") is not None:
            self.reply_kwargs.append(kwargs)
            raise MarkdownFailureError
        if self.fail_entities and "entities" in kwargs:
            self.reply_kwargs.append(kwargs)
            raise MarkdownFailureError
        self.reply_kwargs.append(kwargs)
        self.replies.append(text)

    async def reply_photo(self, *, photo: object) -> None:
        self.photos.append(photo)

    async def reply_document(self, *, document: object) -> None:
        self.documents.append(document)


class DummyDownloadedFile:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def download_as_bytearray(self) -> bytearray:
        return bytearray(self._payload)


class DummyBot:
    def __init__(self) -> None:
        self.actions: list[tuple[int, str]] = []
        self.files: dict[str, bytes] = {}
        self.sent_messages: list[dict[str, object]] = []
        self._next_message_id = 1
        self.edited_reply_markups: list[dict[str, object]] = []

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.actions.append((chat_id, action))

    async def get_file(self, file_id: str) -> DummyDownloadedFile:
        return DummyDownloadedFile(self.files[file_id])

    async def send_message(self, **kwargs: object) -> SimpleNamespace:
        self.sent_messages.append(kwargs)
        msg = SimpleNamespace(message_id=self._next_message_id)
        self._next_message_id += 1
        return msg

    async def edit_message_reply_markup(self, **kwargs: object) -> None:
        self.edited_reply_markups.append(kwargs)


class FailingMarkdownBot(DummyBot):
    async def send_message(self, **kwargs: object) -> SimpleNamespace:
        if "entities" in kwargs:
            raise MarkdownFailureError
        return await super().send_message(**kwargs)


class DummyCallbackQuery:
    def __init__(self, data: str) -> None:
        self.data = data
        self.message = SimpleNamespace(text="Permission required\nRun ls", chat=SimpleNamespace(id=TEST_CHAT_ID))
        self.answers: list[str] = []
        self.reply_markup_cleared = False
        self.edited_text: str | None = None

    async def answer(self, text: str) -> None:
        self.answers.append(text)

    async def edit_message_reply_markup(self, *, reply_markup: object | None = None) -> None:
        self.reply_markup_cleared = reply_markup is None

    async def edit_message_text(self, text: str) -> None:
        self.edited_text = text


class LiveActivityService:
    def __init__(self) -> None:
        self._activity_handler: Callable[[int, AgentActivityBlock], Awaitable[None]] | None = None

    async def new_session(self, *, chat_id: int, workspace):
        del workspace
        return f"s-{chat_id}"

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
        del text, images, files
        if self._activity_handler is not None:
            await self._activity_handler(
                chat_id,
                AgentActivityBlock(
                    kind="think",
                    title="Inspecting history",
                    status="completed",
                    text="Checking latest commit touching tests.",
                ),
            )
        return AgentReply(text="Final response.")

    def get_workspace(self, *, chat_id: int):
        del chat_id

    async def cancel(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    async def stop(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    async def clear(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    def get_permission_policy(self, *, chat_id: int):
        del chat_id

    async def set_session_permission_mode(self, *, chat_id: int, mode):
        del chat_id, mode
        return False

    async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
        del chat_id, enabled
        return False

    def set_activity_event_handler(self, handler):
        self._activity_handler = handler


class ResumeService:
    def __init__(self) -> None:
        self.loaded: tuple[int, str, Path] | None = None
        self.fail_load = False
        self.list_supported = True
        self.items: tuple[ResumableSession, ...] = (
            ResumableSession(
                session_id="s-resume-1",
                workspace=Path("/tmp/ws1"),
                title="First session",
                updated_at="2026-03-02T12:00:00Z",
            ),
            ResumableSession(
                session_id="s-resume-2",
                workspace=Path("/tmp/ws2"),
                title="Second session",
                updated_at="2026-03-02T11:00:00Z",
            ),
        )

    async def new_session(self, *, chat_id: int, workspace):
        del workspace
        return f"s-{chat_id}"

    async def load_session(self, *, chat_id: int, session_id: str, workspace: Path) -> str:
        if self.fail_load:
            raise DummyLoadFailedError()
        self.loaded = (chat_id, session_id, workspace)
        return session_id

    async def list_resumable_sessions(self, *, chat_id: int, workspace: Path | None = None):
        del chat_id
        if not self.list_supported:
            return None
        if workspace is None:
            return self.items
        return tuple(item for item in self.items if item.workspace == workspace)

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
        del chat_id, text, images, files
        return AgentReply(text="ok")

    def get_workspace(self, *, chat_id: int):
        del chat_id

    def supports_session_loading(self, *, chat_id: int):
        del chat_id
        return True

    async def cancel(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    async def stop(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    async def clear(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    def get_permission_policy(self, *, chat_id: int):
        del chat_id

    async def set_session_permission_mode(self, *, chat_id: int, mode):
        del chat_id, mode
        return False

    async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
        del chat_id, enabled
        return False


class ImplicitSessionServiceBase:
    def __init__(self) -> None:
        self._workspace_by_chat: dict[int, Path] = {}

    def get_workspace(self, *, chat_id: int):
        return self._workspace_by_chat.get(chat_id)

    async def cancel(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    async def stop(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    async def clear(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    def get_permission_policy(self, *, chat_id: int):
        del chat_id

    async def set_session_permission_mode(self, *, chat_id: int, mode):
        del chat_id, mode
        return False

    async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
        del chat_id, enabled
        return False


class RecordingImplicitService(ImplicitSessionServiceBase):
    def __init__(self) -> None:
        super().__init__()
        self.new_session_calls: list[tuple[int, Path]] = []

    async def new_session(self, *, chat_id: int, workspace: Path):
        self.new_session_calls.append((chat_id, workspace))
        self._workspace_by_chat[chat_id] = workspace
        return "s-1"

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
        del chat_id, text, images, files
        return AgentReply(text="ok")


class FailingImplicitService(ImplicitSessionServiceBase):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    async def new_session(self, *, chat_id: int, workspace: Path):
        del chat_id, workspace
        raise self._error

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
        del chat_id, text, images, files
        return AgentReply(text="ok")


class PromptWithoutSessionImplicitService(ImplicitSessionServiceBase):
    async def new_session(self, *, chat_id: int, workspace: Path):
        self._workspace_by_chat[chat_id] = workspace
        return "s-1"

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
        del chat_id, text, images, files


class ConcurrentImplicitSessionService(ImplicitSessionServiceBase):
    def __init__(self) -> None:
        super().__init__()
        self.new_session_calls = 0

    async def new_session(self, *, chat_id: int, workspace: Path):
        self.new_session_calls += 1
        await asyncio.sleep(0.01)
        self._workspace_by_chat[chat_id] = workspace
        return "s-1"

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
        del chat_id, text, images, files
        return AgentReply(text="ok")


def make_update(  # noqa: PLR0913
    *,
    user_id: int = 1,
    username: str | None = None,
    chat_id: int = 100,
    text: str | None = None,
    caption: str | None = None,
    photo: Sequence[object] | None = None,
    document: object | None = None,
    message_id: int = 1,
    with_message: bool = True,
):
    message = (
        DummyMessage(text, message_id=message_id, caption=caption, photo=photo, document=document)
        if with_message
        else None
    )
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id, username=username),
        effective_chat=SimpleNamespace(id=chat_id),
        message=message,
    )


def make_context(*, args: list[str] | None = None, application: object | None = None):
    return SimpleNamespace(args=args or [], bot=DummyBot(), application=application)


def make_bridge(*, allowed_ids: set[int] | None = None) -> TelegramBridge:
    config = make_config(token="TOKEN", allowed_user_ids=list(allowed_ids or set()), workspace=".")
    return TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))


async def test_make_config():
    config = make_config(token="T", allowed_user_ids=[1, 2, 2], workspace="~/tmp")
    assert config.token == "T"
    assert config.allowed_user_ids == {1, 2}
    assert config.default_workspace.name == "tmp"


async def test_workspace_from_relative_arg_uses_default_workspace():
    config = make_config(token="T", allowed_user_ids=[], workspace="/tmp/base")
    bridge = TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))

    workspace = bridge._workspace_from_args(["foo"])
    assert workspace == Path("/tmp/base/foo")


async def test_start_and_help():
    bridge = make_bridge()
    update = make_update(with_message=True)
    context = make_context()

    await bridge.start(update, context)
    await bridge.help(update, context)

    assert update.message is not None
    assert "Send a message to start in the default workspace" in update.message.replies[0]
    assert "Commands:" in update.message.replies[1]
    assert "/cancel" in update.message.replies[1]


async def test_start_allows_user_by_username_allowlist():
    config = make_config(token="TOKEN", allowed_user_ids=[], allowed_usernames=["@Alice"], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))
    update = make_update(user_id=999, username="Alice", with_message=True)
    context = make_context()

    await bridge.start(update, context)

    assert update.message is not None
    assert "Send a message to start in the default workspace" in update.message.replies[0]


async def test_restart_requests_app_stop():
    bridge = make_bridge()
    update = make_update(with_message=True)
    stop_calls: list[str] = []
    bridge._app = cast(Application, SimpleNamespace(stop_running=lambda: stop_calls.append("stop")))

    await bridge.restart(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["Restart requested. Re-launching process..."]
    assert stop_calls == ["stop"]


async def test_restart_with_index_resumes_selected_candidate():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert service.loaded == (TEST_CHAT_ID, "s-resume-1", Path("/tmp/ws1"))
    assert update.message is not None
    assert update.message.replies == ["Session restarted: s-resume-1 in /tmp/ws1"]


async def test_restart_with_workspace_arg_only_reports_usage():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.restart(update, make_context(args=["/tmp/ws"]))

    assert update.message is not None
    assert update.message.replies == ["Usage: /restart or /restart N [workspace]"]


async def test_restart_with_too_many_args_reports_usage():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.restart(update, make_context(args=["1", "/tmp/ws", "extra"]))

    assert update.message is not None
    assert update.message.replies == ["Usage: /restart or /restart N [workspace]"]


async def test_restart_with_zero_index_reports_usage():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert service.loaded == (TEST_CHAT_ID, "s-resume-1", Path("/tmp/ws1"))
    assert update.message is not None
    assert update.message.replies == ["Session restarted: s-resume-1 in /tmp/ws1"]


async def test_restart_with_two_indexes_reports_usage():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.restart(update, make_context(args=["1", "2"]))

    assert update.message is not None
    assert update.message.replies == ["Usage: /restart or /restart N [workspace]"]


async def test_restart_with_two_workspace_args_reports_usage():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.restart(update, make_context(args=["/tmp/ws1", "/tmp/ws2"]))

    assert update.message is not None
    assert update.message.replies == ["Usage: /restart or /restart N [workspace]"]


async def test_restart_with_index_reports_list_error():
    class FailingListResumeService(ResumeService):
        async def list_resumable_sessions(self, *, chat_id: int, workspace: Path | None = None):
            del chat_id, workspace
            raise DummyListBoomError()

    service = FailingListResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert update.message is not None
    assert update.message.replies == ["Failed to list resumable sessions: list boom"]


async def test_restart_with_index_reports_list_not_supported():
    service = ResumeService()
    service.list_supported = False
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert update.message is not None
    assert update.message.replies == ["Agent does not support ACP session/list."]


async def test_restart_with_index_reports_empty_results():
    service = ResumeService()
    service.items = ()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert update.message is not None
    assert update.message.replies == ["No resumable sessions found."]


async def test_restart_with_invalid_index_reports_error():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["9"]))

    assert update.message is not None
    assert update.message.replies == ["Invalid restart index 9. Choose 0..1."]


async def test_restart_with_index_reports_load_failure():
    service = ResumeService()
    service.fail_load = True
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert update.message is not None
    assert update.message.replies == ["Failed to resume session s-resume-1: load failed"]


async def test_restart_requires_running_application():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.restart(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["Restart is unavailable: application is not running."]


async def test_restart_access_denied():
    bridge = make_bridge(allowed_ids={999})
    update = make_update(user_id=1, with_message=True)

    await bridge.restart(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["Access denied for this bot."]


async def test_access_denied():
    bridge = make_bridge(allowed_ids={99})
    update = make_update(user_id=1)
    context = make_context()

    await bridge.start(update, context)

    assert update.message is not None
    assert update.message.replies == ["Access denied for this bot."]


async def test_access_allowed_with_allowlist():
    bridge = make_bridge(allowed_ids={1})
    update = make_update(user_id=1)
    context = make_context()

    await bridge.start(update, context)

    assert update.message is not None
    assert len(update.message.replies) == 1
    assert "Send a message to start in the default workspace" in update.message.replies[0]


async def test_denied_paths_for_other_handlers():
    bridge = make_bridge(allowed_ids={42})
    update = make_update(user_id=7, text="hello")
    context = make_context()

    await bridge.help(update, context)
    await bridge.new_session(update, make_context(args=["/tmp"]))
    await bridge.resume_session(update, make_context(args=["/tmp"]))
    await bridge.session(update, context)
    await bridge.cancel(update, context)
    await bridge.stop(update, context)
    await bridge.clear(update, context)
    await bridge.on_message(update, context)

    assert update.message is not None
    assert update.message.replies == [
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
    ]


async def test_new_session_and_session_command():
    bridge = make_bridge()
    update = make_update()

    await bridge.session(update, make_context())
    await bridge.new_session(update, make_context(args=["/tmp"]))
    await bridge.session(update, make_context())

    assert update.message is not None
    assert update.message.replies[0] == "No active session. Use /new first."
    assert "Session started:" in update.message.replies[1]
    assert "Active session workspace:" in update.message.replies[2]


async def test_resume_session_without_app_loads_first_candidate():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update()

    await bridge.resume_session(update, make_context())

    assert service.loaded is not None
    assert service.loaded[1] == "s-resume-1"
    assert update.message is not None
    assert "Session resumed:" in update.message.replies[0]


async def test_resume_session_with_app_sends_picker_message():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context())

    assert bot.sent_messages
    payload = bot.sent_messages[-1]
    assert payload["chat_id"] == TEST_CHAT_ID
    assert "Pick a session to resume" in cast(str, payload["text"])
    assert payload["reply_markup"] is not None


async def test_resume_session_with_workspace_arg_loads_most_recent_for_workspace(tmp_path: Path):
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path)),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context(args=["/tmp/ws2"]))

    assert service.loaded == (TEST_CHAT_ID, "s-resume-2", Path("/tmp/ws2"))
    assert update.message is not None
    assert update.message.replies == ["Session resumed: s-resume-2 in /tmp/ws2"]
    assert bot.sent_messages == []


async def test_resume_session_with_index_arg_loads_selected_candidate():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context(args=["1"]))

    assert service.loaded == (TEST_CHAT_ID, "s-resume-2", Path("/tmp/ws2"))
    assert update.message is not None
    assert update.message.replies == ["Session resumed: s-resume-2 in /tmp/ws2"]


async def test_resume_session_with_invalid_index_reports_error():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context(args=["9"]))

    assert update.message is not None
    assert update.message.replies == ["Invalid resume index 9. Choose 0..1."]


async def test_resume_session_with_zero_index_reports_usage():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context(args=["0"]))

    assert service.loaded == (TEST_CHAT_ID, "s-resume-1", Path("/tmp/ws1"))
    assert update.message is not None
    assert update.message.replies == ["Session resumed: s-resume-1 in /tmp/ws1"]


async def test_resume_session_rejects_combined_index_and_workspace_args():
    bridge = make_bridge()
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context(args=["1", "/tmp/ws1"]))

    assert update.message is not None
    assert update.message.replies == ["Usage: /resume, /resume N, or /resume [workspace]"]


async def test_resume_session_reports_list_not_supported():
    service = ResumeService()
    service.list_supported = False
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update()

    await bridge.resume_session(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["Agent does not support ACP session/list."]


async def test_resume_session_reports_empty_results():
    service = ResumeService()
    service.items = ()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update()

    await bridge.resume_session(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["No resumable sessions found."]


async def test_resume_session_reports_list_error():
    class FailingListResumeService(ResumeService):
        async def list_resumable_sessions(self, *, chat_id: int, workspace: Path | None = None):
            del chat_id, workspace
            raise DummyListBoomError()

    service = FailingListResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update()
    await bridge.resume_session(update, make_context())
    assert update.message is not None
    assert "Failed to list resumable sessions: list boom" in update.message.replies[-1]


async def test_new_session_autocreates_relative_workspace_and_reports_it(tmp_path: Path):
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path))
    bridge = TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))
    update = make_update()

    await bridge.new_session(update, make_context(args=["myproj"]))

    created_path = tmp_path / "myproj"
    assert created_path.is_dir()
    assert update.message is not None
    assert "Session started:" in update.message.replies[0]
    assert f"Created workspace: {created_path}" in update.message.replies[0]


async def test_new_session_reports_invalid_workspace():
    class InvalidWorkspaceService:
        async def new_session(self, *, chat_id: int, workspace):
            del chat_id, workspace
            raise ValueError("/missing")

        async def prompt(self, *, chat_id: int, text: str):
            del chat_id, text

        def get_workspace(self, *, chat_id: int):
            del chat_id

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, InvalidWorkspaceService()))
    update = make_update()

    await bridge.new_session(update, make_context(args=["/missing"]))

    assert update.message is not None
    assert update.message.replies == ["Invalid workspace: /missing"]


async def test_new_session_reports_process_stdio_error():
    class BrokenAgentService:
        async def new_session(self, *, chat_id: int, workspace):
            del chat_id, workspace
            raise RuntimeError

        async def prompt(self, *, chat_id: int, text: str):
            del chat_id, text

        def get_workspace(self, *, chat_id: int):
            del chat_id

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, BrokenAgentService()))
    update = make_update()

    await bridge.new_session(update, make_context(args=["/tmp"]))

    assert update.message is not None
    assert update.message.replies == ["Failed to start session: agent process did not expose stdio pipes."]


async def test_new_session_reports_generic_error():
    class BoomError(Exception):
        pass

    class UnexpectedService:
        async def new_session(self, *, chat_id: int, workspace):
            del chat_id, workspace
            raise BoomError("boom")

        async def prompt(self, *, chat_id: int, text: str):
            del chat_id, text

        def get_workspace(self, *, chat_id: int):
            del chat_id

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, UnexpectedService()))
    update = make_update()

    await bridge.new_session(update, make_context(args=["/tmp"]))

    assert update.message is not None
    assert update.message.replies == ["Failed to start session: boom"]


async def test_on_text_without_and_with_session():
    bridge = make_bridge()
    update = make_update(text="hello")
    context = make_context()

    await bridge.on_message(update, context)
    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message is not None
    assert len(update.message.replies) == EXPECTED_TEXT_REPLIES_WITH_IMPLICIT_AND_EXPLICIT_SESSION
    assert update.message.replies[0].endswith("hello")
    assert update.message.replies[-1].endswith("hello")
    assert context.bot.actions == [(100, "typing"), (100, "typing")]
    assert update.message.reply_kwargs[-1] == {}
    assert "parse_mode" not in update.message.reply_kwargs[-1]


async def test_first_prompt_starts_implicit_session_in_default_workspace(tmp_path: Path):
    service = RecordingImplicitService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path))
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    update = make_update(text="hello")

    await bridge.on_message(update, make_context())

    assert service.new_session_calls == [(TEST_CHAT_ID, tmp_path)]
    assert update.message is not None
    assert update.message.replies == ["ok"]


async def test_implicit_start_lock_is_dropped_once_session_exists(tmp_path: Path):
    service = RecordingImplicitService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path))
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))

    await bridge.on_message(make_update(chat_id=TEST_CHAT_ID, text="hello"), make_context())

    assert TEST_CHAT_ID not in bridge._implicit_start_locks_by_chat


async def test_drop_implicit_start_lock_keeps_lock_when_expected_lock_differs():
    bridge = make_bridge()
    stored_lock = asyncio.Lock()
    different_lock = asyncio.Lock()
    bridge._implicit_start_locks_by_chat[TEST_CHAT_ID] = stored_lock

    bridge._drop_implicit_start_lock(chat_id=TEST_CHAT_ID, expected_lock=different_lock)

    assert bridge._implicit_start_locks_by_chat[TEST_CHAT_ID] is stored_lock


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ValueError("/bad-default"), "Invalid default workspace: /bad-default"),
        (RuntimeError(), "Failed to start session: agent process did not expose stdio pipes."),
        (Exception("boom"), "Failed to start session: boom"),
    ],
)
async def test_first_prompt_reports_implicit_session_start_errors(error: Exception, expected: str):
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    service = FailingImplicitService(error)
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    update = make_update(text="hello")
    context = make_context()

    await bridge.on_message(update, context)

    assert update.message is not None
    assert update.message.replies == [expected]
    assert context.bot.actions == []


async def test_on_message_without_session_after_implicit_start_reports_missing_session():
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(
        config=config,
        agent_service=cast(AgentService, PromptWithoutSessionImplicitService()),
    )
    update = make_update(text="hello")

    await bridge.on_message(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["No active session. Send a message again or use /new [workspace]."]


async def test_concurrent_first_prompts_start_only_one_implicit_session(tmp_path: Path):
    service = ConcurrentImplicitSessionService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path))
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    update_one = make_update(chat_id=TEST_CHAT_ID, text="hello one")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="hello two")
    context_one = make_context()
    context_two = make_context()

    await asyncio.gather(
        bridge.on_message(update_one, context_one),
        bridge.on_message(update_two, context_two),
    )

    assert service.new_session_calls == 1
    assert update_one.message is not None
    assert update_two.message is not None
    assert update_one.message.replies == ["ok"]
    assert update_two.message.replies == ["ok"]
    assert TEST_CHAT_ID not in bridge._implicit_start_locks_by_chat


async def test_on_text_plain_reply_when_response_has_no_entities():
    bridge = make_bridge()
    update = make_update(text="hello")
    assert update.message is not None
    update.message.fail_entities = True
    context = make_context()

    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message.replies[-1].endswith("hello")
    assert update.message.reply_kwargs[-1] == {}


async def test_on_message_with_photo_attachment():
    bridge = make_bridge()
    photo = [SimpleNamespace(file_id="p1")]
    update = make_update(photo=photo)
    context = make_context()
    context.bot.files["p1"] = b"abc"

    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message is not None
    assert "images=1" in update.message.replies[-1]


async def test_on_message_with_document_attachment():
    bridge = make_bridge()
    document = SimpleNamespace(file_id="d1", mime_type="text/plain", file_name="note.txt")
    update = make_update(document=document)
    context = make_context()
    context.bot.files["d1"] = b"hello from file"

    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message is not None
    assert "files=1" in update.message.replies[-1]


async def test_on_message_with_binary_document_attachment():
    bridge = make_bridge()
    document = SimpleNamespace(file_id="bin-doc", mime_type="application/octet-stream", file_name="x.bin")
    update = make_update(document=document)
    context = make_context()
    context.bot.files["bin-doc"] = b"\xff\xfe"

    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message is not None
    assert "files=1" in update.message.replies[-1]


async def test_on_message_with_image_document_attachment():
    bridge = make_bridge()
    document = SimpleNamespace(file_id="img-doc", mime_type="image/png", file_name="x.png")
    update = make_update(document=document)
    context = make_context()
    context.bot.files["img-doc"] = b"\x89PNG"

    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message is not None
    assert "images=1" in update.message.replies[-1]


async def test_outbound_agent_attachments_are_sent():
    class AttachmentService:
        async def new_session(self, *, chat_id: int, workspace):
            del workspace
            return f"s-{chat_id}"

        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del chat_id, text, images, files
            return AgentReply(
                text="ok",
                images=(ImagePayload(data_base64=base64.b64encode(b"img").decode("ascii"), mime_type="image/jpeg"),),
                files=(
                    FilePayload(name="out.txt", text_content="content"),
                    FilePayload(name="out.bin", data_base64=base64.b64encode(b"bin").decode("ascii")),
                ),
            )

        def get_workspace(self, *, chat_id: int):
            del chat_id

        async def cancel(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def stop(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def clear(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        def get_permission_policy(self, *, chat_id: int):
            del chat_id

        async def set_session_permission_mode(self, *, chat_id: int, mode):
            del chat_id, mode
            return False

        async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
            del chat_id, enabled
            return False

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, AttachmentService()))
    update = make_update(text="hello")

    await bridge.on_message(update, make_context())

    assert update.message is not None
    assert update.message.replies[-1] == "ok"
    assert len(update.message.photos) == 1
    assert len(update.message.documents) == EXPECTED_OUTBOUND_DOCUMENTS


async def test_on_message_renders_activity_blocks_before_final_reply():
    class ActivityService:
        async def new_session(self, *, chat_id: int, workspace):
            del workspace
            return f"s-{chat_id}"

        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del chat_id, text, images, files
            return AgentReply(
                text="Done.",
                activity_blocks=(
                    AgentActivityBlock(
                        kind="think",
                        title="Draft plan",
                        status="completed",
                        text="Need to inspect repository files.",
                    ),
                    AgentActivityBlock(
                        kind="execute",
                        title="Run tests",
                        status="completed",
                        text="uv run pytest",
                    ),
                ),
            )

        def get_workspace(self, *, chat_id: int):
            del chat_id

        async def cancel(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def stop(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def clear(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        def get_permission_policy(self, *, chat_id: int):
            del chat_id

        async def set_session_permission_mode(self, *, chat_id: int, mode):
            del chat_id, mode
            return False

        async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
            del chat_id, enabled
            return False

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, ActivityService()))
    update = make_update(text="hello")

    await bridge.on_message(update, make_context())

    assert update.message is not None
    assert len(update.message.replies) == EXPECTED_ACTIVITY_MESSAGES
    assert "💡 Thinking" in update.message.replies[0]
    assert "Draft plan" not in update.message.replies[0]
    assert "⚙️ Running" in update.message.replies[1]
    assert update.message.replies[2] == "Done."


async def test_on_message_sends_live_activity_events_via_app_bot():
    service = LiveActivityService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    update = make_update(text="hello")
    context = make_context()
    bridge._app = cast(Application, SimpleNamespace(bot=context.bot))

    await bridge.on_message(update, context)

    assert update.message is not None
    assert update.message.replies[-1] == "Final response."
    assert context.bot.sent_messages
    assert "💡 Thinking" in cast(str, context.bot.sent_messages[0]["text"])


async def test_on_message_skips_empty_final_text_reply():
    class EmptyTextService:
        async def new_session(self, *, chat_id: int, workspace):
            del workspace
            return f"s-{chat_id}"

        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del chat_id, text, images, files
            return AgentReply(text="")

        def get_workspace(self, *, chat_id: int):
            del chat_id

        async def cancel(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def stop(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def clear(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        def get_permission_policy(self, *, chat_id: int):
            del chat_id

        async def set_session_permission_mode(self, *, chat_id: int, mode):
            del chat_id, mode
            return False

        async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
            del chat_id, enabled
            return False

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, EmptyTextService()))
    update = make_update(text="hello")
    context = make_context()

    await bridge.on_message(update, context)

    assert update.message is not None
    assert update.message.replies == []


async def test_on_message_reports_acp_stdio_limit_error():
    class LimitErrorService:
        async def new_session(self, *, chat_id: int, workspace):
            del workspace
            return f"s-{chat_id}"

        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del chat_id, text, images, files
            raise AgentOutputLimitExceededError(ACP_STDIO_LIMIT_ERROR)

        def get_workspace(self, *, chat_id: int):
            del chat_id

        async def cancel(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def stop(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def clear(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        def get_permission_policy(self, *, chat_id: int):
            del chat_id

        async def set_session_permission_mode(self, *, chat_id: int, mode):
            del chat_id, mode
            return False

        async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
            del chat_id, enabled
            return False

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, LimitErrorService()))
    update = make_update(text="hello")
    context = make_context()

    await bridge.on_message(update, context)

    assert update.message is not None
    assert "Agent output exceeded ACP stdio limit." in update.message.replies[-1]


async def test_on_message_reraises_unrelated_value_error():
    class GenericValueErrorService:
        async def new_session(self, *, chat_id: int, workspace):
            del workspace
            return f"s-{chat_id}"

        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del chat_id, text, images, files
            raise ValueError("unexpected")

        def get_workspace(self, *, chat_id: int):
            del chat_id

        async def cancel(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def stop(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def clear(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        def get_permission_policy(self, *, chat_id: int):
            del chat_id

        async def set_session_permission_mode(self, *, chat_id: int, mode):
            del chat_id, mode
            return False

        async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
            del chat_id, enabled
            return False

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, GenericValueErrorService()))
    update = make_update(text="hello")
    context = make_context()

    with pytest.raises(ValueError, match="unexpected"):
        await bridge.on_message(update, context)


async def test_on_activity_event_without_app_is_noop():
    bridge = make_bridge()
    block = AgentActivityBlock(kind="think", title="x", status="completed", text="y")
    await bridge.on_activity_event(TEST_CHAT_ID, block)


async def test_on_activity_event_markdown_fallback():
    bridge = make_bridge()
    failing_bot = FailingMarkdownBot()
    bridge._app = cast(Application, SimpleNamespace(bot=failing_bot))
    block = AgentActivityBlock(kind="execute", title="Run cmd", status="in_progress", text="")

    await bridge.on_activity_event(TEST_CHAT_ID, block)

    assert failing_bot.sent_messages
    assert "parse_mode" not in failing_bot.sent_messages[-1]


async def test_resume_keyboard_limits_to_ten_entries():
    candidates = tuple(
        ResumableSession(
            session_id=f"s-{index}",
            workspace=Path("/tmp/ws"),
            title=f"title {index}",
            updated_at="2026-03-02T12:00:00Z",
        )
        for index in range(12)
    )
    keyboard = TelegramBridge._resume_keyboard(candidates=candidates)
    assert len(keyboard.inline_keyboard) == RESUME_KEYBOARD_MAX_ROWS
    assert keyboard.inline_keyboard[0][0].text.startswith("0. ")
    assert keyboard.inline_keyboard[1][0].text.startswith("1. ")


async def test_format_activity_block_read_escapes_markdown_and_removes_read_prefix():
    block = AgentActivityBlock(
        kind="read", title="Read test_telegram_bot.py", status="completed", text="Read test_telegram_bot.py"
    )
    rendered = TelegramBridge._format_activity_block(block, workspace=Path("/tmp/ws"))
    assert "*📖 Reading*" in rendered
    assert "`/tmp/ws/test_telegram_bot.py`" in rendered
    assert "\n\nRead /tmp/ws/test_telegram_bot.py" not in rendered


async def test_format_activity_block_edit_uses_absolute_path_and_removes_edit_prefix():
    block = AgentActivityBlock(kind="edit", title="Edit src/telegram_acp_bot/telegram/bot.py", status="completed")
    rendered = TelegramBridge._format_activity_block(block, workspace=Path("/tmp/ws"))
    assert "*✏️ Editing*" in rendered
    assert "`/tmp/ws/src/telegram_acp_bot/telegram/bot.py`" in rendered
    assert "\n\nEdit /tmp/ws/src/telegram_acp_bot/telegram/bot.py" not in rendered


async def test_format_activity_block_read_without_workspace_keeps_relative_path():
    block = AgentActivityBlock(kind="read", title="Read README.md", status="completed", text="Read README.md")
    rendered = TelegramBridge._format_activity_block(block, workspace=None)
    assert "*📖 Reading*" in rendered
    assert f"`{Path.cwd().resolve() / 'README.md'}`" in rendered
    assert "`README.md`" not in rendered


async def test_format_activity_block_read_prefers_file_uri_path():
    block = AgentActivityBlock(
        kind="read",
        title="Read [@README.md](file:///home/tin/lab/telegram-acp/README.md)",
        status="completed",
    )
    rendered = TelegramBridge._format_activity_block(block, workspace=Path("/tmp/ws"))
    assert "`/home/tin/lab/telegram-acp/README.md`" in rendered


async def test_format_activity_block_preserves_thinking_inline_code():
    block = AgentActivityBlock(
        kind="think",
        title="",
        status="completed",
        text="Checking `README.md` and `docs/index.md`.",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "`README.md`" in rendered
    assert "`docs/index.md`" in rendered


async def test_format_activity_block_execute_wraps_command_as_fenced_code_block():
    block = AgentActivityBlock(kind="execute", title="Run git diff -- README.md docs/index.md", status="in_progress")
    rendered = TelegramBridge._format_activity_block(block)
    assert "⚙️ Running" in rendered
    assert "```\ngit diff -- README.md docs/index.md\n```" in rendered


async def test_format_activity_block_execute_multiline_command_uses_fenced_code_block():
    block = AgentActivityBlock(
        kind="execute",
        title="Run git diff -- README.md \\\n  docs/index.md",
        status="in_progress",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "```\ngit diff -- README.md \\\n  docs/index.md\n```" in rendered


async def test_format_activity_block_execute_long_command_uses_fenced_code_block():
    command = (
        "gh api repos/mgaitan/telegram-acp-bot/pulls/67/comments -X POST "
        "-F in_reply_to=2889504154 -f body='Implemented in 55881c9 with detailed context text'"
    )
    block = AgentActivityBlock(kind="execute", title=f"Run {command}", status="in_progress")

    rendered = TelegramBridge._format_activity_block(block)

    assert f"```\n{command}\n```" in rendered


async def test_format_activity_block_execute_command_with_backticks_uses_fenced_code_block():
    command = "gh api -f 'body=implemented in 55881c9. `path` and ACP_TELEGRAM_CHANNEL_ALLOW_PATH'"
    block = AgentActivityBlock(kind="execute", title=f"Run {command}", status="in_progress")

    rendered = TelegramBridge._format_activity_block(block)

    assert f"```\n{command}\n```" in rendered
    assert "\\_" not in rendered


async def test_format_activity_block_execute_command_with_triple_backticks_uses_longer_fence():
    command = "gh api -f 'body=markdown ```code``` example'"
    block = AgentActivityBlock(kind="execute", title=f"Run {command}", status="in_progress")

    rendered = TelegramBridge._format_activity_block(block)

    assert f"````\n{command}\n````" in rendered


async def test_format_activity_block_execute_preserves_escaped_backticks_and_underscores():
    command = r"gh api -f 'body=implemented in 55881c9. \`path\` and ACP_TELEGRAM_CHANNEL_ALLOW_PATH'"
    block = AgentActivityBlock(kind="execute", title=f"Run {command}", status="in_progress")

    rendered = TelegramBridge._format_activity_block(block)

    assert f"```\n{command}\n```" in rendered
    assert r"\`path\`" in rendered
    assert r"\\`path\\`" not in rendered
    assert "ACP_TELEGRAM_CHANNEL_ALLOW_PATH" in rendered


async def test_format_activity_block_search_uses_web_label_when_url_present():
    block = AgentActivityBlock(
        kind="search",
        title='Query: "telegram acp"',
        status="completed",
        text="URL: https://agentclientprotocol.com/",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "*🌐 Searching web*" in rendered


async def test_format_activity_block_search_uses_project_label_for_local_markers():
    block = AgentActivityBlock(
        kind="search",
        title='Query project for "ACP_TELEGRAM_CHANNEL_ALLOW_PATH"',
        status="completed",
        text="ripgrep in workspace",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "*🔎 Querying project*" in rendered


async def test_format_activity_block_search_defaults_to_neutral_querying_label():
    block = AgentActivityBlock(
        kind="search",
        title='Query: "send now"',
        status="completed",
        text="",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "*🔎 Querying*" in rendered


async def test_format_permission_tool_title_empty_returns_empty():
    assert TelegramBridge._format_permission_tool_title("   ") == ""


async def test_format_permission_tool_title_non_run_keeps_title():
    assert TelegramBridge._format_permission_tool_title("Read README.md") == "Read README.md"


async def test_format_activity_block_execute_multiple_run_segments_use_consecutive_fenced_blocks():
    block = AgentActivityBlock(
        kind="execute",
        title="Run which ffmpeg, Run ffmpeg -y -f x11grab -i :0.0 -frames:v 1 /tmp/screenshot-ffmpeg.png",
        status="in_progress",
    )
    rendered = TelegramBridge._format_activity_block(block)
    expected = "```\nwhich ffmpeg\n```\n```\nffmpeg -y -f x11grab -i :0.0 -frames:v 1 /tmp/screenshot-ffmpeg.png\n```"
    assert expected in rendered


async def test_split_execute_commands_keeps_original_on_empty_segments():
    command = "which ffmpeg, Run "
    assert TelegramBridge._split_execute_commands(command) == [command]


async def test_format_fenced_code_without_language_uses_plain_fence():
    rendered = TelegramBridge._format_fenced_code("echo ok")
    assert rendered == "```\necho ok\n```"


async def test_send_helpers_with_no_message():
    update = make_update(with_message=False)
    image = ImagePayload(data_base64=base64.b64encode(b"img").decode("ascii"), mime_type="image/jpeg")
    file_payload = FilePayload(name="out.txt", text_content="content")

    await TelegramBridge._send_image(update, image)
    await TelegramBridge._send_file(update, file_payload)


async def test_reply_activity_block_with_no_message_is_noop():
    update = make_update(with_message=False)
    block = AgentActivityBlock(kind="think", title="t", status="completed", text="x")
    await TelegramBridge._reply_activity_block(update, block)


async def test_reply_activity_block_failed_status_appends_failed_marker():
    update = make_update()
    assert update.message is not None
    block = AgentActivityBlock(kind="other", title="Run command", status="failed", text="boom")

    await TelegramBridge._reply_activity_block(update, block)

    assert update.message.replies[-1].endswith("Failed")


async def test_send_file_with_empty_payload():
    update = make_update()
    assert update.message is not None
    payload = FilePayload(name="empty.bin")
    await TelegramBridge._send_file(update, payload)
    assert len(update.message.documents) == 1


async def test_on_permission_request_sends_buttons():
    bridge = make_bridge()
    dummy_bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=dummy_bot))

    request = PermissionRequest(
        chat_id=TEST_CHAT_ID,
        request_id="abc123",
        tool_title="Run ls",
        tool_call_id="call-1",
        available_actions=("always", "once", "deny"),
    )
    await bridge.on_permission_request(request)

    assert len(dummy_bot.sent_messages) == 1
    payload = dummy_bot.sent_messages[0]
    assert payload["chat_id"] == TEST_CHAT_ID
    assert cast(str, payload["text"]).startswith("⚠️ Permission required")
    assert cast(str, payload["text"]).endswith("ls")
    assert "parse_mode" not in payload
    assert "entities" in payload
    markup = payload["reply_markup"]
    assert markup is not None


async def test_on_permission_request_formats_multiline_run_as_code_block():
    bridge = make_bridge()
    dummy_bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=dummy_bot))

    request = PermissionRequest(
        chat_id=TEST_CHAT_ID,
        request_id="abc-multi",
        tool_title="Run git diff -- README.md \\\n  docs/index.md",
        tool_call_id="call-multi",
        available_actions=("always", "once", "deny"),
    )
    await bridge.on_permission_request(request)

    assert len(dummy_bot.sent_messages) == 1
    payload = dummy_bot.sent_messages[0]
    assert cast(str, payload["text"]).endswith("git diff -- README.md \\\n  docs/index.md")
    assert "parse_mode" not in payload
    assert "entities" in payload


async def test_on_permission_request_markdown_fallback_uses_plain_text():
    bridge = make_bridge()
    failing_bot = FailingMarkdownBot()
    bridge._app = cast(Application, SimpleNamespace(bot=failing_bot))

    request = PermissionRequest(
        chat_id=TEST_CHAT_ID,
        request_id="abc-fallback",
        tool_title="Run ls",
        tool_call_id="call-fallback",
        available_actions=("always", "once", "deny"),
    )
    await bridge.on_permission_request(request)

    assert len(failing_bot.sent_messages) == 1
    payload = failing_bot.sent_messages[0]
    assert payload["text"] == "⚠️ Permission required\n\nls"
    assert "parse_mode" not in payload
    assert "entities" not in payload


async def test_on_permission_request_without_app_is_noop():
    bridge = make_bridge()
    request = PermissionRequest(
        chat_id=TEST_CHAT_ID,
        request_id="noop",
        tool_title="Run ls",
        tool_call_id="call-noop",
        available_actions=("once", "deny"),
    )
    await bridge.on_permission_request(request)


async def test_on_permission_callback_accepts_action():
    class PermissionService:
        def set_permission_request_handler(self, handler):
            del handler

        async def respond_permission_request(self, *, chat_id: int, request_id: str, action: str) -> bool:
            assert chat_id == TEST_CHAT_ID
            assert request_id == "req1"
            assert action == "once"
            return True

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, PermissionService()),
    )
    callback = DummyCallbackQuery("perm|req1|once")
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
        callback_query=callback,
        message=None,
    )

    await bridge.on_permission_callback(cast(Update, update), make_context())
    assert callback.answers[-1] == "Approved this time."
    assert callback.edited_text is not None
    assert "Permission required" in callback.edited_text
    assert "Decision: Approved this time." in callback.edited_text


async def test_on_permission_callback_invalid_cases():
    bridge = make_bridge()
    update_no_query = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=None,
            message=None,
        ),
    )
    await bridge.on_permission_callback(update_no_query, make_context())

    callback = DummyCallbackQuery("invalid")
    update_invalid = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_permission_callback(update_invalid, make_context())
    assert callback.answers[-1] == "Invalid action."

    callback_bad_action = DummyCallbackQuery("perm|req1|weird")
    update_bad_action = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_bad_action,
            message=None,
        ),
    )
    await bridge.on_permission_callback(update_bad_action, make_context())
    assert callback_bad_action.answers[-1] == "Invalid action."

    callback_missing_chat = DummyCallbackQuery("perm|req1|once")
    callback_missing_chat.message = None
    update_missing_chat = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=None,
            callback_query=callback_missing_chat,
            message=None,
        ),
    )
    await bridge.on_permission_callback(update_missing_chat, make_context())
    assert callback_missing_chat.answers[-1] == "Missing chat."


async def test_on_permission_callback_access_denied():
    bridge = make_bridge(allowed_ids={9})
    callback = DummyCallbackQuery("perm|req1|deny")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_permission_callback(update, make_context())
    assert callback.answers[-1] == "Access denied."


async def test_on_permission_callback_expired_request():
    class ExpiredService:
        def set_permission_request_handler(self, handler):
            del handler

        async def respond_permission_request(self, *, chat_id: int, request_id: str, action: str) -> bool:
            del chat_id, request_id, action
            return False

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ExpiredService()),
    )
    callback = DummyCallbackQuery("perm|req1|deny")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_permission_callback(update, make_context())
    assert callback.answers[-1] == "Request expired."


async def test_on_permission_callback_fallback_to_clear_markup_on_edit_error():
    class PermissionService:
        def set_permission_request_handler(self, handler):
            del handler

        async def respond_permission_request(self, *, chat_id: int, request_id: str, action: str) -> bool:
            del chat_id, request_id, action
            return True

    class FailingEditCallbackQuery(DummyCallbackQuery):
        async def edit_message_text(self, text: str) -> None:
            del text
            raise MarkdownFailureError

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, PermissionService()),
    )
    callback = FailingEditCallbackQuery("perm|req1|deny")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_permission_callback(update, make_context())
    assert callback.answers[-1] == "Denied."
    assert callback.reply_markup_cleared


async def test_on_permission_callback_uses_query_message_chat_when_effective_chat_missing():
    class PermissionService:
        def set_permission_request_handler(self, handler):
            del handler

        async def respond_permission_request(self, *, chat_id: int, request_id: str, action: str) -> bool:
            assert chat_id == TEST_CHAT_ID
            assert request_id == "req-chat-fallback"
            assert action == "once"
            return True

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, PermissionService()),
    )
    callback = DummyCallbackQuery("perm|req-chat-fallback|once")
    callback.message = SimpleNamespace(text="Permission required\nRun ls", chat=SimpleNamespace(id=TEST_CHAT_ID))
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=None,
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_permission_callback(update, make_context())
    assert callback.answers[-1] == "Approved this time."
    assert callback.edited_text is not None
    assert "Decision: Approved this time." in callback.edited_text


async def test_on_permission_callback_handles_unexpected_exception():
    class FailingService:
        def set_permission_request_handler(self, handler):
            del handler

        async def respond_permission_request(self, *, chat_id: int, request_id: str, action: str) -> bool:
            del chat_id, request_id, action
            raise RuntimeError("boom")

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, FailingService()),
    )
    callback = DummyCallbackQuery("perm|req1|once")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_permission_callback(update, make_context())
    assert callback.answers[-1] == "Permission action failed."


async def test_on_resume_callback_invalid_cases():
    bridge = make_bridge()
    update_no_query = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=None,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_no_query, make_context())

    callback_invalid = DummyCallbackQuery("resume")
    update_invalid = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_invalid,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_invalid, make_context())
    assert callback_invalid.answers[-1] == "Invalid selection."

    callback_non_digit = DummyCallbackQuery("resume|x")
    update_non_digit = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_non_digit,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_non_digit, make_context())
    assert callback_non_digit.answers[-1] == "Invalid selection."


async def test_on_resume_callback_selection_expired_and_missing_chat():
    bridge = make_bridge()

    callback_expired = DummyCallbackQuery("resume|0")
    update_expired = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_expired,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_expired, make_context())
    assert callback_expired.answers[-1] == "Selection expired."

    callback_missing_chat = DummyCallbackQuery("resume|0")
    callback_missing_chat.message = None
    update_missing_chat = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=None,
            callback_query=callback_missing_chat,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_missing_chat, make_context())
    assert callback_missing_chat.answers[-1] == "Missing chat."


async def test_on_resume_callback_access_denied_and_invalid_index():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[999], workspace="."),
        agent_service=cast(AgentService, service),
    )
    callback_denied = DummyCallbackQuery("resume|0")
    update_denied = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_denied,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_denied, make_context())
    assert callback_denied.answers[-1] == "Access denied."

    bridge_ok = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    bridge_ok._pending_resume_choices_by_chat[TEST_CHAT_ID] = service.items
    callback_invalid_index = DummyCallbackQuery("resume|99")
    update_invalid_index = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_invalid_index,
            message=None,
        ),
    )
    await bridge_ok.on_resume_callback(update_invalid_index, make_context())
    assert callback_invalid_index.answers[-1] == "Invalid selection."


async def test_on_resume_callback_success_and_failure_paths():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    candidates = service.items
    bridge._pending_resume_choices_by_chat[TEST_CHAT_ID] = candidates
    callback = DummyCallbackQuery("resume|1")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=DummyMessage("trigger"),
        ),
    )
    await bridge.on_resume_callback(update, make_context())
    assert callback.answers[-1] == "Session resumed."
    assert callback.edited_text is not None
    assert "Resumed session: s-resume-2" in callback.edited_text
    assert "Workspace: /tmp/ws2" in callback.edited_text
    assert "Title: Second session" in callback.edited_text
    assert TEST_CHAT_ID not in bridge._pending_resume_choices_by_chat

    service.fail_load = True
    bridge._pending_resume_choices_by_chat[TEST_CHAT_ID] = candidates
    callback_fail = DummyCallbackQuery("resume|0")
    update_fail = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_fail,
            message=DummyMessage("trigger"),
        ),
    )
    await bridge.on_resume_callback(update_fail, make_context())
    assert callback_fail.answers[-1] == "Failed to resume."
    assert TEST_CHAT_ID in bridge._pending_resume_choices_by_chat


async def test_on_resume_callback_fallback_to_clear_markup_on_edit_error():
    class FailingEditCallbackQuery(DummyCallbackQuery):
        async def edit_message_text(self, text: str) -> None:
            del text
            raise MarkdownFailureError

    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    bridge._pending_resume_choices_by_chat[TEST_CHAT_ID] = service.items
    callback = FailingEditCallbackQuery("resume|0")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=DummyMessage("trigger"),
        ),
    )

    await bridge.on_resume_callback(update, make_context())
    assert callback.answers[-1] == "Session resumed."
    assert callback.reply_markup_cleared


async def test_on_resume_callback_uses_query_message_chat_when_effective_chat_missing():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    bridge._pending_resume_choices_by_chat[TEST_CHAT_ID] = service.items
    callback = DummyCallbackQuery("resume|0")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=None,
            callback_query=callback,
            message=DummyMessage("trigger"),
        ),
    )
    await bridge.on_resume_callback(update, make_context())
    assert callback.answers[-1] == "Session resumed."


async def test_cancel_stop_clear_without_session():
    bridge = make_bridge()
    update = make_update()
    context = make_context()

    await bridge.cancel(update, context)
    await bridge.stop(update, context)
    await bridge.clear(update, context)

    assert update.message is not None
    assert update.message.replies == [
        "No active session. Use /new first.",
        "No active session. Use /new first.",
        "No active session. Use /new first.",
    ]


async def test_format_activity_block_read_with_absolute_path_keeps_absolute():
    block = AgentActivityBlock(kind="read", title="Read /tmp/absolute.txt", status="completed")
    rendered = TelegramBridge._format_activity_block(block, workspace=Path("/tmp/ws"))
    assert "`/tmp/absolute.txt`" in rendered


async def test_format_read_path_empty_value_returns_empty_text():
    rendered = TelegramBridge._format_read_path("   ", workspace=Path("/tmp/ws"))
    assert rendered == ""


async def test_escape_markdown_preserving_code_escapes_special_chars_outside_code():
    rendered = TelegramBridge._escape_markdown_preserving_code("\\ _ * [ `code_[x]`")
    assert rendered == "\\\\ \\_ \\* \\[ `code_[x]`"


async def test_cancel_stop_clear_with_session():
    bridge = make_bridge()
    update = make_update()

    await bridge.new_session(update, make_context())
    await bridge.cancel(update, make_context())
    await bridge.stop(update, make_context())
    await bridge.clear(update, make_context())

    assert update.message is not None
    assert "Session started:" in update.message.replies[0]
    assert update.message.replies[1:] == [
        "Cancelled current operation.",
        "Stopped current session.",
        "No active session. Use /new first.",
    ]


async def test_clear_with_session():
    bridge = make_bridge()
    update = make_update()

    await bridge.new_session(update, make_context())
    await bridge.clear(update, make_context())

    assert update.message is not None
    assert "Session started:" in update.message.replies[0]
    assert update.message.replies[1] == "Cleared current session."


async def test_on_text_ignores_empty_message():
    bridge = make_bridge()
    update = make_update(text=None)
    context = make_context()

    await bridge.on_message(update, context)

    assert update.message is not None
    assert update.message.replies == []
    assert context.bot.actions == []


async def test_reply_with_no_message_object():
    bridge = make_bridge()
    update = make_update(with_message=False)

    await bridge.help(update, make_context())


async def test_reply_agent_with_no_message_object():
    update = make_update(with_message=False)
    await TelegramBridge._reply_agent(update, "x")


async def test_reply_agent_uses_entities_split_flow(monkeypatch: pytest.MonkeyPatch):
    update = make_update()
    assert update.message is not None

    entity = bot_module.MarkdownMessageEntity(type="bold", offset=0, length=5)
    monkeypatch.setattr(bot_module, "convert", lambda text: (text, [entity]))
    monkeypatch.setattr(
        bot_module,
        "split_entities",
        lambda text, entities, max_utf16_len: [("hello ", entities), ("world", [])],
    )

    await TelegramBridge._reply_agent(update, "hello world")

    assert update.message.replies == ["hello ", "world"]
    assert "entities" in update.message.reply_kwargs[0]
    assert "parse_mode" not in update.message.reply_kwargs[0]
    assert update.message.reply_kwargs[1] == {}


async def test_reply_agent_falls_back_to_plain_text_on_convert_error(
    monkeypatch: pytest.MonkeyPatch,
):
    update = make_update()
    assert update.message is not None

    def boom(_: str):
        raise RuntimeError

    monkeypatch.setattr(bot_module, "convert", boom)

    await TelegramBridge._reply_agent(update, "*x*")

    assert update.message.reply_kwargs[-1] == {}
    assert update.message.replies[-1] == "*x*"


async def test_reply_falls_back_to_plain_when_entity_send_fails():
    update = make_update()
    assert update.message is not None
    update.message.fail_entities = True

    await TelegramBridge._reply(update, "*x*")

    assert "entities" in update.message.reply_kwargs[-2]
    assert update.message.reply_kwargs[-1] == {}
    assert update.message.replies[-1] == "x"


async def test_reply_agent_falls_back_to_plain_when_convert_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    update = make_update()
    assert update.message is not None

    def boom(_: str):
        raise RuntimeError

    monkeypatch.setattr(bot_module, "convert", boom)

    await TelegramBridge._reply_agent(update, "*x*")

    assert update.message.reply_kwargs[-1] == {}
    assert update.message.replies[-1] == "*x*"


async def test_send_markdown_to_chat_falls_back_to_plain_when_entity_send_fails():
    bridge = make_bridge()
    failing_bot = FailingMarkdownBot()
    bridge._app = cast(Application, SimpleNamespace(bot=failing_bot))

    await TelegramBridge._send_markdown_to_chat(
        bot=cast(bot_module.Bot, failing_bot), chat_id=TEST_CHAT_ID, text="*bold*"
    )

    assert len(failing_bot.sent_messages) == 1
    payload = failing_bot.sent_messages[0]
    assert payload["text"] == "bold"
    assert "entities" not in payload


async def test_send_markdown_to_chat_without_entities_omits_entities_kwarg():
    dummy_bot = DummyBot()

    await TelegramBridge._send_markdown_to_chat(
        bot=cast(bot_module.Bot, dummy_bot),
        chat_id=TEST_CHAT_ID,
        text="plain text",
    )

    assert len(dummy_bot.sent_messages) == 1
    payload = dummy_bot.sent_messages[0]
    assert payload["text"] == "plain text"
    assert "entities" not in payload


async def test_send_markdown_to_chat_falls_back_to_plain_when_convert_fails(monkeypatch: pytest.MonkeyPatch):
    dummy_bot = DummyBot()

    def boom(_: str):
        raise ValueError

    monkeypatch.setattr(bot_module, "convert", boom)

    await TelegramBridge._send_markdown_to_chat(
        bot=cast(bot_module.Bot, dummy_bot),
        chat_id=TEST_CHAT_ID,
        text="*bold*",
        reply_markup=InlineKeyboardMarkup([]),
    )

    assert len(dummy_bot.sent_messages) == 1
    payload = dummy_bot.sent_messages[0]
    assert payload["text"] == "*bold*"
    assert payload["reply_markup"] is not None


async def test_on_text_ignores_when_message_is_missing():
    bridge = make_bridge()
    update = make_update(with_message=False)
    context = make_context()

    await bridge.on_message(update, context)
    assert context.bot.actions == []


async def test_chat_id_without_chat_raises():
    update = cast(Update, SimpleNamespace(effective_chat=None))
    with pytest.raises(ChatRequiredError):
        TelegramBridge._chat_id(update)


async def test_build_application_installs_handlers():
    bridge = make_bridge()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")

    app = build_application(config, bridge)
    assert app.handlers
    assert app.update_processor.max_concurrent_updates > 1


async def test_run_polling(monkeypatch):
    calls: list[object] = []

    class DummyApp:
        def run_polling(self, *, allowed_updates):
            calls.append(allowed_updates)

    def fake_build_application(config, bridge):
        del config, bridge
        return DummyApp()

    monkeypatch.setattr(bot_module, "build_application", fake_build_application)

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = make_bridge()
    assert run_polling(config, bridge) == 0
    assert len(calls) == 1


async def test_run_polling_returns_restart_exit_code(monkeypatch):
    class DummyApp:
        def run_polling(self, *, allowed_updates):
            del allowed_updates

    def fake_build_application(config, bridge):
        del config
        bridge._restart_requested = True
        return DummyApp()

    monkeypatch.setattr(bot_module, "build_application", fake_build_application)

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = make_bridge()
    assert run_polling(config, bridge) == RESTART_EXIT_CODE


# ---------------------------------------------------------------------------
# Busy-state tests
# ---------------------------------------------------------------------------


class BlockingService:
    """Service whose prompt blocks until `release()` is called, for busy-state tests."""

    def __init__(self) -> None:
        self._workspace: Path | None = None
        self._prompt_started = asyncio.Event()
        self._prompt_gate = asyncio.Event()
        self.cancelled = False
        self.prompts: list[str] = []

    async def new_session(self, *, chat_id: int, workspace: Path) -> str:
        del chat_id
        self._workspace = workspace
        return "s-blocking"

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()) -> AgentReply:
        del chat_id, images, files
        self.prompts.append(text)
        self._prompt_started.set()
        await self._prompt_gate.wait()
        return AgentReply(text=f"done:{text}")

    def get_workspace(self, *, chat_id: int) -> Path | None:
        del chat_id
        return self._workspace

    async def cancel(self, *, chat_id: int) -> bool:
        del chat_id
        self.cancelled = True
        self._prompt_gate.set()
        return True

    async def stop(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    async def clear(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    def get_permission_policy(self, *, chat_id: int):
        del chat_id

    async def set_session_permission_mode(self, *, chat_id: int, mode):
        del chat_id, mode
        return False

    async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
        del chat_id, enabled
        return False

    def release(self) -> None:
        self._prompt_gate.set()


class FailingCancelService:
    def __init__(self) -> None:
        self._workspace: Path | None = Path(".")

    async def new_session(self, *, chat_id: int, workspace: Path) -> str:
        del chat_id
        self._workspace = workspace
        return "s-fail"

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()) -> AgentReply:
        del chat_id, text, images, files
        return AgentReply(text="ok")

    def get_workspace(self, *, chat_id: int) -> Path | None:
        del chat_id
        return self._workspace

    async def cancel(self, *, chat_id: int) -> bool:
        del chat_id
        raise DummyCancelBoomError

    async def stop(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    async def clear(self, *, chat_id: int) -> bool:
        del chat_id
        return False

    def get_permission_policy(self, *, chat_id: int):
        del chat_id

    async def set_session_permission_mode(self, *, chat_id: int, mode):
        del chat_id, mode
        return False

    async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
        del chat_id, enabled
        return False


async def test_on_message_while_busy_shows_send_now_button():
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first", message_id=11)
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second", message_id=QUEUED_MESSAGE_ID)
    context = make_context(application=SimpleNamespace(bot=bot))

    # Start first message - it will block
    task_one = asyncio.create_task(bridge.on_message(update_one, context))

    # Wait until first prompt is actually running
    await service._prompt_started.wait()

    # Send second message while busy
    await bridge.on_message(update_two, context)

    # The second message should be queued and a "Send now" button should appear
    assert len(bot.sent_messages) == 1
    busy_msg = bot.sent_messages[0]
    assert busy_msg["chat_id"] == TEST_CHAT_ID
    assert busy_msg["reply_to_message_id"] == QUEUED_MESSAGE_ID
    assert "queued" in cast(str, busy_msg["text"]).lower()
    markup = cast(InlineKeyboardMarkup, busy_msg["reply_markup"])
    assert markup is not None
    button = markup.inline_keyboard[0][0]
    assert button.text == "Send now"
    assert button.callback_data is not None
    assert cast(str, button.callback_data).startswith(f"{BUSY_CALLBACK_PREFIX}|")

    # Finish first task
    service.release()
    await task_one


async def test_on_message_queued_runs_automatically_and_button_is_removed():
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    await bridge.on_message(update_two, context)

    # Notify button was shown with message_id=1
    assert bot.sent_messages[0]["reply_markup"] is not None

    # Release first prompt; pump loop should clear button then process second
    service.release()
    await task_one

    # Button should have been removed via edit_message_reply_markup
    assert any(e.get("message_id") == 1 for e in bot.edited_reply_markups)
    # Both updates should have received replies
    assert update_one.message is not None
    assert update_two.message is not None
    assert "done:first" in update_one.message.replies[-1]
    assert "done:second" in update_two.message.replies[-1]


async def test_on_busy_callback_send_now_cancels_and_queued_runs():
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    await bridge.on_message(update_two, context)

    # Grab the token from the "Send now" button
    markup = cast(InlineKeyboardMarkup, bot.sent_messages[0]["reply_markup"])
    token = cast(str, markup.inline_keyboard[0][0].callback_data).split("|", 1)[1]

    # User presses "Send now"
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update_cb, make_context())

    assert "Sending now" in callback.answers[-1]
    assert service.cancelled

    await task_one

    # Second message should have been processed
    assert update_two.message is not None
    assert "done:second" in update_two.message.replies[-1]


async def test_on_busy_callback_stale_token_is_rejected():
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    # Press button with an old/random token (simulates already-processed pending)
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|stale-token")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update_cb, make_context())

    assert callback.answers[-1] == "Already sent."
    assert callback.reply_markup_cleared

    service.release()
    await task_one


async def test_on_busy_callback_stale_after_auto_drain():
    """Queued message ran automatically; old button press returns 'Already sent.'"""
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()
    await bridge.on_message(update_two, context)

    markup = cast(InlineKeyboardMarkup, bot.sent_messages[0]["reply_markup"])
    token = cast(str, markup.inline_keyboard[0][0].callback_data).split("|", 1)[1]

    # Let first task finish naturally -> auto-drains second
    service.release()
    await task_one

    # Now the pending is gone; pressing button should get "Already sent."
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update_cb, make_context())

    assert callback.answers[-1] == "Already sent."


async def test_on_busy_callback_no_query_is_noop():
    bridge = make_bridge()
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=None,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update, make_context())


async def test_on_busy_callback_invalid_data_format():
    bridge = make_bridge()
    callback = DummyCallbackQuery("busy")  # missing token part
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update, make_context())
    assert callback.answers[-1] == "Invalid action."


async def test_on_busy_callback_access_denied():
    bridge = make_bridge(allowed_ids={99})
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|some-token")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update, make_context())
    assert callback.answers[-1] == "Access denied."


async def test_on_busy_callback_missing_chat():
    bridge = make_bridge()
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|some-token")
    callback.message = None
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=None,
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update, make_context())
    assert callback.answers[-1] == "Missing chat."


async def test_busy_queue_replaces_previous_pending_and_removes_old_button():
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    update_three = make_update(chat_id=TEST_CHAT_ID, text="third")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    await bridge.on_message(update_two, context)
    # First notify button sent (message_id=1)
    assert len(bot.sent_messages) == 1

    # Send third message while still busy - should replace second pending
    await bridge.on_message(update_three, context)

    # Old button (message_id=1) should be removed
    assert any(e.get("message_id") == 1 for e in bot.edited_reply_markups)
    # New button sent (message_id=2)
    assert len(bot.sent_messages) == EXPECTED_BUSY_NOTIFY_MESSAGES_AFTER_REPLACE

    service.release()
    await task_one

    # Only "third" should be processed (second was replaced)
    assert update_two.message is not None
    assert update_three.message is not None
    assert update_two.message.replies == []
    assert "done:third" in update_three.message.replies[-1]


async def test_on_busy_callback_edit_failure_is_handled_gracefully():
    """Edit of stale button may fail with TelegramError; that must not propagate."""

    class FailingEditOnStaleCallbackQuery(DummyCallbackQuery):
        async def edit_message_reply_markup(self, *, reply_markup: object | None = None) -> None:
            raise MarkdownFailureError

    bridge = make_bridge()
    callback = FailingEditOnStaleCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|stale")
    callback.message = SimpleNamespace(text="busy", chat=SimpleNamespace(id=TEST_CHAT_ID))
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    # No pending for TEST_CHAT_ID -> "Already sent." + edit attempt (which fails gracefully)
    await bridge.on_busy_callback(update, make_context())
    assert callback.answers[-1] == "Already sent."


async def test_on_busy_callback_send_now_edit_failure_is_handled():
    """TelegramError on edit after 'Sending now' must not propagate."""
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()
    await bridge.on_message(update_two, context)

    markup = cast(InlineKeyboardMarkup, bot.sent_messages[0]["reply_markup"])
    token = cast(str, markup.inline_keyboard[0][0].callback_data).split("|", 1)[1]

    class FailingEditAfterAnswer(DummyCallbackQuery):
        async def edit_message_reply_markup(self, *, reply_markup: object | None = None) -> None:
            if self.answers:
                raise MarkdownFailureError
            await super().edit_message_reply_markup(reply_markup=reply_markup)

    callback = FailingEditAfterAnswer(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update_cb, make_context())

    assert "Sending now" in callback.answers[-1]
    service.release()
    await task_one


async def test_queue_busy_prompt_edit_old_button_failure_is_handled():
    """TelegramError when removing old pending button must not propagate."""
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))

    class FailingEditBot(DummyBot):
        async def edit_message_reply_markup(self, **kwargs: object) -> None:
            raise MarkdownFailureError

    bot = FailingEditBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    update_three = make_update(chat_id=TEST_CHAT_ID, text="third")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    await bridge.on_message(update_two, context)
    # Now queue a third message - old button removal fails gracefully
    await bridge.on_message(update_three, context)

    service.release()
    await task_one


async def test_queue_busy_prompt_send_message_failure_is_handled():
    """TelegramError when sending the notify message must not propagate."""
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))

    class FailingSendBot(DummyBot):
        async def send_message(self, **kwargs: object) -> SimpleNamespace:
            raise MarkdownFailureError

    bot = FailingSendBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    # send_message raises - must not propagate
    await bridge.on_message(update_two, context)

    service.release()
    await task_one


async def test_clear_busy_button_telegram_error_is_swallowed():
    """TelegramError when clearing the busy button must not propagate."""
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))

    class FailingEditBot(DummyBot):
        async def edit_message_reply_markup(self, **kwargs: object) -> None:
            raise MarkdownFailureError

    bot = FailingEditBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    # Queue second message - notify_msg_id is None (send failed), but we manually set one
    await bridge.on_message(update_two, context)
    pending = bridge._pending_prompts_by_chat.get(TEST_CHAT_ID)
    if pending is not None:
        pending.notify_msg_id = 42  # force a non-None id so _clear_busy_button tries to edit

    # Release - _clear_busy_button will try to edit and fail
    service.release()
    await task_one  # must complete without exception


async def test_on_busy_callback_cancel_failure_answers_safely():
    """If cancel() raises, on_busy_callback answers 'Cancel failed.' and returns cleanly."""
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, FailingCancelService()))
    token = "test-token"
    dummy_update = make_update(chat_id=TEST_CHAT_ID, text="hi")
    prompt_input = _PromptInput(chat_id=TEST_CHAT_ID, text="hi", images=(), files=())
    bridge._pending_prompts_by_chat[TEST_CHAT_ID] = _PendingPrompt(
        prompt_input=prompt_input, update=cast(Update, dummy_update), token=token
    )

    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update_cb, make_context())
    assert callback.answers[-1] == "Cancel failed."

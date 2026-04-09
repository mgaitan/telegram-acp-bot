from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from telegram import Bot, CallbackQuery, InlineKeyboardMarkup, MessageEntity, Update
from telegram.error import TelegramError
from telegram.ext import AIORateLimiter, Application

from telegram_acp_bot.acp.echo_service import EchoAgentService
from telegram_acp_bot.acp.models import (
    ActivityMode,
    AgentActivityBlock,
    AgentOutputLimitExceededError,
    AgentReply,
    FilePayload,
    ImagePayload,
    PermissionRequest,
    ResumableSession,
)
from telegram_acp_bot.core.session_registry import SessionRegistry
from telegram_acp_bot.logging_context import LOG_TEXT_PREVIEW_MAX_CHARS, log_text_preview
from telegram_acp_bot.telegram import activity as activity_module
from telegram_acp_bot.telegram import app as app_module
from telegram_acp_bot.telegram import bot as bot_module
from telegram_acp_bot.telegram import bridge as bridge_module
from telegram_acp_bot.telegram.bot import (
    BUSY_CALLBACK_PREFIX,
    BUSY_SENT_TEXT,
    BUSY_STILL_QUEUED_TEXT,
    RESTART_EXIT_CODE,
    RESUME_KEYBOARD_MAX_ROWS,
    UNSUPPORTED_MESSAGE_TEXT,
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
EXPECTED_REPEATED_ACTIVITY_MESSAGES = 2
QUEUED_MESSAGE_ID = 22
COMPACT_STATUS_MSG_ID = 42


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
        voice: object | None = None,
        audio: object | None = None,
    ) -> None:
        self.message_id = message_id
        self.text = text
        self.caption = caption
        self.photo = list(photo) if photo is not None else []
        self.document = document
        self.voice = voice
        self.audio = audio
        self._next_reply_message_id = message_id + 1
        self.replies: list[str] = []
        self.reply_kwargs: list[dict[str, object]] = []
        self.fail_markdown = False
        self.fail_entities = False
        self.photos: list[object] = []
        self.documents: list[object] = []

    async def reply_text(self, text: str, **kwargs: object) -> SimpleNamespace:
        if self.fail_markdown and kwargs.get("parse_mode") is not None:
            self.reply_kwargs.append(kwargs)
            raise MarkdownFailureError
        if self.fail_entities and "entities" in kwargs:
            self.reply_kwargs.append(kwargs)
            raise MarkdownFailureError
        self.reply_kwargs.append(kwargs)
        self.replies.append(text)
        reply = SimpleNamespace(message_id=self._next_reply_message_id)
        self._next_reply_message_id += 1
        return reply

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
        self.sent_photos: list[dict[str, object]] = []
        self.sent_documents: list[dict[str, object]] = []
        self._next_message_id = 1
        self.edited_reply_markups: list[dict[str, object]] = []
        self.edited_messages: list[dict[str, object]] = []
        self.deleted_message_ids: list[tuple[int, int]] = []
        self.reactions: list[dict[str, object]] = []

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.actions.append((chat_id, action))

    async def get_file(self, file_id: str) -> DummyDownloadedFile:
        return DummyDownloadedFile(self.files[file_id])

    async def send_message(self, **kwargs: object) -> SimpleNamespace:
        self.sent_messages.append(kwargs)
        msg = SimpleNamespace(message_id=self._next_message_id)
        self._next_message_id += 1
        return msg

    async def send_photo(self, **kwargs: object) -> SimpleNamespace:
        self.sent_photos.append(kwargs)
        msg = SimpleNamespace(message_id=self._next_message_id)
        self._next_message_id += 1
        return msg

    async def send_document(self, **kwargs: object) -> SimpleNamespace:
        self.sent_documents.append(kwargs)
        msg = SimpleNamespace(message_id=self._next_message_id)
        self._next_message_id += 1
        return msg

    async def edit_message_reply_markup(self, **kwargs: object) -> None:
        self.edited_reply_markups.append(kwargs)

    async def edit_message_text(self, **kwargs: object) -> None:
        self.edited_messages.append(dict(kwargs))

    async def delete_message(self, **kwargs: object) -> None:
        chat_id = cast(int, kwargs.get("chat_id"))
        message_id = cast(int, kwargs.get("message_id"))
        self.deleted_message_ids.append((chat_id, message_id))

    async def set_message_reaction(self, **kwargs: object) -> None:
        self.reactions.append(dict(kwargs))


class FailingMarkdownBot(DummyBot):
    async def send_message(self, **kwargs: object) -> SimpleNamespace:
        if "entities" in kwargs:
            raise MarkdownFailureError
        return await super().send_message(**kwargs)


class FailingEditBot(DummyBot):
    """Always raises on edit_message_text (used to test compact fallback)."""

    async def edit_message_text(self, **kwargs: object) -> None:
        raise MarkdownFailureError


class EntityFailingEditBot(DummyBot):
    """Raises on edit_message_text only when entities are supplied."""

    async def edit_message_text(self, **kwargs: object) -> None:
        if "entities" in kwargs:
            raise MarkdownFailureError
        await super().edit_message_text(**kwargs)


class NotModifiedError(TelegramError):
    def __init__(self) -> None:
        super().__init__("Bad Request: message is not modified")


class NotModifiedAwareBot(DummyBot):
    """Raises the Telegram not-modified error for identical edits."""

    def __init__(self) -> None:
        super().__init__()
        self._message_state: dict[int, dict[str, object]] = {}

    async def send_message(self, **kwargs: object) -> SimpleNamespace:
        message = await super().send_message(**kwargs)
        self._message_state[message.message_id] = {
            "text": kwargs.get("text"),
            "entities": kwargs.get("entities"),
            "reply_markup": kwargs.get("reply_markup"),
        }
        return message

    async def edit_message_text(self, **kwargs: object) -> None:
        message_id = cast(int, kwargs.get("message_id"))
        next_state = {
            "text": kwargs.get("text"),
            "entities": kwargs.get("entities"),
            "reply_markup": kwargs.get("reply_markup"),
        }
        if self._message_state.get(message_id) == next_state:
            raise NotModifiedError
        await super().edit_message_text(**kwargs)
        self._message_state[message_id] = next_state


class DummyCallbackQuery:
    def __init__(self, data: str) -> None:
        self.data = data
        self.message = SimpleNamespace(text="Permission required\nRun ls", chat=SimpleNamespace(id=TEST_CHAT_ID))
        self.answers: list[str] = []
        self.reply_markup_cleared = False
        self.edited_text: str | None = None
        self.edited_kwargs: dict[str, object] = {}

    async def answer(self, text: str) -> None:
        self.answers.append(text)

    async def edit_message_reply_markup(self, *, reply_markup: object | None = None) -> None:
        self.reply_markup_cleared = reply_markup is None

    async def edit_message_text(self, text: str, **kwargs: object) -> None:
        self.edited_text = text
        self.edited_kwargs = kwargs


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


class PromptContextService:
    def __init__(self) -> None:
        self._workspace: Path | None = None
        self.prompt_message_context: list[tuple[str, int | None]] = []

    async def new_session(self, *, chat_id: int, workspace: Path) -> str:
        del chat_id
        self._workspace = workspace
        return "s-context"

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
        del chat_id, text, images, files
        return AgentReply(text="ok")

    async def set_prompt_message_context(self, *, session_id: str, message_id: int | None) -> None:
        self.prompt_message_context.append((session_id, message_id))

    def get_workspace(self, *, chat_id: int) -> Path | None:
        del chat_id
        return self._workspace

    def get_active_session_context(self, *, chat_id: int):
        del chat_id
        if self._workspace is None:
            return None
        return "s-context", self._workspace

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
    voice: object | None = None,
    audio: object | None = None,
    message_id: int = 1,
    with_message: bool = True,
):
    message = (
        DummyMessage(
            text,
            message_id=message_id,
            caption=caption,
            photo=photo,
            document=document,
            voice=voice,
            audio=audio,
        )
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


def make_compact_bridge() -> TelegramBridge:
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".", compact_activity=True)
    return TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))


def make_verbose_bridge() -> TelegramBridge:
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".", activity_mode="verbose")
    return TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))


__all__ = [
    "ACP_STDIO_LIMIT_ERROR",
    "BUSY_CALLBACK_PREFIX",
    "BUSY_SENT_TEXT",
    "BUSY_STILL_QUEUED_TEXT",
    "COMPACT_STATUS_MSG_ID",
    "EXPECTED_ACTIVITY_MESSAGES",
    "EXPECTED_BUSY_NOTIFY_MESSAGES_AFTER_REPLACE",
    "EXPECTED_OUTBOUND_DOCUMENTS",
    "EXPECTED_REPEATED_ACTIVITY_MESSAGES",
    "EXPECTED_TEXT_REPLIES_WITH_IMPLICIT_AND_EXPLICIT_SESSION",
    "LOG_TEXT_PREVIEW_MAX_CHARS",
    "QUEUED_MESSAGE_ID",
    "RESTART_EXIT_CODE",
    "RESUME_KEYBOARD_MAX_ROWS",
    "TEST_CHAT_ID",
    "UNSUPPORTED_MESSAGE_TEXT",
    "AIORateLimiter",
    "ActivityMode",
    "AgentActivityBlock",
    "AgentOutputLimitExceededError",
    "AgentReply",
    "AgentService",
    "Application",
    "Awaitable",
    "Bot",
    "Callable",
    "CallbackQuery",
    "ChatRequiredError",
    "ConcurrentImplicitSessionService",
    "DummyBot",
    "DummyCallbackQuery",
    "DummyCancelBoomError",
    "DummyDownloadedFile",
    "DummyListBoomError",
    "DummyLoadFailedError",
    "DummyMessage",
    "EchoAgentService",
    "EntityFailingEditBot",
    "FailingEditBot",
    "FailingImplicitService",
    "FailingMarkdownBot",
    "FilePayload",
    "ImagePayload",
    "ImplicitSessionServiceBase",
    "InlineKeyboardMarkup",
    "LiveActivityService",
    "MarkdownFailureError",
    "MessageEntity",
    "NotModifiedAwareBot",
    "NotModifiedError",
    "Path",
    "PermissionRequest",
    "PromptContextService",
    "PromptWithoutSessionImplicitService",
    "RecordingImplicitService",
    "ResumableSession",
    "ResumeService",
    "Sequence",
    "SessionRegistry",
    "SimpleNamespace",
    "TelegramBridge",
    "TelegramError",
    "Update",
    "_PendingPrompt",
    "_PromptInput",
    "activity_module",
    "app_module",
    "asyncio",
    "base64",
    "bot_module",
    "bridge_module",
    "build_application",
    "cast",
    "log_text_preview",
    "make_bridge",
    "make_compact_bridge",
    "make_config",
    "make_context",
    "make_update",
    "make_verbose_bridge",
    "pytest",
    "pytestmark",
    "run_polling",
]

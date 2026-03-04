from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from telegram import Update
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
    RESTART_EXIT_CODE,
    RESUME_KEYBOARD_MAX_ROWS,
    AgentService,
    ChatRequiredError,
    TelegramBridge,
    build_application,
    make_config,
    run_polling,
)

pytestmark = pytest.mark.asyncio

EXPECTED_OUTBOUND_DOCUMENTS = 2
TEST_CHAT_ID = 100
EXPECTED_ACTIVITY_MESSAGES = 3
ACP_STDIO_LIMIT_ERROR = "Separator is found, but chunk is longer than limit"


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


class DummyMessage:
    def __init__(
        self,
        text: str | None = None,
        *,
        caption: str | None = None,
        photo: list[object] | None = None,
        document: object | None = None,
    ) -> None:
        self.text = text
        self.caption = caption
        self.photo = photo or []
        self.document = document
        self.replies: list[str] = []
        self.reply_kwargs: list[dict[str, object]] = []
        self.fail_markdown = False
        self.photos: list[object] = []
        self.documents: list[object] = []

    async def reply_text(self, text: str, **kwargs: object) -> None:
        if self.fail_markdown and kwargs.get("parse_mode") is not None:
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

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.actions.append((chat_id, action))

    async def get_file(self, file_id: str) -> DummyDownloadedFile:
        return DummyDownloadedFile(self.files[file_id])

    async def send_message(self, **kwargs: object) -> None:
        self.sent_messages.append(kwargs)


class FailingMarkdownBot(DummyBot):
    async def send_message(self, **kwargs: object) -> None:
        if kwargs.get("parse_mode") is not None:
            raise MarkdownFailureError
        await super().send_message(**kwargs)


class DummyCallbackQuery:
    def __init__(self, data: str) -> None:
        self.data = data
        self.message = SimpleNamespace(text="Permission required for:\nRun ls", chat=SimpleNamespace(id=TEST_CHAT_ID))
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


def make_update(  # noqa: PLR0913
    *,
    user_id: int = 1,
    chat_id: int = 100,
    text: str | None = None,
    caption: str | None = None,
    photo: list[object] | None = None,
    document: object | None = None,
    with_message: bool = True,
):
    message = DummyMessage(text, caption=caption, photo=photo, document=document) if with_message else None
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
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
    assert "Use /new" in update.message.replies[0]
    assert "Commands:" in update.message.replies[1]
    assert "/cancel" in update.message.replies[1]
    assert "/restart" in update.message.replies[1]
    assert "/perm" not in update.message.replies[1]


async def test_restart_requests_app_stop():
    bridge = make_bridge()
    update = make_update(with_message=True)
    stop_calls: list[str] = []
    bridge._app = cast(Application, SimpleNamespace(stop_running=lambda: stop_calls.append("stop")))

    await bridge.restart(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["Restart requested. Re-launching process..."]
    assert stop_calls == ["stop"]


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
    assert "Use /new" in update.message.replies[0]


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


async def test_resume_session_with_workspace_arg_includes_workspace_in_message(tmp_path: Path):
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path)),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context(args=["/tmp/ws2"]))

    assert bot.sent_messages
    payload = bot.sent_messages[-1]
    assert "Pick a session to resume in" in cast(str, payload["text"])


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
    assert update.message.replies == ["Agent does not support ACP `session/list`."]


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
    assert f"Created workspace: `{created_path}`" in update.message.replies[0]


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
    assert update.message.replies[0] == "No active session. Use /new first."
    assert update.message.replies[-1].endswith("hello")
    assert context.bot.actions == [(100, "typing"), (100, "typing")]
    assert update.message.reply_kwargs[-1] == {"parse_mode": "Markdown"}


async def test_on_text_markdown_fallback_to_plain():
    bridge = make_bridge()
    update = make_update(text="hello")
    assert update.message is not None
    update.message.fail_markdown = True
    context = make_context()

    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message.replies[-1].endswith("hello")
    assert update.message.reply_kwargs[-2] == {"parse_mode": "Markdown"}
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
    assert "*💡 Thinking*" in update.message.replies[0]
    assert "Draft plan" not in update.message.replies[0]
    assert "*⚙️ Tool call*" in update.message.replies[1]
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
    assert "*💡 Thinking*" in cast(str, context.bot.sent_messages[0]["text"])


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


async def test_format_activity_block_execute_wraps_command_as_inline_code():
    block = AgentActivityBlock(kind="execute", title="Run git diff -- README.md docs/index.md", status="in_progress")
    rendered = TelegramBridge._format_activity_block(block)
    assert "Run `git diff -- README.md docs/index.md`" in rendered


async def test_format_activity_block_execute_multiline_command_uses_fenced_code_block():
    block = AgentActivityBlock(
        kind="execute",
        title="Run git diff -- README.md \\\n  docs/index.md",
        status="in_progress",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "Run\n```sh\ngit diff -- README.md \\\n  docs/index.md\n```" in rendered


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


async def test_reply_activity_block_failed_status_with_markdown_fallback():
    update = make_update()
    assert update.message is not None
    update.message.fail_markdown = True
    block = AgentActivityBlock(kind="other", title="Run command", status="failed", text="boom")

    await TelegramBridge._reply_activity_block(update, block)

    assert update.message.replies[-1].endswith("_Failed_")
    assert update.message.reply_kwargs[-2] == {"parse_mode": "Markdown"}
    assert update.message.reply_kwargs[-1] == {}


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
    assert "Permission required for:" in cast(str, payload["text"])
    markup = payload["reply_markup"]
    assert markup is not None


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
    assert "Permission required for:" in callback.edited_text
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
    callback.message = SimpleNamespace(text="Permission required for:\nRun ls", chat=SimpleNamespace(id=TEST_CHAT_ID))
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

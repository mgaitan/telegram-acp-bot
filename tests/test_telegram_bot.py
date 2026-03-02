from __future__ import annotations

import asyncio
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
from telegram_acp_bot.acp_app.models import AgentActivityBlock, AgentReply, FilePayload, ImagePayload, PermissionRequest
from telegram_acp_bot.core.session_registry import SessionRegistry
from telegram_acp_bot.telegram import bot as bot_module
from telegram_acp_bot.telegram.bot import (
    AgentService,
    ChatRequiredError,
    TelegramBridge,
    build_application,
    make_config,
    run_polling,
)

EXPECTED_OUTBOUND_DOCUMENTS = 2
TEST_CHAT_ID = 100
EXPECTED_ACTIVITY_MESSAGES = 3
ACP_STDIO_LIMIT_ERROR = "Separator is found, but chunk is longer than limit"


class MarkdownFailureError(TelegramError):
    """Raised by test doubles to emulate Telegram markdown parse failure."""

    def __init__(self) -> None:
        super().__init__("bad markdown")


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


def make_context(*, args: list[str] | None = None):
    return SimpleNamespace(args=args or [], bot=DummyBot())


def make_bridge(*, allowed_ids: set[int] | None = None) -> TelegramBridge:
    config = make_config(token="TOKEN", allowed_user_ids=list(allowed_ids or set()), workspace=".")
    return TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))


def test_make_config():
    config = make_config(token="T", allowed_user_ids=[1, 2, 2], workspace="~/tmp")
    assert config.token == "T"
    assert config.allowed_user_ids == {1, 2}
    assert config.default_workspace.name == "tmp"


def test_workspace_from_relative_arg_uses_default_workspace():
    config = make_config(token="T", allowed_user_ids=[], workspace="/tmp/base")
    bridge = TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))

    workspace = bridge._workspace_from_args(["foo"])
    assert workspace == Path("/tmp/base/foo")


def test_start_and_help():
    bridge = make_bridge()
    update = make_update(with_message=True)
    context = make_context()

    asyncio.run(bridge.start(update, context))
    asyncio.run(bridge.help(update, context))

    assert update.message is not None
    assert "Use /new" in update.message.replies[0]
    assert "Commands:" in update.message.replies[1]
    assert "/cancel" in update.message.replies[1]
    assert "/perm" not in update.message.replies[1]


def test_access_denied():
    bridge = make_bridge(allowed_ids={99})
    update = make_update(user_id=1)
    context = make_context()

    asyncio.run(bridge.start(update, context))

    assert update.message is not None
    assert update.message.replies == ["Access denied for this bot."]


def test_access_allowed_with_allowlist():
    bridge = make_bridge(allowed_ids={1})
    update = make_update(user_id=1)
    context = make_context()

    asyncio.run(bridge.start(update, context))

    assert update.message is not None
    assert len(update.message.replies) == 1
    assert "Use /new" in update.message.replies[0]


def test_denied_paths_for_other_handlers():
    bridge = make_bridge(allowed_ids={42})
    update = make_update(user_id=7, text="hello")
    context = make_context()

    asyncio.run(bridge.help(update, context))
    asyncio.run(bridge.new_session(update, make_context(args=["/tmp"])))
    asyncio.run(bridge.session(update, context))
    asyncio.run(bridge.cancel(update, context))
    asyncio.run(bridge.stop(update, context))
    asyncio.run(bridge.clear(update, context))
    asyncio.run(bridge.on_message(update, context))

    assert update.message is not None
    assert update.message.replies == [
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
    ]


def test_new_session_and_session_command():
    bridge = make_bridge()
    update = make_update()

    asyncio.run(bridge.session(update, make_context()))
    asyncio.run(bridge.new_session(update, make_context(args=["/tmp"])))
    asyncio.run(bridge.session(update, make_context()))

    assert update.message is not None
    assert update.message.replies[0] == "No active session. Use /new first."
    assert "Session started:" in update.message.replies[1]
    assert "Active session workspace:" in update.message.replies[2]


def test_new_session_autocreates_relative_workspace_and_reports_it(tmp_path: Path):
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path))
    bridge = TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))
    update = make_update()

    asyncio.run(bridge.new_session(update, make_context(args=["myproj"])))

    created_path = tmp_path / "myproj"
    assert created_path.is_dir()
    assert update.message is not None
    assert "Session started:" in update.message.replies[0]
    assert f"Created workspace: `{created_path}`" in update.message.replies[0]


def test_new_session_reports_invalid_workspace():
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

    asyncio.run(bridge.new_session(update, make_context(args=["/missing"])))

    assert update.message is not None
    assert update.message.replies == ["Invalid workspace: /missing"]


def test_new_session_reports_process_stdio_error():
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

    asyncio.run(bridge.new_session(update, make_context(args=["/tmp"])))

    assert update.message is not None
    assert update.message.replies == ["Failed to start session: agent process did not expose stdio pipes."]


def test_new_session_reports_generic_error():
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

    asyncio.run(bridge.new_session(update, make_context(args=["/tmp"])))

    assert update.message is not None
    assert update.message.replies == ["Failed to start session: boom"]


def test_on_text_without_and_with_session():
    bridge = make_bridge()
    update = make_update(text="hello")
    context = make_context()

    asyncio.run(bridge.on_message(update, context))
    asyncio.run(bridge.new_session(update, make_context()))
    asyncio.run(bridge.on_message(update, context))

    assert update.message is not None
    assert update.message.replies[0] == "No active session. Use /new first."
    assert update.message.replies[-1].endswith("hello")
    assert context.bot.actions == [(100, "typing"), (100, "typing")]
    assert update.message.reply_kwargs[-1] == {"parse_mode": "Markdown"}


def test_on_text_markdown_fallback_to_plain():
    bridge = make_bridge()
    update = make_update(text="hello")
    assert update.message is not None
    update.message.fail_markdown = True
    context = make_context()

    asyncio.run(bridge.new_session(update, make_context()))
    asyncio.run(bridge.on_message(update, context))

    assert update.message.replies[-1].endswith("hello")
    assert update.message.reply_kwargs[-2] == {"parse_mode": "Markdown"}
    assert update.message.reply_kwargs[-1] == {}


def test_on_message_with_photo_attachment():
    bridge = make_bridge()
    photo = [SimpleNamespace(file_id="p1")]
    update = make_update(photo=photo)
    context = make_context()
    context.bot.files["p1"] = b"abc"

    asyncio.run(bridge.new_session(update, make_context()))
    asyncio.run(bridge.on_message(update, context))

    assert update.message is not None
    assert "images=1" in update.message.replies[-1]


def test_on_message_with_document_attachment():
    bridge = make_bridge()
    document = SimpleNamespace(file_id="d1", mime_type="text/plain", file_name="note.txt")
    update = make_update(document=document)
    context = make_context()
    context.bot.files["d1"] = b"hello from file"

    asyncio.run(bridge.new_session(update, make_context()))
    asyncio.run(bridge.on_message(update, context))

    assert update.message is not None
    assert "files=1" in update.message.replies[-1]


def test_on_message_with_binary_document_attachment():
    bridge = make_bridge()
    document = SimpleNamespace(file_id="bin-doc", mime_type="application/octet-stream", file_name="x.bin")
    update = make_update(document=document)
    context = make_context()
    context.bot.files["bin-doc"] = b"\xff\xfe"

    asyncio.run(bridge.new_session(update, make_context()))
    asyncio.run(bridge.on_message(update, context))

    assert update.message is not None
    assert "files=1" in update.message.replies[-1]


def test_on_message_with_image_document_attachment():
    bridge = make_bridge()
    document = SimpleNamespace(file_id="img-doc", mime_type="image/png", file_name="x.png")
    update = make_update(document=document)
    context = make_context()
    context.bot.files["img-doc"] = b"\x89PNG"

    asyncio.run(bridge.new_session(update, make_context()))
    asyncio.run(bridge.on_message(update, context))

    assert update.message is not None
    assert "images=1" in update.message.replies[-1]


def test_outbound_agent_attachments_are_sent():
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

    asyncio.run(bridge.on_message(update, make_context()))

    assert update.message is not None
    assert update.message.replies[-1] == "ok"
    assert len(update.message.photos) == 1
    assert len(update.message.documents) == EXPECTED_OUTBOUND_DOCUMENTS


def test_on_message_renders_activity_blocks_before_final_reply():
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

    asyncio.run(bridge.on_message(update, make_context()))

    assert update.message is not None
    assert len(update.message.replies) == EXPECTED_ACTIVITY_MESSAGES
    assert "*💡 Thinking*" in update.message.replies[0]
    assert "Draft plan" not in update.message.replies[0]
    assert "*⚙️ Tool call*" in update.message.replies[1]
    assert update.message.replies[2] == "Done."


def test_on_message_sends_live_activity_events_via_app_bot():
    service = LiveActivityService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    update = make_update(text="hello")
    context = make_context()
    bridge._app = cast(Application, SimpleNamespace(bot=context.bot))

    asyncio.run(bridge.on_message(update, context))

    assert update.message is not None
    assert update.message.replies[-1] == "Final response."
    assert context.bot.sent_messages
    assert "*💡 Thinking*" in cast(str, context.bot.sent_messages[0]["text"])


def test_on_message_skips_empty_final_text_reply():
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

    asyncio.run(bridge.on_message(update, context))

    assert update.message is not None
    assert update.message.replies == []


def test_on_message_reports_acp_stdio_limit_error():
    class LimitErrorService:
        async def new_session(self, *, chat_id: int, workspace):
            del workspace
            return f"s-{chat_id}"

        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del chat_id, text, images, files
            raise ValueError(ACP_STDIO_LIMIT_ERROR)

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

    asyncio.run(bridge.on_message(update, context))

    assert update.message is not None
    assert "Agent output exceeded ACP stdio limit." in update.message.replies[-1]


def test_on_message_reraises_unrelated_value_error():
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
        asyncio.run(bridge.on_message(update, context))


def test_on_activity_event_without_app_is_noop():
    bridge = make_bridge()
    block = AgentActivityBlock(kind="think", title="x", status="completed", text="y")
    asyncio.run(bridge.on_activity_event(TEST_CHAT_ID, block))


def test_on_activity_event_markdown_fallback():
    bridge = make_bridge()
    failing_bot = FailingMarkdownBot()
    bridge._app = cast(Application, SimpleNamespace(bot=failing_bot))
    block = AgentActivityBlock(kind="execute", title="Run cmd", status="in_progress", text="")

    asyncio.run(bridge.on_activity_event(TEST_CHAT_ID, block))

    assert failing_bot.sent_messages
    assert "parse_mode" not in failing_bot.sent_messages[-1]


def test_format_activity_block_read_escapes_markdown_and_removes_read_prefix():
    block = AgentActivityBlock(
        kind="read", title="Read test_telegram_bot.py", status="completed", text="Read test_telegram_bot.py"
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "*📖 Reading*" in rendered
    assert "test\\_telegram\\_bot.py" in rendered
    assert "\n\nRead test\\_telegram\\_bot.py" not in rendered


def test_format_activity_block_preserves_thinking_inline_code():
    block = AgentActivityBlock(
        kind="think",
        title="",
        status="completed",
        text="Checking `README.md` and `docs/index.md`.",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "`README.md`" in rendered
    assert "`docs/index.md`" in rendered


def test_format_activity_block_execute_wraps_command_as_inline_code():
    block = AgentActivityBlock(kind="execute", title="Run git diff -- README.md docs/index.md", status="in_progress")
    rendered = TelegramBridge._format_activity_block(block)
    assert "Run `git diff -- README.md docs/index.md`" in rendered


def test_send_helpers_with_no_message():
    update = make_update(with_message=False)
    image = ImagePayload(data_base64=base64.b64encode(b"img").decode("ascii"), mime_type="image/jpeg")
    file_payload = FilePayload(name="out.txt", text_content="content")

    asyncio.run(TelegramBridge._send_image(update, image))
    asyncio.run(TelegramBridge._send_file(update, file_payload))


def test_reply_activity_block_with_no_message_is_noop():
    update = make_update(with_message=False)
    block = AgentActivityBlock(kind="think", title="t", status="completed", text="x")
    asyncio.run(TelegramBridge._reply_activity_block(update, block))


def test_reply_activity_block_failed_status_with_markdown_fallback():
    update = make_update()
    assert update.message is not None
    update.message.fail_markdown = True
    block = AgentActivityBlock(kind="other", title="Run command", status="failed", text="boom")

    asyncio.run(TelegramBridge._reply_activity_block(update, block))

    assert update.message.replies[-1].endswith("_Failed_")
    assert update.message.reply_kwargs[-2] == {"parse_mode": "Markdown"}
    assert update.message.reply_kwargs[-1] == {}


def test_send_file_with_empty_payload():
    update = make_update()
    assert update.message is not None
    payload = FilePayload(name="empty.bin")
    asyncio.run(TelegramBridge._send_file(update, payload))
    assert len(update.message.documents) == 1


def test_on_permission_request_sends_buttons():
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
    asyncio.run(bridge.on_permission_request(request))

    assert len(dummy_bot.sent_messages) == 1
    payload = dummy_bot.sent_messages[0]
    assert payload["chat_id"] == TEST_CHAT_ID
    assert "Permission required for:" in cast(str, payload["text"])
    markup = payload["reply_markup"]
    assert markup is not None


def test_on_permission_request_without_app_is_noop():
    bridge = make_bridge()
    request = PermissionRequest(
        chat_id=TEST_CHAT_ID,
        request_id="noop",
        tool_title="Run ls",
        tool_call_id="call-noop",
        available_actions=("once", "deny"),
    )
    asyncio.run(bridge.on_permission_request(request))


def test_on_permission_callback_accepts_action():
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

    asyncio.run(bridge.on_permission_callback(cast(Update, update), make_context()))
    assert callback.answers[-1] == "Approved this time."
    assert callback.edited_text is not None
    assert "Permission required for:" in callback.edited_text
    assert "Decision: Approved this time." in callback.edited_text


def test_on_permission_callback_invalid_cases():
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
    asyncio.run(bridge.on_permission_callback(update_no_query, make_context()))

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
    asyncio.run(bridge.on_permission_callback(update_invalid, make_context()))
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
    asyncio.run(bridge.on_permission_callback(update_bad_action, make_context()))
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
    asyncio.run(bridge.on_permission_callback(update_missing_chat, make_context()))
    assert callback_missing_chat.answers[-1] == "Missing chat."


def test_on_permission_callback_access_denied():
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
    asyncio.run(bridge.on_permission_callback(update, make_context()))
    assert callback.answers[-1] == "Access denied."


def test_on_permission_callback_expired_request():
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

    asyncio.run(bridge.on_permission_callback(update, make_context()))
    assert callback.answers[-1] == "Request expired."


def test_on_permission_callback_fallback_to_clear_markup_on_edit_error():
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

    asyncio.run(bridge.on_permission_callback(update, make_context()))
    assert callback.answers[-1] == "Denied."
    assert callback.reply_markup_cleared


def test_on_permission_callback_uses_query_message_chat_when_effective_chat_missing():
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

    asyncio.run(bridge.on_permission_callback(update, make_context()))
    assert callback.answers[-1] == "Approved this time."
    assert callback.edited_text is not None
    assert "Decision: Approved this time." in callback.edited_text


def test_on_permission_callback_handles_unexpected_exception():
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

    asyncio.run(bridge.on_permission_callback(update, make_context()))
    assert callback.answers[-1] == "Permission action failed."


def test_cancel_stop_clear_without_session():
    bridge = make_bridge()
    update = make_update()
    context = make_context()

    asyncio.run(bridge.cancel(update, context))
    asyncio.run(bridge.stop(update, context))
    asyncio.run(bridge.clear(update, context))

    assert update.message is not None
    assert update.message.replies == [
        "No active session. Use /new first.",
        "No active session. Use /new first.",
        "No active session. Use /new first.",
    ]


def test_cancel_stop_clear_with_session():
    bridge = make_bridge()
    update = make_update()

    asyncio.run(bridge.new_session(update, make_context()))
    asyncio.run(bridge.cancel(update, make_context()))
    asyncio.run(bridge.stop(update, make_context()))
    asyncio.run(bridge.clear(update, make_context()))

    assert update.message is not None
    assert "Session started:" in update.message.replies[0]
    assert update.message.replies[1:] == [
        "Cancelled current operation.",
        "Stopped current session.",
        "No active session. Use /new first.",
    ]


def test_clear_with_session():
    bridge = make_bridge()
    update = make_update()

    asyncio.run(bridge.new_session(update, make_context()))
    asyncio.run(bridge.clear(update, make_context()))

    assert update.message is not None
    assert "Session started:" in update.message.replies[0]
    assert update.message.replies[1] == "Cleared current session."


def test_on_text_ignores_empty_message():
    bridge = make_bridge()
    update = make_update(text=None)
    context = make_context()

    asyncio.run(bridge.on_message(update, context))

    assert update.message is not None
    assert update.message.replies == []
    assert context.bot.actions == []


def test_reply_with_no_message_object():
    bridge = make_bridge()
    update = make_update(with_message=False)

    asyncio.run(bridge.help(update, make_context()))


def test_reply_agent_with_no_message_object():
    update = make_update(with_message=False)
    asyncio.run(TelegramBridge._reply_agent(update, "x"))


def test_on_text_ignores_when_message_is_missing():
    bridge = make_bridge()
    update = make_update(with_message=False)
    context = make_context()

    asyncio.run(bridge.on_message(update, context))
    assert context.bot.actions == []


def test_chat_id_without_chat_raises():
    update = cast(Update, SimpleNamespace(effective_chat=None))
    with pytest.raises(ChatRequiredError):
        TelegramBridge._chat_id(update)


def test_build_application_installs_handlers():
    bridge = make_bridge()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")

    app = build_application(config, bridge)
    assert app.handlers
    assert app.update_processor.max_concurrent_updates > 1


def test_run_polling(monkeypatch):
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

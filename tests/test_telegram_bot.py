from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from telegram_acp_bot.acp_app.echo_service import EchoAgentService
from telegram_acp_bot.core.session_registry import SessionRegistry
from telegram_acp_bot.telegram import bot as bot_module
from telegram_acp_bot.telegram.bot import ChatRequiredError, TelegramBridge, build_application, make_config, run_polling


class DummyMessage:
    def __init__(self, text: str | None = None) -> None:
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class DummyBot:
    def __init__(self) -> None:
        self.actions: list[tuple[int, str]] = []

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.actions.append((chat_id, action))


def make_update(*, user_id: int = 1, chat_id: int = 100, text: str | None = None, with_message: bool = True):
    message = DummyMessage(text) if with_message else None
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


def test_make_config() -> None:
    config = make_config(token="T", allowed_user_ids=[1, 2, 2], workspace="~/tmp")
    assert config.token == "T"
    assert config.allowed_user_ids == {1, 2}
    assert config.default_workspace.name == "tmp"


def test_start_and_help() -> None:
    bridge = make_bridge()
    update = make_update(with_message=True)
    context = make_context()

    asyncio.run(bridge.start(update, context))
    asyncio.run(bridge.help(update, context))

    assert update.message is not None
    assert "Use /new" in update.message.replies[0]
    assert "Commands:" in update.message.replies[1]
    assert "/cancel" in update.message.replies[1]


def test_access_denied() -> None:
    bridge = make_bridge(allowed_ids={99})
    update = make_update(user_id=1)
    context = make_context()

    asyncio.run(bridge.start(update, context))

    assert update.message is not None
    assert update.message.replies == ["Access denied for this bot."]


def test_access_allowed_with_allowlist() -> None:
    bridge = make_bridge(allowed_ids={1})
    update = make_update(user_id=1)
    context = make_context()

    asyncio.run(bridge.start(update, context))

    assert update.message is not None
    assert len(update.message.replies) == 1
    assert "Use /new" in update.message.replies[0]


def test_denied_paths_for_other_handlers() -> None:
    bridge = make_bridge(allowed_ids={42})
    update = make_update(user_id=7, text="hello")
    context = make_context()

    asyncio.run(bridge.help(update, context))
    asyncio.run(bridge.new_session(update, make_context(args=["/tmp"])))
    asyncio.run(bridge.session(update, context))
    asyncio.run(bridge.cancel(update, context))
    asyncio.run(bridge.stop(update, context))
    asyncio.run(bridge.clear(update, context))
    asyncio.run(bridge.on_text(update, context))

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


def test_new_session_and_session_command() -> None:
    bridge = make_bridge()
    update = make_update()

    asyncio.run(bridge.session(update, make_context()))
    asyncio.run(bridge.new_session(update, make_context(args=["/tmp"])))
    asyncio.run(bridge.session(update, make_context()))

    assert update.message is not None
    assert update.message.replies[0] == "No active session. Use /new first."
    assert "Session started:" in update.message.replies[1]
    assert "Active session workspace:" in update.message.replies[2]


def test_new_session_reports_invalid_workspace() -> None:
    class InvalidWorkspaceService:
        async def new_session(self, *, chat_id: int, workspace):
            del chat_id, workspace
            raise ValueError("/missing")

        async def prompt(self, *, chat_id: int, text: str):
            del chat_id, text

        def get_workspace(self, *, chat_id: int):
            del chat_id

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=InvalidWorkspaceService())
    update = make_update()

    asyncio.run(bridge.new_session(update, make_context(args=["/missing"])))

    assert update.message is not None
    assert update.message.replies == ["Invalid workspace: /missing"]


def test_new_session_reports_process_stdio_error() -> None:
    class BrokenAgentService:
        async def new_session(self, *, chat_id: int, workspace):
            del chat_id, workspace
            raise RuntimeError

        async def prompt(self, *, chat_id: int, text: str):
            del chat_id, text

        def get_workspace(self, *, chat_id: int):
            del chat_id

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=BrokenAgentService())
    update = make_update()

    asyncio.run(bridge.new_session(update, make_context(args=["/tmp"])))

    assert update.message is not None
    assert update.message.replies == ["Failed to start session: agent process did not expose stdio pipes."]


def test_new_session_reports_generic_error() -> None:
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
    bridge = TelegramBridge(config=config, agent_service=UnexpectedService())
    update = make_update()

    asyncio.run(bridge.new_session(update, make_context(args=["/tmp"])))

    assert update.message is not None
    assert update.message.replies == ["Failed to start session: boom"]


def test_on_text_without_and_with_session() -> None:
    bridge = make_bridge()
    update = make_update(text="hello")
    context = make_context()

    asyncio.run(bridge.on_text(update, context))
    asyncio.run(bridge.new_session(update, make_context()))
    asyncio.run(bridge.on_text(update, context))

    assert update.message is not None
    assert update.message.replies[0] == "No active session. Use /new first."
    assert update.message.replies[-1].endswith("hello")
    assert context.bot.actions == [(100, "typing"), (100, "typing")]


def test_cancel_stop_clear_without_session() -> None:
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


def test_cancel_stop_clear_with_session() -> None:
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


def test_clear_with_session() -> None:
    bridge = make_bridge()
    update = make_update()

    asyncio.run(bridge.new_session(update, make_context()))
    asyncio.run(bridge.clear(update, make_context()))

    assert update.message is not None
    assert "Session started:" in update.message.replies[0]
    assert update.message.replies[1] == "Cleared current session."


def test_on_text_ignores_empty_message() -> None:
    bridge = make_bridge()
    update = make_update(text=None)
    context = make_context()

    asyncio.run(bridge.on_text(update, context))

    assert update.message is not None
    assert update.message.replies == []
    assert context.bot.actions == []


def test_reply_with_no_message_object() -> None:
    bridge = make_bridge()
    update = make_update(with_message=False)

    asyncio.run(bridge.help(update, make_context()))


def test_on_text_ignores_when_message_is_missing() -> None:
    bridge = make_bridge()
    update = make_update(with_message=False)
    context = make_context()

    asyncio.run(bridge.on_text(update, context))
    assert context.bot.actions == []


def test_chat_id_without_chat_raises() -> None:
    update = SimpleNamespace(effective_chat=None)
    with pytest.raises(ChatRequiredError):
        TelegramBridge._chat_id(update)


def test_build_application_installs_handlers() -> None:
    bridge = make_bridge()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")

    app = build_application(config, bridge)
    assert app.handlers


def test_run_polling(monkeypatch) -> None:
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

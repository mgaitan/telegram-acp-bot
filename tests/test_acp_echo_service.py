import asyncio
from pathlib import Path

from telegram_acp_bot.acp_app.echo_service import EchoAgentService
from telegram_acp_bot.core.session_registry import SessionRegistry


def test_prompt_without_session_returns_none() -> None:
    service = EchoAgentService(SessionRegistry())

    reply = asyncio.run(service.prompt(chat_id=1, text="hello"))
    assert reply is None


def test_new_session_and_prompt() -> None:
    service = EchoAgentService(SessionRegistry())

    session_id = asyncio.run(service.new_session(chat_id=1, workspace=Path("/work")))
    assert isinstance(session_id, str)

    reply = asyncio.run(service.prompt(chat_id=1, text="hello"))
    assert reply is not None
    assert "hello" in reply.text
    assert session_id.split("-", maxsplit=1)[0] in reply.text


def test_get_workspace() -> None:
    service = EchoAgentService(SessionRegistry())

    assert service.get_workspace(chat_id=1) is None
    asyncio.run(service.new_session(chat_id=1, workspace=Path("/work")))
    assert service.get_workspace(chat_id=1) == Path("/work")

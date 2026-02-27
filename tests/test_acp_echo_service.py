import asyncio
from pathlib import Path

import pytest

from telegram_acp_bot.acp_app.echo_service import EchoAgentService
from telegram_acp_bot.core.session_registry import SessionRegistry


def test_prompt_without_session_returns_none() -> None:
    service = EchoAgentService(SessionRegistry())

    reply = asyncio.run(service.prompt(chat_id=1, text="hello"))
    assert reply is None


def test_new_session_and_prompt(tmp_path: Path) -> None:
    service = EchoAgentService(SessionRegistry())
    workspace = tmp_path / "echo-service-test-workspace"

    session_id = asyncio.run(service.new_session(chat_id=1, workspace=workspace))
    assert isinstance(session_id, str)

    reply = asyncio.run(service.prompt(chat_id=1, text="hello"))
    assert reply is not None
    assert "hello" in reply.text
    assert session_id.split("-", maxsplit=1)[0] in reply.text


def test_get_workspace(tmp_path: Path) -> None:
    service = EchoAgentService(SessionRegistry())
    workspace = tmp_path / "echo-service-test-workspace-2"

    assert service.get_workspace(chat_id=1) is None
    asyncio.run(service.new_session(chat_id=1, workspace=workspace))
    assert service.get_workspace(chat_id=1) == workspace


def test_permission_policy_updates_without_session() -> None:
    service = EchoAgentService(SessionRegistry())
    assert asyncio.run(service.set_session_permission_mode(chat_id=1, mode="approve")) is False
    assert asyncio.run(service.set_next_prompt_auto_approve(chat_id=1, enabled=True)) is False
    assert service.get_permission_policy(chat_id=1) is None


def test_new_session_rejects_file_workspace(tmp_path: Path) -> None:
    service = EchoAgentService(SessionRegistry())
    invalid = tmp_path / "file.txt"
    invalid.write_text("x")

    with pytest.raises(ValueError):
        asyncio.run(service.new_session(chat_id=1, workspace=invalid))

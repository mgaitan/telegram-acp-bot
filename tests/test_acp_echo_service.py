from pathlib import Path

import pytest

from telegram_acp_bot.acp_app.echo_service import EchoAgentService
from telegram_acp_bot.core.session_registry import SessionRegistry

pytestmark = pytest.mark.asyncio


async def test_prompt_without_session_returns_none() -> None:
    service = EchoAgentService(SessionRegistry())

    reply = await service.prompt(chat_id=1, text="hello")
    assert reply is None


async def test_new_session_and_prompt(tmp_path: Path) -> None:
    service = EchoAgentService(SessionRegistry())
    workspace = tmp_path / "echo-service-test-workspace"

    session_id = await service.new_session(chat_id=1, workspace=workspace)
    assert isinstance(session_id, str)

    reply = await service.prompt(chat_id=1, text="hello")
    assert reply is not None
    assert "hello" in reply.text
    assert session_id.split("-", maxsplit=1)[0] in reply.text


async def test_get_workspace(tmp_path: Path) -> None:
    service = EchoAgentService(SessionRegistry())
    workspace = tmp_path / "echo-service-test-workspace-2"

    assert service.get_workspace(chat_id=1) is None
    await service.new_session(chat_id=1, workspace=workspace)
    assert service.get_workspace(chat_id=1) == workspace


async def test_echo_service_supports_session_loading_and_loads_session(tmp_path: Path) -> None:
    service = EchoAgentService(SessionRegistry())
    workspace = tmp_path / "echo-load"

    loaded_id = await service.load_session(chat_id=1, session_id="external-session", workspace=workspace)
    assert isinstance(loaded_id, str)
    assert service.supports_session_loading(chat_id=1) is True
    assert service.get_workspace(chat_id=1) == workspace


async def test_echo_service_list_resumable_sessions(tmp_path: Path) -> None:
    service = EchoAgentService(SessionRegistry())
    workspace = tmp_path / "echo-list"
    await service.new_session(chat_id=1, workspace=workspace)

    listed = await service.list_resumable_sessions(chat_id=1, workspace=workspace)
    assert listed is not None
    assert len(listed) == 1
    assert listed[0].workspace == workspace

    other = await service.list_resumable_sessions(chat_id=1, workspace=tmp_path / "other")
    assert other == ()


async def test_echo_service_list_resumable_sessions_without_active_session() -> None:
    service = EchoAgentService(SessionRegistry())
    listed = await service.list_resumable_sessions(chat_id=999)
    assert listed == ()


async def test_permission_policy_updates_without_session() -> None:
    service = EchoAgentService(SessionRegistry())
    assert await service.set_session_permission_mode(chat_id=1, mode="approve") is False
    assert await service.set_next_prompt_auto_approve(chat_id=1, enabled=True) is False
    assert service.get_permission_policy(chat_id=1) is None


async def test_permission_policy_defaults_and_updates(tmp_path: Path) -> None:
    service = EchoAgentService(SessionRegistry())
    workspace = tmp_path / "echo-policy"
    await service.new_session(chat_id=1, workspace=workspace)

    policy = service.get_permission_policy(chat_id=1)
    assert policy is not None
    assert policy.session_mode == "ask"
    assert policy.next_prompt_auto_approve is False

    assert await service.set_session_permission_mode(chat_id=1, mode="approve")
    assert await service.set_next_prompt_auto_approve(chat_id=1, enabled=True)
    updated = service.get_permission_policy(chat_id=1)
    assert updated is not None
    assert updated.session_mode == "approve"
    assert updated.next_prompt_auto_approve is True


async def test_permission_handler_and_response_are_noop() -> None:
    service = EchoAgentService(SessionRegistry())

    async def handler(request):
        del request

    service.set_permission_request_handler(handler)
    assert await service.respond_permission_request(chat_id=1, request_id="x", action="deny") is False


async def test_new_session_rejects_file_workspace(tmp_path: Path) -> None:
    service = EchoAgentService(SessionRegistry())
    invalid = tmp_path / "file.txt"
    invalid.write_text("x")

    with pytest.raises(ValueError):
        await service.new_session(chat_id=1, workspace=invalid)


async def test_normalize_workspace_allows_none() -> None:
    assert EchoAgentService._normalize_workspace(None) is None

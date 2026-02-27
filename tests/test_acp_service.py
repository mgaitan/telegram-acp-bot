from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from acp import RequestError, text_block
from acp.schema import (
    AgentMessageChunk,
    AudioContentBlock,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextResourceContents,
)

from telegram_acp_bot.acp_app.acp_service import AcpAgentService, _AcpClient
from telegram_acp_bot.core.session_registry import SessionRegistry


class FakeProcess:
    def __init__(self, *, with_pipes: bool = True) -> None:
        self.stdin = object() if with_pipes else None
        self.stdout = object() if with_pipes else None
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class FakeConnection:
    def __init__(self, *, session_id: str = "acp-session") -> None:
        self.client: _AcpClient | None = None
        self.initialized = False
        self.cwd: str | None = None
        self.prompt_calls: list[str] = []
        self._session_id = session_id

    async def initialize(self, **kwargs):
        self.initialized = True
        assert kwargs["protocol_version"] == 1

    async def new_session(self, *, cwd: str, mcp_servers: list) -> SimpleNamespace:
        self.cwd = cwd
        assert mcp_servers == []
        return SimpleNamespace(session_id=self._session_id)

    async def prompt(self, *, session_id: str, prompt: list) -> SimpleNamespace:
        self.prompt_calls.append(session_id)
        assert prompt
        assert self.client is not None
        update = AgentMessageChunk(content=text_block("hello from acp"), sessionUpdate="agent_message_chunk")
        await self.client.session_update(session_id=session_id, update=update)
        return SimpleNamespace(stop_reason="end_turn")


def test_acp_client_capture_text_and_media_markers() -> None:
    client = _AcpClient(auto_approve_permissions=False)
    session_id = "s1"
    client.start_capture(session_id)

    update = AgentMessageChunk(content=text_block("hello"), sessionUpdate="agent_message_chunk")
    asyncio.run(client.session_update(session_id=session_id, update=update))

    assert client.finish_capture(session_id) == "hello"


def test_acp_client_ignores_non_message_updates() -> None:
    client = _AcpClient(auto_approve_permissions=False)
    session_id = "s-ignore"
    client.start_capture(session_id)
    asyncio.run(client.session_update(session_id=session_id, update=SimpleNamespace()))
    assert client.finish_capture(session_id) == ""


def test_acp_client_capture_non_text_content_markers() -> None:
    client = _AcpClient(auto_approve_permissions=False)
    session_id = "s2"
    client.start_capture(session_id)

    updates = [
        AgentMessageChunk(
            content=ImageContentBlock(data="AA==", mimeType="image/png", type="image"),
            sessionUpdate="agent_message_chunk",
        ),
        AgentMessageChunk(
            content=AudioContentBlock(data="AA==", mimeType="audio/wav", type="audio"),
            sessionUpdate="agent_message_chunk",
        ),
        AgentMessageChunk(
            content=ResourceContentBlock(name="r", uri="file:///tmp/r", type="resource_link"),
            sessionUpdate="agent_message_chunk",
        ),
        AgentMessageChunk(
            content=EmbeddedResourceContentBlock(
                resource=TextResourceContents(uri="mem://r", text="x"),
                type="resource",
            ),
            sessionUpdate="agent_message_chunk",
        ),
    ]

    for update in updates:
        asyncio.run(client.session_update(session_id=session_id, update=update))

    assert client.finish_capture(session_id) == "<image><audio>file:///tmp/r<resource>"


def test_acp_client_permission_decision_auto_approve() -> None:
    client = _AcpClient(auto_approve_permissions=True)
    option = SimpleNamespace(option_id="opt-1")
    response = asyncio.run(client.request_permission(options=[option], session_id="s", tool_call=SimpleNamespace()))
    assert response.outcome.outcome == "selected"


def test_acp_client_permission_decision_cancelled() -> None:
    client = _AcpClient(auto_approve_permissions=False)
    response = asyncio.run(client.request_permission(options=[], session_id="s", tool_call=SimpleNamespace()))
    assert response.outcome.outcome == "cancelled"


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("write_text_file", {"content": "c", "path": "p", "session_id": "s"}),
        ("read_text_file", {"path": "p", "session_id": "s"}),
        ("create_terminal", {"command": "ls", "session_id": "s"}),
        ("terminal_output", {"session_id": "s", "terminal_id": "t"}),
        ("release_terminal", {"session_id": "s", "terminal_id": "t"}),
        ("wait_for_terminal_exit", {"session_id": "s", "terminal_id": "t"}),
        ("kill_terminal", {"session_id": "s", "terminal_id": "t"}),
    ],
)
def test_acp_client_unsupported_methods_raise(method_name: str, args: dict[str, str]) -> None:
    client = _AcpClient(auto_approve_permissions=False)
    method = getattr(client, method_name)
    with pytest.raises(RequestError):
        asyncio.run(method(**args))


def test_acp_client_unsupported_ext_methods_raise() -> None:
    client = _AcpClient(auto_approve_permissions=False)
    with pytest.raises(RequestError):
        asyncio.run(client.ext_method("x", {}))
    with pytest.raises(RequestError):
        asyncio.run(client.ext_notification("x", {}))


def test_new_session_rejects_missing_workspace(tmp_path: Path) -> None:
    service = AcpAgentService(SessionRegistry(), program="agent", args=[])
    missing = tmp_path / "missing"

    with pytest.raises(ValueError):
        asyncio.run(service.new_session(chat_id=1, workspace=missing))


def test_new_session_rejects_process_without_stdio(tmp_path: Path) -> None:
    workspace = tmp_path

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return FakeProcess(with_pipes=False)

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn)

    with pytest.raises(RuntimeError):
        asyncio.run(service.new_session(chat_id=1, workspace=workspace))


def test_new_session_and_prompt(tmp_path: Path) -> None:
    process = FakeProcess()
    connection = FakeConnection(session_id="real-session")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        assert input_stream is process.stdin
        assert output_stream is process.stdout
        connection.client = client
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=["--x"],
        spawner=fake_spawn,
        connector=fake_connect,
    )

    session_id = asyncio.run(service.new_session(chat_id=2, workspace=tmp_path))
    assert session_id == "real-session"
    assert connection.initialized
    assert connection.cwd == str(tmp_path.resolve())
    assert service.get_workspace(chat_id=2) == tmp_path.resolve()

    reply = asyncio.run(service.prompt(chat_id=2, text="hi"))
    assert reply is not None
    assert reply.text == "hello from acp"
    assert connection.prompt_calls == ["real-session"]


def test_prompt_without_active_session_returns_none() -> None:
    service = AcpAgentService(SessionRegistry(), program="agent", args=[])
    reply = asyncio.run(service.prompt(chat_id=99, text="hi"))
    assert reply is None


def test_new_session_replaces_previous_and_shuts_down(tmp_path: Path, monkeypatch) -> None:
    first = FakeProcess()
    second = FakeProcess()
    calls: list[FakeProcess] = []

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        process = first if not calls else second
        calls.append(process)
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        conn = FakeConnection(session_id=f"s-{len(calls)}")
        conn.client = client
        return conn

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn, connector=fake_connect)
    asyncio.run(service.new_session(chat_id=5, workspace=tmp_path))
    asyncio.run(service.new_session(chat_id=5, workspace=tmp_path))

    assert first.terminated
    assert not second.terminated

    async def fake_wait_for(fut, **kwargs):
        del kwargs
        fut.close()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    hanging = FakeProcess()
    asyncio.run(service._shutdown(hanging))
    assert hanging.killed

    finished = FakeProcess()
    finished.returncode = 0
    asyncio.run(service._shutdown(finished))
    assert not finished.terminated

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from urllib.parse import quote

import pytest
from acp import RequestError, text_block
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    AudioContentBlock,
    BlobResourceContents,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    PermissionOption,
    RequestPermissionResponse,
    ResourceContentBlock,
    TextResourceContents,
    ToolCall,
    ToolCallProgress,
    ToolCallStart,
)

from telegram_acp_bot.acp_app.acp_service import AcpAgentService, _AcpClient, _PendingPermission
from telegram_acp_bot.acp_app.models import AgentActivityBlock, AgentReply, FilePayload, PromptFile, PromptImage
from telegram_acp_bot.core.session_registry import SessionRegistry

EXPECTED_CAPTURED_FILES = 2


def make_client() -> _AcpClient:
    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del tool_call
        if not options:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

        option_id = options[0].option_id
        return RequestPermissionResponse(outcome=AllowedOutcome(option_id=option_id, outcome="selected"))

    return _AcpClient(permission_decider=allow_first)


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
        update = AgentMessageChunk(content=text_block("hello from acp"), session_update="agent_message_chunk")
        await self.client.session_update(session_id=session_id, update=update)
        return SimpleNamespace(stop_reason="end_turn")

    async def cancel(self, *, session_id: str) -> None:
        self.prompt_calls.append(f"cancel:{session_id}")


class FileResourceConnection(FakeConnection):
    def __init__(self, *, session_id: str, resource: ResourceContentBlock) -> None:
        super().__init__(session_id=session_id)
        self._resource = resource

    async def prompt(self, *, session_id: str, prompt: list) -> SimpleNamespace:
        self.prompt_calls.append(session_id)
        assert prompt
        assert self.client is not None
        await self.client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(content=text_block("resource reply"), session_update="agent_message_chunk"),
        )
        await self.client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(content=self._resource, session_update="agent_message_chunk"),
        )
        return SimpleNamespace(stop_reason="end_turn")


def test_acp_client_capture_text_and_media_markers():
    client = make_client()
    session_id = "s1"
    client.start_capture(session_id)

    update = AgentMessageChunk(content=text_block("hello"), session_update="agent_message_chunk")
    asyncio.run(client.session_update(session_id=session_id, update=update))

    reply = asyncio.run(client.finish_capture(session_id))
    assert reply.text == "hello"
    assert reply.images == ()
    assert reply.files == ()


def test_acp_client_ignores_non_message_updates():
    client = make_client()
    session_id = "s-ignore"
    client.start_capture(session_id)
    asyncio.run(client.session_update(session_id=session_id, update=SimpleNamespace()))
    assert asyncio.run(client.finish_capture(session_id)).text == ""


def test_acp_client_capture_non_text_content_markers():
    client = make_client()
    session_id = "s2"
    client.start_capture(session_id)

    updates = [
        AgentMessageChunk(
            content=ImageContentBlock(data="AA==", mime_type="image/png", type="image"),
            session_update="agent_message_chunk",
        ),
        AgentMessageChunk(
            content=AudioContentBlock(data="AA==", mime_type="audio/wav", type="audio"),
            session_update="agent_message_chunk",
        ),
        AgentMessageChunk(
            content=ResourceContentBlock(name="r", uri="file:///tmp/r", type="resource_link"),
            session_update="agent_message_chunk",
        ),
        AgentMessageChunk(
            content=EmbeddedResourceContentBlock(
                resource=TextResourceContents(uri="mem://r", text="x"),
                type="resource",
            ),
            session_update="agent_message_chunk",
        ),
        AgentMessageChunk(
            content=EmbeddedResourceContentBlock(
                resource=BlobResourceContents(uri="mem://blob.bin", blob=base64.b64encode(b"x").decode("ascii")),
                type="resource",
            ),
            session_update="agent_message_chunk",
        ),
    ]

    for update in updates:
        asyncio.run(client.session_update(session_id=session_id, update=update))

    reply = asyncio.run(client.finish_capture(session_id))
    assert reply.text == "<image><audio>file:///tmp/r<resource><resource>"
    assert len(reply.images) == 1
    assert len(reply.files) == EXPECTED_CAPTURED_FILES + 1


def test_acp_client_permission_decision_auto_approve():
    client = make_client()
    option = PermissionOption(kind="allow_once", name="Allow once", option_id="opt-1")
    tool_call = ToolCall(title="execute", tool_call_id="tc-1")
    client.start_capture("s")
    response = asyncio.run(client.request_permission(options=[option], session_id="s", tool_call=tool_call))
    assert response.outcome.outcome == "selected"
    assert asyncio.run(client.finish_capture("s")).text == ""


def test_acp_client_permission_decision_cancelled():
    async def deny_all(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del tool_call
        del options

        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    client = _AcpClient(permission_decider=deny_all)
    tool_call = ToolCall(title="execute", tool_call_id="tc-2")
    client.start_capture("s")
    response = asyncio.run(client.request_permission(options=[], session_id="s", tool_call=tool_call))
    assert response.outcome.outcome == "cancelled"
    assert asyncio.run(client.finish_capture("s")).text == ""


def test_acp_client_capture_tool_events():
    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del options, tool_call
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    events: list[str] = []
    client = _AcpClient(permission_decider=allow_first, event_reporter=events.append)
    session_id = "s-tool"
    client.start_capture(session_id)

    start = ToolCallStart(title="read file", tool_call_id="tool-1", kind="read", session_update="tool_call")
    progress = ToolCallProgress(
        tool_call_id="tool-1",
        title="read file",
        status="completed",
        session_update="tool_call_update",
    )

    asyncio.run(client.session_update(session_id=session_id, update=start))
    asyncio.run(client.session_update(session_id=session_id, update=progress))
    _ = asyncio.run(client.finish_capture(session_id))
    assert "tool start tool-1 read file (read)" in events[0]
    assert "tool completed tool-1 read file" in events[1]


def test_acp_client_emits_live_activity_blocks():
    events: list[AgentActivityBlock] = []

    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del options, tool_call
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def capture_event(_: str, block: AgentActivityBlock) -> None:
        events.append(block)

    client = _AcpClient(permission_decider=allow_first, activity_reporter=capture_event)
    session_id = "s-live"
    client.start_capture(session_id)

    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=ToolCallStart(
                title="step think", tool_call_id="tool-think", kind="think", session_update="tool_call"
            ),
        )
    )
    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(content=text_block("plan first"), session_update="agent_message_chunk"),
        )
    )
    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=ToolCallStart(
                title="Run command", tool_call_id="tool-exec", kind="execute", session_update="tool_call"
            ),
        )
    )

    assert events[0] == AgentActivityBlock(kind="think", title="step think", status="in_progress", text="plan first")
    assert events[1] == AgentActivityBlock(kind="execute", title="Run command", status="in_progress", text="")


def test_acp_client_flushes_non_tool_text_as_thinking_before_next_tool():
    events: list[AgentActivityBlock] = []

    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del options, tool_call
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def capture_event(_: str, block: AgentActivityBlock) -> None:
        events.append(block)

    client = _AcpClient(permission_decider=allow_first, activity_reporter=capture_event)
    session_id = "s-pending-think"
    client.start_capture(session_id)

    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(content=text_block("first thought"), session_update="agent_message_chunk"),
        )
    )
    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=ToolCallStart(
                title="Run git log", tool_call_id="tool-exec", kind="execute", session_update="tool_call"
            ),
        )
    )
    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=ToolCallProgress(
                tool_call_id="tool-exec",
                title="Run git log",
                status="completed",
                session_update="tool_call_update",
            ),
        )
    )
    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(content=text_block("final output"), session_update="agent_message_chunk"),
        )
    )

    reply = asyncio.run(client.finish_capture(session_id))
    assert events[0] == AgentActivityBlock(kind="think", title="", status="completed", text="first thought")
    assert events[1] == AgentActivityBlock(kind="execute", title="Run git log", status="in_progress", text="")
    assert reply.text == "final output"


def test_acp_client_drops_empty_non_tool_text_when_flushing():
    events: list[AgentActivityBlock] = []

    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del options, tool_call
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def capture_event(_: str, block: AgentActivityBlock) -> None:
        events.append(block)

    client = _AcpClient(permission_decider=allow_first, activity_reporter=capture_event)
    session_id = "s-empty-pending"
    client.start_capture(session_id)
    client._pending_non_tool_text[session_id] = ["   "]

    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=ToolCallStart(title="Run cmd", tool_call_id="tool-exec", kind="execute", session_update="tool_call"),
        )
    )

    assert events == [AgentActivityBlock(kind="execute", title="Run cmd", status="in_progress", text="")]


def test_acp_client_groups_tool_output_into_activity_blocks():
    client = make_client()
    session_id = "s-blocks"
    client.start_capture(session_id)

    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=ToolCallStart(
                title="thinking step", tool_call_id="tool-think", kind="think", session_update="tool_call"
            ),
        )
    )
    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(content=text_block("draft plan"), session_update="agent_message_chunk"),
        )
    )
    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=ToolCallProgress(
                tool_call_id="tool-think",
                title="thinking step",
                status="completed",
                session_update="tool_call_update",
            ),
        )
    )
    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(content=text_block("final answer"), session_update="agent_message_chunk"),
        )
    )

    reply = asyncio.run(client.finish_capture(session_id))
    assert reply.text == "final answer"
    assert reply.activity_blocks == (
        AgentActivityBlock(kind="think", title="thinking step", status="completed", text="draft plan"),
    )


def test_acp_client_moves_trailing_non_think_block_text_to_final_reply():
    client = make_client()
    session_id = "s-trailing"
    client.start_capture(session_id)

    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=ToolCallStart(
                title="Run git show", tool_call_id="tool-exec", kind="execute", session_update="tool_call"
            ),
        )
    )
    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(content=text_block("This should be final."), session_update="agent_message_chunk"),
        )
    )

    reply = asyncio.run(client.finish_capture(session_id))
    assert reply.activity_blocks == (
        AgentActivityBlock(kind="execute", title="Run git show", status="in_progress", text=""),
    )
    assert reply.text == "This should be final."


def test_acp_client_ignores_terminal_progress_for_different_tool():
    client = make_client()
    session_id = "s-mismatch"
    client.start_capture(session_id)

    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=ToolCallStart(title="tool one", tool_call_id="tool-1", kind="read", session_update="tool_call"),
        )
    )
    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=AgentMessageChunk(content=text_block("partial output"), session_update="agent_message_chunk"),
        )
    )
    asyncio.run(
        client.session_update(
            session_id=session_id,
            update=ToolCallProgress(
                tool_call_id="tool-2",
                title="tool two",
                status="completed",
                session_update="tool_call_update",
            ),
        )
    )

    reply = asyncio.run(client.finish_capture(session_id))
    assert reply.activity_blocks == (AgentActivityBlock(kind="read", title="tool one", status="in_progress", text=""),)
    assert reply.text == "partial output"


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
def test_acp_client_unsupported_methods_raise(method_name: str, args: dict[str, str]):
    client = make_client()
    method = getattr(client, method_name)
    with pytest.raises(RequestError):
        asyncio.run(method(**args))


def test_acp_client_unsupported_ext_methods_raise():
    client = make_client()
    with pytest.raises(RequestError):
        asyncio.run(client.ext_method("x", {}))
    with pytest.raises(RequestError):
        asyncio.run(client.ext_notification("x", {}))


def test_new_session_creates_missing_workspace(tmp_path: Path):
    missing = tmp_path / "missing"
    process = FakeProcess()
    connection = FakeConnection(session_id="created")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    session_id = asyncio.run(service.new_session(chat_id=1, workspace=missing))
    assert session_id == "created"
    assert missing.is_dir()


def test_new_session_rejects_file_workspace(tmp_path: Path):
    service = AcpAgentService(SessionRegistry(), program="agent", args=[])
    invalid = tmp_path / "not-a-dir"
    invalid.write_text("x")

    with pytest.raises(ValueError):
        asyncio.run(service.new_session(chat_id=1, workspace=invalid))


def test_new_session_rejects_process_without_stdio(tmp_path: Path):
    workspace = tmp_path

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return FakeProcess(with_pipes=False)

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn)

    with pytest.raises(RuntimeError):
        asyncio.run(service.new_session(chat_id=1, workspace=workspace))


def test_acp_service_rejects_non_positive_stdio_limit():
    with pytest.raises(ValueError):
        AcpAgentService(SessionRegistry(), program="agent", args=[], stdio_limit=0)


def test_new_session_passes_stdio_limit_to_spawner(tmp_path: Path):
    process = FakeProcess()
    limits: list[int] = []

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args
        limits.append(cast(int, kwargs["limit"]))
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        conn = FakeConnection(session_id="s-limit")
        conn.client = client
        return conn

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        stdio_limit=2_000_000,
        spawner=fake_spawn,
        connector=fake_connect,
    )
    asyncio.run(service.new_session(chat_id=1, workspace=tmp_path))
    assert limits == [2_000_000]


def test_new_session_and_prompt(tmp_path: Path):
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


def test_prompt_without_active_session_returns_none():
    service = AcpAgentService(SessionRegistry(), program="agent", args=[])
    reply = asyncio.run(service.prompt(chat_id=99, text="hi"))
    assert reply is None


def test_prompt_resolves_file_uri_resource_as_image(tmp_path: Path):
    workspace = tmp_path / "ws"
    image_path = workspace / "img sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"\x89PNG\r\n")
    uri = image_path.as_uri()

    process = FakeProcess()
    connection = FileResourceConnection(
        session_id="file-image",
        resource=ResourceContentBlock(name="img sample.png", uri=uri, mime_type="image/png", type="resource_link"),
    )

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn, connector=fake_connect)
    asyncio.run(service.new_session(chat_id=1, workspace=workspace))
    reply = asyncio.run(service.prompt(chat_id=1, text="send image"))
    assert reply is not None
    assert len(reply.images) == 1
    assert reply.images[0].mime_type == "image/png"
    assert reply.files == ()


def test_prompt_resolves_file_uri_resource_as_document(tmp_path: Path):
    workspace = tmp_path / "ws"
    text_file = workspace / "note.txt"
    text_file.parent.mkdir(parents=True, exist_ok=True)
    text_file.write_text("hello file")

    process = FakeProcess()
    connection = FileResourceConnection(
        session_id="file-doc",
        resource=ResourceContentBlock(
            name="note.txt", uri=text_file.as_uri(), mime_type="text/plain", type="resource_link"
        ),
    )

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn, connector=fake_connect)
    asyncio.run(service.new_session(chat_id=1, workspace=workspace))
    reply = asyncio.run(service.prompt(chat_id=1, text="send doc"))
    assert reply is not None
    assert reply.images == ()
    assert len(reply.files) == 1
    assert reply.files[0].mime_type == "text/plain"
    assert reply.files[0].data_base64 == base64.b64encode(b"hello file").decode("ascii")


def test_prompt_reports_warning_for_outside_workspace_file_uri(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"x")

    process = FakeProcess()
    connection = FileResourceConnection(
        session_id="file-warning",
        resource=ResourceContentBlock(
            name="outside.png", uri=outside.as_uri(), mime_type="image/png", type="resource_link"
        ),
    )

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn, connector=fake_connect)
    asyncio.run(service.new_session(chat_id=1, workspace=workspace))
    reply = asyncio.run(service.prompt(chat_id=1, text="send outside"))
    assert reply is not None
    assert reply.images == ()
    assert reply.files == ()
    assert "Attachment warning: outside.png: path is outside active workspace" in reply.text


def test_prompt_resolves_percent_encoded_file_uri(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    encoded_name = "encoded file.png"
    encoded_file = workspace / encoded_name
    encoded_file.write_bytes(b"png-bytes")
    encoded_uri = f"file://{quote(str(encoded_file), safe='/')}"

    process = FakeProcess()
    connection = FileResourceConnection(
        session_id="encoded-uri",
        resource=ResourceContentBlock(name=encoded_name, uri=encoded_uri, mime_type="image/png", type="resource_link"),
    )

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn, connector=fake_connect)
    asyncio.run(service.new_session(chat_id=1, workspace=workspace))
    reply = asyncio.run(service.prompt(chat_id=1, text="send encoded"))
    assert reply is not None
    assert len(reply.images) == 1
    assert reply.images[0].mime_type == "image/png"


def test_cancel_and_stop_lifecycle(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(session_id="s1")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    asyncio.run(service.new_session(chat_id=7, workspace=tmp_path))

    assert asyncio.run(service.cancel(chat_id=7))
    assert connection.prompt_calls[-1] == "cancel:s1"
    assert asyncio.run(service.clear(chat_id=7))
    assert not asyncio.run(service.stop(chat_id=7))
    assert not asyncio.run(service.cancel(chat_id=7))


def test_permission_policy_session_and_next_prompt(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(session_id="perm-session")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    asyncio.run(service.new_session(chat_id=9, workspace=tmp_path))

    policy = service.get_permission_policy(chat_id=9)
    assert policy is not None
    assert policy.session_mode == "ask"
    assert not policy.next_prompt_auto_approve

    assert asyncio.run(service.set_session_permission_mode(chat_id=9, mode="approve"))
    assert asyncio.run(service.set_next_prompt_auto_approve(chat_id=9, enabled=True))
    policy = service.get_permission_policy(chat_id=9)
    assert policy is not None
    assert policy.session_mode == "approve"
    assert policy.next_prompt_auto_approve

    assert asyncio.run(service.prompt(chat_id=9, text="hello")) is not None
    image = PromptImage(data_base64=base64.b64encode(b"i").decode("ascii"), mime_type="image/png")
    file_text = PromptFile(name="t.txt", text_content="abc")
    file_bin = PromptFile(name="b.bin", data_base64=base64.b64encode(b"bin").decode("ascii"))
    assert (
        asyncio.run(service.prompt(chat_id=9, text="with files", images=(image,), files=(file_text, file_bin)))
        is not None
    )
    policy = service.get_permission_policy(chat_id=9)
    assert policy is not None
    assert not policy.next_prompt_auto_approve

    assert not asyncio.run(service.set_session_permission_mode(chat_id=999, mode="deny"))
    assert not asyncio.run(service.set_next_prompt_auto_approve(chat_id=999, enabled=True))
    assert service.get_permission_policy(chat_id=999) is None


def test_decide_permission_states(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(session_id="perm")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    asyncio.run(service.new_session(chat_id=1, workspace=tmp_path))
    option = PermissionOption(kind="allow_once", name="Allow once", option_id="opt")

    tool_call = ToolCall(title="run", tool_call_id="tc")

    denied_no_option = asyncio.run(service._decide_permission("perm", [], tool_call))
    assert denied_no_option.outcome.outcome == "cancelled"

    denied_unknown_session = asyncio.run(service._decide_permission("unknown", [option], tool_call))
    assert denied_unknown_session.outcome.outcome == "cancelled"

    asked_mode = asyncio.run(service._decide_permission("perm", [option], tool_call))
    assert asked_mode.outcome.outcome == "cancelled"

    live = service._live_by_chat[1]
    live.permission_mode = "approve"
    approved_session = asyncio.run(service._decide_permission("perm", [option], tool_call))
    assert approved_session.outcome.outcome == "selected"

    live.permission_mode = "deny"
    denied_session = asyncio.run(service._decide_permission("perm", [option], tool_call))
    assert denied_session.outcome.outcome == "cancelled"

    live.active_prompt_auto_approve = True
    approved_prompt = asyncio.run(service._decide_permission("perm", [option], tool_call))
    assert approved_prompt.outcome.outcome == "selected"


def test_decide_permission_ask_mode_with_handler(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(session_id="ask")
    captured: list[tuple[str, tuple[str, ...]]] = []

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    asyncio.run(service.new_session(chat_id=1, workspace=tmp_path))

    async def handler(request):
        captured.append((request.request_id, request.available_actions))
        await service.respond_permission_request(chat_id=1, request_id=request.request_id, action="once")

    service.set_permission_request_handler(handler)
    option = PermissionOption(kind="allow_once", name="Allow once", option_id="opt")
    tool_call = ToolCall(title="run", tool_call_id="tc-ask")
    response = asyncio.run(service._decide_permission("ask", [option], tool_call))
    assert response.outcome.outcome == "selected"
    assert captured
    assert "once" in captured[0][1]
    assert "deny" in captured[0][1]


def test_respond_permission_request_always_enables_session_approve(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(session_id="always")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    asyncio.run(service.new_session(chat_id=1, workspace=tmp_path))

    async def handler(request):
        await service.respond_permission_request(chat_id=1, request_id=request.request_id, action="always")

    service.set_permission_request_handler(handler)
    option = PermissionOption(kind="allow_always", name="Always", option_id="opt-always")
    tool_call = ToolCall(title="run", tool_call_id="tc-always")
    response = asyncio.run(service._decide_permission("always", [option], tool_call))
    assert response.outcome.outcome == "selected"
    policy = service.get_permission_policy(chat_id=1)
    assert policy is not None
    assert policy.session_mode == "approve"


def test_respond_permission_request_rejects_unknown_request(tmp_path: Path):
    service = AcpAgentService(SessionRegistry(), program="agent", args=[], default_permission_mode="ask")
    assert not asyncio.run(service.respond_permission_request(chat_id=1, request_id="missing", action="deny"))


def test_stop_cancels_pending_permission_requests(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(session_id="pending")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    asyncio.run(service.new_session(chat_id=7, workspace=tmp_path))

    async def scenario() -> None:
        future: asyncio.Future[RequestPermissionResponse] = asyncio.get_running_loop().create_future()
        service._pending_permissions["req"] = _PendingPermission(
            request_id="req",
            chat_id=7,
            acp_session_id="pending",
            tool_title="run",
            tool_call_id="tc",
            options=(),
            future=future,
        )
        assert await service.stop(chat_id=7)
        assert future.done()
        assert future.result().outcome.outcome == "cancelled"

    asyncio.run(scenario())


def test_decide_permission_timeout_returns_cancelled(tmp_path: Path, monkeypatch):
    process = FakeProcess()
    connection = FakeConnection(session_id="timeout")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    asyncio.run(service.new_session(chat_id=1, workspace=tmp_path))

    async def handler(request):
        del request

    service.set_permission_request_handler(handler)

    async def fake_wait_for(awaitable, **kwargs):
        del kwargs
        awaitable.cancel()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    option = PermissionOption(kind="allow_once", name="Allow once", option_id="opt")
    tool_call = ToolCall(title="run", tool_call_id="tc-timeout")
    response = asyncio.run(service._decide_permission("timeout", [option], tool_call))
    assert response.outcome.outcome == "cancelled"


def test_build_permission_response_fallbacks():
    deny = AcpAgentService._build_permission_response(options=(), action="deny")
    assert deny.outcome.outcome == "cancelled"

    fallback = AcpAgentService._build_permission_response(options=(), action="once")
    assert fallback.outcome.outcome == "cancelled"


def test_report_permission_event_respects_output_mode(caplog: pytest.LogCaptureFixture):
    service = AcpAgentService(SessionRegistry(), program="agent", args=[], permission_event_output="off")
    with caplog.at_level(logging.INFO):
        service._report_permission_event("x")
    assert not caplog.records

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], permission_event_output="stdout")
    with caplog.at_level(logging.INFO):
        service._report_permission_event("y")
    assert any("ACP permission event: y" in record.message for record in caplog.records)


def test_forward_activity_event_routes_to_matching_chat(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(session_id="activity-session")
    received: list[tuple[int, AgentActivityBlock]] = []

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    async def capture(chat_id: int, block: AgentActivityBlock) -> None:
        received.append((chat_id, block))

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn, connector=fake_connect)
    asyncio.run(service.new_session(chat_id=7, workspace=tmp_path))

    block = AgentActivityBlock(kind="think", title="t", status="completed", text="x")
    asyncio.run(service._forward_activity_event("activity-session", block))
    assert received == []

    service.set_activity_event_handler(capture)
    asyncio.run(service._forward_activity_event("unknown-session", block))
    assert received == []

    asyncio.run(service._forward_activity_event("activity-session", block))
    assert received == [(7, block)]


def test_new_session_replaces_previous_and_shuts_down(tmp_path: Path, monkeypatch):
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


def test_resolve_file_uri_resources_keeps_non_file_payloads(tmp_path: Path):
    service = AcpAgentService(SessionRegistry(), program="agent", args=[])
    response = AgentReply(
        text="ok",
        files=(
            FilePayload(name="inline.txt", text_content="plain text"),
            FilePayload(name="bin.dat", data_base64=base64.b64encode(b"x").decode("ascii")),
        ),
    )
    resolved = service._resolve_file_uri_resources(response=response, workspace=tmp_path)
    assert resolved.files == response.files
    assert resolved.images == ()


def test_resolve_file_uri_resources_reports_unreadable_file(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    file_path = workspace / "a.txt"
    file_path.write_text("x")

    service = AcpAgentService(SessionRegistry(), program="agent", args=[])
    response = AgentReply(
        text="",
        files=(FilePayload(name="a.txt", text_content=file_path.as_uri(), mime_type="text/plain"),),
    )

    def fake_read_bytes(self: Path) -> bytes:
        raise OSError("denied")

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    resolved = service._resolve_file_uri_resources(response=response, workspace=workspace)
    assert resolved.files == ()
    assert "Attachment warning: a.txt: unreadable file" in resolved.text


def test_resolve_local_file_uri_validation(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    file_path = workspace / "ok.txt"
    file_path.write_text("ok")
    folder_path = workspace / "dir"
    folder_path.mkdir()

    _, warning_scheme = AcpAgentService._resolve_local_file_uri("http://example.com/x", workspace)
    assert warning_scheme == "unsupported URI scheme `http`"

    _, warning_host = AcpAgentService._resolve_local_file_uri("file://remote/abc", workspace)
    assert warning_host == "unsupported file URI host `remote`"

    _, warning_empty = AcpAgentService._resolve_local_file_uri("file://", workspace)
    assert warning_empty == "empty file URI path"

    _, warning_missing = AcpAgentService._resolve_local_file_uri((workspace / "missing.txt").as_uri(), workspace)
    assert warning_missing == "file not found"

    _, warning_not_file = AcpAgentService._resolve_local_file_uri(folder_path.as_uri(), workspace)
    assert warning_not_file == "path is not a file"

    def fake_resolve(self: Path, *, strict: bool = False) -> Path:
        del strict
        raise OSError("boom")

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    _, warning_os_error = AcpAgentService._resolve_local_file_uri(file_path.as_uri(), workspace)
    assert warning_os_error is not None
    assert warning_os_error.startswith("invalid file path")

    monkeypatch.undo()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    _, warning_outside = AcpAgentService._resolve_local_file_uri(outside.as_uri(), workspace)
    assert warning_outside == "path is outside active workspace"

    resolved, no_warning = AcpAgentService._resolve_local_file_uri(file_path.as_uri(), workspace)
    assert resolved == file_path.resolve()
    assert no_warning is None

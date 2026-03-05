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
    AgentCapabilities,
    AgentMessageChunk,
    AllowedOutcome,
    AudioContentBlock,
    BlobResourceContents,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    McpServerStdio,
    PermissionOption,
    RequestPermissionResponse,
    ResourceContentBlock,
    SessionCapabilities,
    SessionInfo,
    SessionListCapabilities,
    TextResourceContents,
    ToolCall,
    ToolCallProgress,
    ToolCallStart,
)

from telegram_acp_bot.acp_app.acp_service import AcpAgentService, _AcpClient, _PendingPermission
from telegram_acp_bot.acp_app.models import (
    AgentActivityBlock,
    AgentOutputLimitExceededError,
    AgentReply,
    FilePayload,
    ImagePayload,
    PromptFile,
    PromptImage,
)
from telegram_acp_bot.core.session_registry import SessionRegistry
from telegram_acp_bot.mcp_channel_state import (
    load_last_session_id,
    load_session_chat_map,
    save_last_session_id,
    save_session_chat_map,
)

pytestmark = pytest.mark.asyncio

EXPECTED_CAPTURED_FILES = 2
ACP_STDIO_LIMIT_ERROR = "Separator is found, but chunk is longer than limit"


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
    def __init__(
        self,
        *,
        session_id: str = "acp-session",
        supports_load_session: bool = True,
        supports_session_list: bool = True,
        listed_sessions: list[SessionInfo] | None = None,
    ) -> None:
        self.client: _AcpClient | None = None
        self.initialized = False
        self.cwd: str | None = None
        self.prompt_calls: list[str] = []
        self.list_calls: list[tuple[str | None, str | None]] = []
        self._supports_load_session = supports_load_session
        self._supports_session_list = supports_session_list
        self._listed_sessions = listed_sessions or []
        self._session_id = session_id
        self.new_session_mcp_servers: list | None = None
        self.load_session_mcp_servers: list | None = None

    async def initialize(self, **kwargs):
        self.initialized = True
        assert kwargs["protocol_version"] == 1
        session_capabilities = (
            SessionCapabilities(list=SessionListCapabilities()) if self._supports_session_list else None
        )
        return SimpleNamespace(
            agent_capabilities=AgentCapabilities(
                load_session=self._supports_load_session,
                session_capabilities=session_capabilities,
            )
        )

    async def new_session(self, *, cwd: str, mcp_servers: list) -> SimpleNamespace:
        self.cwd = cwd
        self.new_session_mcp_servers = mcp_servers
        return SimpleNamespace(session_id=self._session_id)

    async def load_session(self, *, cwd: str, session_id: str, mcp_servers: list) -> SimpleNamespace:
        self.cwd = cwd
        self.load_session_mcp_servers = mcp_servers
        return SimpleNamespace(config_options=[], models=[], modes=[])

    async def list_sessions(self, *, cursor: str | None = None, cwd: str | None = None) -> SimpleNamespace:
        self.list_calls.append((cursor, cwd))
        return SimpleNamespace(next_cursor=None, sessions=self._listed_sessions)

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


class OversizeLineConnection(FakeConnection):
    async def prompt(self, *, session_id: str, prompt: list) -> SimpleNamespace:
        del session_id, prompt
        raise ValueError(ACP_STDIO_LIMIT_ERROR)


class GenericValueErrorConnection(FakeConnection):
    async def prompt(self, *, session_id: str, prompt: list) -> SimpleNamespace:
        del session_id, prompt
        raise ValueError("unexpected")


class HangingInitializeConnection(FakeConnection):
    async def initialize(self, **kwargs):
        del kwargs
        await asyncio.sleep(10)


class HangingNewSessionConnection(FakeConnection):
    async def new_session(self, *, cwd: str, mcp_servers: list) -> SimpleNamespace:
        del cwd, mcp_servers
        await asyncio.sleep(10)
        return SimpleNamespace()


class HangingLoadSessionConnection(FakeConnection):
    async def load_session(self, *, cwd: str, session_id: str, mcp_servers: list) -> SimpleNamespace:
        del cwd, session_id, mcp_servers
        await asyncio.sleep(10)
        return SimpleNamespace()


class NoSessionCapabilitiesConnection(FakeConnection):
    async def initialize(self, **kwargs):
        self.initialized = True
        assert kwargs["protocol_version"] == 1
        return SimpleNamespace(agent_capabilities=AgentCapabilities(load_session=True, session_capabilities=None))


async def test_acp_client_capture_text_and_media_markers():
    client = make_client()
    session_id = "s1"
    client.start_capture(session_id)

    update = AgentMessageChunk(content=text_block("hello"), session_update="agent_message_chunk")
    await client.session_update(session_id=session_id, update=update)

    reply = await client.finish_capture(session_id)
    assert reply.text == "hello"
    assert reply.images == ()
    assert reply.files == ()


async def test_acp_client_joins_adjacent_text_chunks_with_spacing():
    client = make_client()
    session_id = "s-text-join"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("Created (deleted)."), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("Done."), session_update="agent_message_chunk"),
    )

    reply = await client.finish_capture(session_id)
    assert reply.text == "Created (deleted). Done."


async def test_acp_client_does_not_insert_space_inside_semver_or_ip_chunks():
    client = make_client()
    session_id = "s-text-semver-ip"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("Version 10."), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("1.2"), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block(" and host 192."), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("168.0.1"), session_update="agent_message_chunk"),
    )

    reply = await client.finish_capture(session_id)
    assert reply.text == "Version 10.1.2 and host 192.168.0.1"


async def test_acp_client_does_not_insert_space_inside_split_word():
    client = make_client()
    session_id = "s-text-word-split"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("Sil"), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("ence"), session_update="agent_message_chunk"),
    )

    reply = await client.finish_capture(session_id)
    assert reply.text == "Silence"


async def test_acp_client_preserves_blank_lines_between_text_chunks():
    client = make_client()
    session_id = "s-text-blank-lines"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("First line\n\n"), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("Second line"), session_update="agent_message_chunk"),
    )

    reply = await client.finish_capture(session_id)
    assert reply.text == "First line\n\nSecond line"


async def test_acp_client_joins_adjacent_text_chunks_without_extra_spacing():
    client = make_client()
    session_id = "s-text-join-preserve"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("hello "), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("world"), session_update="agent_message_chunk"),
    )

    reply = await client.finish_capture(session_id)
    assert reply.text == "hello world"


async def test_acp_client_append_text_chunk_branch_coverage():
    target: list[str] = []
    _AcpClient._append_text_chunk(target, "a")
    assert target == ["a"]

    _AcpClient._append_text_chunk(target, "")
    assert target == ["a"]

    target = [""]
    _AcpClient._append_text_chunk(target, "b")
    assert target == ["", "b"]

    target = ["a "]
    _AcpClient._append_text_chunk(target, "b")
    assert target == ["a ", "b"]

    target = ["a"]
    _AcpClient._append_text_chunk(target, " b")
    assert target == ["a", " b"]

    target = ["a"]
    _AcpClient._append_text_chunk(target, ".")
    assert target == ["a", "."]

    target = ["Sil"]
    _AcpClient._append_text_chunk(target, "ence")
    assert target == ["Sil", "ence"]

    target = ["10."]
    _AcpClient._append_text_chunk(target, "1")
    assert target == ["10.", "1"]

    assert _AcpClient._is_numeric_dot_continuation(previous=".", chunk="1") is False


async def test_acp_client_non_text_chunk_in_active_tool_block_is_captured():
    client = make_client()
    session_id = "s-non-text-active"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=ToolCallStart(title="tool", tool_call_id="tool-1", kind="execute", session_update="tool_call"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(
            content=ImageContentBlock(data="AA==", mime_type="image/png", type="image"),
            session_update="agent_message_chunk",
        ),
    )

    reply = await client.finish_capture(session_id)
    assert reply.images == (ImagePayload(data_base64="AA==", mime_type="image/png"),)


async def test_acp_client_ignores_non_message_updates():
    client = make_client()
    session_id = "s-ignore"
    client.start_capture(session_id)
    await client.session_update(session_id=session_id, update=SimpleNamespace())
    assert (await client.finish_capture(session_id)).text == ""


async def test_acp_client_capture_non_text_content_markers():
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
        await client.session_update(session_id=session_id, update=update)

    reply = await client.finish_capture(session_id)
    assert reply.text == "<image><audio>file:///tmp/r<resource><resource>"
    assert len(reply.images) == 1
    assert len(reply.files) == EXPECTED_CAPTURED_FILES + 1


async def test_acp_client_permission_decision_auto_approve():
    client = make_client()
    option = PermissionOption(kind="allow_once", name="Allow once", option_id="opt-1")
    tool_call = ToolCall(title="execute", tool_call_id="tc-1")
    client.start_capture("s")
    response = await client.request_permission(options=[option], session_id="s", tool_call=tool_call)
    assert response.outcome.outcome == "selected"
    assert (await client.finish_capture("s")).text == ""


async def test_acp_client_permission_decision_cancelled():
    async def deny_all(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del tool_call
        del options

        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    client = _AcpClient(permission_decider=deny_all)
    tool_call = ToolCall(title="execute", tool_call_id="tc-2")
    client.start_capture("s")
    response = await client.request_permission(options=[], session_id="s", tool_call=tool_call)
    assert response.outcome.outcome == "cancelled"
    assert (await client.finish_capture("s")).text == ""


async def test_acp_client_capture_tool_events():
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

    await client.session_update(session_id=session_id, update=start)
    await client.session_update(session_id=session_id, update=progress)
    _ = await client.finish_capture(session_id)
    assert "tool start tool-1 read file (read)" in events[0]
    assert "tool completed tool-1 read file" in events[1]


async def test_acp_client_emits_live_activity_blocks():
    events: list[AgentActivityBlock] = []

    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del options, tool_call
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def capture_event(_: str, block: AgentActivityBlock) -> None:
        events.append(block)

    client = _AcpClient(permission_decider=allow_first, activity_reporter=capture_event)
    session_id = "s-live"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=ToolCallStart(title="step think", tool_call_id="tool-think", kind="think", session_update="tool_call"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("plan first"), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=ToolCallStart(title="Run command", tool_call_id="tool-exec", kind="execute", session_update="tool_call"),
    )

    assert events[0] == AgentActivityBlock(kind="think", title="step think", status="in_progress", text="plan first")
    assert events[1] == AgentActivityBlock(kind="execute", title="Run command", status="in_progress", text="")


async def test_acp_client_flushes_non_tool_text_as_thinking_before_next_tool():
    events: list[AgentActivityBlock] = []

    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del options, tool_call
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def capture_event(_: str, block: AgentActivityBlock) -> None:
        events.append(block)

    client = _AcpClient(permission_decider=allow_first, activity_reporter=capture_event)
    session_id = "s-pending-think"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("first thought"), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=ToolCallStart(title="Run git log", tool_call_id="tool-exec", kind="execute", session_update="tool_call"),
    )
    await client.session_update(
        session_id=session_id,
        update=ToolCallProgress(
            tool_call_id="tool-exec",
            title="Run git log",
            status="completed",
            session_update="tool_call_update",
        ),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("final output"), session_update="agent_message_chunk"),
    )

    reply = await client.finish_capture(session_id)
    assert events[0] == AgentActivityBlock(kind="think", title="", status="completed", text="first thought")
    assert events[1] == AgentActivityBlock(kind="execute", title="Run git log", status="in_progress", text="")
    assert reply.text == "final output"


async def test_acp_client_drops_empty_non_tool_text_when_flushing():
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

    await client.session_update(
        session_id=session_id,
        update=ToolCallStart(title="Run cmd", tool_call_id="tool-exec", kind="execute", session_update="tool_call"),
    )

    assert events == [AgentActivityBlock(kind="execute", title="Run cmd", status="in_progress", text="")]


async def test_acp_client_groups_tool_output_into_activity_blocks():
    client = make_client()
    session_id = "s-blocks"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=ToolCallStart(
            title="thinking step", tool_call_id="tool-think", kind="think", session_update="tool_call"
        ),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("draft plan"), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=ToolCallProgress(
            tool_call_id="tool-think",
            title="thinking step",
            status="completed",
            session_update="tool_call_update",
        ),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("final answer"), session_update="agent_message_chunk"),
    )

    reply = await client.finish_capture(session_id)
    assert reply.text == "final answer"
    assert reply.activity_blocks == (
        AgentActivityBlock(kind="think", title="thinking step", status="completed", text="draft plan"),
    )


async def test_acp_client_moves_trailing_non_think_block_text_to_final_reply():
    client = make_client()
    session_id = "s-trailing"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=ToolCallStart(
            title="Run git show", tool_call_id="tool-exec", kind="execute", session_update="tool_call"
        ),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("This should be final."), session_update="agent_message_chunk"),
    )

    reply = await client.finish_capture(session_id)
    assert reply.activity_blocks == (
        AgentActivityBlock(kind="execute", title="Run git show", status="in_progress", text=""),
    )
    assert reply.text == "This should be final."


async def test_acp_client_ignores_terminal_progress_for_different_tool():
    client = make_client()
    session_id = "s-mismatch"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=ToolCallStart(title="tool one", tool_call_id="tool-1", kind="read", session_update="tool_call"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("partial output"), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=ToolCallProgress(
            tool_call_id="tool-2",
            title="tool two",
            status="completed",
            session_update="tool_call_update",
        ),
    )

    reply = await client.finish_capture(session_id)
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
async def test_acp_client_unsupported_methods_raise(method_name: str, args: dict[str, str]):
    client = make_client()
    method = getattr(client, method_name)
    with pytest.raises(RequestError):
        await method(**args)


async def test_acp_client_unsupported_ext_methods_raise():
    client = make_client()
    with pytest.raises(RequestError):
        await client.ext_method("x", {})
    with pytest.raises(RequestError):
        await client.ext_notification("x", {})


async def test_new_session_creates_missing_workspace(tmp_path: Path):
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
    session_id = await service.new_session(chat_id=1, workspace=missing)
    assert session_id == "created"
    assert missing.is_dir()


async def test_new_session_rejects_file_workspace(tmp_path: Path):
    service = AcpAgentService(SessionRegistry(), program="agent", args=[])
    invalid = tmp_path / "not-a-dir"
    invalid.write_text("x")

    with pytest.raises(ValueError):
        await service.new_session(chat_id=1, workspace=invalid)


async def test_new_session_rejects_process_without_stdio(tmp_path: Path):
    workspace = tmp_path

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return FakeProcess(with_pipes=False)

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn)

    with pytest.raises(RuntimeError):
        await service.new_session(chat_id=1, workspace=workspace)


async def test_acp_service_rejects_non_positive_stdio_limit():
    with pytest.raises(ValueError):
        AcpAgentService(SessionRegistry(), program="agent", args=[], stdio_limit=0)


async def test_acp_service_rejects_non_positive_connect_timeout():
    with pytest.raises(ValueError):
        AcpAgentService(SessionRegistry(), program="agent", args=[], connect_timeout=0)


async def test_new_session_passes_stdio_limit_to_spawner(tmp_path: Path):
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
    await service.new_session(chat_id=1, workspace=tmp_path)
    assert limits == [2_000_000]


async def test_new_session_times_out_when_agent_does_not_handshake(tmp_path: Path):
    process = FakeProcess()
    connection = HangingInitializeConnection(session_id="never")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del client, input_stream, output_stream
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        connect_timeout=0.01,
        spawner=fake_spawn,
        connector=fake_connect,
    )

    with pytest.raises(RuntimeError, match="Timed out waiting for ACP agent handshake"):
        await service.new_session(chat_id=1, workspace=tmp_path)
    assert process.terminated


async def test_new_session_times_out_during_session_new(tmp_path: Path):
    process = FakeProcess()
    connection = HangingNewSessionConnection(session_id="never")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del client, input_stream, output_stream
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        connect_timeout=0.01,
        spawner=fake_spawn,
        connector=fake_connect,
    )
    with pytest.raises(RuntimeError, match="Timed out waiting for ACP agent handshake"):
        await service.new_session(chat_id=1, workspace=tmp_path)
    assert process.terminated


async def test_new_session_and_prompt(tmp_path: Path):
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

    session_id = await service.new_session(chat_id=2, workspace=tmp_path)
    assert session_id == "real-session"
    assert connection.initialized
    assert connection.cwd == str(tmp_path.resolve())
    assert connection.new_session_mcp_servers == []
    assert service.get_workspace(chat_id=2) == tmp_path.resolve()

    reply = await service.prompt(chat_id=2, text="hi")
    assert reply is not None
    assert reply.text == "hello from acp"
    assert connection.prompt_calls == ["real-session"]


async def test_load_session_and_supports_session_loading(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(session_id="real-session", supports_load_session=True)

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
    loaded_id = await service.load_session(chat_id=2, session_id="loaded-session", workspace=tmp_path)
    assert loaded_id == "loaded-session"
    assert connection.cwd == str(tmp_path.resolve())
    assert connection.load_session_mcp_servers == []
    assert service.get_workspace(chat_id=2) == tmp_path.resolve()
    assert service.supports_session_loading(chat_id=2) is True


async def test_service_passes_configured_mcp_servers_to_new_and_load_session(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(session_id="real-session", supports_load_session=True)
    server = McpServerStdio(name="telegram-channel", command="uv", args=["run", "mcp-server"], env=[])

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
        mcp_servers=(server,),
        spawner=fake_spawn,
        connector=fake_connect,
    )

    await service.new_session(chat_id=2, workspace=tmp_path)
    assert connection.new_session_mcp_servers == [server]
    await service.load_session(chat_id=2, session_id="loaded-session", workspace=tmp_path)
    assert connection.load_session_mcp_servers == [server]


async def test_service_persists_channel_session_mapping(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(session_id="mapped-session", supports_load_session=True)
    state_file = tmp_path / "channel-state.json"

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
        channel_state_file=state_file,
        spawner=fake_spawn,
        connector=fake_connect,
    )

    await service.new_session(chat_id=9, workspace=tmp_path)
    assert load_session_chat_map(state_file) == {"mapped-session": 9}
    assert load_last_session_id(state_file) == "mapped-session"

    await service.stop(chat_id=9)
    assert load_session_chat_map(state_file) == {}
    assert load_last_session_id(state_file) is None


async def test_drop_channel_session_mapping_ignores_unknown_session(tmp_path: Path):
    state_file = tmp_path / "channel-state.json"
    save_session_chat_map(state_file, {"known": 11})
    save_last_session_id(state_file, "known")
    service = AcpAgentService(SessionRegistry(), program="agent", args=[], channel_state_file=state_file)

    service._drop_channel_session_mapping(session_id="unknown")

    assert load_session_chat_map(state_file) == {"known": 11}
    assert load_last_session_id(state_file) == "known"


async def test_load_session_rejects_when_capability_is_false(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(supports_load_session=False)

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del client, input_stream, output_stream
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    with pytest.raises(RuntimeError, match="does not support `session/load`"):
        await service.load_session(chat_id=1, session_id="x", workspace=tmp_path)


async def test_load_session_rejects_file_workspace(tmp_path: Path):
    service = AcpAgentService(SessionRegistry(), program="agent", args=[])
    invalid = tmp_path / "not-a-dir"
    invalid.write_text("x")
    with pytest.raises(ValueError):
        await service.load_session(chat_id=1, session_id="x", workspace=invalid)


async def test_load_session_replaces_existing_and_shuts_down(tmp_path: Path):
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
        connection = FakeConnection(supports_load_session=True)
        connection.client = client
        return connection

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn, connector=fake_connect)
    await service.new_session(chat_id=3, workspace=tmp_path)
    await service.load_session(chat_id=3, session_id="reloaded", workspace=tmp_path)
    assert first.terminated


async def test_load_session_times_out(tmp_path: Path):
    process = FakeProcess()
    connection = HangingLoadSessionConnection(supports_load_session=True)

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del client, input_stream, output_stream
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        connect_timeout=0.01,
        spawner=fake_spawn,
        connector=fake_connect,
    )
    with pytest.raises(RuntimeError, match="Timed out waiting for ACP agent handshake"):
        await service.load_session(chat_id=1, session_id="x", workspace=tmp_path)
    assert process.terminated


async def test_list_resumable_sessions_supported_and_filtered(tmp_path: Path):
    workspace = tmp_path / "w"
    workspace.mkdir()
    listed = [
        SessionInfo(
            session_id="s1",
            cwd=str(workspace),
            title="session one",
            updated_at="2026-03-02T12:00:00Z",
        ),
        SessionInfo(
            session_id="s2",
            cwd=str(tmp_path / "other"),
            title="other",
            updated_at="2026-03-01T12:00:00Z",
        ),
    ]
    process = FakeProcess()
    connection = FakeConnection(listed_sessions=listed, supports_session_list=True)

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del client, input_stream, output_stream
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    sessions = await service.list_resumable_sessions(chat_id=1, workspace=workspace)
    assert sessions is not None
    assert [item.session_id for item in sessions] == ["s1"]
    assert connection.list_calls


async def test_list_resumable_sessions_returns_none_when_not_supported(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(supports_session_list=False)

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del client, input_stream, output_stream
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    assert await service.list_resumable_sessions(chat_id=1, workspace=tmp_path) is None
    assert service.supports_session_loading(chat_id=1) is None


async def test_list_resumable_sessions_live_without_list_support_returns_none(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(supports_session_list=False)

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
    await service.new_session(chat_id=8, workspace=tmp_path)
    assert await service.list_resumable_sessions(chat_id=8, workspace=tmp_path) is None


async def test_list_resumable_sessions_uses_live_connection_branch(tmp_path: Path):
    listed = [
        SessionInfo(
            session_id="s-live",
            cwd=str(tmp_path),
            title="live",
            updated_at="2026-03-02T14:00:00Z",
        )
    ]
    process = FakeProcess()
    connection = FakeConnection(listed_sessions=listed, supports_session_list=True)

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
    await service.new_session(chat_id=7, workspace=tmp_path)
    sessions = await service.list_resumable_sessions(chat_id=7, workspace=tmp_path)
    assert sessions is not None
    assert sessions[0].session_id == "s-live"


async def test_list_resumable_sessions_returns_none_when_session_capabilities_missing(tmp_path: Path):
    process = FakeProcess()
    connection = NoSessionCapabilitiesConnection()

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del client, input_stream, output_stream
        return connection

    service = AcpAgentService(
        SessionRegistry(),
        program="agent",
        args=[],
        spawner=fake_spawn,
        connector=fake_connect,
    )
    assert await service.list_resumable_sessions(chat_id=1, workspace=tmp_path) is None


async def test_prompt_without_active_session_returns_none():
    service = AcpAgentService(SessionRegistry(), program="agent", args=[])
    reply = await service.prompt(chat_id=99, text="hi")
    assert reply is None


async def test_prompt_wraps_stdio_limit_error(tmp_path: Path) -> None:
    process = FakeProcess()
    connection = OversizeLineConnection(session_id="limit-session")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn, connector=fake_connect)
    await service.new_session(chat_id=1, workspace=tmp_path)
    with pytest.raises(AgentOutputLimitExceededError):
        await service.prompt(chat_id=1, text="oversize")


async def test_prompt_reraises_unrelated_value_error(tmp_path: Path) -> None:
    process = FakeProcess()
    connection = GenericValueErrorConnection(session_id="value-error-session")

    async def fake_spawn(program: str, *args: str, **kwargs):
        del program, args, kwargs
        return process

    def fake_connect(client, input_stream, output_stream):
        del input_stream, output_stream
        connection.client = client
        return connection

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], spawner=fake_spawn, connector=fake_connect)
    await service.new_session(chat_id=1, workspace=tmp_path)
    with pytest.raises(ValueError, match="unexpected"):
        await service.prompt(chat_id=1, text="boom")


async def test_prompt_resolves_file_uri_resource_as_image(tmp_path: Path):
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
    await service.new_session(chat_id=1, workspace=workspace)
    reply = await service.prompt(chat_id=1, text="send image")
    assert reply is not None
    assert len(reply.images) == 1
    assert reply.images[0].mime_type == "image/png"
    assert reply.files == ()


async def test_prompt_resolves_file_uri_resource_as_document(tmp_path: Path):
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
    await service.new_session(chat_id=1, workspace=workspace)
    reply = await service.prompt(chat_id=1, text="send doc")
    assert reply is not None
    assert reply.images == ()
    assert len(reply.files) == 1
    assert reply.files[0].mime_type == "text/plain"
    assert reply.files[0].data_base64 == base64.b64encode(b"hello file").decode("ascii")


async def test_prompt_reports_warning_for_outside_workspace_file_uri(tmp_path: Path):
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
    await service.new_session(chat_id=1, workspace=workspace)
    reply = await service.prompt(chat_id=1, text="send outside")
    assert reply is not None
    assert reply.images == ()
    assert reply.files == ()
    assert "Attachment warning: outside.png: path is outside active workspace" in reply.text


async def test_prompt_resolves_percent_encoded_file_uri(tmp_path: Path):
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
    await service.new_session(chat_id=1, workspace=workspace)
    reply = await service.prompt(chat_id=1, text="send encoded")
    assert reply is not None
    assert len(reply.images) == 1
    assert reply.images[0].mime_type == "image/png"


async def test_cancel_and_stop_lifecycle(tmp_path: Path):
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
    await service.new_session(chat_id=7, workspace=tmp_path)

    assert await service.cancel(chat_id=7)
    assert connection.prompt_calls[-1] == "cancel:s1"
    assert await service.clear(chat_id=7)
    assert not await service.stop(chat_id=7)
    assert not await service.cancel(chat_id=7)


async def test_permission_policy_session_and_next_prompt(tmp_path: Path):
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
    await service.new_session(chat_id=9, workspace=tmp_path)

    policy = service.get_permission_policy(chat_id=9)
    assert policy is not None
    assert policy.session_mode == "ask"
    assert not policy.next_prompt_auto_approve

    assert await service.set_session_permission_mode(chat_id=9, mode="approve")
    assert await service.set_next_prompt_auto_approve(chat_id=9, enabled=True)
    policy = service.get_permission_policy(chat_id=9)
    assert policy is not None
    assert policy.session_mode == "approve"
    assert policy.next_prompt_auto_approve

    assert await service.prompt(chat_id=9, text="hello") is not None
    image = PromptImage(data_base64=base64.b64encode(b"i").decode("ascii"), mime_type="image/png")
    file_text = PromptFile(name="t.txt", text_content="abc")
    file_bin = PromptFile(name="b.bin", data_base64=base64.b64encode(b"bin").decode("ascii"))
    assert await service.prompt(chat_id=9, text="with files", images=(image,), files=(file_text, file_bin)) is not None
    policy = service.get_permission_policy(chat_id=9)
    assert policy is not None
    assert not policy.next_prompt_auto_approve

    assert not await service.set_session_permission_mode(chat_id=999, mode="deny")
    assert not await service.set_next_prompt_auto_approve(chat_id=999, enabled=True)
    assert service.get_permission_policy(chat_id=999) is None


async def test_decide_permission_states(tmp_path: Path):
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
    await service.new_session(chat_id=1, workspace=tmp_path)
    option = PermissionOption(kind="allow_once", name="Allow once", option_id="opt")

    tool_call = ToolCall(title="run", tool_call_id="tc")

    denied_no_option = await service._decide_permission("perm", [], tool_call)
    assert denied_no_option.outcome.outcome == "cancelled"

    denied_unknown_session = await service._decide_permission("unknown", [option], tool_call)
    assert denied_unknown_session.outcome.outcome == "cancelled"

    asked_mode = await service._decide_permission("perm", [option], tool_call)
    assert asked_mode.outcome.outcome == "cancelled"

    live = service._live_by_chat[1]
    live.permission_mode = "approve"
    approved_session = await service._decide_permission("perm", [option], tool_call)
    assert approved_session.outcome.outcome == "selected"

    live.permission_mode = "deny"
    denied_session = await service._decide_permission("perm", [option], tool_call)
    assert denied_session.outcome.outcome == "cancelled"

    live.active_prompt_auto_approve = True
    approved_prompt = await service._decide_permission("perm", [option], tool_call)
    assert approved_prompt.outcome.outcome == "selected"


async def test_decide_permission_ask_mode_with_handler(tmp_path: Path):
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
    await service.new_session(chat_id=1, workspace=tmp_path)

    async def handler(request):
        captured.append((request.request_id, request.available_actions))
        await service.respond_permission_request(chat_id=1, request_id=request.request_id, action="once")

    service.set_permission_request_handler(handler)
    option = PermissionOption(kind="allow_once", name="Allow once", option_id="opt")
    tool_call = ToolCall(title="run", tool_call_id="tc-ask")
    response = await service._decide_permission("ask", [option], tool_call)
    assert response.outcome.outcome == "selected"
    assert captured
    assert "once" in captured[0][1]
    assert "deny" in captured[0][1]


async def test_respond_permission_request_always_enables_session_approve(tmp_path: Path):
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
    await service.new_session(chat_id=1, workspace=tmp_path)

    async def handler(request):
        await service.respond_permission_request(chat_id=1, request_id=request.request_id, action="always")

    service.set_permission_request_handler(handler)
    option = PermissionOption(kind="allow_always", name="Always", option_id="opt-always")
    tool_call = ToolCall(title="run", tool_call_id="tc-always")
    response = await service._decide_permission("always", [option], tool_call)
    assert response.outcome.outcome == "selected"
    policy = service.get_permission_policy(chat_id=1)
    assert policy is not None
    assert policy.session_mode == "approve"


async def test_respond_permission_request_rejects_unknown_request(tmp_path: Path):
    service = AcpAgentService(SessionRegistry(), program="agent", args=[], default_permission_mode="ask")
    assert not await service.respond_permission_request(chat_id=1, request_id="missing", action="deny")


async def test_respond_permission_request_rejects_unavailable_action(tmp_path: Path):
    del tmp_path
    service = AcpAgentService(SessionRegistry(), program="agent", args=[])
    future: asyncio.Future[RequestPermissionResponse] = asyncio.get_running_loop().create_future()
    service._pending_permissions["req-1"] = _PendingPermission(
        request_id="req-1",
        chat_id=1,
        acp_session_id="s-1",
        tool_title="Run",
        tool_call_id="tc-1",
        options=(PermissionOption(kind="allow_once", name="Allow once", option_id="opt-once"),),
        future=future,
    )

    accepted = await service.respond_permission_request(chat_id=1, request_id="req-1", action="always")
    assert accepted is False
    assert not future.done()


async def test_decide_permission_auto_approve_supports_allow_always_only(tmp_path: Path):
    process = FakeProcess()
    connection = FakeConnection(session_id="approve-always-only")

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
    await service.new_session(chat_id=1, workspace=tmp_path)
    live = service._live_by_chat[1]
    live.permission_mode = "approve"

    option = PermissionOption(kind="allow_always", name="Always", option_id="opt-always")
    tool_call = ToolCall(title="run", tool_call_id="tc-auto-always")
    response = await service._decide_permission("approve-always-only", [option], tool_call)
    assert response.outcome.outcome == "selected"


async def test_stop_cancels_pending_permission_requests(tmp_path: Path):
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
    await service.new_session(chat_id=7, workspace=tmp_path)

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

    await scenario()


async def test_decide_permission_timeout_returns_cancelled(tmp_path: Path, monkeypatch):
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
    await service.new_session(chat_id=1, workspace=tmp_path)

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
    response = await service._decide_permission("timeout", [option], tool_call)
    assert response.outcome.outcome == "cancelled"


async def test_build_permission_response_fallbacks():
    deny = AcpAgentService._build_permission_response(options=(), action="deny")
    assert deny.outcome.outcome == "cancelled"

    fallback = AcpAgentService._build_permission_response(options=(), action="once")
    assert fallback.outcome.outcome == "cancelled"

    no_once = AcpAgentService._build_permission_response(
        options=(PermissionOption(kind="allow_always", name="Always", option_id="opt-always"),),
        action="once",
    )
    assert no_once.outcome.outcome == "cancelled"

    no_always = AcpAgentService._build_permission_response(
        options=(PermissionOption(kind="allow_once", name="Allow once", option_id="opt-once"),),
        action="always",
    )
    assert no_always.outcome.outcome == "cancelled"


async def test_available_actions_reflect_agent_options():
    both = (
        PermissionOption(kind="allow_once", name="Allow once", option_id="opt-once"),
        PermissionOption(kind="allow_always", name="Always", option_id="opt-always"),
    )
    assert AcpAgentService._available_actions(both) == ("always", "once", "deny")

    always_only = (PermissionOption(kind="allow_always", name="Always", option_id="opt-always"),)
    assert AcpAgentService._available_actions(always_only) == ("always", "deny")

    once_only = (PermissionOption(kind="allow_once", name="Allow once", option_id="opt-once"),)
    assert AcpAgentService._available_actions(once_only) == ("once", "deny")

    assert AcpAgentService._auto_approve_action(()) == "deny"


async def test_report_permission_event_respects_output_mode(caplog: pytest.LogCaptureFixture):
    service = AcpAgentService(SessionRegistry(), program="agent", args=[], permission_event_output="off")
    with caplog.at_level(logging.INFO):
        service._report_permission_event("x")
    assert not caplog.records

    service = AcpAgentService(SessionRegistry(), program="agent", args=[], permission_event_output="stdout")
    with caplog.at_level(logging.INFO):
        service._report_permission_event("y")
    assert any("ACP permission event: y" in record.message for record in caplog.records)


async def test_forward_activity_event_routes_to_matching_chat(tmp_path: Path):
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
    await service.new_session(chat_id=7, workspace=tmp_path)

    block = AgentActivityBlock(kind="think", title="t", status="completed", text="x")
    await service._forward_activity_event("activity-session", block)
    assert received == []

    service.set_activity_event_handler(capture)
    await service._forward_activity_event("unknown-session", block)
    assert received == []

    await service._forward_activity_event("activity-session", block)
    assert received == [(7, block)]


async def test_new_session_replaces_previous_and_shuts_down(tmp_path: Path, monkeypatch):
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
    await service.new_session(chat_id=5, workspace=tmp_path)
    await service.new_session(chat_id=5, workspace=tmp_path)

    assert first.terminated
    assert not second.terminated

    async def fake_wait_for(fut, **kwargs):
        del kwargs
        fut.close()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    hanging = FakeProcess()
    await service._shutdown(hanging)
    assert hanging.killed

    finished = FakeProcess()
    finished.returncode = 0
    await service._shutdown(finished)
    assert not finished.terminated


async def test_resolve_file_uri_resources_keeps_non_file_payloads(tmp_path: Path):
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


async def test_resolve_file_uri_resources_reports_unreadable_file(tmp_path: Path, monkeypatch):
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


async def test_resolve_local_file_uri_validation(tmp_path: Path, monkeypatch):
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

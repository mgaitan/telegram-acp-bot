from __future__ import annotations

# ruff: noqa: F403, F405, I001

from tests.acp.support import *


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
    assert target == ["10.", " ", "1"]

    assert _AcpClient._is_numeric_dot_continuation(previous=".", chunk="1") is False
    assert _AcpClient._is_numeric_dot_continuation(previous="Step 1.", chunk="2) detail") is False
    assert _AcpClient._is_numeric_dot_continuation(previous="Version 1.", chunk="2.3") is True


async def test_should_emit_incremental_text_returns_false_for_same_text():
    assert _AcpClient._should_emit_incremental_text(previous_text="same", last_emit_monotonic=0.0, text="same") is False


async def test_emit_incremental_text_block_skips_empty_or_unchanged_content():
    events: list[AgentActivityBlock] = []

    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del options, tool_call
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def capture_event(_: str, block: AgentActivityBlock) -> None:
        events.append(block)

    client = _AcpClient(permission_decider=allow_first, activity_reporter=capture_event)
    session_id = "s-incremental-branches"
    client.start_capture(session_id)

    await client._emit_incremental_text_block(session_id=session_id)
    client._pending_non_tool_text[session_id] = []
    await client._emit_incremental_text_block(session_id=session_id)
    client._pending_non_tool_text[session_id] = ["   "]
    await client._emit_incremental_text_block(session_id=session_id)
    client._pending_non_tool_text[session_id] = ["same"]
    client._pending_non_tool_state[session_id].last_emitted_text = "same"
    client._pending_non_tool_state[session_id].last_emit_monotonic = asyncio.get_running_loop().time()
    await client._emit_incremental_text_block(session_id=session_id)

    assert events == []


async def test_emit_incremental_text_block_skips_empty_and_unchanged_tool_output():
    events: list[AgentActivityBlock] = []

    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del options, tool_call
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def capture_event(_: str, block: AgentActivityBlock) -> None:
        events.append(block)

    client = _AcpClient(permission_decider=allow_first, activity_reporter=capture_event)
    session_id = "s-tool-incremental-branches"
    client.start_capture(session_id)
    await client._open_tool_block(session_id=session_id, tool_call_id="tool-1", kind="execute", title="Run")

    await client._emit_incremental_text_block(session_id=session_id)
    active = client._active_tool_blocks[session_id]
    assert active is not None
    active.chunks.append("same")
    active.last_emitted_text = "same"
    active.last_emit_monotonic = asyncio.get_running_loop().time()
    await client._emit_incremental_text_block(session_id=session_id)

    assert events == []


async def test_flush_pending_non_tool_text_emits_completed_block_when_text_changed():
    events: list[AgentActivityBlock] = []

    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del options, tool_call
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def capture_event(_: str, block: AgentActivityBlock) -> None:
        events.append(block)

    client = _AcpClient(permission_decider=allow_first, activity_reporter=capture_event)
    session_id = "s-flush-pending"
    client.start_capture(session_id)
    client._pending_non_tool_text[session_id] = ["final thought"]

    await client._flush_pending_non_tool_text(session_id=session_id)

    assert events == [AgentActivityBlock(kind="think", title="", status="completed", text="final thought")]


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

    events: list[tuple[str, str]] = []
    client = _AcpClient(
        permission_decider=allow_first, event_reporter=lambda session_id, event: events.append((session_id, event))
    )
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
    assert events[0][0] == session_id
    assert "tool start tool-1 read file (read)" in events[0][1]
    assert events[1][0] == session_id
    assert "tool completed tool-1 read file" in events[1][1]


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

    assert events[0] == AgentActivityBlock(
        kind="think",
        title="step think",
        status="in_progress",
        text="plan first",
        activity_id="tool-think",
    )
    assert events[1] == AgentActivityBlock(
        kind="think",
        title="step think",
        status="in_progress",
        text="plan first",
        activity_id="tool-think",
    )
    assert events[2] == AgentActivityBlock(
        kind="execute",
        title="Run command",
        status="in_progress",
        text="",
        activity_id="tool-exec",
    )


async def test_acp_client_emits_incremental_updates_for_active_tool_text():
    events: list[AgentActivityBlock] = []

    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del options, tool_call
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def capture_event(_: str, block: AgentActivityBlock) -> None:
        events.append(block)

    client = _AcpClient(permission_decider=allow_first, activity_reporter=capture_event)
    session_id = "s-incremental-tool"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=ToolCallStart(title="Run command", tool_call_id="tool-exec", kind="execute", session_update="tool_call"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("first line"), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block(" second line"), session_update="agent_message_chunk"),
    )

    assert events == [
        AgentActivityBlock(
            kind="execute",
            title="Run command",
            status="in_progress",
            text="",
            activity_id="tool-exec",
        ),
        AgentActivityBlock(
            kind="execute",
            title="Run command",
            status="in_progress",
            text="first line",
            activity_id="tool-exec",
        ),
        AgentActivityBlock(
            kind="execute",
            title="Run command",
            status="in_progress",
            text="first line second line",
            activity_id="tool-exec",
        ),
    ]


async def test_acp_client_streams_pending_non_tool_text_as_reply_preview_before_next_tool():
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
    assert events[0] == AgentActivityBlock(
        kind="reply",
        title="",
        status="in_progress",
        text="first thought",
        activity_id="reply",
    )
    assert events[1] == AgentActivityBlock(
        kind="execute",
        title="Run git log",
        status="in_progress",
        text="",
        activity_id="tool-exec",
    )
    assert reply.text == "final output"


async def test_acp_client_emits_incremental_updates_for_pending_reply_preview():
    events: list[AgentActivityBlock] = []

    async def allow_first(_: str, options: list[PermissionOption], tool_call: ToolCall) -> RequestPermissionResponse:
        del options, tool_call
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def capture_event(_: str, block: AgentActivityBlock) -> None:
        events.append(block)

    client = _AcpClient(permission_decider=allow_first, activity_reporter=capture_event)
    session_id = "s-incremental-reply"
    client.start_capture(session_id)

    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block("first thought"), session_update="agent_message_chunk"),
    )
    await client.session_update(
        session_id=session_id,
        update=AgentMessageChunk(content=text_block(" continues"), session_update="agent_message_chunk"),
    )

    assert events == [
        AgentActivityBlock(kind="reply", title="", status="in_progress", text="first thought", activity_id="reply"),
        AgentActivityBlock(
            kind="reply",
            title="",
            status="in_progress",
            text="first thought continues",
            activity_id="reply",
        ),
    ]


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

    assert events == [
        AgentActivityBlock(kind="execute", title="Run cmd", status="in_progress", text="", activity_id="tool-exec")
    ]


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
        AgentActivityBlock(
            kind="think",
            title="thinking step",
            status="completed",
            text="draft plan",
            activity_id="tool-think",
        ),
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
        AgentActivityBlock(
            kind="execute", title="Run git show", status="in_progress", text="", activity_id="tool-exec"
        ),
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
    assert reply.activity_blocks == (
        AgentActivityBlock(kind="read", title="tool one", status="in_progress", text="", activity_id="tool-1"),
    )
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

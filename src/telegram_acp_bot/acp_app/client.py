"""ACP client implementation that accumulates agent output per session."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

from acp import RequestError
from acp.schema import (
    AgentMessageChunk,
    AudioContentBlock,
    BlobResourceContents,
    CreateTerminalResponse,
    EmbeddedResourceContentBlock,
    EnvVariable,
    ImageContentBlock,
    KillTerminalCommandResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    TerminalOutputResponse,
    TextContentBlock,
    TextResourceContents,
    ToolCall,
    ToolCallProgress,
    ToolCallStart,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

from telegram_acp_bot.acp_app.models import (
    AgentActivityBlock,
    AgentReply,
    FilePayload,
    ImagePayload,
    ToolCallStatus,
)
from telegram_acp_bot.acp_app.protocols import (
    INCREMENTAL_TEXT_BOUNDARY_CHARS,
    INCREMENTAL_TEXT_MIN_DELTA_CHARS,
    INCREMENTAL_TEXT_MIN_INTERVAL_SECONDS,
    MIN_NUMERIC_DOT_CHUNK_MIN_LENGTH,
    MIN_NUMERIC_DOT_PREFIX_LENGTH,
    TERMINAL_TOOL_STATUSES,
)
from telegram_acp_bot.acp_app.session import (
    _ActiveToolBlock,
    _PendingTextState,
)


class _AcpClient:
    """ACP client callbacks that accumulate agent text chunks per session."""

    def __init__(
        self,
        *,
        permission_decider: Callable[[str, list[PermissionOption], ToolCall], Awaitable[RequestPermissionResponse]],
        event_reporter: Callable[[str, str], None] | None = None,
        activity_reporter: Callable[[str, AgentActivityBlock], Awaitable[None]] | None = None,
    ) -> None:
        self._permission_decider = permission_decider
        self._event_reporter = event_reporter
        self._activity_reporter = activity_reporter
        self._buffers: dict[str, list[str]] = {}
        self._pending_non_tool_text: dict[str, list[str]] = {}
        self._pending_non_tool_state: dict[str, _PendingTextState] = {}
        self._images: dict[str, list[ImagePayload]] = {}
        self._files: dict[str, list[FilePayload]] = {}
        self._active_tool_blocks: dict[str, _ActiveToolBlock | None] = {}
        self._completed_tool_blocks: dict[str, list[AgentActivityBlock]] = {}

    def start_capture(self, session_id: str) -> None:
        self._buffers[session_id] = []
        self._pending_non_tool_text[session_id] = []
        self._pending_non_tool_state[session_id] = _PendingTextState()
        self._images[session_id] = []
        self._files[session_id] = []
        self._active_tool_blocks[session_id] = None
        self._completed_tool_blocks[session_id] = []

    async def finish_capture(self, session_id: str) -> AgentReply:
        await self._close_active_tool_block(session_id=session_id, status="in_progress", is_prompt_end=True)
        chunks = self._buffers.pop(session_id, [])
        pending_text = self._pending_non_tool_text.pop(session_id, [])
        self._pending_non_tool_state.pop(session_id, None)
        images = tuple(self._images.pop(session_id, []))
        files = tuple(self._files.pop(session_id, []))
        activity_blocks = tuple(self._completed_tool_blocks.pop(session_id, []))
        self._active_tool_blocks.pop(session_id, None)
        final_text = "".join(chunks + pending_text).strip()
        return AgentReply(text=final_text, activity_blocks=activity_blocks, images=images, files=files)

    async def request_permission(
        self, options: list[PermissionOption], session_id: str, tool_call: ToolCall, **kwargs: object
    ) -> RequestPermissionResponse:
        del kwargs
        self._report_event(
            session_id,
            f"permission requested for {tool_call.title} ({tool_call.tool_call_id}), options={len(options)}",
        )
        return await self._permission_decider(session_id, options, tool_call)

    async def session_update(
        self,
        session_id: str,
        update: object,
        **kwargs: object,
    ) -> None:
        del kwargs

        if isinstance(update, ToolCallStart):
            label = update.kind or "other"
            self._report_event(session_id, f"tool start {update.tool_call_id} {update.title} ({label})")
            await self._flush_pending_non_tool_text(session_id=session_id)
            await self._open_tool_block(
                session_id=session_id,
                tool_call_id=update.tool_call_id,
                kind=label,
                title=update.title,
            )
            if label != "think":
                await self._emit_activity_block(
                    session_id=session_id,
                    block=AgentActivityBlock(
                        kind=label,
                        title=update.title,
                        status="in_progress",
                        activity_id=update.tool_call_id,
                    ),
                )
            return
        if isinstance(update, ToolCallProgress):
            status = update.status or "in_progress"
            title = update.title or "tool"
            self._report_event(session_id, f"tool {status} {update.tool_call_id} {title}")
            active_block = self._active_tool_blocks.get(session_id)
            if status in TERMINAL_TOOL_STATUSES and active_block is not None:
                if active_block.tool_call_id != update.tool_call_id:
                    return
                await self._close_active_tool_block(session_id=session_id, status=status)
            return
        if not isinstance(update, AgentMessageChunk):
            return

        await self._capture_agent_message(session_id=session_id, update=update)

    async def _capture_agent_message(self, *, session_id: str, update: AgentMessageChunk) -> None:  # noqa: C901
        content = update.content
        text = "<content>"
        is_text_chunk = isinstance(content, TextContentBlock)
        if isinstance(content, TextContentBlock):
            text = content.text
        elif isinstance(content, ImageContentBlock):
            text = "<image>"
            self._images.setdefault(session_id, []).append(
                ImagePayload(data_base64=content.data, mime_type=content.mime_type)
            )
        elif isinstance(content, AudioContentBlock):
            text = "<audio>"
        elif isinstance(content, ResourceContentBlock):
            text = content.uri or "<resource>"
            self._files.setdefault(session_id, []).append(
                FilePayload(name=content.name, mime_type=content.mime_type, text_content=content.uri)
            )
        elif isinstance(content, EmbeddedResourceContentBlock):
            text = "<resource>"
            resource = content.resource
            if isinstance(resource, TextResourceContents):
                self._files.setdefault(session_id, []).append(
                    FilePayload(
                        name=Path(resource.uri).name or "resource.txt",
                        mime_type=resource.mime_type,
                        text_content=resource.text,
                    )
                )
            elif isinstance(resource, BlobResourceContents):
                self._files.setdefault(session_id, []).append(
                    FilePayload(
                        name=Path(resource.uri).name or "resource.bin",
                        mime_type=resource.mime_type,
                        data_base64=resource.blob,
                    )
                )

        active_block = self._active_tool_blocks.get(session_id)
        target: list[str] | None = None
        if active_block is not None:
            target = active_block.chunks
        elif is_text_chunk:
            target = self._pending_non_tool_text.setdefault(session_id, [])

        if target is not None:
            if is_text_chunk:
                self._append_text_chunk(target, text)
                await self._emit_incremental_text_block(session_id=session_id)
            else:
                target.append(text)
            return

        assert not is_text_chunk
        self._buffers.setdefault(session_id, []).append(text)

    async def _emit_incremental_text_block(self, *, session_id: str) -> None:
        active_block = self._active_tool_blocks.get(session_id)
        if active_block is not None:
            text = "".join(active_block.chunks).strip()
            if not text:
                return
            if not self._should_emit_incremental_text(
                previous_text=active_block.last_emitted_text,
                last_emit_monotonic=active_block.last_emit_monotonic,
                text=text,
            ):
                return
            active_block.last_emitted_text = text
            active_block.last_emit_monotonic = asyncio.get_running_loop().time()
            await self._emit_activity_block(
                session_id=session_id,
                block=AgentActivityBlock(
                    kind=active_block.kind,
                    title=active_block.title,
                    status="in_progress",
                    text=text,
                    activity_id=active_block.tool_call_id,
                ),
            )
            return

        pending = self._pending_non_tool_text.get(session_id)
        if not pending:
            return
        text = "".join(pending).strip()
        if not text:
            return
        state = self._pending_non_tool_state.setdefault(session_id, _PendingTextState())
        if not self._should_emit_incremental_text(
            previous_text=state.last_emitted_text,
            last_emit_monotonic=state.last_emit_monotonic,
            text=text,
        ):
            return
        state.last_emitted_text = text
        state.last_emit_monotonic = asyncio.get_running_loop().time()
        await self._emit_activity_block(
            session_id=session_id,
            block=AgentActivityBlock(kind="reply", title="", status="in_progress", text=text, activity_id="reply"),
        )

    @staticmethod
    def _should_emit_incremental_text(*, previous_text: str, last_emit_monotonic: float, text: str) -> bool:
        if text == previous_text:
            return False
        if not previous_text:
            return True
        if "\n" in text[len(previous_text) :]:
            return True
        if len(text) - len(previous_text) >= INCREMENTAL_TEXT_MIN_DELTA_CHARS:
            return True
        if text.endswith(INCREMENTAL_TEXT_BOUNDARY_CHARS):
            return True
        return (asyncio.get_running_loop().time() - last_emit_monotonic) >= INCREMENTAL_TEXT_MIN_INTERVAL_SECONDS

    @staticmethod
    def _append_text_chunk(target: list[str], chunk: str) -> None:
        if not target:
            target.append(chunk)
            return
        if not chunk:
            return

        previous = target[-1]
        if not previous:
            target.append(chunk)
            return
        if previous[-1].isspace() or chunk[0].isspace():
            target.append(chunk)
            return
        if (
            previous[-1] in {".", "!", "?", ";", ":", ")", "]", "}"}
            and chunk[0].isalnum()
            and not _AcpClient._is_numeric_dot_continuation(previous=previous, chunk=chunk)
        ):
            target.append(" ")
        target.append(chunk)

    @staticmethod
    def _is_numeric_dot_continuation(*, previous: str, chunk: str) -> bool:
        """Return true when two chunks continue a numeric dot token like `10.1`."""

        if previous[-1] != "." or not chunk[0].isdigit():
            return False
        if len(previous) < MIN_NUMERIC_DOT_PREFIX_LENGTH:
            return False
        if not previous[-2].isdigit() or len(chunk) < MIN_NUMERIC_DOT_CHUNK_MIN_LENGTH:
            return False
        return chunk[1].isdigit() or chunk[1] == "."

    async def _open_tool_block(self, *, session_id: str, tool_call_id: str, kind: str, title: str) -> None:
        await self._close_active_tool_block(session_id=session_id, status="in_progress")
        self._active_tool_blocks[session_id] = _ActiveToolBlock(
            tool_call_id=tool_call_id,
            kind=kind,
            title=title,
        )

    async def _close_active_tool_block(self, *, session_id: str, status: str, is_prompt_end: bool = False) -> None:
        active_block = self._active_tool_blocks.get(session_id)
        if active_block is None:
            return

        normalized_status = status if status in TERMINAL_TOOL_STATUSES else "in_progress"
        block_text = "".join(active_block.chunks).strip()
        if is_prompt_end and active_block.kind != "think" and block_text:
            self._buffers.setdefault(session_id, []).append(block_text)
            block_text = ""
        block = AgentActivityBlock(
            kind=active_block.kind,
            title=active_block.title,
            status=cast(ToolCallStatus, normalized_status),
            text=block_text,
            activity_id=active_block.tool_call_id,
        )
        self._completed_tool_blocks.setdefault(session_id, []).append(block)
        self._active_tool_blocks[session_id] = None
        if block.kind == "think" or block.text:
            await self._emit_activity_block(session_id=session_id, block=block)

    def _report_event(self, session_id: str, event: str) -> None:
        if self._event_reporter is not None:
            self._event_reporter(session_id, event)

    async def _emit_activity_block(self, *, session_id: str, block: AgentActivityBlock) -> None:
        if self._activity_reporter is None:
            return
        await self._activity_reporter(session_id, block)

    async def _flush_pending_non_tool_text(self, *, session_id: str) -> None:
        pending = self._pending_non_tool_text.get(session_id)
        if not pending:
            return
        text = "".join(pending).strip()
        emitted_text = self._pending_non_tool_state.get(session_id, _PendingTextState()).last_emitted_text.strip()
        pending.clear()
        self._pending_non_tool_state[session_id] = _PendingTextState()
        if not text:
            return
        block = AgentActivityBlock(kind="think", title="", status="completed", text=text)
        self._completed_tool_blocks.setdefault(session_id, []).append(block)
        if emitted_text != text:
            await self._emit_activity_block(session_id=session_id, block=block)

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: object
    ) -> WriteTextFileResponse | None:
        del content, path, session_id, kwargs
        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: object,
    ) -> ReadTextFileResponse:
        del path, session_id, limit, line, kwargs
        raise RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(  # noqa: PLR0913
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: object,
    ) -> CreateTerminalResponse:
        del command, session_id, args, cwd, env, output_byte_limit, kwargs
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: object) -> TerminalOutputResponse:
        del session_id, terminal_id, kwargs
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> ReleaseTerminalResponse | None:
        del session_id, terminal_id, kwargs
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> WaitForTerminalExitResponse:
        del session_id, terminal_id, kwargs
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> KillTerminalCommandResponse | None:
        del session_id, terminal_id, kwargs
        raise RequestError.method_not_found("terminal/kill")

    async def ext_method(self, method: str, params: dict[str, object]) -> dict[str, object]:
        del method, params
        raise RequestError.method_not_found("ext/method")

    async def ext_notification(self, method: str, params: dict[str, object]) -> None:
        del method, params
        raise RequestError.method_not_found("ext/notification")

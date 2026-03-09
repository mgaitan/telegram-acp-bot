from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
import base64
import logging
import mimetypes
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import unquote, urlparse
from uuid import uuid4

from acp import PROTOCOL_VERSION, RequestError, connect_to_agent, text_block
from acp.core import ClientSideConnection
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AllowedOutcome,
    AudioContentBlock,
    BlobResourceContents,
    ClientCapabilities,
    CreateTerminalResponse,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    EnvVariable,
    ImageContentBlock,
    Implementation,
    KillTerminalCommandResponse,
    McpServerStdio,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    SessionInfo,
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
    AgentOutputLimitExceededError,
    AgentReply,
    FilePayload,
    ImagePayload,
    PermissionDecisionAction,
    PermissionEventOutput,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
    PromptFile,
    PromptImage,
    ResumableSession,
    ToolCallStatus,
)
from telegram_acp_bot.core.session_registry import SessionRegistry
from telegram_acp_bot.logging_context import bind_log_context
from telegram_acp_bot.mcp_channel_state import (
    load_last_session_id,
    load_session_chat_map,
    save_last_session_id,
    save_session_chat_map,
)

logger = logging.getLogger(__name__)
TERMINAL_TOOL_STATUSES = {"completed", "failed"}
MIN_NUMERIC_DOT_PREFIX_LENGTH = 2
MIN_NUMERIC_DOT_CHUNK_MIN_LENGTH = 2
PromptContentBlock = (
    TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock
)


class AcpConnectionFactory(Protocol):
    """Factory protocol to connect a client implementation to an ACP agent."""

    def __call__(self, client: object, input_stream: object, output_stream: object) -> ClientSideConnection: ...


class AcpSpawnFn(Protocol):
    """Spawner protocol for ACP subprocesses."""

    async def __call__(self, program: str, *args: str, **kwargs: object) -> asyncio.subprocess.Process: ...


class ProcessLike(Protocol):
    """Subset of process API used by service shutdown logic."""

    returncode: int | None

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


class AcpHandshakeTimeoutError(RuntimeError):
    """Raised when ACP initialize/new_session handshake does not finish in time."""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(f"Timed out waiting for ACP agent handshake after {timeout_seconds:.1f}s.")


class SessionLoadNotSupportedError(RuntimeError):
    """Raised when the ACP agent does not implement `session/load`."""

    def __init__(self) -> None:
        super().__init__("ACP agent does not support `session/load`.")


@dataclass(slots=True)
class _LiveSession:
    acp_session_id: str
    workspace: Path
    process: ProcessLike
    connection: ClientSideConnection
    client: _AcpClient
    supports_load_session: bool = False
    supports_session_list: bool = False
    permission_mode: PermissionMode = "deny"
    next_prompt_auto_approve: bool = False
    active_prompt_auto_approve: bool = False
    prompt_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class _PendingPermission:
    request_id: str
    chat_id: int
    acp_session_id: str
    tool_title: str
    tool_call_id: str
    options: tuple[PermissionOption, ...]
    future: asyncio.Future[RequestPermissionResponse]


@dataclass(slots=True)
class _ActiveToolBlock:
    tool_call_id: str
    kind: str
    title: str
    chunks: list[str] = field(default_factory=list)


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
        self._images: dict[str, list[ImagePayload]] = {}
        self._files: dict[str, list[FilePayload]] = {}
        self._active_tool_blocks: dict[str, _ActiveToolBlock | None] = {}
        self._completed_tool_blocks: dict[str, list[AgentActivityBlock]] = {}

    def start_capture(self, session_id: str) -> None:
        self._buffers[session_id] = []
        self._pending_non_tool_text[session_id] = []
        self._images[session_id] = []
        self._files[session_id] = []
        self._active_tool_blocks[session_id] = None
        self._completed_tool_blocks[session_id] = []

    async def finish_capture(self, session_id: str) -> AgentReply:
        await self._close_active_tool_block(session_id=session_id, status="in_progress", is_prompt_end=True)
        chunks = self._buffers.pop(session_id, [])
        pending_text = self._pending_non_tool_text.pop(session_id, [])
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
                    block=AgentActivityBlock(kind=label, title=update.title, status="in_progress"),
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

        self._capture_agent_message(session_id=session_id, update=update)

    def _capture_agent_message(self, *, session_id: str, update: AgentMessageChunk) -> None:  # noqa: C901
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
            else:
                target.append(text)
            return

        assert not is_text_chunk
        self._buffers.setdefault(session_id, []).append(text)

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
        pending.clear()
        if not text:
            return
        block = AgentActivityBlock(kind="think", title="", status="completed", text=text)
        self._completed_tool_blocks.setdefault(session_id, []).append(block)
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


class AcpAgentService:
    """ACP-backed service to manage one running agent session per Telegram chat."""

    def __init__(  # noqa: PLR0913
        self,
        registry: SessionRegistry,
        *,
        program: str,
        args: list[str],
        default_permission_mode: PermissionMode = "ask",
        permission_event_output: PermissionEventOutput = "stdout",
        mcp_servers: tuple[McpServerStdio, ...] = (),
        channel_state_file: Path | None = None,
        stdio_limit: int = 8_388_608,
        connect_timeout: float = 30.0,
        connector: AcpConnectionFactory | None = None,
        spawner: AcpSpawnFn | None = None,
    ) -> None:
        if stdio_limit <= 0:
            raise ValueError(stdio_limit)
        if connect_timeout <= 0:
            raise ValueError(connect_timeout)
        self._registry = registry
        self._program = program
        self._args = args
        self._default_permission_mode = default_permission_mode
        self._permission_event_output = permission_event_output
        self._mcp_servers = mcp_servers
        self._channel_state_file = channel_state_file
        self._stdio_limit = stdio_limit
        self._connect_timeout = connect_timeout
        self._connector = connector or cast(AcpConnectionFactory, connect_to_agent)
        self._spawner = spawner or cast(AcpSpawnFn, asyncio.create_subprocess_exec)
        self._live_by_chat: dict[int, _LiveSession] = {}
        self._chat_by_session: dict[str, int] = {}
        self._pending_permissions: dict[str, _PendingPermission] = {}
        self._permission_prompt_handler: Callable[[PermissionRequest], Awaitable[None]] | None = None
        self._activity_event_handler: Callable[[int, AgentActivityBlock], Awaitable[None]] | None = None
        self._channel_state_lock = asyncio.Lock()

    async def new_session(self, *, chat_id: int, workspace: Path) -> str:
        workspace = self._normalize_workspace(workspace)
        with bind_log_context(chat_id=chat_id):
            logger.info("Starting ACP session for chat_id=%s workspace=%s", chat_id, workspace)
        if workspace.exists() and not workspace.is_dir():
            raise ValueError(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        existing = self._live_by_chat.pop(chat_id, None)
        if existing is not None:
            await self._shutdown(existing.process)
            await self._drop_channel_session_mapping(session_id=existing.acp_session_id)
            self._chat_by_session.pop(existing.acp_session_id, None)

        process, connection, client, capabilities = await self._start_initialized_connection(chat_id=chat_id)
        try:
            logger.debug("Requesting ACP new_session for chat_id=%s", chat_id)
            session = await asyncio.wait_for(
                connection.new_session(cwd=str(workspace), mcp_servers=list(self._mcp_servers)),
                timeout=self._connect_timeout,
            )
        except TimeoutError as exc:
            await self._shutdown(process)
            raise AcpHandshakeTimeoutError(self._connect_timeout) from exc

        self._registry.create_or_replace(chat_id=chat_id, workspace=workspace, session_id=session.session_id)
        await self._save_channel_session_mapping(chat_id=chat_id, session_id=session.session_id)
        self._live_by_chat[chat_id] = _LiveSession(
            acp_session_id=session.session_id,
            workspace=workspace,
            process=process,
            connection=connection,
            client=client,
            supports_load_session=self._supports_load_session(capabilities),
            supports_session_list=self._supports_session_list(capabilities),
            permission_mode=self._default_permission_mode,
        )
        self._chat_by_session[session.session_id] = chat_id
        with bind_log_context(chat_id=chat_id, session_id=session.session_id):
            logger.info("ACP session started for chat_id=%s session_id=%s", chat_id, session.session_id)
        return session.session_id

    async def load_session(self, *, chat_id: int, session_id: str, workspace: Path) -> str:
        workspace = self._normalize_workspace(workspace)
        with bind_log_context(chat_id=chat_id, session_id=session_id):
            logger.info("Loading ACP session for chat_id=%s session_id=%s workspace=%s", chat_id, session_id, workspace)
        if workspace.exists() and not workspace.is_dir():
            raise ValueError(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        existing = self._live_by_chat.pop(chat_id, None)
        if existing is not None:
            await self._shutdown(existing.process)
            await self._drop_channel_session_mapping(session_id=existing.acp_session_id)
            self._chat_by_session.pop(existing.acp_session_id, None)

        process, connection, client, capabilities = await self._start_initialized_connection(chat_id=chat_id)
        if not self._supports_load_session(capabilities):
            await self._shutdown(process)
            raise SessionLoadNotSupportedError()
        try:
            await asyncio.wait_for(
                connection.load_session(
                    cwd=str(workspace),
                    session_id=session_id,
                    mcp_servers=list(self._mcp_servers),
                ),
                timeout=self._connect_timeout,
            )
        except TimeoutError as exc:
            await self._shutdown(process)
            raise AcpHandshakeTimeoutError(self._connect_timeout) from exc

        self._registry.create_or_replace(chat_id=chat_id, workspace=workspace, session_id=session_id)
        await self._save_channel_session_mapping(chat_id=chat_id, session_id=session_id)
        self._live_by_chat[chat_id] = _LiveSession(
            acp_session_id=session_id,
            workspace=workspace,
            process=process,
            connection=connection,
            client=client,
            supports_load_session=self._supports_load_session(capabilities),
            supports_session_list=self._supports_session_list(capabilities),
            permission_mode=self._default_permission_mode,
        )
        self._chat_by_session[session_id] = chat_id
        with bind_log_context(chat_id=chat_id, session_id=session_id):
            logger.info("ACP session loaded for chat_id=%s session_id=%s", chat_id, session_id)
        return session_id

    async def list_resumable_sessions(
        self,
        *,
        chat_id: int,
        workspace: Path | None = None,
    ) -> tuple[ResumableSession, ...] | None:
        normalized_workspace = None if workspace is None else self._normalize_workspace(workspace)
        live = self._live_by_chat.get(chat_id)
        if live is not None:
            if not live.supports_session_list:
                return None
            return await self._list_sessions_from_connection(
                connection=live.connection,
                workspace=normalized_workspace,
            )

        process, connection, _client, capabilities = await self._start_initialized_connection(chat_id=chat_id)
        try:
            if not self._supports_session_list(capabilities):
                return None
            return await self._list_sessions_from_connection(connection=connection, workspace=normalized_workspace)
        finally:
            await self._shutdown(process)

    async def prompt(
        self,
        *,
        chat_id: int,
        text: str,
        images: tuple[PromptImage, ...] = (),
        files: tuple[PromptFile, ...] = (),
    ) -> AgentReply | None:
        live = self._live_by_chat.get(chat_id)
        if live is None:
            with bind_log_context(chat_id=chat_id):
                logger.warning("Prompt ignored because no live session exists for chat_id=%s", chat_id)
            return None

        with bind_log_context(chat_id=chat_id, session_id=live.acp_session_id):
            logger.info(
                "Dispatching prompt to ACP session with text=%s chars images=%s files=%s",
                len(text),
                len(images),
                len(files),
            )
            async with live.prompt_lock:
                if live.next_prompt_auto_approve:
                    live.active_prompt_auto_approve = True
                    live.next_prompt_auto_approve = False
                live.client.start_capture(live.acp_session_id)
                prompt_blocks: list[PromptContentBlock] = [text_block(text)]
                prompt_blocks.extend(
                    [
                        ImageContentBlock(data=image.data_base64, mime_type=image.mime_type, type="image")
                        for image in images
                    ]
                )
                for file in files:
                    if file.text_content is not None:
                        text_payload = f"File: {file.name}\n\n{file.text_content}"
                        prompt_blocks.append(text_block(text_payload))
                        continue
                    if file.data_base64 is not None:
                        prompt_blocks.append(
                            text_block(f"Binary file attached: {file.name} ({file.mime_type or 'unknown'})")
                        )

                try:
                    try:
                        await live.connection.prompt(session_id=live.acp_session_id, prompt=prompt_blocks)
                    except ValueError as exc:
                        if "chunk is longer than limit" in str(exc):
                            raise AgentOutputLimitExceededError from exc
                        raise
                    response = await live.client.finish_capture(live.acp_session_id)
                    response = self._resolve_file_uri_resources(response=response, workspace=live.workspace)
                    logger.info("Received ACP reply with text=%s chars", len(response.text))
                    return AgentReply(
                        text=response.text,
                        images=response.images,
                        files=response.files,
                    )
                finally:
                    live.active_prompt_auto_approve = False

    def get_workspace(self, *, chat_id: int) -> Path | None:
        session = self._registry.get(chat_id)
        return None if session is None else session.workspace

    def get_active_session_context(self, *, chat_id: int) -> tuple[str, Path] | None:
        session = self._registry.get(chat_id)
        if session is None:
            return None
        return session.session_id, session.workspace

    def supports_session_loading(self, *, chat_id: int) -> bool | None:
        live = self._live_by_chat.get(chat_id)
        if live is None:
            return None
        return live.supports_load_session

    async def cancel(self, *, chat_id: int) -> bool:
        live = self._live_by_chat.get(chat_id)
        if live is None:
            return False

        await live.connection.cancel(session_id=live.acp_session_id)
        return True

    async def stop(self, *, chat_id: int) -> bool:
        live = self._live_by_chat.pop(chat_id, None)
        if live is None:
            return False

        stale = [request_id for request_id, pending in self._pending_permissions.items() if pending.chat_id == chat_id]
        for request_id in stale:
            pending = self._pending_permissions.pop(request_id)
            if not pending.future.done():
                pending.future.set_result(RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled")))

        await self._shutdown(live.process)
        self._registry.clear(chat_id)
        self._chat_by_session.pop(live.acp_session_id, None)
        await self._drop_channel_session_mapping(session_id=live.acp_session_id)
        return True

    async def clear(self, *, chat_id: int) -> bool:
        return await self.stop(chat_id=chat_id)

    def get_permission_policy(self, *, chat_id: int) -> PermissionPolicy | None:
        live = self._live_by_chat.get(chat_id)
        if live is None:
            return None
        return PermissionPolicy(
            session_mode=live.permission_mode,
            next_prompt_auto_approve=live.next_prompt_auto_approve,
        )

    async def set_session_permission_mode(self, *, chat_id: int, mode: PermissionMode) -> bool:
        live = self._live_by_chat.get(chat_id)
        if live is None:
            return False
        live.permission_mode = mode
        return True

    async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool) -> bool:
        live = self._live_by_chat.get(chat_id)
        if live is None:
            return False
        live.next_prompt_auto_approve = enabled
        return True

    def set_permission_request_handler(
        self,
        handler: Callable[[PermissionRequest], Awaitable[None]] | None,
    ) -> None:
        self._permission_prompt_handler = handler

    def set_activity_event_handler(
        self,
        handler: Callable[[int, AgentActivityBlock], Awaitable[None]] | None,
    ) -> None:
        self._activity_event_handler = handler

    async def respond_permission_request(
        self,
        *,
        chat_id: int,
        request_id: str,
        action: PermissionDecisionAction,
    ) -> bool:
        pending = self._pending_permissions.get(request_id)
        if pending is None or pending.chat_id != chat_id or pending.future.done():
            with bind_log_context(chat_id=chat_id):
                logger.warning(
                    "Permission response ignored: request_id=%s chat_id=%s pending=%s",
                    request_id,
                    chat_id,
                    pending is not None,
                )
            return False

        with bind_log_context(chat_id=chat_id, session_id=pending.acp_session_id):
            logger.info("Permission response received: request_id=%s action=%s", request_id, action)
        available_actions = self._available_actions(pending.options)
        if action not in available_actions:
            with bind_log_context(chat_id=chat_id, session_id=pending.acp_session_id):
                logger.warning(
                    "Permission response rejected: unavailable action request_id=%s chat_id=%s action=%s available=%s",
                    request_id,
                    chat_id,
                    action,
                    ",".join(available_actions),
                )
            return False

        response = self._build_permission_response(
            options=pending.options,
            action=action,
        )
        if action == "always":
            live = self._live_by_chat.get(chat_id)
            if live is not None:
                live.permission_mode = "approve"
        with bind_log_context(chat_id=chat_id, session_id=pending.acp_session_id):
            logger.info("Permission response accepted: request_id=%s chat_id=%s action=%s", request_id, chat_id, action)
        pending.future.set_result(response)
        return True

    async def _shutdown(self, process: ProcessLike) -> None:
        if process.returncode is not None:
            return

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def _start_initialized_connection(
        self,
        *,
        chat_id: int,
    ) -> tuple[ProcessLike, ClientSideConnection, _AcpClient, AgentCapabilities]:
        process = await self._spawner(
            self._program,
            *self._args,
            stdin=aio_subprocess.PIPE,
            stdout=aio_subprocess.PIPE,
            limit=self._stdio_limit,
        )
        with bind_log_context(chat_id=chat_id):
            logger.debug("Spawned ACP process for chat_id=%s pid=%s", chat_id, getattr(process, "pid", "unknown"))
        if process.stdin is None or process.stdout is None:
            raise RuntimeError

        client = _AcpClient(
            permission_decider=self._decide_permission,
            event_reporter=self._report_permission_event,
            activity_reporter=self._forward_activity_event,
        )
        connection = self._connector(client, process.stdin, process.stdout)
        try:
            with bind_log_context(chat_id=chat_id):
                logger.debug("Initializing ACP connection for chat_id=%s", chat_id)
            initialized = await asyncio.wait_for(
                connection.initialize(
                    protocol_version=PROTOCOL_VERSION,
                    client_capabilities=ClientCapabilities(),
                    client_info=Implementation(name="telegram-acp-bot", title="Telegram ACP Bot", version="0.1.1"),
                ),
                timeout=self._connect_timeout,
            )
        except TimeoutError as exc:
            await self._shutdown(process)
            raise AcpHandshakeTimeoutError(self._connect_timeout) from exc
        capabilities = (
            initialized.agent_capabilities
            if initialized is not None and hasattr(initialized, "agent_capabilities")
            else AgentCapabilities()
        )
        return process, connection, client, capabilities

    async def _list_sessions_from_connection(
        self,
        *,
        connection: ClientSideConnection,
        workspace: Path | None,
    ) -> tuple[ResumableSession, ...]:
        cursor: str | None = None
        sessions: list[SessionInfo] = []
        for _ in range(5):
            result = await asyncio.wait_for(
                connection.list_sessions(cursor=cursor, cwd=None if workspace is None else str(workspace)),
                timeout=self._connect_timeout,
            )
            sessions.extend(result.sessions or [])
            cursor = result.next_cursor
            if cursor is None:
                break
        normalized_target = self._resolved_workspace_or_none(workspace)
        mapped: list[ResumableSession] = []
        for info in sessions:
            item_workspace = self._workspace_from_session_cwd(info.cwd)
            if normalized_target is not None and item_workspace != normalized_target:
                continue
            mapped.append(
                ResumableSession(
                    session_id=info.session_id,
                    workspace=item_workspace,
                    title=(info.title or "").strip() or "(untitled session)",
                    updated_at=info.updated_at or "",
                )
            )
        mapped.sort(key=lambda item: item.updated_at, reverse=True)
        return tuple(mapped)

    @staticmethod
    def _supports_load_session(capabilities: AgentCapabilities) -> bool:
        # Some agents expose capabilities with partial/empty values; treat
        # explicit False as unsupported and probe otherwise.
        return capabilities.load_session is not False

    @staticmethod
    def _supports_session_list(capabilities: AgentCapabilities) -> bool:
        session_capabilities = capabilities.session_capabilities
        if session_capabilities is None:
            return False
        # Some agents expose `session_capabilities.list` as null/empty metadata
        # while still implementing `session/list`.
        return session_capabilities.list is not False

    @staticmethod
    def _normalize_workspace(workspace: Path) -> Path:
        return workspace.expanduser().resolve()

    @staticmethod
    def _resolved_workspace_or_none(workspace: Path | None) -> Path | None:
        return None if workspace is None else workspace.resolve()

    @staticmethod
    def _workspace_from_session_cwd(cwd: str | None) -> Path:
        raw_cwd = cwd or "."
        return Path(raw_cwd).expanduser().resolve()

    async def _decide_permission(
        self,
        session_id: str,
        options: list[PermissionOption],
        tool_call: ToolCall,
    ) -> RequestPermissionResponse:
        if not options:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

        for chat_id, live in self._live_by_chat.items():
            if live.acp_session_id != session_id:
                continue
            if live.permission_mode == "approve" or live.active_prompt_auto_approve:
                auto_action = self._auto_approve_action(tuple(options))
                return self._build_permission_response(options=tuple(options), action=auto_action)
            if live.permission_mode == "deny":
                return self._build_permission_response(options=tuple(options), action="deny")

            request_id = uuid4().hex[:12]
            future: asyncio.Future[RequestPermissionResponse] = asyncio.get_running_loop().create_future()
            pending = _PendingPermission(
                request_id=request_id,
                chat_id=chat_id,
                acp_session_id=session_id,
                tool_title=tool_call.title,
                tool_call_id=tool_call.tool_call_id,
                options=tuple(options),
                future=future,
            )
            self._pending_permissions[request_id] = pending
            if self._permission_prompt_handler is None:
                self._pending_permissions.pop(request_id, None)
                return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
            request = PermissionRequest(
                chat_id=pending.chat_id,
                request_id=request_id,
                tool_title=pending.tool_title,
                tool_call_id=pending.tool_call_id,
                available_actions=self._available_actions(options=pending.options),
            )
            await self._permission_prompt_handler(request)

            try:
                response = await asyncio.wait_for(future, timeout=300)
            except TimeoutError:
                response = RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
            finally:
                self._pending_permissions.pop(request_id, None)
            return response

        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    @staticmethod
    def _available_actions(options: tuple[PermissionOption, ...]) -> tuple[PermissionDecisionAction, ...]:
        kinds = {option.kind for option in options}
        actions: list[PermissionDecisionAction] = []
        if "allow_always" in kinds:
            actions.append("always")
        if "allow_once" in kinds:
            actions.append("once")
        actions.append("deny")
        return tuple(actions)

    @staticmethod
    def _build_permission_response(
        *,
        options: tuple[PermissionOption, ...],
        action: PermissionDecisionAction,
    ) -> RequestPermissionResponse:
        if action == "deny":
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

        target_kind = "allow_always" if action == "always" else "allow_once"
        for option in options:
            if option.kind == target_kind:
                return RequestPermissionResponse(outcome=AllowedOutcome(option_id=option.option_id, outcome="selected"))
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    @staticmethod
    def _auto_approve_action(options: tuple[PermissionOption, ...]) -> PermissionDecisionAction:
        kinds = {option.kind for option in options}
        if "allow_once" in kinds:
            return "once"
        if "allow_always" in kinds:
            return "always"
        return "deny"

    def _report_permission_event(self, session_id: str, event: str) -> None:
        if self._permission_event_output != "stdout":
            return
        chat_id = self._chat_id_by_session(session_id)
        with bind_log_context(chat_id=chat_id, session_id=session_id):
            logger.info("ACP permission event: %s", event)

    async def _forward_activity_event(self, session_id: str, block: AgentActivityBlock) -> None:
        handler = self._activity_event_handler
        if handler is None:
            return
        for chat_id, live in self._live_by_chat.items():
            if live.acp_session_id == session_id:
                await handler(chat_id, block)
                return

    def _chat_id_by_session(self, session_id: str) -> int | None:
        return self._chat_by_session.get(session_id)

    def _resolve_file_uri_resources(self, *, response: AgentReply, workspace: Path) -> AgentReply:
        """Resolve `file://` resources from ACP into binary payloads for Telegram delivery."""
        images = list(response.images)
        files: list[FilePayload] = []
        warnings: list[str] = []
        workspace_root = workspace.resolve()

        for payload in response.files:
            if payload.data_base64 is not None or payload.text_content is None:
                files.append(payload)
                continue
            if not payload.text_content.startswith("file://"):
                files.append(payload)
                continue

            resolved_path, warning = self._resolve_local_file_uri(payload.text_content, workspace_root)
            if warning is not None:
                warnings.append(f"{payload.name}: {warning}")
                continue

            assert resolved_path is not None
            try:
                raw = resolved_path.read_bytes()
            except OSError as exc:
                warnings.append(f"{payload.name}: unreadable file ({exc})")
                continue

            mime_type = payload.mime_type or mimetypes.guess_type(resolved_path.name)[0] or "application/octet-stream"
            if mime_type.startswith("image/"):
                images.append(ImagePayload(data_base64=base64.b64encode(raw).decode("ascii"), mime_type=mime_type))
                continue

            files.append(
                FilePayload(
                    name=payload.name or resolved_path.name,
                    mime_type=mime_type,
                    data_base64=base64.b64encode(raw).decode("ascii"),
                )
            )

        text = response.text
        if warnings:
            warning_text = "\n".join(f"Attachment warning: {warning}" for warning in warnings)
            text = f"{text}\n{warning_text}".strip() if text else warning_text

        return AgentReply(
            text=text,
            activity_blocks=response.activity_blocks,
            images=tuple(images),
            files=tuple(files),
        )

    async def _save_channel_session_mapping(self, *, chat_id: int, session_id: str) -> None:
        if self._channel_state_file is None:
            return
        async with self._channel_state_lock:
            await asyncio.to_thread(self._write_channel_session_mapping, chat_id=chat_id, session_id=session_id)

    async def _drop_channel_session_mapping(self, *, session_id: str) -> None:
        if self._channel_state_file is None:
            return
        async with self._channel_state_lock:
            await asyncio.to_thread(self._remove_channel_session_mapping, session_id=session_id)

    async def _set_last_channel_session(self, *, session_id: str) -> None:
        if self._channel_state_file is None:
            return
        async with self._channel_state_lock:
            await asyncio.to_thread(save_last_session_id, self._channel_state_file, session_id)

    def _write_channel_session_mapping(self, *, chat_id: int, session_id: str) -> None:
        assert self._channel_state_file is not None
        mapping = load_session_chat_map(self._channel_state_file)
        mapping[session_id] = chat_id
        save_session_chat_map(self._channel_state_file, mapping)
        save_last_session_id(self._channel_state_file, session_id)

    def _remove_channel_session_mapping(self, *, session_id: str) -> None:
        assert self._channel_state_file is not None
        mapping = load_session_chat_map(self._channel_state_file)
        if session_id not in mapping:
            return
        mapping.pop(session_id, None)
        save_session_chat_map(self._channel_state_file, mapping)
        if load_last_session_id(self._channel_state_file) == session_id:
            save_last_session_id(self._channel_state_file, None)

    @staticmethod
    def _resolve_local_file_uri(uri: str, workspace_root: Path) -> tuple[Path | None, str | None]:
        parsed = urlparse(uri)
        warning: str | None = None
        resolved_path: Path | None = None

        if parsed.scheme != "file":
            warning = f"unsupported URI scheme `{parsed.scheme}`"
        elif parsed.netloc not in {"", "localhost"}:
            warning = f"unsupported file URI host `{parsed.netloc}`"
        else:
            decoded_path = unquote(parsed.path)
            if not decoded_path:
                warning = "empty file URI path"
            else:
                try:
                    resolved = Path(decoded_path).resolve(strict=True)
                except FileNotFoundError:
                    warning = "file not found"
                except OSError as exc:
                    warning = f"invalid file path ({exc})"
                else:
                    if not resolved.is_file():
                        warning = "path is not a file"
                    else:
                        try:
                            resolved.relative_to(workspace_root)
                        except ValueError:
                            warning = "path is outside active workspace"
                        else:
                            resolved_path = resolved

        return resolved_path, warning

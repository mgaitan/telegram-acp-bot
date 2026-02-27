from __future__ import annotations

import asyncio
import asyncio.subprocess as aio_subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from acp import PROTOCOL_VERSION, RequestError, connect_to_agent, text_block
from acp.core import ClientSideConnection
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AllowedOutcome,
    AudioContentBlock,
    AvailableCommandsUpdate,
    ClientCapabilities,
    ConfigOptionUpdate,
    CreateTerminalResponse,
    CurrentModeUpdate,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    EnvVariable,
    ImageContentBlock,
    Implementation,
    KillTerminalCommandResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    SessionInfoUpdate,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCall,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

from telegram_acp_bot.acp_app.echo_service import AgentReply
from telegram_acp_bot.core.session_registry import SessionRegistry


class AcpConnectionFactory(Protocol):
    """Factory protocol to connect a client implementation to an ACP agent."""

    def __call__(self, client: object, input_stream: object, output_stream: object) -> ClientSideConnection: ...


class AcpSpawnFn(Protocol):
    """Spawner protocol for ACP subprocesses."""

    async def __call__(self, program: str, *args: str, **kwargs: object) -> asyncio.subprocess.Process: ...


@dataclass(slots=True)
class _LiveSession:
    acp_session_id: str
    workspace: Path
    process: asyncio.subprocess.Process
    connection: ClientSideConnection
    client: _AcpClient
    prompt_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class _AcpClient:
    """ACP client callbacks that accumulate agent text chunks per session."""

    def __init__(self, *, auto_approve_permissions: bool) -> None:
        self._auto_approve_permissions = auto_approve_permissions
        self._buffers: dict[str, list[str]] = {}

    def start_capture(self, session_id: str) -> None:
        self._buffers[session_id] = []

    def finish_capture(self, session_id: str) -> str:
        chunks = self._buffers.pop(session_id, [])
        return "".join(chunks).strip()

    async def request_permission(
        self, options: list[PermissionOption], session_id: str, tool_call: ToolCall, **kwargs: object
    ) -> RequestPermissionResponse:
        del session_id, tool_call, kwargs

        if self._auto_approve_permissions and options:
            outcome = AllowedOutcome(optionId=options[0].option_id, outcome="selected")
            return RequestPermissionResponse(outcome=outcome)

        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def session_update(
        self,
        session_id: str,
        update: UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AgentPlanUpdate
        | AvailableCommandsUpdate
        | CurrentModeUpdate
        | ConfigOptionUpdate
        | SessionInfoUpdate
        | UsageUpdate,
        **kwargs: object,
    ) -> None:
        del kwargs

        if not isinstance(update, AgentMessageChunk):
            return

        content = update.content
        text = "<content>"
        if isinstance(content, TextContentBlock):
            text = content.text
        elif isinstance(content, ImageContentBlock):
            text = "<image>"
        elif isinstance(content, AudioContentBlock):
            text = "<audio>"
        elif isinstance(content, ResourceContentBlock):
            text = content.uri or "<resource>"
        elif isinstance(content, EmbeddedResourceContentBlock):
            text = "<resource>"

        self._buffers.setdefault(session_id, []).append(text)

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
        auto_approve_permissions: bool = False,
        connector: AcpConnectionFactory = connect_to_agent,
        spawner: AcpSpawnFn = asyncio.create_subprocess_exec,
    ) -> None:
        self._registry = registry
        self._program = program
        self._args = args
        self._auto_approve_permissions = auto_approve_permissions
        self._connector = connector
        self._spawner = spawner
        self._live_by_chat: dict[int, _LiveSession] = {}

    async def new_session(self, *, chat_id: int, workspace: Path) -> str:
        workspace = self._normalize_workspace(workspace)
        if not workspace.is_dir():
            raise ValueError(workspace)

        existing = self._live_by_chat.pop(chat_id, None)
        if existing is not None:
            await self._shutdown(existing.process)

        process = await self._spawner(
            self._program,
            *self._args,
            stdin=aio_subprocess.PIPE,
            stdout=aio_subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError

        client = _AcpClient(auto_approve_permissions=self._auto_approve_permissions)
        connection = self._connector(client, process.stdin, process.stdout)
        await connection.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(),
            client_info=Implementation(name="telegram-acp-bot", title="Telegram ACP Bot", version="0.1.0"),
        )
        session = await connection.new_session(cwd=str(workspace), mcp_servers=[])

        self._registry.create_or_replace(chat_id=chat_id, workspace=workspace, session_id=session.session_id)
        self._live_by_chat[chat_id] = _LiveSession(
            acp_session_id=session.session_id,
            workspace=workspace,
            process=process,
            connection=connection,
            client=client,
        )
        return session.session_id

    async def prompt(self, *, chat_id: int, text: str) -> AgentReply | None:
        live = self._live_by_chat.get(chat_id)
        if live is None:
            return None

        async with live.prompt_lock:
            live.client.start_capture(live.acp_session_id)
            await live.connection.prompt(session_id=live.acp_session_id, prompt=[text_block(text)])
            response_text = live.client.finish_capture(live.acp_session_id)
            return AgentReply(text=response_text or "(no text response)")

    def get_workspace(self, *, chat_id: int) -> Path | None:
        session = self._registry.get(chat_id)
        return None if session is None else session.workspace

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

        await self._shutdown(live.process)
        self._registry.clear(chat_id)
        return True

    async def clear(self, *, chat_id: int) -> bool:
        return await self.stop(chat_id=chat_id)

    async def _shutdown(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return

        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except TimeoutError:
            process.kill()
            await process.wait()

    @staticmethod
    def _normalize_workspace(workspace: Path) -> Path:
        return workspace.expanduser().resolve()

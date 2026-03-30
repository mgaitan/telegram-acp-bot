"""Internal state dataclasses for live ACP sessions and pending interactions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from acp.core import ClientSideConnection
from acp.schema import PermissionOption, RequestPermissionResponse

from telegram_acp_bot.acp.models import PermissionMode

if TYPE_CHECKING:
    from telegram_acp_bot.acp.client import _AcpClient
    from telegram_acp_bot.acp.protocols import ProcessLike


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
    last_emitted_text: str = ""
    last_emit_monotonic: float = 0.0


@dataclass(slots=True)
class _PendingTextState:
    """Tracks incremental emission state for streamed non-tool text."""

    last_emitted_text: str = ""
    last_emit_monotonic: float = 0.0

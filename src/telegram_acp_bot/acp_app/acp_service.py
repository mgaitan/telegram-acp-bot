"""Compatibility re-exports for `telegram_acp_bot.acp_app.acp_service`.

The ACP service has been split into focused submodules. This module re-exports
every public and semi-public name so that existing import paths remain valid.

See also:
- `{py:mod}telegram_acp_bot.acp_app.protocols` — Protocol classes, exceptions, constants
- `{py:mod}telegram_acp_bot.acp_app.session` — internal state dataclasses
- `{py:mod}telegram_acp_bot.acp_app.client` — `_AcpClient`
- `{py:mod}telegram_acp_bot.acp_app.service` — `AcpAgentService`
"""

from __future__ import annotations

# Standard-library re-exports (were at module scope in the original acp_service.py).
import asyncio  # noqa: F401
import asyncio.subprocess as aio_subprocess  # noqa: F401
import base64  # noqa: F401
import logging  # noqa: F401
import mimetypes  # noqa: F401
from collections.abc import Awaitable, Callable  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
from importlib import metadata  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Protocol, cast  # noqa: F401
from urllib.parse import unquote, urlparse  # noqa: F401
from uuid import uuid4  # noqa: F401

# Third-party ACP re-exports.
from acp import PROTOCOL_VERSION, RequestError, connect_to_agent, text_block  # noqa: F401
from acp.core import ClientSideConnection  # noqa: F401
from acp.schema import (  # noqa: F401
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

# Internal re-exports from acp_app submodules.
from telegram_acp_bot.acp_app.client import _AcpClient  # noqa: F401
from telegram_acp_bot.acp_app.models import (  # noqa: F401
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
from telegram_acp_bot.acp_app.protocols import (  # noqa: F401
    INCREMENTAL_TEXT_BOUNDARY_CHARS,
    INCREMENTAL_TEXT_MIN_DELTA_CHARS,
    INCREMENTAL_TEXT_MIN_INTERVAL_SECONDS,
    MIN_NUMERIC_DOT_CHUNK_MIN_LENGTH,
    MIN_NUMERIC_DOT_PREFIX_LENGTH,
    PACKAGE_NAME,
    TERMINAL_TOOL_STATUSES,
    AcpConnectionFactory,
    AcpHandshakeTimeoutError,
    AcpSpawnFn,
    ProcessLike,
    SessionLoadNotSupportedError,
    _package_version,
)
from telegram_acp_bot.acp_app.service import AcpAgentService, PromptContentBlock  # noqa: F401
from telegram_acp_bot.acp_app.session import (  # noqa: F401
    _ActiveToolBlock,
    _LiveSession,
    _PendingPermission,
    _PendingTextState,
)
from telegram_acp_bot.core.session_registry import SessionRegistry  # noqa: F401
from telegram_acp_bot.logging_context import bind_log_context, log_text_preview  # noqa: F401
from telegram_acp_bot.mcp_channel_state import (  # noqa: F401
    load_last_session_id,
    load_session_chat_map,
    save_last_session_id,
    save_session_chat_map,
)


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

from importlib import metadata  # noqa: F401

from telegram_acp_bot.acp_app.client import _AcpClient  # noqa: F401
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
from telegram_acp_bot.acp_app.service import AcpAgentService  # noqa: F401
from telegram_acp_bot.acp_app.session import (  # noqa: F401
    _ActiveToolBlock,
    _LiveSession,
    _PendingPermission,
    _PendingTextState,
)

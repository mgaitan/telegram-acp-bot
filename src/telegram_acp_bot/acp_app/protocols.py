"""Protocols, exceptions, constants, and type aliases for the ACP service layer."""

from __future__ import annotations

import asyncio
from importlib import metadata
from typing import Protocol

from acp.core import ClientSideConnection

TERMINAL_TOOL_STATUSES = {"completed", "failed"}
MIN_NUMERIC_DOT_PREFIX_LENGTH = 2
MIN_NUMERIC_DOT_CHUNK_MIN_LENGTH = 2
INCREMENTAL_TEXT_MIN_INTERVAL_SECONDS = 0.10
INCREMENTAL_TEXT_MIN_DELTA_CHARS = 8
INCREMENTAL_TEXT_BOUNDARY_CHARS = ("\n", ".", "!", "?", ":", ";")
PACKAGE_NAME = "telegram-acp-bot"


def _package_version() -> str:
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return "unknown"


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

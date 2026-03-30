"""Internal data models and protocols for the Telegram transport layer."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from telegram import Update

from telegram_acp_bot.acp.models import (
    AgentActivityBlock,
    AgentReply,
    PermissionDecisionAction,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
    PromptFile,
    PromptImage,
    ResumableSession,
)


@dataclass(slots=True, frozen=True)
class _PromptInput:
    chat_id: int
    text: str
    images: tuple[PromptImage, ...]
    files: tuple[PromptFile, ...]
    cycle_id: str = field(default_factory=lambda: uuid4().hex[:12])


@dataclass(slots=True, frozen=True)
class _ResumeArgs:
    resume_index: int | None
    workspace: Path | None


@dataclass(slots=True, frozen=True)
class _RestartArgs:
    resume_index: int | None
    workspace: Path | None


@dataclass(slots=True)
class _PendingPrompt:
    """A user message queued while the agent is busy processing another prompt."""

    prompt_input: _PromptInput
    update: Update
    token: str
    notify_msg_id: int | None = field(default=None)


@dataclass(slots=True, frozen=True)
class _VerboseActivityMessage:
    """Tracked editable message for one verbose activity stream."""

    activity_id: str
    kind: str
    title: str
    message_id: int
    source_text: str


@dataclass(slots=True, frozen=True)
class _QueuedVerboseBlock:
    """Latest queued in-progress block waiting for the next coalesced flush."""

    chat_id: int
    slot_key: str
    block: AgentActivityBlock


class ChatRequiredError(ValueError):
    """Raised when a Telegram update does not include a chat object."""


class AgentService(Protocol):
    """Minimal protocol for the agent service consumed by `TelegramBridge`."""

    async def new_session(self, *, chat_id: int, workspace: Path) -> str: ...

    async def load_session(self, *, chat_id: int, session_id: str, workspace: Path) -> str: ...

    async def list_resumable_sessions(
        self,
        *,
        chat_id: int,
        workspace: Path | None = None,
    ) -> tuple[ResumableSession, ...] | None: ...

    async def prompt(
        self,
        *,
        chat_id: int,
        text: str,
        images: tuple[PromptImage, ...] = (),
        files: tuple[PromptFile, ...] = (),
    ) -> AgentReply | None: ...

    def get_workspace(self, *, chat_id: int) -> Path | None: ...

    def supports_session_loading(self, *, chat_id: int) -> bool | None: ...

    async def cancel(self, *, chat_id: int) -> bool: ...

    async def stop(self, *, chat_id: int) -> bool: ...

    async def clear(self, *, chat_id: int) -> bool: ...

    def get_permission_policy(self, *, chat_id: int) -> PermissionPolicy | None: ...

    async def set_session_permission_mode(self, *, chat_id: int, mode: PermissionMode) -> bool: ...

    async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool) -> bool: ...

    def set_permission_request_handler(
        self,
        handler: Callable[[PermissionRequest], Awaitable[None]] | None,
    ) -> None: ...

    def set_activity_event_handler(
        self,
        handler: Callable[[int, AgentActivityBlock], Awaitable[None]] | None,
    ) -> None: ...

    async def respond_permission_request(
        self,
        *,
        chat_id: int,
        request_id: str,
        action: PermissionDecisionAction,
    ) -> bool: ...

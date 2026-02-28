from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PermissionMode = Literal["deny", "approve", "ask"]
PermissionDecisionAction = Literal["once", "always", "deny"]
PermissionEventOutput = Literal["stdout", "off"]
StreamEventKind = Literal["message_chunk", "tool_start", "tool_progress", "plan_update"]


@dataclass(slots=True, frozen=True)
class AgentReply:
    """Normalized response from the agent layer."""

    text: str
    images: tuple[ImagePayload, ...] = ()
    files: tuple[FilePayload, ...] = ()


@dataclass(slots=True, frozen=True)
class ImagePayload:
    """Base64 image payload with mime information."""

    data_base64: str
    mime_type: str


@dataclass(slots=True, frozen=True)
class FilePayload:
    """Generic file payload for binary or text resources."""

    name: str
    mime_type: str | None = None
    text_content: str | None = None
    data_base64: str | None = None


@dataclass(slots=True, frozen=True)
class PromptImage:
    """Image input provided by Telegram user."""

    data_base64: str
    mime_type: str


@dataclass(slots=True, frozen=True)
class PromptFile:
    """File input provided by Telegram user."""

    name: str
    mime_type: str | None = None
    text_content: str | None = None
    data_base64: str | None = None


@dataclass(slots=True, frozen=True)
class PermissionPolicy:
    """Permission policy state for one chat session."""

    session_mode: PermissionMode
    next_prompt_auto_approve: bool


@dataclass(slots=True, frozen=True)
class PermissionRequest:
    """A runtime permission request awaiting user confirmation."""

    chat_id: int
    request_id: str
    tool_title: str
    tool_call_id: str
    available_actions: tuple[PermissionDecisionAction, ...]


@dataclass(slots=True, frozen=True)
class AgentStreamEvent:
    """Incremental session update emitted while an ACP prompt is running."""

    kind: StreamEventKind
    text: str

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PermissionMode = Literal["deny", "approve"]


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

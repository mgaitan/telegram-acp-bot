from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PermissionMode = Literal["deny", "approve"]


@dataclass(slots=True, frozen=True)
class AgentReply:
    """Normalized response from the agent layer."""

    text: str


@dataclass(slots=True, frozen=True)
class PermissionPolicy:
    """Permission policy state for one chat session."""

    session_mode: PermissionMode
    next_prompt_auto_approve: bool

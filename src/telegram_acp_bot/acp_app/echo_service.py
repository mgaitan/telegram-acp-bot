from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from telegram_acp_bot.core.session_registry import SessionRegistry


@dataclass(slots=True, frozen=True)
class AgentReply:
    """Normalized response from the agent layer."""

    text: str


class EchoAgentService:
    """Minimal async service used as ACP placeholder for the first MVP."""

    def __init__(self, registry: SessionRegistry) -> None:
        self._registry = registry

    async def new_session(self, *, chat_id: int, workspace: Path) -> str:
        session = self._registry.create_or_replace(chat_id=chat_id, workspace=workspace)
        return session.session_id

    async def prompt(self, *, chat_id: int, text: str) -> AgentReply | None:
        session = self._registry.get(chat_id)
        if session is None:
            return None

        short_id = session.session_id.split("-", maxsplit=1)[0]
        return AgentReply(text=f"[{short_id}] {text}")

    def get_workspace(self, *, chat_id: int) -> Path | None:
        session = self._registry.get(chat_id)
        return None if session is None else session.workspace

    async def cancel(self, *, chat_id: int) -> bool:
        return self._registry.get(chat_id) is not None

    async def stop(self, *, chat_id: int) -> bool:
        if self._registry.get(chat_id) is None:
            return False
        self._registry.clear(chat_id)
        return True

    async def clear(self, *, chat_id: int) -> bool:
        return await self.stop(chat_id=chat_id)

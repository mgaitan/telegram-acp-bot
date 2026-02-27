from __future__ import annotations

from pathlib import Path

from telegram_acp_bot.acp_app.models import AgentReply, PermissionMode, PermissionPolicy
from telegram_acp_bot.core.session_registry import SessionRegistry


class EchoAgentService:
    """Minimal async service used as ACP placeholder for the first MVP."""

    def __init__(self, registry: SessionRegistry) -> None:
        self._registry = registry
        self._session_permission_mode: dict[int, PermissionMode] = {}
        self._next_prompt_auto_approve: dict[int, bool] = {}

    async def new_session(self, *, chat_id: int, workspace: Path) -> str:
        session = self._registry.create_or_replace(chat_id=chat_id, workspace=workspace)
        self._session_permission_mode[chat_id] = "deny"
        self._next_prompt_auto_approve[chat_id] = False
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
        self._session_permission_mode.pop(chat_id, None)
        self._next_prompt_auto_approve.pop(chat_id, None)
        return True

    async def clear(self, *, chat_id: int) -> bool:
        return await self.stop(chat_id=chat_id)

    def get_permission_policy(self, *, chat_id: int) -> PermissionPolicy | None:
        if self._registry.get(chat_id) is None:
            return None
        return PermissionPolicy(
            session_mode=self._session_permission_mode.get(chat_id, "deny"),
            next_prompt_auto_approve=self._next_prompt_auto_approve.get(chat_id, False),
        )

    async def set_session_permission_mode(self, *, chat_id: int, mode: PermissionMode) -> bool:
        if self._registry.get(chat_id) is None:
            return False
        self._session_permission_mode[chat_id] = mode
        return True

    async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool) -> bool:
        if self._registry.get(chat_id) is None:
            return False
        self._next_prompt_auto_approve[chat_id] = enabled
        return True

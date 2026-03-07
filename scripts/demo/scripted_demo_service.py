from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from telegram_acp_bot.acp_app.models import (
    AgentActivityBlock,
    AgentReply,
    FilePayload,
    ImagePayload,
    PermissionDecisionAction,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
    PromptFile,
    PromptImage,
    ResumableSession,
)
from telegram_acp_bot.core.session_registry import SessionRegistry

SAMPLE_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAHgAAABQCAIAAACoyA+qAAAAA3NCSVQICAjb4U/gAAABQElEQVR4nO3QQQ3AIBDAsIP9d26X"
    "IEJC9gR5ZM18A6ft2wG8M2JmxMwImREzI2ZGzIyYGTEzYmbEzIiZETMjZkbMjJgZMTNiZsTMiJkRMyNmRsyMmBkxM2JmxMyImREz"
    "I2ZGzIyYGTEzYmbEzIiZETMjZkbMjJgZMTNiZsTMiJkRMyNmRsyMmBkxM2JmxMyImREzI2ZGzIyYGTEzYmbEzIiZETMjZkbMjJgZ"
    "MTNiZsTMiJkRMyNmRsyMmBkxM2JmxMyImREzI2ZGzIyYGTEzYmbEzIiZETMjZkbMjJgZMTNiZsTMiJkRMyNmRsyMmBkxM2JmxMyI"
    "mREzI2ZGzIyYGTEzYmbEzIiZETMjZkbMjJgZMTNiZsTMiJkRMyNmRsyMmBkxM2JmxMwIux8F+Yv2xjQAAAAASUVORK5CYII="
)


class ScriptedDemoAgentService:
    """Deterministic fake agent backend for landing/demo recordings."""

    def __init__(self, registry: SessionRegistry) -> None:
        self._registry = registry
        self._session_permission_mode: dict[int, PermissionMode] = {}
        self._next_prompt_auto_approve: dict[int, bool] = {}
        self._permission_prompt_handler: Callable[[PermissionRequest], Awaitable[None]] | None = None
        self._activity_event_handler: Callable[[int, AgentActivityBlock], Awaitable[None]] | None = None
        self._cancel_flags: dict[int, asyncio.Event] = {}

    async def new_session(self, *, chat_id: int, workspace: Path) -> str:
        workspace = self._prepare_workspace(workspace)
        session = self._registry.create_or_replace(chat_id=chat_id, workspace=workspace)
        self._session_permission_mode[chat_id] = "ask"
        self._next_prompt_auto_approve[chat_id] = False
        self._cancel_flags[chat_id] = asyncio.Event()
        return session.session_id

    async def load_session(self, *, chat_id: int, session_id: str, workspace: Path) -> str:
        workspace = self._prepare_workspace(workspace)
        session = self._registry.create_or_replace(chat_id=chat_id, workspace=workspace, session_id=session_id)
        self._cancel_flags[chat_id] = asyncio.Event()
        return session.session_id

    async def list_resumable_sessions(
        self,
        *,
        chat_id: int,
        workspace: Path | None = None,
    ) -> tuple[ResumableSession, ...] | None:
        del chat_id
        demo_sessions = (
            ResumableSession(
                session_id="demo-release-main",
                workspace=Path.cwd(),
                title="Release prep on main",
                updated_at="2026-03-06T21:12:00Z",
            ),
            ResumableSession(
                session_id="demo-hotfix-urgent",
                workspace=Path.cwd() / "hotfix",
                title="Hotfix regression triage",
                updated_at="2026-03-06T20:48:00Z",
            ),
            ResumableSession(
                session_id="demo-docs-preview",
                workspace=Path.cwd() / "docs",
                title="Docs preview deploy",
                updated_at="2026-03-06T20:05:00Z",
            ),
        )
        normalized = self._normalize_workspace(workspace)
        if normalized is None:
            return demo_sessions
        return tuple(item for item in demo_sessions if self._normalize_workspace(item.workspace) == normalized)

    async def prompt(
        self,
        *,
        chat_id: int,
        text: str,
        images: tuple[PromptImage, ...] = (),
        files: tuple[PromptFile, ...] = (),
    ) -> AgentReply | None:
        session = self._registry.get(chat_id)
        if session is None:
            return None

        cancel_flag = self._cancel_flags.setdefault(chat_id, asyncio.Event())
        cancel_flag.clear()

        lowered = text.lower()
        media_hint = f" [images={len(images)} files={len(files)}]" if images or files else ""

        if "webcam" in lowered and "image" in lowered:
            await self._emit(chat_id, "think", "Planning webcam capture")
            await self._emit(
                chat_id,
                "execute",
                "ffmpeg -f video4linux2 -i /dev/video0 -frames:v 1 /tmp/webcam-small.jpg",
            )
            await self._emit(chat_id, "execute", "telegram-channel/telegram_send_attachment")
            return AgentReply(
                text=(
                    "Captured `/tmp/webcam-small.jpg` and delivered it over Telegram through the MCP channel."
                    + media_hint
                ),
                images=(ImagePayload(data_base64=SAMPLE_IMAGE_BASE64, mime_type="image/png"),),
            )

        if "attachment" in lowered or "diagnostics" in lowered:
            await self._emit(chat_id, "think", "Preparing diagnostics artifact")
            await self._emit(
                chat_id,
                "execute",
                "python tools/collect_release_diagnostics.py --output /tmp/release-diagnostics.md",
            )
            await self._emit(chat_id, "execute", "telegram-channel/telegram_send_attachment")
            return AgentReply(
                text="Generated diagnostics report and sent it as attachment.",
                files=(
                    FilePayload(
                        name="release-diagnostics.md",
                        mime_type="text/markdown",
                        text_content=(
                            "# Release diagnostics\n\n"
                            "- CI: green\n"
                            "- Docs preview: green\n"
                            "- Pending: final reviewer ack\n"
                        ),
                    ),
                ),
            )

        if "merge" in lowered and "release" in lowered:
            steps = (
                ("think", "Evaluating review status"),
                ("search", "gh pr view 115 --json reviewDecision,statusCheckRollup"),
                ("execute", "gh pr merge 115 --merge --delete-branch"),
                ("execute", "git checkout main && git pull --ff-only"),
                ("edit", "Bump version to 0.1.2 and update changelog"),
                ("execute", "gh release create v0.1.2 --target main --generate-notes"),
            )
            for kind, title in steps:
                canceled = await self._emit(chat_id, kind, title, delay=0.9)
                if canceled:
                    return AgentReply(text="Canceled current workflow. Ready to process queued request.")
            return AgentReply(
                text=(
                    "No review objections found. Merged the PR, fast-forwarded `main`, "
                    "prepared the patch release flow, and drafted release notes." + media_hint
                )
            )

        short_id = session.session_id.split("-", maxsplit=1)[0]
        return AgentReply(text=f"[{short_id}] Acknowledged: {text}{media_hint}".strip())

    async def _emit(self, chat_id: int, kind: str, title: str, *, delay: float = 0.4) -> bool:
        handler = self._activity_event_handler
        if handler is not None:
            await handler(chat_id, AgentActivityBlock(kind=kind, title=title, status="in_progress"))
        await asyncio.sleep(delay)
        return self._cancel_flags.setdefault(chat_id, asyncio.Event()).is_set()

    def get_workspace(self, *, chat_id: int) -> Path | None:
        session = self._registry.get(chat_id)
        return None if session is None else session.workspace

    def supports_session_loading(self, *, chat_id: int) -> bool | None:
        del chat_id
        return True

    async def cancel(self, *, chat_id: int) -> bool:
        if self._registry.get(chat_id) is None:
            return False
        self._cancel_flags.setdefault(chat_id, asyncio.Event()).set()
        return True

    async def stop(self, *, chat_id: int) -> bool:
        if self._registry.get(chat_id) is None:
            return False
        self._registry.clear(chat_id)
        self._session_permission_mode.pop(chat_id, None)
        self._next_prompt_auto_approve.pop(chat_id, None)
        self._cancel_flags.pop(chat_id, None)
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

    def set_permission_request_handler(
        self,
        handler: Callable[[PermissionRequest], Awaitable[None]] | None,
    ) -> None:
        self._permission_prompt_handler = handler

    def set_activity_event_handler(
        self,
        handler: Callable[[int, AgentActivityBlock], Awaitable[None]] | None,
    ) -> None:
        self._activity_event_handler = handler

    async def respond_permission_request(
        self,
        *,
        chat_id: int,
        request_id: str,
        action: PermissionDecisionAction,
    ) -> bool:
        del chat_id, request_id, action
        return False

    @staticmethod
    def _prepare_workspace(workspace: Path) -> Path:
        resolved = workspace.expanduser()
        if resolved.exists() and not resolved.is_dir():
            raise ValueError(resolved)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @staticmethod
    def _normalize_workspace(workspace: Path | None) -> Path | None:
        if workspace is None:
            return None
        return workspace.expanduser().resolve()

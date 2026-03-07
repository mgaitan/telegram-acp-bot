from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
from pathlib import Path
from random import uniform

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

DEMO_DELAY_FACTOR = 1.10


def _load_demo_image_base64() -> str:
    demo_image_path = Path(__file__).with_name("demo.png")
    raw = demo_image_path.read_bytes()
    return base64.b64encode(raw).decode("ascii")


SAMPLE_IMAGE_BASE64 = _load_demo_image_base64()
SAMPLE_PDF_BASE64 = (
    "JVBERi0xLjQKMSAwIG9iajw8Pj5lbmRvYmoKMiAwIG9iajw8IC9UeXBlIC9DYXRhbG9nIC9QYWdlcyAzIDAgUiA+PmVuZG9iagoz"
    "IDAgb2JqPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFs0IDAgUl0gL0NvdW50IDEgPj5lbmRvYmoKNCAwIG9iajw8IC9UeXBlIC9QYWdl"
    "IC9QYXJlbnQgMyAwIFIgL01lZGlhQm94IFswIDAgMzAwIDE0NF0gL0NvbnRlbnRzIDUgMCBSIC9SZXNvdXJjZXMgPDwgL0ZvbnQg"
    "PDwgL0YxIDYgMCBSID4+ID4+ID4+ZW5kb2JqCjUgMCBvYmo8PCAvTGVuZ3RoIDQ0ID4+c3RyZWFtCkJUIC9GMSAxOCBUZiA3MiA5"
    "NiBUZCAoUmVsZWFzZSBkaWFnbm9zdGljcykgVGogRVQKZW5kc3RyZWFtIGVuZG9iago2IDAgb2JqPDwgL1R5cGUgL0ZvbnQgL1N1"
    "YnR5cGUgL1R5cGUxIC9CYXNlRm9udCAvSGVsdmV0aWNhID4+ZW5kb2JqCnhyZWYKMCA3CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAw"
    "MDAwMDAwOSAwMDAwMCBuIAowMDAwMDAwMDMwIDAwMDAwIG4gCjAwMDAwMDAwODcgMDAwMDAgbiAKMDAwMDAwMDE0NCAwMDAwMCBu"
    "IAowMDAwMDAwMjcwIDAwMDAwIG4gCjAwMDAwMDAzNjQgMDAwMDAgbiAKdHJhaWxlcjw8IC9Sb290IDIgMCBSIC9TaXplIDcgPj4K"
    "c3RhcnR4cmVmCjQ0NAolJUVPRgo="
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
                session_id="f548c412-8f9f-4f36-a9d2-6f3e1f9fd3f4",
                workspace=Path.cwd(),
                title="Release prep on main",
                updated_at="2026-03-06T21:12:00Z",
            ),
            ResumableSession(
                session_id="9a6e0f55-7cb6-4e66-a4ae-3f0f9fd5c8bc",
                workspace=Path.cwd() / "hotfix",
                title="Hotfix regression triage",
                updated_at="2026-03-06T20:48:00Z",
            ),
            ResumableSession(
                session_id="3d3f9b2e-97e6-4659-b52c-9e3f0f2fbf73",
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

        if any(keyword in lowered for keyword in ("webcam", "photo", "image", "picture")):
            await self._emit(
                chat_id,
                "think",
                "",
                text="I can do that now. I will capture one frame from `/dev/video0` and send it right away.",
                delay=uniform(1.2, 2.6) * DEMO_DELAY_FACTOR,
            )
            await self._emit(
                chat_id,
                "execute",
                "Run ffmpeg -f video4linux2 -i /dev/video0 -frames:v 1 /tmp/webcam-small.jpg",
                text="",
                delay=uniform(1.6, 3.4) * DEMO_DELAY_FACTOR,
            )
            await self._emit(
                chat_id,
                "execute",
                "Run telegram-channel/telegram_send_attachment",
                text="Uploaded `/tmp/webcam-small.jpg` as a photo via the Telegram MCP channel.",
                delay=uniform(1.0, 2.2) * DEMO_DELAY_FACTOR,
            )
            return AgentReply(
                text=("Done. I captured `/tmp/webcam-small.jpg` and sent it to this chat." + media_hint),
                images=(ImagePayload(data_base64=SAMPLE_IMAGE_BASE64, mime_type="image/png"),),
            )

        if any(keyword in lowered for keyword in ("attachment", "diagnostics", "report", "pdf")):
            await self._emit(
                chat_id,
                "think",
                "",
                text="Generating a compact diagnostics report in PDF format and attaching it here.",
                delay=uniform(1.0, 2.4) * DEMO_DELAY_FACTOR,
            )
            await self._emit(
                chat_id,
                "execute",
                "Run python tools/collect_release_diagnostics.py --format pdf --output /tmp/release-diagnostics.pdf",
                text="CI checks: green. Docs preview: green. Pending: final reviewer ack.",
                delay=uniform(1.4, 3.0) * DEMO_DELAY_FACTOR,
            )
            await self._emit(
                chat_id,
                "execute",
                "Run telegram-channel/telegram_send_attachment",
                text="Attached `/tmp/release-diagnostics.pdf` to this conversation.",
                delay=uniform(1.0, 2.1) * DEMO_DELAY_FACTOR,
            )
            return AgentReply(
                text="Diagnostics PDF is ready and attached.",
                files=(
                    FilePayload(
                        name="release-diagnostics.pdf",
                        mime_type="application/pdf",
                        data_base64=SAMPLE_PDF_BASE64,
                    ),
                ),
            )

        if "merge" in lowered and "release" in lowered:
            steps = (
                (
                    "think",
                    "",
                    "Checking unresolved comments, required reviews, and status checks before merging.",
                    (3.8, 5.4),
                ),
                (
                    "execute",
                    "Run gh pr view 92 --json reviewDecision,reviews,statusCheckRollup",
                    "No blocking review comments found. Required checks look healthy.",
                    (2.2, 3.6),
                ),
                (
                    "execute",
                    "Run gh pr merge 92 --merge --delete-branch, Run git checkout main, Run git pull --ff-only",
                    "",
                    (2.6, 4.1),
                ),
                (
                    "edit",
                    "Edit pyproject.toml",
                    "Bumped patch version and aligned CLI/docs notes for the release.",
                    (1.8, 3.0),
                ),
                (
                    "execute",
                    "Run uv run pytest tests/test_acp_service.py",
                    "Test suite for ACP service passed.",
                    (1.7, 2.9),
                ),
                (
                    "execute",
                    "Run gh release create v0.1.2 --target main --generate-notes",
                    "Draft release created with auto-generated notes.",
                    (1.6, 2.8),
                ),
            )
            for kind, title, text_body, delay_range in steps:
                canceled = await self._emit(
                    chat_id,
                    kind,
                    title,
                    text=text_body,
                    delay=uniform(*delay_range) * DEMO_DELAY_FACTOR,
                )
                if canceled:
                    return AgentReply(text="Canceled current workflow. Ready to process queued request.")
            await asyncio.sleep(uniform(0.8, 1.8) * DEMO_DELAY_FACTOR)
            return AgentReply(
                text=(
                    "Looks good: no unresolved blockers. I merged the PR, synced `main`, "
                    "ran the ACP service tests, and prepared the patch release notes." + media_hint
                )
            )

        short_id = session.session_id.split("-", maxsplit=1)[0]
        return AgentReply(text=f"[{short_id}] Acknowledged: {text}{media_hint}".strip())

    async def _emit(
        self,
        chat_id: int,
        kind: str,
        title: str,
        *,
        text: str = "",
        delay: float = 1.0,
    ) -> bool:
        handler = self._activity_event_handler
        if handler is not None:
            await handler(chat_id, AgentActivityBlock(kind=kind, title=title, status="completed", text=text))
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

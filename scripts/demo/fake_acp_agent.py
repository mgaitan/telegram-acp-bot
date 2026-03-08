# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "agent-client-protocol>=0.8.1",
#   "python-dotenv>=1.2.1",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Never, cast
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    Agent,
    RequestError,
    embedded_blob_resource,
    image_block,
    resource_block,
    run_agent,
    start_tool_call,
    update_agent_message,
    update_agent_message_text,
    update_tool_call,
)
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    AudioContentBlock,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    NewSessionResponse,
    PromptResponse,
    ResourceContentBlock,
    SessionCapabilities,
    SessionInfo,
    SessionListCapabilities,
    SessionResumeCapabilities,
    SseMcpServer,
    TextContentBlock,
    ToolKind,
)
from demo_scenario import DemoScenario, ScenarioAsset, load_demo_scenario
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

MISSING_CONNECTION_ERROR = "Agent connection is not available yet."
STOP_REASON_END_TURN = "end_turn"
STOP_REASON_CANCELLED = "cancelled"
NOTIFICATION_FLUSH_DELAY_SECONDS = 0.05

PromptBlock = (
    TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock
)
McpServer = HttpMcpServer | SseMcpServer | McpServerStdio


@dataclass(slots=True)
class _SessionState:
    cwd: Path
    cancel_event: asyncio.Event
    pending_route_id: str | None = None
    pending_event_index: int = 0
    resume_task: asyncio.Task[None] | None = None


class FakeDemoAcpAgent(Agent):
    """ACP fake agent that replays a declarative demo scenario."""

    def __init__(self, scenario: DemoScenario) -> None:
        self._scenario = scenario
        self._conn: Client | None = None
        self._sessions: dict[str, _SessionState] = {}

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: object | None = None,
        client_info: object | None = None,
        **kwargs: object,
    ) -> InitializeResponse:
        del client_capabilities, client_info, kwargs
        return InitializeResponse(
            protocol_version=min(protocol_version, PROTOCOL_VERSION),
            agent_info=Implementation(name="fake-demo-acp-agent", title="Fake Demo ACP Agent", version="0.1.0"),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[McpServer] | None = None,
        **kwargs: object,
    ) -> NewSessionResponse:
        del mcp_servers, kwargs
        session_id = str(uuid4())
        self._sessions[session_id] = _SessionState(cwd=self._resolved_cwd(cwd), cancel_event=asyncio.Event())
        return NewSessionResponse(session_id=session_id)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[McpServer] | None = None,
        **kwargs: object,
    ) -> LoadSessionResponse:
        del mcp_servers, kwargs
        self._sessions[session_id] = _SessionState(cwd=self._resolved_cwd(cwd), cancel_event=asyncio.Event())
        return LoadSessionResponse()

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: object,
    ) -> ListSessionsResponse:
        del cursor, kwargs
        now_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        base_cwd = self._resolved_cwd(cwd) if cwd else self._resolved_cwd(str(Path.cwd()))
        sessions = [
            SessionInfo(
                session_id="f548c412-8f9f-4f36-a9d2-6f3e1f9fd3f4",
                cwd=str(base_cwd),
                title="Ship it and pray",
                updated_at=now_iso,
            ),
            SessionInfo(
                session_id="9a6e0f55-7cb6-4e66-a4ae-3f0f9fd5c8bc",
                cwd=str(base_cwd / "hotfix"),
                title="Hotfix rodeo with no coffee",
                updated_at=now_iso,
            ),
            SessionInfo(
                session_id="3d3f9b2e-97e6-4659-b52c-9e3f0f2fbf73",
                cwd=str(base_cwd / "docs"),
                title="Docs, dragons, and deadlines",
                updated_at=now_iso,
            ),
        ]
        return ListSessionsResponse(sessions=sessions)

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: object) -> None:
        del mode_id, session_id, kwargs

    async def set_session_model(self, model_id: str, session_id: str, **kwargs: object) -> None:
        del model_id, session_id, kwargs

    async def set_config_option(self, config_id: str, session_id: str, value: str, **kwargs: object) -> None:
        del config_id, session_id, value, kwargs

    async def authenticate(self, method_id: str, **kwargs: object) -> None:
        del method_id, kwargs

    async def prompt(self, prompt: list[PromptBlock], session_id: str, **kwargs: object) -> PromptResponse:
        del kwargs
        session = self._sessions.get(session_id)
        if session is None:
            raise RequestError.invalid_params({"session_id": session_id})

        session.cancel_event = asyncio.Event()
        prompt_text = self._prompt_text(prompt)
        route = self._scenario.route_for_prompt(prompt_text)
        if route is None:
            short_id = session_id.split("-", maxsplit=1)[0]
            await self._notify_agent_text(session_id, f"[{short_id}] Acknowledged: {prompt_text}".strip())
            await self._flush_notifications()
            return PromptResponse(stop_reason=STOP_REASON_END_TURN)

        for index, event in enumerate(route.events, start=1):
            if await self._sleep_with_cancel(session.cancel_event, event.delay_ms):
                self._store_interrupted_route(session=session, route_id=route.id, next_event_index=index - 1)
                reply_text = "" if route.id == "release_flow" else route.cancel_reply_text
                return await self._cancelled_response(session_id, reply_text)
            await self._emit_tool_event(
                session_id=session_id,
                tool_call_id=f"{route.id}-{index}",
                kind=event.kind,
                title=event.title,
                text=event.text,
            )
            if session.cancel_event.is_set():
                self._store_interrupted_route(session=session, route_id=route.id, next_event_index=index)
                reply_text = "" if route.id == "release_flow" else route.cancel_reply_text
                return await self._cancelled_response(session_id, reply_text)

        for asset_id in route.final_images:
            await self._notify_image_asset(session_id, self._scenario.assets[asset_id])
        for asset_id in route.final_files:
            await self._notify_file_asset(session_id, self._scenario.assets[asset_id])
        if route.final_text.strip():
            await self._notify_agent_text(session_id, route.final_text)
        if route.id == "webcam_flow" and session.pending_route_id is not None:
            self._schedule_resume_interrupted_route(session_id=session_id, session=session)

        await self._flush_notifications()
        return PromptResponse(stop_reason=STOP_REASON_END_TURN)

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[McpServer] | None = None,
        **kwargs: object,
    ) -> Never:
        del cwd, session_id, mcp_servers, kwargs
        raise RequestError.method_not_found("session/fork")

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[McpServer] | None = None,
        **kwargs: object,
    ) -> Never:
        del cwd, session_id, mcp_servers, kwargs
        raise RequestError.method_not_found("session/resume")

    async def cancel(self, session_id: str, **kwargs: object) -> None:
        del kwargs
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.cancel_event.set()

    async def ext_method(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "session/list":
            cursor = params.get("cursor")
            cwd = params.get("cwd")
            response = await self.list_sessions(
                cursor=None if cursor is None else str(cursor),
                cwd=None if cwd is None else str(cwd),
            )
            return response.model_dump()
        if method == "session/load":
            session_id = str(params.get("session_id", ""))
            cwd = str(params.get("cwd", Path.cwd()))
            if not session_id:
                raise RequestError.invalid_params({"session_id": session_id})
            response = await self.load_session(cwd=cwd, session_id=session_id)
            return response.model_dump()
        raise RequestError.method_not_found("ext/method")

    async def ext_notification(self, method: str, params: dict[str, object]) -> None:
        del method, params
        raise RequestError.method_not_found("ext/notification")

    @staticmethod
    def _prompt_text(prompt_blocks: list[PromptBlock]) -> str:
        chunks = [block.text for block in prompt_blocks if isinstance(block, TextContentBlock)]
        return "\n".join(item.strip() for item in chunks if item.strip())

    async def _emit_tool_event(self, *, session_id: str, tool_call_id: str, kind: str, title: str, text: str) -> None:
        conn = self._require_conn()
        normalized_kind = cast(ToolKind, kind or "other")
        await conn.session_update(
            session_id=session_id,
            update=start_tool_call(tool_call_id=tool_call_id, title=title or "tool", kind=normalized_kind),
        )
        if text.strip():
            await conn.session_update(session_id=session_id, update=update_agent_message_text(text))
        await conn.session_update(
            session_id=session_id,
            update=update_tool_call(tool_call_id=tool_call_id, status="completed"),
        )

    async def _notify_agent_text(self, session_id: str, text: str) -> None:
        conn = self._require_conn()
        await conn.session_update(session_id=session_id, update=update_agent_message_text(text))

    async def _notify_image_asset(self, session_id: str, asset: ScenarioAsset) -> None:
        conn = self._require_conn()
        await conn.session_update(
            session_id=session_id,
            update=update_agent_message(image_block(data=asset.data_base64(), mime_type=asset.mime_type)),
        )

    async def _notify_file_asset(self, session_id: str, asset: ScenarioAsset) -> None:
        conn = self._require_conn()
        uri = f"file:///tmp/{asset.name}"
        await conn.session_update(
            session_id=session_id,
            update=update_agent_message(
                resource_block(embedded_blob_resource(uri=uri, blob=asset.data_base64(), mime_type=asset.mime_type))
            ),
        )

    async def _cancelled_response(self, session_id: str, reply_text: str) -> PromptResponse:
        if reply_text.strip():
            await self._notify_agent_text(session_id, reply_text)
        await self._flush_notifications()
        return PromptResponse(stop_reason=STOP_REASON_CANCELLED)

    def _store_interrupted_route(self, *, session: _SessionState, route_id: str, next_event_index: int) -> None:
        if route_id != "release_flow":
            return
        session.pending_route_id = route_id
        session.pending_event_index = max(0, next_event_index)

    def _schedule_resume_interrupted_route(self, *, session_id: str, session: _SessionState) -> None:
        if session.resume_task is not None and not session.resume_task.done():
            session.resume_task.cancel()
        session.resume_task = asyncio.create_task(
            self._resume_interrupted_route(session_id=session_id, session=session)
        )

    async def _resume_interrupted_route(self, *, session_id: str, session: _SessionState) -> None:
        route_id = session.pending_route_id
        if route_id is None:
            return
        route = next((item for item in self._scenario.agent_routes if item.id == route_id), None)
        session.pending_route_id = None
        start_index = session.pending_event_index
        session.pending_event_index = 0
        if route is None:
            return

        await asyncio.sleep(1.0)
        for index, event in enumerate(route.events[start_index:], start=start_index + 1):
            await self._sleep_with_cancel(asyncio.Event(), event.delay_ms)
            await self._emit_tool_event(
                session_id=session_id,
                tool_call_id=f"{route.id}-resume-{index}",
                kind=event.kind,
                title=event.title,
                text=event.text,
            )
        if route.final_text.strip():
            await self._notify_agent_text(session_id, route.final_text)
        await self._flush_notifications()

    @staticmethod
    async def _sleep_with_cancel(cancel_event: asyncio.Event, delay_ms: int) -> bool:
        if delay_ms <= 0:
            return cancel_event.is_set()
        timeout_seconds = delay_ms / 1000
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=timeout_seconds)
        except TimeoutError:
            return False
        return True

    @staticmethod
    def _resolved_cwd(raw_cwd: str) -> Path:
        return Path(raw_cwd).expanduser().resolve()

    def _require_conn(self) -> Client:
        if self._conn is None:
            raise RuntimeError(MISSING_CONNECTION_ERROR)
        return self._conn

    @staticmethod
    async def _flush_notifications() -> None:
        await asyncio.sleep(NOTIFICATION_FLUSH_DELAY_SECONDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fake ACP agent driven by demo scenario.")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path(__file__).with_name("demo_story.json"),
        help="Path to demo scenario JSON.",
    )
    parser.add_argument("--log-level", default="INFO", help="Log level for fake ACP agent.")
    return parser.parse_args()


async def _main() -> int:
    load_dotenv(override=False)
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    scenario = load_demo_scenario(args.scenario)
    agent = FakeDemoAcpAgent(scenario)
    await run_agent(agent, use_unstable_protocol=True)
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())

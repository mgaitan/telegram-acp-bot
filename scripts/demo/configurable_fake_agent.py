# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "agent-client-protocol>=0.8.1",
#   "python-dotenv>=1.2.1",
#   "pyyaml>=6.0",
# ]
# ///
"""Configurable fake ACP agent driven by a YAML rule file.

Run with::

    uv run scripts/demo/configurable_fake_agent.py --script my_rules.yaml

or use it as an ``--agent-command`` when starting the Telegram bot::

    telegram-acp-bot --agent-command \\
        "uv run scripts/demo/configurable_fake_agent.py --script my_rules.yaml" \\
        ...

See `scripts/demo/example_rules.yaml` for the supported YAML format.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never, cast
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    Agent,
    RequestError,
    run_agent,
    start_tool_call,
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
    ToolCallStatus,
    ToolKind,
)
from dotenv import load_dotenv
from rule_config import Rule, RuleConfig, Step, load_rule_config, render_step

logger = logging.getLogger(__name__)

MISSING_CONNECTION_ERROR = "Agent connection is not available yet."
STOP_REASON_END_TURN = "end_turn"
NOTIFICATION_FLUSH_DELAY_SECONDS = 0.05

PromptBlock = (
    TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock
)
McpServer = HttpMcpServer | SseMcpServer | McpServerStdio


@dataclass(slots=True)
class _SessionState:
    cwd: Path
    cancel_event: asyncio.Event
    last_user_message: str = ""
    last_rule: str = ""
    session_variables: dict[str, Any] = field(default_factory=dict)


class ConfigurableFakeAcpAgent(Agent):
    """ACP fake agent driven by a YAML rule configuration.

    Each incoming prompt is matched against the configured rules in order.
    The first matching rule's `then` steps are executed. If no rule matches,
    the agent acknowledges the message with a generic echo reply.

    Per-session state (last_user_message, last_rule, and any variables set by
    rules via `set_variables`) is available in step templates as
    ``{{ state.<key> }}``.

    See also `rule_config.load_rule_config` for the YAML format.
    """

    def __init__(self, config: RuleConfig) -> None:
        self._config = config
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
            agent_info=Implementation(
                name="configurable-fake-acp-agent",
                title="Configurable Fake ACP Agent",
                version="0.1.0",
            ),
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
        self._sessions[session_id] = _SessionState(
            cwd=self._resolved_cwd(cwd),
            cancel_event=asyncio.Event(),
        )
        return NewSessionResponse(session_id=session_id)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[McpServer] | None = None,
        **kwargs: object,
    ) -> LoadSessionResponse:
        del mcp_servers, kwargs
        self._sessions[session_id] = _SessionState(
            cwd=self._resolved_cwd(cwd),
            cancel_event=asyncio.Event(),
        )
        return LoadSessionResponse()

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: object,
    ) -> ListSessionsResponse:
        del cursor, kwargs
        now_iso = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        base_cwd = self._resolved_cwd(cwd) if cwd else Path.cwd()
        sessions = [
            SessionInfo(
                session_id=str(uuid4()),
                cwd=str(base_cwd),
                title="Configurable fake session",
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
        message = self._extract_text(prompt)

        # Capture state from previous turn BEFORE applying rule's set_variables.
        state = self._build_state(session)

        rule = self._config.find_rule(message)
        if rule is None:
            await self._echo_reply(session_id=session_id, message=message)
            await self._flush()
            session.last_user_message = message
            return PromptResponse(stop_reason=STOP_REASON_END_TURN)

        # Apply set_variables declared in the rule so templates can reference them.
        if rule.set_variables:
            session.session_variables.update(rule.set_variables)
            state = self._build_state(session)

        await self._execute_rule(session_id=session_id, rule=rule, message=message, state=state)
        await self._flush()

        # Update per-turn state after execution.
        session.last_user_message = message
        session.last_rule = rule.name
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
        if session is not None:
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

    # ------------------------------------------------------------------
    # Rule execution
    # ------------------------------------------------------------------

    async def _execute_rule(self, *, session_id: str, rule: Rule, message: str, state: dict[str, Any]) -> None:
        """Execute all steps of *rule* for the given *message*."""
        open_tool_id: str | None = None

        for step in rule.steps:
            rendered = render_step(step, message=message, variables=self._config.variables, state=state)
            open_tool_id = await self._execute_step(
                session_id=session_id,
                step=rendered,
                open_tool_id=open_tool_id,
            )

        # Close any dangling open tool call (e.g. tool_call without matching tool_result).
        if open_tool_id is not None:
            await self._close_tool(session_id=session_id, tool_call_id=open_tool_id)

    async def _execute_step(
        self,
        *,
        session_id: str,
        step: Step,
        open_tool_id: str | None,
    ) -> str | None:
        """Execute a single *step* and return the new open_tool_id (or `None`)."""
        match step.type:
            case "thought":
                return await self._execute_thought(session_id=session_id, step=step, open_tool_id=open_tool_id)
            case "tool_call":
                return await self._execute_tool_call(session_id=session_id, step=step, open_tool_id=open_tool_id)
            case "tool_result":
                return await self._execute_tool_result(session_id=session_id, step=step, open_tool_id=open_tool_id)
            case "final":
                return await self._execute_final(session_id=session_id, step=step, open_tool_id=open_tool_id)
            case "sleep":
                return await self._execute_sleep(session_id=session_id, step=step, open_tool_id=open_tool_id)
            case "error":
                return await self._execute_error(session_id=session_id, step=step, open_tool_id=open_tool_id)
            case _:
                raise ValueError(f"Unknown step type: {step.type!r}")  # noqa: TRY003

    async def _execute_thought(self, *, session_id: str, step: Step, open_tool_id: str | None) -> str | None:
        tool_id = str(uuid4())
        await self._start_tool(session_id=session_id, tool_call_id=tool_id, kind="think", title="Thinking")
        if step.content.strip():
            await self._send_text(session_id=session_id, text=step.content)
        await self._close_tool(session_id=session_id, tool_call_id=tool_id)
        return open_tool_id

    async def _execute_tool_call(self, *, session_id: str, step: Step, open_tool_id: str | None) -> str | None:
        if open_tool_id is not None:
            await self._close_tool(session_id=session_id, tool_call_id=open_tool_id)
        tool_id = str(uuid4())
        title = step.tool_name or "tool"
        await self._start_tool(session_id=session_id, tool_call_id=tool_id, kind="execute", title=title)
        if step.args:
            args_text = "\n".join(f"{k}: {v}" for k, v in step.args.items())
            await self._send_text(session_id=session_id, text=args_text)
        return tool_id  # leave open — waiting for tool_result

    async def _execute_tool_result(self, *, session_id: str, step: Step, open_tool_id: str | None) -> str | None:
        if open_tool_id is not None:
            if step.content.strip():
                await self._send_text(session_id=session_id, text=step.content)
            await self._close_tool(session_id=session_id, tool_call_id=open_tool_id)
            return None
        if step.content.strip():
            await self._send_text(session_id=session_id, text=step.content)
        return None

    async def _execute_final(self, *, session_id: str, step: Step, open_tool_id: str | None) -> str | None:
        if step.content.strip():
            await self._send_text(session_id=session_id, text=step.content)
        return open_tool_id

    async def _execute_sleep(self, *, session_id: str, step: Step, open_tool_id: str | None) -> str | None:
        del session_id
        if step.sleep_ms > 0:
            await asyncio.sleep(step.sleep_ms / 1000.0)
        return open_tool_id

    async def _execute_error(self, *, session_id: str, step: Step, open_tool_id: str | None) -> str | None:
        # Close any currently open tool as failed first.
        if open_tool_id is not None:
            await self._close_tool(session_id=session_id, tool_call_id=open_tool_id, status="failed")
        # Emit a new tool event that immediately fails.
        tool_id = str(uuid4())
        title = step.tool_name or "error"
        await self._start_tool(session_id=session_id, tool_call_id=tool_id, kind="execute", title=title)
        if step.content.strip():
            await self._send_text(session_id=session_id, text=step.content)
        await self._close_tool(session_id=session_id, tool_call_id=tool_id, status="failed")
        return None

    # ------------------------------------------------------------------
    # Low-level ACP helpers
    # ------------------------------------------------------------------

    async def _start_tool(self, *, session_id: str, tool_call_id: str, kind: str, title: str) -> None:
        conn = self._require_conn()
        await conn.session_update(
            session_id=session_id,
            update=start_tool_call(tool_call_id=tool_call_id, title=title, kind=cast(ToolKind, kind)),
        )

    async def _close_tool(self, *, session_id: str, tool_call_id: str, status: str = "completed") -> None:
        conn = self._require_conn()
        await conn.session_update(
            session_id=session_id,
            update=update_tool_call(tool_call_id=tool_call_id, status=cast(ToolCallStatus, status)),
        )

    async def _send_text(self, *, session_id: str, text: str) -> None:
        conn = self._require_conn()
        await conn.session_update(session_id=session_id, update=update_agent_message_text(text))

    async def _echo_reply(self, *, session_id: str, message: str) -> None:
        short_id = session_id.split("-", maxsplit=1)[0]
        await self._send_text(session_id=session_id, text=f"[{short_id}] {message}".strip())

    def _require_conn(self) -> Client:
        if self._conn is None:
            raise RuntimeError(MISSING_CONNECTION_ERROR)
        return self._conn

    @staticmethod
    def _build_state(session: _SessionState) -> dict[str, Any]:
        return {
            "last_user_message": session.last_user_message,
            "last_rule": session.last_rule,
            **session.session_variables,
        }

    @staticmethod
    async def _flush() -> None:
        await asyncio.sleep(NOTIFICATION_FLUSH_DELAY_SECONDS)

    @staticmethod
    def _extract_text(prompt_blocks: list[PromptBlock]) -> str:
        chunks = [block.text for block in prompt_blocks if isinstance(block, TextContentBlock)]
        return "\n".join(item.strip() for item in chunks if item.strip())

    @staticmethod
    def _resolved_cwd(raw_cwd: str) -> Path:
        return Path(raw_cwd).expanduser().resolve()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configurable fake ACP agent from a YAML rule file.")
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("example_rules.yaml"),
        help="Path to the YAML rules file (default: example_rules.yaml next to this script).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument("--log-level", default="INFO", help="Log level (overridden by --verbose).")
    return parser.parse_args()


async def _main() -> int:
    load_dotenv(override=False)
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=log_level)
    config = load_rule_config(args.script)
    agent = ConfigurableFakeAcpAgent(config)
    await run_agent(agent, use_unstable_protocol=True)
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())

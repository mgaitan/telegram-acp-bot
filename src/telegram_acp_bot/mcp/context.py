"""Request-context helpers shared by Telegram MCP tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from telegram_acp_bot.mcp.state import (
    STATE_FILE_ENV,
    TOKEN_ENV,
    load_last_session_id,
    load_session_chat_map,
)


@dataclass(frozen=True, slots=True)
class RequestContext:
    token: str
    chat_id: int
    session_id: str


def resolve_request_context(*, session_id: str | None) -> RequestContext | str:
    """Resolve Telegram routing context for an MCP tool invocation."""

    token = os.getenv(TOKEN_ENV, "").strip()
    if not token:
        return f"missing {TOKEN_ENV}"
    state_file_raw = os.getenv(STATE_FILE_ENV, "").strip()
    if not state_file_raw:
        return f"missing {STATE_FILE_ENV}"

    state_file = Path(state_file_raw)
    mapping = load_session_chat_map(state_file)
    selected_session_id = (session_id or "").strip() or None
    if selected_session_id is None:
        if len(mapping) == 1:
            selected_session_id = next(iter(mapping))
        else:
            last_session_id = load_last_session_id(state_file)
            if last_session_id and last_session_id in mapping:
                selected_session_id = last_session_id
    if selected_session_id is None:
        if len(mapping) > 1:
            candidates = ", ".join(sorted(mapping))
            return (
                "missing session_id: multiple active sessions exist and no last active session could be inferred. "
                f"Available session_ids: {candidates}"
            )
        return "missing session_id and no active session could be inferred"

    chat_id = mapping.get(selected_session_id)
    if chat_id is None:
        return f"unknown session_id `{selected_session_id}`"
    return RequestContext(token=token, chat_id=chat_id, session_id=selected_session_id)

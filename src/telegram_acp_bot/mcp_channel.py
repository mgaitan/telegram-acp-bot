"""Internal MCP channel exposed by the Telegram ACP bot.

This is a minimal server that is auto-registered for ACP sessions.
It currently exposes attachment delivery over Telegram Bot API.
"""

from __future__ import annotations

import base64
import binascii
import mimetypes
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from telegram import Bot, InputFile

from telegram_acp_bot.mcp_channel_state import (
    STATE_FILE_ENV,
    TOKEN_ENV,
    load_last_session_id,
    load_session_chat_map,
)

mcp = FastMCP(
    name="telegram-channel",
    instructions=(
        "Channel helper tools for Telegram clients. Use these tools when you need channel-specific behavior."
    ),
)


@dataclass(frozen=True, slots=True)
class _RequestContext:
    token: str
    chat_id: int
    session_id: str


@dataclass(frozen=True, slots=True)
class _AttachmentPayload:
    raw: bytes
    filename: str
    guessed_mime: str | None


@mcp.tool(
    name="telegram_channel_info",
    description="Return capabilities currently exposed by the Telegram channel MCP server.",
)
def telegram_channel_info() -> dict[str, object]:
    return {
        "supports_attachment_delivery": True,
        "supports_followup_buttons": False,
        "status": "active",
    }


@mcp.tool(
    name="telegram_send_attachment",
    description=(
        "Send an attachment to the current Telegram chat for the provided ACP session id. "
        "Use this when the user asks to send an image/file."
    ),
)
async def telegram_send_attachment(
    session_id: str | None = None,
    path: str | None = None,
    data_base64: str | None = None,
    name: str | None = None,
    mime_type: str | None = None,
) -> dict[str, object]:
    def fail(error: str) -> dict[str, object]:
        return {"ok": False, "error": error}

    context = _resolve_request_context(session_id=session_id, path=path, data_base64=data_base64)
    if isinstance(context, str):
        return fail(context)

    loaded = _load_attachment_bytes(path=path, data_base64=data_base64, name=name)
    if isinstance(loaded, str):
        return fail(loaded)

    resolved_mime = mime_type or loaded.guessed_mime or "application/octet-stream"
    bot = Bot(token=context.token)
    input_file = InputFile(BytesIO(loaded.raw), filename=loaded.filename)
    if resolved_mime.startswith("image/"):
        await bot.send_photo(chat_id=context.chat_id, photo=input_file)
        delivered_as = "photo"
    else:
        await bot.send_document(chat_id=context.chat_id, document=input_file)
        delivered_as = "document"

    return {
        "ok": True,
        "session_id": context.session_id,
        "chat_id": context.chat_id,
        "delivered_as": delivered_as,
        "name": loaded.filename,
        "mime_type": resolved_mime,
    }


def _load_attachment_bytes(
    *,
    path: str | None,
    data_base64: str | None,
    name: str | None,
) -> _AttachmentPayload | str:
    if path is not None:
        source_path = Path(path).expanduser().resolve(strict=False)
        if not source_path.is_file():
            return f"file not found: {source_path}"
        raw = source_path.read_bytes()
        filename = name or source_path.name
        guessed_mime = mimetypes.guess_type(source_path.name)[0]
        return _AttachmentPayload(raw=raw, filename=filename, guessed_mime=guessed_mime)

    assert data_base64 is not None
    try:
        raw = base64.b64decode(data_base64)
    except (ValueError, binascii.Error):
        return "invalid base64 payload"
    filename = name or "attachment.bin"
    guessed_mime = mimetypes.guess_type(filename)[0]
    return _AttachmentPayload(raw=raw, filename=filename, guessed_mime=guessed_mime)


def _resolve_request_context(
    *,
    session_id: str | None,
    path: str | None,
    data_base64: str | None,
) -> _RequestContext | str:
    if bool(path) == bool(data_base64):
        return "provide exactly one of `path` or `data_base64`"

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
        selected_session_id = load_last_session_id(state_file)
    if selected_session_id is None and len(mapping) == 1:
        selected_session_id = next(iter(mapping))
    if selected_session_id is None:
        return "missing session_id and no active session could be inferred"

    chat_id = mapping.get(selected_session_id)
    if chat_id is None:
        return f"unknown session_id `{selected_session_id}`"
    return _RequestContext(token=token, chat_id=chat_id, session_id=selected_session_id)


def main() -> None:
    """Run the MCP server over stdio."""

    mcp.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()

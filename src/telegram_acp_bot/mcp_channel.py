"""Internal MCP channel exposed by the Telegram ACP bot.

This is a minimal server that is auto-registered for ACP sessions.
It currently exposes attachment delivery over Telegram Bot API.
"""

from __future__ import annotations

import base64
import binascii
import mimetypes
import os
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
        "Channel helper tools for Telegram clients. "
        "Use these tools when you need channel-specific behavior."
    ),
)


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
    if context["error"] is not None:
        return fail(context["error"])
    token = context["token"]
    chat_id = context["chat_id"]
    assert token is not None
    assert chat_id is not None

    loaded = _load_attachment_bytes(path=path, data_base64=data_base64, name=name)
    if loaded["error"] is not None:
        return fail(loaded["error"])
    raw = loaded["raw"]
    assert raw is not None
    filename = loaded["filename"]
    guessed_mime = loaded["guessed_mime"]

    resolved_mime = mime_type or guessed_mime or "application/octet-stream"
    bot = Bot(token=token)
    input_file = InputFile(BytesIO(raw), filename=filename)
    if resolved_mime.startswith("image/"):
        await bot.send_photo(chat_id=chat_id, photo=input_file)
        delivered_as = "photo"
    else:
        await bot.send_document(chat_id=chat_id, document=input_file)
        delivered_as = "document"

    return {
        "ok": True,
        "session_id": context["session_id"],
        "chat_id": chat_id,
        "delivered_as": delivered_as,
        "name": filename,
        "mime_type": resolved_mime,
    }


def _load_attachment_bytes(
    *,
    path: str | None,
    data_base64: str | None,
    name: str | None,
) -> dict[str, bytes | str | None]:
    if path is not None:
        source_path = Path(path).expanduser().resolve(strict=False)
        if not source_path.is_file():
            return {
                "error": f"file not found: {source_path}",
                "raw": None,
                "filename": None,
                "guessed_mime": None,
            }
        raw = source_path.read_bytes()
        filename = name or source_path.name
        guessed_mime = mimetypes.guess_type(source_path.name)[0]
        return {
            "error": None,
            "raw": raw,
            "filename": filename,
            "guessed_mime": guessed_mime,
        }

    assert data_base64 is not None
    try:
        raw = base64.b64decode(data_base64)
    except (ValueError, binascii.Error):
        return {
            "error": "invalid base64 payload",
            "raw": None,
            "filename": None,
            "guessed_mime": None,
        }
    filename = name or "attachment.bin"
    guessed_mime = mimetypes.guess_type(filename)[0]
    return {
        "error": None,
        "raw": raw,
        "filename": filename,
        "guessed_mime": guessed_mime,
    }


def _resolve_request_context(
    *,
    session_id: str | None,
    path: str | None,
    data_base64: str | None,
) -> dict[str, str | int | None]:
    if bool(path) == bool(data_base64):
        return {
            "error": "provide exactly one of `path` or `data_base64`",
            "token": None,
            "chat_id": None,
            "session_id": None,
        }

    token = os.getenv(TOKEN_ENV, "").strip()
    if not token:
        return {"error": f"missing {TOKEN_ENV}", "token": None, "chat_id": None, "session_id": None}
    state_file_raw = os.getenv(STATE_FILE_ENV, "").strip()
    if not state_file_raw:
        return {"error": f"missing {STATE_FILE_ENV}", "token": None, "chat_id": None, "session_id": None}

    state_file = Path(state_file_raw)
    mapping = load_session_chat_map(state_file)
    selected_session_id = (session_id or "").strip() or None
    if selected_session_id is None:
        selected_session_id = load_last_session_id(state_file)
    if selected_session_id is None and len(mapping) == 1:
        selected_session_id = next(iter(mapping))
    if selected_session_id is None:
        return {
            "error": "missing session_id and no active session could be inferred",
            "token": None,
            "chat_id": None,
            "session_id": None,
        }

    chat_id = mapping.get(selected_session_id)
    if chat_id is None:
        return {
            "error": f"unknown session_id `{selected_session_id}`",
            "token": None,
            "chat_id": None,
            "session_id": None,
        }
    return {"error": None, "token": token, "chat_id": chat_id, "session_id": selected_session_id}


def main() -> None:
    """Run the MCP server over stdio."""

    mcp.run("stdio")


if __name__ == "__main__":
    main()

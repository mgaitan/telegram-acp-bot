"""Attachment-delivery tool for the internal Telegram MCP server."""

from __future__ import annotations

import base64
import binascii
import mimetypes
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

from mcp.server.fastmcp import FastMCP
from telegram import Bot, InputFile

from telegram_acp_bot.mcp.context import resolve_request_context

ALLOW_PATH_ENV = "ACP_TELEGRAM_CHANNEL_ALLOW_PATH"
EXACTLY_ONE_ATTACHMENT_SOURCE_ERROR = "provide exactly one of `path` or `data_base64`"


@dataclass(frozen=True, slots=True)
class AttachmentPayload:
    raw: bytes
    filename: str
    guessed_mime: str | None


def register_attachment_tools(mcp: FastMCP) -> None:
    """Register attachment tools on the provided MCP server."""

    mcp.tool(
        name="telegram_send_attachment",
        description=(
            "Send an attachment to the current Telegram chat. "
            "If session_id is omitted, the server auto-resolves it when possible. "
            "Use this when the user asks to send an image/file."
        ),
    )(telegram_send_attachment)


async def telegram_send_attachment(
    session_id: str | None = None,
    path: str | None = None,
    data_base64: str | None = None,
    name: str | None = None,
    mime_type: str | None = None,
) -> dict[str, object]:
    def fail(error: str) -> dict[str, object]:
        return {"ok": False, "error": error}

    if path is not None and not allow_path_inputs():
        return fail(f"`path` input is disabled by default. Set {ALLOW_PATH_ENV}=1 to enable it.")

    context = resolve_request_context(session_id=session_id)
    if isinstance(context, str):
        return fail(context)

    loaded = load_attachment_bytes(path=path, data_base64=data_base64, name=name)
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


def load_attachment_bytes(
    *,
    path: str | None,
    data_base64: str | None,
    name: str | None,
) -> AttachmentPayload | str:
    """Load attachment bytes from a trusted path or base64 payload."""

    if (path is None) == (data_base64 is None):
        return EXACTLY_ONE_ATTACHMENT_SOURCE_ERROR

    if path is not None:
        source_path = Path(path).expanduser().resolve(strict=False)
        if not source_path.is_file():
            return f"file not found: {source_path}"
        raw = source_path.read_bytes()
        filename = name or source_path.name
        guessed_mime = mimetypes.guess_type(filename)[0]
        return AttachmentPayload(raw=raw, filename=filename, guessed_mime=guessed_mime)

    try:
        raw = base64.b64decode(cast(str, data_base64), validate=True)
    except (ValueError, binascii.Error):
        return "invalid base64 payload"
    filename = name or "attachment.bin"
    guessed_mime = mimetypes.guess_type(filename)[0]
    return AttachmentPayload(raw=raw, filename=filename, guessed_mime=guessed_mime)


def allow_path_inputs() -> bool:
    """Return whether trusted `path` inputs are enabled for attachment delivery."""

    value = os.getenv(ALLOW_PATH_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}

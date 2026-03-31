"""Reaction-delivery tool for the internal Telegram MCP server."""

from __future__ import annotations

import unicodedata
from collections.abc import Collection

from mcp.server.fastmcp import FastMCP
from telegram import Bot
from telegram.constants import ReactionEmoji

from telegram_acp_bot.mcp.context import resolve_request_context

STANDARD_REACTION_EMOJIS: Collection[str] = frozenset(member.value for member in ReactionEmoji)
STANDARD_REACTION_EMOJI_LIST: tuple[str, ...] = tuple(sorted(STANDARD_REACTION_EMOJIS))
REACTION_VARIANT_NORMALIZATION = {
    "\N{HEAVY BLACK HEART}\N{VARIATION SELECTOR-16}": "\N{HEAVY BLACK HEART}",
}


def register_reaction_tools(mcp: FastMCP) -> None:
    """Register reaction tools on the provided MCP server."""

    mcp.tool(
        name="telegram_set_message_reaction",
        description=(
            "Set a standard Telegram emoji reaction on a message in the current chat. "
            "Use this for lightweight acknowledgements, celebration, or brief affirmation "
            "instead of a full text reply when appropriate. "
            "Supported inputs are the standard Telegram reaction emoji; common Unicode variants "
            "such as heart-with-variation-selector are normalized automatically."
        ),
    )(telegram_set_message_reaction)


async def telegram_set_message_reaction(
    emoji: str,
    session_id: str | None = None,
    message_id: int | None = None,
    is_big: bool | None = None,
) -> dict[str, object]:
    def fail(error: str) -> dict[str, object]:
        return {"ok": False, "error": error}

    context = resolve_request_context(session_id=session_id)
    if isinstance(context, str):
        return fail(context)

    normalized_emoji = normalize_reaction_emoji(emoji)
    if normalized_emoji not in STANDARD_REACTION_EMOJIS:
        return fail(
            "unsupported reaction emoji: use a standard Telegram reaction emoji. "
            f"Supported reactions: {', '.join(STANDARD_REACTION_EMOJI_LIST)}"
        )

    resolved_message_id = message_id or context.prompt_message_id
    if resolved_message_id is None:
        return fail("missing message_id and no active prompt message could be inferred")

    bot = Bot(token=context.token)
    await bot.set_message_reaction(
        chat_id=context.chat_id,
        message_id=resolved_message_id,
        reaction=normalized_emoji,
        is_big=is_big,
    )
    return {
        "ok": True,
        "session_id": context.session_id,
        "chat_id": context.chat_id,
        "message_id": resolved_message_id,
        "emoji": normalized_emoji,
        "is_big": is_big,
    }


def normalize_reaction_emoji(emoji: str) -> str:
    """Normalize a user-provided emoji string to Telegram's standard reaction form."""

    normalized = unicodedata.normalize("NFC", emoji).strip()
    return REACTION_VARIANT_NORMALIZATION.get(normalized, normalized)

"""Internal MCP server exposed by the Telegram ACP bot."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mcp.server.fastmcp import FastMCP

from telegram_acp_bot.mcp.context import RequestContext, resolve_request_context
from telegram_acp_bot.mcp.state import (
    STATE_FILE_ENV,
    TOKEN_ENV,
    load_last_session_id,
    load_prompt_message_id,
    load_session_chat_map,
)
from telegram_acp_bot.mcp.tools.attachments import (
    ALLOW_PATH_ENV,
    AttachmentPayload,
    allow_path_inputs,
    load_attachment_bytes,
    register_attachment_tools,
    telegram_send_attachment,
)
from telegram_acp_bot.mcp.tools.reactions import (
    STANDARD_REACTION_EMOJI_LIST,
    STANDARD_REACTION_EMOJIS,
    register_reaction_tools,
    telegram_set_message_reaction,
)
from telegram_acp_bot.mcp.tools.scheduling import (
    MISSING_SCHEDULED_DB_ERROR,
    MIXED_RUN_AT_AND_DELAY_ERROR,
    NEGATIVE_DELAY_ERROR,
    RUN_AT_OR_DELAY_ERROR,
    format_scheduled_summary,
    load_scheduled_task_store,
    register_scheduling_tools,
    resolve_run_at,
    schedule_task,
)
from telegram_acp_bot.scheduled_tasks.store import parse_utc_timestamp

_RequestContext = RequestContext
_resolve_request_context = resolve_request_context
_AttachmentPayload = AttachmentPayload
_load_attachment_bytes = load_attachment_bytes

__all__ = [
    "ALLOW_PATH_ENV",
    "MISSING_SCHEDULED_DB_ERROR",
    "MIXED_RUN_AT_AND_DELAY_ERROR",
    "NEGATIVE_DELAY_ERROR",
    "RUN_AT_OR_DELAY_ERROR",
    "STANDARD_REACTION_EMOJIS",
    "STANDARD_REACTION_EMOJI_LIST",
    "STATE_FILE_ENV",
    "TOKEN_ENV",
    "UTC",
    "_AttachmentPayload",
    "_RequestContext",
    "_load_attachment_bytes",
    "_resolve_request_context",
    "allow_path_inputs",
    "datetime",
    "format_scheduled_summary",
    "load_last_session_id",
    "load_prompt_message_id",
    "load_scheduled_task_store",
    "load_session_chat_map",
    "main",
    "mcp",
    "parse_utc_timestamp",
    "register_attachment_tools",
    "register_reaction_tools",
    "register_scheduling_tools",
    "resolve_run_at",
    "schedule_task",
    "telegram_channel_info",
    "telegram_send_attachment",
    "telegram_set_message_reaction",
    "timedelta",
]

mcp = FastMCP(
    name="telegram-channel",
    instructions=(
        "Channel helper tools for Telegram clients. Use these tools when you need channel-specific behavior. "
        "Telegram-native lightweight actions such as reactions can be appropriate for concise acknowledgement, "
        "celebration, or affirmation when a full text reply would be noisier than necessary."
    ),
)


@mcp.tool(
    name="telegram_channel_info",
    description="Return capabilities currently exposed by the Telegram channel MCP server.",
)
def telegram_channel_info() -> dict[str, object]:
    return {
        "supports_attachment_delivery": True,
        "supports_message_reactions": True,
        "supported_reaction_emojis": STANDARD_REACTION_EMOJI_LIST,
        "supports_scheduled_tasks": True,
        "supports_followup_buttons": False,
        "status": "active",
    }


register_attachment_tools(mcp)
register_reaction_tools(mcp)
register_scheduling_tools(mcp)


def main() -> None:
    """Run the MCP server over stdio."""

    mcp.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()

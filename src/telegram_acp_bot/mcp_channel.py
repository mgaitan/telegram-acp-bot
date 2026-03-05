"""Internal MCP channel exposed by the Telegram ACP bot.

This is a minimal server that is auto-registered for ACP sessions.
It currently exposes draft channel tools without side effects.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

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
        "supports_attachment_delivery": False,
        "supports_followup_buttons": False,
        "status": "draft",
    }


def main() -> None:
    """Run the MCP server over stdio."""

    mcp.run("stdio")


if __name__ == "__main__":
    main()

"""Tests for MCP server bootstrap and capability metadata."""

from __future__ import annotations

from tests.mcp.support import mcp_channel


def test_telegram_channel_info_reports_active_capabilities():
    payload = mcp_channel.telegram_channel_info()
    assert payload["status"] == "active"
    assert payload["supports_attachment_delivery"] is True
    assert payload["supports_message_reactions"] is True
    assert "❤" in payload["supported_reaction_emojis"]
    assert payload["supports_scheduled_tasks"] is True
    assert payload["supports_followup_buttons"] is False


def test_main_runs_stdio_server(mocker):
    run = mocker.patch.object(mcp_channel.mcp, "run")
    mcp_channel.main()
    run.assert_called_once_with("stdio")

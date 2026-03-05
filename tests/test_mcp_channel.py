"""Tests for the internal MCP channel server."""

from __future__ import annotations

from telegram_acp_bot import mcp_channel


def test_telegram_channel_info_reports_draft_capabilities():
    payload = mcp_channel.telegram_channel_info()
    assert payload["status"] == "draft"
    assert payload["supports_attachment_delivery"] is False
    assert payload["supports_followup_buttons"] is False


def test_main_runs_stdio_server(mocker):
    run = mocker.patch.object(mcp_channel.mcp, "run")
    mcp_channel.main()
    run.assert_called_once_with("stdio")

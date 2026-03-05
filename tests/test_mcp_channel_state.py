from __future__ import annotations

from pathlib import Path

from telegram_acp_bot.mcp_channel_state import default_state_file


def test_default_state_file_prefers_xdg_runtime_dir(monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)

    path = default_state_file(pid=42)

    assert path == Path("/run/user/1000/telegram-acp-bot/telegram-acp-bot-mcp-state-42.json")


def test_default_state_file_uses_xdg_state_home_when_runtime_missing(monkeypatch):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", "/tmp/custom-state")

    path = default_state_file(pid=42)

    assert path == Path("/tmp/custom-state/telegram-acp-bot/telegram-acp-bot-mcp-state-42.json")


def test_default_state_file_falls_back_to_local_state_home(monkeypatch, mocker):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    mocker.patch("telegram_acp_bot.mcp_channel_state.Path.home", return_value=Path("/home/tester"))

    path = default_state_file(pid=42)

    assert path == Path("/home/tester/.local/state/telegram-acp-bot/telegram-acp-bot-mcp-state-42.json")

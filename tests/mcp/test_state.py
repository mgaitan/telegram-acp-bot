"""Tests for MCP shared state persistence helpers."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from tests.mcp.support import (
    STATE_FILE_PRIVATE_MODE,
    channel_state_module,
    load_last_session_id,
    load_prompt_message_id,
    mcp_channel,
    save_last_session_id,
    save_prompt_message_id,
    save_session_chat_map,
)

PROMPT_MESSAGE_ID = 42


def test_load_session_chat_map_handles_invalid_json(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text("{invalid", encoding="utf-8")

    assert mcp_channel.load_session_chat_map(state_file) == {}


def test_load_session_chat_map_handles_non_dict_sessions(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text('{"sessions": []}', encoding="utf-8")

    assert mcp_channel.load_session_chat_map(state_file) == {}


def test_load_last_session_id_handles_invalid_json(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text("{invalid", encoding="utf-8")

    assert load_last_session_id(state_file) is None


def test_save_last_session_id_coerces_non_dict_sessions(tmp_path: Path):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"sessions": []}), encoding="utf-8")

    save_last_session_id(state_file, "s1")

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["sessions"] == {}
    assert payload["last_session_id"] == "s1"


def test_state_file_permissions_are_restricted(tmp_path: Path):
    state_file = tmp_path / "state.json"

    save_session_chat_map(state_file, {"s1": 123})
    save_last_session_id(state_file, "s1")

    mode = stat.S_IMODE(state_file.stat().st_mode)
    assert mode == STATE_FILE_PRIVATE_MODE


def test_save_prompt_message_id_round_trips(tmp_path: Path):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})

    save_prompt_message_id(state_file, "s1", PROMPT_MESSAGE_ID)

    assert load_prompt_message_id(state_file, "s1") == PROMPT_MESSAGE_ID


def test_save_prompt_message_id_preserves_existing_session_state(tmp_path: Path):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    save_last_session_id(state_file, "s1")

    save_prompt_message_id(state_file, "s1", PROMPT_MESSAGE_ID)

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["sessions"] == {"s1": 123}
    assert payload["last_session_id"] == "s1"
    assert payload["prompt_message_ids"] == {"s1": PROMPT_MESSAGE_ID}


def test_save_prompt_message_id_can_clear_existing_value(tmp_path: Path):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    save_prompt_message_id(state_file, "s1", PROMPT_MESSAGE_ID)

    save_prompt_message_id(state_file, "s1", None)

    assert load_prompt_message_id(state_file, "s1") is None


def test_save_session_chat_map_preserves_prompt_message_ids(tmp_path: Path):
    state_file = tmp_path / "state.json"
    save_prompt_message_id(state_file, "s1", PROMPT_MESSAGE_ID)

    save_session_chat_map(state_file, {"s1": 123})

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["prompt_message_ids"] == {"s1": PROMPT_MESSAGE_ID}


def test_save_last_session_id_preserves_prompt_message_ids(tmp_path: Path):
    state_file = tmp_path / "state.json"
    save_prompt_message_id(state_file, "s1", PROMPT_MESSAGE_ID)

    save_last_session_id(state_file, "s1")

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["prompt_message_ids"] == {"s1": PROMPT_MESSAGE_ID}


def test_state_file_atomic_write_removes_temp_file_on_replace_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"

    def boom(src: Path | str, dst: Path | str) -> None:
        del src, dst
        raise OSError

    monkeypatch.setattr(channel_state_module.os, "replace", boom)

    with pytest.raises(OSError):
        save_session_chat_map(state_file, {"s1": 123})

    assert list(tmp_path.glob(f".{state_file.name}*.tmp")) == []

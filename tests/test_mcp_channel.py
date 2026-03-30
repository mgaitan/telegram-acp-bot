"""Tests for the internal MCP channel server."""

from __future__ import annotations

import base64
import json
import stat
from pathlib import Path
from typing import cast

import pytest

from telegram_acp_bot.mcp import server as mcp_channel
from telegram_acp_bot.mcp import state as channel_state_module
from telegram_acp_bot.mcp.state import (
    STATE_FILE_ENV,
    TOKEN_ENV,
    load_last_session_id,
    save_last_session_id,
    save_session_chat_map,
)
from telegram_acp_bot.scheduled_tasks import ACP_SCHEDULED_TASKS_DB_ENV, ScheduledTaskStore

STATE_FILE_PRIVATE_MODE = 0o600
TEST_SCHEDULED_CHAT_ID = 123


def test_telegram_channel_info_reports_active_capabilities():
    payload = mcp_channel.telegram_channel_info()
    assert payload["status"] == "active"
    assert payload["supports_attachment_delivery"] is True
    assert payload["supports_scheduled_tasks"] is True
    assert payload["supports_followup_buttons"] is False


def test_main_runs_stdio_server(mocker):
    run = mocker.patch.object(mcp_channel.mcp, "run")
    mcp_channel.main()
    run.assert_called_once_with("stdio")


@pytest.mark.asyncio
async def test_send_attachment_from_path_as_photo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker):
    state_file = tmp_path / "state.json"
    image_file = tmp_path / "webcam.jpg"
    image_file.write_bytes(b"jpeg")
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(mcp_channel.ALLOW_PATH_ENV, "1")
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp.tools.attachments.Bot", return_value=bot)

    result = await mcp_channel.telegram_send_attachment(path=str(image_file))

    assert result["ok"] is True
    assert result["delivered_as"] == "photo"
    assert result["session_id"] == "s1"
    bot.send_photo.assert_awaited_once()
    bot.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_send_attachment_from_base64_as_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    save_last_session_id(state_file, "s1")
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp.tools.attachments.Bot", return_value=bot)

    result = await mcp_channel.telegram_send_attachment(
        session_id="s1",
        data_base64=base64.b64encode(b"payload").decode("ascii"),
        name="artifact.bin",
    )

    assert result["ok"] is True
    assert result["delivered_as"] == "document"
    bot.send_document.assert_awaited_once()
    bot.send_photo.assert_not_called()


@pytest.mark.asyncio
async def test_send_attachment_infers_last_active_session_when_multiple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
):
    expected_chat_id = 456
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123, "s2": expected_chat_id})
    save_last_session_id(state_file, "s2")
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp.tools.attachments.Bot", return_value=bot)

    result = await mcp_channel.telegram_send_attachment(
        data_base64=base64.b64encode(b"payload").decode("ascii"),
        name="artifact.bin",
    )

    assert result["ok"] is True
    assert result["session_id"] == "s2"
    assert result["chat_id"] == expected_chat_id
    bot.send_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_attachment_rejects_missing_session_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(mcp_channel.ALLOW_PATH_ENV, "1")

    result = await mcp_channel.telegram_send_attachment(session_id="unknown", path=str(tmp_path / "x.jpg"))
    assert result["ok"] is False
    assert "unknown session_id" in cast(str, result["error"])


@pytest.mark.asyncio
async def test_send_attachment_rejects_invalid_base64_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = await mcp_channel.telegram_send_attachment(session_id="s1", data_base64="***not-base64***")

    assert result["ok"] is False
    assert result["error"] == "invalid base64 payload"


def test_load_attachment_bytes_rejects_missing_path(tmp_path: Path):
    result = mcp_channel._load_attachment_bytes(path=str(tmp_path / "missing.bin"), data_base64=None, name=None)
    assert isinstance(result, str)
    assert result.startswith("file not found:")


@pytest.mark.asyncio
async def test_send_attachment_rejects_path_when_not_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.delenv(mcp_channel.ALLOW_PATH_ENV, raising=False)

    result = await mcp_channel.telegram_send_attachment(session_id="s1", path=str(tmp_path / "x.jpg"))
    assert result["ok"] is False
    assert "`path` input is disabled by default" in cast(str, result["error"])


@pytest.mark.asyncio
async def test_send_attachment_uses_overridden_name_for_mime_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
):
    state_file = tmp_path / "state.json"
    binary_file = tmp_path / "payload.bin"
    binary_file.write_bytes(b"jpeg")
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(mcp_channel.ALLOW_PATH_ENV, "1")
    bot = mocker.AsyncMock()
    mocker.patch("telegram_acp_bot.mcp.tools.attachments.Bot", return_value=bot)

    result = await mcp_channel.telegram_send_attachment(path=str(binary_file), name="forced.jpg")

    assert result["ok"] is True
    assert result["delivered_as"] == "photo"
    bot.send_photo.assert_awaited_once()


def test_resolve_request_context_requires_exactly_one_payload_source():
    result = mcp_channel._resolve_request_context(session_id="s1")
    assert result == f"missing {TOKEN_ENV}"


def test_resolve_request_context_requires_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.setenv(STATE_FILE_ENV, str(tmp_path / "state.json"))
    result = mcp_channel._resolve_request_context(session_id="s1")
    assert result == f"missing {TOKEN_ENV}"


def test_resolve_request_context_requires_state_file_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.delenv(STATE_FILE_ENV, raising=False)
    result = mcp_channel._resolve_request_context(session_id="s1")
    assert result == f"missing {STATE_FILE_ENV}"


def test_resolve_request_context_requires_inferable_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = mcp_channel._resolve_request_context(session_id=None)

    assert result == "missing session_id and no active session could be inferred"


def test_resolve_request_context_requires_explicit_session_when_multiple_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123, "s2": 456})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = mcp_channel._resolve_request_context(session_id=None)

    assert (
        result == "missing session_id: multiple active sessions exist and no last active session could be inferred. "
        "Available session_ids: s1, s2"
    )


def test_resolve_request_context_uses_last_active_session_when_multiple_mappings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123, "s2": 456})
    save_last_session_id(state_file, "s2")
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = mcp_channel._resolve_request_context(session_id=None)

    assert result == mcp_channel._RequestContext(token="TOKEN", chat_id=456, session_id="s2")


def test_resolve_request_context_ignores_stale_last_session_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": 123, "s2": 456})
    save_last_session_id(state_file, "stale")
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))

    result = mcp_channel._resolve_request_context(session_id=None)

    assert isinstance(result, str)
    assert result.startswith(
        "missing session_id: multiple active sessions exist and no last active session could be inferred."
    )


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


def test_state_file_atomic_write_removes_temp_file_on_replace_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"

    def boom(src: Path | str, dst: Path | str) -> None:
        del src, dst
        raise OSError

    monkeypatch.setattr(channel_state_module.os, "replace", boom)

    with pytest.raises(OSError):
        save_session_chat_map(state_file, {"s1": 123})

    assert list(tmp_path.glob(f".{state_file.name}*.tmp")) == []


@pytest.mark.asyncio
async def test_schedule_task_persists_unanchored_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    scheduled_db = tmp_path / "scheduled.sqlite3"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(ACP_SCHEDULED_TASKS_DB_ENV, str(scheduled_db))

    result = await mcp_channel.schedule_task(
        run_at="2026-03-30T21:00:00+00:00",
        mode="notify",
        notify_text="Review the PR now",
    )

    store = ScheduledTaskStore(scheduled_db)
    task = store.get_task(cast(str, result["task_id"]))

    assert result["ok"] is True
    assert result["anchor_message_id"] is None
    assert task is not None
    assert task.chat_id == TEST_SCHEDULED_CHAT_ID
    assert task.session_id == "s1"
    assert task.anchor_message_id is None
    assert task.mode == "notify"
    assert task.notify_text == "Review the PR now"


@pytest.mark.asyncio
async def test_schedule_task_rejects_invalid_timestamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    scheduled_db = tmp_path / "scheduled.sqlite3"
    save_session_chat_map(state_file, {"s1": 123})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(ACP_SCHEDULED_TASKS_DB_ENV, str(scheduled_db))

    result = await mcp_channel.schedule_task(
        run_at="2026-03-30T21:00:00",
        mode="notify",
        notify_text="Review the PR now",
    )

    assert result["ok"] is False
    assert "timezone" in cast(str, result["error"])


@pytest.mark.asyncio
async def test_schedule_task_rejects_invalid_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    scheduled_db = tmp_path / "scheduled.sqlite3"
    save_session_chat_map(state_file, {"s1": TEST_SCHEDULED_CHAT_ID})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(ACP_SCHEDULED_TASKS_DB_ENV, str(scheduled_db))

    result = await mcp_channel.schedule_task(
        run_at="2026-03-30T21:00:00+00:00",
        mode="later",
        notify_text="Review the PR now",
    )

    assert result["ok"] is False
    assert "mode must be one of" in cast(str, result["error"])


@pytest.mark.asyncio
async def test_schedule_task_requires_notify_text_for_notify_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state_file = tmp_path / "state.json"
    scheduled_db = tmp_path / "scheduled.sqlite3"
    save_session_chat_map(state_file, {"s1": TEST_SCHEDULED_CHAT_ID})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(ACP_SCHEDULED_TASKS_DB_ENV, str(scheduled_db))

    result = await mcp_channel.schedule_task(
        run_at="2026-03-30T21:00:00+00:00",
        mode="notify",
    )

    assert result["ok"] is False
    assert result["error"] == "notify mode requires notify_text"


@pytest.mark.asyncio
async def test_schedule_task_requires_prompt_text_for_prompt_agent_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state_file = tmp_path / "state.json"
    scheduled_db = tmp_path / "scheduled.sqlite3"
    save_session_chat_map(state_file, {"s1": TEST_SCHEDULED_CHAT_ID})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(ACP_SCHEDULED_TASKS_DB_ENV, str(scheduled_db))

    result = await mcp_channel.schedule_task(
        run_at="2026-03-30T21:00:00+00:00",
        mode="prompt_agent",
    )

    assert result["ok"] is False
    assert result["error"] == "prompt_agent mode requires prompt_text"


@pytest.mark.asyncio
async def test_schedule_task_accepts_relative_delay_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_file = tmp_path / "state.json"
    scheduled_db = tmp_path / "scheduled.sqlite3"
    save_session_chat_map(state_file, {"s1": TEST_SCHEDULED_CHAT_ID})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(ACP_SCHEDULED_TASKS_DB_ENV, str(scheduled_db))

    before = mcp_channel.datetime.now(mcp_channel.UTC)
    result = await mcp_channel.schedule_task(
        mode="notify",
        notify_text="Review the PR now",
        delay_minutes=10,
    )
    after = mcp_channel.datetime.now(mcp_channel.UTC)
    scheduled = mcp_channel.parse_utc_timestamp(str(result["run_at"]))

    assert result["ok"] is True
    assert before <= scheduled - mcp_channel.timedelta(minutes=10) <= after


@pytest.mark.asyncio
async def test_schedule_task_rejects_mixed_absolute_and_relative_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state_file = tmp_path / "state.json"
    scheduled_db = tmp_path / "scheduled.sqlite3"
    save_session_chat_map(state_file, {"s1": TEST_SCHEDULED_CHAT_ID})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(ACP_SCHEDULED_TASKS_DB_ENV, str(scheduled_db))

    result = await mcp_channel.schedule_task(
        run_at="2026-03-30T21:00:00+00:00",
        mode="notify",
        notify_text="Review the PR now",
        delay_seconds=30,
    )

    assert result["ok"] is False
    assert result["error"] == "provide either run_at or delay inputs, not both"


@pytest.mark.asyncio
async def test_schedule_task_requires_absolute_or_relative_time_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state_file = tmp_path / "state.json"
    scheduled_db = tmp_path / "scheduled.sqlite3"
    save_session_chat_map(state_file, {"s1": TEST_SCHEDULED_CHAT_ID})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(ACP_SCHEDULED_TASKS_DB_ENV, str(scheduled_db))

    result = await mcp_channel.schedule_task(
        mode="notify",
        notify_text="Review the PR now",
    )

    assert result["ok"] is False
    assert result["error"] == "provide run_at or at least one delay input"


@pytest.mark.asyncio
async def test_schedule_task_rejects_negative_delay_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state_file = tmp_path / "state.json"
    scheduled_db = tmp_path / "scheduled.sqlite3"
    save_session_chat_map(state_file, {"s1": TEST_SCHEDULED_CHAT_ID})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.setenv(ACP_SCHEDULED_TASKS_DB_ENV, str(scheduled_db))

    result = await mcp_channel.schedule_task(
        mode="notify",
        notify_text="Review the PR now",
        delay_seconds=-1,
    )

    assert result["ok"] is False
    assert result["error"] == "delay inputs must be zero or positive"


@pytest.mark.asyncio
async def test_schedule_task_reports_missing_scheduled_db_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state_file = tmp_path / "state.json"
    save_session_chat_map(state_file, {"s1": TEST_SCHEDULED_CHAT_ID})
    monkeypatch.setenv(TOKEN_ENV, "TOKEN")
    monkeypatch.setenv(STATE_FILE_ENV, str(state_file))
    monkeypatch.delenv(ACP_SCHEDULED_TASKS_DB_ENV, raising=False)

    result = await mcp_channel.schedule_task(
        run_at="2026-03-30T21:00:00+00:00",
        mode="notify",
        notify_text="Review the PR now",
    )

    assert result["ok"] is False
    assert result["error"] == f"missing {ACP_SCHEDULED_TASKS_DB_ENV}"


@pytest.mark.asyncio
async def test_schedule_task_reports_context_resolution_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.delenv(STATE_FILE_ENV, raising=False)

    result = await mcp_channel.schedule_task(
        run_at="2026-03-30T21:00:00+00:00",
        mode="notify",
        notify_text="Review the PR now",
    )

    assert result["ok"] is False
    assert result["error"] == f"missing {TOKEN_ENV}"

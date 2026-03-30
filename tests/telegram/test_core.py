from __future__ import annotations

# ruff: noqa: F403, F405, I001

from tests.telegram.support import *


async def test_make_config():
    config = make_config(token="T", allowed_user_ids=[1, 2, 2], workspace="~/tmp")
    assert config.token == "T"
    assert config.allowed_user_ids == {1, 2}
    assert config.default_workspace.name == "tmp"


async def test_workspace_from_relative_arg_uses_default_workspace():
    config = make_config(token="T", allowed_user_ids=[], workspace="/tmp/base")
    bridge = TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))

    workspace = bridge._workspace_from_args(["foo"])
    assert workspace == Path("/tmp/base/foo")


async def test_start_and_help():
    bridge = make_bridge()
    update = make_update(with_message=True)
    context = make_context()

    await bridge.start(update, context)
    await bridge.help(update, context)

    assert update.message is not None
    assert "Send a message to start in the default workspace" in update.message.replies[0]
    assert "Commands:" in update.message.replies[1]
    assert "/cancel" in update.message.replies[1]
    assert "/mode" in update.message.replies[1]
    assert "/scheduled" in update.message.replies[1]


async def test_mode_command_sets_verbose_activity_mode():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.mode(update, make_context(args=["verbose"]))

    assert update.message is not None
    assert update.message.replies == ["Activity mode set to verbose."]
    assert bridge._activity_mode(chat_id=TEST_CHAT_ID) == "verbose"


async def test_mode_command_reports_current_mode_without_args():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.mode(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["Current activity mode: normal\nUsage: /mode normal, compact, or verbose"]


async def test_mode_command_rejects_invalid_mode():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.mode(update, make_context(args=["loud"]))

    assert update.message is not None
    assert update.message.replies == ["Usage: /mode normal, compact, or verbose"]


async def test_mode_command_stops_when_access_is_denied():
    bridge = make_bridge(allowed_ids={99})
    update = make_update(with_message=True)

    await bridge.mode(update, make_context(args=["verbose"]))

    assert update.message is not None
    assert update.message.replies == ["Access denied for this bot."]


async def test_start_allows_user_by_username_allowlist():
    config = make_config(token="TOKEN", allowed_user_ids=[], allowed_usernames=["@Alice"], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))
    update = make_update(user_id=999, username="Alice", with_message=True)
    context = make_context()

    await bridge.start(update, context)

    assert update.message is not None
    assert "Send a message to start in the default workspace" in update.message.replies[0]


async def test_activity_mode_base_handler_defaults():
    bridge = make_bridge()
    handler = bot_module._ActivityModeHandler(bridge)

    await handler.on_permission_request(
        request=PermissionRequest(
            chat_id=TEST_CHAT_ID,
            request_id="req",
            tool_title="Run pwd",
            tool_call_id="call-base",
            available_actions=("once",),
        ),
        message="Permission request",
        keyboard=InlineKeyboardMarkup([]),
    )
    assert await handler.finalize_reply(chat_id=TEST_CHAT_ID, update=cast(Update, make_update()), text="hello") is None
    await handler.handle_empty_reply(chat_id=TEST_CHAT_ID)
    await handler.clear_chat_state(chat_id=TEST_CHAT_ID)

    with pytest.raises(NotImplementedError):
        await handler.on_activity_event(
            chat_id=TEST_CHAT_ID,
            block=AgentActivityBlock(kind="think", title="", status="in_progress", text="x"),
        )


async def test_normal_activity_handler_skips_reply_and_duplicate_stream_updates():
    bridge = make_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    handler = bot_module._NormalActivityModeHandler(bridge)

    await handler.on_activity_event(
        chat_id=TEST_CHAT_ID,
        block=AgentActivityBlock(kind="reply", title="", status="in_progress", text="preview", activity_id="reply"),
    )
    await handler.on_activity_event(
        chat_id=TEST_CHAT_ID,
        block=AgentActivityBlock(kind="execute", title="Run", status="in_progress", text="", activity_id="dup"),
    )
    await handler.on_activity_event(
        chat_id=TEST_CHAT_ID,
        block=AgentActivityBlock(kind="execute", title="Run", status="in_progress", text="", activity_id="dup"),
    )
    await handler.on_activity_event(
        chat_id=TEST_CHAT_ID,
        block=AgentActivityBlock(kind="execute", title="Run", status="completed", text="", activity_id="dup"),
    )

    assert len(bot.sent_messages) == EXPECTED_REPEATED_ACTIVITY_MESSAGES


async def test_compact_permission_request_replaces_status_message_when_edit_fails():

    class EditFailingBot(DummyBot):
        async def edit_message_text(self, **kwargs: object) -> None:
            raise MarkdownFailureError

    bridge = make_compact_bridge()
    bot = EditFailingBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    bridge._compact_status_msg_id[TEST_CHAT_ID] = COMPACT_STATUS_MSG_ID

    await bridge._activity_handler(chat_id=TEST_CHAT_ID).on_permission_request(
        request=PermissionRequest(
            chat_id=TEST_CHAT_ID,
            request_id="req-compact",
            tool_title="Run pwd",
            tool_call_id="call-compact",
            available_actions=("once",),
        ),
        message="Permission request",
        keyboard=InlineKeyboardMarkup([]),
    )

    assert bot.deleted_message_ids == [(TEST_CHAT_ID, COMPACT_STATUS_MSG_ID)]
    assert len(bot.sent_messages) == 1
    assert bridge._compact_status_msg_id[TEST_CHAT_ID] == 1


async def test_compact_and_verbose_handlers_return_early_without_app_or_reply_updates():
    compact_handler = make_compact_bridge()._activity_handler(chat_id=TEST_CHAT_ID)
    verbose_handler = make_verbose_bridge()._activity_handler(chat_id=TEST_CHAT_ID)

    await compact_handler.on_activity_event(
        chat_id=TEST_CHAT_ID,
        block=AgentActivityBlock(kind="reply", title="", status="in_progress", text="preview", activity_id="reply"),
    )
    await verbose_handler.on_activity_event(
        chat_id=TEST_CHAT_ID,
        block=AgentActivityBlock(kind="think", title="Thinking", status="in_progress", text="", activity_id="t1"),
    )

    assert (
        await verbose_handler.finalize_reply(chat_id=TEST_CHAT_ID, update=cast(Update, make_update()), text="hello")
        is None
    )


async def test_compact_permission_request_returns_early_without_app():
    handler = make_compact_bridge()._activity_handler(chat_id=TEST_CHAT_ID)

    await handler.on_permission_request(
        request=PermissionRequest(
            chat_id=TEST_CHAT_ID,
            request_id="req-no-app",
            tool_title="Run pwd",
            tool_call_id="call-no-app",
            available_actions=("once",),
        ),
        message="Permission request",
        keyboard=InlineKeyboardMarkup([]),
    )


async def test_restart_requests_app_stop():
    service = EchoAgentService(SessionRegistry())
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=service,
    )
    update = make_update(with_message=True)
    stop_calls: list[str] = []
    bridge._app = cast(Application, SimpleNamespace(stop_running=lambda: stop_calls.append("stop")))
    session_id = await service.new_session(chat_id=TEST_CHAT_ID, workspace=Path("/tmp/restart-workspace"))

    await bridge.restart(update, make_context())

    assert update.message is not None
    assert update.message.replies == [
        f"Restart requested. Re-launching process...\nSession restarted: {session_id} in /tmp/restart-workspace"
    ]
    assert stop_calls == ["stop"]


async def test_restart_with_index_resumes_selected_candidate():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert service.loaded == (TEST_CHAT_ID, "s-resume-1", Path("/tmp/ws1"))
    assert update.message is not None
    assert update.message.replies == ["Session resumed: s-resume-1 in /tmp/ws1"]


async def test_resume_candidate_with_restart_notice_keeps_relaunch_copy():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)
    candidate = service.items[0]

    resumed = await bridge._resume_candidate(
        update=cast(Update, update),
        chat_id=TEST_CHAT_ID,
        candidate=candidate,
        success_label="Session restarted",
        include_restart_notice=True,
    )

    assert resumed is True
    assert update.message is not None
    assert update.message.replies == [
        "Restart requested. Re-launching process...\nSession restarted: s-resume-1 in /tmp/ws1"
    ]


async def test_restart_with_workspace_arg_only_reports_usage():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.restart(update, make_context(args=["/tmp/ws"]))

    assert update.message is not None
    assert update.message.replies == ["Usage: /restart or /restart N [workspace]"]


async def test_restart_with_too_many_args_reports_usage():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.restart(update, make_context(args=["1", "/tmp/ws", "extra"]))

    assert update.message is not None
    assert update.message.replies == ["Usage: /restart or /restart N [workspace]"]


async def test_restart_with_zero_index_resumes_first_candidate():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert service.loaded == (TEST_CHAT_ID, "s-resume-1", Path("/tmp/ws1"))
    assert update.message is not None
    assert update.message.replies == ["Session resumed: s-resume-1 in /tmp/ws1"]


async def test_restart_with_running_app_and_no_active_session_reports_missing_session():
    bridge = make_bridge()
    update = make_update(with_message=True)
    stop_calls: list[str] = []
    bridge._app = cast(Application, SimpleNamespace(stop_running=lambda: stop_calls.append("stop")))

    await bridge.restart(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["No active session. Use /new first."]
    assert stop_calls == []


async def test_restart_with_two_indexes_reports_usage():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.restart(update, make_context(args=["1", "2"]))

    assert update.message is not None
    assert update.message.replies == ["Usage: /restart or /restart N [workspace]"]


async def test_restart_with_two_workspace_args_reports_usage():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.restart(update, make_context(args=["/tmp/ws1", "/tmp/ws2"]))

    assert update.message is not None
    assert update.message.replies == ["Usage: /restart or /restart N [workspace]"]


async def test_restart_with_index_reports_list_error():
    class FailingListResumeService(ResumeService):
        async def list_resumable_sessions(self, *, chat_id: int, workspace: Path | None = None):
            del chat_id, workspace
            raise DummyListBoomError()

    service = FailingListResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert update.message is not None
    assert update.message.replies == ["Failed to list resumable sessions: list boom"]


async def test_restart_with_index_reports_list_not_supported():
    service = ResumeService()
    service.list_supported = False
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert update.message is not None
    assert update.message.replies == ["Agent does not support ACP session/list."]


async def test_restart_with_index_reports_empty_results():
    service = ResumeService()
    service.items = ()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert update.message is not None
    assert update.message.replies == ["No resumable sessions found."]


async def test_restart_with_invalid_index_reports_error():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["9"]))

    assert update.message is not None
    assert update.message.replies == ["Invalid restart index 9. Choose 0..1."]


async def test_restart_with_index_reports_load_failure():
    service = ResumeService()
    service.fail_load = True
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID, with_message=True)

    await bridge.restart(update, make_context(args=["0"]))

    assert update.message is not None
    assert update.message.replies == ["Failed to resume session s-resume-1: load failed"]


async def test_restart_requires_running_application():
    bridge = make_bridge()
    update = make_update(with_message=True)

    await bridge.restart(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["Restart is unavailable: application is not running."]


async def test_restart_access_denied():
    bridge = make_bridge(allowed_ids={999})
    update = make_update(user_id=1, with_message=True)

    await bridge.restart(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["Access denied for this bot."]


async def test_access_denied():
    bridge = make_bridge(allowed_ids={99})
    update = make_update(user_id=1)
    context = make_context()

    await bridge.start(update, context)

    assert update.message is not None
    assert update.message.replies == ["Access denied for this bot."]


async def test_access_allowed_with_allowlist():
    bridge = make_bridge(allowed_ids={1})
    update = make_update(user_id=1)
    context = make_context()

    await bridge.start(update, context)

    assert update.message is not None
    assert len(update.message.replies) == 1
    assert "Send a message to start in the default workspace" in update.message.replies[0]


async def test_denied_paths_for_other_handlers():
    bridge = make_bridge(allowed_ids={42})
    update = make_update(user_id=7, text="hello")
    context = make_context()

    await bridge.help(update, context)
    await bridge.new_session(update, make_context(args=["/tmp"]))
    await bridge.resume_session(update, make_context(args=["/tmp"]))
    await bridge.session(update, context)
    await bridge.cancel(update, context)
    await bridge.stop(update, context)
    await bridge.clear(update, context)
    await bridge.on_message(update, context)

    assert update.message is not None
    assert update.message.replies == [
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
        "Access denied for this bot.",
    ]


async def test_new_session_and_session_command():
    bridge = make_bridge()
    update = make_update()

    await bridge.session(update, make_context())
    await bridge.new_session(update, make_context(args=["/tmp"]))
    await bridge.session(update, make_context())

    assert update.message is not None
    assert update.message.replies[0] == "No active session. Use /new first."
    assert "Session started:" in update.message.replies[1]
    assert "Active session workspace:" in update.message.replies[2]


async def test_resume_session_without_app_loads_first_candidate():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update()

    await bridge.resume_session(update, make_context())

    assert service.loaded is not None
    assert service.loaded[1] == "s-resume-1"
    assert update.message is not None
    assert "Session resumed:" in update.message.replies[0]


async def test_resume_session_with_app_sends_picker_message():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context())

    assert bot.sent_messages
    payload = bot.sent_messages[-1]
    assert payload["chat_id"] == TEST_CHAT_ID
    assert "Pick a session to resume" in cast(str, payload["text"])
    assert payload["reply_markup"] is not None


async def test_resume_session_with_workspace_arg_loads_most_recent_for_workspace(tmp_path: Path):
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path)),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context(args=["/tmp/ws2"]))

    assert service.loaded == (TEST_CHAT_ID, "s-resume-2", Path("/tmp/ws2"))
    assert update.message is not None
    assert update.message.replies == ["Session resumed: s-resume-2 in /tmp/ws2"]
    assert bot.sent_messages == []


async def test_resume_session_with_index_arg_loads_selected_candidate():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context(args=["1"]))

    assert service.loaded == (TEST_CHAT_ID, "s-resume-2", Path("/tmp/ws2"))
    assert update.message is not None
    assert update.message.replies == ["Session resumed: s-resume-2 in /tmp/ws2"]


async def test_resume_session_with_invalid_index_reports_error():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context(args=["9"]))

    assert update.message is not None
    assert update.message.replies == ["Invalid resume index 9. Choose 0..1."]


async def test_resume_session_with_zero_index_reports_usage():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context(args=["0"]))

    assert service.loaded == (TEST_CHAT_ID, "s-resume-1", Path("/tmp/ws1"))
    assert update.message is not None
    assert update.message.replies == ["Session resumed: s-resume-1 in /tmp/ws1"]


async def test_resume_session_rejects_combined_index_and_workspace_args():
    bridge = make_bridge()
    update = make_update(chat_id=TEST_CHAT_ID)

    await bridge.resume_session(update, make_context(args=["1", "/tmp/ws1"]))

    assert update.message is not None
    assert update.message.replies == ["Usage: /resume, /resume N, or /resume [workspace]"]


async def test_resume_session_reports_list_not_supported():
    service = ResumeService()
    service.list_supported = False
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update()

    await bridge.resume_session(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["Agent does not support ACP session/list."]


async def test_resume_session_reports_empty_results():
    service = ResumeService()
    service.items = ()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update()

    await bridge.resume_session(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["No resumable sessions found."]


async def test_resume_session_reports_list_error():
    class FailingListResumeService(ResumeService):
        async def list_resumable_sessions(self, *, chat_id: int, workspace: Path | None = None):
            del chat_id, workspace
            raise DummyListBoomError()

    service = FailingListResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    update = make_update()
    await bridge.resume_session(update, make_context())
    assert update.message is not None
    assert "Failed to list resumable sessions: list boom" in update.message.replies[-1]


async def test_new_session_autocreates_relative_workspace_and_reports_it(tmp_path: Path):
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path))
    bridge = TelegramBridge(config=config, agent_service=EchoAgentService(SessionRegistry()))
    update = make_update()

    await bridge.new_session(update, make_context(args=["myproj"]))

    created_path = tmp_path / "myproj"
    assert created_path.is_dir()
    assert update.message is not None
    assert "Session started:" in update.message.replies[0]
    assert f"Created workspace: {created_path}" in update.message.replies[0]


async def test_new_session_reports_invalid_workspace():
    class InvalidWorkspaceService:
        async def new_session(self, *, chat_id: int, workspace):
            del chat_id, workspace
            raise ValueError("/missing")

        async def prompt(self, *, chat_id: int, text: str):
            del chat_id, text

        def get_workspace(self, *, chat_id: int):
            del chat_id

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, InvalidWorkspaceService()))
    update = make_update()

    await bridge.new_session(update, make_context(args=["/missing"]))

    assert update.message is not None
    assert update.message.replies == ["Invalid workspace: /missing"]


async def test_new_session_reports_process_stdio_error():
    class BrokenAgentService:
        async def new_session(self, *, chat_id: int, workspace):
            del chat_id, workspace
            raise RuntimeError

        async def prompt(self, *, chat_id: int, text: str):
            del chat_id, text

        def get_workspace(self, *, chat_id: int):
            del chat_id

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, BrokenAgentService()))
    update = make_update()

    await bridge.new_session(update, make_context(args=["/tmp"]))

    assert update.message is not None
    assert update.message.replies == ["Failed to start session: agent process did not expose stdio pipes."]


async def test_new_session_reports_generic_error():
    class BoomError(Exception):
        pass

    class UnexpectedService:
        async def new_session(self, *, chat_id: int, workspace):
            del chat_id, workspace
            raise BoomError("boom")

        async def prompt(self, *, chat_id: int, text: str):
            del chat_id, text

        def get_workspace(self, *, chat_id: int):
            del chat_id

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, UnexpectedService()))
    update = make_update()

    await bridge.new_session(update, make_context(args=["/tmp"]))

    assert update.message is not None
    assert update.message.replies == ["Failed to start session: boom"]


async def test_on_text_without_and_with_session():
    bridge = make_bridge()
    update = make_update(text="hello")
    context = make_context()

    await bridge.on_message(update, context)
    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message is not None
    assert len(update.message.replies) == EXPECTED_TEXT_REPLIES_WITH_IMPLICIT_AND_EXPLICIT_SESSION
    assert update.message.replies[0].endswith("hello")
    assert update.message.replies[-1].endswith("hello")
    assert context.bot.actions == [(100, "typing"), (100, "typing")]
    assert update.message.reply_kwargs[-1] == {}
    assert "parse_mode" not in update.message.reply_kwargs[-1]


async def test_first_prompt_starts_implicit_session_in_default_workspace(tmp_path: Path):
    service = RecordingImplicitService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path))
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    update = make_update(text="hello")

    await bridge.on_message(update, make_context())

    assert service.new_session_calls == [(TEST_CHAT_ID, tmp_path)]
    assert update.message is not None
    assert update.message.replies == ["ok"]


async def test_implicit_start_lock_is_dropped_once_session_exists(tmp_path: Path):
    service = RecordingImplicitService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path))
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))

    await bridge.on_message(make_update(chat_id=TEST_CHAT_ID, text="hello"), make_context())

    assert TEST_CHAT_ID not in bridge._implicit_start_locks_by_chat


async def test_drop_implicit_start_lock_keeps_lock_when_expected_lock_differs():
    bridge = make_bridge()
    stored_lock = asyncio.Lock()
    different_lock = asyncio.Lock()
    bridge._implicit_start_locks_by_chat[TEST_CHAT_ID] = stored_lock

    bridge._drop_implicit_start_lock(chat_id=TEST_CHAT_ID, expected_lock=different_lock)

    assert bridge._implicit_start_locks_by_chat[TEST_CHAT_ID] is stored_lock


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ValueError("/bad-default"), "Invalid default workspace: /bad-default"),
        (RuntimeError(), "Failed to start session: agent process did not expose stdio pipes."),
        (Exception("boom"), "Failed to start session: boom"),
    ],
)
async def test_first_prompt_reports_implicit_session_start_errors(error: Exception, expected: str):
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    service = FailingImplicitService(error)
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    update = make_update(text="hello")
    context = make_context()

    await bridge.on_message(update, context)

    assert update.message is not None
    assert update.message.replies == [expected]
    assert context.bot.actions == []


async def test_on_message_without_session_after_implicit_start_reports_missing_session():
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(
        config=config,
        agent_service=cast(AgentService, PromptWithoutSessionImplicitService()),
    )
    update = make_update(text="hello")

    await bridge.on_message(update, make_context())

    assert update.message is not None
    assert update.message.replies == ["No active session. Send a message again or use /new [workspace]."]


async def test_concurrent_first_prompts_start_only_one_implicit_session(tmp_path: Path):
    service = ConcurrentImplicitSessionService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=str(tmp_path))
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    update_one = make_update(chat_id=TEST_CHAT_ID, text="hello one")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="hello two")
    context_one = make_context()
    context_two = make_context()

    await asyncio.gather(
        bridge.on_message(update_one, context_one),
        bridge.on_message(update_two, context_two),
    )

    assert service.new_session_calls == 1
    assert update_one.message is not None
    assert update_two.message is not None
    assert update_one.message.replies == ["ok"]
    assert update_two.message.replies == ["ok"]
    assert TEST_CHAT_ID not in bridge._implicit_start_locks_by_chat

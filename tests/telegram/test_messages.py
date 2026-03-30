from __future__ import annotations

# ruff: noqa: F403, F405, I001

from tests.telegram.support import *


async def test_on_text_plain_reply_when_response_has_no_entities():
    bridge = make_bridge()
    update = make_update(text="hello")
    assert update.message is not None
    update.message.fail_entities = True
    context = make_context()

    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message.replies[-1].endswith("hello")
    assert update.message.reply_kwargs[-1] == {}


async def test_on_message_with_photo_attachment():
    bridge = make_bridge()
    photo = [SimpleNamespace(file_id="p1")]
    update = make_update(photo=photo)
    context = make_context()
    context.bot.files["p1"] = b"abc"

    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message is not None
    assert "images=1" in update.message.replies[-1]


async def test_on_message_with_document_attachment():
    bridge = make_bridge()
    document = SimpleNamespace(file_id="d1", mime_type="text/plain", file_name="note.txt")
    update = make_update(document=document)
    context = make_context()
    context.bot.files["d1"] = b"hello from file"

    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message is not None
    assert "files=1" in update.message.replies[-1]


async def test_on_message_with_binary_document_attachment():
    bridge = make_bridge()
    document = SimpleNamespace(file_id="bin-doc", mime_type="application/octet-stream", file_name="x.bin")
    update = make_update(document=document)
    context = make_context()
    context.bot.files["bin-doc"] = b"\xff\xfe"

    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message is not None
    assert "files=1" in update.message.replies[-1]


async def test_on_message_with_image_document_attachment():
    bridge = make_bridge()
    document = SimpleNamespace(file_id="img-doc", mime_type="image/png", file_name="x.png")
    update = make_update(document=document)
    context = make_context()
    context.bot.files["img-doc"] = b"\x89PNG"

    await bridge.new_session(update, make_context())
    await bridge.on_message(update, context)

    assert update.message is not None
    assert "images=1" in update.message.replies[-1]


async def test_outbound_agent_attachments_are_sent():
    class AttachmentService:
        async def new_session(self, *, chat_id: int, workspace):
            del workspace
            return f"s-{chat_id}"

        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del chat_id, text, images, files
            return AgentReply(
                text="ok",
                images=(ImagePayload(data_base64=base64.b64encode(b"img").decode("ascii"), mime_type="image/jpeg"),),
                files=(
                    FilePayload(name="out.txt", text_content="content"),
                    FilePayload(name="out.bin", data_base64=base64.b64encode(b"bin").decode("ascii")),
                ),
            )

        def get_workspace(self, *, chat_id: int):
            del chat_id

        async def cancel(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def stop(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def clear(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        def get_permission_policy(self, *, chat_id: int):
            del chat_id

        async def set_session_permission_mode(self, *, chat_id: int, mode):
            del chat_id, mode
            return False

        async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
            del chat_id, enabled
            return False

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, AttachmentService()))
    update = make_update(text="hello")

    await bridge.on_message(update, make_context())

    assert update.message is not None
    assert update.message.replies[-1] == "ok"
    assert len(update.message.photos) == 1
    assert len(update.message.documents) == EXPECTED_OUTBOUND_DOCUMENTS


async def test_on_message_renders_activity_blocks_before_final_reply():
    class ActivityService:
        async def new_session(self, *, chat_id: int, workspace):
            del workspace
            return f"s-{chat_id}"

        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del chat_id, text, images, files
            return AgentReply(
                text="Done.",
                activity_blocks=(
                    AgentActivityBlock(
                        kind="think",
                        title="Draft plan",
                        status="completed",
                        text="Need to inspect repository files.",
                    ),
                    AgentActivityBlock(
                        kind="execute",
                        title="Run tests",
                        status="completed",
                        text="uv run pytest",
                    ),
                ),
            )

        def get_workspace(self, *, chat_id: int):
            del chat_id

        async def cancel(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def stop(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def clear(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        def get_permission_policy(self, *, chat_id: int):
            del chat_id

        async def set_session_permission_mode(self, *, chat_id: int, mode):
            del chat_id, mode
            return False

        async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
            del chat_id, enabled
            return False

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, ActivityService()))
    update = make_update(text="hello")

    await bridge.on_message(update, make_context())

    assert update.message is not None
    assert len(update.message.replies) == EXPECTED_ACTIVITY_MESSAGES
    assert "💡 Thinking" in update.message.replies[0]
    assert "Draft plan" not in update.message.replies[0]
    assert "⚙️ Running" in update.message.replies[1]
    assert update.message.replies[2] == "Done."


async def test_on_message_sends_live_activity_events_via_app_bot():
    service = LiveActivityService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    update = make_update(text="hello")
    context = make_context()
    bridge._app = cast(Application, SimpleNamespace(bot=context.bot))

    await bridge.on_message(update, context)

    assert update.message is not None
    assert update.message.replies[-1] == "Final response."
    assert context.bot.sent_messages
    assert "💡 Thinking" in cast(str, context.bot.sent_messages[0]["text"])


async def test_on_message_skips_empty_final_text_reply():
    class EmptyTextService:
        async def new_session(self, *, chat_id: int, workspace):
            del workspace
            return f"s-{chat_id}"

        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del chat_id, text, images, files
            return AgentReply(text="")

        def get_workspace(self, *, chat_id: int):
            del chat_id

        async def cancel(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def stop(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def clear(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        def get_permission_policy(self, *, chat_id: int):
            del chat_id

        async def set_session_permission_mode(self, *, chat_id: int, mode):
            del chat_id, mode
            return False

        async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
            del chat_id, enabled
            return False

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, EmptyTextService()))
    update = make_update(text="hello")
    context = make_context()

    await bridge.on_message(update, context)

    assert update.message is not None
    assert update.message.replies == []


async def test_on_message_reports_acp_stdio_limit_error():
    class LimitErrorService:
        async def new_session(self, *, chat_id: int, workspace):
            del workspace
            return f"s-{chat_id}"

        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del chat_id, text, images, files
            raise AgentOutputLimitExceededError(ACP_STDIO_LIMIT_ERROR)

        def get_workspace(self, *, chat_id: int):
            del chat_id

        async def cancel(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def stop(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def clear(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        def get_permission_policy(self, *, chat_id: int):
            del chat_id

        async def set_session_permission_mode(self, *, chat_id: int, mode):
            del chat_id, mode
            return False

        async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
            del chat_id, enabled
            return False

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, LimitErrorService()))
    update = make_update(text="hello")
    context = make_context()

    await bridge.on_message(update, context)

    assert update.message is not None
    assert "Agent output exceeded ACP stdio limit." in update.message.replies[-1]


async def test_on_message_reraises_unrelated_value_error():
    class GenericValueErrorService:
        async def new_session(self, *, chat_id: int, workspace):
            del workspace
            return f"s-{chat_id}"

        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del chat_id, text, images, files
            raise ValueError("unexpected")

        def get_workspace(self, *, chat_id: int):
            del chat_id

        async def cancel(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def stop(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        async def clear(self, *, chat_id: int) -> bool:
            del chat_id
            return False

        def get_permission_policy(self, *, chat_id: int):
            del chat_id

        async def set_session_permission_mode(self, *, chat_id: int, mode):
            del chat_id, mode
            return False

        async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool):
            del chat_id, enabled
            return False

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, GenericValueErrorService()))
    update = make_update(text="hello")
    context = make_context()

    with pytest.raises(ValueError, match="unexpected"):
        await bridge.on_message(update, context)


async def test_on_activity_event_without_app_is_noop():
    bridge = make_bridge()
    block = AgentActivityBlock(kind="think", title="x", status="completed", text="y")
    await bridge.on_activity_event(TEST_CHAT_ID, block)


async def test_on_activity_event_markdown_fallback():
    bridge = make_bridge()
    failing_bot = FailingMarkdownBot()
    bridge._app = cast(Application, SimpleNamespace(bot=failing_bot))
    block = AgentActivityBlock(kind="execute", title="Run cmd", status="in_progress", text="")

    await bridge.on_activity_event(TEST_CHAT_ID, block)

    assert failing_bot.sent_messages
    assert "parse_mode" not in failing_bot.sent_messages[-1]


async def test_resume_keyboard_limits_to_ten_entries():
    candidates = tuple(
        ResumableSession(
            session_id=f"s-{index}",
            workspace=Path("/tmp/ws"),
            title=f"title {index}",
            updated_at="2026-03-02T12:00:00Z",
        )
        for index in range(12)
    )
    keyboard = TelegramBridge._resume_keyboard(candidates=candidates)
    assert len(keyboard.inline_keyboard) == RESUME_KEYBOARD_MAX_ROWS
    assert keyboard.inline_keyboard[0][0].text.startswith("0. ")
    assert keyboard.inline_keyboard[1][0].text.startswith("1. ")


async def test_format_activity_block_read_escapes_markdown_and_removes_read_prefix():
    block = AgentActivityBlock(
        kind="read", title="Read test_telegram_bot.py", status="completed", text="Read test_telegram_bot.py"
    )
    rendered = TelegramBridge._format_activity_block(block, workspace=Path("/tmp/ws"))
    assert "*📖 Reading*" in rendered
    assert "`/tmp/ws/test_telegram_bot.py`" in rendered
    assert "\n\nRead /tmp/ws/test_telegram_bot.py" not in rendered


async def test_format_activity_block_edit_uses_absolute_path_and_removes_edit_prefix():
    block = AgentActivityBlock(kind="edit", title="Edit src/telegram_acp_bot/telegram/bot.py", status="completed")
    rendered = TelegramBridge._format_activity_block(block, workspace=Path("/tmp/ws"))
    assert "*✏️ Editing*" in rendered
    assert "`/tmp/ws/src/telegram_acp_bot/telegram/bot.py`" in rendered
    assert "\n\nEdit /tmp/ws/src/telegram_acp_bot/telegram/bot.py" not in rendered


async def test_format_activity_block_read_without_workspace_keeps_relative_path():
    block = AgentActivityBlock(kind="read", title="Read README.md", status="completed", text="Read README.md")
    rendered = TelegramBridge._format_activity_block(block, workspace=None)
    assert "*📖 Reading*" in rendered
    assert f"`{Path.cwd().resolve() / 'README.md'}`" in rendered
    assert "`README.md`" not in rendered


async def test_format_activity_block_read_prefers_file_uri_path():
    block = AgentActivityBlock(
        kind="read",
        title="Read [@README.md](file:///home/tin/lab/telegram-acp/README.md)",
        status="completed",
    )
    rendered = TelegramBridge._format_activity_block(block, workspace=Path("/tmp/ws"))
    assert "`/home/tin/lab/telegram-acp/README.md`" in rendered


async def test_format_activity_block_preserves_thinking_inline_code():
    block = AgentActivityBlock(
        kind="think",
        title="",
        status="completed",
        text="Checking `README.md` and `docs/index.md`.",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "`README.md`" in rendered
    assert "`docs/index.md`" in rendered


async def test_format_activity_block_think_allows_basic_markdown_markers():
    block = AgentActivityBlock(
        kind="think",
        title="",
        status="completed",
        text="Working on **#90** before final patch.",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "**#90**" in rendered
    assert r"\*\*#90\*\*" not in rendered


async def test_format_activity_block_execute_wraps_command_as_fenced_code_block():
    block = AgentActivityBlock(kind="execute", title="Run git diff -- README.md docs/index.md", status="in_progress")
    rendered = TelegramBridge._format_activity_block(block)
    assert "⚙️ Running" in rendered
    assert "```\ngit diff -- README.md docs/index.md\n```" in rendered


async def test_format_activity_block_execute_multiline_command_uses_fenced_code_block():
    block = AgentActivityBlock(
        kind="execute",
        title="Run git diff -- README.md \\\n  docs/index.md",
        status="in_progress",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "```\ngit diff -- README.md \\\n  docs/index.md\n```" in rendered


async def test_format_activity_block_execute_long_command_uses_fenced_code_block():
    command = (
        "gh api repos/mgaitan/telegram-acp-bot/pulls/67/comments -X POST "
        "-F in_reply_to=2889504154 -f body='Implemented in 55881c9 with detailed context text'"
    )
    block = AgentActivityBlock(kind="execute", title=f"Run {command}", status="in_progress")

    rendered = TelegramBridge._format_activity_block(block)

    assert f"```\n{command}\n```" in rendered


async def test_format_activity_block_execute_command_with_backticks_uses_fenced_code_block():
    command = "gh api -f 'body=implemented in 55881c9. `path` and ACP_TELEGRAM_CHANNEL_ALLOW_PATH'"
    block = AgentActivityBlock(kind="execute", title=f"Run {command}", status="in_progress")

    rendered = TelegramBridge._format_activity_block(block)

    assert f"```\n{command}\n```" in rendered
    assert "\\_" not in rendered


async def test_format_activity_block_execute_command_with_triple_backticks_uses_longer_fence():
    command = "gh api -f 'body=markdown ```code``` example'"
    block = AgentActivityBlock(kind="execute", title=f"Run {command}", status="in_progress")

    rendered = TelegramBridge._format_activity_block(block)

    assert f"````\n{command}\n````" in rendered


async def test_format_activity_block_execute_preserves_escaped_backticks_and_underscores():
    command = r"gh api -f 'body=implemented in 55881c9. \`path\` and ACP_TELEGRAM_CHANNEL_ALLOW_PATH'"
    block = AgentActivityBlock(kind="execute", title=f"Run {command}", status="in_progress")

    rendered = TelegramBridge._format_activity_block(block)

    assert f"```\n{command}\n```" in rendered
    assert r"\`path\`" in rendered
    assert r"\\`path\\`" not in rendered
    assert "ACP_TELEGRAM_CHANNEL_ALLOW_PATH" in rendered


async def test_format_activity_block_search_uses_web_label_when_url_present():
    block = AgentActivityBlock(
        kind="search",
        title='Query: "telegram acp"',
        status="completed",
        text="URL: https://agentclientprotocol.com/",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "*🌐 Searching web*" in rendered


async def test_format_activity_block_search_uses_neutral_label_for_local_markers():
    block = AgentActivityBlock(
        kind="search",
        title='Query project for "ACP_TELEGRAM_CHANNEL_ALLOW_PATH"',
        status="completed",
        text="ripgrep in workspace",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "*🔎 Querying*" in rendered


async def test_format_activity_block_search_defaults_to_neutral_querying_label():
    block = AgentActivityBlock(
        kind="search",
        title='Query: "send now"',
        status="completed",
        text="",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "*🔎 Querying*" in rendered


async def test_format_activity_block_search_report_word_is_not_misclassified_as_repo():
    block = AgentActivityBlock(
        kind="search",
        title='Query: "annual report"',
        status="completed",
        text="",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "*🔎 Querying*" in rendered


async def test_format_activity_block_search_uses_neutral_label_for_file_uri():
    block = AgentActivityBlock(
        kind="search",
        title='Query: "config"',
        status="completed",
        text="file:///home/user/project/pyproject.toml",
    )
    rendered = TelegramBridge._format_activity_block(block)
    assert "*🔎 Querying*" in rendered


async def test_format_activity_block_reply_and_fallback_helpers():
    block = AgentActivityBlock(kind="reply", title="ignored", status="completed", text="final")

    assert TelegramBridge._format_activity_block(block) == "final"
    assert TelegramBridge._activity_label(block) == "✍️ Replying"
    assert TelegramBridge._activity_id(
        AgentActivityBlock(kind="think", title="Title", status="completed", text="")
    ) == ("think:Title")


async def test_format_permission_tool_title_empty_returns_empty():
    assert TelegramBridge._format_permission_tool_title("   ") == ""


async def test_format_permission_tool_title_non_run_keeps_title():
    assert TelegramBridge._format_permission_tool_title("Read README.md") == "Read README.md"


async def test_format_activity_block_execute_multiple_run_segments_use_consecutive_fenced_blocks():
    block = AgentActivityBlock(
        kind="execute",
        title="Run which ffmpeg, Run ffmpeg -y -f x11grab -i :0.0 -frames:v 1 /tmp/screenshot-ffmpeg.png",
        status="in_progress",
    )
    rendered = TelegramBridge._format_activity_block(block)
    expected = "```\nwhich ffmpeg\n```\n```\nffmpeg -y -f x11grab -i :0.0 -frames:v 1 /tmp/screenshot-ffmpeg.png\n```"
    assert expected in rendered


async def test_split_execute_commands_keeps_original_on_empty_segments():
    command = "which ffmpeg, Run "
    assert TelegramBridge._split_execute_commands(command) == [command]


async def test_format_fenced_code_without_language_uses_plain_fence():
    rendered = TelegramBridge._format_fenced_code("echo ok")
    assert rendered == "```\necho ok\n```"


async def test_send_helpers_with_no_message():
    update = make_update(with_message=False)
    image = ImagePayload(data_base64=base64.b64encode(b"img").decode("ascii"), mime_type="image/jpeg")
    file_payload = FilePayload(name="out.txt", text_content="content")

    await TelegramBridge._send_image(update, image)
    await TelegramBridge._send_file(update, file_payload)


async def test_reply_activity_block_with_no_message_is_noop():
    update = make_update(with_message=False)
    block = AgentActivityBlock(kind="think", title="t", status="completed", text="x")
    await TelegramBridge._reply_activity_block(update, block)


async def test_reply_activity_block_failed_status_appends_failed_marker():
    update = make_update()
    assert update.message is not None
    block = AgentActivityBlock(kind="other", title="Run command", status="failed", text="boom")

    await TelegramBridge._reply_activity_block(update, block)

    assert update.message.replies[-1].endswith("Failed")


async def test_send_file_with_empty_payload():
    update = make_update()
    assert update.message is not None
    payload = FilePayload(name="empty.bin")
    await TelegramBridge._send_file(update, payload)
    assert len(update.message.documents) == 1


async def test_on_permission_request_sends_buttons():
    bridge = make_bridge()
    dummy_bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=dummy_bot))

    request = PermissionRequest(
        chat_id=TEST_CHAT_ID,
        request_id="abc123",
        tool_title="Run ls",
        tool_call_id="call-1",
        available_actions=("always", "once", "deny"),
    )
    await bridge.on_permission_request(request)

    assert len(dummy_bot.sent_messages) == 1
    payload = dummy_bot.sent_messages[0]
    assert payload["chat_id"] == TEST_CHAT_ID
    assert cast(str, payload["text"]).startswith("⚠️ Permission required")
    assert cast(str, payload["text"]).endswith("ls")
    assert "parse_mode" not in payload
    assert "entities" in payload
    markup = payload["reply_markup"]
    assert markup is not None


async def test_on_permission_request_compact_reuses_status_message():
    bridge = make_compact_bridge()
    dummy_bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=dummy_bot))
    bridge._compact_status_msg_id[TEST_CHAT_ID] = COMPACT_STATUS_MSG_ID

    request = PermissionRequest(
        chat_id=TEST_CHAT_ID,
        request_id="abc123",
        tool_title="Run ls",
        tool_call_id="call-1",
        available_actions=("always", "once", "deny"),
    )
    await bridge.on_permission_request(request)

    assert dummy_bot.sent_messages == []
    assert len(dummy_bot.edited_messages) == 1
    payload = dummy_bot.edited_messages[0]
    assert payload["chat_id"] == TEST_CHAT_ID
    assert payload["message_id"] == COMPACT_STATUS_MSG_ID
    assert cast(str, payload["text"]).startswith("⚠️ Permission required")
    assert payload["reply_markup"] is not None


async def test_on_permission_request_formats_multiline_run_as_code_block():
    bridge = make_bridge()
    dummy_bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=dummy_bot))

    request = PermissionRequest(
        chat_id=TEST_CHAT_ID,
        request_id="abc-multi",
        tool_title="Run git diff -- README.md \\\n  docs/index.md",
        tool_call_id="call-multi",
        available_actions=("always", "once", "deny"),
    )
    await bridge.on_permission_request(request)

    assert len(dummy_bot.sent_messages) == 1
    payload = dummy_bot.sent_messages[0]
    assert cast(str, payload["text"]).endswith("git diff -- README.md \\\n  docs/index.md")
    assert "parse_mode" not in payload
    assert "entities" in payload


async def test_on_permission_request_markdown_fallback_uses_plain_text():
    bridge = make_bridge()
    failing_bot = FailingMarkdownBot()
    bridge._app = cast(Application, SimpleNamespace(bot=failing_bot))

    request = PermissionRequest(
        chat_id=TEST_CHAT_ID,
        request_id="abc-fallback",
        tool_title="Run ls",
        tool_call_id="call-fallback",
        available_actions=("always", "once", "deny"),
    )
    await bridge.on_permission_request(request)

    assert len(failing_bot.sent_messages) == 1
    payload = failing_bot.sent_messages[0]
    assert payload["text"] == "⚠️ Permission required\n\nls"
    assert "parse_mode" not in payload
    assert "entities" not in payload


async def test_on_permission_request_without_app_is_noop():
    bridge = make_bridge()
    request = PermissionRequest(
        chat_id=TEST_CHAT_ID,
        request_id="noop",
        tool_title="Run ls",
        tool_call_id="call-noop",
        available_actions=("once", "deny"),
    )
    await bridge.on_permission_request(request)


async def test_on_permission_callback_accepts_action():
    class PermissionService:
        def set_permission_request_handler(self, handler):
            del handler

        async def respond_permission_request(self, *, chat_id: int, request_id: str, action: str) -> bool:
            assert chat_id == TEST_CHAT_ID
            assert request_id == "req1"
            assert action == "once"
            return True

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, PermissionService()),
    )
    callback = DummyCallbackQuery("perm|req1|once")
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
        callback_query=callback,
        message=None,
    )

    await bridge.on_permission_callback(cast(Update, update), make_context())
    assert callback.answers[-1] == "Approved this time."
    assert callback.edited_text is not None
    assert "Permission required" in callback.edited_text
    assert "Decision: Approved this time." in callback.edited_text


async def test_on_permission_callback_preserves_original_entities():
    class PermissionService:
        def set_permission_request_handler(self, handler):
            del handler

        async def respond_permission_request(self, *, chat_id: int, request_id: str, action: str) -> bool:
            assert chat_id == TEST_CHAT_ID
            assert request_id == "req-entities"
            assert action == "once"
            return True

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, PermissionService()),
    )
    callback = DummyCallbackQuery("perm|req-entities|once")
    callback.message = SimpleNamespace(
        text="⚠️ Permission required\n\ngit diff -- README.md",
        entities=[MessageEntity(type=MessageEntity.PRE, offset=22, length=22)],
        chat=SimpleNamespace(id=TEST_CHAT_ID),
    )
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_permission_callback(update, make_context())

    assert callback.answers[-1] == "Approved this time."
    assert callback.edited_text is not None
    assert "Decision: Approved this time." in callback.edited_text
    assert "entities" in callback.edited_kwargs
    assert callback.edited_kwargs["entities"] == callback.message.entities


async def test_on_permission_callback_invalid_cases():
    bridge = make_bridge()
    update_no_query = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=None,
            message=None,
        ),
    )
    await bridge.on_permission_callback(update_no_query, make_context())

    callback = DummyCallbackQuery("invalid")
    update_invalid = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_permission_callback(update_invalid, make_context())
    assert callback.answers[-1] == "Invalid action."

    callback_bad_action = DummyCallbackQuery("perm|req1|weird")
    update_bad_action = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_bad_action,
            message=None,
        ),
    )
    await bridge.on_permission_callback(update_bad_action, make_context())
    assert callback_bad_action.answers[-1] == "Invalid action."

    callback_missing_chat = DummyCallbackQuery("perm|req1|once")
    callback_missing_chat.message = None
    update_missing_chat = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=None,
            callback_query=callback_missing_chat,
            message=None,
        ),
    )
    await bridge.on_permission_callback(update_missing_chat, make_context())
    assert callback_missing_chat.answers[-1] == "Missing chat."


async def test_on_permission_callback_access_denied():
    bridge = make_bridge(allowed_ids={9})
    callback = DummyCallbackQuery("perm|req1|deny")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_permission_callback(update, make_context())
    assert callback.answers[-1] == "Access denied."


async def test_on_permission_callback_expired_request():
    class ExpiredService:
        def set_permission_request_handler(self, handler):
            del handler

        async def respond_permission_request(self, *, chat_id: int, request_id: str, action: str) -> bool:
            del chat_id, request_id, action
            return False

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, ExpiredService()),
    )
    callback = DummyCallbackQuery("perm|req1|deny")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_permission_callback(update, make_context())
    assert callback.answers[-1] == "Request expired."


async def test_on_permission_callback_fallback_to_clear_markup_on_edit_error():
    class PermissionService:
        def set_permission_request_handler(self, handler):
            del handler

        async def respond_permission_request(self, *, chat_id: int, request_id: str, action: str) -> bool:
            del chat_id, request_id, action
            return True

    class FailingEditCallbackQuery(DummyCallbackQuery):
        async def edit_message_text(self, text: str, **kwargs: object) -> None:
            del text, kwargs
            raise MarkdownFailureError

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, PermissionService()),
    )
    callback = FailingEditCallbackQuery("perm|req1|deny")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_permission_callback(update, make_context())
    assert callback.answers[-1] == "Denied."
    assert callback.reply_markup_cleared


async def test_on_permission_callback_uses_query_message_chat_when_effective_chat_missing():
    class PermissionService:
        def set_permission_request_handler(self, handler):
            del handler

        async def respond_permission_request(self, *, chat_id: int, request_id: str, action: str) -> bool:
            assert chat_id == TEST_CHAT_ID
            assert request_id == "req-chat-fallback"
            assert action == "once"
            return True

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, PermissionService()),
    )
    callback = DummyCallbackQuery("perm|req-chat-fallback|once")
    callback.message = SimpleNamespace(text="Permission required\nRun ls", chat=SimpleNamespace(id=TEST_CHAT_ID))
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=None,
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_permission_callback(update, make_context())
    assert callback.answers[-1] == "Approved this time."
    assert callback.edited_text is not None
    assert "Decision: Approved this time." in callback.edited_text


async def test_on_permission_callback_handles_unexpected_exception():
    class FailingService:
        def set_permission_request_handler(self, handler):
            del handler

        async def respond_permission_request(self, *, chat_id: int, request_id: str, action: str) -> bool:
            del chat_id, request_id, action
            raise RuntimeError("boom")

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, FailingService()),
    )
    callback = DummyCallbackQuery("perm|req1|once")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_permission_callback(update, make_context())
    assert callback.answers[-1] == "Permission action failed."


async def test_on_resume_callback_invalid_cases():
    bridge = make_bridge()
    update_no_query = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=None,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_no_query, make_context())

    callback_invalid = DummyCallbackQuery("resume")
    update_invalid = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_invalid,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_invalid, make_context())
    assert callback_invalid.answers[-1] == "Invalid selection."

    callback_non_digit = DummyCallbackQuery("resume|x")
    update_non_digit = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_non_digit,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_non_digit, make_context())
    assert callback_non_digit.answers[-1] == "Invalid selection."


async def test_on_resume_callback_selection_expired_and_missing_chat():
    bridge = make_bridge()

    callback_expired = DummyCallbackQuery("resume|0")
    update_expired = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_expired,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_expired, make_context())
    assert callback_expired.answers[-1] == "Selection expired."

    callback_missing_chat = DummyCallbackQuery("resume|0")
    callback_missing_chat.message = None
    update_missing_chat = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=None,
            callback_query=callback_missing_chat,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_missing_chat, make_context())
    assert callback_missing_chat.answers[-1] == "Missing chat."


async def test_on_resume_callback_access_denied_and_invalid_index():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[999], workspace="."),
        agent_service=cast(AgentService, service),
    )
    callback_denied = DummyCallbackQuery("resume|0")
    update_denied = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_denied,
            message=None,
        ),
    )
    await bridge.on_resume_callback(update_denied, make_context())
    assert callback_denied.answers[-1] == "Access denied."

    bridge_ok = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    bridge_ok._pending_resume_choices_by_chat[TEST_CHAT_ID] = service.items
    callback_invalid_index = DummyCallbackQuery("resume|99")
    update_invalid_index = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_invalid_index,
            message=None,
        ),
    )
    await bridge_ok.on_resume_callback(update_invalid_index, make_context())
    assert callback_invalid_index.answers[-1] == "Invalid selection."


async def test_on_resume_callback_success_and_failure_paths():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    candidates = service.items
    bridge._pending_resume_choices_by_chat[TEST_CHAT_ID] = candidates
    callback = DummyCallbackQuery("resume|1")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=DummyMessage("trigger"),
        ),
    )
    await bridge.on_resume_callback(update, make_context())
    assert callback.answers[-1] == "Session resumed."
    assert callback.edited_text is not None
    assert "Resumed session: s-resume-2" in callback.edited_text
    assert "Workspace: /tmp/ws2" in callback.edited_text
    assert "Title: Second session" in callback.edited_text
    assert TEST_CHAT_ID not in bridge._pending_resume_choices_by_chat

    service.fail_load = True
    bridge._pending_resume_choices_by_chat[TEST_CHAT_ID] = candidates
    callback_fail = DummyCallbackQuery("resume|0")
    update_fail = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback_fail,
            message=DummyMessage("trigger"),
        ),
    )
    await bridge.on_resume_callback(update_fail, make_context())
    assert callback_fail.answers[-1] == "Failed to resume."
    assert TEST_CHAT_ID in bridge._pending_resume_choices_by_chat


async def test_on_resume_callback_fallback_to_clear_markup_on_edit_error():
    class FailingEditCallbackQuery(DummyCallbackQuery):
        async def edit_message_text(self, text: str, **kwargs: object) -> None:
            del text, kwargs
            raise MarkdownFailureError

    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    bridge._pending_resume_choices_by_chat[TEST_CHAT_ID] = service.items
    callback = FailingEditCallbackQuery("resume|0")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=DummyMessage("trigger"),
        ),
    )

    await bridge.on_resume_callback(update, make_context())
    assert callback.answers[-1] == "Session resumed."
    assert callback.reply_markup_cleared


async def test_on_resume_callback_uses_query_message_chat_when_effective_chat_missing():
    service = ResumeService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace="."),
        agent_service=cast(AgentService, service),
    )
    bridge._pending_resume_choices_by_chat[TEST_CHAT_ID] = service.items
    callback = DummyCallbackQuery("resume|0")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=None,
            callback_query=callback,
            message=DummyMessage("trigger"),
        ),
    )
    await bridge.on_resume_callback(update, make_context())
    assert callback.answers[-1] == "Session resumed."


async def test_cancel_stop_clear_without_session():
    bridge = make_bridge()
    update = make_update()
    context = make_context()

    await bridge.cancel(update, context)
    await bridge.stop(update, context)
    await bridge.clear(update, context)

    assert update.message is not None
    assert update.message.replies == [
        "No active session. Use /new first.",
        "No active session. Use /new first.",
        "No active session. Use /new first.",
    ]


async def test_format_activity_block_read_with_absolute_path_keeps_absolute():
    block = AgentActivityBlock(kind="read", title="Read /tmp/absolute.txt", status="completed")
    rendered = TelegramBridge._format_activity_block(block, workspace=Path("/tmp/ws"))
    assert "`/tmp/absolute.txt`" in rendered


async def test_format_read_path_empty_value_returns_empty_text():
    rendered = TelegramBridge._format_read_path("   ", workspace=Path("/tmp/ws"))
    assert rendered == ""


async def test_escape_markdown_preserving_code_escapes_special_chars_outside_code():
    rendered = TelegramBridge._escape_markdown_preserving_code("\\ _ * [ `code_[x]`")
    assert rendered == "\\\\ \\_ \\* \\[ `code_[x]`"


async def test_cancel_stop_clear_with_session():
    bridge = make_bridge()
    update = make_update()

    await bridge.new_session(update, make_context())
    await bridge.cancel(update, make_context())
    await bridge.stop(update, make_context())
    await bridge.clear(update, make_context())

    assert update.message is not None
    assert "Session started:" in update.message.replies[0]
    assert update.message.replies[1:] == [
        "Cancelled current operation.",
        "Stopped current session.",
        "No active session. Use /new first.",
    ]


async def test_clear_with_session():
    bridge = make_bridge()
    update = make_update()

    await bridge.new_session(update, make_context())
    await bridge.clear(update, make_context())

    assert update.message is not None
    assert "Session started:" in update.message.replies[0]
    assert update.message.replies[1] == "Cleared current session."


async def test_on_text_ignores_empty_message():
    bridge = make_bridge()
    update = make_update(text=None)
    context = make_context()

    await bridge.on_message(update, context)

    assert update.message is not None
    assert update.message.replies == []
    assert context.bot.actions == []


async def test_reply_with_no_message_object():
    bridge = make_bridge()
    update = make_update(with_message=False)

    await bridge.help(update, make_context())


async def test_reply_agent_with_no_message_object():
    update = make_update(with_message=False)
    await TelegramBridge._reply_agent(update, "x")


async def test_reply_agent_uses_entities_split_flow(monkeypatch: pytest.MonkeyPatch):
    update = make_update()
    assert update.message is not None

    entity = bot_module.MarkdownMessageEntity(type="bold", offset=0, length=5)
    monkeypatch.setattr(bridge_module, "convert", lambda text: (text, [entity]))
    monkeypatch.setattr(
        bridge_module,
        "split_entities",
        lambda text, entities, max_utf16_len: [("hello ", entities), ("world", [])],
    )

    await TelegramBridge._reply_agent(update, "hello world")

    assert update.message.replies == ["hello ", "world"]
    assert "entities" in update.message.reply_kwargs[0]
    assert "parse_mode" not in update.message.reply_kwargs[0]
    assert update.message.reply_kwargs[1] == {}


async def test_reply_agent_falls_back_to_plain_text_on_convert_error(
    monkeypatch: pytest.MonkeyPatch,
):
    update = make_update()
    assert update.message is not None

    def boom(_: str):
        raise RuntimeError

    monkeypatch.setattr(bridge_module, "convert", boom)

    await TelegramBridge._reply_agent(update, "*x*")

    assert update.message.reply_kwargs[-1] == {}
    assert update.message.replies[-1] == "*x*"


async def test_reply_falls_back_to_plain_when_entity_send_fails():
    update = make_update()
    assert update.message is not None
    update.message.fail_entities = True

    await TelegramBridge._reply(update, "*x*")

    assert "entities" in update.message.reply_kwargs[-2]
    assert update.message.reply_kwargs[-1] == {}
    assert update.message.replies[-1] == "x"


async def test_reply_agent_falls_back_to_plain_when_convert_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    update = make_update()
    assert update.message is not None

    def boom(_: str):
        raise RuntimeError

    monkeypatch.setattr(bridge_module, "convert", boom)

    await TelegramBridge._reply_agent(update, "*x*")

    assert update.message.reply_kwargs[-1] == {}
    assert update.message.replies[-1] == "*x*"


async def test_send_markdown_to_chat_falls_back_to_plain_when_entity_send_fails():
    bridge = make_bridge()
    failing_bot = FailingMarkdownBot()
    bridge._app = cast(Application, SimpleNamespace(bot=failing_bot))

    await TelegramBridge._send_markdown_to_chat(
        bot=cast(bot_module.Bot, failing_bot), chat_id=TEST_CHAT_ID, text="*bold*"
    )

    assert len(failing_bot.sent_messages) == 1
    payload = failing_bot.sent_messages[0]
    assert payload["text"] == "bold"
    assert "entities" not in payload


async def test_send_markdown_to_chat_without_entities_omits_entities_kwarg():
    dummy_bot = DummyBot()

    await TelegramBridge._send_markdown_to_chat(
        bot=cast(bot_module.Bot, dummy_bot),
        chat_id=TEST_CHAT_ID,
        text="plain text",
    )

    assert len(dummy_bot.sent_messages) == 1
    payload = dummy_bot.sent_messages[0]
    assert payload["text"] == "plain text"
    assert "entities" not in payload


async def test_send_markdown_to_chat_falls_back_to_plain_when_convert_fails(monkeypatch: pytest.MonkeyPatch):
    dummy_bot = DummyBot()

    def boom(_: str):
        raise ValueError

    monkeypatch.setattr(bridge_module, "convert", boom)

    await TelegramBridge._send_markdown_to_chat(
        bot=cast(bot_module.Bot, dummy_bot),
        chat_id=TEST_CHAT_ID,
        text="*bold*",
        reply_markup=InlineKeyboardMarkup([]),
    )

    assert len(dummy_bot.sent_messages) == 1
    payload = dummy_bot.sent_messages[0]
    assert payload["text"] == "*bold*"
    assert payload["reply_markup"] is not None


async def test_on_text_ignores_when_message_is_missing():
    bridge = make_bridge()
    update = make_update(with_message=False)
    context = make_context()

    await bridge.on_message(update, context)
    assert context.bot.actions == []


async def test_chat_id_without_chat_raises():
    update = cast(Update, SimpleNamespace(effective_chat=None))
    with pytest.raises(ChatRequiredError):
        TelegramBridge._chat_id(update)


async def test_active_session_context_returns_none_when_provider_is_not_callable():
    bridge = make_bridge()
    bridge._agent_service = cast(AgentService, SimpleNamespace(get_active_session_context="invalid"))

    assert bridge._active_session_context(chat_id=TEST_CHAT_ID) is None


async def test_build_application_installs_handlers():
    bridge = make_bridge()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")

    app = build_application(config, bridge)
    assert app.handlers
    assert app.update_processor.max_concurrent_updates > 1
    assert isinstance(app.bot.rate_limiter, AIORateLimiter)


async def test_run_polling(monkeypatch):
    calls: list[object] = []

    class DummyApp:
        def run_polling(self, *, allowed_updates):
            calls.append(allowed_updates)

    def fake_build_application(config, bridge):
        del config, bridge
        return DummyApp()

    monkeypatch.setattr(app_module, "build_application", fake_build_application)

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = make_bridge()
    assert run_polling(config, bridge) == 0
    assert len(calls) == 1


async def test_run_polling_returns_restart_exit_code(monkeypatch):
    class DummyApp:
        def run_polling(self, *, allowed_updates):
            del allowed_updates

    def fake_build_application(config, bridge):
        del config
        bridge._restart_requested = True
        return DummyApp()

    monkeypatch.setattr(app_module, "build_application", fake_build_application)

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = make_bridge()
    assert run_polling(config, bridge) == RESTART_EXIT_CODE


# ---------------------------------------------------------------------------
# Busy-state tests
# ---------------------------------------------------------------------------

from __future__ import annotations

# ruff: noqa: F403, F405, I001

from tests.telegram.support import *


async def test_make_config_compact_activity_defaults_to_false():
    config = make_config(token="T", allowed_user_ids=[1], workspace=".")
    assert config.compact_activity is False


async def test_make_config_compact_activity_can_be_set():
    config = make_config(token="T", allowed_user_ids=[1], workspace=".", compact_activity=True)
    assert config.compact_activity is True


async def test_compact_on_activity_event_sends_initial_status_message():
    bridge = make_compact_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    block = AgentActivityBlock(kind="think", title="Checking", status="in_progress", text="")

    await bridge.on_activity_event(TEST_CHAT_ID, block)

    assert len(bot.sent_messages) == 1
    status_text = cast(str, bot.sent_messages[0]["text"])
    assert status_text == "💡 Thinking."
    assert TEST_CHAT_ID in bridge._compact_status_msg_id


async def test_compact_on_activity_event_edits_existing_status_message():
    bridge = make_compact_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    bridge._compact_status_msg_id[TEST_CHAT_ID] = COMPACT_STATUS_MSG_ID
    block = AgentActivityBlock(kind="execute", title="Run ls", status="in_progress", text="")

    await bridge.on_activity_event(TEST_CHAT_ID, block)

    assert len(bot.sent_messages) == 0
    assert len(bot.edited_messages) == 1
    assert bot.edited_messages[0]["message_id"] == COMPACT_STATUS_MSG_ID
    assert cast(str, bot.edited_messages[0]["text"]) == "⚙️ Running."


async def test_compact_on_activity_event_send_failure_is_silent():
    bridge = make_compact_bridge()

    class SendFailBot(DummyBot):
        async def send_message(self, **kwargs: object) -> SimpleNamespace:
            raise MarkdownFailureError

    bot = SendFailBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    block = AgentActivityBlock(kind="think", title="x", status="in_progress", text="")

    await bridge.on_activity_event(TEST_CHAT_ID, block)

    assert TEST_CHAT_ID not in bridge._compact_status_msg_id


async def test_compact_on_activity_event_edit_failure_is_silent():
    bridge = make_compact_bridge()
    bot = FailingEditBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    bridge._compact_status_msg_id[TEST_CHAT_ID] = COMPACT_STATUS_MSG_ID
    block = AgentActivityBlock(kind="think", title="x", status="in_progress", text="")

    await bridge.on_activity_event(TEST_CHAT_ID, block)

    assert bridge._compact_status_msg_id[TEST_CHAT_ID] == COMPACT_STATUS_MSG_ID


async def test_compact_dispatch_reply_edits_status_message():
    bridge = make_compact_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    bridge._compact_status_msg_id[TEST_CHAT_ID] = COMPACT_STATUS_MSG_ID
    update = make_update(text="hello")
    reply = AgentReply(text="Final answer.")

    await bridge._dispatch_reply(chat_id=TEST_CHAT_ID, update=update, reply=reply)

    assert len(bot.edited_messages) == 1
    assert "Final answer." in cast(str, bot.edited_messages[0]["text"])
    assert update.message is not None
    assert update.message.replies == []
    assert TEST_CHAT_ID not in bridge._compact_status_msg_id


async def test_compact_dispatch_reply_fallback_on_edit_failure():
    bridge = make_compact_bridge()
    bot = FailingEditBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    bridge._compact_status_msg_id[TEST_CHAT_ID] = COMPACT_STATUS_MSG_ID
    update = make_update(text="hello")
    reply = AgentReply(text="Final answer.")

    await bridge._dispatch_reply(chat_id=TEST_CHAT_ID, update=update, reply=reply)

    assert (TEST_CHAT_ID, COMPACT_STATUS_MSG_ID) in bot.deleted_message_ids
    assert update.message is not None
    assert "Final answer." in update.message.replies[-1]
    assert TEST_CHAT_ID not in bridge._compact_status_msg_id


async def test_compact_dispatch_reply_no_status_message_sends_normally():
    bridge = make_compact_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(text="hello")
    reply = AgentReply(text="Final answer.")

    await bridge._dispatch_reply(chat_id=TEST_CHAT_ID, update=update, reply=reply)

    assert bot.edited_messages == []
    assert update.message is not None
    assert "Final answer." in update.message.replies[-1]


async def test_compact_dispatch_reply_empty_text_deletes_status():
    bridge = make_compact_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    bridge._compact_status_msg_id[TEST_CHAT_ID] = COMPACT_STATUS_MSG_ID
    update = make_update(text="hello")
    reply = AgentReply(text="")

    await bridge._dispatch_reply(chat_id=TEST_CHAT_ID, update=update, reply=reply)

    assert (TEST_CHAT_ID, COMPACT_STATUS_MSG_ID) in bot.deleted_message_ids
    assert update.message is not None
    assert update.message.replies == []


async def test_compact_dispatch_reply_empty_text_no_status_is_noop():
    bridge = make_compact_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(text="hello")
    reply = AgentReply(text="")

    await bridge._dispatch_reply(chat_id=TEST_CHAT_ID, update=update, reply=reply)

    assert bot.deleted_message_ids == []
    assert update.message is not None
    assert update.message.replies == []


async def test_finalize_compact_reply_without_app_sends_normally():
    bridge = make_compact_bridge()
    bridge._compact_status_msg_id[TEST_CHAT_ID] = COMPACT_STATUS_MSG_ID
    update = make_update(text="hello")

    await bridge._finalize_compact_reply(chat_id=TEST_CHAT_ID, update=update, text="Final answer.")

    assert update.message is not None
    assert "Final answer." in update.message.replies[-1]
    assert TEST_CHAT_ID not in bridge._compact_status_msg_id


async def test_clear_compact_status_without_app_is_noop():
    bridge = make_compact_bridge()
    bridge._compact_status_msg_id[TEST_CHAT_ID] = COMPACT_STATUS_MSG_ID

    await bridge._clear_compact_status(TEST_CHAT_ID)

    assert TEST_CHAT_ID not in bridge._compact_status_msg_id


async def test_clear_compact_status_no_status_is_noop():
    bridge = make_compact_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    await bridge._clear_compact_status(TEST_CHAT_ID)

    assert bot.deleted_message_ids == []


async def test_clear_compact_status_delete_failure_is_silent():
    bridge = make_compact_bridge()

    class DeleteFailBot(DummyBot):
        async def delete_message(self, **kwargs: object) -> None:
            raise MarkdownFailureError

    bot = DeleteFailBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    bridge._compact_status_msg_id[TEST_CHAT_ID] = COMPACT_STATUS_MSG_ID

    await bridge._clear_compact_status(TEST_CHAT_ID)

    assert TEST_CHAT_ID not in bridge._compact_status_msg_id


async def test_compact_live_activity_full_flow():
    """Integration: compact mode sends one status, then edits it with the final answer."""
    service = LiveActivityService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".", compact_activity=True)
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(chat_id=TEST_CHAT_ID, text="hello")
    context = make_context()
    context.bot = bot

    await bridge.on_message(update, context)

    assert update.message is not None
    assert update.message.replies == []
    assert len(bot.sent_messages) == 1
    # Status text keeps the normal emoji language in compact mode.
    status_text = cast(str, bot.sent_messages[0]["text"])
    assert status_text == "💡 Thinking."
    assert len(bot.edited_messages) == 1
    assert "Final response." in cast(str, bot.edited_messages[0]["text"])


async def test_edit_markdown_in_chat_returns_true_on_success():

    bot = DummyBot()
    result = await TelegramBridge._edit_markdown_in_chat(
        bot=cast(Bot, bot), chat_id=TEST_CHAT_ID, message_id=COMPACT_STATUS_MSG_ID, text="Hello world"
    )
    assert result is True
    assert bot.edited_messages


async def test_edit_markdown_in_chat_returns_true_for_text_with_entities():

    bot = DummyBot()
    result = await TelegramBridge._edit_markdown_in_chat(
        bot=cast(Bot, bot), chat_id=TEST_CHAT_ID, message_id=COMPACT_STATUS_MSG_ID, text="*bold text*"
    )
    assert result is True
    assert bot.edited_messages


async def test_edit_markdown_in_chat_returns_false_on_multi_chunk():

    bot = DummyBot()
    long_text = "word " * 1500
    result = await TelegramBridge._edit_markdown_in_chat(
        bot=cast(Bot, bot), chat_id=TEST_CHAT_ID, message_id=COMPACT_STATUS_MSG_ID, text=long_text
    )
    assert result is False
    assert not bot.edited_messages


async def test_edit_markdown_in_chat_entity_edit_fails_falls_back_to_plain():

    bot = EntityFailingEditBot()
    result = await TelegramBridge._edit_markdown_in_chat(
        bot=cast(Bot, bot), chat_id=TEST_CHAT_ID, message_id=COMPACT_STATUS_MSG_ID, text="*bold text*"
    )
    assert result is True
    assert bot.edited_messages
    assert "entities" not in bot.edited_messages[-1]


async def test_edit_markdown_in_chat_treats_not_modified_as_success():

    bot = NotModifiedAwareBot()
    sent = await bot.send_message(chat_id=TEST_CHAT_ID, text="Hello")
    result = await TelegramBridge._edit_markdown_in_chat(
        bot=cast(Bot, bot), chat_id=TEST_CHAT_ID, message_id=sent.message_id, text="Hello"
    )
    assert result is True
    assert bot.edited_messages == []


async def test_edit_markdown_in_chat_returns_false_when_all_entity_edits_fail():

    bot = FailingEditBot()
    result = await TelegramBridge._edit_markdown_in_chat(
        bot=cast(Bot, bot), chat_id=TEST_CHAT_ID, message_id=COMPACT_STATUS_MSG_ID, text="*bold text*"
    )
    assert result is False


async def test_edit_markdown_in_chat_returns_false_when_plain_edit_fails():

    bot = FailingEditBot()
    result = await TelegramBridge._edit_markdown_in_chat(
        bot=cast(Bot, bot), chat_id=TEST_CHAT_ID, message_id=COMPACT_STATUS_MSG_ID, text="Simple text"
    )
    assert result is False


async def test_edit_markdown_in_chat_convert_error_falls_back_to_plain(mocker):

    mocker.patch("telegram_acp_bot.telegram.bridge.convert", side_effect=RuntimeError("bad"))
    bot = DummyBot()
    result = await TelegramBridge._edit_markdown_in_chat(
        bot=cast(Bot, bot), chat_id=TEST_CHAT_ID, message_id=COMPACT_STATUS_MSG_ID, text="Hello"
    )
    assert result is True
    assert bot.edited_messages


async def test_edit_markdown_in_chat_convert_error_returns_false_when_edit_fails(mocker):

    mocker.patch("telegram_acp_bot.telegram.bridge.convert", side_effect=RuntimeError("bad"))
    bot = FailingEditBot()
    result = await TelegramBridge._edit_markdown_in_chat(
        bot=cast(Bot, bot), chat_id=TEST_CHAT_ID, message_id=COMPACT_STATUS_MSG_ID, text="Hello"
    )
    assert result is False


async def test_verbose_final_reply_replaces_streamed_reply_preview_when_markdown_finishes():

    class StreamingMarkdownReplyService(LiveActivityService):
        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del text, images, files
            assert self._activity_handler is not None
            await self._activity_handler(
                chat_id,
                AgentActivityBlock(kind="reply", title="", status="in_progress", text="**bold", activity_id="reply"),
            )
            return AgentReply(text="**bold**")

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=".", activity_mode="verbose"),
        agent_service=cast(AgentService, StreamingMarkdownReplyService()),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(chat_id=TEST_CHAT_ID, text="hello")
    context = make_context()
    context.bot = bot

    await bridge.on_message(update, context)

    assert len(bot.sent_messages) == 1
    assert len(bot.edited_messages) == 1
    assert cast(int, bot.edited_messages[0]["message_id"]) == 1
    assert cast(str, bot.edited_messages[0]["text"]) == "bold"
    assert bot.edited_messages[0]["entities"]
    assert update.message is not None
    assert update.message.replies == []


async def test_verbose_final_reply_keeps_markdown_when_final_matches_streamed_preview():

    class StableStreamingMarkdownReplyService(LiveActivityService):
        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del text, images, files
            reply = (
                'Sí, pero te conviene separar `"extraer emails plausibles"` de '
                '`"validar RFC completo"`.\n\n'
                "```python\n"
                "print('ok')\n"
                "```"
            )
            assert self._activity_handler is not None
            await self._activity_handler(
                chat_id,
                AgentActivityBlock(kind="reply", title="", status="in_progress", text=reply, activity_id="reply"),
            )
            return AgentReply(text=reply)

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=".", activity_mode="verbose"),
        agent_service=cast(AgentService, StableStreamingMarkdownReplyService()),
    )
    bot = NotModifiedAwareBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(chat_id=TEST_CHAT_ID, text="hello")
    context = make_context()
    context.bot = bot

    await bridge.on_message(update, context)

    assert len(bot.sent_messages) == 1
    assert bot.edited_messages == []
    payload = bot.sent_messages[0]
    assert "entities" in payload
    assert payload["entities"]
    assert update.message is not None
    assert update.message.replies == []


async def test_verbose_finalize_reply_handles_empty_text_and_empty_chunks(mocker):
    bridge = make_verbose_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    handler = cast(bot_module._VerboseActivityModeHandler, bridge._activity_handler(chat_id=TEST_CHAT_ID))
    handler._store_message(
        chat_id=TEST_CHAT_ID,
        slot_key="activity:reply",
        message=bot_module._VerboseActivityMessage(
            activity_id="reply",
            kind="reply",
            title="",
            message_id=1,
            source_text="preview",
        ),
    )

    assert await handler.finalize_reply(chat_id=TEST_CHAT_ID, update=cast(Update, make_update()), text="") is True

    handler._store_message(
        chat_id=TEST_CHAT_ID,
        slot_key="activity:reply",
        message=bot_module._VerboseActivityMessage(
            activity_id="reply",
            kind="reply",
            title="",
            message_id=2,
            source_text="preview",
        ),
    )
    mocker.patch.object(TelegramBridge, "_render_markdown_chunks", return_value=[])
    assert (
        await handler.finalize_reply(
            chat_id=TEST_CHAT_ID, update=cast(Update, make_update()), text="still empty chunks"
        )
        is True
    )


async def test_verbose_final_reply_deletes_preview_when_final_edit_fails():

    class StreamingReplyService(LiveActivityService):
        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del text, images, files
            assert self._activity_handler is not None
            await self._activity_handler(
                chat_id,
                AgentActivityBlock(kind="reply", title="", status="in_progress", text="preview", activity_id="reply"),
            )
            return AgentReply(text="Final response.")

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=".", activity_mode="verbose"),
        agent_service=cast(AgentService, StreamingReplyService()),
    )
    bot = FailingEditBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(chat_id=TEST_CHAT_ID, text="hello")
    context = make_context()
    context.bot = bot

    await bridge.on_message(update, context)

    assert bot.deleted_message_ids == [(TEST_CHAT_ID, 1)]
    assert update.message is not None
    assert update.message.replies == ["Final response."]


async def test_verbose_in_progress_updates_are_coalesced_before_editing(mocker):
    mocker.patch.object(activity_module, "VERBOSE_STREAM_TICK_SECONDS", 0.01)
    bridge = make_verbose_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    await bridge.on_activity_event(
        TEST_CHAT_ID,
        AgentActivityBlock(kind="reply", title="", status="in_progress", text="one", activity_id="reply"),
    )
    await bridge.on_activity_event(
        TEST_CHAT_ID,
        AgentActivityBlock(kind="reply", title="", status="in_progress", text="one two", activity_id="reply"),
    )
    await bridge.on_activity_event(
        TEST_CHAT_ID,
        AgentActivityBlock(kind="reply", title="", status="in_progress", text="one two three", activity_id="reply"),
    )
    await asyncio.sleep(0.03)

    assert len(bot.sent_messages) == 1
    assert len(bot.edited_messages) == 1
    assert cast(str, bot.edited_messages[-1]["text"]) == "one two three"


async def test_verbose_reset_clears_older_pending_preview_before_flush(mocker):
    mocker.patch.object(activity_module, "VERBOSE_STREAM_TICK_SECONDS", 0.01)
    bridge = make_verbose_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    await bridge.on_activity_event(
        TEST_CHAT_ID,
        AgentActivityBlock(kind="reply", title="", status="in_progress", text="one", activity_id="reply"),
    )
    await bridge.on_activity_event(
        TEST_CHAT_ID,
        AgentActivityBlock(kind="reply", title="", status="in_progress", text="one two", activity_id="reply"),
    )
    await bridge.on_activity_event(
        TEST_CHAT_ID,
        AgentActivityBlock(kind="reply", title="", status="in_progress", text="restart", activity_id="reply"),
    )
    await asyncio.sleep(0.03)

    assert len(bot.sent_messages) == EXPECTED_REPEATED_ACTIVITY_MESSAGES
    assert cast(str, bot.sent_messages[-1]["text"]) == "restart"
    assert all(cast(str, item.get("text", "")) != "one two" for item in bot.edited_messages)


async def test_verbose_in_progress_long_preview_stays_single_message(mocker):
    mocker.patch.object(activity_module, "VERBOSE_STREAM_TICK_SECONDS", 0.01)
    bridge = make_verbose_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    long_text = "a" * 5000
    longer_text = "a" * 5200

    await bridge.on_activity_event(
        TEST_CHAT_ID,
        AgentActivityBlock(kind="reply", title="", status="in_progress", text=long_text, activity_id="reply"),
    )
    await bridge.on_activity_event(
        TEST_CHAT_ID,
        AgentActivityBlock(kind="reply", title="", status="in_progress", text=longer_text, activity_id="reply"),
    )
    await asyncio.sleep(0.03)

    assert len(bot.sent_messages) == 1
    assert len(bot.edited_messages) == 1


async def test_verbose_in_progress_preview_helpers_handle_missing_preview(mocker):
    bridge = make_verbose_bridge()
    handler = cast(bot_module._VerboseActivityModeHandler, bridge._activity_handler(chat_id=TEST_CHAT_ID))
    bot = DummyBot()

    mocker.patch.object(TelegramBridge, "_render_markdown_preview_chunk", return_value=None)

    assert (
        await handler._edit_in_progress_preview(
            bot=cast(Bot, bot),
            chat_id=TEST_CHAT_ID,
            message_id=123,
            text="preview",
        )
        is False
    )
    assert await handler._send_in_progress_preview(bot=cast(Bot, bot), chat_id=TEST_CHAT_ID, text="preview") is None


async def test_render_markdown_preview_chunk_handles_empty_and_fallback(mocker):
    short_text = "plain preview"
    mocker.patch.object(TelegramBridge, "_render_markdown_chunks", side_effect=RuntimeError("boom"))
    short_preview = TelegramBridge._render_markdown_preview_chunk(short_text)

    assert short_preview == (short_text, None)

    long_text = "a" * (bot_module.TELEGRAM_MAX_UTF16_MESSAGE_LENGTH + 50)
    long_preview = TelegramBridge._render_markdown_preview_chunk(long_text)

    assert long_preview is not None
    assert long_preview[1] is None
    assert long_preview[0].endswith("...")
    assert bot_module.utf16_len(long_preview[0]) <= bot_module.TELEGRAM_MAX_UTF16_MESSAGE_LENGTH

    mocker.patch.object(TelegramBridge, "_render_markdown_chunks", return_value=[])
    assert TelegramBridge._render_markdown_preview_chunk("ignored") is None


async def test_verbose_completed_event_without_active_message_sends_and_clears():
    bridge = make_verbose_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    await bridge.on_activity_event(
        TEST_CHAT_ID,
        AgentActivityBlock(kind="execute", title="Run", status="completed", text="output", activity_id="exec-1"),
    )

    assert len(bot.sent_messages) == 1


async def test_verbose_internal_helpers_cover_remaining_branches(mocker):
    bridge = make_verbose_bridge()
    handler = cast(bot_module._VerboseActivityModeHandler, bridge._activity_handler(chat_id=TEST_CHAT_ID))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    assert await handler.finalize_reply(chat_id=TEST_CHAT_ID, update=cast(Update, make_update()), text="hello") is False

    handler._clear_message(chat_id=TEST_CHAT_ID, slot_key="missing")

    bridge._app = None
    await handler._apply_block_locked(
        chat_id=TEST_CHAT_ID,
        slot_key="activity:reply",
        block=AgentActivityBlock(kind="reply", title="", status="in_progress", text="preview", activity_id="reply"),
    )

    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    handler._store_message(
        chat_id=TEST_CHAT_ID,
        slot_key="activity:reply",
        message=bot_module._VerboseActivityMessage(
            activity_id="reply",
            kind="reply",
            title="",
            message_id=7,
            source_text="preview",
        ),
    )
    mocker.patch.object(TelegramBridge, "_edit_markdown_in_chat", return_value=True)
    await handler._apply_block_locked(
        chat_id=TEST_CHAT_ID,
        slot_key="activity:reply",
        block=AgentActivityBlock(kind="reply", title="", status="completed", text="done", activity_id="reply"),
    )
    assert "activity:reply" not in handler._messages_by_chat.get(TEST_CHAT_ID, {})

    handler._store_message(
        chat_id=TEST_CHAT_ID,
        slot_key="activity:reply",
        message=bot_module._VerboseActivityMessage(
            activity_id="reply",
            kind="reply",
            title="",
            message_id=8,
            source_text="preview",
        ),
    )
    mocker.patch.object(handler, "_edit_in_progress_preview", return_value=False)
    mocked_send = mocker.patch.object(handler, "_send_in_progress_preview", return_value=None)
    await handler._apply_block_locked(
        chat_id=TEST_CHAT_ID,
        slot_key="activity:reply",
        block=AgentActivityBlock(kind="reply", title="", status="in_progress", text="retry", activity_id="reply"),
    )
    assert mocked_send.called
    assert "activity:reply" not in handler._messages_by_chat.get(TEST_CHAT_ID, {})

    mocked_message = SimpleNamespace(message_id=9)
    mocker.patch.object(TelegramBridge, "_send_markdown_to_chat", return_value=mocked_message)
    await handler._apply_block_locked(
        chat_id=TEST_CHAT_ID,
        slot_key="activity:exec-2",
        block=AgentActivityBlock(kind="execute", title="Run", status="completed", text="done", activity_id="exec-2"),
    )
    assert "activity:exec-2" not in handler._messages_by_chat.get(TEST_CHAT_ID, {})


async def test_compact_stale_status_cleared_when_reply_is_none():
    """Compact status message is deleted when the prompt cycle exits without a final reply."""

    class LimitErrorCompactService(LiveActivityService):
        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            if self._activity_handler is not None:
                await self._activity_handler(
                    chat_id,
                    AgentActivityBlock(kind="think", title="Thinking…", status="in_progress", text=""),
                )
            raise AgentOutputLimitExceededError(ACP_STDIO_LIMIT_ERROR)

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".", compact_activity=True)
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, LimitErrorCompactService()))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(chat_id=TEST_CHAT_ID, text="hello")
    context = make_context()
    context.bot = bot

    await bridge.on_message(update, context)

    assert TEST_CHAT_ID not in bridge._compact_status_msg_id
    assert len(bot.deleted_message_ids) == 1
    assert update.message is not None
    assert "Agent output exceeded ACP stdio limit." in update.message.replies[-1]


async def test_animate_compact_status_returns_when_app_is_missing():
    bridge = make_compact_bridge()
    await bridge._animate_compact_status(chat_id=TEST_CHAT_ID, message_id=1)


async def test_compact_rotating_dots_cycle(mocker):
    """Background compact animation rotates dots: . → .. → ... → ."""
    stop_after_steps = 3
    bridge = make_compact_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    block = AgentActivityBlock(kind="think", title="x", status="in_progress", text="")

    await bridge.on_activity_event(TEST_CHAT_ID, block)

    steps = 0

    async def fake_sleep(_: float) -> None:
        nonlocal steps
        steps += 1
        if steps >= stop_after_steps:
            bridge._compact_status_label.pop(TEST_CHAT_ID, None)

    mocker.patch("telegram_acp_bot.telegram.bridge.asyncio.sleep", side_effect=fake_sleep)
    await bridge._animate_compact_status(chat_id=TEST_CHAT_ID, message_id=1)

    texts = [cast(str, bot.sent_messages[0]["text"])] + [cast(str, m["text"]) for m in bot.edited_messages]
    assert texts[0].startswith("💡 Thinking")
    # Dot sequence advances while the animation task is alive.
    assert texts[0].endswith(".")
    assert texts[1].endswith("..")
    assert texts[2].endswith("...")


async def test_compact_concurrent_activity_sends_only_one_message():
    """Race-condition guard: concurrent activity events produce exactly one send."""
    bridge = make_compact_bridge()
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    block = AgentActivityBlock(kind="think", title="x", status="in_progress", text="")

    # Schedule three events concurrently; without the per-chat lock this would
    # create three separate status messages because all three coroutines would
    # read existing_msg_id=None before the first send_message completes.
    await asyncio.gather(
        bridge.on_activity_event(TEST_CHAT_ID, block),
        bridge.on_activity_event(TEST_CHAT_ID, block),
        bridge.on_activity_event(TEST_CHAT_ID, block),
    )

    assert len(bot.sent_messages) == 1, "Only one status message should be created"
    assert TEST_CHAT_ID in bridge._compact_status_msg_id
    assert TEST_CHAT_ID in bridge._compact_status_tasks


async def test_compact_live_activity_full_flow_multi_activity():
    """Integration: compact mode keeps one status message across multiple activity updates."""

    class MultiActivityService(LiveActivityService):
        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del text, images, files
            if self._activity_handler is not None:
                for kind in ("think", "execute"):
                    await self._activity_handler(
                        chat_id,
                        AgentActivityBlock(kind=kind, title="Working", status="in_progress", text=""),
                    )
            return AgentReply(text="Done!")

    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".", compact_activity=True)
    service = MultiActivityService()
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    update = make_update(chat_id=TEST_CHAT_ID, text="hello")
    context = make_context()
    context.bot = bot

    await bridge.on_message(update, context)

    # Exactly one status message sent, zero direct replies.
    assert len(bot.sent_messages) == 1
    assert update.message is not None
    assert update.message.replies == []


async def test_normal_activity_state_is_cleared_after_successful_prompt_cycle():

    class ReusedActivityIdService(LiveActivityService):
        async def prompt(self, *, chat_id: int, text: str, images=(), files=()):
            del images, files
            assert self._activity_handler is not None
            await self._activity_handler(
                chat_id,
                AgentActivityBlock(
                    kind="execute",
                    title="Run command",
                    status="in_progress",
                    text="",
                    activity_id="shared-id",
                ),
            )
            return AgentReply(text=f"done:{text}")

    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=".", activity_mode="normal"),
        agent_service=cast(AgentService, ReusedActivityIdService()),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    first_update = make_update(chat_id=TEST_CHAT_ID, text="first")
    second_update = make_update(chat_id=TEST_CHAT_ID, text="second")
    first_context = make_context()
    first_context.bot = bot
    second_context = make_context()
    second_context.bot = bot

    await bridge.on_message(first_update, first_context)
    await bridge.on_message(second_update, second_context)

    activity_texts = [
        cast(str, payload["text"]) for payload in bot.sent_messages if "Running" in cast(str, payload["text"])
    ]
    assert len(activity_texts) == EXPECTED_REPEATED_ACTIVITY_MESSAGES

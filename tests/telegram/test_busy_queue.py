from __future__ import annotations

# ruff: noqa: F403, F405, I001

from tests.telegram.support import *


class BlockingService:
    """Service whose prompt blocks until `release()` is called, for busy-state tests."""

    def __init__(self) -> None:
        self._workspace: Path | None = None
        self._prompt_started = asyncio.Event()
        self._prompt_gate = asyncio.Event()
        self.cancelled = False
        self.prompts: list[str] = []

    async def new_session(self, *, chat_id: int, workspace: Path) -> str:
        del chat_id
        self._workspace = workspace
        return "s-blocking"

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()) -> AgentReply:
        del chat_id, images, files
        self.prompts.append(text)
        self._prompt_started.set()
        await self._prompt_gate.wait()
        return AgentReply(text=f"done:{text}")

    def get_workspace(self, *, chat_id: int) -> Path | None:
        del chat_id
        return self._workspace

    async def cancel(self, *, chat_id: int) -> bool:
        del chat_id
        self.cancelled = True
        self._prompt_gate.set()
        return True

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

    def release(self) -> None:
        self._prompt_gate.set()


class BlockingActivityService(BlockingService):
    """Blocking service that emits a visible activity event before waiting."""

    def __init__(self, *, kind: str = "think", title: str = "Thinking") -> None:
        super().__init__()
        self._activity_handler: Callable[[int, AgentActivityBlock], Awaitable[None]] | None = None
        self._kind = kind
        self._title = title

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()) -> AgentReply:
        del images, files
        self.prompts.append(text)
        if self._activity_handler is not None:
            await self._activity_handler(
                chat_id,
                AgentActivityBlock(kind=self._kind, title=self._title, status="in_progress", text=""),
            )
        self._prompt_started.set()
        await self._prompt_gate.wait()
        return AgentReply(text=f"done:{text}")

    def set_activity_event_handler(self, handler):
        self._activity_handler = handler


class WindowAwareBlockingService:
    """Blocking service that reports whether a prompt is currently active."""

    def __init__(self) -> None:
        self._workspace: Path | None = None
        self._prompt_started = asyncio.Event()
        self._prompt_gate = asyncio.Event()
        self._prompt_progress = asyncio.Condition()
        self.active_prompt = False
        self.prompts: list[str] = []

    async def new_session(self, *, chat_id: int, workspace: Path) -> str:
        del chat_id
        self._workspace = workspace
        return "s-window-aware"

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()) -> AgentReply:
        del chat_id, images, files
        async with self._prompt_progress:
            self.prompts.append(text)
            self._prompt_progress.notify_all()
        self.active_prompt = True
        self._prompt_started.set()
        await self._prompt_gate.wait()
        self.active_prompt = False
        self._prompt_started = asyncio.Event()
        self._prompt_gate = asyncio.Event()
        return AgentReply(text=f"done:{text}")

    def get_workspace(self, *, chat_id: int) -> Path | None:
        del chat_id
        return self._workspace

    def supports_session_loading(self, *, chat_id: int) -> bool | None:
        del chat_id
        return None

    async def wait_for_prompt_count(self, count: int) -> None:
        async with self._prompt_progress:
            await self._prompt_progress.wait_for(lambda: len(self.prompts) >= count)

    async def cancel(self, *, chat_id: int) -> bool:
        del chat_id
        if not self.active_prompt:
            return False
        self._prompt_gate.set()
        return True

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

    def set_permission_request_handler(self, handler):
        del handler

    def set_activity_event_handler(self, handler):
        del handler

    async def respond_permission_request(self, *, chat_id: int, request_id: str, action):
        del chat_id, request_id, action
        return False


class FailingCancelService:
    def __init__(self) -> None:
        self._workspace: Path | None = Path(".")

    async def new_session(self, *, chat_id: int, workspace: Path) -> str:
        del chat_id
        self._workspace = workspace
        return "s-fail"

    async def prompt(self, *, chat_id: int, text: str, images=(), files=()) -> AgentReply:
        del chat_id, text, images, files
        return AgentReply(text="ok")

    def get_workspace(self, *, chat_id: int) -> Path | None:
        del chat_id
        return self._workspace

    async def cancel(self, *, chat_id: int) -> bool:
        del chat_id
        raise DummyCancelBoomError

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


async def test_on_message_while_busy_shows_send_now_button():
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first", message_id=11)
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second", message_id=QUEUED_MESSAGE_ID)
    context = make_context(application=SimpleNamespace(bot=bot))

    # Start first message - it will block
    task_one = asyncio.create_task(bridge.on_message(update_one, context))

    # Wait until first prompt is actually running
    await service._prompt_started.wait()

    # Send second message while busy
    await bridge.on_message(update_two, context)

    # The second message should be queued and a "Send now" button should appear
    assert len(bot.sent_messages) == 1
    busy_msg = bot.sent_messages[0]
    assert busy_msg["chat_id"] == TEST_CHAT_ID
    assert busy_msg["reply_to_message_id"] == QUEUED_MESSAGE_ID
    assert "queued" in cast(str, busy_msg["text"]).lower()
    markup = cast(InlineKeyboardMarkup, busy_msg["reply_markup"])
    assert markup is not None
    button = markup.inline_keyboard[0][0]
    assert button.text == "Send now"
    assert button.callback_data is not None
    assert cast(str, button.callback_data).startswith(f"{BUSY_CALLBACK_PREFIX}|")

    # Finish first task
    service.release()
    await task_one


async def test_on_message_queued_runs_automatically_and_button_is_removed():
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    await bridge.on_message(update_two, context)

    # Notify button was shown with message_id=1
    assert bot.sent_messages[0]["reply_markup"] is not None

    # Release first prompt; pump loop should clear button then process second
    service.release()
    await task_one

    assert any(edit.get("message_id") == 1 and edit.get("text") == BUSY_SENT_TEXT for edit in bot.edited_messages)
    # Both updates should have received replies
    assert update_one.message is not None
    assert update_two.message is not None
    assert "done:first" in update_one.message.replies[-1]
    assert "done:second" in update_two.message.replies[-1]


async def test_auto_drain_keeps_send_now_button_until_reply_dispatch_finishes(mocker):
    service = BlockingService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=".", activity_mode="verbose"),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    dispatch_entered = asyncio.Event()
    release_dispatch = asyncio.Event()
    original_dispatch_reply = bridge._dispatch_reply
    dispatch_calls = 0

    async def delayed_dispatch_reply(*, chat_id: int, update: Update, reply: AgentReply) -> None:
        nonlocal dispatch_calls
        dispatch_calls += 1
        dispatch_entered.set()
        if dispatch_calls == 1:
            assert not any(e.get("message_id") == 1 for e in bot.edited_reply_markups)
        await release_dispatch.wait()
        await original_dispatch_reply(chat_id=chat_id, update=update, reply=reply)

    mocker.patch.object(bridge, "_dispatch_reply", side_effect=delayed_dispatch_reply)

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()
    await bridge.on_message(update_two, context)

    assert bot.sent_messages[0]["reply_markup"] is not None

    service.release()
    await dispatch_entered.wait()

    assert not any(e.get("message_id") == 1 for e in bot.edited_reply_markups)

    release_dispatch.set()
    await task_one

    assert any(edit.get("message_id") == 1 and edit.get("text") == BUSY_SENT_TEXT for edit in bot.edited_messages)


async def test_on_busy_callback_updates_notification_while_prompt_is_already_dequeued(mocker):
    service = BlockingService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=".", activity_mode="verbose"),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    dispatch_entered = asyncio.Event()
    release_dispatch = asyncio.Event()
    original_dispatch_reply = bridge._dispatch_reply

    async def delayed_dispatch_reply(*, chat_id: int, update: Update, reply: AgentReply) -> None:
        dispatch_entered.set()
        await release_dispatch.wait()
        await original_dispatch_reply(chat_id=chat_id, update=update, reply=reply)

    mocker.patch.object(bridge, "_dispatch_reply", side_effect=delayed_dispatch_reply)

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()
    await bridge.on_message(update_two, context)

    markup = cast(InlineKeyboardMarkup, bot.sent_messages[0]["reply_markup"])
    token = cast(str, markup.inline_keyboard[0][0].callback_data).split("|", 1)[1]
    notify_msg_id = 1

    service.release()
    await dispatch_entered.wait()

    assert TEST_CHAT_ID not in bridge._pending_prompts_by_chat
    assert bridge._dequeued_prompts_by_chat[TEST_CHAT_ID].token == token

    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_busy_callback(update_cb, make_context())

    assert callback.answers[-1] == "✅ Sent."
    assert not service.cancelled
    assert any(
        edit.get("message_id") == notify_msg_id and edit.get("text") == "✅ Sent." for edit in bot.edited_messages
    )

    release_dispatch.set()
    await task_one


async def test_on_busy_callback_keeps_prompt_queued_when_cancel_returns_false_during_dispatch(mocker):
    service = WindowAwareBlockingService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=".", activity_mode="verbose"),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    update_three = make_update(chat_id=TEST_CHAT_ID, text="third")
    context = make_context(application=SimpleNamespace(bot=bot))

    dispatch_entered = asyncio.Event()
    release_dispatch = asyncio.Event()
    original_dispatch_reply = bridge._dispatch_reply

    async def delayed_dispatch_reply(*, chat_id: int, update: Update, reply: AgentReply) -> None:
        dispatch_entered.set()
        await release_dispatch.wait()
        await original_dispatch_reply(chat_id=chat_id, update=update, reply=reply)

    mocker.patch.object(bridge, "_dispatch_reply", side_effect=delayed_dispatch_reply)

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await asyncio.wait_for(service.wait_for_prompt_count(1), timeout=1)
    await bridge.on_message(update_two, context)

    service._prompt_gate.set()
    await asyncio.wait_for(dispatch_entered.wait(), timeout=1)
    await bridge.on_message(update_three, context)

    markup = cast(InlineKeyboardMarkup, bot.sent_messages[-1]["reply_markup"])
    token = cast(str, markup.inline_keyboard[0][0].callback_data).split("|", 1)[1]
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_busy_callback(update_cb, make_context())

    assert callback.answers[-1] == BUSY_STILL_QUEUED_TEXT
    assert bridge._pending_prompts_by_chat[TEST_CHAT_ID].token == token
    queued_notification_id = 2
    assert all(edit.get("message_id") != queued_notification_id for edit in bot.edited_messages)

    release_dispatch.set()
    await asyncio.wait_for(service.wait_for_prompt_count(2), timeout=1)
    service._prompt_gate.set()
    await asyncio.wait_for(service.wait_for_prompt_count(3), timeout=1)
    service._prompt_gate.set()
    await asyncio.wait_for(task_one, timeout=1)


async def test_on_busy_callback_send_now_cancels_and_queued_runs():
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    await bridge.on_message(update_two, context)

    # Grab the token from the "Send now" button
    markup = cast(InlineKeyboardMarkup, bot.sent_messages[0]["reply_markup"])
    token = cast(str, markup.inline_keyboard[0][0].callback_data).split("|", 1)[1]

    # User presses "Send now"
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update_cb, make_context())

    assert callback.answers[-1] == "✅ Sent."
    assert any(edit.get("text") == "✅ Sent." for edit in bot.edited_messages)
    assert service.cancelled

    await task_one

    # Second message should have been processed
    assert update_two.message is not None
    assert "done:second" in update_two.message.replies[-1]


async def test_on_busy_callback_stale_token_is_rejected():
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    # Press button with an old/random token (simulates already-processed pending)
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|stale-token")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update_cb, make_context())

    assert callback.answers[-1] == "Already sent."
    assert callback.reply_markup_cleared

    service.release()
    await task_one


async def test_on_busy_callback_stale_after_auto_drain():
    """Queued message ran automatically; old button press returns 'Already sent.'"""
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()
    await bridge.on_message(update_two, context)

    markup = cast(InlineKeyboardMarkup, bot.sent_messages[0]["reply_markup"])
    token = cast(str, markup.inline_keyboard[0][0].callback_data).split("|", 1)[1]

    # Let first task finish naturally -> auto-drains second
    service.release()
    await task_one

    # Now the pending is gone; pressing button should get "Already sent."
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update_cb, make_context())

    assert callback.answers[-1] == "Already sent."


async def test_dispatch_reply_failure_clears_dequeued_prompt_state(mocker):
    service = BlockingService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=".", activity_mode="verbose"),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()
    await bridge.on_message(update_two, context)

    markup = cast(InlineKeyboardMarkup, bot.sent_messages[0]["reply_markup"])
    token = cast(str, markup.inline_keyboard[0][0].callback_data).split("|", 1)[1]

    async def failing_dispatch_reply(*, chat_id: int, update: Update, reply: AgentReply) -> None:
        del chat_id, update, reply
        raise MarkdownFailureError

    mocker.patch.object(bridge, "_dispatch_reply", side_effect=failing_dispatch_reply)

    service.release()
    with pytest.raises(MarkdownFailureError):
        await task_one

    assert TEST_CHAT_ID not in bridge._dequeued_prompts_by_chat

    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_busy_callback(update_cb, make_context())

    assert callback.answers[-1] == "Already sent."


async def test_on_busy_callback_no_query_is_noop():
    bridge = make_bridge()
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=None,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update, make_context())


async def test_on_busy_callback_invalid_data_format():
    bridge = make_bridge()
    callback = DummyCallbackQuery("busy")  # missing token part
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update, make_context())
    assert callback.answers[-1] == "Invalid action."


async def test_on_busy_callback_access_denied():
    bridge = make_bridge(allowed_ids={99})
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|some-token")
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update, make_context())
    assert callback.answers[-1] == "Access denied."


async def test_on_busy_callback_missing_chat():
    bridge = make_bridge()
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|some-token")
    callback.message = None
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=None,
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update, make_context())
    assert callback.answers[-1] == "Missing chat."


async def test_busy_queue_replaces_previous_pending_and_removes_old_button():
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    update_three = make_update(chat_id=TEST_CHAT_ID, text="third")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    await bridge.on_message(update_two, context)
    # First notify button sent (message_id=1)
    assert len(bot.sent_messages) == 1

    # Send third message while still busy - should replace second pending
    await bridge.on_message(update_three, context)

    # Old button (message_id=1) should be removed
    assert any(e.get("message_id") == 1 for e in bot.edited_reply_markups)
    # New button sent (message_id=2)
    assert len(bot.sent_messages) == EXPECTED_BUSY_NOTIFY_MESSAGES_AFTER_REPLACE

    service.release()
    await task_one

    # Only "third" should be processed (second was replaced)
    assert update_two.message is not None
    assert update_three.message is not None
    assert update_two.message.replies == []
    assert "done:third" in update_three.message.replies[-1]


async def test_on_busy_callback_edit_failure_is_handled_gracefully():
    """Edit of stale button may fail with TelegramError; that must not propagate."""

    class FailingEditOnStaleCallbackQuery(DummyCallbackQuery):
        async def edit_message_reply_markup(self, *, reply_markup: object | None = None) -> None:
            raise MarkdownFailureError

    bridge = make_bridge()
    callback = FailingEditOnStaleCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|stale")
    callback.message = SimpleNamespace(text="busy", chat=SimpleNamespace(id=TEST_CHAT_ID))
    update = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    # No pending for TEST_CHAT_ID -> "Already sent." + edit attempt (which fails gracefully)
    await bridge.on_busy_callback(update, make_context())
    assert callback.answers[-1] == "Already sent."


async def test_on_busy_callback_send_now_edit_failure_is_handled():
    """TelegramError on edit after 'Sent now' must not propagate."""
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()
    await bridge.on_message(update_two, context)

    markup = cast(InlineKeyboardMarkup, bot.sent_messages[0]["reply_markup"])
    token = cast(str, markup.inline_keyboard[0][0].callback_data).split("|", 1)[1]

    class FailingEditAfterAnswer(DummyCallbackQuery):
        async def edit_message_reply_markup(self, *, reply_markup: object | None = None) -> None:
            if self.answers:
                raise MarkdownFailureError
            await super().edit_message_reply_markup(reply_markup=reply_markup)

    callback = FailingEditAfterAnswer(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update_cb, make_context())

    assert callback.answers[-1] == "✅ Sent."
    service.release()
    await task_one


async def test_queue_busy_prompt_edit_old_button_failure_is_handled():
    """TelegramError when removing old pending button must not propagate."""
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))

    class FailingEditBot(DummyBot):
        async def edit_message_reply_markup(self, **kwargs: object) -> None:
            raise MarkdownFailureError

    bot = FailingEditBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    update_three = make_update(chat_id=TEST_CHAT_ID, text="third")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    await bridge.on_message(update_two, context)
    # Now queue a third message - old button removal fails gracefully
    await bridge.on_message(update_three, context)

    service.release()
    await task_one


async def test_queue_busy_prompt_send_message_failure_is_handled():
    """TelegramError when sending the notify message must not propagate."""
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))

    class FailingSendBot(DummyBot):
        async def send_message(self, **kwargs: object) -> SimpleNamespace:
            raise MarkdownFailureError

    bot = FailingSendBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    # send_message raises - must not propagate
    await bridge.on_message(update_two, context)

    service.release()
    await task_one


async def test_clear_busy_button_telegram_error_is_swallowed():
    """TelegramError when clearing the busy button must not propagate."""
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))

    class FailingEditBot(DummyBot):
        async def edit_message_reply_markup(self, **kwargs: object) -> None:
            raise MarkdownFailureError

    bot = FailingEditBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first")
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second")
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()

    # Queue second message - notify_msg_id is None (send failed), but we manually set one
    await bridge.on_message(update_two, context)
    pending = bridge._pending_prompts_by_chat.get(TEST_CHAT_ID)
    if pending is not None:
        pending.notify_msg_id = 42  # force a non-None id so _clear_busy_button tries to edit

    # Release - _clear_busy_button will try to edit and fail
    service.release()
    await task_one  # must complete without exception


async def test_clear_busy_button_falls_back_to_markup_when_text_edit_fails():
    service = BlockingService()
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, service))

    class TextFailingBot(DummyBot):
        async def edit_message_text(self, **kwargs: object) -> None:
            raise MarkdownFailureError

    bot = TextFailingBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))
    notify_msg_id = 42

    pending = _PendingPrompt(
        prompt_input=_PromptInput(chat_id=TEST_CHAT_ID, text="queued", images=(), files=()),
        update=cast(Update, make_update(chat_id=TEST_CHAT_ID, text="queued")),
        token="queued-token",
        notify_msg_id=notify_msg_id,
    )

    await bridge._clear_busy_button(pending)

    assert any(edit.get("message_id") == notify_msg_id for edit in bot.edited_reply_markups)


async def test_on_busy_callback_cancel_failure_answers_safely():
    """If cancel() raises, on_busy_callback answers 'Cancel failed.' and returns cleanly."""
    config = make_config(token="TOKEN", allowed_user_ids=[], workspace=".")
    bridge = TelegramBridge(config=config, agent_service=cast(AgentService, FailingCancelService()))
    bridge._app = cast(Application, SimpleNamespace(bot=DummyBot()))
    token = "test-token"
    notify_msg_id = 42
    dummy_update = make_update(chat_id=TEST_CHAT_ID, text="hi")
    prompt_input = _PromptInput(chat_id=TEST_CHAT_ID, text="hi", images=(), files=())
    bridge._pending_prompts_by_chat[TEST_CHAT_ID] = _PendingPrompt(
        prompt_input=prompt_input,
        update=cast(Update, dummy_update),
        token=token,
        notify_msg_id=notify_msg_id,
    )

    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )
    await bridge.on_busy_callback(update_cb, make_context())
    assert callback.answers[-1] == "Cancel failed."
    assert bridge._pending_prompts_by_chat[TEST_CHAT_ID].notify_msg_id == notify_msg_id


@pytest.mark.parametrize(("mode", "expect_direct_replies"), [("normal", True), ("compact", False), ("verbose", True)])
async def test_busy_queue_notification_is_visible_with_activity_in_all_modes(mode: str, expect_direct_replies: bool):
    service = BlockingActivityService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=".", activity_mode=cast(ActivityMode, mode)),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first", message_id=11)
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second", message_id=QUEUED_MESSAGE_ID)
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()
    await bridge.on_message(update_two, context)

    queue_payload = next(payload for payload in bot.sent_messages if "queued" in cast(str, payload["text"]).lower())
    assert queue_payload["chat_id"] == TEST_CHAT_ID
    assert queue_payload["reply_to_message_id"] == QUEUED_MESSAGE_ID
    assert queue_payload["allow_sending_without_reply"] is True
    markup = cast(InlineKeyboardMarkup, queue_payload["reply_markup"])
    assert markup is not None
    assert markup.inline_keyboard[0][0].text == "Send now"

    service.release()
    await task_one

    assert update_one.message is not None
    assert update_two.message is not None
    if expect_direct_replies:
        assert "done:first" in update_one.message.replies[-1]
        assert "done:second" in update_two.message.replies[-1]
    else:
        assert update_one.message.replies == []
        assert update_two.message.replies == []
        assert any("done:second" in cast(str, item["text"]) for item in bot.edited_messages)


async def test_update_busy_notification_query_fallback_clears_reply_markup():
    bridge = make_bridge()
    pending = _PendingPrompt(
        prompt_input=_PromptInput(chat_id=TEST_CHAT_ID, text="queued", images=(), files=()),
        update=cast(Update, make_update(chat_id=TEST_CHAT_ID, text="queued")),
        token="queued-token",
    )
    callback = DummyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|queued-token")

    await bridge._update_busy_notification(
        chat_id=TEST_CHAT_ID,
        pending=pending,
        query=cast(CallbackQuery, callback),
        text=BUSY_SENT_TEXT,
    )

    assert callback.edited_text == BUSY_SENT_TEXT
    assert callback.edited_kwargs["reply_markup"] is None


async def test_update_busy_notification_falls_back_to_dismiss_when_query_edit_fails():
    bridge = make_bridge()

    class FailingBusyCallbackQuery(DummyCallbackQuery):
        async def edit_message_text(self, text: str, **kwargs: object) -> None:
            del text, kwargs
            raise MarkdownFailureError

    pending = _PendingPrompt(
        prompt_input=_PromptInput(chat_id=TEST_CHAT_ID, text="queued", images=(), files=()),
        update=cast(Update, make_update(chat_id=TEST_CHAT_ID, text="queued")),
        token="queued-token",
    )
    callback = FailingBusyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|queued-token")

    await bridge._update_busy_notification(
        chat_id=TEST_CHAT_ID,
        pending=pending,
        query=cast(CallbackQuery, callback),
        text=BUSY_SENT_TEXT,
    )

    assert callback.reply_markup_cleared


@pytest.mark.parametrize("mode", ["normal", "compact", "verbose"])
async def test_on_busy_callback_updates_stored_notification_when_query_edit_fails(mode: str):
    service = BlockingActivityService()
    bridge = TelegramBridge(
        config=make_config(token="TOKEN", allowed_user_ids=[], workspace=".", activity_mode=cast(ActivityMode, mode)),
        agent_service=cast(AgentService, service),
    )
    bot = DummyBot()
    bridge._app = cast(Application, SimpleNamespace(bot=bot))

    update_one = make_update(chat_id=TEST_CHAT_ID, text="first", message_id=11)
    update_two = make_update(chat_id=TEST_CHAT_ID, text="second", message_id=QUEUED_MESSAGE_ID)
    context = make_context(application=SimpleNamespace(bot=bot))

    task_one = asyncio.create_task(bridge.on_message(update_one, context))
    await service._prompt_started.wait()
    await bridge.on_message(update_two, context)

    pending = bridge._pending_prompts_by_chat[TEST_CHAT_ID]
    token = pending.token
    notify_msg_id = pending.notify_msg_id
    assert notify_msg_id is not None

    class FailingBusyCallbackQuery(DummyCallbackQuery):
        async def edit_message_text(self, text: str, **kwargs: object) -> None:
            del text, kwargs
            raise MarkdownFailureError

        async def edit_message_reply_markup(self, *, reply_markup: object | None = None) -> None:
            del reply_markup
            raise MarkdownFailureError

    callback = FailingBusyCallbackQuery(f"{BUSY_CALLBACK_PREFIX}|{token}")
    update_cb = cast(
        Update,
        SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            effective_chat=SimpleNamespace(id=TEST_CHAT_ID),
            callback_query=callback,
            message=None,
        ),
    )

    await bridge.on_busy_callback(update_cb, make_context())

    assert callback.answers[-1] == "✅ Sent."
    assert any(
        edit.get("message_id") == notify_msg_id and edit.get("text") == "✅ Sent." for edit in bot.edited_messages
    )

    await task_one


async def test_log_text_preview_compacts_and_truncates():
    short = log_text_preview("  hola    mundo  ")
    assert short == "hola mundo"

    long_text = "x" * 400
    preview = log_text_preview(long_text)
    assert preview.endswith("...")
    assert len(preview) == LOG_TEXT_PREVIEW_MAX_CHARS + 3

    assert log_text_preview("   ") == "<empty>"


# ---------------------------------------------------------------------------
# Compact activity mode tests
# ---------------------------------------------------------------------------

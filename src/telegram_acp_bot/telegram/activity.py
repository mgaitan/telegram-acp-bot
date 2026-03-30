"""Activity-mode strategy handlers for the Telegram transport layer.

Each concrete `_ActivityModeHandler` implements a distinct UX strategy for how
agent activity events are surfaced to the user in Telegram.  The active strategy
for a given chat is selected by `TelegramBridge` at runtime.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING

from telegram import Bot, InlineKeyboardMarkup, Message, Update
from telegram.error import TelegramError

from telegram_acp_bot.acp_app.models import AgentActivityBlock, PermissionRequest
from telegram_acp_bot.telegram.models import _QueuedVerboseBlock, _VerboseActivityMessage

if TYPE_CHECKING:
    from telegram_acp_bot.telegram.bridge import TelegramBridge

VERBOSE_STREAM_TICK_SECONDS = 0.12


class _ActivityModeHandler:
    """Behavioral strategy for one Telegram activity mode."""

    def __init__(self, bridge: TelegramBridge) -> None:
        self._bridge = bridge

    async def on_permission_request(
        self,
        *,
        request: PermissionRequest,
        message: str,
        keyboard: InlineKeyboardMarkup,
    ) -> None:
        app = self._bridge._app
        if app is None:
            return
        await self._bridge._send_markdown_to_chat(
            bot=app.bot,
            chat_id=request.chat_id,
            text=message,
            reply_markup=keyboard,
        )

    async def on_activity_event(self, *, chat_id: int, block: AgentActivityBlock) -> None:
        raise NotImplementedError

    async def finalize_reply(self, *, chat_id: int, update: Update, text: str) -> bool:
        del chat_id, update, text
        return False

    async def handle_empty_reply(self, *, chat_id: int) -> None:
        del chat_id

    async def clear_chat_state(self, *, chat_id: int) -> None:
        del chat_id


class _NormalActivityModeHandler(_ActivityModeHandler):
    """Preserve the legacy per-event activity UX without streaming edits."""

    def __init__(self, bridge: TelegramBridge) -> None:
        super().__init__(bridge)
        self._seen_streams_by_chat: dict[int, set[str]] = {}

    async def on_activity_event(self, *, chat_id: int, block: AgentActivityBlock) -> None:
        app = self._bridge._app
        if app is None or block.kind == "reply":
            return
        if block.activity_id:
            if block.status == "in_progress":
                seen = self._seen_streams_by_chat.setdefault(chat_id, set())
                if block.activity_id in seen:
                    return
                seen.add(block.activity_id)
            else:
                self._seen_streams_by_chat.get(chat_id, set()).discard(block.activity_id)
        workspace = self._bridge._activity_workspace(chat_id=chat_id)
        text = self._bridge._format_activity_block(block, workspace=workspace)
        await self._bridge._send_markdown_to_chat(bot=app.bot, chat_id=chat_id, text=text)

    async def clear_chat_state(self, *, chat_id: int) -> None:
        self._seen_streams_by_chat.pop(chat_id, None)


class _CompactActivityModeHandler(_ActivityModeHandler):
    """Single in-place status message during prompt execution."""

    async def on_permission_request(
        self,
        *,
        request: PermissionRequest,
        message: str,
        keyboard: InlineKeyboardMarkup,
    ) -> None:
        app = self._bridge._app
        if app is None:
            return
        lock = self._bridge._compact_status_locks.setdefault(request.chat_id, asyncio.Lock())
        async with lock:
            status_msg_id = self._bridge._compact_status_msg_id.get(request.chat_id)
            self._bridge._cancel_compact_animation(request.chat_id)
            if status_msg_id is not None:
                edited = await self._bridge._edit_markdown_in_chat(
                    bot=app.bot,
                    chat_id=request.chat_id,
                    message_id=status_msg_id,
                    text=message,
                    reply_markup=keyboard,
                )
                if edited:
                    return
                with suppress(TelegramError):
                    await app.bot.delete_message(chat_id=request.chat_id, message_id=status_msg_id)

            sent = await self._bridge._send_markdown_to_chat(
                bot=app.bot,
                chat_id=request.chat_id,
                text=message,
                reply_markup=keyboard,
            )
            if sent is not None:
                self._bridge._compact_status_msg_id[request.chat_id] = sent.message_id

    async def on_activity_event(self, *, chat_id: int, block: AgentActivityBlock) -> None:
        app = self._bridge._app
        if app is None or block.kind == "reply":
            return
        label = self._bridge._activity_label(block)
        lock = self._bridge._compact_status_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            self._bridge._compact_status_label[chat_id] = label
            status_text = f"{label}."
            existing_msg_id = self._bridge._compact_status_msg_id.get(chat_id)
            if existing_msg_id is None:
                with suppress(TelegramError):
                    msg = await app.bot.send_message(chat_id=chat_id, text=status_text)
                    existing_msg_id = msg.message_id
                    self._bridge._compact_status_msg_id[chat_id] = existing_msg_id
            else:
                with suppress(TelegramError):
                    await app.bot.edit_message_text(chat_id=chat_id, message_id=existing_msg_id, text=status_text)
            if existing_msg_id is not None:
                self._bridge._ensure_compact_animation(chat_id=chat_id, message_id=existing_msg_id)

    async def finalize_reply(self, *, chat_id: int, update: Update, text: str) -> bool:
        if self._bridge._app is None or chat_id not in self._bridge._compact_status_msg_id:
            return False
        await self._bridge._finalize_compact_reply(chat_id=chat_id, update=update, text=text)
        return True

    async def handle_empty_reply(self, *, chat_id: int) -> None:
        await self._bridge._clear_compact_status(chat_id)

    async def clear_chat_state(self, *, chat_id: int) -> None:
        await self._bridge._clear_compact_status(chat_id)


class _VerboseActivityModeHandler(_ActivityModeHandler):
    """Track each live activity stream independently and edit it in place."""

    def __init__(self, bridge: TelegramBridge) -> None:
        super().__init__(bridge)
        self._locks: dict[int, asyncio.Lock] = {}
        self._messages_by_chat: dict[int, dict[str, _VerboseActivityMessage]] = {}
        self._pending_by_chat: dict[int, dict[str, _QueuedVerboseBlock]] = {}
        self._flush_tasks_by_chat: dict[int, asyncio.Task[None]] = {}

    async def on_activity_event(self, *, chat_id: int, block: AgentActivityBlock) -> None:
        app = self._bridge._app
        if app is None:
            return
        slot_key = self._slot_key(block)
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            active = self._messages_by_chat.get(chat_id, {}).get(slot_key)
            if active is not None and block.text and not block.text.startswith(active.source_text):
                self._clear_message(chat_id=chat_id, slot_key=slot_key)
                self._clear_pending(chat_id=chat_id, slot_key=slot_key)
                active = None
            if block.status == "in_progress":
                if active is None:
                    await self._apply_block_locked(chat_id=chat_id, slot_key=slot_key, block=block)
                    return
                self._pending_by_chat.setdefault(chat_id, {})[slot_key] = _QueuedVerboseBlock(
                    chat_id=chat_id,
                    slot_key=slot_key,
                    block=block,
                )
                self._ensure_flush_task(chat_id=chat_id)
                return
            self._clear_pending(chat_id=chat_id, slot_key=slot_key)
            await self._apply_block_locked(chat_id=chat_id, slot_key=slot_key, block=block)

    async def finalize_reply(self, *, chat_id: int, update: Update, text: str) -> bool:
        del update
        app = self._bridge._app
        if app is None:
            return False
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            self._clear_pending(chat_id=chat_id, slot_key="activity:reply")
            active = self._messages_by_chat.get(chat_id, {}).get("activity:reply")
            if active is None:
                return False
            self._clear_message(chat_id=chat_id, slot_key="activity:reply")
            if not text:
                return True
            chunks = self._bridge._render_markdown_chunks(text)
            if not chunks:
                return True
            first_text, first_entities = chunks[0]
            edited = await self._bridge._edit_rendered_chunk_in_chat(
                bot=app.bot,
                chat_id=chat_id,
                message_id=active.message_id,
                text=first_text,
                entities=first_entities,
            )
            if edited:
                await self._bridge._send_rendered_chunks_to_chat(bot=app.bot, chat_id=chat_id, chunks=chunks[1:])
                return True
            with suppress(TelegramError):
                await app.bot.delete_message(chat_id=chat_id, message_id=active.message_id)
            return False

    async def clear_chat_state(self, *, chat_id: int) -> None:
        self._cancel_flush_task(chat_id)
        self._pending_by_chat.pop(chat_id, None)
        self._messages_by_chat.pop(chat_id, None)

    def _slot_key(self, block: AgentActivityBlock) -> str:
        return f"activity:{self._bridge._activity_id(block)}"

    def _store_message(self, *, chat_id: int, slot_key: str, message: _VerboseActivityMessage) -> None:
        self._messages_by_chat.setdefault(chat_id, {})[slot_key] = message

    def _clear_message(self, *, chat_id: int, slot_key: str) -> None:
        messages = self._messages_by_chat.get(chat_id)
        if messages is None:
            return
        messages.pop(slot_key, None)
        if not messages:
            self._messages_by_chat.pop(chat_id, None)

    def _clear_pending(self, *, chat_id: int, slot_key: str) -> None:
        pending = self._pending_by_chat.get(chat_id)
        if pending is None:
            return
        pending.pop(slot_key, None)
        if not pending:
            self._pending_by_chat.pop(chat_id, None)
            self._cancel_flush_task(chat_id)

    def _cancel_flush_task(self, chat_id: int) -> None:
        task = self._flush_tasks_by_chat.pop(chat_id, None)
        if task is not None:
            task.cancel()

    def _ensure_flush_task(self, *, chat_id: int) -> None:
        task = self._flush_tasks_by_chat.get(chat_id)
        if task is not None and not task.done():
            return
        self._flush_tasks_by_chat[chat_id] = asyncio.create_task(self._run_flush_loop(chat_id=chat_id))

    async def _run_flush_loop(self, *, chat_id: int) -> None:
        try:
            while True:
                await asyncio.sleep(VERBOSE_STREAM_TICK_SECONDS)
                lock = self._locks.setdefault(chat_id, asyncio.Lock())
                async with lock:
                    pending = self._pending_by_chat.get(chat_id)
                    if not pending:
                        self._pending_by_chat.pop(chat_id, None)
                        return
                    queued_blocks = list(pending.values())
                    self._pending_by_chat.pop(chat_id, None)
                    for queued in queued_blocks:
                        await self._apply_block_locked(
                            chat_id=queued.chat_id,
                            slot_key=queued.slot_key,
                            block=queued.block,
                        )
        finally:
            current = asyncio.current_task()
            task = self._flush_tasks_by_chat.get(chat_id)
            if task is current:
                self._flush_tasks_by_chat.pop(chat_id, None)

    async def _edit_in_progress_preview(
        self,
        *,
        bot: Bot,
        chat_id: int,
        message_id: int,
        text: str,
    ) -> bool:
        preview = self._bridge._render_markdown_preview_chunk(text)
        if preview is None:
            return False
        preview_text, preview_entities = preview
        return await self._bridge._edit_rendered_chunk_in_chat(
            bot=bot,
            chat_id=chat_id,
            message_id=message_id,
            text=preview_text,
            entities=preview_entities,
        )

    async def _send_in_progress_preview(self, *, bot: Bot, chat_id: int, text: str) -> Message | None:
        preview = self._bridge._render_markdown_preview_chunk(text)
        if preview is None:
            return None
        return await self._bridge._send_rendered_chunks_to_chat(bot=bot, chat_id=chat_id, chunks=[preview])

    async def _apply_block_locked(self, *, chat_id: int, slot_key: str, block: AgentActivityBlock) -> None:
        app = self._bridge._app
        if app is None:
            return
        text = (
            block.text
            if block.kind == "reply"
            else self._bridge._format_activity_block(
                block,
                workspace=self._bridge._activity_workspace(chat_id=chat_id),
            )
        )
        active = self._messages_by_chat.get(chat_id, {}).get(slot_key)
        if active is not None:
            if block.status == "in_progress":
                edited = await self._edit_in_progress_preview(
                    bot=app.bot,
                    chat_id=chat_id,
                    message_id=active.message_id,
                    text=text,
                )
            else:
                edited = await self._bridge._edit_markdown_in_chat(
                    bot=app.bot,
                    chat_id=chat_id,
                    message_id=active.message_id,
                    text=text,
                )
            if edited:
                if block.status == "in_progress":
                    self._store_message(
                        chat_id=chat_id,
                        slot_key=slot_key,
                        message=_VerboseActivityMessage(
                            activity_id=active.activity_id,
                            kind=block.kind,
                            title=block.title,
                            message_id=active.message_id,
                            source_text=block.text,
                        ),
                    )
                else:
                    self._clear_message(chat_id=chat_id, slot_key=slot_key)
                return
            self._clear_message(chat_id=chat_id, slot_key=slot_key)

        if block.status == "in_progress":
            sent = await self._send_in_progress_preview(
                bot=app.bot,
                chat_id=chat_id,
                text=text,
            )
        else:
            sent = await self._bridge._send_markdown_to_chat(bot=app.bot, chat_id=chat_id, text=text)
        if sent is None:
            return
        if block.status == "in_progress":
            self._store_message(
                chat_id=chat_id,
                slot_key=slot_key,
                message=_VerboseActivityMessage(
                    activity_id=self._bridge._activity_id(block),
                    kind=block.kind,
                    title=block.title,
                    message_id=sent.message_id,
                    source_text=block.text,
                ),
            )
            return
        self._clear_message(chat_id=chat_id, slot_key=slot_key)

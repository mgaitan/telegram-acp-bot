from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse

from telegram import (
    Bot,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    Message,
    MessageEntity,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters
from telegramify_markdown import MessageEntity as MarkdownMessageEntity
from telegramify_markdown import convert, split_entities

from telegram_acp_bot.acp_app.models import (
    AgentActivityBlock,
    AgentOutputLimitExceededError,
    AgentReply,
    FilePayload,
    ImagePayload,
    PermissionDecisionAction,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
    PromptFile,
    PromptImage,
    ResumableSession,
)

PERMISSION_CALLBACK_PREFIX = "perm"
RESUME_CALLBACK_PREFIX = "resume"
PERMISSION_CALLBACK_PARTS = 3
RESUME_CALLBACK_PARTS = 2
RESUME_KEYBOARD_MAX_ROWS = 10
RESTART_EXIT_CODE = 75
TELEGRAM_MAX_UTF16_MESSAGE_LENGTH = 4096
logger = logging.getLogger(__name__)
KIND_LABELS = {
    "think": "💡 Thinking",
    "execute": "⚙️ Tool call",
    "read": "📖 Reading",
    "search": "🔎 Searching",
    "edit": "✏️ Editing",
    "write": "✍️ Writing",
}


@dataclass(slots=True, frozen=True)
class BotConfig:
    """Runtime settings for Telegram transport."""

    token: str
    allowed_user_ids: set[int]
    default_workspace: Path


@dataclass(slots=True, frozen=True)
class _PromptInput:
    chat_id: int
    text: str
    images: tuple[PromptImage, ...]
    files: tuple[PromptFile, ...]


class ChatRequiredError(ValueError):
    """Raised when a Telegram update does not include a chat object."""


class AgentService(Protocol):
    """Service interface expected by Telegram handlers."""

    async def new_session(self, *, chat_id: int, workspace: Path) -> str: ...

    async def load_session(self, *, chat_id: int, session_id: str, workspace: Path) -> str: ...

    async def list_resumable_sessions(
        self,
        *,
        chat_id: int,
        workspace: Path | None = None,
    ) -> tuple[ResumableSession, ...] | None: ...

    async def prompt(
        self,
        *,
        chat_id: int,
        text: str,
        images: tuple[PromptImage, ...] = (),
        files: tuple[PromptFile, ...] = (),
    ) -> AgentReply | None: ...

    def get_workspace(self, *, chat_id: int) -> Path | None: ...

    def supports_session_loading(self, *, chat_id: int) -> bool | None: ...

    async def cancel(self, *, chat_id: int) -> bool: ...

    async def stop(self, *, chat_id: int) -> bool: ...

    async def clear(self, *, chat_id: int) -> bool: ...

    def get_permission_policy(self, *, chat_id: int) -> PermissionPolicy | None: ...

    async def set_session_permission_mode(self, *, chat_id: int, mode: PermissionMode) -> bool: ...

    async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool) -> bool: ...

    def set_permission_request_handler(
        self,
        handler: Callable[[PermissionRequest], Awaitable[None]] | None,
    ) -> None: ...

    def set_activity_event_handler(
        self,
        handler: Callable[[int, AgentActivityBlock], Awaitable[None]] | None,
    ) -> None: ...

    async def respond_permission_request(
        self,
        *,
        chat_id: int,
        request_id: str,
        action: PermissionDecisionAction,
    ) -> bool: ...


class TelegramBridge:
    """Telegram command and message handlers for the MVP bot."""

    def __init__(self, config: BotConfig, agent_service: AgentService) -> None:
        self._config = config
        self._agent_service = agent_service
        self._app: Application | None = None
        self._restart_requested = False
        self._implicit_start_locks_by_chat: dict[int, asyncio.Lock] = {}
        self._pending_resume_choices_by_chat: dict[int, tuple[ResumableSession, ...]] = {}
        if hasattr(self._agent_service, "set_permission_request_handler"):
            self._agent_service.set_permission_request_handler(self.on_permission_request)
        if hasattr(self._agent_service, "set_activity_event_handler"):
            self._agent_service.set_activity_event_handler(self.on_activity_event)

    def install(self, app: Application) -> None:
        self._app = app
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.help))
        app.add_handler(CommandHandler("new", self.new_session))
        app.add_handler(CommandHandler("resume", self.resume_session))
        app.add_handler(CommandHandler("session", self.session))
        app.add_handler(CommandHandler("cancel", self.cancel))
        app.add_handler(CommandHandler("stop", self.stop))
        app.add_handler(CommandHandler("clear", self.clear))
        app.add_handler(CommandHandler("restart", self.restart))
        app.add_handler(CallbackQueryHandler(self.on_permission_callback, pattern=r"^perm\|"))
        app.add_handler(CallbackQueryHandler(self.on_resume_callback, pattern=r"^resume\|"))
        app.add_handler(
            MessageHandler((filters.TEXT | filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND, self.on_message)
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._require_access(update):
            return
        await self._reply(
            update,
            "Send a message to start in the default workspace, or use /new [workspace] or /resume [workspace].",
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._require_access(update):
            return
        await self._reply(
            update,
            "Commands: /new [workspace], /resume [workspace], /session, /cancel, /stop, /clear, /restart, /help",
        )

    async def new_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_access(update):
            return

        chat_id = self._chat_id(update)
        workspace = self._workspace_from_args(self._context_args(context))
        workspace_was_missing = not workspace.exists()
        started = await self._start_session(
            update=update,
            chat_id=chat_id,
            workspace=workspace,
            invalid_workspace_label="Invalid workspace",
        )
        if started is None:
            return
        session_id, active_workspace = started
        response = f"Session started: `{session_id}` in `{active_workspace}`"
        if workspace_was_missing:
            response = f"{response}\nCreated workspace: `{active_workspace}`"
        await self._reply(update, response)

    async def resume_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_access(update):
            return

        chat_id = self._chat_id(update)
        args = self._context_args(context)
        workspace = self._workspace_from_args(args) if args else None
        try:
            candidates = await self._agent_service.list_resumable_sessions(chat_id=chat_id, workspace=workspace)
        except Exception as exc:  # noqa: BLE001
            await self._reply(update, f"Failed to list resumable sessions: {exc}")
            return
        if candidates is None:
            await self._reply(update, "Agent does not support ACP `session/list`.")
            return
        if not candidates:
            await self._reply(update, "No resumable sessions found.")
            return

        if self._app is None:
            candidate = candidates[0]
            session_id = await self._agent_service.load_session(
                chat_id=chat_id,
                session_id=candidate.session_id,
                workspace=candidate.workspace,
            )
            await self._reply(update, f"Session resumed: `{session_id}` in `{candidate.workspace}`")
            return

        self._pending_resume_choices_by_chat[chat_id] = candidates
        keyboard = self._resume_keyboard(candidates=candidates)
        message = "Pick a session to resume:"
        if workspace is not None:
            message = f"Pick a session to resume in `{workspace}`:"
        await self._app.bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN,
        )

    async def session(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._require_access(update):
            return

        workspace = self._agent_service.get_workspace(chat_id=self._chat_id(update))
        if workspace is None:
            await self._reply(update, "No active session. Use /new first.")
            return

        await self._reply(update, f"Active session workspace: `{workspace}`")

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._require_access(update):
            return

        cancelled = await self._agent_service.cancel(chat_id=self._chat_id(update))
        if cancelled:
            await self._reply(update, "Cancelled current operation.")
            return
        await self._reply(update, "No active session. Use /new first.")

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._require_access(update):
            return

        stopped = await self._agent_service.stop(chat_id=self._chat_id(update))
        if stopped:
            await self._reply(update, "Stopped current session.")
            return
        await self._reply(update, "No active session. Use /new first.")

    async def clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._require_access(update):
            return

        cleared = await self._agent_service.clear(chat_id=self._chat_id(update))
        if cleared:
            await self._reply(update, "Cleared current session.")
            return
        await self._reply(update, "No active session. Use /new first.")

    async def restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._require_access(update):
            return

        if self._app is None:
            await self._reply(update, "Restart is unavailable: application is not running.")
            return
        await self._reply(update, "Restart requested. Re-launching process...")
        self._restart_requested = True
        self._app.stop_running()

    async def on_permission_request(self, request: PermissionRequest) -> None:
        if self._app is None:
            return

        keyboard = self._permission_keyboard(request)
        title = TelegramBridge._format_permission_tool_title(request.tool_title)
        message_parts = ["*⚠️ Permission required for:*"]
        if title:
            message_parts.append(TelegramBridge._render_activity_part(title))
        message = "\n\n".join(message_parts)
        await TelegramBridge._send_markdown_to_chat(
            bot=self._app.bot,
            chat_id=request.chat_id,
            text=message,
            reply_markup=keyboard,
        )

    async def on_activity_event(self, chat_id: int, block: AgentActivityBlock) -> None:
        app = self._app
        if app is None:
            return

        workspace = self._activity_workspace(chat_id=chat_id)
        text = self._format_activity_block(block, workspace=workspace)
        await TelegramBridge._send_markdown_to_chat(bot=app.bot, chat_id=chat_id, text=text)

    async def on_permission_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        query = update.callback_query
        if query is None:
            return
        try:
            if not await self._require_access(update):
                await query.answer("Access denied.")
                return

            data = query.data or ""
            logger.info("Permission callback received: %s", data)
            parts = data.split("|", maxsplit=2)
            if len(parts) != PERMISSION_CALLBACK_PARTS:
                await query.answer("Invalid action.")
                return
            _, request_id, raw_action = parts
            if raw_action not in {"once", "always", "deny"}:
                await query.answer("Invalid action.")
                return

            chat = update.effective_chat
            chat_id = chat.id if chat is not None else None
            query_message = getattr(query, "message", None)
            if chat_id is None and query_message is not None:
                chat_id = query_message.chat.id
            if chat_id is None:
                await query.answer("Missing chat.")
                return

            accepted = await self._agent_service.respond_permission_request(
                chat_id=chat_id,
                request_id=request_id,
                action=cast(PermissionDecisionAction, raw_action),
            )
            if not accepted:
                logger.warning("Permission callback rejected: request_id=%s chat_id=%s", request_id, chat_id)
                await query.answer("Request expired.")
                return

            labels = {"once": "Approved this time.", "always": "Approved for this session.", "deny": "Denied."}
            logger.info("Permission callback accepted: request_id=%s action=%s", request_id, raw_action)
            await query.answer(labels[raw_action])
            try:
                query_message = getattr(query, "message", None)
                original = getattr(query_message, "text", None) if query_message is not None else None
                base_text = original or "Permission request"
                await query.edit_message_text(f"{base_text}\nDecision: {labels[raw_action]}")
            except TelegramError:
                await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            logger.exception("Unhandled error while processing permission callback")
            await query.answer("Permission action failed.")

    async def on_resume_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        query = update.callback_query
        if query is None:
            return
        if not await self._require_access(update):
            await query.answer("Access denied.")
            return
        selection = await self._resolve_resume_selection(update=update, query=query)
        if selection is None:
            return
        chat_id, candidate = selection

        try:
            session_id = await self._agent_service.load_session(
                chat_id=chat_id,
                session_id=candidate.session_id,
                workspace=candidate.workspace,
            )
        except Exception as exc:
            logger.exception(
                "Resume failed for chat_id=%s session_id=%s workspace=%s",
                chat_id,
                candidate.session_id,
                candidate.workspace,
            )
            await query.answer("Failed to resume.")
            await self._reply(update, f"Failed to resume session `{candidate.session_id}`: {exc}")
            return

        await query.answer("Session resumed.")
        selected_message = f"Resumed session: {session_id}\nWorkspace: {candidate.workspace}\nTitle: {candidate.title}"
        try:
            await query.edit_message_text(selected_message)
        except TelegramError:
            await query.edit_message_reply_markup(reply_markup=None)
        self._pending_resume_choices_by_chat.pop(chat_id, None)
        await self._reply(update, f"Session resumed: `{session_id}` in `{candidate.workspace}`")

    async def _resolve_resume_selection(
        self,
        *,
        update: Update,
        query: CallbackQuery,
    ) -> tuple[int, ResumableSession] | None:
        index = self._resume_index(query.data or "")
        if index is None:
            await query.answer("Invalid selection.")
            return None

        chat_id = self._chat_id_from_update_or_query(update=update, query=query)
        if chat_id is None:
            await query.answer("Missing chat.")
            return None

        candidates = self._pending_resume_choices_by_chat.get(chat_id)
        if candidates is None:
            await query.answer("Selection expired.")
            return None
        if index < 0 or index >= len(candidates):
            await query.answer("Invalid selection.")
            return None

        return chat_id, candidates[index]

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_access(update):
            return

        prompt_input = await self._prompt_input(update=update, context=context)
        if prompt_input is None:
            return

        if not await self._ensure_session_for_chat(update=update, chat_id=prompt_input.chat_id):
            return

        reply = await self._request_reply(
            update=update,
            context=context,
            prompt_input=prompt_input,
        )
        if reply is None:
            return

        if self._app is None:
            workspace = self._activity_workspace(chat_id=prompt_input.chat_id)
            for block in reply.activity_blocks:
                await self._reply_activity_block(update, block, workspace=workspace)
        if reply.text.strip():
            await self._reply_agent(update, reply.text)
        await self._send_attachments(update, reply)

    async def _start_implicit_session(self, *, update: Update, chat_id: int) -> bool:
        workspace = self._config.default_workspace
        started = await self._start_session(
            update=update,
            chat_id=chat_id,
            workspace=workspace,
            invalid_workspace_label="Invalid default workspace",
        )
        return started is not None

    async def _prompt_input(self, *, update: Update, context: ContextTypes.DEFAULT_TYPE) -> _PromptInput | None:
        message = update.message
        if message is None:
            return None
        text = message.text or message.caption or ""
        images = await self._extract_prompt_images(message=message, context=context)
        files = await self._extract_prompt_files(message=message, context=context)
        if not text and not images and not files:
            return None
        return _PromptInput(chat_id=self._chat_id(update), text=text, images=images, files=files)

    async def _ensure_session_for_chat(self, *, update: Update, chat_id: int) -> bool:
        if self._agent_service.get_workspace(chat_id=chat_id) is not None:
            self._drop_implicit_start_lock(chat_id=chat_id)
            return True
        lock = self._implicit_start_lock(chat_id)
        async with lock:
            if self._agent_service.get_workspace(chat_id=chat_id) is not None:
                self._drop_implicit_start_lock(chat_id=chat_id, expected_lock=lock)
                return True
            started = await self._start_implicit_session(update=update, chat_id=chat_id)
            if self._agent_service.get_workspace(chat_id=chat_id) is not None:
                self._drop_implicit_start_lock(chat_id=chat_id, expected_lock=lock)
            return started

    async def _start_session(
        self,
        *,
        update: Update,
        chat_id: int,
        workspace: Path,
        invalid_workspace_label: str,
    ) -> tuple[str, Path] | None:
        try:
            session_id = await self._agent_service.new_session(chat_id=chat_id, workspace=workspace)
        except ValueError as exc:
            message = str(exc) or str(workspace)
            await self._reply(update, f"{invalid_workspace_label}: {message}")
            return None
        except RuntimeError:
            await self._reply(update, "Failed to start session: agent process did not expose stdio pipes.")
            return None
        except Exception as exc:  # noqa: BLE001
            await self._reply(update, f"Failed to start session: {exc}")
            return None
        active_workspace = self._agent_service.get_workspace(chat_id=chat_id) or workspace
        return session_id, active_workspace

    def _implicit_start_lock(self, chat_id: int) -> asyncio.Lock:
        lock = self._implicit_start_locks_by_chat.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._implicit_start_locks_by_chat[chat_id] = lock
        return lock

    def _drop_implicit_start_lock(self, *, chat_id: int, expected_lock: asyncio.Lock | None = None) -> None:
        current_lock = self._implicit_start_locks_by_chat.get(chat_id)
        if current_lock is None:
            return
        if expected_lock is not None and current_lock is not expected_lock:
            return
        self._implicit_start_locks_by_chat.pop(chat_id, None)

    async def _request_reply(
        self,
        *,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        prompt_input: _PromptInput,
    ) -> AgentReply | None:
        await context.bot.send_chat_action(chat_id=prompt_input.chat_id, action=ChatAction.TYPING)
        try:
            reply = await self._agent_service.prompt(
                chat_id=prompt_input.chat_id,
                text=prompt_input.text,
                images=prompt_input.images,
                files=prompt_input.files,
            )
        except AgentOutputLimitExceededError:
            await self._reply(
                update,
                "Agent output exceeded ACP stdio limit. Restart with a higher `--acp-stdio-limit` "
                "(or `ACP_STDIO_LIMIT`).",
            )
            return None
        if reply is not None:
            return reply
        await self._reply(update, "No active session. Send a message again or use /new [workspace].")
        return None

    async def _extract_prompt_images(
        self,
        *,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> tuple[PromptImage, ...]:
        images: list[PromptImage] = []
        if message.photo:
            photo = message.photo[-1]
            tg_file = await context.bot.get_file(photo.file_id)
            raw = bytes(await tg_file.download_as_bytearray())
            images.append(PromptImage(data_base64=base64.b64encode(raw).decode("ascii"), mime_type="image/jpeg"))

        document = message.document
        if document is not None and document.mime_type and document.mime_type.startswith("image/"):
            tg_file = await context.bot.get_file(document.file_id)
            raw = bytes(await tg_file.download_as_bytearray())
            images.append(PromptImage(data_base64=base64.b64encode(raw).decode("ascii"), mime_type=document.mime_type))

        return tuple(images)

    async def _extract_prompt_files(
        self,
        *,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> tuple[PromptFile, ...]:
        document = message.document
        if document is None:
            return ()
        if document.mime_type and document.mime_type.startswith("image/"):
            return ()

        tg_file = await context.bot.get_file(document.file_id)
        raw = bytes(await tg_file.download_as_bytearray())
        name = document.file_name or "attachment.bin"
        try:
            text_content = raw.decode("utf-8")
            return (PromptFile(name=name, mime_type=document.mime_type, text_content=text_content),)
        except UnicodeDecodeError:
            return (
                PromptFile(
                    name=name,
                    mime_type=document.mime_type,
                    data_base64=base64.b64encode(raw).decode("ascii"),
                ),
            )

    async def _send_attachments(self, update: Update, reply: AgentReply) -> None:
        for image in reply.images:
            await self._send_image(update, image)
        for file_payload in reply.files:
            await self._send_file(update, file_payload)

    @staticmethod
    def _permission_keyboard(request: PermissionRequest) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = []
        buttons: list[InlineKeyboardButton] = []
        for action in request.available_actions:
            label = {"always": "Always", "once": "This time", "deny": "Deny"}[action]
            callback_data = f"{PERMISSION_CALLBACK_PREFIX}|{request.request_id}|{action}"
            buttons.append(InlineKeyboardButton(text=label, callback_data=callback_data))
        rows.append(buttons)
        return InlineKeyboardMarkup(rows)

    @staticmethod
    def _resume_keyboard(*, candidates: tuple[ResumableSession, ...]) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = []
        for index, candidate in enumerate(candidates[:RESUME_KEYBOARD_MAX_ROWS]):
            title = candidate.title.strip() or candidate.session_id
            label = title[:48]
            callback_data = f"{RESUME_CALLBACK_PREFIX}|{index}"
            rows.append([InlineKeyboardButton(text=label, callback_data=callback_data)])
        return InlineKeyboardMarkup(rows)

    async def _require_access(self, update: Update) -> bool:
        allowed = self._config.allowed_user_ids
        if not allowed:
            return True

        user_id = update.effective_user.id if update.effective_user else None
        if user_id in allowed:
            return True

        await self._reply(update, "Access denied for this bot.")
        return False

    @staticmethod
    def _chat_id(update: Update) -> int:
        chat = update.effective_chat
        if chat is None:
            raise ChatRequiredError
        return chat.id

    @staticmethod
    def _chat_id_from_update_or_query(*, update: Update, query: object) -> int | None:
        chat = update.effective_chat
        chat_id = chat.id if chat is not None else None
        if chat_id is not None:
            return chat_id
        query_message = getattr(query, "message", None)
        if query_message is None:
            return None
        return query_message.chat.id

    @staticmethod
    def _resume_index(data: str) -> int | None:
        parts = data.split("|", maxsplit=1)
        if len(parts) != RESUME_CALLBACK_PARTS:
            return None
        _, raw_index = parts
        if not raw_index.isdigit():
            return None
        return int(raw_index)

    def _workspace_from_args(self, args: list[str]) -> Path:
        if not args:
            return self._config.default_workspace

        candidate = Path(args[0]).expanduser()
        if candidate.is_absolute():
            return candidate
        return self._config.default_workspace / candidate

    @staticmethod
    def _context_args(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
        args = context.args
        if not args:
            return []
        return list(args)

    @staticmethod
    async def _reply(update: Update, text: str) -> None:
        if update.message is None:
            return
        await TelegramBridge._reply_markdown_message(update.message, text=text)

    @staticmethod
    async def _reply_agent(update: Update, text: str) -> None:
        if update.message is None:
            return
        await TelegramBridge._reply_markdown_message(update.message, text=text)

    @staticmethod
    def _to_telegram_entity(entity: MarkdownMessageEntity) -> MessageEntity:
        return MessageEntity(
            type=entity.type,
            offset=entity.offset,
            length=entity.length,
            url=entity.url,
            language=entity.language,
            custom_emoji_id=entity.custom_emoji_id,
        )

    @staticmethod
    async def _reply_activity_block(
        update: Update, block: AgentActivityBlock, *, workspace: Path | None = None
    ) -> None:
        if update.message is None:
            return

        text = TelegramBridge._format_activity_block(block, workspace=workspace)
        await TelegramBridge._reply_markdown_message(update.message, text=text)

    @staticmethod
    def _format_activity_block(block: AgentActivityBlock, *, workspace: Path | None = None) -> str:
        label = KIND_LABELS.get(block.kind, "⚙️ Tool call")
        text_parts = [f"*{label}*"]
        normalized_title = TelegramBridge._normalize_activity_title(block, workspace=workspace)
        normalized_text = TelegramBridge._normalize_activity_text(block, workspace=workspace)
        if normalized_title and normalized_text and normalized_title == normalized_text:
            normalized_title = ""
        if normalized_title:
            text_parts.append(TelegramBridge._render_activity_part(normalized_title))
        if normalized_text:
            text_parts.append(TelegramBridge._render_activity_part(normalized_text))
        if block.status == "failed":
            text_parts.append("_Failed_")
        return "\n\n".join(text_parts)

    @staticmethod
    def _normalize_activity_title(block: AgentActivityBlock, *, workspace: Path | None = None) -> str:
        title = block.title.strip()
        if block.kind == "think":
            return ""
        if block.kind == "execute" and title.startswith("Run "):
            command = title[4:].strip()
            if command:
                return f"Run\n```sh\n{TelegramBridge._normalize_execute_commands(command)}\n```"
        path_prefix = TelegramBridge._path_prefix_for_kind(block.kind)
        if path_prefix and title.startswith(path_prefix):
            return TelegramBridge._format_read_path(title[len(path_prefix) :], workspace=workspace)
        return title

    @staticmethod
    def _normalize_activity_text(block: AgentActivityBlock, *, workspace: Path | None = None) -> str:
        text = block.text.strip()
        path_prefix = TelegramBridge._path_prefix_for_kind(block.kind)
        if path_prefix and text.startswith(path_prefix) and "\n" not in text:
            return TelegramBridge._format_read_path(text[len(path_prefix) :], workspace=workspace)
        return text

    @staticmethod
    def _escape_markdown_preserving_code(text: str) -> str:
        escaped: list[str] = []
        in_code = False
        for char in text:
            if char == "`":
                in_code = not in_code
                escaped.append(char)
                continue
            if in_code:
                escaped.append(char)
                continue
            if char in {"\\", "_", "*", "["}:
                escaped.append(f"\\{char}")
                continue
            escaped.append(char)
        return "".join(escaped)

    @staticmethod
    def _render_activity_part(text: str) -> str:
        if "```" in text:
            return text
        return TelegramBridge._escape_markdown_preserving_code(text)

    @staticmethod
    def _format_read_path(raw_path: str, *, workspace: Path | None) -> str:
        raw = raw_path.strip()
        if not raw:
            return ""

        # Prefer explicit file URIs when present in tool titles/text.
        if "file://" in raw:
            parsed = urlparse(raw[raw.index("file://") :])
            if parsed.scheme == "file" and parsed.path:
                safe_uri_path = parsed.path.rstrip(")").replace("`", "\\`")
                return f"`{safe_uri_path}`"

        path = Path(raw).expanduser()
        if path.is_absolute():
            safe_abs_path = str(path.resolve(strict=False)).replace("`", "\\`")
            return f"`{safe_abs_path}`"
        base_workspace = workspace or Path.cwd()
        workspace_path = (base_workspace / path).resolve(strict=False)
        safe_workspace_path = str(workspace_path).replace("`", "\\`")
        return f"`{safe_workspace_path}`"

    @staticmethod
    def _path_prefix_for_kind(kind: str) -> str | None:
        if kind == "read":
            return "Read "
        if kind == "edit":
            return "Edit "
        return None

    @staticmethod
    def _format_permission_tool_title(tool_title: str) -> str:
        title = tool_title.strip()
        if not title:
            return ""
        if title.startswith("Run "):
            return TelegramBridge._normalize_activity_title(
                AgentActivityBlock(kind="execute", title=title, status="in_progress")
            )
        return title

    @staticmethod
    def _normalize_execute_commands(command: str) -> str:
        if ", Run " not in command:
            return command
        commands = [part.strip() for part in command.split(", Run ")]
        if commands and all(commands):
            return "\n".join(commands)
        return command

    @staticmethod
    async def _reply_markdown_message(message: Message, *, text: str) -> None:
        try:
            rendered_text, rendered_entities = convert(text)
            chunks = split_entities(rendered_text, rendered_entities, max_utf16_len=TELEGRAM_MAX_UTF16_MESSAGE_LENGTH)
            for chunk_text, chunk_entities in chunks:
                entities = [TelegramBridge._to_telegram_entity(entity) for entity in chunk_entities]
                try:
                    await message.reply_text(chunk_text, entities=entities)
                except TelegramError:
                    await message.reply_text(chunk_text)
        except (RuntimeError, ValueError, TypeError):
            await message.reply_text(text)

    @staticmethod
    async def _send_markdown_to_chat(
        *,
        bot: Bot,
        chat_id: int,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        try:
            rendered_text, rendered_entities = convert(text)
            chunks = split_entities(rendered_text, rendered_entities, max_utf16_len=TELEGRAM_MAX_UTF16_MESSAGE_LENGTH)
            for index, (chunk_text, chunk_entities) in enumerate(chunks):
                kwargs: dict[str, object] = {"chat_id": chat_id, "text": chunk_text}
                if chunk_entities:
                    kwargs["entities"] = [TelegramBridge._to_telegram_entity(entity) for entity in chunk_entities]
                if reply_markup is not None and index == 0:
                    kwargs["reply_markup"] = reply_markup
                try:
                    await bot.send_message(**kwargs)
                except TelegramError:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=chunk_text,
                        reply_markup=reply_markup if index == 0 else None,
                    )
        except (RuntimeError, ValueError, TypeError):
            await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)

    def _activity_workspace(self, *, chat_id: int) -> Path:
        return self._agent_service.get_workspace(chat_id=chat_id) or self._config.default_workspace

    @staticmethod
    async def _send_image(update: Update, payload: ImagePayload) -> None:
        if update.message is None:
            return

        raw = base64.b64decode(payload.data_base64)
        extension = "jpg" if payload.mime_type == "image/jpeg" else "bin"
        input_file = InputFile(BytesIO(raw), filename=f"agent-image.{extension}")
        await update.message.reply_photo(photo=input_file)

    @staticmethod
    async def _send_file(update: Update, payload: FilePayload) -> None:
        if update.message is None:
            return

        if payload.text_content is not None:
            raw = payload.text_content.encode("utf-8")
        elif payload.data_base64 is not None:
            raw = base64.b64decode(payload.data_base64)
        else:
            raw = b""

        input_file = InputFile(BytesIO(raw), filename=payload.name)
        await update.message.reply_document(document=input_file)


def build_application(config: BotConfig, bridge: TelegramBridge) -> Application:
    # Permission prompts are awaited inside message handlers, so callback queries
    # must be processed concurrently to avoid deadlocking the update loop.
    app = Application.builder().token(config.token).concurrent_updates(True).build()
    bridge.install(app)
    return app


def run_polling(config: BotConfig, bridge: TelegramBridge) -> int:
    app = build_application(config, bridge)
    app.run_polling(allowed_updates=Update.ALL_TYPES)
    if bridge._restart_requested:
        return RESTART_EXIT_CODE
    return 0


def make_config(*, token: str, allowed_user_ids: list[int], workspace: str) -> BotConfig:
    return BotConfig(
        token=token,
        allowed_user_ids=set(allowed_user_ids),
        default_workspace=Path(workspace).expanduser(),
    )

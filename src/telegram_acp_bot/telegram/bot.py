from __future__ import annotations

import base64
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol, cast

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from telegram_acp_bot.acp_app.models import (
    AgentActivityBlock,
    AgentReply,
    FilePayload,
    ImagePayload,
    PermissionDecisionAction,
    PermissionMode,
    PermissionPolicy,
    PermissionRequest,
    PromptFile,
    PromptImage,
)

PERMISSION_CALLBACK_PREFIX = "perm"
PERMISSION_CALLBACK_PARTS = 3
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


class ChatRequiredError(ValueError):
    """Raised when a Telegram update does not include a chat object."""


class AgentService(Protocol):
    """Service interface expected by Telegram handlers."""

    async def new_session(self, *, chat_id: int, workspace: Path) -> str: ...

    async def prompt(
        self,
        *,
        chat_id: int,
        text: str,
        images: tuple[PromptImage, ...] = (),
        files: tuple[PromptFile, ...] = (),
    ) -> AgentReply | None: ...

    def get_workspace(self, *, chat_id: int) -> Path | None: ...

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
        if hasattr(self._agent_service, "set_permission_request_handler"):
            self._agent_service.set_permission_request_handler(self.on_permission_request)
        if hasattr(self._agent_service, "set_activity_event_handler"):
            self._agent_service.set_activity_event_handler(self.on_activity_event)

    def install(self, app: Application) -> None:
        self._app = app
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.help))
        app.add_handler(CommandHandler("new", self.new_session))
        app.add_handler(CommandHandler("session", self.session))
        app.add_handler(CommandHandler("cancel", self.cancel))
        app.add_handler(CommandHandler("stop", self.stop))
        app.add_handler(CommandHandler("clear", self.clear))
        app.add_handler(CallbackQueryHandler(self.on_permission_callback, pattern=r"^perm\|"))
        app.add_handler(
            MessageHandler((filters.TEXT | filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND, self.on_message)
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._require_access(update):
            return
        await self._reply(update, "Use /new [workspace] to start a session, then send plain text prompts.")

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not await self._require_access(update):
            return
        await self._reply(
            update,
            "Commands: /new [workspace], /session, /cancel, /stop, /clear, /help",
        )

    async def new_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_access(update):
            return

        chat_id = self._chat_id(update)
        workspace = self._workspace_from_args(self._context_args(context))
        workspace_was_missing = not workspace.exists()
        try:
            session_id = await self._agent_service.new_session(chat_id=chat_id, workspace=workspace)
        except ValueError as exc:
            message = str(exc) or str(workspace)
            await self._reply(update, f"Invalid workspace: {message}")
            return
        except RuntimeError:
            await self._reply(update, "Failed to start session: agent process did not expose stdio pipes.")
            return
        except Exception as exc:  # noqa: BLE001
            await self._reply(update, f"Failed to start session: {exc}")
            return
        active_workspace = self._agent_service.get_workspace(chat_id=chat_id) or workspace
        response = f"Session started: `{session_id}` in `{active_workspace}`"
        if workspace_was_missing:
            response = f"{response}\nCreated workspace: `{active_workspace}`"
        await self._reply(update, response)

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

    async def on_permission_request(self, request: PermissionRequest) -> None:
        if self._app is None:
            return

        keyboard = self._permission_keyboard(request)
        message = f"Permission required for:\n{request.tool_title}"
        await self._app.bot.send_message(
            chat_id=request.chat_id,
            text=message,
            reply_markup=keyboard,
        )

    async def on_activity_event(self, chat_id: int, block: AgentActivityBlock) -> None:
        app = self._app
        if app is None:
            return

        text = self._format_activity_block(block)
        try:
            await app.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
        except TelegramError:
            await app.bot.send_message(chat_id=chat_id, text=text)

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

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_access(update):
            return

        message = update.message
        if message is None:
            return

        text = message.text or message.caption or ""
        images = await self._extract_prompt_images(message=message, context=context)
        files = await self._extract_prompt_files(message=message, context=context)
        if not text and not images and not files:
            return

        chat_id = self._chat_id(update)
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        try:
            reply = await self._agent_service.prompt(chat_id=chat_id, text=text, images=images, files=files)
        except ValueError as exc:
            message = str(exc)
            if "chunk is longer than limit" in message:
                await self._reply(
                    update,
                    "Agent output exceeded ACP stdio limit. Restart with a higher `--acp-stdio-limit` "
                    "(or `ACP_STDIO_LIMIT`).",
                )
                return
            raise
        if reply is None:
            await self._reply(update, "No active session. Use /new first.")
            return

        if self._app is None:
            for block in reply.activity_blocks:
                await self._reply_activity_block(update, block)
        if reply.text.strip():
            await self._reply_agent(update, reply.text)
        await self._send_attachments(update, reply)

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
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except TelegramError:
            await update.message.reply_text(text)

    @staticmethod
    async def _reply_agent(update: Update, text: str) -> None:
        if update.message is None:
            return

        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except TelegramError:
            await update.message.reply_text(text)

    @staticmethod
    async def _reply_activity_block(update: Update, block: AgentActivityBlock) -> None:
        if update.message is None:
            return

        text = TelegramBridge._format_activity_block(block)
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except TelegramError:
            await update.message.reply_text(text)

    @staticmethod
    def _format_activity_block(block: AgentActivityBlock) -> str:
        label = KIND_LABELS.get(block.kind, "⚙️ Tool call")
        text_parts = [f"*{label}*"]
        normalized_title = TelegramBridge._normalize_activity_title(block)
        normalized_text = TelegramBridge._normalize_activity_text(block)
        if normalized_title and normalized_text and normalized_title == normalized_text:
            normalized_title = ""
        if normalized_title:
            text_parts.append(TelegramBridge._escape_markdown_preserving_code(normalized_title))
        if normalized_text:
            text_parts.append(TelegramBridge._escape_markdown_preserving_code(normalized_text))
        if block.status == "failed":
            text_parts.append("_Failed_")
        return "\n\n".join(text_parts)

    @staticmethod
    def _normalize_activity_title(block: AgentActivityBlock) -> str:
        title = block.title.strip()
        if block.kind == "think":
            return ""
        if block.kind == "execute" and title.startswith("Run "):
            command = title[4:].strip()
            if command:
                safe_command = command.replace("`", "\\`")
                return f"Run `{safe_command}`"
        if block.kind == "read" and title.startswith("Read "):
            return title[5:]
        return title

    @staticmethod
    def _normalize_activity_text(block: AgentActivityBlock) -> str:
        text = block.text.strip()
        if block.kind == "read" and text.startswith("Read ") and "\n" not in text:
            return text[5:]
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
    return 0


def make_config(*, token: str, allowed_user_ids: list[int], workspace: str) -> BotConfig:
    return BotConfig(
        token=token,
        allowed_user_ids=set(allowed_user_ids),
        default_workspace=Path(workspace).expanduser(),
    )

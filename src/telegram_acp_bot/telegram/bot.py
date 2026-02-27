from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from telegram_acp_bot.acp_app.models import AgentReply, PermissionMode, PermissionPolicy

PERMISSION_SET_ARG_COUNT = 2


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

    async def prompt(self, *, chat_id: int, text: str) -> AgentReply | None: ...

    def get_workspace(self, *, chat_id: int) -> Path | None: ...

    async def cancel(self, *, chat_id: int) -> bool: ...

    async def stop(self, *, chat_id: int) -> bool: ...

    async def clear(self, *, chat_id: int) -> bool: ...

    def get_permission_policy(self, *, chat_id: int) -> PermissionPolicy | None: ...

    async def set_session_permission_mode(self, *, chat_id: int, mode: PermissionMode) -> bool: ...

    async def set_next_prompt_auto_approve(self, *, chat_id: int, enabled: bool) -> bool: ...


class TelegramBridge:
    """Telegram command and message handlers for the MVP bot."""

    def __init__(self, config: BotConfig, agent_service: AgentService) -> None:
        self._config = config
        self._agent_service = agent_service

    def install(self, app: Application) -> None:
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.help))
        app.add_handler(CommandHandler("new", self.new_session))
        app.add_handler(CommandHandler("session", self.session))
        app.add_handler(CommandHandler("cancel", self.cancel))
        app.add_handler(CommandHandler("stop", self.stop))
        app.add_handler(CommandHandler("clear", self.clear))
        app.add_handler(CommandHandler("perm", self.permissions))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_text))

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
            "Commands: /new [workspace], /session, /cancel, /stop, /clear, /perm, /help",
        )

    async def new_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_access(update):
            return

        chat_id = self._chat_id(update)
        workspace = self._workspace_from_args(context.args)
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
        await self._reply(update, f"Session started: `{session_id}` in `{active_workspace}`")

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

    async def permissions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:  # noqa: PLR0911
        if not await self._require_access(update):
            return

        chat_id = self._chat_id(update)
        if not context.args:
            policy = self._agent_service.get_permission_policy(chat_id=chat_id)
            if policy is None:
                await self._reply(update, "No active session. Use /new first.")
                return
            await self._reply(
                update,
                f"Permissions: session={policy.session_mode}, next_prompt={policy.next_prompt_auto_approve}",
            )
            return

        subcommand = context.args[0].lower()
        if subcommand == "session" and len(context.args) >= PERMISSION_SET_ARG_COUNT:
            mode = context.args[1].lower()
            if mode not in {"approve", "deny"}:
                await self._reply(update, "Usage: /perm session approve|deny")
                return
            changed = await self._agent_service.set_session_permission_mode(chat_id=chat_id, mode=mode)
            if changed:
                await self._reply(update, f"Updated session permission mode to {mode}.")
                return
            await self._reply(update, "No active session. Use /new first.")
            return

        if subcommand == "next" and len(context.args) >= PERMISSION_SET_ARG_COUNT:
            raw_value = context.args[1].lower()
            if raw_value not in {"on", "off"}:
                await self._reply(update, "Usage: /perm next on|off")
                return
            changed = await self._agent_service.set_next_prompt_auto_approve(
                chat_id=chat_id,
                enabled=raw_value == "on",
            )
            if changed:
                await self._reply(update, f"Updated next prompt auto-approve to {raw_value}.")
                return
            await self._reply(update, "No active session. Use /new first.")
            return

        await self._reply(update, "Usage: /perm | /perm session approve|deny | /perm next on|off")

    async def on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_access(update):
            return

        message = update.message
        if message is None or not message.text:
            return

        chat_id = self._chat_id(update)
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        reply = await self._agent_service.prompt(chat_id=chat_id, text=message.text)
        if reply is None:
            await self._reply(update, "No active session. Use /new first.")
            return

        await self._reply_agent(update, reply.text)

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
        return Path(args[0]).expanduser() if args else self._config.default_workspace

    @staticmethod
    async def _reply(update: Update, text: str) -> None:
        if update.message is not None:
            await update.message.reply_text(text)

    @staticmethod
    async def _reply_agent(update: Update, text: str) -> None:
        if update.message is None:
            return

        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except TelegramError:
            await update.message.reply_text(text)


def build_application(config: BotConfig, bridge: TelegramBridge) -> Application:
    app = Application.builder().token(config.token).build()
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

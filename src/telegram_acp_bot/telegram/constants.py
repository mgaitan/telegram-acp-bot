"""Constants for the Telegram transport layer."""

from __future__ import annotations

PERMISSION_CALLBACK_PREFIX = "perm"
RESUME_CALLBACK_PREFIX = "resume"
BUSY_CALLBACK_PREFIX = "busy"
SCHEDULED_CALLBACK_PREFIX = "scheduled"
PERMISSION_CALLBACK_PARTS = 3
RESUME_CALLBACK_PARTS = 2
BUSY_CALLBACK_PARTS = 2
SCHEDULED_CALLBACK_PARTS = 3
BUSY_SENT_TEXT = "✅ Sent."
BUSY_QUEUED_TEXT = "⏳ Agent is busy. Your message is queued."
BUSY_STILL_QUEUED_TEXT = "Still queued."
MAX_RESUME_ARGS = 1
MAX_RESTART_ARGS = 2
RESUME_KEYBOARD_MAX_ROWS = 10
SCHEDULED_KEYBOARD_MAX_ROWS = 10
RESTART_EXIT_CODE = 75
TELEGRAM_MAX_UTF16_MESSAGE_LENGTH = 4096
BOT_COMMANDS: tuple[tuple[str, str], ...] = (
    ("start", "Start or resume in the default workspace"),
    ("new", "Create a new agent session [workspace]"),
    ("resume", "Resume a previous session [N|workspace]"),
    ("scheduled", "List and cancel scheduled follow-ups"),
    ("mode", "Show or set activity mode [normal|compact|verbose]"),
    ("session", "Show the active session workspace"),
    ("cancel", "Cancel the current agent operation"),
    ("stop", "Stop the current session"),
    ("clear", "Clear the current session"),
    ("restart", "Restart the bot [N [workspace]]"),
    ("help", "Show available commands"),
)
KIND_LABELS = {
    "think": "💡 Thinking",
    "execute": "⚙️ Running",
    "read": "📖 Reading",
    "edit": "✏️ Editing",
    "write": "✍️ Writing",
}
SEARCH_LABEL_WEB = "🌐 Searching web"
SEARCH_LABEL_NEUTRAL = "🔎 Querying"
REPLY_LABEL = "✍️ Replying"
ACTIVITY_MODE_CHOICES: tuple[str, ...] = ("normal", "compact", "verbose")
ACTIVITY_MODE_HELP = "normal, compact, or verbose"
UNSUPPORTED_MESSAGE_TEXT = (
    "Unsupported message type. I can process text, photos, documents, and voice/audio messages."
)

# Design and architecture notes

This page explains stable design decisions behind the Telegram ACP bot.

## Product principles

- Direct interaction: plain chat messages are prompts.
- Fast feedback: runtime activity is shown incrementally.
- Safe by default: risky operations require explicit permission by default.
- Recoverable sessions: chats can stop, resume, and reset sessions.
- Operational clarity: users can inspect active session and workspace.

## Session and conversation model

- One Telegram chat maps to one active ACP session.
- The first prompt starts a session implicitly in the default workspace.
- `/new` is used for explicit workspace/session switching.
- When a prompt is already running, new prompts are queued and can be forced with **Send now**.

See also {doc}`how-to` for user-facing command and behavior details.

## Runtime visibility model

- Tool/runtime activity is rendered as compact Telegram messages.
- Permission requests are shown as independent messages with inline actions.
- Final assistant text is emitted as a separate message after activity updates.
- File and image outputs are delivered as Telegram attachments when possible.

## Architecture split

The codebase is organized into layers with clear responsibilities:

1. `telegram_acp_bot.telegram`
   Telegram transport handlers and rendering concerns.
2. `telegram_acp_bot.acp_app`
   ACP session lifecycle, process/transport handling, and event mapping.
3. `telegram_acp_bot.core`
   Shared state helpers and domain-level utilities.

This separation keeps Telegram-specific behavior isolated from ACP integration logic.

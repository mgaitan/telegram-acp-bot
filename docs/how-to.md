# How To Use The Bot From Telegram

This guide shows the main command flow for day-to-day usage.

## 1. Start a new session

Create a session in the default workspace:

```text
/new
```

Create a session in a specific workspace:

```text
/new /home/user/project
```

The bot will reply with the ACP `session_id` and workspace path.

## 2. Ask for work

After `/new`, send normal text messages.

Example:

```text
Please review the latest commit and suggest a fix.
```

You can also attach images/documents; the bot forwards them to ACP prompt input.

## 3. Resume an existing ACP session

List resumable sessions and pick one from buttons:

```text
/resume
```

List resumable sessions for a specific workspace:

```text
/resume /home/user/project
```

When you tap a button, the bot calls ACP `session/load` and switches the active conversation for that chat.

If your agent does not support ACP `session/list`, `/resume` will report it.

## 4. Inspect current session

```text
/session
```

Shows the active session workspace.

## 5. Cancel, stop, and clear

Cancel current in-flight operation:

```text
/cancel
```

Stop current session process:

```text
/stop
```

Clear current session:

```text
/clear
```

## 6. Restart bot process (dev workflow)

```text
/restart
```

The bot exits polling and re-execs the process (or uses `ACP_RESTART_COMMAND` if configured).

## Notes

- Session scope is per Telegram chat.
- `/new` replaces the active session for that chat.
- `/resume` is ACP-native (`session/list` + `session/load`) and depends on agent capabilities.

## Agent-to-Telegram attachments (explicit ACP extension)

This bot advertises a client extension method named `_telegram/send_attachment` via ACP `initialize` `_meta`.

Supported params:

- `sessionId` (required, string)
- exactly one of:
  - `uri` (string)
  - `dataBase64` (string)
- `mimeType` (optional, string)
- `name` (optional, string, defaults to `attachment.bin`)

Behavior is explicit: if the agent does not call `_telegram/send_attachment`, no extension-based attachment is sent.

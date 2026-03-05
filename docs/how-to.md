# How To Use The Bot From Telegram

This guide shows the main command flow for day-to-day usage.

## 1. Start working (implicit session)

Send a normal prompt. If the chat has no active session, the bot creates one automatically in the default workspace.

Example:

```text
Please review the latest commit and suggest a fix.
```

You can also attach images/documents; the bot forwards them to ACP prompt input.

## 2. Start or switch sessions with `/new`

Create a session in the default workspace:

```text
/new
```

Create a session in a specific workspace:

```text
/new /home/user/project
```

The bot will reply with the ACP `session_id` and workspace path.

Use `/new` when you want a different session/workspace than the current one.

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
- The first prompt creates a session implicitly when none is active (including after `/restart`).
- `/new` replaces the active session for that chat and is intended for explicitly switching to another workspace/session.
- `/resume` is ACP-native (`session/list` + `session/load`) and depends on agent capabilities.

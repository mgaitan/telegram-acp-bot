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

Resume by list index (0-based, where `0` is the most recent):

```text
/resume 0
```

List resumable sessions for a specific workspace:

```text
/resume /home/user/project
```

This resumes the most recent session in that workspace.

When you tap a button, the bot calls ACP `session/load` and switches the active conversation for that chat.

`/resume` accepts either an index or a workspace path (not both in the same command).

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

## 6. Busy-state handling

When you send a message while the agent is already processing a previous prompt, the bot:

- Queues your message automatically.
- Shows a temporary **Send now** inline button.

### Send now

Pressing **Send now** immediately cancels the current in-flight operation and processes your queued message.

If the current task finishes naturally before you press the button, your queued message runs automatically and the button is removed (pressing it after that shows "Already sent." with no side effects).

## 7. Restart bot process (dev workflow)

```text
/restart
```

The bot exits polling and re-execs the process (or uses `ACP_RESTART_COMMAND` if configured).

Restart by resuming a specific saved session (without restarting the process):

```text
/restart 0
```

You can combine index and workspace filter:

```text
/restart 2 /home/user/project
```

## Notes

- Session scope is per Telegram chat.
- The first prompt creates a session implicitly when none is active (including after `/restart`).
- `/new` replaces the active session for that chat and is intended for explicitly switching to another workspace/session.
- `/resume` is ACP-native (`session/list` + `session/load`) and depends on agent capabilities.

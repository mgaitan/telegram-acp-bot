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

## 6. Switch activity mode

Show the current mode:

```text
/mode
```

Set the mode explicitly:

```text
/mode normal
/mode compact
/mode verbose
```

The available modes are:

- `normal`: separate activity messages, no streaming edits.
- `compact`: one in-progress status message that becomes the final answer.
- `verbose`: in-place append-only streaming for reply text and active tool output.

(how-to-busy-queue)=
## 7. Busy-state handling

When you send a message while the agent is already processing a previous prompt, the bot:

- Queues your message automatically.
- Replies to that queued message with the busy notice.
- Shows a temporary **Send now** inline button.

### Send now

Pressing **Send now** immediately cancels the current in-flight operation and processes your queued message.
The queued notice is updated to `✅ Sent.` so chat state matches what happened.

If the current task finishes naturally before you press the button, your queued message runs automatically and the button is removed (pressing it after that shows "Already sent." with no side effects).

## 8. Schedule a prompt directly

You can schedule a deferred agent prompt without going through the agent, using `/schedule`. This is useful when the model rejects your request due to quota limits or when you want to queue work for later without starting an agent session first.

```text
/schedule <time> <prompt text>
```

Supported time formats:

- `30s` — 30 seconds from now
- `10m` — 10 minutes from now
- `2h` — 2 hours from now
- `1d` — 1 day from now
- Natural-language dates such as `tomorrow 9am` or `mañana 9am`
- ISO timestamp with timezone, e.g. `2026-04-11T10:00:00+00:00`

Example:

```text
/schedule tomorrow 9am Check for new review comments on the open PR
Summarize what changed
Flag anything blocking merge

/schedule mañana 9am Generate the weekly summary report
Include pending reviews
Mention overdue follow-ups

/schedule 2026-04-12T09:00:00+00:00 Generate the weekly summary report
Include pending reviews
Mention overdue follow-ups
```

The bot replies with the scheduled execution time. When the time arrives, the stored prompt is sent to the agent and the reply is posted back to the chat as a reply to your `/schedule` message.

The prompt may also continue on following lines. Everything after the time spec is stored verbatim, so multiline prompts keep their line breaks.

Natural-language parsing uses English and Spanish by default. You can customize the accepted languages with {term}`ACP_SCHEDULE_LANGUAGES` or `telegram.schedule_languages` in the config file.

Single-line prompts still work too:

```text
/schedule 2h Generate the weekly summary report
```

If no active session exists when the scheduled time arrives, the bot will report that the task could not be run automatically.

## 9. Inspect and cancel scheduled follow-ups

When the agent schedules a deferred follow-up, you can inspect the pending work directly from Telegram:

```text
/scheduled
```

The bot replies with a compact summary of the scheduled tasks for the current chat. Pending tasks are shown with inline **Cancel** buttons, so you do not have to copy or type task ids manually.

A typical interaction looks like this:

```text
/scheduled

Scheduled tasks for this chat:

Pending:
1. prompt_agent | 2026-03-30 22:10:00 UTC
  Check the PR again and report any new comments.
2. notify | 2026-03-30 22:30:00 UTC
  Remind me to restart the bot.
```

From there you can tap **Cancel 1**, **Cancel 2**, and so on, or use **Cancel all pending** if you want to clear the queue for that chat.

For this first version, cancellation is intentionally limited to tasks that are still `pending`. Tasks that are already `running` are visible for inspection, but they are not turned into interruptible jobs by this command.

## 10. Restart bot process (dev workflow)

```text
/restart
```

With an active session, the bot replies with:

```text
Restart requested. Re-launching process...
Session restarted: <session_id> in <workspace>
```

Then it exits polling and re-execs the process (or uses `ACP_RESTART_COMMAND` if configured).

If there is no active session, it replies:

```text
No active session. Use /new first.
```

Resume a specific saved session in-process with the restart command:

```text
/restart 0
```

You can combine index and workspace filter:

```text
/restart 2 /home/user/project
```

That path replies with:

```text
Session resumed: <session_id> in <workspace>
```

## Notes

- Session scope is per Telegram chat.
- The first prompt creates a session implicitly when none is active (including after `/restart`).
- `/new` replaces the active session for that chat and is intended for explicitly switching to another workspace/session.
- `/resume` is ACP-native (`session/list` + `session/load`) and depends on agent capabilities.
- `/mode` is stored per Telegram chat, so different chats can use different activity modes.

## Logging and traceability

Every application log line includes contextual identifiers:

- `chat_id`: Telegram chat id.
- `session_id`: ACP session id (when a session is active).
- `prompt_cycle_id`: per-prompt cycle/task id generated for each incoming user prompt.

This lets you trace:

- all activity from one ACP session (`session_id`),
- all logs tied to a single prompt execution (including queued prompts) via `prompt_cycle_id`.

Use {term}`ACP_LOG_FORMAT`=`json` to emit structured logs for log aggregators.

By default, the bot also emits compact text previews for prompts and replies:

- `Prompt received: ...`
- `Reply sent: ...`

The human-oriented `text` log format uses `rich` output in the terminal to highlight the logger, chat/session/cycle ids, and the message preview.

To keep logs focused on auditability, verbose transport/framework logs (for example `httpx` Telegram API request lines) are downgraded to warning level by default.

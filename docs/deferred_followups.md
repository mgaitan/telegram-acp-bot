# Deferred Follow-ups

Deferred follow-ups let the agent promise something now and do it later, without keeping the original ACP prompt alive. The feature is deliberately small, but it already covers the most useful cases: “check this again in 10 minutes”, “remind me at 9 PM”, or “send me a short follow-up in 30 seconds”.

The important user-facing idea is that scheduling should feel conversational, not infrastructural. The user asks for something later. The agent answers normally in the thread. When the scheduled time arrives, the follow-up appears as a reply to that scheduling confirmation.

For example, the interaction is meant to feel like this:

```text
User: check this PR again in 10 minutes
Bot: Okay. I’ll check it again in 10 minutes.

... later ...

Bot (replying to that confirmation): There are 3 new comments on the PR.
```

The bot does not need to expose internal status messages such as “Scheduled” or “Completed”. The useful message is the one the agent already sends to the user. That message becomes the anchor for the later follow-up.

## What Exists Today

The current implementation supports one-shot tasks only. Tasks are persisted in the SQLite database configured by {term}`ACP_SCHEDULED_TASKS_DB`, and an in-process scheduler loop claims due work inside the bot runtime. There are two execution modes.

`notify` is the simpler mode. It sends a plain reminder or follow-up text later.

`prompt_agent` is more interesting. It re-enters the agent flow later with a stored prompt and sends the resulting answer back to Telegram as a reply to the scheduling confirmation.

Recurring schedules and automatic session rehydration are intentionally out of scope for this first version.

The bot does, however, provide a small management surface for already-scheduled work. The `/scheduled` command shows the pending and running follow-ups for the current chat, and pending items can be cancelled from inline buttons without typing task ids by hand.

Users can also schedule prompts directly via the `/schedule` slash command, without going through the agent at all. This is especially useful when the model rejects a request due to quota limits or when immediate execution is not possible.

(deferred-followups-ux)=
## How The UX Works

When the agent calls `schedule_task`, the MCP tool persists the task and returns a summary. The tool itself does not send a Telegram message. Instead, the agent uses that tool result to answer the user in normal language. Once that answer reaches Telegram, the bot records that reply as the task’s anchor message.

That small design choice keeps the chat cleaner. The visible message is the conversational confirmation from the agent, not a second technical artifact from the transport layer.

When the scheduled time arrives, the scheduler claims the task and runs it. A `notify` task sends its text as a reply to the anchor. A `prompt_agent` task sends the stored prompt through the normal agent pipeline and replies to the same anchor with the final result.

```{mermaid}
:align: center
sequenceDiagram
    participant U as Telegram user
    participant B as Bot runtime
    participant A as ACP agent
    participant M as MCP schedule_task
    participant S as Scheduler

    U->>B: "Check this PR again in 10 minutes"
    B->>A: ACP prompt
    A->>M: schedule_task(...)
    M-->>A: { ok: true, summary: ... }
    A-->>B: "Okay. I'll check again in 10 minutes."
    B-->>U: scheduling confirmation
    B->>B: bind task to that message id
    S->>B: task becomes due
    B->>A: stored prompt, later
    A-->>B: final reply
    B-->>U: reply to scheduling confirmation
```

## Relative And Absolute Time Inputs

For short delays, relative time is the preferred path. Requests like “in 30 seconds” or “in 10 minutes” should become `delay_seconds`, `delay_minutes`, or `delay_hours` on the MCP side. That avoids forcing the agent to calculate a wall-clock timestamp for something that is naturally relative.

Absolute scheduling is still supported through `run_at`, which must be an ISO timestamp with an explicit timezone offset. That is appropriate for requests such as “at 21:00 UTC” or “tomorrow at 09:00-03:00”.

## Session Behavior

`prompt_agent` only runs if the chat has an active ACP session when the scheduled time arrives. The implementation intentionally reuses the chat’s current live session rather than trying to resurrect one invisibly. That makes failures easier to understand and avoids hidden magic.

If there is no active session, the task fails visibly by replying to the anchor message with an explanation such as `Could not run automatically: no active session.` The same principle applies to delivery errors: the user should see a plain explanation in the thread rather than having the task disappear silently.

## Scheduling Directly Via Slash Command

In addition to the agent-driven path, users can schedule a `prompt_agent` task directly from the chat without involving the agent at all. This is useful when the model rejects a prompt due to quota limits or when immediate execution is not possible.

```text
/schedule <time> <prompt text>
```

Supported time formats are: `30s`, `10m`, `2h`, `1d`, natural-language dates such as `tomorrow 9am` or `mañana 9am`, or an ISO timestamp with an explicit timezone offset such as `2026-04-11T10:00:00+00:00`.

The prompt text can span multiple lines. Everything after the time spec is preserved as-is and stored in the scheduled `prompt_agent` task.

Example interaction:

```text
User: /schedule tomorrow 9am Check for new review comments on the open PR
Summarize what changed
Flag anything blocking merge
Bot: Scheduled for 2026-04-11 10:30 UTC. Use /scheduled to view or cancel.

... 30 minutes later ...

Bot (replying to that command): There are 3 new comments on the PR.
```

Single-line prompts remain valid as a shorter form:

```text
User: /schedule 2h Generate the weekly summary report
Bot: Scheduled for 2026-04-11 12:00 UTC. Use /scheduled to view or cancel.
```

Natural-language parsing uses English and Spanish by default, and can be restricted or extended with {term}`ACP_SCHEDULE_LANGUAGES` or `telegram.schedule_languages`.

The `/schedule` command sets the command message itself as the anchor, so the scheduled reply will appear directly in that thread. Session behavior and failure reporting follow the same rules as agent-driven scheduling.

## Inspecting And Cancelling Scheduled Work

The scheduling flow is conversational, but it is still useful to inspect what is queued. For that reason, the bot exposes a Telegram-side command:

```text
/scheduled
```

This command is intentionally modest. It is not a rich dashboard. It simply shows the pending and running tasks for the current chat in a readable list, with inline **Cancel** buttons for the tasks that are still pending.

The list uses small consecutive numbers such as `1.` and `2.` so the interface feels more like `/resume` and less like a database browser. Internally the bot still uses the real task ids for callback handling, but the user only sees the short numbered list.

That choice keeps the product surface small. The agent still owns scheduling itself through `schedule_task`, while the human gets a direct way to inspect or cancel follow-ups without leaving Telegram or touching the SQLite database.

## A Note About Multiple MCP Calls

During development you may sometimes see two `schedule_task` tool calls in the Telegram activity stream for what looks like a single request. That is usually the agent retrying or refining its tool usage while reasoning. It does not automatically mean that two follow-ups were scheduled.

The durable source of truth is the task row stored in {term}`ACP_SCHEDULED_TASKS_DB`. If one row was persisted, then one follow-up survived.

## Related Pages

See {doc}`mcp` for the broader MCP architecture and {doc}`configuration` for the runtime settings involved in scheduling.

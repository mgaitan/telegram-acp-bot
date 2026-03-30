# Deferred Follow-ups

Deferred follow-ups let the agent schedule a one-shot action for later without
keeping the original ACP prompt alive.

This is useful for requests such as:

- `check this PR again in 10 minutes`
- `notify me at 9 PM`

## Current scope

The first iteration is intentionally narrow:

- one-shot tasks only
- local SQLite persistence
- an in-process scheduler inside the bot runtime
- two execution modes:
  - `notify`
  - `prompt_agent`

Not included yet:

- recurring schedules
- list/cancel commands in Telegram
- automatic session resume when no active chat session exists

## User-facing behavior

When a task is scheduled, the bot creates a single anchor message in Telegram:

- `Scheduled for 2026-03-30 21:00 UTC.`

When the task becomes due, the bot edits that same message:

- `Running now...`

When execution finishes, the bot edits the anchor again:

- `Completed.`
- or a failure message such as `Could not run automatically: no active session.`

For `prompt_agent`, the actual agent reply is delivered through the normal
Telegram reply pipeline as a reply to that anchor message.

## Session behavior

`prompt_agent` only runs when the chat already has an active ACP session.

If no active session exists at execution time:

- the task is marked as failed
- the anchor message is updated with a visible explanation

This keeps the first version predictable and avoids hidden session rehydration.

## Persistence model

Deferred tasks are stored in a local SQLite database configured via
{term}`ACP_SCHEDULED_TASKS_DB`.

Each task stores:

- target `chat_id`
- the Telegram anchor `message_id`
- execution `mode`
- payload (`notify_text` or `prompt_text`)
- `run_at`
- lifecycle status

## Flow

```{mermaid}
flowchart TD
    U[User in Telegram] --> B[Bot receives prompt]
    B --> A[ACP agent handles immediate work]
    A --> M[MCP tool schedule_task]
    M --> S[(SQLite scheduled tasks)]
    M --> T[Telegram anchor message]
    S --> W[Scheduler claims due task]
    W --> E[Execute notify or prompt_agent]
    E --> T2[Edit anchor message]
    E --> R[Reply to anchor with final agent output]
```

## Related pages

- {doc}`mcp`
- {doc}`configuration`

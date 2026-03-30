# MCP Integration

This project includes a built-in MCP server that the bot advertises automatically to the ACP agent. If you run:

```bash
uv run telegram-acp-bot
```

every new or resumed ACP session receives an MCP stdio server named `telegram-channel`. In practice, this is how the agent performs Telegram-specific side effects without extending ACP itself.

ACP remains responsible for the conversational session with the agent. MCP is the place for actions that belong to the Telegram channel: sending an attachment to the current chat, or scheduling a follow-up that should happen later. That split is the main design idea. It keeps the ACP side protocol-native while still giving the agent a clean way to interact with Telegram.

The implementation now lives under `telegram_acp_bot.mcp`, and the built-in server entrypoint is:

```bash
python -m telegram_acp_bot.mcp.server
```

## How The Pieces Fit Together

The easiest way to think about the MCP layer is as a small local service that knows how to translate an ACP-side tool call into a Telegram-side action.

```{mermaid}
:align: center
flowchart LR
    U[Telegram user] --> B[telegram-acp-bot]
    B --> A[ACP agent session]
    A --> M[Internal MCP server]
    M --> T[Telegram Bot API]
    M --> S[(Runtime state / SQLite)]
```

The bot process owns all of this. The MCP server is not a separate deployment. It is launched as a stdio subprocess and receives the minimal runtime state it needs to route tool calls back to the correct Telegram chat.

For configuration details, see {doc}`configuration` and {ref}`mcp-channel-environment-variables`. In normal use, the important environment variables are injected by the bot runtime itself: {term}`ACP_TELEGRAM_CHANNEL_STATE_FILE`, {term}`ACP_TELEGRAM_BOT_TOKEN`, and, when scheduling is enabled, {term}`ACP_SCHEDULED_TASKS_DB`.

## Available Tools

The built-in server currently exposes three tools.

`telegram_channel_info` returns capability metadata. It is mostly useful as a lightweight “what can this channel do?” probe.

`telegram_send_attachment` delivers a file or image to the current Telegram chat. The agent can provide either a base64 payload or, when explicitly enabled, a trusted local path. A typical call looks like this:

```text
telegram_send_attachment(
  data_base64="...",
  name="review.png",
  mime_type="image/png"
)
```

If the MIME type resolves to `image/*`, the server sends a Telegram photo. Otherwise it sends a document. In most deployments, `data_base64` is the safer default. The `path` input is intentionally disabled unless {term}`ACP_TELEGRAM_CHANNEL_ALLOW_PATH` is enabled.

`schedule_task` stores a one-shot deferred follow-up for the current chat. It supports two styles of time input. For absolute times, the agent can pass `run_at` as an ISO timestamp with an explicit timezone offset. For relative requests such as “in 10 minutes” or “in 30 seconds”, the preferred form is a delay input like `delay_minutes=10` or `delay_seconds=30`. That keeps the tool call simple and avoids making the agent compute a wall-clock timestamp for short delays.

A typical relative scheduling call looks like this:

```text
schedule_task(
  mode="prompt_agent",
  delay_minutes=10,
  prompt_text="Check the PR again and tell me if there are new comments."
)
```

## Session Routing

Both attachment delivery and scheduling need to know which Telegram chat should receive the side effect. The MCP layer resolves that from a local state file maintained by the bot runtime.

`AcpAgentService` persists a `session_id -> chat_id` mapping, and the MCP server reads that mapping when a tool call arrives. If a tool call does not provide `session_id`, the server tries to infer it. The inference order is intentionally conservative: if there is only one active mapped session, it uses that; otherwise it tries the last active session remembered by the channel state. If neither rule produces a safe answer, the tool returns an explicit error rather than guessing.

This routing model is what allows an agent response such as “send me the screenshot” or “remind me in 20 minutes” to affect the same Telegram chat that originated the ACP session.

## Why You May See Multiple Tool Calls

One source of confusion during development is that Telegram may show more than one `Tool call` block for the same MCP tool during a single agent turn. That does not necessarily mean the side effect happened twice.

The agent is free to call a tool more than once while deciding parameters or retrying a failed attempt. For example, it may first try one shape of `schedule_task` and then retry with corrected arguments. The reliable source of truth is the underlying side effect itself. For deferred follow-ups, that source of truth is the row persisted in {term}`ACP_SCHEDULED_TASKS_DB`. If only one task row was created, only one schedule survived.

## Example Sequence

The attachment flow is a good concrete example because it is short and easy to visualize:

```{mermaid}
:align: center
sequenceDiagram
    participant U as Telegram user
    participant B as Bot runtime
    participant A as ACP agent
    participant M as MCP telegram-channel
    participant T as Telegram Bot API

    U->>B: "Send me the generated screenshot"
    B->>A: ACP prompt
    A->>M: telegram_send_attachment(data_base64="...", name="artifact.png")
    M->>M: resolve session_id and chat_id
    M->>T: sendPhoto(chat_id, artifact.png)
    T-->>U: image delivered
    M-->>A: { ok: true, delivered_as: "photo" }
    A-->>B: final reply text
    B-->>U: confirmation message
```

## Current Limits

The MCP layer is intentionally small. It does not implement follow-up suggestion buttons yet, and multi-session ambiguity can still require an explicit `session_id`. That is a tradeoff in favor of predictable routing and debuggable behavior.

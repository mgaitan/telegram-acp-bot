# MCP Integration

This project includes a built-in MCP server that is automatically exposed to the ACP agent.

You do not need extra configuration for this. If you run:

```bash
uv run telegram-acp-bot
```

the bot advertises an MCP stdio server named `telegram-channel` when ACP sessions are created or loaded.

## Configuration

The internal MCP server is enabled automatically, but attachment behavior depends on specific environment variables.

- For a full reference, see {doc}`configuration` and the section {ref}`mcp-channel-environment-variables`.
- {term}`ACP_TELEGRAM_CHANNEL_ALLOW_PATH` enables trusted `path` inputs for `telegram_send_attachment`.
- {term}`ACP_TELEGRAM_CHANNEL_STATE_FILE` and {term}`ACP_TELEGRAM_BOT_TOKEN` are typically injected by the bot runtime for the internal MCP server.

## Why MCP here

ACP is used for conversation/session flow.
MCP is used for channel-specific tools (Telegram side effects).

This avoids custom ACP client extensions and keeps integration with `codex-acp` protocol-native.

## Built-in MCP server

The built-in server entrypoint is:

- `python -m telegram_acp_bot.mcp_channel`

It currently exposes:

1. `telegram_channel_info`
  Returns channel capability metadata.
2. `telegram_send_attachment`
  Sends a file/image attachment to Telegram for the active session.
3. `schedule_task`
  Schedules a one-shot deferred follow-up for the current Telegram chat.

## `telegram_send_attachment`

Inputs:

- `session_id` (optional)
- Exactly one of:
  - `path` (disabled by default; enable with `ACP_TELEGRAM_CHANNEL_ALLOW_PATH=1`)
  - `data_base64`
- `name` (optional)
- `mime_type` (optional)

Behavior:

- If mime resolves to `image/*`, it sends as Telegram photo.
- Otherwise it sends as Telegram document.
- For security, prefer `data_base64` unless you explicitly trust local file-path inputs.

## `schedule_task`

Inputs:

- `run_at` as ISO 8601 with an explicit timezone offset
- `mode`: `notify` or `prompt_agent`
- `notify_text` for `notify`
- `prompt_text` for `prompt_agent`
- `reply_to_message_id` (optional)
- `session_id` (optional)

Behavior:

- The tool resolves the current Telegram chat from the internal MCP channel state.
- It creates a Telegram anchor message immediately.
- It persists the deferred task in the SQLite database configured by {term}`ACP_SCHEDULED_TASKS_DB`.
- The bot scheduler later edits that anchor message as the task moves through execution.

## Session routing

To map MCP calls to the right Telegram chat:

- `AcpAgentService` stores `session_id -> chat_id` in a local state file.
- The MCP server reads that mapping before sending attachments.
- If `session_id` is omitted, MCP tries to infer it in this order:
  - single active mapped session
  - `last_session_id` saved in channel state (if still active)
- If ambiguity remains, MCP returns an explicit error including candidate `session_id` values.

## Sequence

```{mermaid}
:align: center
sequenceDiagram
    participant U as Telegram User
    participant B as telegram-acp-bot
    participant A as ACP Agent
    participant M as MCP telegram-channel
    participant T as Telegram Bot API

    U->>B: "Send me /webcam.jpg"
    B->>A: ACP prompt
    A->>M: telegram_send_attachment(session_id="s1", data_base64="...")
    M->>M: resolve session_id (explicit, single mapping, or last active)
    M->>M: map session_id -> chat_id
    M->>T: sendPhoto/sendDocument(chat_id, file)
    T-->>U: Attachment delivered
    M-->>A: { ok: true, delivered_as: ... }
    A-->>B: Final ACP response text
    B-->>U: Confirmation message
```

## Limitations

- Follow-up suggestion buttons are not implemented yet in MCP.
- Multi-session ambiguity can still require explicit `session_id` when no valid last active session is available.

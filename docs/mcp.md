# MCP Integration

This project includes a built-in MCP server that is automatically exposed to the ACP agent.

You do not need extra configuration for this. If you run:

```bash
uv run acp-bot
```

the bot advertises an MCP stdio server named `telegram-channel` when ACP sessions are created or loaded.

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

## Session routing

To map MCP calls to the right Telegram chat:

- `AcpAgentService` stores `session_id -> chat_id` in a local state file.
- The MCP server reads that mapping before sending attachments.
- If `session_id` is omitted, MCP infers it only when there is exactly one active mapped session.
- When multiple mapped sessions exist, provide `session_id` explicitly.

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
    M->>M: resolve session_id (explicit or single active mapping)
    M->>M: map session_id -> chat_id
    M->>T: sendPhoto/sendDocument(chat_id, file)
    T-->>U: Attachment delivered
    M-->>A: { ok: true, delivered_as: ... }
    A-->>B: Final ACP response text
    B-->>U: Confirmation message
```

## Limitations

- Follow-up suggestion buttons are not implemented yet in MCP.
- Multi-session ambiguity can require explicit `session_id`.

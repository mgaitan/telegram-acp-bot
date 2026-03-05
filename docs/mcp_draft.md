# MCP Channel Draft

## Goal

Replace client-specific ACP extension methods with a simple MCP-based channel for Telegram-specific features.

Initial targets:
- Send attachments from agent workflows.
- Offer follow-up prompt suggestions as Telegram buttons.

## Key Decision

Use an MCP server instead of ACP client extension methods.

Why:
- `codex-acp` already supports MCP server negotiation.
- The flow is standard and tool-driven.
- No custom ACP extension bridge is required in `codex-acp`.

## Transport Choice

HTTP is **not required**.

Start with an MCP server over **stdio**. This keeps setup simple and works well for local bot deployments.

About async:
- It can run in the same Python process/event loop if we embed an MCP server implementation.
- It can also run as a child process over stdio.
- For a first iteration, a child process is simpler to isolate and restart.

## Minimal Tool Set (Phase 1)

1. `telegram_send_attachment`
- Inputs: optional `session_id`, optional `name`, optional `mime_type`, and either `path` or `data_base64`.
- Effect: send a Telegram document/photo in the current chat.

2. `telegram_suggest_followups`
- Inputs: `title` (optional), `options` (`label` + `prompt`), `ttl_seconds` (optional).
- Effect: render inline buttons; clicking sends `prompt` as a user message.

## Session Routing

Each MCP call must map to the active Telegram chat/session context.

Implementation note:
- Keep an in-memory map from ACP session id to Telegram chat id.
- MCP tools resolve the target chat from this map.

Current implementation:
- The bot persists `session_id -> chat_id` to a local state file.
- The internal MCP server reads that mapping and delivers attachments through Telegram Bot API.
- If `session_id` is omitted, the server infers it from the last active/prompted session.

## Safety Rules

- Validate payload size and MIME type before sending files.
- Limit number of follow-up options (for example 2-5).
- Limit button label and prompt length.
- Expire stale follow-up buttons.

## Rollout Plan

1. [x] Implement `telegram_send_attachment`.
2. [x] Add tests for session routing and payload validation.
3. [ ] Add `telegram_suggest_followups`.
4. [ ] Add user setting to disable suggestions.

## Open Questions

- Keep MCP tool calls side-effecting (send immediately), or return preview first?
- Should follow-up suggestions replace previous ones or stack in chat?
- Do we need per-chat feature flags from bot settings?

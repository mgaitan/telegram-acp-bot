# MCP Integration

This project includes a built-in MCP server that the bot advertises automatically to the ACP agent. The practical reason for having it is simple: some things belong to the Telegram channel itself, not to the ACP conversation protocol.

That becomes visible in requests such as these:

```text
Send me the generated screenshot.
React with a thumbs up if that was enough.
Check for review in 2 minutes.
Remind me at 9 PM to restart the bot.
```

In all of those cases, the agent needs a channel-side capability. It may need to send a Telegram attachment, or schedule something that should happen later in the same chat. Those are exactly the kinds of actions exposed through the internal MCP tools.

If you run:

```bash
uv run telegram-acp-bot
```

every new or resumed ACP session receives an MCP stdio server named `telegram-channel`. In practice, this is how the agent performs Telegram-specific side effects without extending ACP itself.

ACP remains responsible for the conversational session with the agent. MCP is the place for actions that belong to the Telegram channel. That split is the main design idea. It keeps the ACP side protocol-native while still giving the agent a clean way to interact with Telegram.

## What The Tools Enable

The built-in server currently exposes four tools.

`telegram_channel_info` is a lightweight capability probe. It lets the agent confirm what this Telegram channel integration can currently do.

`telegram_send_attachment` lets the agent deliver a file or image to the current Telegram chat. This is the tool behind flows like “send me the generated screenshot” or “upload the review artifact”.

`telegram_set_message_reaction` lets the agent add a standard Telegram emoji reaction to a message in the current chat. This is the tool behind lightweight moments such as “acknowledge that briefly” or “react instead of sending a noisy one-line reply”.

`schedule_task` lets the agent schedule a one-shot deferred follow-up. This is the tool behind flows like “check for review in 2 minutes” or “remind me tomorrow at 9 PM”.

Here is a concrete example of the second case. A user may say:

```text
Check for review in 2 minutes.
```

The agent can translate that into a relative scheduling call such as:

```text
schedule_task(
  mode="prompt_agent",
  delay_minutes=2,
  prompt_text="Check the review status again and report any changes."
)
```

The point of the relative delay inputs is that the agent does not need to compute an absolute timestamp for short delays. For this kind of prompt, `delay_minutes=2` is the natural representation.

## How It Is Implemented

The implementation lives under `telegram_acp_bot.mcp`, and the built-in server entrypoint is:

```bash
python -m telegram_acp_bot.mcp.server
```

The easiest way to think about the MCP layer is as a small local service that translates an ACP-side tool call into a Telegram-side action.

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

(mcp-feature-attachments)=
## Reply With Attachments

`telegram_send_attachment` accepts either a base64 payload or, when explicitly enabled, a trusted local path. This is the feature behind flows where the agent answers not only with text, but also with a file or image sent back to Telegram. A typical call looks like this:

```text
telegram_send_attachment(
  data_base64="...",
  name="review.png",
  mime_type="image/png"
)
```

If the MIME type resolves to `image/*`, the server sends a Telegram photo. Otherwise it sends a document. In most deployments, `data_base64` is the safer default. The `path` input is intentionally disabled unless {term}`ACP_TELEGRAM_CHANNEL_ALLOW_PATH` is enabled.

(mcp-feature-reactions)=
## React To Messages

`telegram_set_message_reaction` is the small, Telegram-native counterpart to a normal text reply. It is useful when the agent wants to acknowledge something briefly, celebrate success, or confirm that it understood the user's last message without adding another sentence to the chat.

A typical call looks like this:

```text
telegram_set_message_reaction(
  emoji="👍"
)
```

When `message_id` is omitted, the server tries to infer the active Telegram prompt message for the current turn. That makes the tool natural to use during a normal conversational reply. If there is no active prompt message to infer, the tool returns an explicit error instead of guessing.

The first version intentionally supports only standard Telegram reaction emoji. Custom emoji and stickers are a separate design space, and they are tracked independently from this lightweight reaction flow.

(mcp-feature-scheduling)=
## Schedule Tasks

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

## External MCP Servers

In addition to the built-in `telegram-channel` server, you can register external MCP servers via the `mcp_servers` key in the {doc}`config file <configuration>`.

All configured servers are merged with the internal one and passed to every ACP session at startup. If you configure a server named `telegram-channel`, it overrides the internal server.

### Stdio server (local process)

```json
{
  "mcp_servers": {
    "echo": {
      "command": "uv",
      "args": ["run", "examples/echo_agent.py"],
      "env": {
        "MY_API_KEY": "secret"
      }
    }
  }
}
```

Fields:

- key name — server identifier exposed to ACP; must match `^[a-z0-9-_]+$`
- `command` (required) — path to the executable
- `args` (optional) — list of string arguments
- `env` (optional) — object of `string → string` environment variables

### Remote server (HTTP)

```json
{
  "mcp_servers": {
    "remote": {
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

Fields:

- key name — server identifier exposed to ACP; must match `^[a-z0-9-_]+$`
- `url` (required) — URL of the remote MCP server
- `headers` (optional) — object of `string → string` HTTP headers

### Mixed example

```json
{
  "mcp_servers": {
    "local": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/home/alice/projects"]
    },
    "remote": {
      "url": "https://mcp.example.com/mcp",
      "headers": {
        "Authorization": "Bearer token"
      }
    }
  }
}
```

### Validation

Each server must define exactly one transport: `command` (stdio) **or** `url` (remote), not both. The bot validates the config at startup and reports clear errors for invalid definitions.

### Troubleshooting

**Missing transport**
```json
{ "mcp_servers": { "broken": { "args": ["--help"] } } }
```
Error: `MCP server 'broken' must define 'command' (stdio) or 'url' (remote)`

**Both transports defined**
```json
{ "mcp_servers": { "broken": { "command": "x", "url": "http://x" } } }
```
Error: `MCP server 'broken' cannot define both 'command' and 'url'`

**Wrong type for args**
```json
{ "mcp_servers": { "broken": { "command": "x", "args": "--flag" } } }
```
Error: `MCP server 'broken' 'args' must be a list of strings`

## Current Limits

The MCP layer is intentionally small. It does not implement follow-up suggestion buttons yet, and multi-session ambiguity can still require an explicit `session_id`. That is a tradeoff in favor of predictable routing and debuggable behavior.

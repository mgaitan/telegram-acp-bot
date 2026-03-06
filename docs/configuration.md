# Configuration

This page documents runtime configuration via environment variables.

For ready-to-use command examples per backend, see {doc}`agents`.

```{glossary}
TELEGRAM_BOT_TOKEN
  Telegram bot token (from BotFather). Required unless passed as `--telegram-token`.

TELEGRAM_ALLOWED_USER_IDS
  Comma-separated allowlist of Telegram numeric user IDs.
  Example: `123456,987654`.

TELEGRAM_ALLOWED_USERNAMES
  Comma-separated allowlist of Telegram usernames.
  Usernames are normalized to lowercase and can include or omit `@`.
  Example: `alice,@bob`.

ACP_AGENT_COMMAND
  Command line used to launch the ACP agent process.
  Examples: `npx @zed-industries/codex-acp`, `uv run examples/echo_agent.py`.
  Required unless passed as `--agent-command`.

ACP_RESTART_COMMAND
  Optional command used by `/restart` to relaunch the bot process.
  Recommended when you run with `uv run ...` and need to preserve its flags.
  Example: `uv run acp-bot --telegram-token ... --agent-command ...`.

ACP_PERMISSION_MODE
  Default permission policy for ACP tool calls.
  Allowed values: `ask`, `approve`, `deny`.
  Maps to `--permission-mode`.

ACP_PERMISSION_EVENT_OUTPUT
  Permission/tool event log output mode.
  Allowed values: `stdout`, `off`.
  Maps to `--permission-event-output`.

ACP_STDIO_LIMIT
  Asyncio stdio reader limit in bytes for ACP transport.
  Increase this if the agent emits very large JSON lines.
  Maps to `--acp-stdio-limit`.

ACP_CONNECT_TIMEOUT
  Timeout in seconds for ACP initialize/new_session handshake.
  Prevents `/new` from hanging forever if the agent does not speak ACP over stdio.
  Maps to `--acp-connect-timeout`.

ACP_LOG_LEVEL
  Application log level.
  Common values: `DEBUG`, `INFO`, `WARNING`, `ERROR`.
```

## Example `.env`

```ini
TELEGRAM_BOT_TOKEN=123456:abc
TELEGRAM_ALLOWED_USER_IDS=123456789
# TELEGRAM_ALLOWED_USERNAMES=alice,@bob
ACP_AGENT_COMMAND="npx @zed-industries/codex-acp"
ACP_RESTART_COMMAND="uv run acp-bot --telegram-token 123456:abc --agent-command \"npx @zed-industries/codex-acp\""
ACP_PERMISSION_MODE=ask
ACP_PERMISSION_EVENT_OUTPUT=stdout
ACP_STDIO_LIMIT=8388608
ACP_CONNECT_TIMEOUT=30
ACP_LOG_LEVEL=INFO
```

## MCP behavior

The bot always advertises an internal MCP stdio server named `telegram-channel`
to the ACP agent. No extra configuration is required.

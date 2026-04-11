# Configuration

telegram-acp-bot uses a three-layer configuration model. Settings are resolved in this order:

```
CLI flags  >  Environment variables  >  Config file  >  Built-in defaults
```

You can mix layers freely. A typical setup uses a config file for static values and environment variables for secrets that should not live on disk.

## Config file

Pass `--config <path>` (or set it in a wrapper script) to load settings from a JSON file.

```bash
telegram-acp-bot --config ./telegram-acp.json
```

The file must be a JSON object. All keys are optional. Missing keys fall back to environment variables or built-in defaults.

### Schema

```json
{
  "telegram": {
    "bot_token": "123456:abc",
    "allowed_user_ids": [123456789],
    "allowed_usernames": ["alice", "@bob"]
  },
  "acp": {
    "agent_command": "npx @zed-industries/codex-acp",
    "restart_command": "uv run telegram-acp-bot --config ./telegram-acp.json",
    "permission_mode": "ask",
    "permission_event_output": "stdout",
    "stdio_limit": 8388608,
    "connect_timeout": 30,
    "log_format": "text",
    "log_level": "INFO",
    "activity_mode": "normal",
    "scheduled_tasks_db": "/home/alice/.local/state/telegram-acp-bot/scheduled-tasks.sqlite3",
    "workspace": "/home/alice/projects"
  }
}
```

### Validation

The config file is validated at startup. Invalid values (wrong types, unknown enum values) are reported as clear error messages and the bot does not start.

## Environment variables

Environment variables override any matching config file value. They are the recommended way to pass secrets like the bot token in container or CI environments.

```{glossary}
TELEGRAM_BOT_TOKEN
  Telegram bot token (from BotFather). Required unless set via config file or `--telegram-token`.

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
  Required unless set via config file or `--agent-command`.

ACP_RESTART_COMMAND
  Optional command used by `/restart` to relaunch the bot process.
  Recommended when you run with `uv run ...` and need to preserve its flags.

ACP_PERMISSION_MODE
  Default permission policy for ACP tool calls.
  Allowed values: `ask`, `approve`, `deny`.

ACP_PERMISSION_EVENT_OUTPUT
  Permission/tool event log output mode.
  Allowed values: `stdout`, `off`.

ACP_STDIO_LIMIT
  Asyncio stdio reader limit in bytes for ACP transport.
  Increase this if the agent emits very large JSON lines.

ACP_CONNECT_TIMEOUT
  Timeout in seconds for ACP initialize/new_session handshake.

ACP_LOG_LEVEL
  Application log level.
  Common values: `DEBUG`, `INFO`, `WARNING`, `ERROR`.

ACP_LOG_FORMAT
  Application log format.
  Allowed values: `text`, `json`.

ACP_ACTIVITY_MODE
  Controls how intermediate agent activity events are shown in Telegram.
  Allowed values: `normal`, `compact`, `verbose`.
  `normal` (default) emits each visible activity event as its own message.
  `compact` collapses all events into a single in-place status message
  that is replaced by the final answer when the agent responds.
  `verbose` streams append-only updates in place for active reply text and
  tool activity, and then finalizes those messages when the prompt completes.

ACP_SCHEDULED_TASKS_DB
  Path to the SQLite database used for deferred follow-up tasks.

ACP_TELEGRAM_CHANNEL_ALLOW_PATH
  Enables `path` inputs for the internal MCP `telegram_send_attachment` tool.
  Disabled by default. Set to `1` (or `true`/`yes`/`on`) only when file-path inputs are trusted.

ACP_TELEGRAM_CHANNEL_ALLOW_PUBLISH
  Enables external long-form publishing for the internal MCP `telegram_publish_markdown` tool.
  Disabled by default because published pages are hosted outside Telegram.

ACP_TELEGRAM_CHANNEL_STATE_FILE
  Path to the MCP channel shared state file (`session_id -> chat_id`, plus last active session).
  Usually injected by the bot runtime for the internal MCP server.

ACP_TELEGRAM_BOT_TOKEN
  Telegram bot token used by the internal MCP server when sending attachments.
  Usually injected by the bot runtime for the internal MCP server.

ACP_TELEGRAM_CHANNEL_PUBLISH_SHORT_NAME
  Telegraph account short name used when creating transient publishing accounts.
  Defaults to `telegram-acp-bot`.

ACP_TELEGRAM_CHANNEL_PUBLISH_AUTHOR_NAME
  Default author name for `telegram_publish_markdown`.
  Defaults to `telegram-acp-bot`.

ACP_TELEGRAM_CHANNEL_PUBLISH_AUTHOR_URL
  Optional author URL attached to published Telegraph pages.
```

## CLI flags

CLI flags take the highest priority and override both environment variables and the config file. Run `telegram-acp-bot --help` for the full list.

Key flags that map to config file or environment variable equivalents:

| CLI flag | Env var | Config key |
|---|---|---|
| `--config` | — | — |
| `--telegram-token` | `TELEGRAM_BOT_TOKEN` | `telegram.bot_token` |
| `--allowed-user-id` | `TELEGRAM_ALLOWED_USER_IDS` | `telegram.allowed_user_ids` |
| `--allowed-username` | `TELEGRAM_ALLOWED_USERNAMES` | `telegram.allowed_usernames` |
| `--agent-command` | `ACP_AGENT_COMMAND` | `acp.agent_command` |
| `--restart-command` | `ACP_RESTART_COMMAND` | `acp.restart_command` |
| `--permission-mode` | `ACP_PERMISSION_MODE` | `acp.permission_mode` |
| `--permission-event-output` | `ACP_PERMISSION_EVENT_OUTPUT` | `acp.permission_event_output` |
| `--acp-stdio-limit` | `ACP_STDIO_LIMIT` | `acp.stdio_limit` |
| `--acp-connect-timeout` | `ACP_CONNECT_TIMEOUT` | `acp.connect_timeout` |
| `--log-format` | `ACP_LOG_FORMAT` | `acp.log_format` |
| `--activity-mode` / `-m` | `ACP_ACTIVITY_MODE` | `acp.activity_mode` |
| `--scheduled-tasks-db` | `ACP_SCHEDULED_TASKS_DB` | `acp.scheduled_tasks_db` |
| `--workspace` | — | `acp.workspace` |

## Migration guide

### From `.env` to `config.json`

If you currently use a `.env` file:

```ini
TELEGRAM_BOT_TOKEN=123456:abc
TELEGRAM_ALLOWED_USER_IDS=123456789
ACP_AGENT_COMMAND=npx @zed-industries/codex-acp
ACP_PERMISSION_MODE=approve
```

The equivalent config file is:

```json
{
  "telegram": {
    "bot_token": "123456:abc",
    "allowed_user_ids": [123456789]
  },
  "acp": {
    "agent_command": "npx @zed-industries/codex-acp",
    "permission_mode": "approve"
  }
}
```

Then run:

```bash
telegram-acp-bot --config ./telegram-acp.json
```

### Mixed setup (env override for secrets)

Keep the token in the environment and everything else in the config file:

```json
{
  "telegram": {
    "allowed_user_ids": [123456789]
  },
  "acp": {
    "agent_command": "npx @zed-industries/codex-acp"
  }
}
```

```bash
TELEGRAM_BOT_TOKEN=123456:abc telegram-acp-bot --config ./telegram-acp.json
```

The environment variable wins over the config file for `bot_token`, so you can rotate the token without touching the file.

## MCP behavior

The bot always advertises an internal MCP stdio server named `telegram-channel`
to the ACP agent. No extra configuration is required.

For adding external MCP servers, see {doc}`mcp`.

(mcp-channel-environment-variables)=
## MCP channel environment variables

- {term}`ACP_TELEGRAM_CHANNEL_ALLOW_PATH` controls whether MCP attachment delivery accepts `path` input.
  Default behavior is disabled, so agents must use `data_base64`.
- {term}`ACP_TELEGRAM_CHANNEL_ALLOW_PUBLISH` controls whether MCP long-form publishing is allowed.
  Default behavior is disabled, so agents cannot publish outside Telegram unless you opt in.
- {term}`ACP_TELEGRAM_CHANNEL_STATE_FILE` points to runtime state used to resolve session routing for MCP calls.
- {term}`ACP_TELEGRAM_BOT_TOKEN` is the token consumed by the MCP server when calling Telegram Bot API.
- {term}`ACP_TELEGRAM_CHANNEL_PUBLISH_SHORT_NAME` sets the Telegraph short name for published pages.
- {term}`ACP_TELEGRAM_CHANNEL_PUBLISH_AUTHOR_NAME` sets the default author label for published pages.
- {term}`ACP_TELEGRAM_CHANNEL_PUBLISH_AUTHOR_URL` sets the optional author URL for published pages.

Security notes:

- Keep {term}`ACP_TELEGRAM_CHANNEL_ALLOW_PATH` disabled unless your agent is trusted to read local files.
- Keep {term}`ACP_TELEGRAM_CHANNEL_ALLOW_PUBLISH` disabled unless your agent is trusted to publish content outside Telegram.
- {term}`ACP_TELEGRAM_CHANNEL_STATE_FILE` and {term}`ACP_TELEGRAM_BOT_TOKEN` are typically managed by the bot process and should not be shared across unrelated runtimes.

# CLI Reference

The bot entrypoint is `acp-bot`.

```{richterm} env PYTHONPATH=../src uv run -m telegram_acp_bot --help
:hide-command: true
```

## Arguments

- `--telegram-token`
  Default: {term}`TELEGRAM_BOT_TOKEN`.
  Required if {term}`TELEGRAM_BOT_TOKEN` is not set.

- `--agent-command`
  Default: {term}`ACP_AGENT_COMMAND`.
  Required if {term}`ACP_AGENT_COMMAND` is not set.

- `--restart-command`
  Default: {term}`ACP_RESTART_COMMAND`.
  Optional command used by `/restart` to relaunch the process.

- `--allowed-user-id`
  Repeatable allowlist of Telegram user IDs.
  If omitted, the bot accepts messages from any user.

- `--workspace`
  Default workspace used by `/new` when no workspace path is provided.

- `--permission-mode`
  Default: {term}`ACP_PERMISSION_MODE`.
  Allowed values: `ask`, `approve`, `deny`.

- `--permission-event-output`
  Default: {term}`ACP_PERMISSION_EVENT_OUTPUT`.
  Allowed values: `stdout`, `off`.

- `--acp-stdio-limit`
  Default: {term}`ACP_STDIO_LIMIT`.
  Asyncio stdio reader limit in bytes.

- `--acp-connect-timeout`
  Default: {term}`ACP_CONNECT_TIMEOUT`.
  Timeout in seconds for ACP `initialize` + `new_session` handshake.

- `--mcp-server-stdio-name`
  Default: {term}`ACP_MCP_SERVER_STDIO_NAME`.
  Optional MCP stdio server name to pass to ACP `new_session` / `load_session`.
  Must be set together with `--mcp-server-stdio-command`.

- `--mcp-server-stdio-command`
  Default: {term}`ACP_MCP_SERVER_STDIO_COMMAND`.
  Optional MCP stdio server command line to pass to ACP `new_session` / `load_session`.
  Must be set together with `--mcp-server-stdio-name`.

- `-V`, `--version`
  Print CLI version and exit.

## Notes

- `/restart` behavior:
  - If {term}`ACP_RESTART_COMMAND` (or `--restart-command`) is set, that command is executed.
  - Otherwise, the bot re-execs itself using `sys.executable + sys.argv`.

# CLI Reference

The bot entrypoint is `telegram-acp-bot`.

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
  At least one allowlist entry is required across IDs/usernames.

- `--allowed-username`
  Repeatable allowlist of Telegram usernames.
  `@` prefix is optional; values are normalized to lowercase.
  At least one allowlist entry is required across IDs/usernames.

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

- `--log-format`
  Default: {term}`ACP_LOG_FORMAT`.
  Allowed values: `text`, `json`.

- `--activity-mode`
  Default: {term}`ACP_ACTIVITY_MODE`.
  Allowed values: `verbose`, `compact`.
  Controls how intermediate agent activity events appear in the chat.
  `verbose` sends each in-progress tool-call as its own message and updates it in-place
  when the tool completes (with command output, thinking text, etc.).
  `compact` collapses all intermediate events into a single status message edited in-place,
  then replaced by the final answer.

- `-V`, `--version`
  Print CLI version and exit.

## Notes

- `/restart` behavior:
  - If {term}`ACP_RESTART_COMMAND` (or `--restart-command`) is set, that command is executed.
  - Otherwise, the bot re-execs itself using `sys.executable + sys.argv`.
  - It requires an active session and reports session context (`session_id`, `workspace`) in the response.
  - `/restart N [workspace]` selects and loads a resumable session in-process (no process relaunch), but keeps the same restart acknowledgment text for UX consistency.
- Access control behavior:
  - Configure at least one allowlist entry via `--allowed-user-id`, `--allowed-username`,
    {term}`TELEGRAM_ALLOWED_USER_IDS`, or {term}`TELEGRAM_ALLOWED_USERNAMES`.
- MCP behavior:
  - `telegram-acp-bot` always advertises an internal MCP stdio server (`telegram-channel`) to the ACP agent.

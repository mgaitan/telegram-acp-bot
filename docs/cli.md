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

- `-m`, `--activity-mode`
  Default: {term}`ACP_ACTIVITY_MODE`.
  Allowed values: `normal`, `compact`, `verbose`.
  In `normal`, the bot sends activity/tool updates as separate messages without
  streaming edits.
  In `compact`, the bot keeps a single in-progress reply message per prompt,
  updates that message in place, and keeps the normal activity emoji in the
  message text while work is in progress.
  In `verbose`, the bot streams append-only updates in place for active reply
  text and tool activity, then finalizes the same message when possible.

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

## register-commands

Register (or delete) the bot's slash commands in Telegram via `setMyCommands`.

```bash
telegram-acp-bot register-commands --telegram-token <TOKEN> [options]
```

### Options

- `--telegram-token`
  Default: {term}`TELEGRAM_BOT_TOKEN`.
  Required if {term}`TELEGRAM_BOT_TOKEN` is not set.

- `--scope`
  BotCommandScope to target.
  Allowed values: `default`, `all_private_chats`, `all_group_chats`, `all_chat_administrators`.
  Default: `default` (global scope for all users).

- `--language-code`
  IETF language code for language-specific registration (e.g. `en`, `es`).
  Omit to register for all languages.

- `--dry-run`
  Print the commands that would be registered without calling the Telegram API.

- `--delete`
  Delete registered commands for the given scope/language instead of setting them
  (`deleteMyCommands`). Useful for cleanup flows.

### Examples

Register commands for all users (default scope):

```bash
telegram-acp-bot register-commands --telegram-token "$TELEGRAM_BOT_TOKEN"
```

Preview what would be registered without calling the API:

```bash
telegram-acp-bot register-commands --telegram-token "$TELEGRAM_BOT_TOKEN" --dry-run
```

Register commands only for private chats in English:

```bash
telegram-acp-bot register-commands \
  --telegram-token "$TELEGRAM_BOT_TOKEN" \
  --scope all_private_chats \
  --language-code en
```

Delete all registered commands for the default scope:

```bash
telegram-acp-bot register-commands --telegram-token "$TELEGRAM_BOT_TOKEN" --delete
```

### Notes

- Command definitions are sourced from the codebase constant
  `telegram_acp_bot.telegram.bot.BOT_COMMANDS`.  Running `register-commands` is idempotent —
  it is safe to call multiple times.
- Exit code `0` on success; `1` on Telegram API error.

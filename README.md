# Telegram ACP bot

[![CI](https://github.com/mgaitan/telegram-acp-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/mgaitan/telegram-acp-bot/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-blue.svg?style=flat)](https://mgaitan.github.io/telegram-acp-bot/)
[![PyPI version](https://img.shields.io/pypi/v/telegram-acp-bot?logo=pypi&logoColor=white)](https://pypi.org/project/telegram-acp-bot/)
[![Changelog](https://img.shields.io/github/v/release/mgaitan/telegram-acp-bot?include_prereleases&label=changelog)](https://github.com/mgaitan/telegram-acp-bot/releases)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/mgaitan/telegram-acp-bot/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mgaitan/telegram-acp-bot/blob/main/LICENSE)

A Telegram bot that implements the [Agent Client Protocol](https://agentclientprotocol.com/) to interact with AI agents.

Project documentation: <https://mgaitan.github.io/telegram-acp-bot/>

## Status

This project is in **alpha** and under active development. Development has included extensive use of AI agents. Human and agents contributions are welcome.

## Quick Start

Run directly without installing via `uvx`:

```bash
uvx telegram-acp-bot --help
```

Run the latest development version from git:

```bash
uvx git+https://github.com/mgaitan/telegram-acp-bot --help
```

Run the bot with a real ACP agent:

```bash
TELEGRAM_BOT_TOKEN=123456:abc \
ACP_AGENT_COMMAND="npx @zed-industries/codex-acp" \
uvx telegram-acp-bot
```

Current interaction capabilities:
- `/new [workspace]`, `/resume [N|workspace]`, `/session`, `/cancel`, `/stop`, `/clear`, `/restart [N [workspace]]`
- Interactive permission prompts with inline buttons (`Always`, `This time`, `Deny`)
- Plain text prompts
- Tool activity updates are sent as separate messages per ACP tool kind
- Image and document attachments from Telegram messages
- ACP `file://` resources are sent as attachments when they resolve to files inside the active workspace
- Agent markdown output (with fallback to plain text when Telegram rejects entities)

Message flow:
- The bot sends activity blocks while the prompt is running.
- Common labels are `💡 Thinking`, `⚙️ Running`, `📖 Reading`, `✏️ Editing`, `✍️ Writing`, `🌐 Searching web`, `🔎 Querying project`, and fallback `🔎 Querying`.
- Permission prompts for risky actions are sent as independent messages with inline buttons.
- The final answer is sent as a separate message after activity blocks.
- If the final text is empty, no dummy "(no text response)" message is sent.

For development, `/restart` stops polling and relaunches the process.
If `ACP_RESTART_COMMAND` (or `--restart-command`) is configured, that command is used (recommended when running with `uv run ...` and extra flags).
Otherwise, it falls back to re-execing the current process (`sys.executable + sys.argv`).

## Telegram Bot Token

Create your token with [@BotFather](https://t.me/BotFather):

1. Open BotFather and run `/newbot`.
2. Choose a bot name and username.
3. Copy the token returned by BotFather.

Store the token in your local `.env` file (gitignored):

At least one allowlist entry is required (`TELEGRAM_ALLOWED_USER_IDS` or `TELEGRAM_ALLOWED_USERNAMES`).

```env
TELEGRAM_BOT_TOKEN=123456:abc
TELEGRAM_ALLOWED_USER_IDS=123456789
# TELEGRAM_ALLOWED_USERNAMES=alice,@bob
ACP_AGENT_COMMAND="npx @zed-industries/codex-acp"
ACP_RESTART_COMMAND="uv run telegram-acp-bot --telegram-token <TOKEN> --agent-command \"npx @zed-industries/codex-acp\""
ACP_PERMISSION_MODE=ask
ACP_PERMISSION_EVENT_OUTPUT=stdout
ACP_STDIO_LIMIT=8388608
```

## Agent Command

Set `ACP_AGENT_COMMAND` to the ACP-compatible agent command you want the bot to run.

Example:

```env
ACP_AGENT_COMMAND="npx @zed-industries/codex-acp"
```

To install the tool permanently:

```bash
uv tool install telegram-acp-bot
```

## Development

- Install dependencies with `uv sync`.
- Then run `uv run telegram-acp-bot`
- New dependency releases are delayed by one week via `uv` cooldown (`[tool.uv].exclude-newer = "1 week"`), with per-package overrides when required (for example, `ty`).
- Run the QA bundle with [`ty`](https://github.com/astral-sh/ty):

```bash
uv run ty check
```

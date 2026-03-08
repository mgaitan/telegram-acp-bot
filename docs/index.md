# Telegram ACP bot

[![ci](https://github.com/mgaitan/telegram-acp-bot/workflows/ci/badge.svg)](https://github.com/mgaitan/telegram-acp-bot/actions?query=workflow%3Aci)
[![docs](https://img.shields.io/badge/docs-blue.svg?style=flat)](https://mgaitan.github.io/telegram-acp-bot/)
[![pypi version](https://img.shields.io/pypi/v/telegram-acp-bot?logo=pypi&logoColor=white)](https://pypi.org/project/telegram-acp-bot/)
[![Changelog](https://img.shields.io/github/v/release/mgaitan/telegram-acp-bot?include_prereleases&label=changelog)](https://github.com/mgaitan/telegram-acp-bot/releases)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/mgaitan/telegram-acp-bot/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mgaitan/telegram-acp-bot/blob/main/LICENSE)

A Telegram bot that implements Agent Client Protocol to interact with AI agents

## Quick Start

Run directly without installing via `uvx`

```bash
uvx telegram-acp-bot --help
```

Run the latest development version from git:

```bash
uvx git+https://github.com/mgaitan/telegram-acp-bot --help
```

```{richterm} env PYTHONPATH=../src uv run -m telegram_acp_bot --help
:hide-command: true
```

To install the tool permanently:

```bash
uv tool install telegram-acp-bot
```

Run the bot with a real ACP agent:

```bash
TELEGRAM_BOT_TOKEN=123456:abc \
ACP_AGENT_COMMAND="codex-acp" \
uvx telegram-acp-bot
```

You can also set `TELEGRAM_BOT_TOKEN` and `ACP_AGENT_COMMAND` in a local `.env` file.

Supported bot commands:
- `/new [workspace]`
- `/resume [N|workspace]`
- `/session`
- `/cancel`
- `/stop`
- `/clear`
- `/restart [N [workspace]]`

Session behavior:
- The first user prompt in a chat starts a session implicitly in the default workspace.
- Use `/new [workspace]` when you want to explicitly switch to another session/workspace.

Attachment behavior:
- Telegram inbound photos/documents are forwarded to ACP prompts.
- ACP `file://` resources are delivered back as Telegram attachments when the file is inside the active workspace.

Tool activity behavior:
- ACP tool updates are emitted as separate Telegram messages grouped by tool kind (`think`, `execute`, `read`, etc.).
- Labels currently used in chat are: `💡 Thinking`, `⚙️ Running`, `📖 Reading`, `✏️ Editing`, `✍️ Writing`, `🌐 Searching web`, and `🔎 Querying` (or Spanish equivalents when `ACP_UI_LANGUAGE` is set to `es`).
- `Thinking` blocks preserve basic markdown emphasis (for example, `**issue refs**`) while command/code views remain code-formatted.
- Search activity blocks render compact details when available (`Query: "..."` and `URL: ...`) extracted from block title/text.
- Permission prompts for risky actions are sent as independent messages with inline buttons.
- The final assistant answer is sent as a separate message after those activity blocks.
- If the final text payload is empty, no dummy "(no text response)" message is emitted.

Permission behavior:
- By default (`ACP_PERMISSION_MODE=ask`), permission requests are shown in Telegram with inline buttons.
- You can set startup defaults with:
  - `ACP_PERMISSION_MODE=ask|approve|deny`
  - `ACP_PERMISSION_EVENT_OUTPUT=stdout|off`
  - `ACP_STDIO_LIMIT=8388608` (increase for very large ACP stdout JSON lines)


```{toctree}
:maxdepth: 2
:caption: Documentation

how-to.md
agents.md
cli.md
mcp.md
configuration.md
about_the_docs.md
../CONTRIBUTING.md
../CODE_OF_CONDUCT.md
```

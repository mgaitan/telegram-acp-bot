# Telegram ACP bot

[![ci](https://github.com/mgaitan/telegram-acp-bot/workflows/ci/badge.svg)](https://github.com/mgaitan/telegram-acp-bot/actions?query=workflow%3Aci)
[![docs](https://img.shields.io/badge/docs-blue.svg?style=flat)](https://mgaitan.github.io/telegram-acp-bot/)
[![pypi version](https://img.shields.io/pypi/v/telegram-acp-bot.svg)](https://pypi.org/project/telegram-acp-bot/)
[![Changelog](https://img.shields.io/github/v/release/mgaitan/telegram-acp-bot?include_prereleases&label=changelog)](https://github.com/mgaitan/telegram-acp-bot/releases)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/mgaitan/telegram-acp-bot/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mgaitan/telegram-acp-bot/blob/main/LICENSE)

A Telegram bot that implements Agent Client Protocol to interact with AI agents

## Quick Start

Run directly without installing via `uvx`

```bash
uvx --with=telegram-acp-bot acp-bot --help
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
uvx --with=telegram-acp-bot acp-bot
```

You can also set `TELEGRAM_BOT_TOKEN` and `ACP_AGENT_COMMAND` in a local `.env` file.


```{toctree}
:maxdepth: 2
:caption: Documentation

../CONTRIBUTING.md
../CODE_OF_CONDUCT.md
about_the_docs.md
preliminary_ux.md
```

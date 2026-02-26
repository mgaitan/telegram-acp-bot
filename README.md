# Telegram ACP bot

[![CI](https://github.com/mgaitan/telegram-acp-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/mgaitan/telegram-acp-bot/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-blue.svg?style=flat)](https://mgaitan.github.io/telegram-acp-bot/)
[![pypi version](https://img.shields.io/pypi/v/telegram-acp-bot.svg)](https://pypi.org/project/telegram-acp-bot/)
[![Changelog](https://img.shields.io/github/v/release/mgaitan/telegram-acp-bot?include_prereleases&label=changelog)](https://github.com/mgaitan/telegram-acp-bot/releases)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/mgaitan/telegram-acp-bot/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mgaitan/telegram-acp-bot/blob/main/LICENSE)

A Telegram bot that implements Agent Client Protocol to interact with AI agents

## Quick Start

Run directly without installing via `uvx`:

```bash
uvx --with=telegram-acp-bot acp-bot
```

To install the tool permanently:

```bash
uv tool install telegram-acp-bot
```

## Development

- Install dependencies with `uv sync`.
- New dependency releases are delayed by one week via `uv` cooldown (`[tool.uv].exclude-newer = "1 week"`), with per-package overrides when required (for example, `ty`).
- Run the QA bundle with [`ty`](https://github.com/astral-sh/ty):

```bash
uv run ty check
```

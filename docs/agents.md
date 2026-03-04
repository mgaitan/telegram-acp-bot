# Agent Setup

This chapter explains how to configure common ACP agents with this bot.

Use {term}`ACP_AGENT_COMMAND` to define the command executed by `/new`.

## Quick compatibility map

| Agent | ACP support | Recommended command |
| --- | --- | --- |
| [Gemini CLI](https://google-gemini.github.io/gemini-cli/) | Native ACP support (experimental flag required) | `ACP_AGENT_COMMAND="npx @google/gemini-cli --experimental-acp"` |
| GitHub Copilot CLI | Native ACP support (`--acp`) | `ACP_AGENT_COMMAND="copilot --acp"` |
| [Codex CLI](https://developers.openai.com/codex/cli) | Via ACP adapter | `ACP_AGENT_COMMAND="npx @zed-industries/codex-acp"` |
| [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) / [Claude Code](https://code.claude.com/docs/en/cli-reference) | Via ACP adapter | `ACP_AGENT_COMMAND="npx @zed-industries/claude-agent-acp"` |

## Gemini CLI (native ACP)

1. Install and authenticate Gemini CLI first.
2. Set `GEMINI_API_KEY` (or another auth method supported by Gemini CLI).
3. Start the bot with Gemini as agent:

```bash
GEMINI_API_KEY=... \
ACP_AGENT_COMMAND="npx @google/gemini-cli --experimental-acp" \
uv run acp-bot
```

If `/new` does not return, set a lower {term}`ACP_CONNECT_TIMEOUT` and `ACP_LOG_LEVEL=DEBUG` to see where handshake stops.
See also [Gemini CLI repository](https://github.com/google-gemini/gemini-cli).

## Codex CLI (via adapter)

[Codex CLI](https://github.com/openai/codex) is an OpenAI coding agent. For ACP clients, use the [Codex ACP adapter](https://github.com/zed-industries/codex-acp).

```bash
OPENAI_API_KEY=... \
ACP_AGENT_COMMAND="npx @zed-industries/codex-acp" \
uv run acp-bot
```

You can also install the adapter binary from [adapter releases](https://github.com/zed-industries/codex-acp/releases) and use `codex-acp` directly.

## GitHub Copilot CLI (native ACP)

If you already have Copilot CLI installed and authenticated, run:

```bash
ACP_AGENT_COMMAND="copilot --acp" \
uv run acp-bot
```

## Claude Agent / Claude Code (via adapter)

Claude integration for ACP clients is provided by [`@zed-industries/claude-agent-acp`](https://github.com/zed-industries/claude-agent-acp).

```bash
ANTHROPIC_API_KEY=... \
ACP_AGENT_COMMAND="npx @zed-industries/claude-agent-acp" \
uv run acp-bot
```

If you prefer global install:

```bash
npm install -g @zed-industries/claude-agent-acp
ANTHROPIC_API_KEY=... ACP_AGENT_COMMAND="claude-agent-acp" uv run acp-bot
```

## Other agents

For other ACP-compatible agents, check:

- [ACP agents list](https://agentclientprotocol.com/get-started/agents)
- [ACP registry](https://agentclientprotocol.com/registry/index)

When trying a new agent, start with:

1. `ACP_LOG_LEVEL=DEBUG`
2. A short {term}`ACP_CONNECT_TIMEOUT` (for example `15`)
3. A command that is known to run non-interactively on stdio

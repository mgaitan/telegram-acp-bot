# Telegram ACP bot

[![CI](https://github.com/mgaitan/telegram-acp-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/mgaitan/telegram-acp-bot/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-blue.svg?style=flat)](https://mgaitan.github.io/telegram-acp-bot/)
[![GitHub](https://img.shields.io/badge/GitHub-repository-181717?logo=github)](https://github.com/mgaitan/telegram-acp-bot)
[![pypi version](https://img.shields.io/pypi/v/telegram-acp-bot?logo=pypi&logoColor=white)](https://pypi.org/project/telegram-acp-bot/)
[![Changelog](https://img.shields.io/github/v/release/mgaitan/telegram-acp-bot?include_prereleases&label=changelog)](https://github.com/mgaitan/telegram-acp-bot/releases)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/mgaitan/telegram-acp-bot/actions/workflows/ci.yml)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mgaitan/telegram-acp-bot/blob/main/LICENSE)

```{raw} html
<style>
.hero {
  display: grid;
  gap: 2rem;
  align-items: center;
  margin: 2rem 0 1.5rem;
}

.hero-copy h2 {
  margin: 0 0 0.75rem;
  font-size: clamp(1.7rem, 2.8vw, 2.5rem);
  line-height: 1.15;
}

.hero-copy p {
  margin: 0;
  font-size: 1.05rem;
}

.phone {
  width: min(100%, 380px);
  margin: 0 auto;
  padding: 12px;
  border-radius: 36px;
  background: linear-gradient(145deg, #141b26, #2b3748);
  box-shadow: 0 20px 40px rgb(0 0 0 / 20%);
}

.phone iframe {
  display: block;
  width: 100%;
  aspect-ratio: 9 / 16;
  border: 0;
  border-radius: 24px;
}

@media (min-width: 860px) {
  .hero {
    grid-template-columns: 1.2fr 0.8fr;
  }

  .phone {
    margin: 0 0 0 auto;
  }
}

@media (prefers-color-scheme: dark) {
  .phone {
    background: linear-gradient(145deg, #0f1520, #263445);
  }
}
</style>

<section class="hero">
  <div class="hero-copy">
    <h2>The agent works on your computer. You control it from Telegram.</h2>
    <p>
      A Telegram bot that brings the <a href="https://agentclientprotocol.com/" target="_blank" rel="noreferrer noopener">Agent Client Protocol (ACP)</a>
      to your pocket.
    </p>
  </div>
  <div class="phone" aria-label="Demo video in a phone frame">
    <iframe
      src="https://www.youtube.com/embed/QvLoZkhAbqA"
      title="Telegram ACP bot demo"
      loading="lazy"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
      allowfullscreen
    ></iframe>
  </div>
</section>
```

Software is changing incredibly fast, and we no longer need to write every line of code by hand. To interact with an agent, a simple and friendly chat interface is enough, and you already carry one on your phone.

ACP is the open protocol adopted by the industry to connect user interfaces to AI agents such as Codex, Claude Code, Gemini CLI, and [many others](https://agentclientprotocol.com/registry/index) that work on your machine.

Telegram ACP bot is a full-featured, [open-source](https://github.com/mgaitan/telegram-acp-bot) ACP client that allows you to control your agent wherever you are. The good part? You can start an agent session on your computer from your preferred editor/CLI/IDE and continue it later on Telegram, or do it the other way around.

**Don't watch the computer. Watch the sky.**

Start with:

```bash
TELEGRAM_BOT_TOKEN=123456:abc ACP_AGENT_COMMAND="your acp capable agent" uvx telegram-acp-bot
```

```{toctree}
:maxdepth: 2
:caption: Documentation

how-to.md
agents.md
cli.md
mcp.md
configuration.md
demo.md
about_the_docs.md
../CONTRIBUTING.md
../CODE_OF_CONDUCT.md
```

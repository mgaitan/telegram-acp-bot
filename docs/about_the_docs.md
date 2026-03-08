# About this documentation

This documentation is built using [Sphinx](https://www.sphinx-doc.org/)
with [myst-parser](https://myst-parser.readthedocs.io/).

The theme used is
[sphinx-book-theme](https://sphinx-book-theme.readthedocs.io/en/stable/).

## How to contribute

The documentation is written in [myst-parser](https://myst-parser.readthedocs.io/en/latest/syntax/typography.html) Markdown.

The myst extensions
[colon_fences](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#code-fences-using-colons),
[linkify](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#linkify)
and [deflist](https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#definition-lists) are enabled and you can also use the extra [content blocks](https://sphinx-book-theme.readthedocs.io/en/stable/content/content-blocks.html)
from our theme.

In addition, you can use all the [directives available in Sphinx](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html),
as explained in [this guide](https://myst-parser.readthedocs.io/en/v4.0.0/using/intro.html#intro-writing).

We also ship [richterm](https://github.com/mgaitan/richterm) to capture CLI output as SVG in the docs.

## Including diagrams

We support [mermaid diagrams](https://mermaid.js.org/], powered [sphinxcontrib.mermaid](https://github.com/mgaitan/sphinxcontrib-mermaid):

```{mermaid}
:align: center
graph LR;
    Hi-->there;
```

with this syntax:

````markdown
```{mermaid}
:align: center
graph LR;
  Hi-->there;
```
````

## Linking to the repo

There is shortcut to add links to the monorepo at Github, by prefixing `gh:` plus the
relative path. For example:

```markdown
[ci workflow](gh:.github/workflows/ci.yml)
```

Produces this link: [e2e workflow](gh:.github/workflows/ci.yml)

Check `myst_url_schemes` at [docs/conf.py](gh:docs/conf.py) for details on how it's implemented.


## How to build the documentation

From the root directory run

```bash
$ make docs
```

This will run `sphinx-build` using `uv run` with the docs requirements.

It should exit without error nor warnings.

If you want to check everything looks ok, open the generated html in the browser.

```bash
$ make docs-open
````

Also you can build the documentation in `epub` with `make docs-epub`

## How the documentation is published online

- GitHub Actions workflow: `[](gh:.github/workflows/cd.yml)` publishes docs to GitHub Pages.
- Triggers:
  - On releases (PyPI publish + docs deploy).
  - Manual via `workflow_dispatch` (used for the initial docs build and any ad-hoc redeploy).
- To trigger manually from your repo: `gh workflow run cd.yml --ref main` (requires `gh` CLI auth) or use the Actions UI.

## Demo capture workflow (maintainers)

For landing-page demo assets we keep a reproducible Telegram Web recording flow.

### 1. Configure `.env`

Reuse your local `.env`:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ALLOWED_USER_IDS` (or `TELEGRAM_ALLOWED_USERNAMES`)
- `TELEGRAM_DEMO_BOT_USERNAME` (or `TELEGRAM_BOT_USERNAME`)

### 2. Install browser runtime (one-time)

```bash
uv run --group playwright playwright install chromium
```

### 3. Optional: run scripted bot backend (fake ACP agent)

Use the helper script that runs `TelegramBridge` with `AcpAgentService` against a fake ACP agent process:

```bash
uv run scripts/demo/run_demo_bot.py
```

This helper uses scripted responses with slight randomized timing and always uses `scripts/demo/fake_acp_agent.py` unless you override
`--agent-command` explicitly.

Both the fake agent and Telegram Web demo script consume the same declarative story file:

- `scripts/demo/demo_story.json`

### 4. Login once via QR and persist session

```bash
uv run --group playwright python scripts/demo/telegram_web_demo.py --mode login
```

This stores Telegram Web session data under `.cache/telegram-web-profile`.

### 5. Record scripted interaction (vertical format)

```bash
uv run --group playwright python scripts/demo/telegram_web_demo.py --mode record
```

The recorder uses an iPhone-like viewport (`390x780`) and records video at the same size (`390x780`).
It stores `.webm` files under `artifacts/demo-videos`.
The dialogue timing and payloads come from `scripts/demo/demo_story.json` (including the synthetic image/PDF assets,
with `scripts/demo/demo.png` used as the demo image payload).
To pass recorder-specific flags, pass them directly:

```bash
uv run --group playwright python scripts/demo/telegram_web_demo.py --mode record --device-scale-factor 1.0
```

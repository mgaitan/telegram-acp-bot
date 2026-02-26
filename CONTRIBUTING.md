# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

## Environment setup

Nothing easier!

Fork and clone the repository, then:

```bash
cd telegram-acp-bot
uv sync
```

:::{note}
You'll need to install [uv](https://github.com/astral-sh/uv).

In Linux or macOS, you can install it with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

In Windows, you can install it with:
```bash
curl -LsSf https://astral.sh/uv/install.ps1 | powershell
```
:::

You now have the dependencies installed.

You can run the application with `uv run acp-bot [ARGS...]`.

## Development

As usual:

1. create a new branch: `git switch -c feature-or-bugfix-name`
1. edit the code and/or the documentation

**Before committing:**


1. run `uv run pytest` to run the tests (fix any issue)
1. run `uv run ruff check --fix` to check everything (fix any warning)
1. run `uv run ruff format` to auto-format the code
1. if you updated the documentation or the project dependencies:
    1. run `make docs` and check that everything looks good

If you are unsure about how to fix or ignore a warning, just let the continuous integration fail, and we will help you during review.

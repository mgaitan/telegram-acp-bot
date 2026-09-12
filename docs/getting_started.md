# Getting Started (Tutorial)

This tutorial gives you a complete first pass through local setup, checks, and docs.

## 1. Create the environment

From the project root:

```bash
uv sync
```

This resolves dependencies and creates the local virtual environment.

## 2. Run the CLI from source

If this project includes a CLI command:

```bash
uv run telegram-acp-bot --help
```

When invoking modules directly from source, set {term}`PYTHONPATH` so imports resolve cleanly:

```bash
PYTHONPATH=src uv run -m telegram_acp_bot --help
```
## 3. Run quality checks

```bash
make qa
make test
```

If `prek` is installed, `make qa` runs the local QA bundle with hooks.

## 4. Build the documentation

```bash
make docs
```

To open generated HTML:

```bash
make docs-open
```

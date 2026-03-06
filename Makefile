.PHONY: install
install: ## Install the virtual environment and git hooks via prek
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@if command -v prek >/dev/null 2>&1; then \
		echo "🪝 Installing git hooks with prek"; \
		prek install; \
	else \
		echo "ℹ️  prek is not installed. Install it with: uv tool install prek"; \
	fi

.PHONY: test qa
test: ## Run tests with coverage
	@echo "🧪 Running tests with coverage"
	@uv run pytest

qa: ## Run local QA checks via prek
	@if command -v prek >/dev/null 2>&1; then \
		echo "🔍 Running prek checks"; \
		prek run --all-files; \
	else \
		echo "ℹ️  prek is not installed. Falling back to direct QA commands. Install it with: uv tool install prek"; \
		set -e; \
		uv run --only-group lint ruff check; \
		uv run --only-group lint ruff format --check; \
		uv run --group qa ty check; \
	fi

.PHONY: bump
bump:
	uv version --bump minor

.PHONY: release
release: ## Create a GitHub release for the current version
	@version=$$(uv version --short); \
	git commit -am "Bump $$version"; \
	git push origin main; \
	gh release create "$$version" --generate-notes

.PHONY: docs docs-html docs-epub docs-open html epub open

DOCS_SOURCE := docs
DOCS_BUILD := $(DOCS_SOURCE)/_build

docs: ## Build HTML documentation (default).
	@$(MAKE) docs-html

html: docs-html

docs-html: ## Build documentation as static HTML.
	@echo "📖 Building HTML documentation"
	@uv run --group docs sphinx-build $(DOCS_SOURCE) $(DOCS_BUILD)/html -b html -W

epub: docs-epub

docs-epub: ## Build documentation as EPUB.
	@echo "📖 Building EPUB documentation"
	@uv run --group docs sphinx-build $(DOCS_SOURCE) $(DOCS_BUILD)/epub -b epub -W

docs-open open: docs-html ## Build docs and open them in the browser.
	@uv run -m webbrowser docs/_build/html/index.html

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_.-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help

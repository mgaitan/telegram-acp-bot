Project: Telegram ACP bot

Repository: https://github.com/mgaitan/telegram-acp-bot

Description: A Telegram bot that implements Agent Client Protocol to interact with AI agents

## Stack

- **Python:** Python 3.14. Check main dependencies in `pyproject.toml`.
- **Tests:** pytest, factory-boy, pytest-mock. `make test` to run tests. or `uv run pytest`
- **Productivity:** Dependencies managed with `uv` via `pyproject.toml`;
- **lint/format:** Ruff.
- **Git:** and Github

## Python preferences

- Modern and idiomatic practices that emphasize clarity and predictable behavior. Examples of modern features:
   - Pathlib for file operations
   - Data model methods (like __len__, __add__, etc.)
   - Stdlib or pydantic dataclasses
   - Advanced itertools
   - Pattern matching
   - Walrus operator
   - Enums subclasses
- Dependency changes use `uv add` or `uv remove`
- Use `uv run python - <<'PY' ...` instead `python - <<'PY'` for inline scripts
- Docstrings in Markdown ("myst") format, expressing intentions rather than implementation details.
  Make references to other code if appropriate. Eg: "See also `{py:func}`other_module.helper_function`.".
- Explicit and robust type annotations using built-in generics (`list`, `dict`, etc.), union types with `|`, etc.
- Prefer flat code: use early returns, guard clauses, fixtures over context managers on tests, etc.
- Never hallucinate APIs or behaviours. If uncertain, inspect the code and/or check online documentation (ensure it's the correct version declared by uv.lock) or ask the developer

## Git/Github preferences

- Ensure you are in a proper branch for each new feature or bugfix.
- Never commit or push automatically unless instructed otherwise.
- Prefer `gh` CLI for all interactions with Github if possible. Eg. Use it to open PRs / manage issues.
- Consider mentioning yourself in the commits as co-author if you helped enough. Use the standard Git co-author trailer:
  "Co-authored-by: Name <email>".

## Documentation

- Documentation sources are in `docs/` using Sphinx and Markdown with myst-parser.  Run `make docs` to build them.
- Provide documentation for:
  - **Users:** intention, behaviour, minimal examples.
  - **Contributors (humans + agents):** technical details, internal references, short code explanations.
- Any feature or behaviour change must include updated documentation, just like tests.
- Follow the style in the same file or directory: match tone, headings, formatting, and Markdown conventions unless instructed otherwise.
- Use internal references (add anchors if needed) to link across relation sections.
- If useful, maintain a glossary with key domain definitions.
- sphinx-mermaid is available: create diagrams whenever they help understanding.

## Language preferences

- All committable text must be in simple English (including documentation, code comments, docstrings, commit messages, PR descriptions, etc.).
- However, when interacting with the developer in chat, respond in the language they use.
- Avoid sexist or exclusionary language. Always prefer gender-neutral phrasing.

Project: Telegram ACP bot

Repository: https://github.com/mgaitan/telegram-acp-bot

Description: A Telegram bot that implements Agent Client Protocol to interact with AI agents

## Stack

- **Python:** Python 3.14. Check main dependencies in `pyproject.toml`.
- **Tests:** pytest, factory-boy, pytest-mock. `make test` to run tests. or `uv run pytest`
- **Productivity:** Dependencies managed with `uv` via `pyproject.toml`;
- **lint/format:** Ruff.
- **Git:** and GitHub

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
- In tests (`tests/**/*.py`), avoid adding a return annotation (`-> None`) to `test_*` functions.
  Ruff `ANN` rules are intentionally ignored there.
- Validate type safety with `uv run ty check` after relevant code changes.
- Prefer flat code: use early returns, guard clauses, fixtures over context managers on tests, etc.
- Never hallucinate APIs or behaviours. If uncertain, inspect the code and/or check online documentation (ensure it's the correct version declared by uv.lock) or ask the developer

## Git/GitHub preferences

- Ensure you are in a proper branch for each new feature or bugfix.
- Never commit or push automatically unless instructed otherwise.
- Never merge your own PRs without explicit maintainer approval in chat, even if CI is green.
- Prefer `gh` CLI for all interactions with GitHub if possible. Eg. Use it to open PRs / manage issues.
- For `gh pr` interactions, prefer `--body-file` with a temporary file created in the system temp directory (for example via `mktemp` or `$TMPDIR`), and delete the file after the command completes.
- For issue categorization, use GitHub labels instead of title prefixes like `Bug:`, `Feat:`, or `Docs:`.
- Before assigning labels, inspect labels already used in the target repository and follow that taxonomy first.
- If labels are missing or incomplete, use this baseline set: `bug`, `enhancement`, `documentation`, `maintenance`, `refactor`, `test`, `ci`, `dependencies`, `security`, `performance`, `ux`, `question`.
- Consider mentioning yourself in the commits as co-author if you helped enough. Use the standard Git co-author trailer:
  "Co-authored-by: Name <email>".
  For common AI collaborators, use:
  - "Co-authored-by: codex <codex@openai.com>".
  - "Co-authored-by: Claude <noreply@anthropic.com>".

## Documentation

- Documentation sources are in `docs/` using Sphinx and Markdown with myst-parser.  Run `make docs` to build them.
- Organize docs using [Diataxis](https://diataxis.fr/): tutorial, how-to, reference, and explanation.
- Use `docs/about_the_docs.md` as the canonical chapter for documentation principles and design rationale.
- Keep env var definitions in `docs/configuration.md` using the Sphinx `glossary` directive.
- When mentioning an env var in docs, reference it as a glossary term (`{term}`).
- Documentation previews for PRs with docs changes are published under `https://mgaitan.github.io/telegram-acp-bot/_preview/pr-<PR_NUMBER>/`.
- Release/manual docs publish uses the canonical site URL: `https://mgaitan.github.io/telegram-acp-bot/`.
- Provide documentation for:
  - **Users:** intention, behaviour, minimal examples.
  - **Contributors (humans + agents):** technical details, internal references, short code explanations.
- Any feature or behaviour change must include updated documentation, just like tests.
- Follow the style in the same file or directory: match tone, headings, formatting, and Markdown conventions unless instructed otherwise.
- Use internal references (add anchors if needed) to link across relation sections.
- If useful, maintain a glossary with key domain definitions.
- sphinx-mermaid is available: create diagrams whenever they help understanding.

## Language preferences

- The public language is English: all committable text and GitHub interactions must be in simple English (including documentation, comments, docstrings, commit messages, PR descriptions, etc.).
- However, when interacting with the developer in chat, respond in the language they use.
- Avoid sexist or exclusionary language. Always prefer gender-neutral phrasing.

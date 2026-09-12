# Development Workflow (How-to)

This guide covers the most common maintenance tasks.

## Use the pinned Python

The generated `.python-version` records the exact Python interpreter that ran
Copier. `uv` uses this file automatically, and template updates preserve it.

## Run local QA before pushing

```bash
make qa
make test
```

CI runs workflow linting, QA, and tests for every pull request and push to
`main`; Markdown-only changes are not skipped.

If `prek` is not installed, install it once with:

```bash
uv tool install prek
```

## Update dependencies safely

This template uses `uv` groups and cooldown windows.
After dependency changes:

```bash
uv sync
make qa
make test
```

## Publish a release

```bash
make bump
make release
```

Release workflows publish package artifacts and canonical docs.

### Verify release attestations

Each wheel and source distribution has a PEP 740 publish attestation signed
through Sigstore and uploaded to PyPI with the package.

Verify a published artifact by passing its PyPI file URL:

```bash
uvx pypi-attestations verify pypi \
  --repository https://github.com/mgaitan/telegram-acp-bot \
  https://files.pythonhosted.org/path/to/distribution.whl
```

## Preview documentation in pull requests

Pull requests build the documentation in CI but do not publish a preview
automatically. To publish a preview manually, dispatch the workflow with the
pull request number:

```bash
gh workflow run cd.yml --ref main \
  -f deploy_target=preview \
  -f pr_number=<PR_NUMBER>
```

The preview is published under:

```bash
https://mgaitan.github.io/telegram-acp-bot/_preview/
```

`gh` authentication can use {term}`GH_TOKEN`.

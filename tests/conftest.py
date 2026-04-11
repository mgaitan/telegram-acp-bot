"""Configuration for the pytest test suite."""

import pytest


@pytest.fixture(autouse=True)
def isolate_home_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Redirect HOME and XDG_CONFIG_HOME to a temporary directory.

    Prevents `_find_config_file()` from accidentally discovering real config
    files in the developer's home directory, keeping tests hermetic.
    The CWD-relative candidate (`.telegram_acp_bot/config.json`) is unaffected,
    so tests that exercise auto-discovery can still do so by creating the file
    under their own `tmp_path` and changing into it.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))

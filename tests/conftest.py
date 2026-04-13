"""Configuration for the pytest test suite."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_home_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect HOME and XDG_CONFIG_HOME to a temporary directory.

    Prevents `_find_config_file()` from accidentally discovering real config
    files in the developer's home directory, keeping tests hermetic.
    The CWD-relative candidate (`.telegram_acp_bot/config.json`) is unaffected,
    so tests that exercise auto-discovery can still do so by creating the file
    under their own `tmp_path` and changing into it.
    """
    home_dir = tmp_path / "home"
    xdg_config_home = tmp_path / "xdg-config"
    home_dir.mkdir()
    xdg_config_home.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))

from __future__ import annotations

import telegram_acp_bot.acp.protocols as protocols_module


def test_package_version_uses_installed_distribution(mocker):
    mocker.patch.object(protocols_module.metadata, "version", return_value="0.2.0")

    assert protocols_module._package_version() == "0.2.0"


def test_package_version_returns_unknown_when_distribution_is_missing(mocker):
    mocker.patch.object(
        protocols_module.metadata,
        "version",
        side_effect=protocols_module.metadata.PackageNotFoundError("telegram-acp-bot"),
    )

    assert protocols_module._package_version() == "unknown"

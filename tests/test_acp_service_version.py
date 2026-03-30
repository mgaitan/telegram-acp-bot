from __future__ import annotations

import telegram_acp_bot.acp_app.acp_service as acp_service_module


def test_package_version_uses_installed_distribution(mocker):
    mocker.patch.object(acp_service_module.metadata, "version", return_value="0.2.0")

    assert acp_service_module._package_version() == "0.2.0"


def test_package_version_returns_unknown_when_distribution_is_missing(mocker):
    mocker.patch.object(
        acp_service_module.metadata,
        "version",
        side_effect=acp_service_module.metadata.PackageNotFoundError("telegram-acp-bot"),
    )

    assert acp_service_module._package_version() == "unknown"

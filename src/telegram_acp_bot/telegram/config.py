"""Runtime configuration for the Telegram transport layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from telegram_acp_bot.acp.models import ActivityMode


@dataclass(slots=True, frozen=True)
class BotConfig:
    """Runtime settings for Telegram transport."""

    token: str
    allowed_user_ids: set[int]
    allowed_usernames: set[str]
    default_workspace: Path
    activity_mode: ActivityMode = "normal"
    schedule_languages: tuple[str, ...] = ("en", "es")

    @property
    def compact_activity(self) -> bool:
        return self.activity_mode == "compact"


def make_config(  # noqa: PLR0913
    *,
    token: str,
    allowed_user_ids: list[int],
    workspace: str,
    allowed_usernames: list[str] | None = None,
    activity_mode: ActivityMode = "normal",
    compact_activity: bool | None = None,
    schedule_languages: list[str] | None = None,
) -> BotConfig:
    normalized_usernames = {
        username.lstrip("@").strip().lower() for username in (allowed_usernames or []) if username.strip()
    }
    normalized_schedule_languages = tuple(
        language.strip().lower() for language in (schedule_languages or ["en", "es"]) if language.strip()
    ) or ("en", "es")
    if compact_activity is not None:
        activity_mode = "compact" if compact_activity else "normal"
    return BotConfig(
        token=token,
        allowed_user_ids=set(allowed_user_ids),
        allowed_usernames=normalized_usernames,
        default_workspace=Path(workspace).expanduser(),
        activity_mode=activity_mode,
        schedule_languages=normalized_schedule_languages,
    )

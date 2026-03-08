from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

STEP_KIND_SEND_NOW = "click_send_now"
STEP_KIND_RESUME_CHOICE = "click_resume_choice"
ASSET_KIND_IMAGE = "image"
ASSET_KIND_FILE = "file"

MISSING_ROUTE_MESSAGE = "Scenario route requires at least one `match_any` keyword."
UNKNOWN_ACTION_MESSAGE = "Unknown user action type in scenario."
UNKNOWN_ASSET_KIND_MESSAGE = "Unknown asset kind in scenario."


@dataclass(slots=True, frozen=True)
class SendNowAction:
    type: Literal["click_send_now"]
    tap_marker: bool = True


@dataclass(slots=True, frozen=True)
class ResumeChoiceAction:
    type: Literal["click_resume_choice"]
    index: int = 0


@dataclass(slots=True, frozen=True)
class WaitForText:
    pattern: str
    timeout_seconds: float = 20.0
    after_ms: int = 0


@dataclass(slots=True, frozen=True)
class UserStep:
    id: str
    text: str
    wait_for_text: WaitForText | None = None
    actions: tuple[SendNowAction | ResumeChoiceAction, ...] = ()


@dataclass(slots=True, frozen=True)
class AgentEvent:
    kind: str
    title: str
    text: str = ""
    delay_ms: int = 0


@dataclass(slots=True, frozen=True)
class AgentRoute:
    id: str
    match_any: tuple[str, ...]
    events: tuple[AgentEvent, ...]
    final_text: str
    final_images: tuple[str, ...] = ()
    final_files: tuple[str, ...] = ()
    cancel_reply_text: str = ""


@dataclass(slots=True, frozen=True)
class ScenarioAsset:
    id: str
    kind: Literal["image", "file"]
    path: Path
    mime_type: str
    name: str

    def data_base64(self) -> str:
        return base64.b64encode(self.path.read_bytes()).decode("ascii")


@dataclass(slots=True, frozen=True)
class DemoRuntime:
    typing_delay_ms: int = 40
    pause_min_seconds: float = 1.1
    pause_max_seconds: float = 3.1
    final_pause_seconds: float = 3.2
    image_reply_pattern: str = r"captured\\s+`?/tmp/webcam-small\\.jpg`?"


@dataclass(slots=True, frozen=True)
class DemoScenario:
    runtime: DemoRuntime
    user_steps: tuple[UserStep, ...]
    agent_routes: tuple[AgentRoute, ...]
    assets: dict[str, ScenarioAsset]

    def route_for_prompt(self, text: str) -> AgentRoute | None:
        lowered = text.lower()
        best_route: AgentRoute | None = None
        best_score = 0
        for route in self.agent_routes:
            score = sum(1 for keyword in route.match_any if keyword.lower() in lowered)
            if score > best_score:
                best_score = score
                best_route = route
        return best_route


DEFAULT_SCENARIO_PATH = Path(__file__).with_name("demo_story.json")


def _parse_wait_for_text(raw: dict[str, Any] | None) -> WaitForText | None:
    if raw is None:
        return None
    pattern = str(raw.get("pattern", "")).strip()
    if not pattern:
        return None
    return WaitForText(
        pattern=pattern,
        timeout_seconds=float(raw.get("timeout_seconds", 20.0)),
        after_ms=int(raw.get("after_ms", 0)),
    )


def _parse_user_actions(raw_actions: list[dict[str, Any]] | None) -> tuple[SendNowAction | ResumeChoiceAction, ...]:
    if not raw_actions:
        return ()
    actions: list[SendNowAction | ResumeChoiceAction] = []
    for raw in raw_actions:
        action_type = str(raw.get("type", "")).strip()
        if action_type == STEP_KIND_SEND_NOW:
            actions.append(SendNowAction(type=STEP_KIND_SEND_NOW, tap_marker=bool(raw.get("tap_marker", True))))
            continue
        if action_type == STEP_KIND_RESUME_CHOICE:
            actions.append(ResumeChoiceAction(type=STEP_KIND_RESUME_CHOICE, index=int(raw.get("index", 0))))
            continue
        raise ValueError(UNKNOWN_ACTION_MESSAGE)
    return tuple(actions)


def _parse_user_steps(raw_steps: list[dict[str, Any]]) -> tuple[UserStep, ...]:
    return tuple(
        UserStep(
            id=str(raw["id"]),
            text=str(raw["text"]),
            wait_for_text=_parse_wait_for_text(raw.get("wait_for_text")),
            actions=_parse_user_actions(raw.get("actions")),
        )
        for raw in raw_steps
    )


def _parse_agent_routes(raw_routes: list[dict[str, Any]]) -> tuple[AgentRoute, ...]:
    routes: list[AgentRoute] = []
    for raw in raw_routes:
        keywords = tuple(str(item) for item in raw.get("match_any", []) if str(item).strip())
        if not keywords:
            raise ValueError(MISSING_ROUTE_MESSAGE)
        events = tuple(
            AgentEvent(
                kind=str(item["kind"]),
                title=str(item.get("title", "")),
                text=str(item.get("text", "")),
                delay_ms=int(item.get("delay_ms", 0)),
            )
            for item in raw.get("events", [])
        )
        routes.append(
            AgentRoute(
                id=str(raw["id"]),
                match_any=keywords,
                events=events,
                final_text=str(raw.get("final_text", "")),
                final_images=tuple(str(item) for item in raw.get("final_images", [])),
                final_files=tuple(str(item) for item in raw.get("final_files", [])),
                cancel_reply_text=str(raw.get("cancel_reply_text", "")),
            )
        )
    return tuple(routes)


def _parse_assets(base_dir: Path, raw_assets: dict[str, dict[str, Any]]) -> dict[str, ScenarioAsset]:
    assets: dict[str, ScenarioAsset] = {}
    for asset_id, raw in raw_assets.items():
        kind = str(raw.get("kind", "")).strip().lower()
        if kind not in {ASSET_KIND_IMAGE, ASSET_KIND_FILE}:
            raise ValueError(UNKNOWN_ASSET_KIND_MESSAGE)
        normalized_kind = cast(Literal["image", "file"], kind)
        relative_path = Path(str(raw["path"]))
        asset_path = (base_dir / relative_path).resolve()
        assets[asset_id] = ScenarioAsset(
            id=asset_id,
            kind=normalized_kind,
            path=asset_path,
            mime_type=str(raw["mime_type"]),
            name=str(raw.get("name", asset_path.name)),
        )
    return assets


def load_demo_scenario(path: Path | None = None) -> DemoScenario:
    scenario_path = (path or DEFAULT_SCENARIO_PATH).resolve()
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    base_dir = scenario_path.parent
    runtime_payload = payload.get("runtime", {})
    return DemoScenario(
        runtime=DemoRuntime(
            typing_delay_ms=int(runtime_payload.get("typing_delay_ms", 40)),
            pause_min_seconds=float(runtime_payload.get("pause_min_seconds", 1.1)),
            pause_max_seconds=float(runtime_payload.get("pause_max_seconds", 3.1)),
            final_pause_seconds=float(runtime_payload.get("final_pause_seconds", 3.2)),
            image_reply_pattern=str(
                runtime_payload.get("image_reply_pattern", r"captured\\s+`?/tmp/webcam-small\\.jpg`?")
            ),
        ),
        user_steps=_parse_user_steps(payload["user_steps"]),
        agent_routes=_parse_agent_routes(payload["agent_routes"]),
        assets=_parse_assets(base_dir, payload.get("assets", {})),
    )

"""Configuration model and loader for the configurable fake ACP agent.

Parses a YAML file describing conditional response rules.
Each rule has an optional `when` condition and a `then` sequence of steps.
Rules are evaluated in order; the first matching rule is executed.
A rule without a `when` key acts as the fallback default.

## YAML format

```yaml
version: 1

defaults:
  variables:
    threshold: 10

rules:
  - name: greeting
    when:
      message_regex: "(?i)hello"
    then:
      - type: thought
        content: "The user is greeting us."
      - type: final
        content: "Hello! How can I help you?"

  - name: fallback
    then:
      - type: final
        content: "I don't understand: {{ message }}"
```

## Step types

- `thought` — emits a *think* tool event (visible as a collapsible block).
- `tool_call` — opens an *execute* tool event showing `tool_name` + `args`.
- `tool_result` — completes the last open tool event with `content`.
- `final` — emits the final reply text.

## Template variables

Content fields support `{{ message }}` and `{{ variables.<key> }}` placeholders,
which are resolved at render time against the incoming message and the configured
default variables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

StepType = Literal["thought", "final", "tool_call", "tool_result"]

UNKNOWN_STEP_TYPE_MESSAGE = "Unknown step type"
MISSING_RULE_STEPS_MESSAGE = "Rule has no steps in 'then'"


@dataclass(slots=True, frozen=True)
class NumericGtCondition:
    """Condition that passes when the numeric value of *value_from* exceeds *gt*."""

    value_from: str  # currently only "message" is supported
    gt: float


@dataclass(slots=True, frozen=True)
class Condition:
    """Combined condition for a rule's `when` clause.

    All non-`None` sub-conditions must pass for the rule to match.
    """

    message_regex: str | None = None
    numeric_gt: NumericGtCondition | None = None

    def matches(self, message: str) -> bool:
        """Return `True` when every active sub-condition passes."""
        if self.message_regex is not None and not re.search(self.message_regex, message):
            return False
        if self.numeric_gt is not None:
            try:
                value = float(message.strip())
            except ValueError:
                return False
            if value <= self.numeric_gt.gt:
                return False
        return True


@dataclass(slots=True, frozen=True)
class Step:
    """A single action in a rule's `then` sequence.

    See `rule_config` module for a description of each step type.
    """

    type: StepType
    content: str = ""
    tool_name: str = ""
    args: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Rule:
    """A conditional response rule.

    A rule without a `condition` (i.e. no `when` key in YAML) acts as the
    unconditional fallback and always matches.
    """

    name: str
    condition: Condition | None  # None → fallback (always matches)
    steps: tuple[Step, ...]

    def matches(self, message: str) -> bool:
        """Return `True` when this rule should be executed for *message*."""
        if self.condition is None:
            return True
        return self.condition.matches(message)


@dataclass(slots=True, frozen=True)
class RuleConfig:
    """Top-level configuration loaded from a YAML file.

    See also `load_rule_config`.
    """

    version: int
    variables: dict[str, Any]
    rules: tuple[Rule, ...]

    def find_rule(self, message: str) -> Rule | None:
        """Return the first rule that matches *message*, or `None`."""
        for rule in self.rules:
            if rule.matches(message):
                return rule
        return None


def render_template(text: str, *, message: str, variables: dict[str, Any]) -> str:
    """Substitute `{{ message }}` and `{{ variables.<key> }}` in *text*.

    Unknown variable references are left as-is.
    """
    result = re.sub(r"\{\{\s*message\s*\}\}", lambda _: message, text)

    def _var_sub(m: re.Match[str]) -> str:
        key = m.group(1).strip()
        return str(variables.get(key, m.group(0)))

    return re.sub(r"\{\{\s*variables\.(\w+)\s*\}\}", _var_sub, result)


def render_step(step: Step, *, message: str, variables: dict[str, Any]) -> Step:
    """Return a new `Step` with template placeholders resolved."""
    rendered_content = render_template(step.content, message=message, variables=variables)
    rendered_args = {k: render_template(v, message=message, variables=variables) for k, v in step.args.items()}
    return Step(type=step.type, content=rendered_content, tool_name=step.tool_name, args=rendered_args)


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------


def _parse_step(raw: dict[str, Any]) -> Step:
    step_type = str(raw.get("type", "")).strip()
    if step_type not in {"thought", "final", "tool_call", "tool_result"}:
        raise ValueError(UNKNOWN_STEP_TYPE_MESSAGE)
    return Step(
        type=step_type,  # type: ignore[arg-type]
        content=str(raw.get("content", "")),
        tool_name=str(raw.get("tool_name", "")),
        args={str(k): str(v) for k, v in raw.get("args", {}).items()},
    )


def _parse_condition(raw: dict[str, Any] | None) -> Condition | None:
    if raw is None:
        return None
    numeric_gt_raw = raw.get("numeric_gt")
    numeric_gt: NumericGtCondition | None = None
    if numeric_gt_raw is not None:
        numeric_gt = NumericGtCondition(
            value_from=str(numeric_gt_raw.get("value_from", "message")),
            gt=float(numeric_gt_raw.get("gt", 0)),
        )
    return Condition(
        message_regex=str(raw["message_regex"]) if "message_regex" in raw else None,
        numeric_gt=numeric_gt,
    )


def load_rule_config(path: Path) -> RuleConfig:
    """Parse *path* as a YAML rule configuration and return a `RuleConfig`.

    The file must be a valid YAML document matching the format described in
    this module's docstring.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    version = int(payload.get("version", 1))
    defaults = payload.get("defaults") or {}
    variables: dict[str, Any] = dict(defaults.get("variables") or {})

    rules: list[Rule] = []
    for raw_rule in payload.get("rules") or []:
        condition = _parse_condition(raw_rule.get("when"))
        raw_steps = raw_rule.get("then") or []
        if not raw_steps:
            raise ValueError(MISSING_RULE_STEPS_MESSAGE)
        steps = tuple(_parse_step(s) for s in raw_steps)
        rules.append(
            Rule(
                name=str(raw_rule.get("name", "")),
                condition=condition,
                steps=steps,
            )
        )
    return RuleConfig(version=version, variables=variables, rules=tuple(rules))

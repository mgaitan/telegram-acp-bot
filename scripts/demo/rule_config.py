"""Configuration model and loader for the configurable fake ACP agent.

Parses a YAML file describing conditional response rules.
Each rule has an optional `when` condition and a `then` sequence of steps.
Rules are evaluated in order; the first matching rule is executed.
A rule without a `when` key (or with `when: {default: true}`) acts as the
unconditional fallback.

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

  - name: search-flow
    when:
      message_contains: "search"
    set_variables:
      last_intent: search
    then:
      - type: tool_call
        tool_name: search
        args:
          query: "{{ message }}"
      - type: tool_result
        content: "Fake result for {{ message }}"
      - type: final
        content: "Done. Last intent was {{ state.last_intent }}."

  - name: big-number
    when:
      all:
        - message_regex: "^\\d+$"
        - numeric_gt:
            value_from: message
            gt: 100
    then:
      - type: final
        content: "Big number!"

  - name: pause-demo
    when:
      message_contains: "slow"
    then:
      - type: thought
        content: "Thinking…"
      - type: sleep
        sleep_ms: 1000
      - type: final
        content: "Done after a pause."

  - name: fallback
    then:
      - type: final
        content: "I don't understand: {{ message }}"
```

## Condition types

Flat conditions (all present keys must pass, acting as implicit AND):
- `message_regex` — regex search against the raw message.
- `message_contains` — simple substring test.
- `numeric_gt` — numeric value greater-than comparison.
- `numeric_lt` — numeric value less-than comparison.
- `default: true` — unconditional match (explicit fallback).

Compound conditions:
- `all` — list of flat conditions; all must pass.
- `any` — list of flat conditions; at least one must pass.

## Step types

- `thought` — emits a *think* tool event (visible as a collapsible block).
- `tool_call` — opens an *execute* tool event showing `tool_name` + `args`.
- `tool_result` — completes the last open tool event with `content`.
- `final` — emits the final reply text.
- `sleep` — waits for `sleep_ms` milliseconds (useful for pacing in demos).
- `error` — emits a tool event that finishes with *failed* status.

## Template variables

Content fields support `{{ message }}`, `{{ variables.<key> }}`, and
`{{ state.<key> }}` placeholders, resolved at render time.

`{{ state.<key> }}` exposes per-session state:
- `state.last_user_message` — the user message from the *previous* turn.
- `state.last_rule` — the rule name matched in the *previous* turn.
- Any key set via a rule's `set_variables` dictionary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, get_args

import yaml

StepType = Literal["thought", "final", "tool_call", "tool_result", "sleep", "error"]

UNKNOWN_STEP_TYPE_MESSAGE = "Unknown step type"
MISSING_RULE_STEPS_MESSAGE = "Rule has no steps in 'then'"

# Derived at import time from the Literal so additions only need a single source-of-truth edit.
_VALID_STEP_TYPES: frozenset[str] = frozenset(get_args(StepType))


@dataclass(slots=True, frozen=True)
class NumericGtCondition:
    """Condition that passes when the numeric value of *value_from* exceeds *gt*.

    `value_from` specifies the source to convert to a number.
    Currently only `"message"` (the raw user message text) is supported.
    """

    value_from: str = "message"
    gt: float = 0.0


@dataclass(slots=True, frozen=True)
class NumericLtCondition:
    """Condition that passes when the numeric value of *value_from* is less than *lt*.

    `value_from` specifies the source to convert to a number.
    Currently only `"message"` (the raw user message text) is supported.
    """

    value_from: str = "message"
    lt: float = 0.0


@dataclass(slots=True, frozen=True)
class Condition:
    """A rule's `when` clause.

    Flat keys form an implicit AND: every non-`None` check must pass.
    `all_of` / `any_of` allow explicit compound logic.
    `default=True` makes the condition always match (explicit fallback).
    """

    message_regex: str | None = None
    message_contains: str | None = None
    numeric_gt: NumericGtCondition | None = None
    numeric_lt: NumericLtCondition | None = None
    default: bool = False
    all_of: tuple[Condition, ...] = ()
    any_of: tuple[Condition, ...] = ()

    def matches(self, message: str) -> bool:
        """Return `True` when every active sub-condition passes."""
        if self.default:
            return True
        return (
            self._check_regex(message)
            and self._check_contains(message)
            and self._check_numeric_gt(message)
            and self._check_numeric_lt(message)
            and self._check_all_of(message)
            and self._check_any_of(message)
        )

    def _check_regex(self, message: str) -> bool:
        return self.message_regex is None or bool(re.search(self.message_regex, message))

    def _check_contains(self, message: str) -> bool:
        return self.message_contains is None or self.message_contains in message

    def _check_numeric_gt(self, message: str) -> bool:
        if self.numeric_gt is None:
            return True
        try:
            return float(message.strip()) > self.numeric_gt.gt
        except ValueError:
            return False

    def _check_numeric_lt(self, message: str) -> bool:
        if self.numeric_lt is None:
            return True
        try:
            return float(message.strip()) < self.numeric_lt.lt
        except ValueError:
            return False

    def _check_all_of(self, message: str) -> bool:
        return not self.all_of or all(c.matches(message) for c in self.all_of)

    def _check_any_of(self, message: str) -> bool:
        return not self.any_of or any(c.matches(message) for c in self.any_of)


@dataclass(slots=True, frozen=True)
class Step:
    """A single action in a rule's `then` sequence."""

    type: StepType
    content: str = ""
    tool_name: str = ""
    args: dict[str, str] = field(default_factory=dict)
    sleep_ms: int = 0  # used by type: sleep


@dataclass(slots=True, frozen=True)
class Rule:
    """A conditional response rule.

    A rule without a `condition` (i.e. no `when` key in YAML) acts as the
    unconditional fallback and always matches.
    """

    name: str
    condition: Condition | None  # None → fallback (always matches)
    steps: tuple[Step, ...]
    set_variables: dict[str, Any] = field(default_factory=dict)

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


def render_template(
    text: str,
    *,
    message: str,
    variables: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> str:
    """Substitute template placeholders in *text*.

    Supported placeholders:
    - `{{ message }}` — the current user message.
    - `{{ variables.<key> }}` — a config-level variable.
    - `{{ state.<key> }}` — a session-state value (optional).

    Unknown references are left unchanged.
    """
    result = re.sub(r"\{\{\s*message\s*\}\}", lambda _: message, text)

    def _var_sub(m: re.Match[str]) -> str:
        key = m.group(1).strip()
        return str(variables.get(key, m.group(0)))

    result = re.sub(r"\{\{\s*variables\.(\w+)\s*\}\}", _var_sub, result)

    if state:

        def _state_sub(m: re.Match[str]) -> str:
            key = m.group(1).strip()
            return str(state.get(key, m.group(0)))

        result = re.sub(r"\{\{\s*state\.(\w+)\s*\}\}", _state_sub, result)

    return result


def render_step(
    step: Step,
    *,
    message: str,
    variables: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> Step:
    """Return a new `Step` with template placeholders resolved."""
    rendered_content = render_template(step.content, message=message, variables=variables, state=state)
    rendered_args = {
        k: render_template(v, message=message, variables=variables, state=state) for k, v in step.args.items()
    }
    return Step(
        type=step.type,
        content=rendered_content,
        tool_name=step.tool_name,
        args=rendered_args,
        sleep_ms=step.sleep_ms,
    )


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------


def _parse_step(raw: dict[str, Any]) -> Step:
    step_type = str(raw.get("type", "")).strip()
    if step_type not in _VALID_STEP_TYPES:
        raise ValueError(UNKNOWN_STEP_TYPE_MESSAGE)
    return Step(
        type=cast(StepType, step_type),
        content=str(raw.get("content", "")),
        tool_name=str(raw.get("tool_name", "")),
        args={str(k): str(v) for k, v in raw.get("args", {}).items()},
        sleep_ms=int(raw.get("sleep_ms", 0)),
    )


def _parse_atomic_condition(raw: dict[str, Any]) -> Condition:
    """Parse a single flat (non-compound) condition dict."""
    numeric_gt_raw = raw.get("numeric_gt")
    numeric_gt: NumericGtCondition | None = None
    if numeric_gt_raw is not None:
        numeric_gt = NumericGtCondition(
            value_from=str(numeric_gt_raw.get("value_from", "message")),
            gt=float(numeric_gt_raw.get("gt", 0)),
        )
    numeric_lt_raw = raw.get("numeric_lt")
    numeric_lt: NumericLtCondition | None = None
    if numeric_lt_raw is not None:
        numeric_lt = NumericLtCondition(
            value_from=str(numeric_lt_raw.get("value_from", "message")),
            lt=float(numeric_lt_raw.get("lt", 0)),
        )
    return Condition(
        message_regex=str(raw["message_regex"]) if "message_regex" in raw else None,
        message_contains=str(raw["message_contains"]) if "message_contains" in raw else None,
        numeric_gt=numeric_gt,
        numeric_lt=numeric_lt,
        default=bool(raw.get("default", False)),
    )


def _parse_condition(raw: dict[str, Any] | None) -> Condition | None:
    if raw is None:
        return None
    all_raw: list[dict[str, Any]] | None = raw.get("all")
    any_raw: list[dict[str, Any]] | None = raw.get("any")
    if all_raw is not None or any_raw is not None:
        all_of = tuple(_parse_atomic_condition(item) for item in (all_raw or []))
        any_of = tuple(_parse_atomic_condition(item) for item in (any_raw or []))
        return Condition(all_of=all_of, any_of=any_of)
    return _parse_atomic_condition(raw)


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
        set_variables: dict[str, Any] = dict(raw_rule.get("set_variables") or {})
        rules.append(
            Rule(
                name=str(raw_rule.get("name", "")),
                condition=condition,
                steps=steps,
                set_variables=set_variables,
            )
        )
    return RuleConfig(version=version, variables=variables, rules=tuple(rules))

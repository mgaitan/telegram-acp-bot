"""Tests for scripts/demo/rule_config.py and configurable_fake_agent.py."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, cast

import pytest
from acp import RequestError, text_block
from acp.interfaces import Client
from acp.schema import AgentMessageChunk, ToolCallProgress, ToolCallStart
from configurable_fake_agent import ConfigurableFakeAcpAgent
from rule_config import (
    Condition,
    NumericGtCondition,
    NumericLtCondition,
    Rule,
    RuleConfig,
    Step,
    load_rule_config,
    render_step,
    render_template,
)

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(rules: list[Rule], variables: dict[str, Any] | None = None) -> RuleConfig:
    return RuleConfig(version=1, variables=variables or {}, rules=tuple(rules))


def _make_rule(
    name: str,
    condition: Condition | None,
    steps: list[Step],
    set_variables: dict[str, Any] | None = None,
) -> Rule:
    return Rule(
        name=name,
        condition=condition,
        steps=tuple(steps),
        set_variables=set_variables or {},
    )


class _MockConn:
    """Captures session_update calls."""

    def __init__(self) -> None:
        self.updates: list[object] = []

    async def session_update(self, session_id: str, update: object) -> None:
        del session_id
        self.updates.append(update)


async def _make_agent_with_session(config: RuleConfig) -> tuple[ConfigurableFakeAcpAgent, _MockConn, str]:
    agent = ConfigurableFakeAcpAgent(config)
    conn = _MockConn()
    agent.on_connect(cast(Client, conn))
    resp = await agent.new_session(cwd="/tmp")
    return agent, conn, resp.session_id


# ===========================================================================
# Condition matching
# ===========================================================================


def test_condition_message_regex_matches() -> None:
    cond = Condition(message_regex=r"(?i)hello")
    assert cond.matches("Hello world")
    assert not cond.matches("goodbye")


def test_condition_message_contains_matches() -> None:
    cond = Condition(message_contains="search")
    assert cond.matches("please search for cats")
    assert not cond.matches("nothing here")


def test_condition_numeric_gt_matches() -> None:
    cond = Condition(numeric_gt=NumericGtCondition(gt=10))
    assert cond.matches("11")
    assert cond.matches("100")
    assert not cond.matches("10")
    assert not cond.matches("5")
    assert not cond.matches("abc")


def test_condition_numeric_lt_matches() -> None:
    cond = Condition(numeric_lt=NumericLtCondition(lt=10))
    assert cond.matches("9")
    assert cond.matches("0")
    assert not cond.matches("10")
    assert not cond.matches("11")
    assert not cond.matches("xyz")


def test_condition_default_always_matches() -> None:
    cond = Condition(default=True)
    assert cond.matches("")
    assert cond.matches("anything at all")


def test_condition_all_of_requires_every_sub_condition() -> None:
    cond = Condition(
        all_of=(
            Condition(message_regex=r"^\d+$"),
            Condition(numeric_gt=NumericGtCondition(gt=100)),
        )
    )
    assert cond.matches("150")
    assert not cond.matches("50")  # numeric_gt fails
    assert not cond.matches("abc")  # regex fails


def test_condition_any_of_passes_with_one_match() -> None:
    cond = Condition(
        any_of=(
            Condition(message_contains="hello"),
            Condition(message_contains="hi"),
        )
    )
    assert cond.matches("hello world")
    assert cond.matches("hi there")
    assert not cond.matches("goodbye")


def test_condition_all_of_empty_passes() -> None:
    cond = Condition(all_of=())
    assert cond.matches("anything")


def test_condition_any_of_empty_passes() -> None:
    cond = Condition(any_of=())
    assert cond.matches("anything")


def test_condition_combined_flat_and_compound() -> None:
    # message_regex + numeric_gt both at top level (implicit AND)
    cond = Condition(message_regex=r"^\d+$", numeric_gt=NumericGtCondition(gt=5))
    assert cond.matches("10")
    assert not cond.matches("3")  # gt fails
    assert not cond.matches("abc")  # regex fails


# ===========================================================================
# Rule matching and ordering
# ===========================================================================


def test_rule_without_condition_always_matches() -> None:
    rule = _make_rule("fallback", None, [Step(type="final", content="ok")])
    assert rule.matches("anything")
    assert rule.matches("")


def test_find_rule_returns_first_match() -> None:
    r1 = _make_rule("first", Condition(message_contains="foo"), [Step(type="final", content="1")])
    r2 = _make_rule("second", Condition(message_contains="foo"), [Step(type="final", content="2")])
    config = _make_config([r1, r2])
    found = config.find_rule("foo bar")
    assert found is not None
    assert found.name == "first"


def test_find_rule_returns_none_when_no_match() -> None:
    r = _make_rule("only", Condition(message_contains="needle"), [Step(type="final", content="ok")])
    config = _make_config([r])
    assert config.find_rule("nothing") is None


def test_find_rule_fallback_after_no_specific_match() -> None:
    specific = _make_rule("specific", Condition(message_regex="foo"), [Step(type="final", content="foo")])
    fallback = _make_rule("fallback", None, [Step(type="final", content="fallback")])
    config = _make_config([specific, fallback])
    found = config.find_rule("bar")
    assert found is not None
    assert found.name == "fallback"


def test_find_rule_default_true_acts_as_fallback() -> None:
    specific = _make_rule("specific", Condition(message_contains="hello"), [Step(type="final", content="hi")])
    fallback = _make_rule("fallback", Condition(default=True), [Step(type="final", content="fallback")])
    config = _make_config([specific, fallback])
    found_specific = config.find_rule("hello")
    found_fallback = config.find_rule("other")
    assert found_specific is not None and found_specific.name == "specific"
    assert found_fallback is not None and found_fallback.name == "fallback"


# ===========================================================================
# Template rendering
# ===========================================================================


def test_render_template_message() -> None:
    result = render_template("You said: {{ message }}", message="hi", variables={})
    assert result == "You said: hi"


def test_render_template_variable() -> None:
    result = render_template("threshold={{ variables.threshold }}", message="x", variables={"threshold": 10})
    assert result == "threshold=10"


def test_render_template_unknown_variable_unchanged() -> None:
    result = render_template("{{ variables.missing }}", message="x", variables={})
    assert result == "{{ variables.missing }}"


def test_render_template_state_last_rule() -> None:
    result = render_template(
        "last: {{ state.last_rule }}",
        message="x",
        variables={},
        state={"last_rule": "greeting"},
    )
    assert result == "last: greeting"


def test_render_template_state_last_user_message() -> None:
    result = render_template(
        "prev: {{ state.last_user_message }}",
        message="now",
        variables={},
        state={"last_user_message": "before"},
    )
    assert result == "prev: before"


def test_render_template_state_custom_variable() -> None:
    result = render_template(
        "intent={{ state.last_intent }}",
        message="x",
        variables={},
        state={"last_intent": "search"},
    )
    assert result == "intent=search"


def test_render_template_state_unknown_key_unchanged() -> None:
    result = render_template("{{ state.missing }}", message="x", variables={}, state={})
    assert result == "{{ state.missing }}"


def test_render_template_no_state_leaves_state_refs_unchanged() -> None:
    result = render_template("{{ state.last_rule }}", message="x", variables={})
    assert result == "{{ state.last_rule }}"


def test_render_step_substitutes_content_and_args() -> None:
    step = Step(type="final", content="{{ message }}", args={"q": "{{ message }}"})
    rendered = render_step(step, message="hello", variables={})
    assert rendered.content == "hello"
    assert rendered.args["q"] == "hello"


def test_render_step_passes_state() -> None:
    step = Step(type="final", content="{{ state.last_rule }}")
    rendered = render_step(step, message="x", variables={}, state={"last_rule": "prev-rule"})
    assert rendered.content == "prev-rule"


# ===========================================================================
# YAML loading
# ===========================================================================


def test_load_rule_config_basic(tmp_path: Path) -> None:
    expected_threshold = 5
    expected_rule_count = 2
    yaml_content = textwrap.dedent("""\
        version: 1
        defaults:
          variables:
            threshold: 5
        rules:
          - name: hi
            when:
              message_regex: "hello"
            then:
              - type: final
                content: "Hello!"
          - name: default
            then:
              - type: final
                content: "fallback"
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_content)
    config = load_rule_config(p)
    assert config.version == 1
    assert config.variables["threshold"] == expected_threshold
    assert len(config.rules) == expected_rule_count
    assert config.rules[0].name == "hi"
    assert config.rules[1].condition is None  # no when → fallback


def test_load_rule_config_all_condition(tmp_path: Path) -> None:
    expected_all_of_count = 2
    yaml_content = textwrap.dedent("""\
        version: 1
        rules:
          - name: big
            when:
              all:
                - message_regex: "^\\\\d+$"
                - numeric_gt:
                    value_from: message
                    gt: 100
            then:
              - type: final
                content: "big"
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_content)
    config = load_rule_config(p)
    rule = config.rules[0]
    assert rule.condition is not None
    assert len(rule.condition.all_of) == expected_all_of_count
    assert rule.matches("150")
    assert not rule.matches("50")


def test_load_rule_config_any_condition(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        version: 1
        rules:
          - name: greet
            when:
              any:
                - message_contains: "hello"
                - message_contains: "hi"
            then:
              - type: final
                content: "hey"
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_content)
    config = load_rule_config(p)
    assert config.rules[0].matches("hello")
    assert config.rules[0].matches("hi there")
    assert not config.rules[0].matches("goodbye")


def test_load_rule_config_set_variables(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        version: 1
        rules:
          - name: r
            when:
              message_contains: x
            set_variables:
              foo: bar
            then:
              - type: final
                content: ok
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_content)
    config = load_rule_config(p)
    assert config.rules[0].set_variables == {"foo": "bar"}


def test_load_rule_config_sleep_step(tmp_path: Path) -> None:
    expected_sleep_ms = 200
    yaml_content = textwrap.dedent("""\
        version: 1
        rules:
          - name: s
            then:
              - type: sleep
                sleep_ms: 200
              - type: final
                content: done
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_content)
    config = load_rule_config(p)
    assert config.rules[0].steps[0].type == "sleep"
    assert config.rules[0].steps[0].sleep_ms == expected_sleep_ms


def test_load_rule_config_error_step(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        version: 1
        rules:
          - name: e
            then:
              - type: error
                tool_name: op
                content: "boom"
              - type: final
                content: failed
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_content)
    config = load_rule_config(p)
    assert config.rules[0].steps[0].type == "error"
    assert config.rules[0].steps[0].tool_name == "op"


def test_load_rule_config_numeric_lt(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        version: 1
        rules:
          - name: small
            when:
              numeric_lt:
                value_from: message
                lt: 10
            then:
              - type: final
                content: small
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_content)
    config = load_rule_config(p)
    assert config.rules[0].matches("5")
    assert not config.rules[0].matches("15")


def test_load_rule_config_invalid_step_type(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        version: 1
        rules:
          - name: bad
            then:
              - type: unknown_step
                content: oops
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_content)
    with pytest.raises(ValueError, match="Unknown step type"):
        load_rule_config(p)


def test_load_rule_config_missing_steps_raises(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        version: 1
        rules:
          - name: empty
            then: []
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_content)
    with pytest.raises(ValueError, match="Rule has no steps"):
        load_rule_config(p)


def test_load_rule_config_default_true(tmp_path: Path) -> None:
    yaml_content = textwrap.dedent("""\
        version: 1
        rules:
          - name: fb
            when:
              default: true
            then:
              - type: final
                content: fallback
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_content)
    config = load_rule_config(p)
    assert config.rules[0].condition is not None
    assert config.rules[0].matches("anything")


# ===========================================================================
# Agent prompt handling
# ===========================================================================


async def test_agent_greeting_rule() -> None:
    config = _make_config([_make_rule("hi", Condition(message_contains="hello"), [Step(type="final", content="Hi!")])])
    agent, conn, sid = await _make_agent_with_session(config)
    result = await agent.prompt([text_block("say hello please")], session_id=sid)
    assert result.stop_reason == "end_turn"
    assert len(conn.updates) > 0


async def test_agent_fallback_rule() -> None:
    config = _make_config(
        [
            _make_rule("specific", Condition(message_contains="foo"), [Step(type="final", content="foo!")]),
            _make_rule("fallback", None, [Step(type="final", content="fallback")]),
        ]
    )
    agent, conn, sid = await _make_agent_with_session(config)
    conn.updates.clear()
    result = await agent.prompt([text_block("bar")], session_id=sid)
    assert result.stop_reason == "end_turn"


async def test_agent_echo_when_no_rule_matches() -> None:
    config = _make_config(
        [_make_rule("only", Condition(message_contains="special"), [Step(type="final", content="special")])]
    )
    agent, conn, sid = await _make_agent_with_session(config)
    conn.updates.clear()
    result = await agent.prompt([text_block("unmatched text")], session_id=sid)
    assert result.stop_reason == "end_turn"
    assert len(conn.updates) == 1  # echo reply only


async def test_agent_numeric_branching() -> None:
    high = _make_rule(
        "high",
        Condition(
            all_of=(
                Condition(message_regex=r"^\d+$"),
                Condition(numeric_gt=NumericGtCondition(gt=50)),
            )
        ),
        [Step(type="final", content="big")],
    )
    low = _make_rule("low", None, [Step(type="final", content="small")])
    config = _make_config([high, low])
    agent, conn, sid = await _make_agent_with_session(config)

    conn.updates.clear()
    await agent.prompt([text_block("100")], session_id=sid)
    high_updates = len(conn.updates)

    conn.updates.clear()
    await agent.prompt([text_block("10")], session_id=sid)
    low_updates = len(conn.updates)

    assert high_updates > 0
    assert low_updates > 0


async def test_agent_session_state_last_user_message() -> None:
    first_rule = _make_rule("first", Condition(message_contains="first"), [Step(type="final", content="got first")])
    second_rule = _make_rule(
        "second",
        Condition(message_contains="second"),
        [Step(type="final", content="prev={{ state.last_user_message }}")],
    )
    config = _make_config([first_rule, second_rule])
    agent, conn, sid = await _make_agent_with_session(config)

    await agent.prompt([text_block("this is first")], session_id=sid)

    conn.updates.clear()
    await agent.prompt([text_block("now second")], session_id=sid)

    texts = [u.content.text for u in conn.updates if isinstance(u, AgentMessageChunk)]
    assert any("this is first" in t for t in texts)


async def test_agent_session_state_last_rule() -> None:
    r1 = _make_rule("alpha", Condition(message_contains="alpha"), [Step(type="final", content="alpha")])
    r2 = _make_rule(
        "beta",
        Condition(message_contains="beta"),
        [Step(type="final", content="prev={{ state.last_rule }}")],
    )
    config = _make_config([r1, r2])
    agent, conn, sid = await _make_agent_with_session(config)

    await agent.prompt([text_block("alpha please")], session_id=sid)

    conn.updates.clear()
    await agent.prompt([text_block("now beta")], session_id=sid)

    texts = [u.content.text for u in conn.updates if isinstance(u, AgentMessageChunk)]
    assert any("alpha" in t for t in texts)


async def test_agent_set_variables_available_in_state() -> None:
    r = _make_rule(
        "setter",
        Condition(message_contains="set"),
        [Step(type="final", content="intent={{ state.my_var }}")],
        set_variables={"my_var": "hello"},
    )
    config = _make_config([r])
    agent, conn, sid = await _make_agent_with_session(config)
    conn.updates.clear()
    await agent.prompt([text_block("please set it")], session_id=sid)

    texts = [u.content.text for u in conn.updates if isinstance(u, AgentMessageChunk)]
    assert any("hello" in t for t in texts)


async def test_agent_sleep_step_does_not_break_flow() -> None:
    r = _make_rule(
        "slow",
        None,
        [
            Step(type="sleep", sleep_ms=1),
            Step(type="final", content="done"),
        ],
    )
    config = _make_config([r])
    agent, _conn, sid = await _make_agent_with_session(config)
    result = await agent.prompt([text_block("go")], session_id=sid)
    assert result.stop_reason == "end_turn"


async def test_agent_error_step_emits_failed_tool() -> None:
    r = _make_rule(
        "err",
        None,
        [
            Step(type="error", tool_name="op", content="boom"),
            Step(type="final", content="after error"),
        ],
    )
    config = _make_config([r])
    agent, conn, sid = await _make_agent_with_session(config)
    conn.updates.clear()
    result = await agent.prompt([text_block("trigger")], session_id=sid)
    assert result.stop_reason == "end_turn"

    statuses = [u.status for u in conn.updates if isinstance(u, ToolCallProgress)]
    assert "failed" in statuses


async def test_agent_thought_step_emits_think_tool() -> None:
    r = _make_rule(
        "thinker",
        None,
        [
            Step(type="thought", content="Thinking..."),
            Step(type="final", content="done"),
        ],
    )
    config = _make_config([r])
    agent, conn, sid = await _make_agent_with_session(config)
    conn.updates.clear()
    await agent.prompt([text_block("go")], session_id=sid)

    kinds = [u.kind for u in conn.updates if isinstance(u, ToolCallStart)]
    assert "think" in kinds


async def test_agent_tool_call_and_result_pair() -> None:
    r = _make_rule(
        "search",
        None,
        [
            Step(type="tool_call", tool_name="search", args={"q": "test"}),
            Step(type="tool_result", content="found it"),
            Step(type="final", content="here"),
        ],
    )
    config = _make_config([r])
    agent, conn, sid = await _make_agent_with_session(config)
    conn.updates.clear()
    result = await agent.prompt([text_block("search for something")], session_id=sid)
    assert result.stop_reason == "end_turn"

    starts = [u for u in conn.updates if isinstance(u, ToolCallStart)]
    closes = [u for u in conn.updates if isinstance(u, ToolCallProgress)]
    assert len(starts) == 1
    assert any(u.status == "completed" for u in closes)


async def test_agent_invalid_session_raises() -> None:
    config = _make_config([_make_rule("r", None, [Step(type="final", content="ok")])])
    agent, _conn, _ = await _make_agent_with_session(config)
    with pytest.raises(RequestError):
        await agent.prompt([text_block("hi")], session_id="nonexistent-session-id")


async def test_agent_load_session() -> None:
    config = _make_config([_make_rule("r", None, [Step(type="final", content="ok")])])
    agent = ConfigurableFakeAcpAgent(config)
    conn = _MockConn()
    agent.on_connect(cast(Client, conn))
    await agent.load_session(cwd="/tmp", session_id="sess-123")
    result = await agent.prompt([text_block("hi")], session_id="sess-123")
    assert result.stop_reason == "end_turn"


async def test_agent_cancel_does_not_raise_for_unknown_session() -> None:
    config = _make_config([_make_rule("r", None, [Step(type="final", content="ok")])])
    agent = ConfigurableFakeAcpAgent(config)
    conn = _MockConn()
    agent.on_connect(cast(Client, conn))
    await agent.cancel("no-such-session")  # should not raise


async def test_agent_list_sessions() -> None:
    config = _make_config([_make_rule("r", None, [Step(type="final", content="ok")])])
    agent = ConfigurableFakeAcpAgent(config)
    conn = _MockConn()
    agent.on_connect(cast(Client, conn))
    ls = await agent.list_sessions(cwd="/tmp")
    assert len(ls.sessions) == 1


async def test_agent_ordered_rule_precedence() -> None:
    """First matching rule wins; later rules are skipped."""
    r1 = _make_rule("first", Condition(message_contains="x"), [Step(type="final", content="first")])
    r2 = _make_rule("second", Condition(message_contains="x"), [Step(type="final", content="second")])
    r3 = _make_rule("fallback", None, [Step(type="final", content="fallback")])
    config = _make_config([r1, r2, r3])
    agent, conn, sid = await _make_agent_with_session(config)

    conn.updates.clear()
    await agent.prompt([text_block("x")], session_id=sid)

    texts = [u.content.text for u in conn.updates if isinstance(u, AgentMessageChunk)]
    assert "first" in texts
    assert "second" not in texts


async def test_agent_multi_turn_stateful_interaction() -> None:
    """Verify that state persists correctly across multiple turns."""
    r_set = _make_rule(
        "set-intent",
        Condition(message_contains="search"),
        [Step(type="final", content="searching")],
        set_variables={"intent": "search"},
    )
    r_check = _make_rule(
        "check-intent",
        Condition(message_contains="what"),
        [Step(type="final", content="intent={{ state.intent }}, last={{ state.last_rule }}")],
    )
    r_fallback = _make_rule("fallback", None, [Step(type="final", content="ok")])
    config = _make_config([r_set, r_check, r_fallback])
    agent, conn, sid = await _make_agent_with_session(config)

    # Turn 1: trigger set-intent
    await agent.prompt([text_block("search for cats")], session_id=sid)

    # Turn 2: check state
    conn.updates.clear()
    await agent.prompt([text_block("what was the intent?")], session_id=sid)

    texts = [u.content.text for u in conn.updates if isinstance(u, AgentMessageChunk)]
    assert any("search" in t for t in texts)
    assert any("set-intent" in t for t in texts)


# ===========================================================================
# Example YAML file smoke test
# ===========================================================================


def test_example_rules_yaml_loads() -> None:
    example_path = Path(__file__).parent.parent / "scripts" / "demo" / "example_rules.yaml"
    assert example_path.exists(), f"example_rules.yaml not found at {example_path}"
    config = load_rule_config(example_path)
    assert config.version == 1
    assert len(config.rules) > 0


async def test_example_rules_yaml_agent_smoke() -> None:
    example_path = Path(__file__).parent.parent / "scripts" / "demo" / "example_rules.yaml"
    config = load_rule_config(example_path)
    agent, conn, sid = await _make_agent_with_session(config)

    for msg in ["hello", "search python", "fail now", "slow response", "repeat last", "random text"]:
        conn.updates.clear()
        result = await agent.prompt([text_block(msg)], session_id=sid)
        assert result.stop_reason == "end_turn"

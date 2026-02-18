#!/usr/bin/env python3
"""Deferred tool result behavior tests."""

from __future__ import annotations

from src.features import Features, reset_features, set_features
from src.repl_environment import REPLEnvironment, TOOL_OUTPUT_END, TOOL_OUTPUT_START
from src.api.routes.chat_pipeline.repl_executor import _tools_success


class _FakeInvocation:
    def __init__(self, success: bool):
        self.success = success


def setup_function() -> None:
    reset_features()


def teardown_function() -> None:
    reset_features()


def test_maybe_wrap_tool_output_legacy_mode() -> None:
    set_features(Features(deferred_tool_results=False))
    repl = REPLEnvironment(context="x")

    output = repl._maybe_wrap_tool_output("hello")

    assert output.startswith(TOOL_OUTPUT_START)
    assert output.endswith(TOOL_OUTPUT_END)
    assert repl.artifacts.get("_tool_outputs") == ["hello"]


def test_maybe_wrap_tool_output_deferred_mode() -> None:
    set_features(Features(deferred_tool_results=True))
    repl = REPLEnvironment(context="x")

    output = repl._maybe_wrap_tool_output("hello")

    assert output == "hello"
    assert repl.artifacts.get("_tool_outputs") in (None, [])


def test_list_dir_deferred_does_not_populate_tool_outputs(tmp_path) -> None:
    set_features(Features(deferred_tool_results=True))
    repl = REPLEnvironment(context="x")

    output = repl._list_dir(str(tmp_path))

    assert TOOL_OUTPUT_START not in output
    assert repl.artifacts.get("_tool_outputs") in (None, [])


def test_get_state_hides_tool_outputs_and_shows_user_variables() -> None:
    set_features(Features(deferred_tool_results=True))
    repl = REPLEnvironment(context="x", artifacts={"_tool_outputs": ["secret"]})
    repl.execute("foo = 123")

    state = repl.get_state()

    assert "_tool_outputs" not in state
    assert "## Available Variables (from previous turns)" in state
    assert "foo (int) = 123" in state


def test_tools_success_fallback_uses_invocation_log() -> None:
    invocations = [_FakeInvocation(success=False), _FakeInvocation(success=True)]
    assert _tools_success("answer", [], 2, invocation_log=invocations) is True


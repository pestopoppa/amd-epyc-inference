"""Unit tests for orchestration graph nodes.

Tests each node class's transition logic with mock LLM/REPL deps.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


from src.graph.nodes import (
    ArchitectCodingNode,
    ArchitectNode,
    CoderEscalationNode,
    CoderNode,
    FrontdoorNode,
    IngestNode,
    WorkerNode,
    _classify_error,
    _rescue_from_last_output,
    select_start_node,
)
from src.graph.state import GraphConfig, TaskDeps, TaskResult, TaskState
from src.graph.graph import orchestration_graph
from src.escalation import ErrorCategory
from src.roles import Role


# ── Fixtures ───────────────────────────────────────────────────────────


@dataclass
class MockREPLResult:
    output: str = ""
    is_final: bool = False
    error: str | None = None


@dataclass
class MockREPLConfig:
    timeout_seconds: int = 30


class MockREPL:
    def __init__(self, results: list[MockREPLResult] | None = None):
        self._results = results or [MockREPLResult(output="hello", is_final=True)]
        self._idx = 0
        self.config = MockREPLConfig()
        self.artifacts = {}
        self._tool_invocations = 0

    def execute(self, code: str) -> MockREPLResult:
        if self._idx < len(self._results):
            result = self._results[self._idx]
            self._idx += 1
            return result
        return MockREPLResult(output="", is_final=True)

    def get_state(self) -> dict:
        return {"files": {}, "vars": {}}


class MockPrimitives:
    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ["FINAL('hello')"]
        self._idx = 0

    def llm_call(self, prompt, role=None, n_tokens=None, **kwargs) -> str:
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return "FINAL('done')"


def make_deps(
    repl_results: list[MockREPLResult] | None = None,
    llm_responses: list[str] | None = None,
    config: GraphConfig | None = None,
) -> TaskDeps:
    return TaskDeps(
        primitives=MockPrimitives(llm_responses),
        repl=MockREPL(repl_results),
        config=config or GraphConfig(max_retries=2, max_escalations=2, max_turns=10),
    )


def make_state(**kwargs) -> TaskState:
    defaults = {
        "task_id": "test-001",
        "prompt": "test prompt",
        "current_role": Role.FRONTDOOR,
        "role_history": ["frontdoor"],
        "max_turns": 10,
    }
    defaults.update(kwargs)
    return TaskState(**defaults)


# ── Error classification tests ─────────────────────────────────────────


class TestClassifyError:
    def test_timeout(self):
        assert _classify_error("Request timed out") == ErrorCategory.TIMEOUT

    def test_schema(self):
        assert _classify_error("JSON validation failed") == ErrorCategory.SCHEMA

    def test_format(self):
        assert _classify_error("ruff format check failed") == ErrorCategory.FORMAT

    def test_early_abort(self):
        assert _classify_error("Generation aborted: quality low") == ErrorCategory.EARLY_ABORT

    def test_infrastructure(self):
        assert _classify_error("Backend connection 502") == ErrorCategory.INFRASTRUCTURE

    def test_code(self):
        assert _classify_error("SyntaxError in module") == ErrorCategory.CODE

    def test_logic(self):
        assert _classify_error("Assertion failed: wrong output") == ErrorCategory.LOGIC

    def test_unknown(self):
        assert _classify_error("Something happened") == ErrorCategory.UNKNOWN


# ── Node selection tests ───────────────────────────────────────────────


class TestSelectStartNode:
    def test_frontdoor(self):
        node = select_start_node(Role.FRONTDOOR)
        assert isinstance(node, FrontdoorNode)

    def test_worker_math(self):
        node = select_start_node(Role.WORKER_MATH)
        assert isinstance(node, WorkerNode)

    def test_coder_primary(self):
        node = select_start_node(Role.CODER_PRIMARY)
        assert isinstance(node, CoderNode)

    def test_coder_escalation(self):
        node = select_start_node(Role.CODER_ESCALATION)
        assert isinstance(node, CoderEscalationNode)

    def test_ingest(self):
        node = select_start_node(Role.INGEST_LONG_CONTEXT)
        assert isinstance(node, IngestNode)

    def test_architect_general(self):
        node = select_start_node(Role.ARCHITECT_GENERAL)
        assert isinstance(node, ArchitectNode)

    def test_architect_coding(self):
        node = select_start_node(Role.ARCHITECT_CODING)
        assert isinstance(node, ArchitectCodingNode)

    def test_string_role(self):
        node = select_start_node("coder_primary")
        assert isinstance(node, CoderNode)

    def test_unknown_role(self):
        node = select_start_node("nonexistent")
        assert isinstance(node, FrontdoorNode)


# ── Node transition tests (via full graph run) ────────────────────────


class TestFrontdoorNode:
    @pytest.mark.asyncio
    async def test_success_ends(self):
        """Frontdoor with successful REPL result should End."""
        state = make_state(current_role=Role.FRONTDOOR)
        deps = make_deps(
            repl_results=[MockREPLResult(output="answer", is_final=True)],
        )
        result = await orchestration_graph.run(FrontdoorNode(), state=state, deps=deps)
        assert isinstance(result.output, TaskResult)
        assert result.output.success is True
        assert "answer" in result.output.answer

    @pytest.mark.asyncio
    async def test_error_retries_then_escalates(self):
        """Frontdoor should retry then escalate to coder on repeated errors."""
        state = make_state(current_role=Role.FRONTDOOR)
        deps = make_deps(
            repl_results=[
                MockREPLResult(error="SyntaxError in code"),
                MockREPLResult(error="SyntaxError in code"),
                MockREPLResult(output="fixed", is_final=True),
            ],
            llm_responses=[
                "x = broken_code()",  # error turn — no FINAL to avoid rescue
                "x = broken_code()",  # error turn — no FINAL to avoid rescue
                "FINAL('fixed')",     # success turn after escalation
            ],
            config=GraphConfig(max_retries=2, max_escalations=2, max_turns=10),
        )
        result = await orchestration_graph.run(FrontdoorNode(), state=state, deps=deps)
        assert isinstance(result.output, TaskResult)
        # Should have escalated to coder then succeeded
        assert len(state.role_history) > 1

    @pytest.mark.asyncio
    async def test_max_turns(self):
        """Should stop at max_turns."""
        state = make_state(current_role=Role.FRONTDOOR, max_turns=1)
        deps = make_deps(
            repl_results=[MockREPLResult(output="partial", is_final=False)],
        )
        result = await orchestration_graph.run(FrontdoorNode(), state=state, deps=deps)
        assert result.output.success is False
        assert "Max turns" in result.output.answer


class TestCoderNode:
    @pytest.mark.asyncio
    async def test_success(self):
        state = make_state(current_role=Role.CODER_PRIMARY)
        deps = make_deps(
            repl_results=[MockREPLResult(output="code output", is_final=True)],
        )
        result = await orchestration_graph.run(CoderNode(), state=state, deps=deps)
        assert result.output.success is True

    @pytest.mark.asyncio
    async def test_escalates_to_architect(self):
        """Coder should escalate to ArchitectNode on repeated errors."""
        state = make_state(current_role=Role.CODER_PRIMARY)
        deps = make_deps(
            repl_results=[
                MockREPLResult(error="SyntaxError"),
                MockREPLResult(error="SyntaxError"),
                MockREPLResult(output="architect fixed it", is_final=True),
            ],
            config=GraphConfig(max_retries=2, max_escalations=2, max_turns=10),
        )
        result = await orchestration_graph.run(CoderNode(), state=state, deps=deps)
        assert isinstance(result.output, TaskResult)


class TestArchitectNode:
    @pytest.mark.asyncio
    async def test_terminal_failure(self):
        """Architect is terminal — should fail after retries exhausted."""
        state = make_state(current_role=Role.ARCHITECT_GENERAL)
        deps = make_deps(
            repl_results=[
                MockREPLResult(error="Logic error"),
                MockREPLResult(error="Logic error"),
                MockREPLResult(error="Logic error"),
            ],
            llm_responses=[
                "x = attempt()",  # error turn — no FINAL to avoid rescue
                "x = attempt()",  # error turn — no FINAL to avoid rescue
                "x = attempt()",  # error turn — no FINAL to avoid rescue
            ],
            config=GraphConfig(max_retries=2, max_escalations=0, max_turns=10),
        )
        result = await orchestration_graph.run(ArchitectNode(), state=state, deps=deps)
        assert result.output.success is False
        assert "FAILED" in result.output.answer


class TestEscalationCountIncrement:
    """Verify the bug fix: escalation_count is actually incremented."""

    @pytest.mark.asyncio
    async def test_escalation_count_incremented(self):
        state = make_state(current_role=Role.FRONTDOOR)
        deps = make_deps(
            repl_results=[
                MockREPLResult(error="SyntaxError"),
                MockREPLResult(error="SyntaxError"),
                # After escalation to coder
                MockREPLResult(output="fixed", is_final=True),
            ],
            llm_responses=[
                "x = broken()",   # error turn — no FINAL to avoid rescue
                "x = broken()",   # error turn — no FINAL to avoid rescue
                "FINAL('fixed')", # success turn after escalation
            ],
            config=GraphConfig(max_retries=2, max_escalations=2, max_turns=10),
        )
        await orchestration_graph.run(FrontdoorNode(), state=state, deps=deps)
        # escalation_count should be > 0 since we escalated
        assert state.escalation_count > 0


# ── Rescue from last output tests ─────────────────────────────────────


class TestRescueFromLastOutput:
    """Test _rescue_from_last_output() extraction logic."""

    def test_rescues_final_pattern(self):
        text = 'Some reasoning...\nFINAL("42")\nMore text'
        assert _rescue_from_last_output(text) == "42"

    def test_rescues_prose_answer(self):
        text = "After analyzing the options.\nThe answer is D"
        assert _rescue_from_last_output(text) == "D"

    def test_rescues_code_block(self):
        text = "Here is the solution:\n```python\ndef solve(n):\n    return n * 2\n```"
        result = _rescue_from_last_output(text)
        assert result is not None
        assert "def solve" in result

    def test_returns_none_on_empty(self):
        assert _rescue_from_last_output("") is None
        assert _rescue_from_last_output("   ") is None

    def test_returns_none_on_no_pattern(self):
        assert _rescue_from_last_output("Just some random text without any answer") is None

    def test_prefers_final_over_prose(self):
        text = 'The answer is B\nFINAL("C")'
        assert _rescue_from_last_output(text) == "C"

    def test_small_code_block_ignored(self):
        text = "```\nx=1\n```"
        # Code block with < 20 chars should be ignored
        assert _rescue_from_last_output(text) is None


class TestMaxTurnsRescue:
    """Test that max-turns triggers rescue instead of immediate failure."""

    @pytest.mark.asyncio
    async def test_max_turns_rescues_prose_answer(self):
        """When max turns hit and last_output has a prose answer, rescue it."""
        state = make_state(current_role=Role.FRONTDOOR, max_turns=1)
        deps = make_deps(
            # REPL output contains a rescuable prose answer pattern
            repl_results=[MockREPLResult(output="The answer is B", is_final=False)],
        )
        result = await orchestration_graph.run(FrontdoorNode(), state=state, deps=deps)
        # Turn 1 executes (non-final, sets last_output="The answer is B"),
        # then turn 2 hits max_turns → rescue extracts "B"
        assert result.output.success is True
        assert "B" in result.output.answer

    @pytest.mark.asyncio
    async def test_max_turns_no_rescue_without_answer(self):
        """When max turns hit and last_output has no extractable answer, fail."""
        state = make_state(current_role=Role.FRONTDOOR, max_turns=1)
        state.last_output = "Still thinking about it..."
        deps = make_deps(
            repl_results=[MockREPLResult(output="partial", is_final=False)],
        )
        result = await orchestration_graph.run(FrontdoorNode(), state=state, deps=deps)
        assert result.output.success is False
        assert "Max turns" in result.output.answer

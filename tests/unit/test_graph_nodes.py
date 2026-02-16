"""Unit tests for orchestration graph nodes.

Tests each node class's transition logic with mock LLM/REPL deps.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


from src.graph.helpers import (
    _build_think_harder_config,
    _classify_error,
    _detect_role_cycle,
    _extract_final_from_raw,
    _extract_prose_answer,
    _is_comment_only,
    _rescue_from_last_output,
    _resolve_answer,
    _update_workspace_from_turn,
)
from src.graph.nodes import (
    ArchitectCodingNode,
    ArchitectNode,
    CoderEscalationNode,
    CoderNode,
    FrontdoorNode,
    IngestNode,
    WorkerNode,
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


class TestThinkHarderConfig:
    def test_build_think_harder_config_high_roi(self):
        state = make_state(current_role=Role.FRONTDOOR)
        state.think_harder_roi_by_role = {
            str(Role.FRONTDOOR): {"attempts": 20.0, "successes": 18.0}
        }
        cfg = _build_think_harder_config(state)
        assert cfg["n_tokens"] >= 3500
        assert cfg["temperature"] >= 0.45
        assert bool(cfg["cot_prefix"])

    def test_build_think_harder_config_low_roi(self):
        state = make_state(current_role=Role.FRONTDOOR)
        state.think_harder_roi_by_role = {
            str(Role.FRONTDOOR): {"attempts": 20.0, "successes": 2.0}
        }
        cfg = _build_think_harder_config(state)
        assert cfg["n_tokens"] <= 2600
        assert cfg["temperature"] <= 0.35
        assert cfg["cot_prefix"] == ""


class TestWorkspaceBroadcast:
    def test_workspace_uses_proposal_selection_broadcast(self):
        state = make_state(current_role=Role.FRONTDOOR)

        _update_workspace_from_turn(
            state=state,
            role=Role.FRONTDOOR,
            output="We will parse config then run tests.",
            error=None,
        )
        _update_workspace_from_turn(
            state=state,
            role=Role.CODER_PRIMARY,
            output="",
            error="Schema mismatch on field `steps`",
        )

        ws = state.workspace_state
        assert ws.get("broadcast_version", 0) >= 2
        assert isinstance(ws.get("proposals"), list) and ws["proposals"]
        assert any("parse config" in x.get("text", "") for x in ws.get("commitments", []))
        assert any("Schema mismatch" in x.get("text", "") for x in ws.get("open_questions", []))


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


# ====================================================================
# Characterization tests for pure helper functions
# ====================================================================


class TestClassifyErrorCharacterization:
    """Exhaustive characterization of _classify_error keyword matching."""

    # -- TIMEOUT --
    def test_timeout_keyword(self):
        assert _classify_error("timeout occurred") == ErrorCategory.TIMEOUT

    def test_timed_out_keyword(self):
        assert _classify_error("request timed out after 30s") == ErrorCategory.TIMEOUT

    def test_deadline_keyword(self):
        assert _classify_error("deadline exceeded") == ErrorCategory.TIMEOUT

    # -- SCHEMA --
    def test_json_validation(self):
        assert _classify_error("json validation failed") == ErrorCategory.SCHEMA

    def test_schema_keyword(self):
        assert _classify_error("schema mismatch in IR") == ErrorCategory.SCHEMA

    def test_jsonschema_keyword(self):
        assert _classify_error("jsonschema error on field X") == ErrorCategory.SCHEMA

    # -- FORMAT --
    def test_ruff_format_error(self):
        assert _classify_error("ruff format error") == ErrorCategory.FORMAT

    def test_lint_keyword(self):
        assert _classify_error("lint check failed") == ErrorCategory.FORMAT

    def test_markdown_keyword(self):
        assert _classify_error("markdown style violation") == ErrorCategory.FORMAT

    def test_style_keyword(self):
        assert _classify_error("style check failed") == ErrorCategory.FORMAT

    # -- EARLY_ABORT --
    def test_generation_aborted(self):
        assert _classify_error("generation aborted") == ErrorCategory.EARLY_ABORT

    def test_early_abort_keyword(self):
        assert _classify_error("early abort detected") == ErrorCategory.EARLY_ABORT

    def test_abort_keyword_alone(self):
        assert _classify_error("abort signal received") == ErrorCategory.EARLY_ABORT

    # -- INFRASTRUCTURE --
    def test_backend_unreachable(self):
        assert _classify_error("backend unreachable") == ErrorCategory.INFRASTRUCTURE

    def test_connection_error(self):
        assert _classify_error("connection refused on port 8080") == ErrorCategory.INFRASTRUCTURE

    def test_502_error(self):
        assert _classify_error("server returned 502") == ErrorCategory.INFRASTRUCTURE

    def test_503_error(self):
        assert _classify_error("HTTP 503 service unavailable") == ErrorCategory.INFRASTRUCTURE

    # -- CODE --
    def test_syntax_error(self):
        assert _classify_error("syntax error in code") == ErrorCategory.CODE

    def test_typeerror(self):
        assert _classify_error("TypeError: 'NoneType'") == ErrorCategory.CODE

    def test_nameerror(self):
        assert _classify_error("NameError: name 'x' is not defined") == ErrorCategory.CODE

    def test_test_fail(self):
        assert _classify_error("test fail in test_foo.py") == ErrorCategory.CODE

    def test_import_error(self):
        assert _classify_error("import error: no module named foo") == ErrorCategory.CODE

    # -- LOGIC --
    def test_incorrect_logic(self):
        assert _classify_error("incorrect logic in solution") == ErrorCategory.LOGIC

    def test_wrong_output(self):
        assert _classify_error("wrong output for test case 3") == ErrorCategory.LOGIC

    def test_assertion_error(self):
        assert _classify_error("assertion failed: expected 42") == ErrorCategory.LOGIC

    # -- UNKNOWN --
    def test_unknown_fallback(self):
        assert _classify_error("something random") == ErrorCategory.UNKNOWN

    def test_empty_string(self):
        assert _classify_error("") == ErrorCategory.UNKNOWN

    # -- case insensitivity --
    def test_case_insensitive_timeout(self):
        assert _classify_error("TIMEOUT OCCURRED") == ErrorCategory.TIMEOUT

    def test_case_insensitive_json(self):
        assert _classify_error("JSON Validation Failed") == ErrorCategory.SCHEMA


class TestExtractFinalFromRawCharacterization:
    """Characterize FINAL() pattern extraction from raw LLM output."""

    def test_double_quoted(self):
        assert _extract_final_from_raw('some text FINAL("answer") more') == "answer"

    def test_single_quoted(self):
        assert _extract_final_from_raw("FINAL('hello')") == "hello"

    def test_triple_single_quoted_multiline(self):
        text = "FINAL('''line one\nline two\nline three''')"
        result = _extract_final_from_raw(text)
        assert result is not None
        assert "line one" in result
        assert "line three" in result

    def test_triple_double_quoted_multiline(self):
        text = 'FINAL("""first\nsecond""")'
        result = _extract_final_from_raw(text)
        assert result is not None
        assert "first" in result
        assert "second" in result

    def test_unquoted_token(self):
        assert _extract_final_from_raw("FINAL(42)") == "42"

    def test_no_final_returns_none(self):
        assert _extract_final_from_raw("no final call here") is None

    def test_empty_string(self):
        assert _extract_final_from_raw("") is None

    def test_final_with_whitespace(self):
        result = _extract_final_from_raw('FINAL( "spaced" )')
        assert result == "spaced"

    def test_final_in_longer_text(self):
        text = 'Let me compute...\nresult = 42\nFINAL("42")\nDone.'
        assert _extract_final_from_raw(text) == "42"


class TestExtractProseAnswerCharacterization:
    """Characterize prose answer extraction patterns."""

    def test_the_answer_is(self):
        assert _extract_prose_answer("The answer is D") == "D"

    def test_i_choose(self):
        assert _extract_prose_answer("I choose B") == "B"

    def test_answer_colon(self):
        assert _extract_prose_answer("Answer: C") == "C"

    def test_therefore(self):
        assert _extract_prose_answer("Therefore, the answer is A") == "A"

    def test_i_will_go_with(self):
        assert _extract_prose_answer("I'll go with B") == "B"

    def test_my_answer_is(self):
        assert _extract_prose_answer("My answer is 42") == "42"

    def test_the_correct_option_is(self):
        assert _extract_prose_answer("The correct option is A") == "A"

    def test_bare_mcq_letter_fallback(self):
        assert _extract_prose_answer("some reasoning\nD\nmore text") == "D"

    def test_no_pattern_returns_none(self):
        assert _extract_prose_answer("This is just some text without an answer pattern.") is None

    def test_empty_string(self):
        assert _extract_prose_answer("") is None

    def test_so_the_answer_is(self):
        assert _extract_prose_answer("So the answer is B") == "B"

    def test_i_select(self):
        assert _extract_prose_answer("I select A") == "A"

    def test_the_correct_answer_is(self):
        assert _extract_prose_answer("The correct answer is C") == "C"

    def test_the_correct_choice_is(self):
        assert _extract_prose_answer("The correct choice is B") == "B"


class TestRescueFromLastOutputCharacterization:
    """Characterize the last-resort answer rescue pipeline (priority order)."""

    def test_rescue_final(self):
        text = 'Some reasoning\nFINAL("the answer")\nDone'
        assert _rescue_from_last_output(text) == "the answer"

    def test_rescue_prose(self):
        text = "After analysis.\nThe answer is D"
        assert _rescue_from_last_output(text) == "D"

    def test_rescue_code_block(self):
        code = "def solution():\n    return [1, 2, 3]\n\nprint(solution())"
        text = f"Here is my solution:\n```python\n{code}\n```"
        result = _rescue_from_last_output(text)
        assert result is not None
        assert "def solution" in result

    def test_empty_returns_none(self):
        assert _rescue_from_last_output("") is None

    def test_whitespace_only_returns_none(self):
        assert _rescue_from_last_output("   \n\t  ") is None

    def test_short_code_block_ignored(self):
        text = "```\nx = 1\n```"
        assert _rescue_from_last_output(text) is None

    def test_final_takes_priority_over_prose(self):
        text = 'The answer is B\nFINAL("A")'
        assert _rescue_from_last_output(text) == "A"

    def test_prose_takes_priority_over_code(self):
        text = "The answer is D\n```python\nprint('hello world and some more text')\n```"
        assert _rescue_from_last_output(text) == "D"

    def test_no_rescue_possible(self):
        text = "I am still thinking about this problem..."
        assert _rescue_from_last_output(text) is None


class TestIsCommentOnlyCharacterization:
    """Characterize comment-only code detection."""

    def test_single_comment(self):
        assert _is_comment_only("# just a comment") is True

    def test_multiple_comments(self):
        assert _is_comment_only("# line one\n# line two\n# line three") is True

    def test_comments_with_blanks(self):
        assert _is_comment_only("# comment\n\n# another\n") is True

    def test_executable_code(self):
        assert _is_comment_only("x = 1") is False

    def test_mixed_comment_and_code(self):
        assert _is_comment_only("# comment\nx = 1") is False

    def test_empty_string(self):
        assert _is_comment_only("") is True

    def test_blank_lines_only(self):
        assert _is_comment_only("\n\n\n") is True

    def test_comment_then_print(self):
        assert _is_comment_only("# setup\nprint('hello')") is False

    def test_indented_comment(self):
        assert _is_comment_only("    # indented comment") is True

    def test_function_def(self):
        assert _is_comment_only("def foo(): pass") is False


class TestDetectRoleCycleCharacterization:
    """Characterize role-cycle detection for escalation loops."""

    def test_period_2_cycle(self):
        assert _detect_role_cycle(["A", "B", "A", "B"]) is True

    def test_period_3_cycle(self):
        assert _detect_role_cycle(["A", "B", "C", "A", "B", "C"]) is True

    def test_no_cycle_short(self):
        assert _detect_role_cycle(["A", "B"]) is False

    def test_no_cycle_unique(self):
        assert _detect_role_cycle(["A", "B", "C", "D"]) is False

    def test_no_cycle_three_items(self):
        assert _detect_role_cycle(["A", "B", "C"]) is False

    def test_period_2_with_prefix(self):
        assert _detect_role_cycle(["X", "A", "B", "A", "B"]) is True

    def test_empty_list(self):
        assert _detect_role_cycle([]) is False


class TestResolveAnswerCharacterization:
    """Characterize answer resolution from output and tool outputs."""

    def test_output_preferred(self):
        assert _resolve_answer("hello", ["tool1"]) == "hello"

    def test_tool_output_fallback(self):
        assert _resolve_answer("", ["tool result"]) == "tool result"

    def test_empty_both(self):
        assert _resolve_answer("", []) == ""

    def test_strips_whitespace(self):
        assert _resolve_answer("  answer  ", []) == "answer"

    def test_multiple_tool_outputs(self):
        result = _resolve_answer("", ["result1", "result2"])
        assert "result1" in result
        assert "result2" in result

    def test_whitespace_only_output_uses_tools(self):
        assert _resolve_answer("   ", []) == ""

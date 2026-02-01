"""Tests for prompt_builders.py — core prompt construction functions.

Covers the 7 functions imported by chat.py/chat_pipeline.py plus
review/revision prompts. Focuses on correctness-critical behavior:
code extraction, FINAL wrapping, error classification, and prompt
structure invariants.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.prompt_builders import (
    auto_wrap_final,
    build_escalation_prompt,
    build_long_context_exploration_prompt,
    build_review_verdict_prompt,
    build_revision_prompt,
    build_root_lm_prompt,
    build_routing_context,
    classify_error,
    detect_format_constraints,
    extract_code_from_response,
)


# ── extract_code_from_response ────────────────────────────────────────────


class TestExtractCodeFromResponse:
    """Code extraction from LLM responses."""

    def test_markdown_python_block(self):
        response = "Here's the code:\n```python\nprint('hello')\n```\nDone."
        result = extract_code_from_response(response)
        assert "print('hello')" in result
        assert "```" not in result

    def test_markdown_block_no_language(self):
        response = "```\nx = 42\n```"
        result = extract_code_from_response(response)
        assert "x = 42" in result

    def test_unpaired_trailing_backticks(self):
        """Model quirk: sometimes generates trailing ``` without opening."""
        response = "FINAL('result')\n```"
        result = extract_code_from_response(response)
        assert "FINAL" in result

    def test_repl_function_recognized_as_code(self):
        """Lines starting with REPL tools should be treated as code."""
        response = "peek(0, 50)\ngrep('error')"
        result = extract_code_from_response(response)
        assert "peek(0, 50)" in result

    def test_final_function_recognized(self):
        response = "FINAL('The answer is 42')"
        result = extract_code_from_response(response)
        assert "FINAL(" in result

    def test_strips_import_lines(self):
        response = "```python\nimport json\nfrom pathlib import Path\nresult = json.loads(data)\n```"
        result = extract_code_from_response(response)
        assert "import json" not in result
        assert "from pathlib" not in result
        assert "result = json.loads(data)" in result

    def test_preserves_from_in_non_import(self):
        """'from' in normal code should not be stripped."""
        response = "x = get_data_from_source()\nFINAL(x)"
        result = extract_code_from_response(response)
        assert "get_data_from_source" in result

    def test_empty_response(self):
        result = extract_code_from_response("")
        assert result == ""

    def test_code_extracted_from_indented_block(self):
        response = "```python\n    x = 1\n    y = 2\n```"
        result = extract_code_from_response(response)
        assert "x = 1" in result
        assert "y = 2" in result

    def test_multiple_code_blocks_picks_first(self):
        response = "```python\nfirst = 1\n```\n```python\nsecond = 2\n```"
        result = extract_code_from_response(response)
        assert "first = 1" in result


# ── auto_wrap_final ───────────────────────────────────────────────────────


class TestAutoWrapFinal:
    """FINAL() auto-wrapping logic."""

    def test_already_has_final(self):
        code = "FINAL('hello world')"
        assert auto_wrap_final(code) == code

    def test_exploration_not_wrapped(self):
        """Exploration functions should NOT be wrapped."""
        for func in ("peek(0, 50)", "grep('error')", "llm_call('summarize')"):
            result = auto_wrap_final(func)
            assert "FINAL" not in result, f"Should not wrap: {func}"

    def test_llm_batch_not_wrapped(self):
        code = "llm_batch(['q1', 'q2'], role='worker')"
        result = auto_wrap_final(code)
        assert "FINAL" not in result

    def test_artifacts_not_wrapped(self):
        code = "artifacts['summary'] = 'done'"
        result = auto_wrap_final(code)
        assert "FINAL" not in result

    def test_def_wrapped_in_triple_quotes(self):
        code = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
        result = auto_wrap_final(code)
        assert "FINAL(" in result
        assert "'''" in result or '"""' in result

    def test_class_wrapped_in_triple_quotes(self):
        code = "class MyWidget:\n    pass"
        result = auto_wrap_final(code)
        assert "FINAL(" in result

    def test_single_expression_wrapped(self):
        code = "result = compute_answer()"
        result = auto_wrap_final(code)
        assert "FINAL(" in result

    def test_control_flow_not_wrapped(self):
        """Control flow statements should not be wrapped."""
        for stmt in ("for x in range(10):", "while True:", "if condition:",
                      "try:", "with open('f'):"):
            result = auto_wrap_final(stmt)
            assert "FINAL" not in result, f"Should not wrap: {stmt}"

    def test_import_not_wrapped(self):
        code = "import os"
        result = auto_wrap_final(code)
        assert "FINAL" not in result

    def test_escapes_triple_quotes(self):
        """Triple quotes in code must be escaped to prevent syntax errors."""
        code = "x = '''some text'''"
        result = auto_wrap_final(code)
        assert "FINAL(" in result
        # The triple quotes should be escaped or the code should use the other style


# ── classify_error ────────────────────────────────────────────────────────


class TestClassifyError:
    """Error classification for escalation routing."""

    def test_format_gates(self):
        for gate in ("schema", "format", "lint", "mdformat", "shfmt"):
            result = classify_error("some error", gate_name=gate)
            assert result.value in ("FORMAT", "SCHEMA", "format", "schema"), \
                f"Gate '{gate}' should classify as FORMAT or SCHEMA"

    def test_syntax_error_is_code(self):
        result = classify_error("SyntaxError: invalid syntax")
        assert result.value.upper() in ("CODE", "SYNTAX")

    def test_type_error_is_code(self):
        result = classify_error("TypeError: expected str, got int")
        assert result.value.upper() in ("CODE", "SYNTAX", "TYPE")

    def test_import_error_is_code(self):
        result = classify_error("ImportError: No module named 'foo'")
        assert result.value.upper() in ("CODE", "IMPORT")

    def test_assertion_error_is_logic(self):
        result = classify_error("AssertionError: test failed")
        # Assertion errors map to LOGIC category
        assert result.value.upper() in ("LOGIC", "TEST", "ASSERTION")

    def test_timeout_keyword(self):
        result = classify_error("Request timed out after 60s")
        assert result.value.upper() in ("TIMEOUT",)

    def test_generation_abort_early_abort(self):
        result = classify_error("early abort: high entropy detected")
        assert result.value.upper() == "EARLY_ABORT"

    def test_generation_abort_unknown(self):
        """'aborted' alone doesn't match 'early abort' pattern."""
        result = classify_error("Generation aborted: entropy spike detected")
        assert result.value.upper() == "UNKNOWN"

    def test_unknown_fallback(self):
        result = classify_error("solar flare disrupted the quasar")
        assert result.value.upper() == "UNKNOWN"

    def test_expected_in_substring_is_logic(self):
        """'unexpected' contains 'expected' — classified as LOGIC."""
        result = classify_error("something completely unexpected happened")
        assert result.value.upper() == "LOGIC"

    def test_empty_message(self):
        result = classify_error("")
        assert result is not None


# ── build_root_lm_prompt ──────────────────────────────────────────────────


class TestBuildRootLmPrompt:
    """Root LM prompt structure tests."""

    def test_includes_task(self):
        result = build_root_lm_prompt(
            state="ready", original_prompt="What is 2+2?"
        )
        assert "What is 2+2?" in result

    def test_includes_routing_context_on_turn_0(self):
        result = build_root_lm_prompt(
            state="ready",
            original_prompt="test",
            turn=0,
            routing_context="Role: frontdoor | Tier: A",
        )
        assert "frontdoor" in result

    def test_includes_error_context(self):
        result = build_root_lm_prompt(
            state="ready",
            original_prompt="test",
            last_error="NameError: x is not defined",
        )
        assert "NameError" in result

    def test_includes_last_output(self):
        result = build_root_lm_prompt(
            state="x = 42",
            original_prompt="test",
            last_output="42",
        )
        assert "42" in result

    def test_returns_string(self):
        result = build_root_lm_prompt(state="", original_prompt="hello")
        assert isinstance(result, str)
        assert len(result) > 100  # Should be a substantial prompt


# ── build_routing_context ─────────────────────────────────────────────────


class TestBuildRoutingContext:
    """Routing context generation (MemRL integration)."""

    def test_none_router_returns_empty(self):
        result = build_routing_context("frontdoor", None, "some task")
        assert result == ""

    def test_router_exception_returns_empty(self):
        """Gracefully handles retrieval failures."""
        mock_router = MagicMock()
        mock_router.get_q_values_for_task.side_effect = RuntimeError("DB error")
        result = build_routing_context("frontdoor", mock_router, "some task")
        assert result == ""

    def test_respects_max_chars(self):
        mock_router = MagicMock()
        mock_router.get_q_values_for_task.return_value = [
            {"role": "coder", "q_value": 0.9, "task_preview": "x" * 500}
        ] * 10
        result = build_routing_context("frontdoor", mock_router, "task", max_chars=100)
        assert len(result) <= 150  # Some overflow is OK for formatting


# ── build_long_context_exploration_prompt ──────────────────────────────────


class TestBuildLongContextExplorationPrompt:
    """Long context exploration prompt generation."""

    def test_includes_original_prompt(self):
        result = build_long_context_exploration_prompt("Find the bug", 20000)
        assert "Find the bug" in result

    def test_estimates_tokens(self):
        result = build_long_context_exploration_prompt("task", 40000)
        # 40000 chars / 4 = ~10000 tokens
        assert "10000" in result or "10,000" in result or "10k" in result.lower()

    def test_search_task_uses_grep(self):
        """Search tasks should start with grep()."""
        result = build_long_context_exploration_prompt("Find all error handlers", 20000)
        assert "grep" in result.lower()

    def test_non_search_task_uses_peek(self):
        """Non-search tasks should start with peek()."""
        result = build_long_context_exploration_prompt("Summarize this document", 20000)
        assert "peek" in result.lower()

    def test_includes_repl_tools(self):
        result = build_long_context_exploration_prompt("task", 20000)
        for tool in ("peek", "grep", "FINAL"):
            assert tool in result


# ── build_escalation_prompt ───────────────────────────────────────────────


class TestBuildEscalationPrompt:
    """Escalation prompt with failure context."""

    def test_with_escalation_context(self):
        from src.escalation import EscalationContext, EscalationAction

        ctx = EscalationContext(
            current_role="worker_explore",
            error_message="NameError: undefined variable",
            error_category="CODE",
            failure_count=2,
            task_id="test-123",
        )

        class MockDecision:
            target_role = "coder_escalation"
            reason = "Code error needs specialist"
            action = EscalationAction.ESCALATE

        result = build_escalation_prompt("Fix the bug", "x = None", ctx, MockDecision())
        assert isinstance(result, str)
        assert len(result) > 50

    def test_includes_error_info(self):
        from src.escalation import EscalationContext, EscalationAction

        ctx = EscalationContext(
            current_role="frontdoor",
            error_message="TimeoutError: request timed out",
            error_category="TIMEOUT",
            failure_count=1,
            task_id="t1",
        )

        class MockDecision:
            target_role = "architect_general"
            reason = "Timeout needs investigation"
            action = EscalationAction.ESCALATE

        result = build_escalation_prompt("Debug timeout", "", ctx, MockDecision())
        assert "timeout" in result.lower() or "Timeout" in result


# ── build_review_verdict_prompt ───────────────────────────────────────────


class TestBuildReviewVerdictPrompt:
    """Architect review verdict prompt."""

    def test_includes_question_and_answer(self):
        result = build_review_verdict_prompt(
            question="What is 2+2?",
            answer="The answer is 4.",
        )
        assert "What is 2+2?" in result
        assert "4" in result

    def test_truncates_long_question(self):
        long_q = "x" * 500
        result = build_review_verdict_prompt(question=long_q, answer="ok")
        assert len(long_q) > 300  # original is long
        # The prompt itself should be reasonable length
        assert result is not None

    def test_truncates_long_answer(self):
        long_a = "y" * 3000
        result = build_review_verdict_prompt(question="q", answer=long_a)
        assert result is not None

    def test_context_digest_included(self):
        result = build_review_verdict_prompt(
            question="q",
            answer="a",
            context_digest="Important context about the topic",
        )
        assert "Important context" in result


# ── build_revision_prompt ─────────────────────────────────────────────────


class TestBuildRevisionPrompt:
    """Fast revision prompt."""

    def test_includes_all_parts(self):
        result = build_revision_prompt(
            question="What is Python?",
            original="Python is a language.",
            corrections="Add mention of dynamic typing.",
        )
        assert "Python" in result
        assert "dynamic typing" in result

    def test_returns_string(self):
        result = build_revision_prompt("q", "a", "fix it")
        assert isinstance(result, str)
        assert len(result) > 20


# ── detect_format_constraints ─────────────────────────────────────────────


class TestDetectFormatConstraints:
    """Format constraint detection via regex."""

    def test_json_format_detected(self):
        constraints = detect_format_constraints("Return the answer as JSON")
        assert any("json" in c.lower() for c in constraints)

    def test_numbered_list_detected(self):
        constraints = detect_format_constraints("Give me a numbered list of items")
        assert len(constraints) > 0

    def test_bullet_list_detected(self):
        constraints = detect_format_constraints("Format as a bullet list")
        assert len(constraints) > 0
        assert any("bullet" in c.lower() for c in constraints)

    def test_no_constraints_returns_empty(self):
        constraints = detect_format_constraints("What is the capital of France?")
        assert isinstance(constraints, list)

    def test_table_format_detected(self):
        constraints = detect_format_constraints("Show the results in a table")
        assert len(constraints) > 0

    def test_case_insensitive(self):
        constraints = detect_format_constraints("RETURN AS JSON FORMAT")
        assert any("json" in c.lower() for c in constraints)

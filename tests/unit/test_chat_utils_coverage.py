"""Tests for chat_utils.py - utility functions and constants for chat endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.api.routes.chat_utils import (
    DEFAULT_TIMEOUT_S,
    ROLE_TIMEOUTS,
    RoutingResult,
    _estimate_tokens,
    _formalize_output,
    _is_stub_final,
    _resolve_answer,
    _should_formalize,
    _strip_tool_outputs,
    _truncate_looped_answer,
)
from src.roles import Role


class TestEstimateTokens:
    """Tests for _estimate_tokens."""

    def test_empty_string(self):
        """Empty string returns 0 tokens."""
        assert _estimate_tokens("") == 0

    def test_short_string(self):
        """Short string estimation (4 chars per token)."""
        assert _estimate_tokens("test") == 1

    def test_longer_string(self):
        """Longer string estimation."""
        # 20 chars / 4 = 5 tokens
        assert _estimate_tokens("a" * 20) == 5

    def test_unicode_characters(self):
        """Unicode characters counted by char length."""
        # UTF-8 multi-byte chars still count as 1 char each in Python
        assert _estimate_tokens("你好世界") == 1  # 4 chars / 4 = 1


class TestIsStubFinal:
    """Tests for _is_stub_final stub detection."""

    @pytest.mark.parametrize(
        "stub",
        [
            "complete",
            "Complete.",
            "COMPLETE",
            "See above",
            "see above.",
            "Analysis complete",
            "Estimation complete",
            "Done",
            "DONE.",
            "Finished",
            "See results above",
            "see output above",
            "See structured output above",
            "See integrated results above",
            "See the structured output above",
        ],
    )
    def test_detects_stub_patterns(self, stub: str):
        """Known stub patterns are detected."""
        assert _is_stub_final(stub) is True

    def test_real_content_not_stub(self):
        """Real content is not detected as stub."""
        assert _is_stub_final("The answer is 42.") is False

    def test_partial_match_detected(self):
        """Partial match within text detects stub."""
        assert _is_stub_final("Analysis complete. Refer to output.") is True

    def test_whitespace_handling(self):
        """Whitespace is stripped before detection."""
        assert _is_stub_final("  complete  ") is True
        assert _is_stub_final("\n\tcomplete.\n") is True


class TestStripToolOutputs:
    """Tests for _strip_tool_outputs."""

    def test_empty_text(self):
        """Empty text returns empty."""
        assert _strip_tool_outputs("", []) == ""
        assert _strip_tool_outputs("", ["output"]) == ""

    def test_no_tool_outputs(self):
        """No tool outputs leaves text unchanged."""
        text = "This is regular output."
        assert _strip_tool_outputs(text, []) == text

    def test_strips_structured_delimiter(self):
        """Strips structured tool output delimiters."""
        text = "Before <<<TOOL_OUTPUT>>>some tool data<<<END_TOOL_OUTPUT>>> After"
        result = _strip_tool_outputs(text, [])
        assert "<<<TOOL_OUTPUT>>>" not in result
        assert "some tool data" not in result
        assert "Before" in result
        assert "After" in result

    def test_strips_multiline_structured_output(self):
        """Strips multiline tool output between delimiters."""
        text = """Before
<<<TOOL_OUTPUT>>>
Line 1
Line 2
<<<END_TOOL_OUTPUT>>>
After"""
        result = _strip_tool_outputs(text, [])
        assert "Line 1" not in result
        assert "Before" in result
        assert "After" in result

    def test_strips_exact_tool_outputs(self):
        """Strips exact tool output strings (legacy fallback)."""
        text = "Result: {'role': 'coder'} done"
        result = _strip_tool_outputs(text, ["{'role': 'coder'}"])
        assert "{'role': 'coder'}" not in result

    def test_strips_common_prefixes(self):
        """Strips common tool output prefixes."""
        text = "Current Role: coder_escalation\nSome content"
        result = _strip_tool_outputs(text, [])
        assert "Current Role:" not in result

    def test_strips_available_files_prefix(self):
        """Strips Available files prefix."""
        text = "Available files: /path/to/file.py\nSome content"
        result = _strip_tool_outputs(text, [])
        assert "Available files:" not in result

    def test_strips_routing_advice_prefix(self):
        """Strips Routing advice prefix."""
        text = "Routing advice: use coder\nSome content"
        result = _strip_tool_outputs(text, [])
        assert "Routing advice:" not in result

    def test_collapses_multiple_blank_lines(self):
        """Multiple blank lines collapsed to double newline."""
        text = "Line 1\n\n\n\n\nLine 2"
        result = _strip_tool_outputs(text, [])
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result


class TestResolveAnswer:
    """Tests for _resolve_answer."""

    @dataclass
    class MockResult:
        """Mock ExecutionResult for testing."""

        output: str = ""
        final_answer: str = ""

    def test_final_answer_only(self):
        """Final answer returned when no captured output."""
        result = self.MockResult(output="", final_answer="The answer is 42")
        assert _resolve_answer(result) == "The answer is 42"

    def test_captured_output_only(self):
        """Empty final answer returns empty (captured output ignored without stub)."""
        result = self.MockResult(output="Printed content", final_answer="")
        # Without a stub final, final_answer is returned even if empty
        assert _resolve_answer(result) == ""

    def test_stub_final_returns_captured(self):
        """Stub final answer returns captured output."""
        result = self.MockResult(
            output="This is the real analysis",
            final_answer="Analysis complete. See above.",
        )
        assert _resolve_answer(result) == "This is the real analysis"

    def test_both_present_combined(self):
        """Both captured and final combined when different."""
        result = self.MockResult(
            output="Step 1 output",
            final_answer="Final conclusion",
        )
        answer = _resolve_answer(result)
        assert "Step 1 output" in answer
        assert "Final conclusion" in answer

    def test_final_in_captured_returns_final(self):
        """Final returned when final is substring of captured."""
        result = self.MockResult(
            output="Long analysis with Final summary included",
            final_answer="Final summary",
        )
        answer = _resolve_answer(result)
        # When final IS in captured, just return final
        assert answer == "Final summary"

    def test_tool_outputs_stripped(self):
        """Tool outputs stripped from captured output."""
        result = self.MockResult(
            output="<<<TOOL_OUTPUT>>>role: coder<<<END_TOOL_OUTPUT>>>Real content",
            final_answer="Done",
        )
        answer = _resolve_answer(result, tool_outputs=["role: coder"])
        assert "role: coder" not in answer

    def test_whitespace_handling(self):
        """Whitespace in output handled correctly."""
        result = self.MockResult(
            output="  Spaced content  ",
            final_answer="complete",  # Stub
        )
        answer = _resolve_answer(result)
        assert answer == "Spaced content"


class TestTruncateLoopedAnswer:
    """Tests for _truncate_looped_answer."""

    def test_empty_answer(self):
        """Empty answer returned unchanged."""
        assert _truncate_looped_answer("", "prompt text") == ""

    def test_empty_prompt(self):
        """Empty prompt returns answer unchanged."""
        assert _truncate_looped_answer("answer", "") == "answer"

    def test_short_prompt_no_truncation(self):
        """Short prompt (<40 chars) returns answer unchanged."""
        answer = "This is the answer"
        prompt = "Short"
        assert _truncate_looped_answer(answer, prompt) == answer

    def test_no_loop_detected(self):
        """Answer without prompt echo returned unchanged."""
        answer = "The answer is 42"
        prompt = "What is the meaning of life, the universe, and everything?"
        assert _truncate_looped_answer(answer, prompt) == answer

    def test_loop_detected_truncates(self):
        """Answer with prompt loop truncated."""
        prompt = "Please explain the concept of recursion in programming"
        answer = f"Recursion is when a function calls itself. That's the basic idea.{prompt[-80:]}"
        result = _truncate_looped_answer(answer, prompt)
        assert "recursion in programming" not in result.lower()
        assert "Recursion is when" in result

    def test_loop_at_beginning_not_truncated(self):
        """Loop at very beginning (idx=0) not truncated."""
        prompt = "This is a long enough prompt for testing purposes here"
        answer = prompt[-80:] + " Extra content"
        # idx = 0, so no truncation
        result = _truncate_looped_answer(answer, prompt)
        assert result == answer

    def test_short_truncation_preserved(self):
        """Truncation that leaves <20 chars preserved as original."""
        prompt = "x" * 100
        # Answer has prompt suffix at position 5, truncation would leave only 5 chars
        answer = "short" + prompt[-80:]
        result = _truncate_looped_answer(answer, prompt)
        # Since truncated length would be 5 < 20, original returned
        assert result == answer


class TestShouldFormalize:
    """Tests for _should_formalize."""

    def test_feature_disabled(self):
        """Formalization disabled by feature flag."""
        with patch("src.api.routes.chat_utils.features") as mock_features:
            mock_features.return_value.output_formalizer = False
            should, spec = _should_formalize("Write a poem")
        assert should is False
        assert spec == ""

    def test_no_constraints_detected(self):
        """No format constraints returns False."""
        with patch("src.api.routes.chat_utils.features") as mock_features:
            mock_features.return_value.output_formalizer = True
            with patch("src.api.routes.chat_utils.detect_format_constraints") as mock_detect:
                mock_detect.return_value = []
                should, spec = _should_formalize("Write a poem")
        assert should is False
        assert spec == ""

    def test_constraints_detected(self):
        """Detected constraints return True with spec."""
        with patch("src.api.routes.chat_utils.features") as mock_features:
            mock_features.return_value.output_formalizer = True
            with patch("src.api.routes.chat_utils.detect_format_constraints") as mock_detect:
                mock_detect.return_value = ["JSON format", "camelCase keys"]
                should, spec = _should_formalize("Return JSON with camelCase")
        assert should is True
        assert spec == "JSON format; camelCase keys"


class TestFormalizeOutput:
    """Tests for _formalize_output."""

    def test_successful_formalization(self):
        """Successful formalization returns reformatted answer."""
        mock_primitives = MagicMock()
        mock_primitives.llm_call.return_value = "  Reformatted answer  "

        with patch("src.api.routes.chat_utils.build_formalizer_prompt") as mock_build:
            mock_build.return_value = "Formalize this..."
            result = _formalize_output(
                "Original answer",
                "User prompt",
                "JSON format",
                mock_primitives,
            )

        assert result == "Reformatted answer"
        mock_primitives.llm_call.assert_called_once()

    def test_formalization_empty_result(self):
        """Empty reformatted result returns original."""
        mock_primitives = MagicMock()
        mock_primitives.llm_call.return_value = ""

        with patch("src.api.routes.chat_utils.build_formalizer_prompt"):
            result = _formalize_output(
                "Original answer",
                "User prompt",
                "JSON format",
                mock_primitives,
            )

        assert result == "Original answer"

    def test_formalization_short_result(self):
        """Short reformatted result (<5 chars) returns original."""
        mock_primitives = MagicMock()
        mock_primitives.llm_call.return_value = "abc"

        with patch("src.api.routes.chat_utils.build_formalizer_prompt"):
            result = _formalize_output(
                "Original answer",
                "User prompt",
                "JSON format",
                mock_primitives,
            )

        assert result == "Original answer"

    def test_formalization_exception_returns_original(self):
        """Exception during formalization returns original."""
        mock_primitives = MagicMock()
        mock_primitives.llm_call.side_effect = RuntimeError("LLM error")

        with patch("src.api.routes.chat_utils.build_formalizer_prompt"):
            result = _formalize_output(
                "Original answer",
                "User prompt",
                "JSON format",
                mock_primitives,
            )

        assert result == "Original answer"


class TestRoutingResult:
    """Tests for RoutingResult dataclass."""

    def test_role_property_with_decision(self):
        """Role property returns first routing decision."""
        result = RoutingResult(
            task_id="test-123",
            task_ir={},
            use_mock=False,
            routing_decision=["coder_escalation", "worker_general"],
        )
        assert result.role == "coder_escalation"

    def test_role_property_empty_decision(self):
        """Role property returns frontdoor when no decision."""
        result = RoutingResult(
            task_id="test-123",
            task_ir={},
            use_mock=False,
            routing_decision=[],
        )
        assert result.role == str(Role.FRONTDOOR)

    def test_timeout_for_role(self):
        """timeout_for_role returns role-specific timeout."""
        result = RoutingResult(
            task_id="test-123",
            task_ir={},
            use_mock=False,
        )
        # Should return timeout from ROLE_TIMEOUTS or default
        timeout = result.timeout_for_role("architect_general")
        assert isinstance(timeout, int)
        assert timeout > 0

    def test_timeout_for_unknown_role(self):
        """Unknown role returns default timeout."""
        result = RoutingResult(
            task_id="test-123",
            task_ir={},
            use_mock=False,
        )
        timeout = result.timeout_for_role("unknown_role_xyz")
        assert timeout == DEFAULT_TIMEOUT_S

    def test_default_timeout_value(self):
        """RoutingResult has default timeout."""
        result = RoutingResult(
            task_id="test-123",
            task_ir={},
            use_mock=False,
        )
        assert result.timeout_s == DEFAULT_TIMEOUT_S


class TestConstants:
    """Tests for module-level constants."""

    def test_role_timeouts_dict(self):
        """ROLE_TIMEOUTS is a dict with role strings."""
        assert isinstance(ROLE_TIMEOUTS, dict)
        # Should have at least some entries
        assert len(ROLE_TIMEOUTS) > 0

    def test_default_timeout_positive(self):
        """DEFAULT_TIMEOUT_S is a positive integer."""
        assert isinstance(DEFAULT_TIMEOUT_S, int)
        assert DEFAULT_TIMEOUT_S > 0

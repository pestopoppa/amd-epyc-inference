"""Tests for src/api/routes/chat_utils.py.

Covers: _resolve_answer, _truncate_looped_answer, _is_stub_final,
_strip_tool_outputs, _estimate_tokens, _should_formalize, RoutingResult.
"""

from unittest.mock import MagicMock, patch


from src.api.routes.chat_utils import (
    RoutingResult,
    _estimate_tokens,
    _resolve_answer,
    _should_formalize,
    _strip_tool_outputs,
    _truncate_looped_answer,
)


# ── _resolve_answer ──────────────────────────────────────────────────────


class TestResolveAnswer:
    """Test answer extraction from ExecutionResult."""

    def test_returns_final_answer_when_no_output(self):
        result = MagicMock(output="", final_answer="The answer is 42")
        assert _resolve_answer(result) == "The answer is 42"

    def test_returns_output_when_stub_final(self):
        result = MagicMock(
            output="Detailed analysis here...",
            final_answer="Analysis complete. See above.",
        )
        with patch("src.classifiers.is_stub_final", return_value=True):
            assert _resolve_answer(result) == "Detailed analysis here..."

    def test_returns_final_when_not_stub(self):
        result = MagicMock(
            output="Some debug output",
            final_answer="The real answer",
        )
        with patch("src.classifiers.is_stub_final", return_value=False):
            answer = _resolve_answer(result)
            assert "The real answer" in answer

    def test_strips_tool_outputs(self):
        result = MagicMock(
            output="<<<TOOL_OUTPUT>>>role info<<<END_TOOL_OUTPUT>>>\nActual content",
            final_answer="See above.",
        )
        with patch("src.classifiers.is_stub_final", return_value=True):
            answer = _resolve_answer(result, tool_outputs=["role info"])
            assert "<<<TOOL_OUTPUT>>>" not in answer
            assert "Actual content" in answer


# ── _truncate_looped_answer ──────────────────────────────────────────────


class TestTruncateLoopedAnswer:
    """Test prompt echo truncation."""

    def test_passthrough_on_clean_answer(self):
        answer = "This is a clean answer with no echoing."
        prompt = "What is the meaning of life? Please explain in detail."
        assert _truncate_looped_answer(answer, prompt) == answer

    def test_truncates_on_prompt_echo(self):
        prompt = "X" * 100
        answer = f"The answer is 42. Here is some padding text.{prompt[-80:]}"
        result = _truncate_looped_answer(answer, prompt)
        assert "The answer is 42." in result
        assert len(result) < len(answer)

    def test_passthrough_on_short_prompt(self):
        assert _truncate_looped_answer("answer", "hi") == "answer"

    def test_passthrough_on_empty(self):
        assert _truncate_looped_answer("", "prompt") == ""
        assert _truncate_looped_answer("answer", "") == "answer"

    def test_does_not_truncate_to_nothing(self):
        prompt = "A" * 100
        answer = prompt[-80:]  # Answer IS the probe
        result = _truncate_looped_answer(answer, prompt)
        # Should not truncate to empty/tiny
        assert len(result) > 0


# ── _strip_tool_outputs ──────────────────────────────────────────────────


class TestStripToolOutputs:
    """Test tool output stripping from captured stdout."""

    def test_strips_structured_delimiters(self):
        text = "Hello <<<TOOL_OUTPUT>>>some tool output<<<END_TOOL_OUTPUT>>> world"
        result = _strip_tool_outputs(text, [])
        assert "<<<TOOL_OUTPUT>>>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strips_legacy_exact_match(self):
        text = "Role: frontdoor\nActual output"
        result = _strip_tool_outputs(text, ["Role: frontdoor"])
        assert "Role: frontdoor" not in result

    def test_strips_common_prefixes(self):
        text = "Current Role: frontdoor\nAvailable files: []\nAnswer here"
        result = _strip_tool_outputs(text, [])
        assert "Current Role:" not in result

    def test_passthrough_on_empty(self):
        assert _strip_tool_outputs("", []) == ""

    def test_collapses_blank_lines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = _strip_tool_outputs(text, [])
        assert "\n\n\n" not in result


# ── _estimate_tokens ─────────────────────────────────────────────────────


class TestEstimateTokens:
    """Test rough token estimation."""

    def test_basic_estimation(self):
        assert _estimate_tokens("abcd") == 1
        assert _estimate_tokens("abcdefgh") == 2

    def test_empty_string(self):
        assert _estimate_tokens("") == 0


# ── _should_formalize ────────────────────────────────────────────────────


class TestShouldFormalize:
    """Test format constraint detection."""

    @patch("src.api.routes.chat_utils.features")
    def test_disabled_when_feature_off(self, mock_features):
        mock_features.return_value = MagicMock(output_formalizer=False)
        should, spec = _should_formalize("Format as JSON")
        assert should is False

    @patch("src.api.routes.chat_utils.features")
    @patch("src.api.routes.chat_utils.detect_format_constraints")
    def test_enabled_with_constraints(self, mock_detect, mock_features):
        mock_features.return_value = MagicMock(output_formalizer=True)
        mock_detect.return_value = ["JSON format"]
        should, spec = _should_formalize("Format as JSON")
        assert should is True
        assert "JSON format" in spec

    @patch("src.api.routes.chat_utils.features")
    @patch("src.api.routes.chat_utils.detect_format_constraints")
    def test_disabled_without_constraints(self, mock_detect, mock_features):
        mock_features.return_value = MagicMock(output_formalizer=True)
        mock_detect.return_value = []
        should, spec = _should_formalize("Tell me a joke")
        assert should is False


# ── RoutingResult ────────────────────────────────────────────────────────


class TestRoutingResult:
    """Test RoutingResult dataclass."""

    def test_role_property_returns_first(self):
        r = RoutingResult(task_id="t1", task_ir={}, use_mock=False, routing_decision=["coder_escalation"])
        assert r.role == "coder_escalation"

    def test_role_property_defaults_to_frontdoor(self):
        r = RoutingResult(task_id="t1", task_ir={}, use_mock=False)
        assert "frontdoor" in r.role.lower()

    def test_timeout_for_role(self):
        r = RoutingResult(task_id="t1", task_ir={}, use_mock=False)
        timeout = r.timeout_for_role("architect_general")
        assert isinstance(timeout, int)
        assert timeout > 0

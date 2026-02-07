"""Tests for chat_pipeline stages (stages.py + split modules).

Covers: _execute_mock, _parse_plan_steps, _annotate_error, _quality_escalate.
"""

from unittest.mock import MagicMock, patch
import time


from src.api.routes.chat_utils import RoutingResult
from src.api.routes.chat_pipeline.stages import (
    _annotate_error,
    _execute_mock,
    _quality_escalate,
)
from src.api.routes.chat_pipeline.proactive_stage import _parse_plan_steps


# ── _execute_mock ────────────────────────────────────────────────────────


class TestExecuteMock:
    """Test mock mode execution."""

    def test_returns_mock_response(self):
        request = MagicMock(prompt="Hello world", context=None)
        routing = RoutingResult(task_id="t1", task_ir={}, use_mock=True)
        state = MagicMock(progress_logger=None)
        start = time.perf_counter()

        result = _execute_mock(request, routing, state, start)

        assert result.mock_mode is True
        assert result.real_mode is False
        assert "[MOCK]" in result.answer
        assert "Hello world" in result.answer

    def test_includes_context_info(self):
        request = MagicMock(prompt="test", context="some context text")
        routing = RoutingResult(task_id="t1", task_ir={}, use_mock=True)
        state = MagicMock(progress_logger=None)
        start = time.perf_counter()

        result = _execute_mock(request, routing, state, start)
        assert "context" in result.answer.lower()

    def test_logs_progress(self):
        request = MagicMock(prompt="test", context=None)
        routing = RoutingResult(task_id="t1", task_ir={}, use_mock=True)
        logger = MagicMock()
        state = MagicMock(progress_logger=logger)
        start = time.perf_counter()

        _execute_mock(request, routing, state, start)
        logger.log_task_completed.assert_called_once()


# ── _parse_plan_steps ────────────────────────────────────────────────────


class TestParsePlanSteps:
    """Test architect JSON plan step parsing."""

    def test_parses_valid_steps(self):
        raw = '[{"id": "S1", "action": "research"}, {"id": "S2", "action": "implement"}]'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 2
        assert steps[0]["id"] == "S1"
        assert steps[1]["action"] == "implement"

    def test_adds_defaults(self):
        raw = '[{"id": "S1", "action": "research"}]'
        steps = _parse_plan_steps(raw)
        assert steps[0]["actor"] == "worker"
        assert steps[0]["depends_on"] == []
        assert steps[0]["outputs"] == []

    def test_strips_markdown_fences(self):
        raw = '```json\n[{"id": "S1", "action": "go"}]\n```'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 1

    def test_handles_trailing_commas(self):
        raw = '[{"id": "S1", "action": "go"},]'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 1

    def test_returns_empty_on_invalid_json(self):
        assert _parse_plan_steps("not json at all") == []

    def test_returns_empty_on_non_list(self):
        assert _parse_plan_steps('{"id": "S1"}') == []

    def test_skips_invalid_steps(self):
        raw = '[{"id": "S1", "action": "go"}, {"bad": true}, "string"]'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 1

    def test_rejects_steps_without_id(self):
        raw = '[{"action": "go"}]'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 0

    def test_rejects_steps_without_action(self):
        raw = '[{"id": "S1"}]'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 0


# ── _annotate_error ──────────────────────────────────────────────────────


class TestAnnotateError:
    """Test error pattern detection in responses."""

    def test_timeout_gets_504(self):
        resp = MagicMock(answer="[ERROR: Request timed out after 60s]", error_code=None, error_detail=None)
        result = _annotate_error(resp)
        assert result.error_code == 504

    def test_backend_failure_gets_502(self):
        resp = MagicMock(answer="[ERROR: Backend connection failed]", error_code=None, error_detail=None)
        result = _annotate_error(resp)
        assert result.error_code == 502

    def test_generic_error_gets_500(self):
        resp = MagicMock(answer="[ERROR: Something unexpected]", error_code=None, error_detail=None)
        result = _annotate_error(resp)
        assert result.error_code == 500

    def test_failed_prefix_gets_500(self):
        resp = MagicMock(answer="[FAILED: max retries]", error_code=None, error_detail=None)
        result = _annotate_error(resp)
        assert result.error_code == 500

    def test_clean_answer_no_annotation(self):
        resp = MagicMock(answer="The answer is 42", error_code=None, error_detail=None)
        result = _annotate_error(resp)
        assert result.error_code is None

    def test_empty_answer_no_annotation(self):
        resp = MagicMock(answer="", error_code=None, error_detail=None)
        result = _annotate_error(resp)
        assert result.error_code is None

    def test_none_answer_no_crash(self):
        resp = MagicMock(answer=None, error_code=None, error_detail=None)
        result = _annotate_error(resp)
        assert result.error_code is None


# ── _quality_escalate ───────────────────────────────────────────────────


class TestQualityEscalate:
    """Test shared quality escalation helper."""

    @patch("src.api.routes.chat_pipeline.stages.features")
    def test_passthrough_when_monitor_disabled(self, mock_features):
        mock_features.return_value.generation_monitor = False
        primitives = MagicMock()
        answer, role = _quality_escalate("good answer", "prompt", primitives, "frontdoor")
        assert answer == "good answer"
        assert role == "frontdoor"

    @patch("src.api.routes.chat_pipeline.stages.features")
    @patch("src.api.routes.chat_pipeline.stages._detect_output_quality_issue")
    def test_passthrough_when_no_issue(self, mock_detect, mock_features):
        mock_features.return_value.generation_monitor = True
        mock_detect.return_value = None
        primitives = MagicMock()
        answer, role = _quality_escalate("good answer", "prompt", primitives, "frontdoor")
        assert answer == "good answer"
        assert role == "frontdoor"

    @patch("src.api.routes.chat_pipeline.stages.features")
    @patch("src.api.routes.chat_pipeline.stages._detect_output_quality_issue")
    def test_escalates_on_quality_issue(self, mock_detect, mock_features):
        mock_features.return_value.generation_monitor = True
        mock_detect.return_value = "repetitive"
        primitives = MagicMock()
        primitives.llm_call.return_value = "  Better answer  "
        answer, role = _quality_escalate("bad answer", "prompt", primitives, "frontdoor")
        assert answer == "Better answer"

    @patch("src.api.routes.chat_pipeline.stages.features")
    @patch("src.api.routes.chat_pipeline.stages._detect_output_quality_issue")
    def test_returns_original_on_escalation_failure(self, mock_detect, mock_features):
        mock_features.return_value.generation_monitor = True
        mock_detect.return_value = "repetitive"
        primitives = MagicMock()
        primitives.llm_call.side_effect = RuntimeError("model error")
        answer, role = _quality_escalate("bad answer", "prompt", primitives, "frontdoor")
        assert answer == "bad answer"
        assert role == "frontdoor"

    def test_passthrough_on_error_prefix(self):
        primitives = MagicMock()
        answer, role = _quality_escalate("[ERROR: timeout]", "prompt", primitives, "frontdoor")
        assert answer == "[ERROR: timeout]"

    def test_passthrough_on_empty(self):
        primitives = MagicMock()
        answer, role = _quality_escalate("", "prompt", primitives, "frontdoor")
        assert answer == ""

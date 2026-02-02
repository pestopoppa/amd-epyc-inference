"""Tests for proactive parallel delegation pipeline (Item C wiring).

Tests: _parse_plan_steps(), _execute_proactive() complexity gating,
architect bypass, and full mock flow.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.api.routes.chat_pipeline import _parse_plan_steps
from src.proactive_delegation import (
    ComplexitySignals,
    IterationContext,
    ReviewDecision,
    TaskComplexity,
)


# ── _parse_plan_steps tests ───────────────────────────────────────────


class TestParsePlanSteps:
    """Test JSON plan parsing with various formats."""

    def test_valid_json_array(self):
        raw = json.dumps([
            {"id": "S1", "action": "analyze code", "actor": "worker"},
            {"id": "S2", "action": "write tests", "actor": "coder", "depends_on": ["S1"]},
        ])
        steps = _parse_plan_steps(raw)
        assert len(steps) == 2
        assert steps[0]["id"] == "S1"
        assert steps[1]["depends_on"] == ["S1"]

    def test_markdown_fenced_json(self):
        raw = "```json\n" + json.dumps([
            {"id": "S1", "action": "step one"},
            {"id": "S2", "action": "step two"},
        ]) + "\n```"
        steps = _parse_plan_steps(raw)
        assert len(steps) == 2

    def test_trailing_comma_tolerance(self):
        raw = '[{"id":"S1","action":"do X"},{"id":"S2","action":"do Y"},]'
        steps = _parse_plan_steps(raw)
        assert len(steps) == 2

    def test_invalid_json_returns_empty(self):
        assert _parse_plan_steps("not json at all") == []

    def test_non_array_returns_empty(self):
        assert _parse_plan_steps('{"id": "S1"}') == []

    def test_missing_required_fields_filtered(self):
        raw = json.dumps([
            {"id": "S1", "action": "valid step"},
            {"action": "missing id"},
            {"id": "S3"},  # missing action
            {"id": "S4", "action": "also valid"},
        ])
        steps = _parse_plan_steps(raw)
        assert len(steps) == 2
        assert steps[0]["id"] == "S1"
        assert steps[1]["id"] == "S4"

    def test_defaults_applied(self):
        raw = json.dumps([{"id": "S1", "action": "do thing"}])
        steps = _parse_plan_steps(raw)
        assert steps[0]["actor"] == "worker"
        assert steps[0]["depends_on"] == []
        assert steps[0]["outputs"] == []

    def test_empty_input(self):
        assert _parse_plan_steps("") == []
        assert _parse_plan_steps("   ") == []

    def test_non_dict_items_filtered(self):
        raw = json.dumps([
            {"id": "S1", "action": "valid"},
            "not a dict",
            42,
            {"id": "S2", "action": "also valid"},
        ])
        steps = _parse_plan_steps(raw)
        assert len(steps) == 2


# ── _execute_proactive gating tests ──────────────────────────────────


class TestExecuteProactiveGating:
    """Test that _execute_proactive correctly gates on features and complexity."""

    @pytest.fixture
    def mock_request(self):
        req = MagicMock()
        req.prompt = "Design a distributed caching system with consistency guarantees"
        req.context = ""
        req.real_mode = True
        req.mock_mode = False
        return req

    @pytest.fixture
    def mock_routing(self):
        routing = MagicMock()
        routing.task_id = "test-001"
        routing.task_ir = {"task_type": "chat", "objective": "test"}
        routing.routing_decision = ["frontdoor"]
        routing.formalization_applied = False
        return routing

    @pytest.fixture
    def mock_primitives(self):
        p = MagicMock()
        p._backends = {}
        p.total_tokens_generated = 0
        p.total_prompt_eval_ms = 0
        p.total_generation_ms = 0
        p._last_predicted_tps = 0
        p.total_http_overhead_ms = 0
        return p

    @pytest.fixture
    def mock_state(self):
        s = MagicMock()
        s.registry = MagicMock()
        s.progress_logger = None
        s.hybrid_router = None
        return s

    @pytest.mark.asyncio
    async def test_returns_none_when_feature_disabled(
        self, mock_request, mock_routing, mock_primitives, mock_state,
    ):
        from src.api.routes.chat_pipeline import _execute_proactive

        with patch("src.api.routes.chat_pipeline.stages.features") as mock_feat:
            mock_feat.return_value = MagicMock(parallel_execution=False)
            result = await _execute_proactive(
                mock_request, mock_routing, mock_primitives, mock_state, 0.0,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_real_mode(
        self, mock_request, mock_routing, mock_primitives, mock_state,
    ):
        from src.api.routes.chat_pipeline import _execute_proactive

        mock_request.real_mode = False
        with patch("src.api.routes.chat_pipeline.stages.features") as mock_feat:
            mock_feat.return_value = MagicMock(parallel_execution=True)
            result = await _execute_proactive(
                mock_request, mock_routing, mock_primitives, mock_state, 0.0,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_non_complex_tasks(
        self, mock_request, mock_routing, mock_primitives, mock_state,
    ):
        from src.api.routes.chat_pipeline import _execute_proactive
        from src.proactive_delegation import TaskComplexity

        with patch("src.api.routes.chat_pipeline.stages.features") as mock_feat:
            mock_feat.return_value = MagicMock(parallel_execution=True)
            with patch(
                "src.proactive_delegation.classify_task_complexity",
                return_value=(TaskComplexity.SIMPLE, MagicMock()),
            ):
                result = await _execute_proactive(
                    mock_request, mock_routing, mock_primitives, mock_state, 0.0,
                )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_architect_already_selected(
        self, mock_request, mock_routing, mock_primitives, mock_state,
    ):
        from src.api.routes.chat_pipeline import _execute_proactive
        from src.proactive_delegation import TaskComplexity

        mock_routing.routing_decision = ["architect_general"]

        with patch("src.api.routes.chat_pipeline.stages.features") as mock_feat:
            mock_feat.return_value = MagicMock(parallel_execution=True)
            with patch(
                "src.proactive_delegation.classify_task_complexity",
                return_value=(TaskComplexity.COMPLEX, MagicMock()),
            ):
                result = await _execute_proactive(
                    mock_request, mock_routing, mock_primitives, mock_state, 0.0,
                )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_plan_too_short(
        self, mock_request, mock_routing, mock_primitives, mock_state,
    ):
        from src.api.routes.chat_pipeline import _execute_proactive
        from src.proactive_delegation import TaskComplexity

        # Architect returns single-step plan
        mock_primitives.llm_call = MagicMock(return_value=json.dumps([
            {"id": "S1", "action": "do everything"},
        ]))

        with patch("src.api.routes.chat_pipeline.stages.features") as mock_feat:
            mock_feat.return_value = MagicMock(parallel_execution=True)
            with patch(
                "src.proactive_delegation.classify_task_complexity",
                return_value=(TaskComplexity.COMPLEX, MagicMock()),
            ):
                result = await _execute_proactive(
                    mock_request, mock_routing, mock_primitives, mock_state, 0.0,
                )
        assert result is None


# ── build_task_decomposition_prompt test ──────────────────────────────


class TestBuildTaskDecompositionPrompt:
    """Test prompt generation for architect task decomposition."""

    def test_prompt_contains_objective(self):
        from src.prompt_builders import build_task_decomposition_prompt

        prompt = build_task_decomposition_prompt("Build a REST API")
        assert "Build a REST API" in prompt
        assert "JSON" in prompt

    def test_prompt_truncates_long_objective(self):
        from src.prompt_builders import build_task_decomposition_prompt

        long_obj = "x" * 1000
        prompt = build_task_decomposition_prompt(long_obj)
        # Should be truncated to 500 chars
        assert len(prompt) < 1200

    def test_prompt_includes_context_note(self):
        from src.prompt_builders import build_task_decomposition_prompt

        prompt = build_task_decomposition_prompt("task", "some context here")
        assert "Context" in prompt


# ── Custom exception tests ────────────────────────────────────────────


class TestCustomExceptions:
    """Test custom delegation exception hierarchy."""

    def test_delegation_error_is_exception(self):
        from src.proactive_delegation import DelegationError
        assert issubclass(DelegationError, Exception)
        e = DelegationError("test failure")
        assert str(e) == "test failure"

    def test_architect_plan_error_inherits(self):
        from src.proactive_delegation import ArchitectPlanError, DelegationError
        assert issubclass(ArchitectPlanError, DelegationError)
        e = ArchitectPlanError("bad plan")
        assert isinstance(e, DelegationError)
        assert str(e) == "bad plan"

    def test_step_execution_error_inherits(self):
        from src.proactive_delegation import StepExecutionError, DelegationError
        assert issubclass(StepExecutionError, DelegationError)

    def test_step_execution_error_fields(self):
        from src.proactive_delegation import StepExecutionError
        cause = ValueError("bad input")
        e = StepExecutionError("S1", "coder_primary", cause=cause)
        assert e.step_id == "S1"
        assert e.role == "coder_primary"
        assert e.cause is cause
        assert "S1" in str(e)
        assert "coder_primary" in str(e)
        assert "bad input" in str(e)

    def test_step_execution_error_no_cause(self):
        from src.proactive_delegation import StepExecutionError
        e = StepExecutionError("S2", "worker_general")
        assert e.cause is None
        assert "S2" in str(e)
        assert "bad input" not in str(e)

    def test_exceptions_catchable_by_base(self):
        from src.proactive_delegation import (
            DelegationError, ArchitectPlanError, StepExecutionError,
        )
        with pytest.raises(DelegationError):
            raise ArchitectPlanError("plan failed")
        with pytest.raises(DelegationError):
            raise StepExecutionError("S1", "worker")


# ── IterationContext tests ────────────────────────────────────────────


class TestIterationContext:
    """Test iteration tracking and limits using real IterationContext objects."""

    def test_can_iterate_fresh(self):
        """New context should allow iteration."""
        ctx = IterationContext(max_iterations=3, max_total_iterations=10)
        assert ctx.can_iterate("sub1") is True

    def test_can_iterate_at_limit(self):
        """Should block iteration at per-subtask limit."""
        ctx = IterationContext(max_iterations=2, max_total_iterations=10)
        ctx.record_iteration("sub1", ReviewDecision.REQUEST_CHANGES, "fix X")
        assert ctx.can_iterate("sub1") is True
        ctx.record_iteration("sub1", ReviewDecision.REQUEST_CHANGES, "fix Y")
        assert ctx.can_iterate("sub1") is False

    def test_total_iterations_limit(self):
        """Should block iteration at global limit."""
        ctx = IterationContext(max_iterations=5, max_total_iterations=3)
        ctx.record_iteration("s1", ReviewDecision.REQUEST_CHANGES)
        ctx.record_iteration("s2", ReviewDecision.REQUEST_CHANGES)
        ctx.record_iteration("s3", ReviewDecision.REQUEST_CHANGES)
        assert ctx.can_iterate("s4") is False

    def test_record_iteration_history(self):
        """Should track iteration history correctly."""
        ctx = IterationContext(max_iterations=5, max_total_iterations=10)
        ctx.record_iteration("sub1", ReviewDecision.APPROVE, "looks good")
        assert len(ctx.iteration_history) == 1
        assert ctx.iteration_history[0]["subtask_id"] == "sub1"
        assert ctx.iteration_history[0]["decision"] == "approve"
        assert ctx.iteration_history[0]["feedback"] == "looks good"

    def test_get_summary(self):
        """Should provide accurate summary statistics."""
        ctx = IterationContext(max_iterations=3, max_total_iterations=10)
        ctx.record_iteration("s1", ReviewDecision.APPROVE)
        ctx.record_iteration("s2", ReviewDecision.REJECT)
        summary = ctx.get_summary()
        assert summary["total_iterations"] == 2
        assert summary["subtask_counts"] == {"s1": 1, "s2": 1}
        assert summary["max_reached"] is False


# ── ComplexitySignals tests ───────────────────────────────────────────


class TestComplexitySignals:
    """Test complexity signal detection using real ComplexitySignals objects."""

    def test_default_signals(self):
        """Default signals should have sensible values."""
        signals = ComplexitySignals()
        assert signals.word_count == 0
        assert signals.has_code_keywords is False
        assert signals.question_type == "unknown"

    def test_classify_trivial(self):
        """Simple questions should classify as TRIVIAL or SIMPLE."""
        from src.proactive_delegation import classify_task_complexity
        complexity, signals = classify_task_complexity("What is 2+2?")
        assert complexity in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE)

    def test_classify_complex_triggers(self):
        """Complex architecture prompts should classify as MODERATE or COMPLEX."""
        from src.proactive_delegation import classify_task_complexity
        complexity, signals = classify_task_complexity(
            "Design and implement a distributed caching system with "
            "consistency guarantees, sharding, and replication across "
            "multiple data centers with failover handling"
        )
        assert complexity in (TaskComplexity.MODERATE, TaskComplexity.COMPLEX)
        assert signals.has_architecture_keywords or signals.has_multi_step_keywords

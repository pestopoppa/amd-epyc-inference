"""Tests for proactive parallel delegation pipeline (Item C wiring).

Tests: _parse_plan_steps(), _execute_proactive() complexity gating,
architect bypass, and full mock flow.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.api.routes.chat_pipeline import _parse_plan_steps


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

        with patch("src.api.routes.chat_pipeline.features") as mock_feat:
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
        with patch("src.api.routes.chat_pipeline.features") as mock_feat:
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

        with patch("src.api.routes.chat_pipeline.features") as mock_feat:
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

        with patch("src.api.routes.chat_pipeline.features") as mock_feat:
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

        with patch("src.api.routes.chat_pipeline.features") as mock_feat:
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

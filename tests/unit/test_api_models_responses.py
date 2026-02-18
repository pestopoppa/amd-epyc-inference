#!/usr/bin/env python3
"""Unit tests for API response models."""

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Load the models file directly to avoid src.api.__init__ which transitively
# imports pydantic_graph and other heavy dependencies not needed for unit tests.
_ROOT = Path(__file__).resolve().parents[2] / "src" / "api" / "models"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f"{name.split('.')[-1]}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module("src.api.models.responses")
ChatResponse = _mod.ChatResponse
DelegationEvent = _mod.DelegationEvent
GateResultModel = _mod.GateResultModel
GatesResponse = _mod.GatesResponse
HealthResponse = _mod.HealthResponse
StatsResponse = _mod.StatsResponse
ToolTiming = _mod.ToolTiming


class TestToolTiming:
    """Test ToolTiming model."""

    def test_minimal(self):
        tt = ToolTiming(tool_name="python_exec")
        assert tt.tool_name == "python_exec"
        assert tt.elapsed_ms == 0.0
        assert tt.success is True

    def test_failed_invocation(self):
        tt = ToolTiming(tool_name="web_search", elapsed_ms=1500.0, success=False)
        assert tt.success is False
        assert tt.elapsed_ms == 1500.0

    def test_tool_name_required(self):
        with pytest.raises(ValidationError):
            ToolTiming()  # type: ignore[call-arg]


class TestDelegationEvent:
    """Test DelegationEvent model."""

    def test_minimal(self):
        de = DelegationEvent(from_role="frontdoor", to_role="coder")
        assert de.from_role == "frontdoor"
        assert de.to_role == "coder"

    def test_defaults(self):
        de = DelegationEvent(from_role="a", to_role="b")
        assert de.task_summary == ""
        assert de.success is None
        assert de.elapsed_ms == 0.0
        assert de.tokens_generated == 0

    def test_full(self):
        de = DelegationEvent(
            from_role="architect",
            to_role="coder",
            task_summary="implement auth",
            success=True,
            elapsed_ms=3200.0,
            tokens_generated=500,
        )
        assert de.task_summary == "implement auth"
        assert de.tokens_generated == 500

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            DelegationEvent(from_role="a")  # type: ignore[call-arg]


class TestChatResponse:
    """Test ChatResponse field defaults and required fields."""

    def _minimal(self, **kwargs):
        defaults = {
            "answer": "Hello",
            "turns": 1,
            "elapsed_seconds": 0.5,
            "mock_mode": True,
        }
        defaults.update(kwargs)
        return ChatResponse(**defaults)

    def test_minimal(self):
        resp = self._minimal()
        assert resp.answer == "Hello"
        assert resp.turns == 1

    def test_defaults(self):
        resp = self._minimal()
        assert resp.tokens_used == 0
        assert resp.real_mode is False
        assert resp.cache_stats is None
        assert resp.routed_to == ""
        assert resp.role_history == []
        assert resp.routing_strategy == ""
        assert resp.mode == ""
        assert resp.tokens_generated == 0
        assert resp.formalization_applied is False
        assert resp.tools_used == 0
        assert resp.tools_called == []
        assert resp.tool_timings == []
        assert resp.tool_chains == []
        assert resp.delegation_events == []
        assert resp.tools_success is None
        assert resp.delegation_success is None
        assert resp.prompt_eval_ms == 0.0
        assert resp.generation_ms == 0.0
        assert resp.predicted_tps == 0.0
        assert resp.http_overhead_ms == 0.0
        assert resp.error_code is None
        assert resp.error_detail is None
        assert resp.cheap_first_attempted is False
        assert resp.cheap_first_passed is None
        assert resp.think_harder_attempted is False
        assert resp.think_harder_succeeded is None
        assert resp.grammar_enforced is False
        assert resp.parallel_tools_used is False
        assert resp.cache_affinity_bonus == 0.0
        assert resp.cost_dimensions == {}
        assert resp.skills_retrieved == 0
        assert resp.skill_ids == []

    def test_with_tool_timings(self):
        resp = self._minimal(
            tool_timings=[
                ToolTiming(tool_name="python_exec", elapsed_ms=100.0),
                ToolTiming(tool_name="web_search", elapsed_ms=200.0, success=False),
            ]
        )
        assert len(resp.tool_timings) == 2

    def test_with_delegation_events(self):
        resp = self._minimal(
            delegation_events=[
                DelegationEvent(from_role="a", to_role="b", success=True)
            ]
        )
        assert len(resp.delegation_events) == 1

    def test_with_tool_chains(self):
        resp = self._minimal(
            tool_chains=[
                {
                    "chain_id": "ch_123",
                    "caller_type": "chain",
                    "tools": ["read_file", "list_directory"],
                    "elapsed_ms": 22.5,
                    "success": True,
                }
            ]
        )
        assert len(resp.tool_chains) == 1
        assert resp.tool_chains[0]["chain_id"] == "ch_123"

    def test_required_fields_missing(self):
        with pytest.raises(ValidationError):
            ChatResponse(answer="Hi")  # type: ignore[call-arg]


class TestHealthResponse:
    """Test HealthResponse model."""

    def test_minimal(self):
        hr = HealthResponse(status="ok")
        assert hr.status == "ok"

    def test_defaults(self):
        hr = HealthResponse(status="ok")
        assert hr.models_loaded == 0
        assert hr.mock_mode_available is True
        assert hr.version == "0.1.0"
        assert hr.backend_health is None
        assert hr.backend_probes is None
        assert hr.knowledge_tools is None

    def test_degraded_with_probes(self):
        hr = HealthResponse(
            status="degraded",
            models_loaded=2,
            backend_probes={"coder": {"url": "http://localhost:8081", "ok": True}},
        )
        assert hr.status == "degraded"
        assert hr.models_loaded == 2


class TestGateResultModel:
    """Test GateResultModel."""

    def test_passed_gate(self):
        gr = GateResultModel(
            gate_name="shellcheck", passed=True, exit_code=0, elapsed_seconds=1.2
        )
        assert gr.passed is True
        assert gr.errors == []
        assert gr.warnings == []

    def test_failed_gate_with_errors(self):
        gr = GateResultModel(
            gate_name="format",
            passed=False,
            exit_code=1,
            elapsed_seconds=0.5,
            errors=["file.sh: line 10: syntax error"],
            warnings=["unused variable"],
        )
        assert gr.passed is False
        assert len(gr.errors) == 1
        assert len(gr.warnings) == 1


class TestGatesResponse:
    """Test GatesResponse model."""

    def test_all_passed(self):
        results = [
            GateResultModel(
                gate_name="shellcheck", passed=True, exit_code=0, elapsed_seconds=1.0
            ),
            GateResultModel(
                gate_name="format", passed=True, exit_code=0, elapsed_seconds=0.5
            ),
        ]
        resp = GatesResponse(
            results=results, all_passed=True, total_elapsed_seconds=1.5
        )
        assert resp.all_passed is True
        assert len(resp.results) == 2
        assert resp.total_elapsed_seconds == 1.5

    def test_some_failed(self):
        results = [
            GateResultModel(
                gate_name="lint", passed=False, exit_code=1, elapsed_seconds=0.3
            )
        ]
        resp = GatesResponse(
            results=results, all_passed=False, total_elapsed_seconds=0.3
        )
        assert resp.all_passed is False


class TestStatsResponse:
    """Test StatsResponse model."""

    def test_instantiation(self):
        stats = StatsResponse(
            total_requests=100,
            total_turns=350,
            average_turns_per_request=3.5,
            mock_requests=80,
            real_requests=20,
        )
        assert stats.total_requests == 100
        assert stats.total_turns == 350
        assert stats.average_turns_per_request == 3.5
        assert stats.mock_requests == 80
        assert stats.real_requests == 20

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            StatsResponse(total_requests=1)  # type: ignore[call-arg]

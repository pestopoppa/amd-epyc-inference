#!/usr/bin/env python3
"""Tests for seeding_types — dataclasses, constants, and shared state."""

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "benchmark"))

from seeding_types import (
    ACTION_ARCHITECT,
    ACTION_SELF_DIRECT,
    ACTION_SELF_REPL,
    ACTION_WORKER,
    ARCHITECT_MODES,
    ARCHITECT_ROLES,
    ComparativeResult,
    ESCALATION_REWARD,
    HEAVY_PORTS,
    HealthCheckError,
    ROLE_COST_TIER,
    ROLE_PORT,
    RoleResult,
    THREE_WAY_ACTIONS,
    _State,
    state,
)


class TestRoleResult:
    """RoleResult dataclass fields and defaults."""

    def test_required_fields(self):
        rr = RoleResult(role="frontdoor", mode="direct", answer="42", passed=True, elapsed_seconds=1.5)
        assert rr.role == "frontdoor"
        assert rr.mode == "direct"
        assert rr.answer == "42"
        assert rr.passed is True
        assert rr.elapsed_seconds == 1.5

    def test_default_values(self):
        rr = RoleResult(role="x", mode="y", answer="", passed=False, elapsed_seconds=0.0)
        assert rr.error is None
        assert rr.error_type == "none"
        assert rr.tokens_generated == 0
        assert rr.tools_used == 0
        assert rr.tools_called == []
        assert rr.delegation_events == []
        assert rr.tools_success is None
        assert rr.delegation_success is None
        assert rr.routed_to == ""
        assert rr.role_history == []
        assert rr.predicted_tps == 0.0
        assert rr.generation_ms == 0.0

    def test_asdict_roundtrip(self):
        rr = RoleResult(
            role="coder", mode="repl", answer="hello", passed=True,
            elapsed_seconds=2.0, tokens_generated=50, predicted_tps=25.0,
        )
        d = asdict(rr)
        assert d["role"] == "coder"
        assert d["tokens_generated"] == 50
        assert isinstance(d["tools_called"], list)

    def test_mutable_defaults_not_shared(self):
        """Each instance gets its own list for mutable defaults."""
        rr1 = RoleResult(role="a", mode="b", answer="", passed=False, elapsed_seconds=0.0)
        rr2 = RoleResult(role="c", mode="d", answer="", passed=False, elapsed_seconds=0.0)
        rr1.tools_called.append("tool1")
        assert rr2.tools_called == []


class TestComparativeResult:
    """ComparativeResult dataclass and serialization."""

    def test_required_fields(self):
        cr = ComparativeResult(suite="thinking", question_id="q1", prompt="p", expected="e")
        assert cr.suite == "thinking"
        assert cr.question_id == "q1"

    def test_default_values(self):
        cr = ComparativeResult(suite="s", question_id="q", prompt="p", expected="e")
        assert cr.dataset_source == "yaml"
        assert cr.prompt_hash == ""
        assert cr.timestamp == ""
        assert cr.role_results == {}
        assert cr.rewards == {}
        assert cr.rewards_injected == 0

    def test_asdict_roundtrip_with_nested(self):
        rr = RoleResult(role="coder", mode="direct", answer="ok", passed=True, elapsed_seconds=1.0)
        cr = ComparativeResult(
            suite="math", question_id="m1", prompt="2+2", expected="4",
            role_results={"coder:direct": rr},
            rewards={"coder:direct": 1.0},
        )
        d = asdict(cr)
        assert "coder:direct" in d["role_results"]
        assert d["role_results"]["coder:direct"]["role"] == "coder"
        assert d["rewards"]["coder:direct"] == 1.0


class TestState:
    """Shared mutable state singleton."""

    def test_shutdown_starts_false(self):
        s = _State()
        assert s.shutdown is False

    def test_close_poll_client_noop_when_none(self):
        s = _State()
        s.close_poll_client()  # Should not raise

    def test_shutdown_toggle(self):
        s = _State()
        s.shutdown = True
        assert s.shutdown is True
        s.shutdown = False
        assert s.shutdown is False

    def test_global_state_exists(self):
        assert state is not None
        assert isinstance(state, _State)


class TestConstants:
    """Verify key constants have expected structure."""

    def test_three_way_actions(self):
        assert ACTION_SELF_DIRECT in THREE_WAY_ACTIONS
        assert ACTION_SELF_REPL in THREE_WAY_ACTIONS
        assert ACTION_ARCHITECT in THREE_WAY_ACTIONS
        assert ACTION_WORKER in THREE_WAY_ACTIONS
        assert len(THREE_WAY_ACTIONS) == 4

    def test_role_cost_tier_ordering(self):
        # Workers are cheapest
        assert ROLE_COST_TIER["worker_explore"] < ROLE_COST_TIER["frontdoor"]
        # Architects are most expensive
        assert ROLE_COST_TIER["architect_coding"] > ROLE_COST_TIER["coder_escalation"]

    def test_architect_roles_and_modes(self):
        assert "architect_general" in ARCHITECT_ROLES
        assert "architect_coding" in ARCHITECT_ROLES
        assert "direct" in ARCHITECT_MODES
        assert "delegated" in ARCHITECT_MODES

    def test_role_port_complete(self):
        for role in ["frontdoor", "coder_escalation", "architect_general", "worker_vision"]:
            assert role in ROLE_PORT

    def test_heavy_ports_subset_of_model_ports(self):
        for port in HEAVY_PORTS:
            assert isinstance(port, int)

    def test_escalation_reward_positive(self):
        assert ESCALATION_REWARD > 0
        assert ESCALATION_REWARD <= 1.0

    def test_health_check_error_is_exception(self):
        with pytest.raises(HealthCheckError):
            raise HealthCheckError("test")

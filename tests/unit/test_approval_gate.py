#!/usr/bin/env python3
"""Tests for the approval gate module."""

import pytest
from unittest.mock import MagicMock

from src.features import Features, set_features, reset_features
from src.graph.approval_gate import (
    ApprovalDecision,
    AutoApproveCallback,
    HaltReason,
    HaltState,
    should_halt,
    request_approval_for_escalation,
)


@pytest.fixture(autouse=True)
def _reset_features():
    """Reset features after each test."""
    yield
    reset_features()


class TestHaltState:
    """Tests for HaltState dataclass."""

    def test_creation(self):
        halt = HaltState(
            reason=HaltReason.ESCALATION,
            from_role="coder_escalation",
            to_role="architect_general",
            description="Retries exhausted",
        )
        assert halt.reason == HaltReason.ESCALATION
        assert halt.from_role == "coder_escalation"
        assert halt.to_role == "architect_general"

    def test_destructive_tool_halt(self):
        halt = HaltState(
            reason=HaltReason.DESTRUCTIVE_TOOL,
            side_effects=["modifies_files", "system_state"],
        )
        assert halt.reason == HaltReason.DESTRUCTIVE_TOOL
        assert "modifies_files" in halt.side_effects


class TestAutoApproveCallback:
    """Tests for default auto-approve callback."""

    def test_always_approves(self):
        cb = AutoApproveCallback()
        halt = HaltState(reason=HaltReason.ESCALATION)
        assert cb.request_approval(halt) == ApprovalDecision.APPROVE

    def test_approves_destructive(self):
        cb = AutoApproveCallback()
        halt = HaltState(reason=HaltReason.DESTRUCTIVE_TOOL)
        assert cb.request_approval(halt) == ApprovalDecision.APPROVE


class TestShouldHalt:
    """Tests for should_halt() function."""

    def test_disabled_returns_none(self):
        set_features(Features(approval_gates=False))
        assert should_halt("coder_escalation", "architect_general") is None

    def test_tier_crossing_triggers_escalation(self):
        set_features(Features(approval_gates=True))
        result = should_halt("worker_general", "coder_escalation")
        assert result == HaltReason.ESCALATION

    def test_same_tier_no_halt(self):
        set_features(Features(approval_gates=True))
        # Worker to worker — same tier
        result = should_halt("worker_general", "worker_math")
        assert result is None

    def test_architect_triggers_high_cost(self):
        set_features(Features(approval_gates=True))
        # Coder to architect — both Tier B, so triggers HIGH_COST
        result = should_halt("coder_escalation", "architect_general")
        assert result == HaltReason.HIGH_COST


class TestRequestApproval:
    """Tests for request_approval_for_escalation()."""

    def test_no_callback_auto_approves(self):
        set_features(Features(approval_gates=False))
        ctx = MagicMock()
        ctx.deps.approval_callback = None
        decision = request_approval_for_escalation(
            ctx, "coder_escalation", "architect_general", "test"
        )
        assert decision == ApprovalDecision.APPROVE

    def test_callback_called_on_halt(self):
        set_features(Features(approval_gates=True, resume_tokens=False))
        callback = MagicMock()
        callback.request_approval.return_value = ApprovalDecision.REJECT

        ctx = MagicMock()
        ctx.deps.approval_callback = callback
        ctx.state.pending_approval = None

        decision = request_approval_for_escalation(
            ctx, "worker_general", "coder_escalation", "test reason"
        )
        assert decision == ApprovalDecision.REJECT
        callback.request_approval.assert_called_once()

        # Verify halt state was passed
        halt = callback.request_approval.call_args[0][0]
        assert halt.reason == HaltReason.ESCALATION
        assert halt.from_role == "worker_general"
        assert halt.to_role == "coder_escalation"

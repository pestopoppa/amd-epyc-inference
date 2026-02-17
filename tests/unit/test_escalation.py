#!/usr/bin/env python3
"""Unit tests for src/escalation.py."""

import pytest

from src.escalation import (
    ErrorCategory,
    EscalationAction,
    EscalationConfig,
    EscalationContext,
    EscalationDecision,
    EscalationPolicy,
    decide,
    get_policy,
)
from src.roles import Role


class TestErrorCategory:
    """Test ErrorCategory enum."""

    def test_error_categories(self):
        """Test ErrorCategory enum values."""
        assert ErrorCategory.CODE == "code"
        assert ErrorCategory.LOGIC == "logic"
        assert ErrorCategory.TIMEOUT == "timeout"
        assert ErrorCategory.SCHEMA == "schema"
        assert ErrorCategory.FORMAT == "format"
        assert ErrorCategory.EARLY_ABORT == "early_abort"
        assert ErrorCategory.UNKNOWN == "unknown"

    def test_error_category_from_string(self):
        """Test ErrorCategory can be created from string."""
        assert ErrorCategory("code") == ErrorCategory.CODE
        assert ErrorCategory("timeout") == ErrorCategory.TIMEOUT


class TestEscalationContext:
    """Test EscalationContext normalization."""

    def test_role_string_to_enum(self):
        """Test role string is converted to enum."""
        context = EscalationContext(current_role="coder_escalation")
        assert context.current_role == Role.CODER_ESCALATION

    def test_error_category_string_to_enum(self):
        """Test error category string is converted to enum."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            error_category="timeout",
        )
        assert context.error_category == ErrorCategory.TIMEOUT

    def test_unknown_role_remains_string(self):
        """Test unknown role remains as string."""
        context = EscalationContext(current_role="unknown_role")
        assert context.current_role == "unknown_role"

    def test_invalid_error_category_becomes_unknown(self):
        """Test invalid error category becomes UNKNOWN."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            error_category="invalid",
        )
        assert context.error_category == ErrorCategory.UNKNOWN


class TestEscalationDecision:
    """Test EscalationDecision properties."""

    def test_should_escalate(self):
        """Test should_escalate property."""
        decision = EscalationDecision(action=EscalationAction.ESCALATE)
        assert decision.should_escalate is True

        decision2 = EscalationDecision(action=EscalationAction.RETRY)
        assert decision2.should_escalate is False

    def test_should_retry(self):
        """Test should_retry property."""
        decision = EscalationDecision(action=EscalationAction.RETRY)
        assert decision.should_retry is True

        decision2 = EscalationDecision(action=EscalationAction.ESCALATE)
        assert decision2.should_retry is False

    def test_should_delegate(self):
        """Test should_delegate property."""
        decision = EscalationDecision(action=EscalationAction.DELEGATE)
        assert decision.should_delegate is True

    def test_is_terminal(self):
        """Test is_terminal property."""
        fail_decision = EscalationDecision(action=EscalationAction.FAIL)
        assert fail_decision.is_terminal is True

        skip_decision = EscalationDecision(action=EscalationAction.SKIP)
        assert skip_decision.is_terminal is True

        retry_decision = EscalationDecision(action=EscalationAction.RETRY)
        assert retry_decision.is_terminal is False


class TestEscalationPolicy:
    """Test EscalationPolicy decision logic."""

    @pytest.fixture
    def policy(self):
        """Create a policy with known config."""
        config = EscalationConfig(
            max_retries=3,
            max_escalations=2,
            optional_gates=frozenset({"lint", "format"}),
        )
        return EscalationPolicy(config)

    def test_retry_before_escalation(self, policy):
        """Test policy retries before escalating."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=0,
            error_category=ErrorCategory.CODE,
        )
        decision = policy.decide(context)

        assert decision.action == EscalationAction.RETRY
        assert decision.retries_remaining == 2

    def test_escalate_after_max_retries(self, policy):
        """Test policy escalates after max retries."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=3,
            error_category=ErrorCategory.CODE,
            escalation_count=0,
        )
        decision = policy.decide(context)

        assert decision.action == EscalationAction.ESCALATE
        assert decision.target_role == Role.CODER_ESCALATION

    def test_skip_optional_gate_on_timeout(self, policy):
        """Test policy skips optional gates on timeout."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=1,
            error_category=ErrorCategory.TIMEOUT,
            gate_name="lint",
        )
        decision = policy.decide(context)

        assert decision.action == EscalationAction.SKIP
        assert "optional gate" in decision.reason.lower()

    def test_no_skip_required_gate_on_timeout(self, policy):
        """Test policy doesn't skip required gates on timeout."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=1,
            error_category=ErrorCategory.TIMEOUT,
            gate_name="typecheck",  # Not in optional_gates
        )
        decision = policy.decide(context)

        # Should retry, not skip
        assert decision.action == EscalationAction.RETRY

    def test_early_abort_immediate_escalation(self, policy):
        """Test early abort triggers immediate escalation."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=0,  # Even on first failure
            error_category=ErrorCategory.EARLY_ABORT,
            escalation_count=0,
        )
        decision = policy.decide(context)

        assert decision.action == EscalationAction.ESCALATE
        assert decision.target_role == Role.CODER_ESCALATION

    def test_format_errors_never_escalate(self, policy):
        """Test format errors retry only, never escalate."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=3,  # Exhausted retries
            error_category=ErrorCategory.FORMAT,
        )
        decision = policy.decide(context)

        # Should fail, not escalate
        assert decision.action == EscalationAction.FAIL
        assert "format error" in decision.reason.lower()

    def test_schema_errors_escalate_on_capability_gap_signature(self, policy):
        """Schema errors escalate only when retries exhausted and signature indicates capability gap."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=3,  # Exhausted retries
            error_category=ErrorCategory.SCHEMA,
            error_message="Schema mismatch: required property 'steps' missing",
            escalation_count=0,
        )
        decision = policy.decide(context)

        assert decision.action == EscalationAction.ESCALATE
        assert decision.target_role == Role.CODER_ESCALATION

    def test_schema_parser_errors_fail_after_retries(self, policy):
        """Schema parser/transient signatures fail after retries instead of escalating."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=3,  # Exhausted retries
            error_category=ErrorCategory.SCHEMA,
            error_message="JSON decode error: expecting value at line 1",
            escalation_count=0,
        )
        decision = policy.decide(context)

        assert decision.action == EscalationAction.FAIL

    def test_terminal_role_explore_fallback(self, policy):
        """Test terminal role (architect) falls back to EXPLORE."""
        context = EscalationContext(
            current_role=Role.ARCHITECT_GENERAL,
            failure_count=3,
            error_category=ErrorCategory.CODE,
        )
        decision = policy.decide(context)

        # Architect has no escalation target
        assert decision.action == EscalationAction.EXPLORE
        assert "exploration" in decision.reason.lower()

    def test_max_escalations_enforced(self, policy):
        """Test max escalations limit is enforced."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=3,
            error_category=ErrorCategory.CODE,
            escalation_count=2,  # Already at max
        )
        decision = policy.decide(context)

        assert decision.action == EscalationAction.FAIL
        assert "max escalations" in decision.reason.lower()

    def test_unknown_role_fails(self, policy):
        """Test unknown role fails immediately."""
        context = EscalationContext(current_role="unknown_role")
        decision = policy.decide(context)

        assert decision.action == EscalationAction.FAIL
        assert "unknown role" in decision.reason.lower()

    def test_get_escalation_path(self, policy):
        """Test get_escalation_path returns full chain."""
        path = policy.get_escalation_path(Role.WORKER_GENERAL)

        assert len(path) == 3
        assert path[0] == Role.WORKER_GENERAL
        assert path[1] == Role.CODER_ESCALATION
        assert path[2] == Role.ARCHITECT_CODING


class TestThinkHarder:
    """Test THINK_HARDER escalation action."""

    @pytest.fixture
    def policy(self):
        config = EscalationConfig(
            max_retries=3,
            max_escalations=2,
            optional_gates=frozenset({"lint", "format"}),
        )
        return EscalationPolicy(config)

    def test_think_harder_on_penultimate_retry(self, policy):
        """THINK_HARDER fires on penultimate retry (failure_count == max_retries - 1)."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=2,  # max_retries=3, so this is penultimate
            error_category=ErrorCategory.CODE,
        )
        decision = policy.decide(context)

        assert decision.action == EscalationAction.THINK_HARDER
        assert decision.target_role == Role.WORKER_GENERAL  # Same model
        assert decision.config_override is not None
        assert "cot_prefix" in decision.config_override
        assert decision.config_override["n_tokens"] == 4096

    def test_think_harder_has_correct_properties(self, policy):
        """THINK_HARDER decision has should_think_harder=True."""
        decision = EscalationDecision(action=EscalationAction.THINK_HARDER)
        assert decision.should_think_harder is True
        assert decision.should_escalate is False
        assert decision.should_retry is False

    def test_regular_retry_before_penultimate(self, policy):
        """Normal RETRY still fires before penultimate retry."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=0,
            error_category=ErrorCategory.CODE,
        )
        decision = policy.decide(context)
        assert decision.action == EscalationAction.RETRY

    def test_escalate_after_think_harder(self, policy):
        """After THINK_HARDER fails, next failure triggers ESCALATE."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=3,  # All retries (including think-harder) exhausted
            error_category=ErrorCategory.CODE,
            escalation_count=0,
        )
        decision = policy.decide(context)
        assert decision.action == EscalationAction.ESCALATE

    def test_think_harder_not_for_format_errors(self, policy):
        """Format errors don't trigger THINK_HARDER (just retry)."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=2,
            error_category=ErrorCategory.FORMAT,
        )
        decision = policy.decide(context)
        # Format errors use no_escalate_categories path → RETRY (not THINK_HARDER)
        assert decision.action == EscalationAction.RETRY
        assert decision.action != EscalationAction.THINK_HARDER


class TestGlobalPolicy:
    """Test global policy functions."""

    def test_get_policy_singleton(self):
        """Test get_policy returns same instance."""
        policy1 = get_policy()
        policy2 = get_policy()
        assert policy1 is policy2

    def test_decide_uses_global_policy(self):
        """Test decide() function uses global policy."""
        context = EscalationContext(
            current_role=Role.WORKER_GENERAL,
            failure_count=0,
        )
        decision = decide(context)

        # Should get a valid decision
        assert isinstance(decision, EscalationDecision)

"""Unit tests for RoutingFacade."""

from src.escalation import EscalationAction, EscalationContext, EscalationPolicy, ErrorCategory
from src.failure_router import LearnedEscalationResult
from src.routing_facade import RoutingFacade


class _DummyLearned:
    def __init__(self, result: LearnedEscalationResult):
        self._result = result

    def query(self, _context):
        return self._result


def test_facade_uses_rules_by_default():
    facade = RoutingFacade(policy=EscalationPolicy(), learned=None)
    ctx = EscalationContext(
        current_role="frontdoor",
        failure_count=0,
        error_category=ErrorCategory.CODE,
    )
    decision = facade.decide(ctx)
    assert decision.action == EscalationAction.RETRY


def test_facade_uses_learned_when_confident():
    learned = _DummyLearned(
        LearnedEscalationResult(
            should_use_learned=True,
            suggested_action="escalate",
            suggested_role="architect_general",
            confidence=0.9,
            similar_cases=5,
        )
    )
    facade = RoutingFacade(policy=EscalationPolicy(), learned=learned, confidence_threshold=0.7)
    ctx = EscalationContext(
        current_role="coder_primary",
        failure_count=2,
        error_category=ErrorCategory.CODE,
    )
    decision = facade.decide(ctx)
    assert decision.action == EscalationAction.ESCALATE
    assert decision.target_role.value == "architect_general"
    assert "learned" in decision.reason


def test_facade_rejects_learned_for_format():
    learned = _DummyLearned(
        LearnedEscalationResult(
            should_use_learned=True,
            suggested_action="escalate",
            suggested_role="architect_general",
            confidence=0.9,
            similar_cases=5,
        )
    )
    facade = RoutingFacade(policy=EscalationPolicy(), learned=learned, confidence_threshold=0.7)
    ctx = EscalationContext(
        current_role="coder_primary",
        failure_count=2,
        error_category=ErrorCategory.FORMAT,
    )
    decision = facade.decide(ctx)
    assert decision.action != EscalationAction.ESCALATE


def test_facade_rejects_learned_for_infra():
    learned = _DummyLearned(
        LearnedEscalationResult(
            should_use_learned=True,
            suggested_action="escalate",
            suggested_role="architect_general",
            confidence=0.9,
            similar_cases=5,
        )
    )
    facade = RoutingFacade(policy=EscalationPolicy(), learned=learned, confidence_threshold=0.7)
    ctx = EscalationContext(
        current_role="coder_primary",
        failure_count=2,
        error_category=ErrorCategory.INFRASTRUCTURE,
    )
    decision = facade.decide(ctx)
    assert "learned" not in decision.reason

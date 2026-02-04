#!/usr/bin/env python3
"""RoutingFacade: unified entry point for escalation decisions.

Rules (EscalationPolicy) are authoritative. Learned escalation is advisory.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.escalation import (
    EscalationAction,
    EscalationContext,
    EscalationDecision,
    EscalationPolicy,
    ErrorCategory,
)
from src.roles import Role

if True:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from src.failure_router import LearnedEscalationPolicy


@dataclass
class RoutingFacade:
    """Single entry point for all escalation/routing decisions."""

    policy: EscalationPolicy
    learned: "LearnedEscalationPolicy | None" = None
    confidence_threshold: float = 0.7

    def __post_init__(self) -> None:
        self._strategy_counts = {"learned": 0, "rules": 0}

    def decide(self, context: EscalationContext) -> EscalationDecision:
        """Decide escalation with learned advisory + rule-based fallback."""
        if self.learned is not None:
            learned_result = self.learned.query(_to_failure_context(context))
            if (
                learned_result
                and learned_result.should_use_learned
                and learned_result.confidence >= self.confidence_threshold
            ):
                decision = self._validate_learned(context, learned_result)
                if decision is not None:
                    self._strategy_counts["learned"] += 1
                    return decision

        self._strategy_counts["rules"] += 1
        return self.policy.decide(context)

    def _validate_learned(self, context: EscalationContext, learned_result) -> EscalationDecision | None:
        """Reject learned suggestions that violate rule constraints."""
        if context.error_category in (ErrorCategory.FORMAT, ErrorCategory.SCHEMA):
            if learned_result.suggested_action == "escalate":
                return None
        if context.error_category == ErrorCategory.INFRASTRUCTURE:
            return None

        action = _map_action(learned_result.suggested_action)
        if action is None:
            return None

        if action == EscalationAction.RETRY:
            retries_remaining = max(self.policy.config.max_retries - context.failure_count - 1, 0)
            return EscalationDecision(
                action=EscalationAction.RETRY,
                target_role=context.current_role if isinstance(context.current_role, Role) else None,
                reason=(
                    f"learned retry (confidence={learned_result.confidence:.2f}, "
                    f"similar_cases={learned_result.similar_cases})"
                ),
                retries_remaining=retries_remaining,
            )

        if action == EscalationAction.ESCALATE:
            if context.escalation_count >= self.policy.config.max_escalations:
                return None
            target = _resolve_role(learned_result.suggested_role)
            if target is None:
                return None
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                target_role=target,
                reason=(
                    f"learned escalate (confidence={learned_result.confidence:.2f}, "
                    f"similar_cases={learned_result.similar_cases})"
                ),
            )

        if action == EscalationAction.FAIL:
            return EscalationDecision(
                action=EscalationAction.FAIL,
                reason=(
                    f"learned fail (confidence={learned_result.confidence:.2f}, "
                    f"similar_cases={learned_result.similar_cases})"
                ),
            )

        return None


def _map_action(action: str | None) -> EscalationAction | None:
    if not action:
        return None
    lowered = action.lower()
    if lowered == "retry":
        return EscalationAction.RETRY
    if lowered == "escalate":
        return EscalationAction.ESCALATE
    if lowered == "fail":
        return EscalationAction.FAIL
    return None


def _resolve_role(role: str | None) -> Role | None:
    if not role:
        return None
    return Role.from_string(role)


def _to_failure_context(context: EscalationContext):
    """Convert EscalationContext to FailureContext for learned policy."""
    from src.failure_router import FailureContext
    role_str = (
        context.current_role.value
        if isinstance(context.current_role, Role)
        else str(context.current_role)
    )
    return FailureContext(
        role=role_str,
        failure_count=context.failure_count,
        error_category=context.error_category,
        gate_name=context.gate_name,
        error_message=context.error_message,
        task_id=context.task_id,
        escalation_count=context.escalation_count,
    )

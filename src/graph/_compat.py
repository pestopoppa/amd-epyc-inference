"""Compatibility bridge for gradual migration of old-API callers.

Provides converters between the legacy EscalationContext/EscalationDecision
types and the new graph TaskState/TaskResult types. These helpers allow
call sites that haven't been fully migrated to continue working.
"""

from __future__ import annotations

from src.escalation import (
    EscalationAction,
    EscalationConfig,
    EscalationContext,
    EscalationDecision,
)
from src.graph.state import GraphConfig, TaskResult, TaskState
from src.roles import Role


def context_to_state(ctx: EscalationContext) -> TaskState:
    """Convert an EscalationContext to a TaskState.

    Used by callers that still create EscalationContext objects but need
    to interact with the graph.
    """
    role = ctx.current_role if isinstance(ctx.current_role, Role) else Role.FRONTDOOR
    return TaskState(
        task_id=ctx.task_id,
        current_role=role,
        consecutive_failures=ctx.failure_count,
        escalation_count=ctx.escalation_count,
        last_error=ctx.error_message,
        role_history=[str(role)],
    )


def result_to_decision(result: TaskResult) -> EscalationDecision:
    """Convert a TaskResult to an EscalationDecision.

    Used by callers that expect the old EscalationDecision return type.
    """
    if result.success:
        last_role = Role.from_string(result.role_history[-1]) if result.role_history else None
        return EscalationDecision(
            action=EscalationAction.RETRY,
            target_role=last_role,
            reason="Graph execution succeeded",
            retries_remaining=0,
        )

    # Failure — map to appropriate action
    answer = result.answer
    if "Max turns" in answer:
        return EscalationDecision(
            action=EscalationAction.FAIL,
            reason="Max turns reached",
        )
    if "Terminal role" in answer:
        return EscalationDecision(
            action=EscalationAction.EXPLORE,
            reason="Terminal role exhausted",
        )

    return EscalationDecision(
        action=EscalationAction.FAIL,
        reason=answer[:200] if answer else "Graph execution failed",
    )


def config_to_graph_config(esc_config: EscalationConfig) -> GraphConfig:
    """Convert an EscalationConfig to a GraphConfig."""
    return GraphConfig(
        max_retries=esc_config.max_retries,
        max_escalations=esc_config.max_escalations,
        optional_gates=esc_config.optional_gates,
        no_escalate_categories=esc_config.no_escalate_categories,
    )

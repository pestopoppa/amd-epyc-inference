"""Halt-and-resume approval protocol (Lobster pattern).

Provides human approval gates at escalation boundaries and destructive
tool invocations. Graph halts, serializes state via resume token,
waits for approval, then resumes.

Only active when features().approval_gates is True.

Depends on:
- Phase 1A (side_effect_tracking) for destructive tool detection
- Phase 3B (resume_tokens) for state serialization

Halt triggers:
1. Escalation boundary — tier crossing (Worker→Coder, Coder→Architect)
2. Destructive tool — tool.destructive == True
3. High-cost invocation — architect-tier models (configurable threshold)

Usage:
    from src.graph.approval_gate import should_halt, HaltReason, AutoApproveCallback

    # In a graph node, before escalation:
    if should_halt(state, from_role, to_role, deps):
        halt = HaltState(reason=HaltReason.ESCALATION, ...)
        decision = deps.approval_callback.request_approval(halt)
        if decision == ApprovalDecision.REJECT:
            return End(TaskResult(success=False, answer="Rejected"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


class HaltReason(str, Enum):
    """Why the graph is halting for approval."""

    ESCALATION = "escalation"  # Tier crossing
    DESTRUCTIVE_TOOL = "destructive_tool"  # Tool with destructive=True
    HIGH_COST = "high_cost"  # Architect-tier invocation


class ApprovalDecision(str, Enum):
    """User's decision on a halt request."""

    APPROVE = "approve"
    REJECT = "reject"


@dataclass
class HaltState:
    """State captured when graph halts for approval."""

    reason: HaltReason
    from_role: str = ""
    to_role: str = ""
    resume_token: str = ""
    side_effects: list[str] = field(default_factory=list)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ApprovalCallback(Protocol):
    """Protocol for approval decision providers.

    Implementations can be synchronous (CLI prompt), async (web UI),
    or auto-approve (preserves current behavior).
    """

    def request_approval(self, halt: HaltState) -> ApprovalDecision: ...


class AutoApproveCallback:
    """Default callback that auto-approves everything.

    Preserves current behavior when approval_gates is disabled or
    no callback is injected into TaskDeps.
    """

    def request_approval(self, halt: HaltState) -> ApprovalDecision:
        log.debug("Auto-approving: %s (%s → %s)", halt.reason, halt.from_role, halt.to_role)
        return ApprovalDecision.APPROVE


# High-cost roles that trigger approval (architect tier)
_HIGH_COST_ROLES = {"architect_general", "architect_coding"}


def should_halt(
    from_role: str,
    to_role: str,
) -> HaltReason | None:
    """Determine if a transition should trigger an approval gate.

    Args:
        from_role: Current role.
        to_role: Target role.

    Returns:
        HaltReason if approval needed, None otherwise.
    """
    from src.features import features as _get_features

    if not _get_features().approval_gates:
        return None

    # Escalation boundary (tier crossing)
    from src.roles import get_tier

    from_tier = get_tier(from_role)
    to_tier = get_tier(to_role)
    if from_tier != to_tier:
        return HaltReason.ESCALATION

    # High-cost invocation
    if to_role in _HIGH_COST_ROLES:
        return HaltReason.HIGH_COST

    return None


def request_approval_for_escalation(
    ctx: Any,  # GraphRunContext
    from_role: str,
    to_role: str,
    reason: str,
) -> ApprovalDecision:
    """Request approval for an escalation, returning the decision.

    If no callback or feature disabled, auto-approves.

    Args:
        ctx: GraphRunContext with state and deps.
        from_role: Current role.
        to_role: Target role after escalation.
        reason: Human-readable reason for escalation.

    Returns:
        ApprovalDecision.APPROVE or ApprovalDecision.REJECT.
    """
    halt_reason = should_halt(from_role, to_role)
    if halt_reason is None:
        return ApprovalDecision.APPROVE

    callback = getattr(ctx.deps, "approval_callback", None)
    if callback is None:
        return ApprovalDecision.APPROVE

    # Build resume token if available
    resume_token = ""
    from src.features import features as _get_features

    if _get_features().resume_tokens:
        try:
            from src.graph.resume_token import ResumeToken

            token = ResumeToken.from_state(ctx.state, type(ctx).__name__)
            resume_token = token.encode()
        except Exception:
            pass

    halt = HaltState(
        reason=halt_reason,
        from_role=from_role,
        to_role=to_role,
        resume_token=resume_token,
        description=reason,
    )

    ctx.state.pending_approval = halt
    decision = callback.request_approval(halt)
    ctx.state.pending_approval = None

    log.info(
        "Approval gate: %s → %s: %s (decision: %s)",
        from_role, to_role, halt_reason, decision,
    )
    return decision

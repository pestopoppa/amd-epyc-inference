#!/usr/bin/env python3
"""Unified escalation logic for hierarchical orchestration.

This module is the single source of truth for escalation policy. All other
modules (api.py, failure_router.py, executor.py) should use this module
for escalation decisions.

Escalation Chain:
    worker → coder → architect (terminal)
    ingest → architect (terminal)
    frontdoor → coder → architect (terminal)

Usage:
    from src.escalation import EscalationPolicy, EscalationContext

    policy = EscalationPolicy()
    context = EscalationContext(
        current_role=Role.WORKER_GENERAL,
        failure_count=2,
        error_category=ErrorCategory.CODE,
    )
    decision = policy.decide(context)
    if decision.should_escalate:
        next_role = decision.target_role
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from src.roles import Role, get_escalation_chain

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors that affect escalation decisions.

    Different error categories trigger different escalation behaviors:
    - CODE/LOGIC: Standard retry → escalate flow
    - FORMAT/SCHEMA: Retry only, never escalate (formatting issues)
    - TIMEOUT: Skip if optional gate, fail otherwise
    - EARLY_ABORT: Immediate escalation (model showed failure signs)
    """

    CODE = "code"  # Syntax errors, type errors, test failures
    LOGIC = "logic"  # Wrong output, failed assertions
    TIMEOUT = "timeout"  # Gate or execution timeout
    SCHEMA = "schema"  # IR/JSON schema violations
    FORMAT = "format"  # Style/format issues
    EARLY_ABORT = "early_abort"  # Generation aborted due to predicted failure
    INFRASTRUCTURE = "infrastructure"  # Backend or network failure (not task failure)
    UNKNOWN = "unknown"  # Unclassified errors


class EscalationAction(str, Enum):
    """Actions that can result from an escalation decision."""

    RETRY = "retry"  # Retry with same role
    ESCALATE = "escalate"  # Escalate to next tier
    DELEGATE = "delegate"  # Route DOWN to a specific lower-tier role
    REVIEW = "review"  # Quality check by higher-tier model
    FAIL = "fail"  # Terminal failure
    SKIP = "skip"  # Skip the gate/step (for optional gates)
    EXPLORE = "explore"  # Fall back to REPL exploration (for terminal roles)


@dataclass
class EscalationContext:
    """Context for making an escalation decision.

    Attributes:
        current_role: The role that failed.
        failure_count: Number of failures at this role for this task.
        error_category: Category of the error.
        error_message: The error message (optional).
        gate_name: Name of the gate that failed (if applicable).
        task_id: Task identifier for tracking.
        escalation_count: How many times we've already escalated.
    """

    current_role: Role | str
    failure_count: int = 0
    error_category: ErrorCategory | str = ErrorCategory.UNKNOWN
    error_message: str = ""
    gate_name: str = ""
    task_id: str = ""
    escalation_count: int = 0
    # Model-initiated routing fields
    target_role_requested: str = ""  # Specific role requested by model

    def __post_init__(self) -> None:
        """Normalize types."""
        # Convert role string to enum
        if isinstance(self.current_role, str):
            role = Role.from_string(self.current_role)
            if role is not None:
                self.current_role = role

        # Convert error category string to enum
        if isinstance(self.error_category, str):
            try:
                self.error_category = ErrorCategory(self.error_category)
            except ValueError:
                self.error_category = ErrorCategory.UNKNOWN


@dataclass
class EscalationDecision:
    """Result of an escalation decision.

    Attributes:
        action: The action to take.
        target_role: The role to route to (for RETRY/ESCALATE).
        reason: Human-readable explanation.
        include_context: Whether to include error context.
        retries_remaining: How many retries are left.
    """

    action: EscalationAction
    target_role: Role | None = None
    reason: str = ""
    include_context: bool = True
    retries_remaining: int = 0

    @property
    def should_escalate(self) -> bool:
        """Check if decision is to escalate."""
        return self.action == EscalationAction.ESCALATE

    @property
    def should_retry(self) -> bool:
        """Check if decision is to retry."""
        return self.action == EscalationAction.RETRY

    @property
    def should_delegate(self) -> bool:
        """Check if decision is to delegate downward."""
        return self.action == EscalationAction.DELEGATE

    @property
    def is_terminal(self) -> bool:
        """Check if decision is terminal (FAIL or SKIP)."""
        return self.action in {EscalationAction.FAIL, EscalationAction.SKIP}


def _esc_cfg():
    from src.config import get_config

    return get_config().escalation


@dataclass
class EscalationConfig:
    """Configuration for escalation policy.

    Attributes:
        max_retries: Maximum retries before escalation.
        max_escalations: Maximum escalations per task.
        optional_gates: Gates that can be skipped on timeout.
        no_escalate_categories: Error categories that never trigger escalation.
    """

    max_retries: int = field(default_factory=lambda: _esc_cfg().max_retries)
    max_escalations: int = field(default_factory=lambda: _esc_cfg().max_escalations)
    optional_gates: frozenset[str] = field(default_factory=lambda: _esc_cfg().optional_gates)
    no_escalate_categories: frozenset[ErrorCategory] = field(
        default_factory=lambda: frozenset(
            {
                ErrorCategory.FORMAT,
                ErrorCategory.SCHEMA,
            }
        )
    )


class EscalationPolicy:
    """Unified escalation policy for the orchestration system.

    This class encapsulates all escalation logic. It replaces the scattered
    escalation implementations in failure_router.py, executor.py, and api.py.

    Policy Rules:
    1. Format/Schema errors: Retry only, never escalate
    2. Timeout on optional gate: Skip the gate
    3. Early abort: Immediate escalation (don't waste retries)
    4. Standard errors: Retry up to max_retries, then escalate
    5. At architect (top): No escalation possible, fail

    Usage:
        policy = EscalationPolicy()
        decision = policy.decide(context)
    """

    def __init__(self, config: EscalationConfig | None = None):
        """Initialize the escalation policy.

        Args:
            config: Policy configuration. Uses defaults if None.
        """
        self.config = config or EscalationConfig()

    def decide(self, context: EscalationContext) -> EscalationDecision:
        """Make an escalation decision based on context.

        Args:
            context: Information about the failure.

        Returns:
            EscalationDecision with action and target role.
        """
        # Get the role's escalation target
        if isinstance(context.current_role, Role):
            target = context.current_role.escalates_to()
        else:
            # Unknown role - default to fail
            return EscalationDecision(
                action=EscalationAction.FAIL,
                reason=f"Unknown role: {context.current_role}",
            )

        # Handle timeout on optional gates
        if context.error_category == ErrorCategory.TIMEOUT:
            if context.gate_name in self.config.optional_gates:
                return EscalationDecision(
                    action=EscalationAction.SKIP,
                    target_role=context.current_role,
                    reason=f"Skipping optional gate '{context.gate_name}' due to timeout",
                    include_context=False,
                )

        # Handle early abort - escalate immediately
        if context.error_category == ErrorCategory.EARLY_ABORT:
            if target is None:
                return EscalationDecision(
                    action=EscalationAction.FAIL,
                    reason=f"Early abort at {context.current_role} with no escalation available",
                )
            if context.escalation_count >= self.config.max_escalations:
                return EscalationDecision(
                    action=EscalationAction.FAIL,
                    reason=f"Early abort: max escalations ({self.config.max_escalations}) reached",
                )
            return EscalationDecision(
                action=EscalationAction.ESCALATE,
                target_role=target,
                reason=f"Early abort detected: {context.error_message[:100]}",
            )

        # Handle format/schema errors - retry only
        if context.error_category in self.config.no_escalate_categories:
            if context.failure_count < self.config.max_retries:
                return EscalationDecision(
                    action=EscalationAction.RETRY,
                    target_role=context.current_role,
                    reason=f"Retry {context.error_category.value} error (attempt {context.failure_count + 1}/{self.config.max_retries})",
                    retries_remaining=self.config.max_retries - context.failure_count - 1,
                )
            return EscalationDecision(
                action=EscalationAction.FAIL,
                reason=f"Max retries ({self.config.max_retries}) exceeded for {context.error_category.value} error",
            )

        # Standard retry/escalate logic
        if context.failure_count < self.config.max_retries:
            return EscalationDecision(
                action=EscalationAction.RETRY,
                target_role=context.current_role,
                reason=f"Retry with same role (attempt {context.failure_count + 1}/{self.config.max_retries})",
                retries_remaining=self.config.max_retries - context.failure_count - 1,
            )

        # Retries exhausted - try to escalate
        if target is None:
            # Terminal role (architect) — fall back to REPL exploration
            # instead of hard failure. The chat handler will switch to
            # exploration mode with chunk_and_summarize.
            return EscalationDecision(
                action=EscalationAction.EXPLORE,
                target_role=context.current_role,  # Stay at current role
                reason=f"Terminal role {context.current_role}: falling back to REPL exploration",
            )

        if context.escalation_count >= self.config.max_escalations:
            return EscalationDecision(
                action=EscalationAction.FAIL,
                reason=f"Max escalations ({self.config.max_escalations}) reached",
            )

        return EscalationDecision(
            action=EscalationAction.ESCALATE,
            target_role=target,
            reason=f"Escalating from {context.current_role} to {target} after {context.failure_count} failures",
        )

    def get_escalation_path(self, role: Role | str) -> list[Role]:
        """Get the full escalation path from a role.

        Args:
            role: Starting role.

        Returns:
            List of roles in escalation order.
        """
        return get_escalation_chain(role)


# Global default policy
_default_policy: EscalationPolicy | None = None


def get_policy() -> EscalationPolicy:
    """Get the global default escalation policy.

    Returns:
        EscalationPolicy instance.
    """
    global _default_policy
    if _default_policy is None:
        _default_policy = EscalationPolicy()
    return _default_policy


def set_policy(policy: EscalationPolicy) -> None:
    """Set the global default escalation policy.

    Args:
        policy: Policy to use globally.
    """
    global _default_policy
    _default_policy = policy


def decide(context: EscalationContext) -> EscalationDecision:
    """Make an escalation decision using the global policy.

    Convenience function for quick decisions without creating a policy.

    Args:
        context: Failure context.

    Returns:
        Escalation decision.
    """
    return get_policy().decide(context)

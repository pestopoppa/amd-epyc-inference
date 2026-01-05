#!/usr/bin/env python3
"""Failure routing for hierarchical orchestration.

This module routes failures to the appropriate escalation level based on
role, failure count, and error category. Implements the escalation chain:
worker → coder → architect.

Usage:
    from src.failure_router import FailureRouter, FailureContext

    router = FailureRouter()
    context = FailureContext(
        role="worker",
        failure_count=2,
        error_category="code",
        gate_name="unit",
        error_message="Test failed",
    )
    next_role = router.route_failure(context)
    # Returns "coder" (escalation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    """Categories of errors for routing decisions."""

    CODE = "code"  # Syntax errors, type errors, test failures
    LOGIC = "logic"  # Wrong output, failed assertions
    TIMEOUT = "timeout"  # Gate or execution timeout
    SCHEMA = "schema"  # IR/JSON schema violations
    FORMAT = "format"  # Style/format issues
    EARLY_ABORT = "early_abort"  # Generation aborted due to predicted failure
    UNKNOWN = "unknown"  # Unclassified errors


@dataclass
class EscalationChain:
    """Defines escalation path for a role.

    Attributes:
        role: The current role in the chain.
        escalates_to: The role to escalate to, or None if at top.
        max_retries: Maximum retries before escalation.
        max_escalations: Maximum times this role can escalate.
    """

    role: str
    escalates_to: str | None
    max_retries: int = 2
    max_escalations: int = 2


@dataclass
class FailureContext:
    """Context about a failure for routing decisions.

    Attributes:
        role: Current role that failed.
        failure_count: Number of times this role has failed on this task.
        error_category: Category of the error.
        gate_name: Name of the gate that failed (if applicable).
        error_message: The error message.
        task_id: Optional task ID for tracking.
        escalation_count: How many times we've escalated already.
    """

    role: str
    failure_count: int
    error_category: str | ErrorCategory
    gate_name: str = ""
    error_message: str = ""
    task_id: str = ""
    escalation_count: int = 0

    def __post_init__(self) -> None:
        """Convert error_category to ErrorCategory if string."""
        if isinstance(self.error_category, str):
            try:
                self.error_category = ErrorCategory(self.error_category)
            except ValueError:
                self.error_category = ErrorCategory.UNKNOWN


@dataclass
class RoutingDecision:
    """Result of a routing decision.

    Attributes:
        action: The action to take ("retry", "escalate", "fail", "skip").
        next_role: The role to route to (same role for retry).
        reason: Human-readable explanation of the decision.
        should_include_context: Whether to include full error context.
        max_retries_remaining: How many more retries are allowed.
    """

    action: str  # "retry", "escalate", "fail", "skip"
    next_role: str | None
    reason: str
    should_include_context: bool = True
    max_retries_remaining: int = 0


class FailureRouter:
    """Routes failures to appropriate handlers.

    Implements escalation chains:
    - worker → coder → architect → fail
    - ingest → architect → fail
    - coder → architect → fail

    First failure: retry same role
    Second failure: escalate one tier
    At architect with failure: terminal failure

    Special rules:
    - Format/schema errors: Always retry same role (never escalate)
    - Timeout errors: Skip the gate if optional, fail if required
    """

    # Standard escalation chains
    ESCALATION_CHAINS: dict[str, EscalationChain] = {
        "worker": EscalationChain("worker", "coder", max_retries=2, max_escalations=2),
        "coder": EscalationChain("coder", "architect", max_retries=2, max_escalations=1),
        "architect": EscalationChain("architect", None, max_retries=3, max_escalations=0),
        "ingest": EscalationChain("ingest", "architect", max_retries=1, max_escalations=1),
    }

    # Error categories that should not trigger escalation
    NO_ESCALATE_CATEGORIES: set[ErrorCategory] = {
        ErrorCategory.FORMAT,
        ErrorCategory.SCHEMA,
    }

    # Gates that are optional (can be skipped on timeout)
    OPTIONAL_GATES: set[str] = {
        "typecheck",
        "integration",
        "shellcheck",
    }

    def __init__(
        self,
        custom_chains: dict[str, EscalationChain] | None = None,
        optional_gates: set[str] | None = None,
    ):
        """Initialize the failure router.

        Args:
            custom_chains: Optional custom escalation chains to merge.
            optional_gates: Optional set of gate names that can be skipped.
        """
        self.chains = dict(self.ESCALATION_CHAINS)
        if custom_chains:
            self.chains.update(custom_chains)

        self.optional_gates = set(self.OPTIONAL_GATES)
        if optional_gates:
            self.optional_gates.update(optional_gates)

        # Track escalation history per task
        self._escalation_history: dict[str, list[str]] = {}

    def route_failure(self, context: FailureContext) -> RoutingDecision:
        """Determine how to handle a failure.

        Args:
            context: Information about the failure.

        Returns:
            RoutingDecision with action and next role.
        """
        chain = self.chains.get(context.role)
        if chain is None:
            return RoutingDecision(
                action="fail",
                next_role=None,
                reason=f"Unknown role: {context.role}",
                should_include_context=False,
            )

        # Handle timeout on optional gates
        if context.error_category == ErrorCategory.TIMEOUT:
            if context.gate_name in self.optional_gates:
                return RoutingDecision(
                    action="skip",
                    next_role=context.role,
                    reason=f"Skipping optional gate '{context.gate_name}' due to timeout",
                    should_include_context=False,
                )

        # Early abort: escalate immediately (don't retry - model showed failure signs)
        if context.error_category == ErrorCategory.EARLY_ABORT:
            if chain.escalates_to is None:
                # At top of chain (architect)
                return RoutingDecision(
                    action="fail",
                    next_role=None,
                    reason=f"Early abort at {context.role} with no escalation available",
                    should_include_context=True,
                )
            if context.escalation_count >= chain.max_escalations:
                return RoutingDecision(
                    action="fail",
                    next_role=None,
                    reason=f"Early abort: max escalations ({chain.max_escalations}) reached",
                    should_include_context=True,
                )
            # Immediate escalation - retrying same role wastes compute
            return RoutingDecision(
                action="escalate",
                next_role=chain.escalates_to,
                reason=f"Early abort detected at {context.role}: {context.error_message}",
                should_include_context=True,
            )

        # Format/schema errors: always retry, never escalate
        if context.error_category in self.NO_ESCALATE_CATEGORIES:
            if context.failure_count < chain.max_retries:
                return RoutingDecision(
                    action="retry",
                    next_role=context.role,
                    reason=f"Retry {context.error_category.value} error (attempt {context.failure_count + 1}/{chain.max_retries})",
                    max_retries_remaining=chain.max_retries - context.failure_count - 1,
                )
            else:
                return RoutingDecision(
                    action="fail",
                    next_role=None,
                    reason=f"Max retries ({chain.max_retries}) exceeded for {context.error_category.value} error",
                    should_include_context=True,
                )

        # Standard retry/escalate logic
        if context.failure_count < chain.max_retries:
            # Still have retries left
            return RoutingDecision(
                action="retry",
                next_role=context.role,
                reason=f"Retry with same role (attempt {context.failure_count + 1}/{chain.max_retries})",
                should_include_context=True,
                max_retries_remaining=chain.max_retries - context.failure_count - 1,
            )

        # Retries exhausted, check if we can escalate
        if chain.escalates_to is None:
            # At top of chain (architect)
            return RoutingDecision(
                action="fail",
                next_role=None,
                reason=f"No escalation possible from {context.role}",
                should_include_context=True,
            )

        if context.escalation_count >= chain.max_escalations:
            # Already escalated max times
            return RoutingDecision(
                action="fail",
                next_role=None,
                reason=f"Max escalations ({chain.max_escalations}) reached from {context.role}",
                should_include_context=True,
            )

        # Escalate to next tier
        return RoutingDecision(
            action="escalate",
            next_role=chain.escalates_to,
            reason=f"Escalating from {context.role} to {chain.escalates_to} after {context.failure_count} failures",
            should_include_context=True,
        )

    def get_escalation_path(self, role: str) -> list[str]:
        """Get the full escalation path from a role.

        Args:
            role: Starting role.

        Returns:
            List of roles in escalation order (including starting role).
        """
        path = [role]
        current = role

        while current in self.chains:
            chain = self.chains[current]
            if chain.escalates_to is None:
                break
            path.append(chain.escalates_to)
            current = chain.escalates_to

        return path

    def record_escalation(self, task_id: str, from_role: str, to_role: str) -> None:
        """Record an escalation for tracking.

        Args:
            task_id: The task ID.
            from_role: Role escalating from.
            to_role: Role escalating to.
        """
        if task_id not in self._escalation_history:
            self._escalation_history[task_id] = []
        self._escalation_history[task_id].append(f"{from_role}->{to_role}")

    def get_escalation_history(self, task_id: str) -> list[str]:
        """Get escalation history for a task.

        Args:
            task_id: The task ID.

        Returns:
            List of escalation strings (e.g., ["worker->coder", "coder->architect"]).
        """
        return self._escalation_history.get(task_id, [])

    def clear_history(self, task_id: str | None = None) -> None:
        """Clear escalation history.

        Args:
            task_id: Specific task to clear, or None to clear all.
        """
        if task_id is None:
            self._escalation_history.clear()
        elif task_id in self._escalation_history:
            del self._escalation_history[task_id]

    def should_include_error_context(self, context: FailureContext) -> bool:
        """Determine if full error context should be included.

        Args:
            context: The failure context.

        Returns:
            True if error details should be passed to next handler.
        """
        # Always include context for code/logic errors
        if context.error_category in {ErrorCategory.CODE, ErrorCategory.LOGIC}:
            return True

        # Include context on first retry
        if context.failure_count <= 1:
            return True

        # Don't include verbose context for format/schema on repeated failures
        if context.error_category in self.NO_ESCALATE_CATEGORIES:
            return context.failure_count <= 2

        return True

    def get_chain(self, role: str) -> EscalationChain | None:
        """Get the escalation chain for a role.

        Args:
            role: The role name.

        Returns:
            EscalationChain or None if role not found.
        """
        return self.chains.get(role)

    def add_chain(self, chain: EscalationChain) -> None:
        """Add or update an escalation chain.

        Args:
            chain: The escalation chain to add.
        """
        self.chains[chain.role] = chain

    def format_failure_report(self, context: FailureContext, decision: RoutingDecision) -> str:
        """Format a human-readable failure report.

        Args:
            context: The failure context.
            decision: The routing decision.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 50,
            "FAILURE REPORT",
            "=" * 50,
            f"Role: {context.role}",
            f"Failure Count: {context.failure_count}",
            f"Error Category: {context.error_category.value if isinstance(context.error_category, ErrorCategory) else context.error_category}",
        ]

        if context.gate_name:
            lines.append(f"Gate: {context.gate_name}")

        if context.task_id:
            lines.append(f"Task ID: {context.task_id}")

        if context.error_message:
            lines.append(f"\nError Message:\n{context.error_message[:500]}")

        lines.extend([
            "",
            "-" * 50,
            "DECISION",
            "-" * 50,
            f"Action: {decision.action.upper()}",
            f"Next Role: {decision.next_role or 'N/A'}",
            f"Reason: {decision.reason}",
        ])

        if decision.max_retries_remaining > 0:
            lines.append(f"Retries Remaining: {decision.max_retries_remaining}")

        lines.append("=" * 50)

        return "\n".join(lines)

#!/usr/bin/env python3
"""Failure routing for hierarchical orchestration.

This module routes failures to the appropriate escalation level based on
role, failure count, and error category. Implements the escalation chain:
worker → coder → architect.

Phase 4 (MemRL): LearnedEscalationPolicy queries episodic memory for similar
failures to inform escalation decisions. Falls back to rule-based routing when
not confident.

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

    # With learned escalation (requires MemRL components):
    from orchestration.repl_memory import TwoPhaseRetriever, ProgressLogger
    router = FailureRouter(retriever=retriever, progress_logger=logger)
    # Router will query episodic memory before making decisions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestration.repl_memory.progress_logger import ProgressLogger
    from orchestration.repl_memory.retriever import TwoPhaseRetriever

logger = logging.getLogger(__name__)


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
        task_type: Type of task (for future routing enhancements).
        code_generated: Generated code snippet (for debugging).
    """

    role: str
    failure_count: int
    error_category: str | ErrorCategory
    gate_name: str = ""
    error_message: str = ""
    task_id: str = ""
    escalation_count: int = 0
    task_type: str = ""
    code_generated: str | None = None

    def __post_init__(self) -> None:
        """Normalize role and error_category."""
        # Normalize role to string value
        self.role = _normalize_role(self.role)

        # Convert error_category to ErrorCategory if string
        if isinstance(self.error_category, str):
            try:
                self.error_category = ErrorCategory(self.error_category)
            except ValueError:
                self.error_category = ErrorCategory.UNKNOWN


def _normalize_role(role: str) -> str:
    """Normalize role to string value.

    Handles:
    - Role enum objects (extract .value)
    - Role enum repr strings like "Role.CODER_PRIMARY" -> "coder_primary"
    - Normal strings (pass through)
    """
    # Handle Role enum objects
    if hasattr(role, "value"):
        return role.value
    # Handle enum repr strings like "Role.CODER_PRIMARY"
    if isinstance(role, str) and role.startswith("Role."):
        # Extract the value part after "Role."
        enum_name = role[5:]  # Remove "Role." prefix
        # Convert CODER_PRIMARY to coder_primary
        return enum_name.lower()
    return role


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


@dataclass
class LearnedEscalationResult:
    """Result of querying episodic memory for escalation guidance.

    Attributes:
        should_use_learned: Whether to use learned action instead of rules.
        suggested_action: Suggested action ("retry", "escalate", "fail").
        suggested_role: Role to route to (for escalate action).
        confidence: Confidence score (0-1).
        similar_cases: Number of similar cases found.
        memory_id: ID of the most relevant memory (for logging).
    """

    should_use_learned: bool = False
    suggested_action: str = ""
    suggested_role: str | None = None
    confidence: float = 0.0
    similar_cases: int = 0
    memory_id: str | None = None


class LearnedEscalationPolicy:
    """Queries episodic memory to inform escalation decisions.

    This is the key Phase 4 component that enables learned escalation.
    It uses two-phase retrieval to find similar past failures and their
    outcomes, then suggests actions based on what worked before.

    Cold start: Returns should_use_learned=False until enough data is
    collected. The FailureRouter falls back to rule-based escalation.
    """

    def __init__(
        self,
        retriever: "TwoPhaseRetriever",
        min_samples: int = 3,
        confidence_threshold: float = 0.6,
    ):
        """Initialize the learned escalation policy.

        Args:
            retriever: TwoPhaseRetriever for episodic memory queries.
            min_samples: Minimum samples needed to trust learned actions.
            confidence_threshold: Minimum confidence to use learned action.
        """
        self.retriever = retriever
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold

    def query(self, context: FailureContext) -> LearnedEscalationResult:
        """Query episodic memory for similar failures.

        Args:
            context: The failure context to find matches for.

        Returns:
            LearnedEscalationResult with suggestion (or not confident).
        """
        # Build failure context dictionary for embedding
        failure_dict = {
            "role": context.role,
            "error_category": (
                context.error_category.value
                if isinstance(context.error_category, ErrorCategory)
                else context.error_category
            ),
            "gate_name": context.gate_name,
            "error_message": context.error_message[:500],  # Truncate
            "failure_count": context.failure_count,
        }

        try:
            # Query episodic memory
            results = self.retriever.retrieve_for_escalation(failure_dict)

            if not results:
                return LearnedEscalationResult(similar_cases=0)

            # Check if we should use learned routing
            if not self.retriever.should_use_learned(results, self.min_samples):
                return LearnedEscalationResult(
                    similar_cases=len(results),
                    confidence=results[0].combined_score if results else 0.0,
                )

            # Parse the best action
            best = results[0]
            action_parts = best.memory.action.split(":")
            suggested_action = action_parts[0] if action_parts else "retry"
            suggested_role = action_parts[1] if len(action_parts) > 1 else None

            return LearnedEscalationResult(
                should_use_learned=True,
                suggested_action=suggested_action,
                suggested_role=suggested_role,
                confidence=best.combined_score,
                similar_cases=len(results),
                memory_id=best.memory.id,
            )

        except Exception as e:
            logger.warning(f"Learned escalation query failed: {e}")
            return LearnedEscalationResult()


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

    Phase 4 (MemRL): When retriever is provided, queries episodic memory
    for similar failures before making rule-based decisions. Falls back
    to rules when not confident or during cold start.
    """

    # Standard escalation chains (use generic names)
    ESCALATION_CHAINS: dict[str, EscalationChain] = {
        "worker": EscalationChain("worker", "coder", max_retries=2, max_escalations=2),
        "coder": EscalationChain("coder", "architect", max_retries=2, max_escalations=1),
        "architect": EscalationChain("architect", None, max_retries=3, max_escalations=0),
        "ingest": EscalationChain("ingest", "architect", max_retries=1, max_escalations=1),
        "frontdoor": EscalationChain("frontdoor", "coder", max_retries=2, max_escalations=2),
    }

    # Map specific role names to generic chain names
    ROLE_TO_CHAIN: dict[str, str] = {
        # Workers -> worker chain
        "worker_general": "worker",
        "worker_math": "worker",
        "worker_summarize": "worker",
        # Coders -> coder chain
        "coder_primary": "coder",
        "coder_escalation": "coder",
        # Architects -> architect chain
        "architect_general": "architect",
        "architect_coding": "architect",
        # Ingest -> ingest chain
        "ingest_long_context": "ingest",
        # Frontdoor uses its own chain
        "frontdoor": "frontdoor",
        # Thinking -> coder chain (escalates to architect)
        "thinking_reasoning": "coder",
        # Toolrunner -> worker chain
        "toolrunner": "worker",
    }

    # Map generic chain names to specific role names for escalation targets
    CHAIN_TO_ROLE: dict[str, str] = {
        "worker": "worker_general",
        "coder": "coder_primary",
        "architect": "architect_general",
        "ingest": "ingest_long_context",
        "frontdoor": "frontdoor",
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
        retriever: "TwoPhaseRetriever | None" = None,
        progress_logger: "ProgressLogger | None" = None,
    ):
        """Initialize the failure router.

        Args:
            custom_chains: Optional custom escalation chains to merge.
            optional_gates: Optional set of gate names that can be skipped.
            retriever: Optional TwoPhaseRetriever for learned escalation.
            progress_logger: Optional ProgressLogger for logging decisions.
        """
        self.chains = dict(self.ESCALATION_CHAINS)
        if custom_chains:
            self.chains.update(custom_chains)

        self.optional_gates = set(self.OPTIONAL_GATES)
        if optional_gates:
            self.optional_gates.update(optional_gates)

        # Track escalation history per task
        self._escalation_history: dict[str, list[str]] = {}

        # Phase 4: Learned escalation support
        self.learned_policy: LearnedEscalationPolicy | None = None
        if retriever is not None:
            self.learned_policy = LearnedEscalationPolicy(retriever)

        self.progress_logger = progress_logger

        # Track strategy usage for monitoring
        self._strategy_counts = {"learned": 0, "rules": 0}

    def route_failure(self, context: FailureContext) -> RoutingDecision:
        """Determine how to handle a failure.

        Uses hybrid strategy:
        1. Query learned policy (if available and confident)
        2. Fall back to rule-based escalation

        Logs escalation decisions via progress_logger when provided.

        Args:
            context: Information about the failure.

        Returns:
            RoutingDecision with action and next role (using specific role names).
        """
        # Ensure role is normalized (belt and suspenders - also done in __post_init__)
        original_role = context.role
        context.role = _normalize_role(context.role)
        if original_role != context.role:
            logger.info(f"Normalized role: {original_role!r} -> {context.role!r}")

        # Map specific role name to generic chain name
        chain_name = self.ROLE_TO_CHAIN.get(context.role, context.role)
        chain = self.chains.get(chain_name)
        if chain is None:
            return RoutingDecision(
                action="fail",
                next_role=None,
                reason=f"Unknown role: {context.role} (chain: {chain_name})",
                should_include_context=False,
            )

        # Phase 4: Try learned escalation first
        learned_result: LearnedEscalationResult | None = None
        if self.learned_policy is not None:
            learned_result = self.learned_policy.query(context)
            if learned_result.should_use_learned:
                decision = self._apply_learned_decision(context, chain, learned_result)
                if decision is not None:
                    self._strategy_counts["learned"] += 1
                    self._log_decision(context, decision, "learned", learned_result)
                    return decision

        # Rule-based escalation (default path)
        self._strategy_counts["rules"] += 1
        decision = self._rule_based_route(context, chain)

        # Translate generic chain names to specific role names
        if decision.next_role is not None:
            decision = RoutingDecision(
                action=decision.action,
                next_role=self._chain_to_specific_role(decision.next_role, context.role),
                reason=decision.reason,
                should_include_context=decision.should_include_context,
                max_retries_remaining=decision.max_retries_remaining,
            )

        self._log_decision(context, decision, "rules", learned_result)
        return decision

    def _chain_to_specific_role(self, chain_name: str, original_role: str) -> str:
        """Map a generic chain name to a specific role name.

        If the chain_name matches the original role's chain, return the original.
        Otherwise, map to the default specific role for that chain.

        Args:
            chain_name: Generic chain name (e.g., "coder", "architect").
            original_role: The original specific role that failed.

        Returns:
            Specific role name.
        """
        # If returning to same chain, use original role
        original_chain = self.ROLE_TO_CHAIN.get(original_role, original_role)
        if chain_name == original_chain:
            return original_role

        # Map to default specific role for this chain
        return self.CHAIN_TO_ROLE.get(chain_name, chain_name)

    def _apply_learned_decision(
        self,
        context: FailureContext,
        chain: EscalationChain,
        learned: LearnedEscalationResult,
    ) -> RoutingDecision | None:
        """Apply a learned escalation decision.

        Validates the learned action against current state and chain limits.

        Args:
            context: Failure context.
            chain: Escalation chain for current role.
            learned: Result from learned policy.

        Returns:
            RoutingDecision or None if learned action is invalid.
        """
        action = learned.suggested_action.lower()

        if action == "retry":
            if context.failure_count < chain.max_retries:
                return RoutingDecision(
                    action="retry",
                    next_role=context.role,
                    reason=f"Learned: retry (confidence {learned.confidence:.2f}, {learned.similar_cases} similar cases)",
                    max_retries_remaining=chain.max_retries - context.failure_count - 1,
                )
            # Can't retry - fall through to rules

        elif action == "escalate":
            target = learned.suggested_role or chain.escalates_to
            if target is not None and context.escalation_count < chain.max_escalations:
                return RoutingDecision(
                    action="escalate",
                    next_role=target,
                    reason=f"Learned: escalate to {target} (confidence {learned.confidence:.2f}, {learned.similar_cases} similar cases)",
                    should_include_context=True,
                )
            # Can't escalate - fall through to rules

        elif action == "fail":
            return RoutingDecision(
                action="fail",
                next_role=None,
                reason=f"Learned: fail (confidence {learned.confidence:.2f}, {learned.similar_cases} similar cases)",
                should_include_context=True,
            )

        # Learned action not applicable, return None to use rules
        return None

    def _log_decision(
        self,
        context: FailureContext,
        decision: RoutingDecision,
        strategy: str,
        learned_result: LearnedEscalationResult | None,
    ) -> None:
        """Log escalation decision via progress_logger.

        Args:
            context: Failure context.
            decision: The routing decision made.
            strategy: "learned" or "rules".
            learned_result: Result from learned policy (may be None).
        """
        if self.progress_logger is None:
            return

        if decision.action == "escalate" and decision.next_role:
            self.progress_logger.log_escalation(
                task_id=context.task_id or "unknown",
                from_tier=context.role,
                to_tier=decision.next_role,
                reason=decision.reason,
                memory_id=learned_result.memory_id if learned_result else None,
            )

    def _rule_based_route(
        self,
        context: FailureContext,
        chain: EscalationChain,
    ) -> RoutingDecision:
        """Rule-based escalation logic (original implementation).

        Args:
            context: Failure context.
            chain: Escalation chain for current role.

        Returns:
            RoutingDecision based on rules.
        """
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

    def get_strategy_counts(self) -> dict[str, int]:
        """Get counts of learned vs rule-based decisions.

        Returns:
            Dictionary with "learned" and "rules" counts.
        """
        return dict(self._strategy_counts)

    def reset_strategy_counts(self) -> None:
        """Reset strategy counts to zero."""
        self._strategy_counts = {"learned": 0, "rules": 0}

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

        lines.extend(
            [
                "",
                "-" * 50,
                "DECISION",
                "-" * 50,
                f"Action: {decision.action.upper()}",
                f"Next Role: {decision.next_role or 'N/A'}",
                f"Reason: {decision.reason}",
            ]
        )

        if decision.max_retries_remaining > 0:
            lines.append(f"Retries Remaining: {decision.max_retries_remaining}")

        lines.append("=" * 50)

        return "\n".join(lines)

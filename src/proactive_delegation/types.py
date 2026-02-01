"""Data types for proactive delegation workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ── Custom Exceptions ─────────────────────────────────────────────────


class DelegationError(Exception):
    """Base exception for proactive delegation failures."""


class ArchitectPlanError(DelegationError):
    """Architect failed to generate a valid plan."""


class StepExecutionError(DelegationError):
    """A delegation step failed execution."""

    def __init__(self, step_id: str, role: str, cause: Exception | None = None):
        self.step_id = step_id
        self.role = role
        self.cause = cause
        msg = f"Step {step_id} ({role}) failed"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)


# ── Enums ─────────────────────────────────────────────────────────────


class ReviewDecision(Enum):
    """Architect's review decision."""

    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    ESCALATE = "escalate"
    REJECT = "reject"


class TaskComplexity(Enum):
    """Task complexity level determining delegation path."""

    TRIVIAL = "trivial"  # Frontdoor answers directly (factual, chat)
    SIMPLE = "simple"    # Frontdoor executes in REPL (single code task)
    MODERATE = "moderate"  # Frontdoor delegates to single specialist
    COMPLEX = "complex"  # Architect generates TaskIR, multi-specialist


@dataclass
class ComplexitySignals:
    """Signals used to estimate task complexity and routing."""

    word_count: int = 0
    has_code_keywords: bool = False
    has_multi_step_keywords: bool = False
    has_architecture_keywords: bool = False
    question_type: str = "unknown"  # factual, how-to, implementation, design, architect_requested, thinking_requested
    estimated_files: int = 0
    # Escalation flags (orthogonal to complexity)
    thinking_requested: bool = False  # /think, ultrathink -> use thinking_reasoning model
    architect_requested: bool = False  # /architect, /plan -> use architect for planning


def _deleg_cfg():
    from src.config import get_config
    return get_config().delegation


@dataclass
class IterationContext:
    """Track iteration state to prevent infinite loops.

    Attributes:
        max_iterations: Maximum review-fix cycles per subtask
        max_total_iterations: Maximum total iterations across all subtasks
        current_iteration: Current iteration count for active subtask
        total_iterations: Total iterations across all subtasks
        iteration_history: Log of iteration decisions
    """

    max_iterations: int = field(default_factory=lambda: _deleg_cfg().max_iterations)
    max_total_iterations: int = field(default_factory=lambda: _deleg_cfg().max_total_iterations)
    current_iteration: int = 0
    total_iterations: int = 0
    subtask_iterations: dict[str, int] = field(default_factory=dict)
    iteration_history: list[dict[str, Any]] = field(default_factory=list)

    def can_iterate(self, subtask_id: str) -> bool:
        """Check if another iteration is allowed for this subtask."""
        subtask_count = self.subtask_iterations.get(subtask_id, 0)
        return (
            subtask_count < self.max_iterations
            and self.total_iterations < self.max_total_iterations
        )

    def record_iteration(
        self,
        subtask_id: str,
        decision: ReviewDecision,
        feedback: str | None = None,
    ) -> None:
        """Record an iteration for tracking."""
        self.subtask_iterations[subtask_id] = (
            self.subtask_iterations.get(subtask_id, 0) + 1
        )
        self.total_iterations += 1
        self.current_iteration = self.subtask_iterations[subtask_id]

        self.iteration_history.append(
            {
                "subtask_id": subtask_id,
                "iteration": self.current_iteration,
                "total": self.total_iterations,
                "decision": decision.value,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_summary(self) -> dict[str, Any]:
        """Get iteration summary for logging."""
        return {
            "total_iterations": self.total_iterations,
            "subtask_counts": dict(self.subtask_iterations),
            "max_reached": self.total_iterations >= self.max_total_iterations,
        }


@dataclass
class ArchitectReview:
    """Result of architect reviewing specialist output."""

    subtask_id: str
    decision: ReviewDecision
    feedback: str = ""
    score: float = 0.0
    suggested_changes: list[str] = field(default_factory=list)
    approved_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "subtask_id": self.subtask_id,
            "decision": self.decision.value,
            "feedback": self.feedback,
            "score": self.score,
            "suggested_changes": self.suggested_changes,
            "approved_output": self.approved_output,
        }


@dataclass
class SubtaskResult:
    """Result from a specialist executing a subtask."""

    subtask_id: str
    role: str
    output: str
    success: bool
    error: str | None = None
    tokens_used: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class AggregatedResult:
    """Final aggregated result from multiple specialists."""

    task_id: str
    objective: str
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    aggregated_output: str = ""
    all_approved: bool = False
    total_iterations: int = 0
    roles_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "subtask_results": [
                {
                    "subtask_id": r.subtask_id,
                    "role": r.role,
                    "output": r.output[:500] + "..." if len(r.output) > 500 else r.output,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.subtask_results
            ],
            "aggregated_output": self.aggregated_output,
            "all_approved": self.all_approved,
            "total_iterations": self.total_iterations,
            "roles_used": self.roles_used,
        }


@dataclass
class PlanReviewResult:
    """Result of architect reviewing a plan before execution."""

    decision: str = "ok"
    score: float = 1.0
    feedback: str = ""
    patches: list[dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""

    @property
    def is_ok(self) -> bool:
        """True if architect approved the plan without changes."""
        return self.decision == "ok"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision": self.decision,
            "score": self.score,
            "feedback": self.feedback,
            "patches": self.patches,
        }

"""Proactive delegation workflow for hierarchical orchestration.

Phase 5 implementation: Transform from reactive escalation to proactive orchestration.

COMPLEXITY-AWARE ROUTING:
    TRIVIAL  -> Frontdoor answers directly (no delegation)
    SIMPLE   -> Frontdoor executes in REPL (no architect)
    MODERATE -> Frontdoor delegates to single specialist (no architect)
    COMPLEX  -> Architect generates TaskIR, multi-specialist workflow

Usage:
    from src.proactive_delegation import ProactiveDelegator, TaskComplexity
    delegator = ProactiveDelegator(registry=..., primitives=...)
    complexity, action, signals, confidence = delegator.route_by_complexity(objective)
"""

# Types
from src.proactive_delegation.types import (
    AggregatedResult,
    ArchitectPlanError,
    ArchitectReview,
    ComplexitySignals,
    DelegationError,
    IterationContext,
    PlanReviewResult,
    ReviewDecision,
    StepExecutionError,
    SubtaskResult,
    TaskComplexity,
)

# Complexity classification and constants
from src.proactive_delegation.complexity import (
    ARCHITECT_TRIGGERS,
    ROLE_MAPPING,
    THINKING_TRIGGERS,
    classify_task_complexity,
    has_architect_trigger,
    has_thinking_trigger,
)

# Services
from src.proactive_delegation.review_service import (
    AggregationService,
    ArchitectReviewService,
)

# Delegator
from src.proactive_delegation.delegator import ProactiveDelegator

__all__ = [
    # Exceptions
    "DelegationError",
    "ArchitectPlanError",
    "StepExecutionError",
    # Types
    "ReviewDecision",
    "TaskComplexity",
    "ComplexitySignals",
    "IterationContext",
    "ArchitectReview",
    "SubtaskResult",
    "AggregatedResult",
    "PlanReviewResult",
    # Complexity
    "ARCHITECT_TRIGGERS",
    "THINKING_TRIGGERS",
    "ROLE_MAPPING",
    "classify_task_complexity",
    "has_architect_trigger",
    "has_thinking_trigger",
    # Services
    "AggregationService",
    "ArchitectReviewService",
    # Delegator
    "ProactiveDelegator",
]

"""Task complexity classification and role mapping."""

from __future__ import annotations

import logging

from src.proactive_delegation.types import (
    ComplexitySignals,
    TaskComplexity,
)

logger = logging.getLogger(__name__)


# Architect escalation triggers - force architect model for TASK PLANNING
# Use when you need the task broken down into subtasks
ARCHITECT_TRIGGERS = [
    "/architect",  # Explicit command
    "/plan",  # Request planning
    "[PLAN]",  # Tag-style
    "[ARCHITECT]",  # Tag-style
    "architect this",  # Natural language
    "plan this carefully",
    "break this down",
    "decompose this task",
    "create a spec",
]

# Thinking escalation triggers - force thinking model for DEEP REASONING
# Equivalent to Claude Code's "ultrathink" - routes to thinking_reasoning role
# Use when you need more thorough analysis, not task decomposition
THINKING_TRIGGERS = [
    "/think",  # Explicit command
    "ultrathink",  # Claude Code compatibility
    "[THINK]",  # Tag-style
    "think deeply",  # Natural language
    "think carefully",
    "reason through",
    "analyze carefully",
    "step by step",
    "show your reasoning",
]


# Role mapping for TaskIR agents to registry roles
ROLE_MAPPING = {
    "frontdoor": "frontdoor",
    "coder": "coder_escalation",
    "ingest": "ingest_long_context",
    "architect": "architect_general",
    "worker": "worker_general",
    "docwriter": "worker_general",
    "math": "worker_math",
    "vision": "worker_vision",
    "toolrunner": "toolrunner",
    "formalizer": "formalizer",
}


def has_architect_trigger(objective: str) -> bool:
    """Check if objective requests architect for task planning/decomposition."""
    objective_lower = objective.lower()
    return any(trigger.lower() in objective_lower for trigger in ARCHITECT_TRIGGERS)


def has_thinking_trigger(objective: str) -> bool:
    """Check if objective requests thinking model for deep reasoning.

    Equivalent to Claude Code's "ultrathink" - escalates to thinking_reasoning
    role which uses a model optimized for chain-of-thought reasoning.
    """
    objective_lower = objective.lower()
    return any(trigger.lower() in objective_lower for trigger in THINKING_TRIGGERS)


def classify_task_complexity(objective: str) -> tuple[TaskComplexity, ComplexitySignals]:
    """Classify task complexity to determine delegation path.

    This is a fast heuristic classifier. For borderline cases, the frontdoor
    model can make the final decision via REPL exploration.

    ESCALATION TRIGGERS:
        /architect, /plan, [PLAN], "think carefully", "break this down"
        -> Forces COMPLEX classification regardless of heuristics

    Returns:
        (complexity_level, signals) - complexity and the signals used

    Decision tree:
        TRIVIAL: factual questions, greetings, simple math
        SIMPLE: single-file code tasks, fix typo, add function
        MODERATE: multi-function changes, refactoring single module
        COMPLEX: multi-file changes, architecture, system design
    """
    objective_lower = objective.lower()
    words = objective.split()
    signals = ComplexitySignals(word_count=len(words))

    # Check escalation triggers (can be combined)
    signals.architect_requested = has_architect_trigger(objective)
    signals.thinking_requested = has_thinking_trigger(objective)

    # Architect trigger forces COMPLEX (task decomposition)
    if signals.architect_requested:
        signals.question_type = "architect_requested"
        return TaskComplexity.COMPLEX, signals

    # Thinking trigger alone doesn't change complexity, just routing
    if signals.thinking_requested:
        signals.question_type = "thinking_requested"
        # Don't return early - continue to assess complexity normally
        # The thinking_requested flag tells the router to use thinking model

    # Trivial indicators (answer directly)
    trivial_patterns = [
        "what is",
        "who is",
        "when did",
        "how many",
        "define ",
        "hello",
        "hi ",
        "thanks",
        "thank you",
        "help",
        "what's the weather",
        "tell me about",
    ]
    if any(p in objective_lower for p in trivial_patterns) and len(words) < 15:
        signals.question_type = "factual"
        return TaskComplexity.TRIVIAL, signals

    # Code keywords
    code_keywords = [
        "implement",
        "code",
        "function",
        "class",
        "method",
        "write",
        "create",
        "build",
        "fix",
        "debug",
        "refactor",
        "add",
        "remove",
        "update",
        "modify",
        "change",
    ]
    signals.has_code_keywords = any(k in objective_lower for k in code_keywords)

    # Multi-step keywords
    multi_step_keywords = [
        "and then",
        "after that",
        "followed by",
        "steps",
        "first",
        "second",
        "third",
        "finally",
        "multiple",
        "several",
        "all the",
        "each",
    ]
    signals.has_multi_step_keywords = any(k in objective_lower for k in multi_step_keywords)

    # Architecture keywords (likely needs architect)
    architecture_keywords = [
        "architecture",
        "design",
        "system",
        "distributed",
        "scalable",
        "fault-tolerant",
        "consensus",
        "protocol",
        "api design",
        "database schema",
        "microservice",
        "trade-off",
        "compare",
        "evaluate",
        "pros and cons",
    ]
    signals.has_architecture_keywords = any(k in objective_lower for k in architecture_keywords)

    # File count estimation
    file_patterns = [
        ("files", 3),
        ("modules", 3),
        ("components", 3),
        ("services", 4),
        ("endpoints", 2),
        ("tests", 2),
    ]
    for pattern, count in file_patterns:
        if pattern in objective_lower:
            signals.estimated_files = max(signals.estimated_files, count)

    # Decision logic
    if signals.has_architecture_keywords or signals.estimated_files >= 3:
        signals.question_type = "design"
        return TaskComplexity.COMPLEX, signals

    if signals.has_multi_step_keywords and signals.has_code_keywords:
        signals.question_type = "implementation"
        return TaskComplexity.MODERATE, signals

    if signals.has_code_keywords:
        if len(words) > 30 or signals.estimated_files >= 2:
            signals.question_type = "implementation"
            return TaskComplexity.MODERATE, signals
        signals.question_type = "implementation"
        return TaskComplexity.SIMPLE, signals

    # Default based on length
    if len(words) > 50:
        signals.question_type = "implementation"
        return TaskComplexity.MODERATE, signals

    signals.question_type = "how-to"
    return TaskComplexity.SIMPLE, signals

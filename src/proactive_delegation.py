"""Proactive delegation workflow for hierarchical orchestration.

Phase 5 implementation: Transform from reactive escalation to proactive orchestration.

COMPLEXITY-AWARE ROUTING:
    TRIVIAL  → Frontdoor answers directly (no delegation)
    SIMPLE   → Frontdoor executes in REPL (no architect)
    MODERATE → Frontdoor delegates to single specialist (no architect)
    COMPLEX  → Architect generates TaskIR, multi-specialist workflow

ESCALATION TRIGGERS:
    /architect, /plan  → Force architect for TASK PLANNING (TaskIR breakdown)
    /think, ultrathink → Force thinking model for DEEP REASONING (like Claude Code)

    Architect escalation: changes complexity to COMPLEX, routes to architect_general
    Thinking escalation: keeps complexity, routes to thinking_reasoning model

TOKEN EFFICIENCY:
- Architect prompts use abbreviated JSON keys (d, s, f, c)
- Review: ~30-50 tokens output, 128 max
- TaskIR generation: ~100-150 tokens output, 256 max
- Only COMPLEX tasks invoke expensive architect model

MEMRL INTEGRATION:
- HybridRouter provides learned routing suggestions from episodic memory
- Delegation decisions logged for Q-scoring and learning
- Confidence scores from MemRL can upgrade/downgrade complexity

Example:
    delegator = ProactiveDelegator(...)
    complexity, action, signals, confidence = delegator.route_by_complexity(objective)
    role = delegator.get_target_role(action, signals)
    # role = "thinking_reasoning" if signals.thinking_requested
    # role = "architect_general" if action == "architect"
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.registry_loader import RegistryLoader
    from src.llm_primitives import LLMPrimitives
    from orchestration.repl_memory.progress_logger import ProgressLogger

logger = logging.getLogger(__name__)


class ReviewDecision(Enum):
    """Architect's review decision."""

    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    ESCALATE = "escalate"
    REJECT = "reject"


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
    """Result of architect reviewing specialist output.

    Attributes:
        subtask_id: ID of the subtask being reviewed
        decision: Approve, request changes, escalate, or reject
        feedback: Specific feedback for the specialist
        score: Quality score 0.0-1.0 (for MemRL learning)
        suggested_changes: Structured list of changes needed
        approved_output: Final output if approved
    """

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
    """Result from a specialist executing a subtask.

    Attributes:
        subtask_id: ID of the subtask
        role: Role that executed (e.g., "coder", "worker_math")
        output: Raw output from specialist
        success: Whether execution succeeded
        error: Error message if failed
        tokens_used: Tokens consumed
        elapsed_seconds: Time taken
    """

    subtask_id: str
    role: str
    output: str
    success: bool
    error: str | None = None
    tokens_used: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class AggregatedResult:
    """Final aggregated result from multiple specialists.

    Attributes:
        task_id: Original task ID
        objective: Task objective
        subtask_results: Results from each specialist
        aggregated_output: Combined final output
        all_approved: Whether all subtasks passed review
        total_iterations: Total review iterations
        roles_used: List of roles involved
    """

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


class AggregationService:
    """Service for combining outputs from multiple specialists.

    Strategies:
    - concatenate: Simple concatenation with headers
    - merge_code: Merge code outputs intelligently
    - structured: Combine into structured JSON
    """

    def aggregate(
        self,
        results: list[SubtaskResult],
        strategy: str = "concatenate",
    ) -> str:
        """Aggregate multiple specialist outputs into final result.

        Args:
            results: List of specialist results
            strategy: Aggregation strategy

        Returns:
            Combined output string
        """
        if not results:
            return ""

        if strategy == "concatenate":
            return self._aggregate_concatenate(results)
        elif strategy == "merge_code":
            return self._aggregate_merge_code(results)
        elif strategy == "structured":
            return self._aggregate_structured(results)
        else:
            logger.warning(f"Unknown aggregation strategy '{strategy}', using concatenate")
            return self._aggregate_concatenate(results)

    def _aggregate_concatenate(self, results: list[SubtaskResult]) -> str:
        """Simple concatenation with section headers."""
        sections = []
        for result in results:
            if result.success and result.output:
                header = f"## {result.subtask_id} ({result.role})"
                sections.append(f"{header}\n\n{result.output}")
        return "\n\n---\n\n".join(sections)

    def _aggregate_merge_code(self, results: list[SubtaskResult]) -> str:
        """Merge code outputs, handling imports and dependencies."""
        imports = set()
        code_blocks = []

        for result in results:
            if not result.success or not result.output:
                continue

            lines = result.output.split("\n")
            current_block = []

            for line in lines:
                # Extract imports
                if line.startswith("import ") or line.startswith("from "):
                    imports.add(line)
                else:
                    current_block.append(line)

            if current_block:
                code_blocks.append(
                    f"# From {result.subtask_id} ({result.role})\n"
                    + "\n".join(current_block)
                )

        # Combine imports at top, then code blocks
        output_parts = []
        if imports:
            output_parts.append("\n".join(sorted(imports)))
        if code_blocks:
            output_parts.append("\n\n".join(code_blocks))

        return "\n\n".join(output_parts)

    def _aggregate_structured(self, results: list[SubtaskResult]) -> str:
        """Combine into structured JSON output."""
        structured = {
            "results": [
                {
                    "subtask_id": r.subtask_id,
                    "role": r.role,
                    "success": r.success,
                    "output": r.output,
                    "error": r.error,
                }
                for r in results
            ],
            "summary": {
                "total": len(results),
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
            },
        }
        return json.dumps(structured, indent=2)


@dataclass
class PlanReviewResult:
    """Result of architect reviewing a plan before execution.

    Attributes:
        decision: ok, reorder, drop, add, or reroute
        score: Architect's confidence 0.0-1.0
        feedback: Brief feedback string (<15 words)
        patches: List of step-level patches to apply
        raw_response: Raw architect JSON response for debugging
    """

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


class ArchitectReviewService:
    """Service for architect to review specialist outputs.

    The architect evaluates outputs against the original spec and provides
    feedback for iteration or approval.

    IMPORTANT: Prompts are designed for minimal token output from expensive
    architect models (Qwen3-235B, Qwen3-Coder-480B).
    """

    # Concise review prompt - minimize architect output tokens
    REVIEW_PROMPT_TEMPLATE = """Review specialist output. Be BRIEF.

Objective: {objective}
Subtask: {action}
Output (truncated):
{output}

Reply JSON only (no explanation):
{{"d":"approve|changes|escalate|reject","s":0.0-1.0,"f":"<10 words","c":["fix1"]}}

d=decision, s=score, f=feedback, c=changes (optional, max 3 items)"""

    # Even more compact for simple approve/reject decisions
    QUICK_REVIEW_PROMPT = """Review: {action}
Output: {output_preview}
Reply: {{"d":"approve|changes","s":0.0-1.0,"f":"<5 words"}}"""

    # TaskIR generation prompt (for future use when frontdoor queries architect)
    TASKIR_GENERATION_PROMPT = """Break down task into subtasks. Be MINIMAL.

Task: {objective}

Reply JSON only:
{{"steps":[{{"id":"S1","actor":"coder|worker|math","action":"<10 words","out":["file.py"]}}]}}

Rules:
- Max 5 steps
- actor: coder (code), worker (docs/tests), math (proofs)
- action: imperative, <10 words
- out: expected output files"""

    # Max tokens for architect responses (expensive model)
    MAX_REVIEW_TOKENS = 128
    MAX_TASKIR_TOKENS = 256

    def __init__(
        self,
        primitives: "LLMPrimitives",
        architect_role: str = "architect_general",
    ):
        """Initialize the review service.

        Args:
            primitives: LLM primitives for calling architect
            architect_role: Role name for architect model
        """
        self.primitives = primitives
        self.architect_role = architect_role

    def review(
        self,
        spec: dict[str, Any],
        subtask: dict[str, Any],
        output: str,
        quick_mode: bool = False,
    ) -> ArchitectReview:
        """Have architect review a specialist's output.

        Args:
            spec: Original TaskIR or spec
            subtask: The subtask definition (step from plan)
            output: Specialist's output to review
            quick_mode: Use ultra-compact prompt for simple decisions

        Returns:
            ArchitectReview with decision and feedback
        """
        subtask_id = subtask.get("id", "unknown")
        action = subtask.get("action", "")

        # Truncate output aggressively to save input tokens
        output_truncated = output[:500] + "..." if len(output) > 500 else output

        # Build concise review prompt - only include objective, not full spec
        objective = spec.get("objective", "")[:200]

        if quick_mode:
            prompt = self.QUICK_REVIEW_PROMPT.format(
                action=action[:50],
                output_preview=output[:200],
            )
        else:
            prompt = self.REVIEW_PROMPT_TEMPLATE.format(
                objective=objective,
                action=action,
                output=output_truncated,
            )

        try:
            # Call architect with strict token limit
            response = self.primitives.llm_call(
                prompt,
                role=self.architect_role,
                n_tokens=self.MAX_REVIEW_TOKENS,
            )

            # Parse abbreviated JSON response
            review_data = self._parse_review_response(response)

            # Map abbreviated keys to full names
            decision_str = review_data.get("d", review_data.get("decision", "changes"))
            # Normalize decision string
            if decision_str == "changes":
                decision_str = "request_changes"
            decision = ReviewDecision(decision_str)

            return ArchitectReview(
                subtask_id=subtask_id,
                decision=decision,
                feedback=review_data.get("f", review_data.get("feedback", "")),
                score=float(review_data.get("s", review_data.get("score", 0.5))),
                suggested_changes=review_data.get("c", review_data.get("suggested_changes", [])),
                approved_output=output if decision == ReviewDecision.APPROVE else None,
            )

        except Exception as e:
            logger.warning(f"Architect review failed: {e}", exc_info=True)
            # Default to request_changes on failure
            return ArchitectReview(
                subtask_id=subtask_id,
                decision=ReviewDecision.REQUEST_CHANGES,
                feedback=f"Review failed: {e}",
                score=0.3,
            )

    # Max tokens for plan review responses
    MAX_PLAN_REVIEW_TOKENS = 128

    def review_plan(
        self,
        objective: str,
        task_type: str,
        plan_steps: list[dict[str, Any]],
        timeout_seconds: float = 15.0,
    ) -> PlanReviewResult | None:
        """Have architect review a plan before specialist execution.

        Returns PlanReviewResult on success, None on timeout/error.
        Non-blocking: never prevents request completion.

        Args:
            objective: Task objective.
            task_type: Task type (code, chat, etc.).
            plan_steps: Plan steps from TaskIR.
            timeout_seconds: Max time for review (default 15s).

        Returns:
            PlanReviewResult or None on failure.
        """
        from src.prompt_builders import build_plan_review_prompt

        prompt = build_plan_review_prompt(objective, task_type, plan_steps)

        try:
            response = self.primitives.llm_call(
                prompt,
                role=self.architect_role,
                n_tokens=self.MAX_PLAN_REVIEW_TOKENS,
            )

            # Parse abbreviated JSON response
            review_data = self._parse_review_response(response)

            decision = review_data.get("d", review_data.get("decision", "ok"))
            # Normalize valid decisions
            valid_decisions = {"ok", "reorder", "drop", "add", "reroute"}
            if decision not in valid_decisions:
                decision = "ok"

            return PlanReviewResult(
                decision=decision,
                score=float(review_data.get("s", review_data.get("score", 0.5))),
                feedback=review_data.get("f", review_data.get("feedback", "")),
                patches=review_data.get("p", review_data.get("patches", [])),
                raw_response=response[:200],
            )

        except Exception as e:
            logger.warning(f"Plan review failed: {e}", exc_info=True)
            return None  # Non-blocking — proceed without review

    def generate_taskir(self, objective: str) -> dict[str, Any]:
        """Have architect generate minimal TaskIR for an objective.

        Use this when frontdoor receives a complex task and needs
        architect to break it down into subtasks.

        Args:
            objective: The high-level task objective

        Returns:
            Minimal TaskIR dict with plan.steps

        Example output:
            {"steps": [
                {"id": "S1", "actor": "coder", "action": "Implement PBFT", "out": ["pbft.py"]},
                {"id": "S2", "actor": "math", "action": "Verify bounds", "out": ["proof.md"]}
            ]}
        """
        prompt = self.TASKIR_GENERATION_PROMPT.format(
            objective=objective[:300],  # Truncate long objectives
        )

        try:
            response = self.primitives.llm_call(
                prompt,
                role=self.architect_role,
                n_tokens=self.MAX_TASKIR_TOKENS,
            )

            taskir_data = self._parse_review_response(response)

            # Normalize abbreviated format to full TaskIR
            steps = taskir_data.get("steps", [])
            normalized_steps = []
            for i, step in enumerate(steps):
                normalized_steps.append({
                    "id": step.get("id", f"S{i+1}"),
                    "actor": step.get("actor", "worker"),
                    "action": step.get("action", ""),
                    "outputs": step.get("out", step.get("outputs", [])),
                    "inputs": step.get("in", step.get("inputs", [])),
                })

            return {
                "task_id": f"arch-{uuid.uuid4().hex[:8]}",
                "objective": objective,
                "plan": {"steps": normalized_steps},
            }

        except Exception as e:
            logger.warning(f"TaskIR generation failed: {e}", exc_info=True)
            # Return single-step fallback
            return {
                "task_id": f"arch-{uuid.uuid4().hex[:8]}",
                "objective": objective,
                "plan": {
                    "steps": [
                        {"id": "S1", "actor": "coder", "action": objective[:50], "outputs": ["output.txt"]}
                    ]
                },
            }

    def _parse_review_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from architect response."""
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Could not parse review response: {response[:200]}")
            return {"decision": "request_changes", "feedback": "Parse error"}


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
    thinking_requested: bool = False  # /think, ultrathink → use thinking_reasoning model
    architect_requested: bool = False  # /architect, /plan → use architect for planning


# Architect escalation triggers - force architect model for TASK PLANNING
# Use when you need the task broken down into subtasks
ARCHITECT_TRIGGERS = [
    "/architect",           # Explicit command
    "/plan",                # Request planning
    "[PLAN]",               # Tag-style
    "[ARCHITECT]",          # Tag-style
    "architect this",       # Natural language
    "plan this carefully",
    "break this down",
    "decompose this task",
    "create a spec",
]

# Thinking escalation triggers - force thinking model for DEEP REASONING
# Equivalent to Claude Code's "ultrathink" - routes to thinking_reasoning role
# Use when you need more thorough analysis, not task decomposition
THINKING_TRIGGERS = [
    "/think",               # Explicit command
    "ultrathink",           # Claude Code compatibility
    "[THINK]",              # Tag-style
    "think deeply",         # Natural language
    "think carefully",
    "reason through",
    "analyze carefully",
    "step by step",
    "show your reasoning",
]


def has_architect_trigger(objective: str) -> bool:
    """Check if objective requests architect for task planning/decomposition."""
    objective_lower = objective.lower()
    return any(trigger.lower() in objective_lower for trigger in ARCHITECT_TRIGGERS)


def has_thinking_trigger(objective: str) -> bool:
    """Check if objective requests thinking model for deep reasoning.

    Equivalent to Claude Code's "ultrathink" - escalates to thinking_reasoning
    role which uses a model optimized for chain-of-thought reasoning.

    Use for:
    - Complex analysis requiring multi-step reasoning
    - Problems needing thorough exploration of edge cases
    - When you want visible reasoning process
    """
    objective_lower = objective.lower()
    return any(trigger.lower() in objective_lower for trigger in THINKING_TRIGGERS)


def classify_task_complexity(objective: str) -> tuple[TaskComplexity, ComplexitySignals]:
    """Classify task complexity to determine delegation path.

    This is a fast heuristic classifier. For borderline cases, the frontdoor
    model can make the final decision via REPL exploration.

    ESCALATION TRIGGERS:
        /architect, /plan, [PLAN], "think carefully", "break this down"
        → Forces COMPLEX classification regardless of heuristics

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
        "what is", "who is", "when did", "how many", "define ",
        "hello", "hi ", "thanks", "thank you", "help",
        "what's the weather", "tell me about",
    ]
    if any(p in objective_lower for p in trivial_patterns) and len(words) < 15:
        signals.question_type = "factual"
        return TaskComplexity.TRIVIAL, signals

    # Code keywords
    code_keywords = [
        "implement", "code", "function", "class", "method",
        "write", "create", "build", "fix", "debug", "refactor",
        "add", "remove", "update", "modify", "change",
    ]
    signals.has_code_keywords = any(k in objective_lower for k in code_keywords)

    # Multi-step keywords
    multi_step_keywords = [
        "and then", "after that", "followed by", "steps",
        "first", "second", "third", "finally",
        "multiple", "several", "all the", "each",
    ]
    signals.has_multi_step_keywords = any(k in objective_lower for k in multi_step_keywords)

    # Architecture keywords (likely needs architect)
    architecture_keywords = [
        "architecture", "design", "system", "distributed",
        "scalable", "fault-tolerant", "consensus", "protocol",
        "api design", "database schema", "microservice",
        "trade-off", "compare", "evaluate", "pros and cons",
    ]
    signals.has_architecture_keywords = any(k in objective_lower for k in architecture_keywords)

    # File count estimation
    file_patterns = [
        ("files", 3), ("modules", 3), ("components", 3),
        ("services", 4), ("endpoints", 2), ("tests", 2),
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


# Role mapping for TaskIR agents to registry roles
ROLE_MAPPING = {
    "frontdoor": "frontdoor",
    "coder": "coder_primary",
    "ingest": "ingest_long_context",
    "architect": "architect_general",
    "worker": "worker_general",
    "docwriter": "worker_general",
    "math": "worker_math",
    "vision": "worker_vision",
    "toolrunner": "toolrunner",
    "formalizer": "formalizer",
}


class ProactiveDelegator:
    """Orchestrates proactive delegation workflow.

    Complexity-aware routing:
        TRIVIAL  → Frontdoor answers directly (no delegation)
        SIMPLE   → Frontdoor executes in REPL (no architect)
        MODERATE → Frontdoor delegates to single specialist (no architect)
        COMPLEX  → Architect generates TaskIR, multi-specialist workflow

    Only COMPLEX tasks invoke the expensive architect model.
    """

    def __init__(
        self,
        registry: "RegistryLoader",
        primitives: "LLMPrimitives",
        progress_logger: "ProgressLogger | None" = None,
        hybrid_router: Any | None = None,
        max_iterations: int = 3,
        max_total_iterations: int = 10,
        skip_complexity_check: bool = False,
    ):
        """Initialize the delegator.

        Args:
            registry: Model registry for role lookup
            primitives: LLM primitives for calling models
            progress_logger: Optional MemRL progress logger for Q-scoring
            hybrid_router: Optional HybridRouter for learned routing
            max_iterations: Max iterations per subtask
            max_total_iterations: Max total iterations
            skip_complexity_check: If True, always use full delegation
        """
        self.registry = registry
        self.primitives = primitives
        self.progress_logger = progress_logger
        self.hybrid_router = hybrid_router
        self.review_service = ArchitectReviewService(primitives)
        self.aggregation_service = AggregationService()
        self.skip_complexity_check = skip_complexity_check
        self.iteration_context = IterationContext(
            max_iterations=max_iterations,
            max_total_iterations=max_total_iterations,
        )

    def route_by_complexity(
        self,
        objective: str,
        task_ir: dict[str, Any] | None = None,
    ) -> tuple[TaskComplexity, str, ComplexitySignals, float]:
        """Determine delegation path based on task complexity + MemRL.

        Combines heuristic complexity classification with learned Q-values
        from episodic memory (if HybridRouter available).

        Escalation triggers:
        - /architect, /plan → forces architect for task planning
        - /think, ultrathink → uses thinking_reasoning model (set in signals)

        Returns:
            (complexity, action, signals, confidence) where:
            - action: "direct", "repl", "specialist", or "architect"
            - signals.thinking_requested: True if should use thinking model
            - confidence: 0.0-1.0 from MemRL (1.0 if no MemRL)
        """
        complexity, signals = classify_task_complexity(objective)
        confidence = 1.0

        if self.skip_complexity_check:
            return TaskComplexity.COMPLEX, "architect", signals, confidence

        # Consult MemRL HybridRouter if available
        if self.hybrid_router and task_ir:
            try:
                # HybridRouter returns (roles, strategy) - "learned" or "rules"
                roles, strategy = self.hybrid_router.route(task_ir)

                # If learned routing suggests escalation to architect, upgrade complexity
                if strategy == "learned" and any("architect" in r for r in roles):
                    complexity = TaskComplexity.COMPLEX
                    confidence = 0.8  # High confidence from learned routing

                # If learned routing suggests thinking model
                elif strategy == "learned" and any("thinking" in r for r in roles):
                    signals.thinking_requested = True
                    confidence = 0.8

                # If learned routing suggests coder directly, may downgrade to MODERATE
                elif strategy == "learned" and any("coder" in r for r in roles):
                    if complexity == TaskComplexity.COMPLEX:
                        complexity = TaskComplexity.MODERATE
                        confidence = 0.7
            except Exception as e:
                logger.warning(f"HybridRouter query failed, using heuristics: {e}")
                confidence = 0.5

        action_map = {
            TaskComplexity.TRIVIAL: "direct",
            TaskComplexity.SIMPLE: "repl",
            TaskComplexity.MODERATE: "specialist",
            TaskComplexity.COMPLEX: "architect",
        }

        return complexity, action_map[complexity], signals, confidence

    def get_target_role(
        self,
        action: str,
        signals: ComplexitySignals,
    ) -> str:
        """Get the target role based on action and escalation flags.

        Args:
            action: The delegation action (direct, repl, specialist, architect)
            signals: Complexity signals including escalation flags

        Returns:
            Role name to use for execution
        """
        # Thinking escalation overrides default role for the action
        if signals.thinking_requested:
            return "thinking_reasoning"

        role_map = {
            "direct": "frontdoor",
            "repl": "frontdoor",
            "specialist": "coder_primary",
            "architect": "architect_general",
        }
        return role_map.get(action, "frontdoor")

    def log_delegation_decision(
        self,
        task_id: str,
        complexity: TaskComplexity,
        action: str,
        confidence: float,
    ) -> None:
        """Log delegation decision for MemRL Q-learning.

        Args:
            task_id: Task identifier
            complexity: Determined complexity level
            action: Delegation action taken
            confidence: Confidence in the decision
        """
        if self.progress_logger:
            self.progress_logger.log_delegation(
                task_id=task_id,
                complexity=complexity.value,
                action=action,
                confidence=confidence,
            )

    async def delegate(self, task_ir: dict[str, Any]) -> AggregatedResult:
        """Execute proactive delegation workflow.

        Args:
            task_ir: TaskIR JSON with objective, agents, plan

        Returns:
            AggregatedResult with combined outputs
        """
        task_id = task_ir.get("task_id", str(uuid.uuid4()))
        objective = task_ir.get("objective", "")

        # Log start
        if self.progress_logger:
            self.progress_logger.log_task_started(
                task_id=task_id,
                task_ir=task_ir,
                routing_decision=["proactive_delegation"],
                routing_strategy="proactive",
            )

        result = AggregatedResult(
            task_id=task_id,
            objective=objective,
        )

        # Extract subtasks from plan
        plan = task_ir.get("plan", {})
        steps = plan.get("steps", [])

        if not steps:
            logger.warning(f"No steps in TaskIR plan for task {task_id}")
            result.aggregated_output = "[ERROR: No subtasks in plan]"
            return result

        # Execute subtasks: wave-based if parallel_execution enabled, else sequential
        from src.features import features as _get_features

        plan_start = time.monotonic()
        used_parallel = False

        if _get_features().parallel_execution and len(steps) > 1:
            from src.parallel_step_executor import compute_waves, StepExecutor

            used_parallel = True
            waves = compute_waves(steps)
            max_concurrent = plan.get("parallelism", {}).get(
                "max_concurrent_steps", 2,
            )
            executor = StepExecutor(
                primitives=self.primitives,
                review_service=self.review_service,
                iteration_context=self.iteration_context,
                max_burst_concurrent=max_concurrent,
            )
            subtask_results = await executor.execute_plan(
                task_ir, waves, ROLE_MAPPING,
            )
            for sr in subtask_results:
                result.subtask_results.append(sr)
                if sr.role not in result.roles_used:
                    result.roles_used.append(sr.role)
        else:
            for step in steps:
                subtask_result = await self._execute_with_review(task_ir, step)
                result.subtask_results.append(subtask_result)
                if subtask_result.role not in result.roles_used:
                    result.roles_used.append(subtask_result.role)

        plan_elapsed = time.monotonic() - plan_start

        # Critical path metrics (post-hoc observability)
        if used_parallel and len(result.subtask_results) > 1:
            try:
                from src.metrics.critical_path import compute_critical_path
                from src.parallel_step_executor import extract_step_timings

                timings = extract_step_timings(result.subtask_results, steps)
                cp_report = compute_critical_path(
                    timings, wall_clock_seconds=plan_elapsed,
                )
                logger.info(
                    "Critical path: %.1fs (%d steps), parallelism ratio: %.2f, "
                    "total work: %.1fs, wall clock: %.1fs",
                    cp_report.critical_path_seconds,
                    len(cp_report.critical_path_steps),
                    cp_report.parallelism_ratio,
                    cp_report.total_work_seconds,
                    cp_report.wall_clock_seconds,
                )
            except Exception as e:
                logger.debug("Critical path computation skipped: %s", e)

        # Aggregate results
        result.aggregated_output = self.aggregation_service.aggregate(
            result.subtask_results,
            strategy="concatenate",
        )
        result.all_approved = all(r.success for r in result.subtask_results)
        result.total_iterations = self.iteration_context.total_iterations

        # Log completion
        if self.progress_logger:
            self.progress_logger.log_task_completed(
                task_id=task_id,
                success=result.all_approved,
                details=f"Proactive delegation: {len(result.subtask_results)} subtasks, "
                f"{result.total_iterations} iterations",
            )

        return result

    async def _execute_with_review(
        self,
        task_ir: dict[str, Any],
        step: dict[str, Any],
    ) -> SubtaskResult:
        """Execute a subtask with architect review loop.

        Args:
            task_ir: Full TaskIR for context
            step: Step definition from plan

        Returns:
            SubtaskResult after iterations complete
        """
        subtask_id = step.get("id", f"S{uuid.uuid4().hex[:4]}")
        actor = step.get("actor", "worker")
        action = step.get("action", "")

        # Map actor to registry role
        role = ROLE_MAPPING.get(actor, "worker_general")

        # Build prompt for specialist
        prompt = self._build_specialist_prompt(task_ir, step)

        # Iteration loop
        current_output = ""
        feedback_history: list[str] = []

        while self.iteration_context.can_iterate(subtask_id):
            # Include feedback from previous iterations
            if feedback_history:
                prompt_with_feedback = (
                    prompt
                    + "\n\n## Previous Feedback\n"
                    + "\n".join(f"- {fb}" for fb in feedback_history[-3:])
                    + "\n\nAddress the feedback above."
                )
            else:
                prompt_with_feedback = prompt

            # Call specialist
            try:
                current_output = self.primitives.llm_call(
                    prompt_with_feedback,
                    role=role,
                    n_tokens=1024,
                )
            except Exception as e:
                logger.warning(f"Specialist {role} failed: {e}", exc_info=True)
                return SubtaskResult(
                    subtask_id=subtask_id,
                    role=role,
                    output="",
                    success=False,
                    error=str(e),
                )

            # Architect review
            review = self.review_service.review(
                spec=task_ir,
                subtask=step,
                output=current_output,
            )

            # Record iteration
            self.iteration_context.record_iteration(
                subtask_id=subtask_id,
                decision=review.decision,
                feedback=review.feedback,
            )

            # Log escalation if needed
            if review.decision == ReviewDecision.ESCALATE and self.progress_logger:
                self.progress_logger.log_escalation(
                    task_id=task_ir.get("task_id", ""),
                    from_tier=role,
                    to_tier="architect_general",
                    reason=f"Review escalation: {review.feedback}",
                )

            # Check decision
            if review.decision == ReviewDecision.APPROVE:
                return SubtaskResult(
                    subtask_id=subtask_id,
                    role=role,
                    output=review.approved_output or current_output,
                    success=True,
                )
            elif review.decision == ReviewDecision.REJECT:
                return SubtaskResult(
                    subtask_id=subtask_id,
                    role=role,
                    output=current_output,
                    success=False,
                    error=f"Rejected: {review.feedback}",
                )
            elif review.decision == ReviewDecision.ESCALATE:
                # Escalate to higher-tier role
                role = self._escalate_role(role)

            # Add feedback for next iteration
            if review.feedback:
                feedback_history.append(review.feedback)

        # Max iterations reached
        return SubtaskResult(
            subtask_id=subtask_id,
            role=role,
            output=current_output,
            success=False,
            error=f"Max iterations ({self.iteration_context.max_iterations}) reached",
        )

    def _build_specialist_prompt(
        self,
        task_ir: dict[str, Any],
        step: dict[str, Any],
    ) -> str:
        """Build prompt for specialist from TaskIR and step."""
        objective = task_ir.get("objective", "")
        action = step.get("action", "")
        inputs = step.get("inputs", [])
        outputs = step.get("outputs", [])

        prompt_parts = [
            f"# Task: {action}",
            "",
            f"## Overall Objective",
            objective,
            "",
        ]

        if inputs:
            prompt_parts.extend(
                [
                    "## Inputs",
                    "\n".join(f"- {i}" for i in inputs),
                    "",
                ]
            )

        if outputs:
            prompt_parts.extend(
                [
                    "## Expected Outputs",
                    "\n".join(f"- {o}" for o in outputs),
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "## Instructions",
                "Complete the task above. Provide your output directly.",
            ]
        )

        return "\n".join(prompt_parts)

    def _escalate_role(self, current_role: str) -> str:
        """Get escalated role for current role."""
        escalation_map = {
            "worker_general": "coder_primary",
            "worker_math": "coder_primary",
            "worker_vision": "coder_primary",
            "coder_primary": "architect_general",
            "coder_escalation": "architect_general",
            "frontdoor": "coder_primary",
            "architect_general": "architect_coding",
        }
        return escalation_map.get(current_role, "architect_general")

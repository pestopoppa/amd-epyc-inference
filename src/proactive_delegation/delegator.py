"""ProactiveDelegator — orchestrates proactive delegation workflow."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, TYPE_CHECKING

from src.proactive_delegation.types import (
    AggregatedResult,
    ComplexitySignals,
    IterationContext,
    ReviewDecision,
    StepExecutionError,
    SubtaskResult,
    TaskComplexity,
)
from src.proactive_delegation.complexity import (
    ROLE_MAPPING,
    classify_task_complexity,
)
from src.proactive_delegation.review_service import (
    AggregationService,
    ArchitectReviewService,
)
from src.task_ir import canonicalize_task_ir

if TYPE_CHECKING:
    from src.registry_loader import RegistryLoader
    from src.llm_primitives import LLMPrimitives
    from orchestration.repl_memory.progress_logger import ProgressLogger

logger = logging.getLogger(__name__)


class ProactiveDelegator:
    """Orchestrates proactive delegation workflow.

    Complexity-aware routing:
        TRIVIAL  -> Frontdoor answers directly (no delegation)
        SIMPLE   -> Frontdoor executes in REPL (no architect)
        MODERATE -> Frontdoor delegates to single specialist (no architect)
        COMPLEX  -> Architect generates TaskIR, multi-specialist workflow

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
        """Initialize the delegator."""
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
                task_ir = canonicalize_task_ir(task_ir)
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
        """Get the target role based on action and escalation flags."""
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
        """Log delegation decision for MemRL Q-learning."""
        if self.progress_logger:
            self.progress_logger.log_delegation(
                task_id=task_id,
                complexity=complexity.value,
                action=action,
                confidence=confidence,
            )

    async def delegate(self, task_ir: dict[str, Any]) -> AggregatedResult:
        """Execute proactive delegation workflow."""
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
                "max_concurrent_steps",
                2,
            )
            executor = StepExecutor(
                primitives=self.primitives,
                review_service=self.review_service,
                iteration_context=self.iteration_context,
                hybrid_router=self.hybrid_router,
                max_burst_concurrent=max_concurrent,
            )
            subtask_results = await executor.execute_plan(
                task_ir,
                waves,
                ROLE_MAPPING,
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

        # Delegation telemetry
        for sr in result.subtask_results:
            result.delegation_events.append(
                {
                    "from_role": "proactive_delegation",
                    "to_role": sr.role,
                    "task_summary": sr.subtask_id,
                    "success": sr.success,
                    "elapsed_ms": round(sr.elapsed_seconds * 1000),
                    "tokens_generated": sr.tokens_used,
                }
            )

        plan_elapsed = time.monotonic() - plan_start

        # Critical path metrics (post-hoc observability)
        if used_parallel and len(result.subtask_results) > 1:
            try:
                from src.metrics.critical_path import compute_critical_path
                from src.parallel_step_executor import extract_step_timings

                timings = extract_step_timings(result.subtask_results, steps)
                cp_report = compute_critical_path(
                    timings,
                    wall_clock_seconds=plan_elapsed,
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
        """Execute a subtask with architect review loop."""
        subtask_id = step.get("id", f"S{uuid.uuid4().hex[:4]}")
        actor = step.get("actor", "worker")
        step.get("action", "")

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
                exc = StepExecutionError(subtask_id, role, cause=e)
                logger.warning("%s", exc, exc_info=True)
                return SubtaskResult(
                    subtask_id=subtask_id,
                    role=role,
                    output="",
                    success=False,
                    error=str(exc),
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
            "## Overall Objective",
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

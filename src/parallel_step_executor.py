"""Wave-based step executor for TaskIR plans.

Computes execution waves from step dependencies (depends_on, parallel_group),
then executes each wave with correct ordering. Only burst worker steps
(Tier C WARM) run concurrently — HOT tier models share a single EPYC CPU
and cannot overlap without contention.

Usage:
    from src.parallel_step_executor import compute_waves, StepExecutor

    waves = compute_waves(task_ir["plan"]["steps"])
    executor = StepExecutor(primitives=primitives)
    results = await executor.execute_plan(task_ir, waves)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives
    from src.metrics.critical_path import StepTiming
    from src.proactive_delegation import (
        ArchitectReviewService,
        IterationContext,
        ReviewDecision,
        SubtaskResult,
    )

logger = logging.getLogger(__name__)

# Roles that map to WARM burst workers (8102/8112) — can genuinely overlap
BURST_WORKER_ROLES = frozenset({
    "worker_fast", "worker_fast_1", "worker_fast_2",
    "fast_1", "fast_2",
})


# ── Wave Computation ─────────────────────────────────────────────────────


@dataclass
class Wave:
    """A group of steps that can execute within the same dependency level.

    Attributes:
        index: Wave number (0-based). Wave 0 has no dependencies.
        step_ids: Step IDs in this wave.
        steps: Full step dicts from TaskIR plan.
    """

    index: int
    step_ids: list[str]
    steps: list[dict[str, Any]]


def compute_waves(steps: list[dict[str, Any]]) -> list[Wave]:
    """Compute execution waves from step dependencies via topological sort.

    Uses Kahn's algorithm to group steps into waves by dependency level.
    Steps in the same wave have all their dependencies satisfied by
    prior waves. Steps sharing a ``parallel_group`` are merged into the
    same wave (the latest wave required by any group member).

    Args:
        steps: Plan steps from TaskIR. Each step must have an ``id`` field.
            Optional fields: ``depends_on`` (list of step IDs),
            ``parallel_group`` (string group name).

    Returns:
        Ordered list of Waves. Wave 0 has no dependencies, Wave 1 depends
        only on Wave 0 steps, etc.

    Raises:
        ValueError: If circular dependencies are detected, duplicate step
            IDs exist, or ``depends_on`` references a nonexistent step.
    """
    if not steps:
        return []

    # Build ID → step map and validate uniqueness
    step_map: dict[str, dict[str, Any]] = {}
    for step in steps:
        step_id = step.get("id", "")
        if not step_id:
            raise ValueError(f"Step missing 'id' field: {step}")
        if step_id in step_map:
            raise ValueError(f"Duplicate step ID: {step_id}")
        step_map[step_id] = step

    all_ids = set(step_map)

    # Validate depends_on references
    for step_id, step in step_map.items():
        for dep in step.get("depends_on", []):
            if dep not in all_ids:
                raise ValueError(
                    f"Step {step_id} depends on nonexistent step {dep}"
                )

    # Kahn's algorithm: compute in-degree and adjacency
    in_degree: dict[str, int] = {sid: 0 for sid in all_ids}
    dependents: dict[str, list[str]] = defaultdict(list)

    for step_id, step in step_map.items():
        deps = step.get("depends_on", [])
        in_degree[step_id] = len(deps)
        for dep in deps:
            dependents[dep].append(step_id)

    # Assign wave indices via BFS
    wave_index: dict[str, int] = {}
    remaining = set(all_ids)
    current_wave = 0

    while remaining:
        # Find all steps with in_degree 0 among remaining
        ready = [sid for sid in remaining if in_degree[sid] == 0]
        if not ready:
            cycle_members = ", ".join(sorted(remaining))
            raise ValueError(
                f"Circular dependency detected among steps: {cycle_members}"
            )

        for sid in ready:
            wave_index[sid] = current_wave
            remaining.discard(sid)
            # Decrement in-degree for dependents
            for dep_id in dependents[sid]:
                in_degree[dep_id] -= 1

        current_wave += 1

    # Merge parallel_group members to the same wave (latest required)
    groups: dict[str, list[str]] = defaultdict(list)
    for step_id, step in step_map.items():
        pg = step.get("parallel_group")
        if pg:
            groups[pg].append(step_id)

    for group_name, member_ids in groups.items():
        max_wave = max(wave_index[sid] for sid in member_ids)
        for sid in member_ids:
            wave_index[sid] = max_wave

    # Build Wave objects
    waves_dict: dict[int, list[str]] = defaultdict(list)
    for step_id, wi in wave_index.items():
        waves_dict[wi].append(step_id)

    waves: list[Wave] = []
    for wi in sorted(waves_dict):
        sids = sorted(waves_dict[wi])  # Deterministic ordering
        waves.append(Wave(
            index=wi,
            step_ids=sids,
            steps=[step_map[sid] for sid in sids],
        ))

    return waves


# ── Step Timing Extraction ───────────────────────────────────────────────


def extract_step_timings(
    results: "list[SubtaskResult]",
    steps: list[dict[str, Any]],
) -> "list[StepTiming]":
    """Build StepTiming list from execution results and plan step definitions.

    Merges ``elapsed_seconds`` from SubtaskResult with ``depends_on`` from
    the original plan steps so that :func:`compute_critical_path` can
    determine the longest dependency chain.

    Args:
        results: Execution results (one per step) with timing data.
        steps: Original plan steps from TaskIR (carry ``depends_on``).

    Returns:
        List of StepTiming, one per result, preserving result order.
    """
    from src.metrics.critical_path import StepTiming

    deps_map: dict[str, tuple[str, ...]] = {
        s["id"]: tuple(s.get("depends_on", []))
        for s in steps
    }
    return [
        StepTiming(
            step_id=r.subtask_id,
            elapsed_seconds=r.elapsed_seconds,
            depends_on=deps_map.get(r.subtask_id, ()),
        )
        for r in results
    ]


# ── Step Executor ────────────────────────────────────────────────────────


@dataclass
class StepExecutor:
    """Execute TaskIR steps in dependency-ordered waves.

    Waves execute sequentially (respecting dependencies). Within a wave:
    - Burst worker steps (WARM tier) run concurrently via asyncio.gather
    - All other steps run sequentially (HOT tier models share CPU)

    Attributes:
        primitives: LLM abstraction for inference calls.
        review_service: Optional architect review for quality gating.
        iteration_context: Optional iteration tracker (for review loops).
        max_burst_concurrent: Max concurrent burst worker calls per wave.
        step_outputs: Accumulated outputs keyed by step_id.
    """

    primitives: LLMPrimitives
    review_service: ArchitectReviewService | None = None
    iteration_context: IterationContext | None = None
    hybrid_router: Any | None = None
    max_burst_concurrent: int = 2
    step_outputs: dict[str, str] = field(default_factory=dict)

    async def execute_plan(
        self,
        task_ir: dict[str, Any],
        waves: list[Wave],
        role_mapping: dict[str, str] | None = None,
    ) -> list[SubtaskResult]:
        """Execute all waves sequentially, steps within each wave by tier.

        Args:
            task_ir: Full TaskIR dict for context/objective.
            waves: Pre-computed execution waves from compute_waves().
            role_mapping: Optional actor → role name mapping.

        Returns:
            List of SubtaskResult, one per step, in plan order.
        """
        from src.proactive_delegation import SubtaskResult

        role_mapping = role_mapping or {}
        all_results: dict[str, SubtaskResult] = {}
        failed_steps: set[str] = set()

        for wave in waves:
            wave_results = await self._execute_wave(
                task_ir, wave, role_mapping, failed_steps,
            )
            for result in wave_results:
                all_results[result.subtask_id] = result
                if not result.success:
                    failed_steps.add(result.subtask_id)

        # Return in original plan step order
        step_order = []
        for wave in waves:
            for step in wave.steps:
                step_id = step["id"]
                if step_id in all_results:
                    step_order.append(all_results[step_id])

        return step_order

    async def _execute_wave(
        self,
        task_ir: dict[str, Any],
        wave: Wave,
        role_mapping: dict[str, str],
        failed_steps: set[str],
    ) -> list[SubtaskResult]:
        """Execute all steps in a single wave.

        Burst worker steps run concurrently; all others run sequentially.
        Steps whose dependencies failed are skipped.
        """
        from src.proactive_delegation import SubtaskResult

        results: list[SubtaskResult] = []

        # Partition steps by execution strategy
        burst_steps: list[dict[str, Any]] = []
        sequential_steps: list[dict[str, Any]] = []

        for step in wave.steps:
            # Check if any dependency failed
            deps = step.get("depends_on", [])
            if any(d in failed_steps for d in deps):
                results.append(SubtaskResult(
                    subtask_id=step["id"],
                    role=role_mapping.get(step.get("actor", "worker"), "worker_general"),
                    output="",
                    success=False,
                    error="Skipped: dependency failed",
                ))
                continue

            role = role_mapping.get(step.get("actor", "worker"), "worker_general")
            if role in BURST_WORKER_ROLES:
                burst_steps.append(step)
            else:
                sequential_steps.append(step)

        # Execute sequential steps first (HOT tier, one at a time)
        for step in sequential_steps:
            role = role_mapping.get(step.get("actor", "worker"), "worker_general")
            result = await self._execute_step(task_ir, step, role)
            results.append(result)
            if result.success:
                self.step_outputs[step["id"]] = result.output

        # Execute burst worker steps concurrently
        if burst_steps:
            sem = asyncio.Semaphore(self.max_burst_concurrent)

            async def _bounded_execute(s: dict[str, Any]) -> SubtaskResult:
                r = role_mapping.get(s.get("actor", "worker"), "worker_fast")
                async with sem:
                    return await self._execute_step(task_ir, s, r)

            burst_results = await asyncio.gather(
                *[_bounded_execute(s) for s in burst_steps],
                return_exceptions=True,
            )

            for step, result in zip(burst_steps, burst_results):
                if isinstance(result, Exception):
                    results.append(SubtaskResult(
                        subtask_id=step["id"],
                        role="worker_fast",
                        output="",
                        success=False,
                        error=str(result),
                    ))
                else:
                    results.append(result)
                    if result.success:
                        self.step_outputs[step["id"]] = result.output

        return results

    async def _execute_step(
        self,
        task_ir: dict[str, Any],
        step: dict[str, Any],
        role: str,
    ) -> SubtaskResult:
        """Execute a single step with optional architect review.

        Args:
            task_ir: Full TaskIR for context.
            step: Step definition from plan.
            role: Resolved role name for LLM call.

        Returns:
            SubtaskResult with output and success status.
        """
        from src.proactive_delegation import SubtaskResult, ReviewDecision

        step_id = step["id"]
        prompt = self._build_step_prompt(task_ir, step)

        # Resolve persona: explicit step field → MemRL auto-selection → None
        persona = step.get("persona") or step.get("persona_hint")
        if not persona and self.hybrid_router:
            persona = self._auto_select_persona(task_ir, step, role)

        start = time.monotonic()
        try:
            output = await asyncio.to_thread(
                self.primitives.llm_call,
                prompt,
                role=role,
                n_tokens=1024,
                persona=persona,
            )
        except Exception as e:
            logger.warning("Step %s (%s) failed: %s", step_id, role, e)
            return SubtaskResult(
                subtask_id=step_id,
                role=role,
                output="",
                success=False,
                error=str(e),
                elapsed_seconds=time.monotonic() - start,
            )

        elapsed = time.monotonic() - start

        # Optional architect review (single pass, no iteration loop)
        if self.review_service is not None:
            try:
                review = self.review_service.review(
                    spec=task_ir, subtask=step, output=output,
                )
                if review.decision == ReviewDecision.APPROVE:
                    output = review.approved_output or output
                elif review.decision == ReviewDecision.REJECT:
                    return SubtaskResult(
                        subtask_id=step_id,
                        role=role,
                        output=output,
                        success=False,
                        error=f"Rejected: {review.feedback}",
                        elapsed_seconds=elapsed,
                    )
                # REQUEST_CHANGES and ESCALATE: accept output as-is in parallel mode
                # (full iteration loop is in ProactiveDelegator's sequential path)
            except Exception as e:
                logger.warning("Review failed for step %s: %s", step_id, e)

        return SubtaskResult(
            subtask_id=step_id,
            role=role,
            output=output,
            success=True,
            elapsed_seconds=elapsed,
        )

    def _build_step_prompt(
        self,
        task_ir: dict[str, Any],
        step: dict[str, Any],
    ) -> str:
        """Build prompt for a step, injecting context from prior steps.

        Includes outputs from steps listed in ``step['inputs']`` that
        were completed in earlier waves.
        """
        parts: list[str] = []
        objective = task_ir.get("objective", "")
        action = step.get("action", "")

        parts.append(f"## Task\n{action}")

        if objective:
            parts.append(f"## Overall Objective\n{objective}")

        # Inject context from prior step outputs
        inputs = step.get("inputs", [])
        context_parts: list[str] = []
        for input_key in inputs:
            if input_key in self.step_outputs:
                context_parts.append(
                    f"### Output from {input_key}\n{self.step_outputs[input_key]}"
                )
        if context_parts:
            parts.append("## Prior Context\n" + "\n\n".join(context_parts))

        # Expected outputs
        outputs = step.get("outputs", [])
        if outputs:
            parts.append(f"## Expected Outputs\n" + ", ".join(outputs))

        return "\n\n".join(parts)

    def _auto_select_persona(
        self,
        task_ir: dict[str, Any],
        step: dict[str, Any],
        role: str,
    ) -> str | None:
        """Auto-select persona via MemRL Q-values for this (role, task_type) combo.

        Queries the HybridRouter's retriever for persona-related episodes,
        picks the highest Q-value persona above the confidence threshold.

        Returns:
            Persona name if confident match found, None otherwise.
        """
        try:
            retriever = self.hybrid_router.retriever
            task_type = task_ir.get("task_type", "chat")
            action_desc = step.get("action", "")[:100]
            query_ir = {
                "task_type": task_type,
                "objective": action_desc,
            }
            results = retriever.retrieve_for_routing(query_ir)
            if not results:
                return None

            # Filter for persona actions (format: "persona:{name}")
            persona_results = [
                r for r in results
                if r.memory.action.startswith("persona:")
            ]
            if not persona_results:
                return None

            # Pick highest Q-value above threshold
            best = max(persona_results, key=lambda r: r.q_value)
            if best.q_value >= 0.6:
                persona_name = best.memory.action.removeprefix("persona:")
                logger.debug(
                    "Auto-selected persona %s (Q=%.2f) for step %s",
                    persona_name, best.q_value, step.get("id", "?"),
                )
                return persona_name
        except Exception as exc:
            logger.debug("Persona auto-selection failed: %s", exc)
        return None

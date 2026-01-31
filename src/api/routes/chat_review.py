"""Architect review and quality gates for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: output quality detection, MemRL-conditional review gates,
architect verdict, fast revision, and plan review pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.prompt_builders import (
    build_review_verdict_prompt,
    build_revision_prompt,
)

if TYPE_CHECKING:
    from src.api.state import AppState
    from src.llm_primitives import LLMPrimitives


def _detect_output_quality_issue(answer: str) -> str | None:
    """Detect quality issues in model output using text-based heuristics.

    Inspired by GenerationMonitor's entropy/repetition signals but operating
    on complete output text (no streaming logits needed). Returns a description
    of the issue if detected, None if output looks fine.

    This is SAFE routing: only triggers on detected failure patterns,
    never on input keywords (which caused Wave 2 regressions).

    Args:
        answer: The model's complete output.

    Returns:
        Issue description if quality problem detected, None otherwise.
    """
    if not answer or len(answer) < 20:
        return None  # Too short to analyze

    words = answer.split()
    n_words = len(words)

    # 1. High n-gram repetition (degeneration loops)
    if n_words >= 20:
        trigrams = [" ".join(words[i:i+3]) for i in range(n_words - 2)]
        if trigrams:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            if unique_ratio < 0.5:  # More than 50% repeated trigrams
                return f"high_repetition (unique_ratio={unique_ratio:.2f})"

    # 2. Self-contradictory trace (model says X then says not-X)
    lines = answer.strip().split("\n")
    if n_words >= 50:
        # Check for confused analysis: very short non-empty lines mixed with long ones
        # indicates garbled/confused trace
        short_lines = sum(1 for l in lines if 0 < len(l.strip()) < 10)
        total_lines = sum(1 for l in lines if l.strip())
        if total_lines > 5 and short_lines / total_lines > 0.6:
            return "garbled_output (mostly very short lines)"

    # 3. Empty or near-empty after stripping common prefixes
    stripped = answer.strip()
    for prefix in ["```", "```python", "```json", "Here is", "The answer is"]:
        stripped = stripped.removeprefix(prefix).strip()
    if len(stripped) < 10:
        return "near_empty_output"

    return None


def _should_review(state: "AppState", task_id: str, role: str, answer: str) -> bool:
    """MemRL-conditional: review only when confidence < threshold.

    Checks Q-values for the current role+task combination. If average
    Q-value is below 0.6, the role historically struggles with this
    task type and a brief architect review is warranted.

    Args:
        state: Application state with hybrid_router.
        task_id: Current task ID.
        role: The role that generated the answer.
        answer: The answer to potentially review.

    Returns:
        True if architect review should be triggered.
    """
    if not state.hybrid_router:
        return False
    if "architect" in str(role):
        return False  # Architects ARE the reviewer — don't self-review
    if len(answer) < 50:
        return False  # Trivial answers don't need review
    try:
        # Get Q-values for this role from MemRL
        retriever = state.hybrid_router.retriever
        task_ir = {"task_type": "chat", "objective": answer[:100]}
        results = retriever.retrieve_for_routing(task_ir)
        if not results:
            return False
        # Filter for current role
        role_results = [r for r in results if r.memory.action == str(role)]
        if not role_results:
            return False
        avg_q = sum(r.q_value for r in role_results) / len(role_results)
        return avg_q < 0.6
    except Exception:
        return False


def _architect_verdict(
    question: str,
    answer: str,
    primitives: "LLMPrimitives",
    worker_digests: list[dict] | None = None,
    context_digest: str = "",
) -> str | None:
    """Get architect's hyper-concise verdict on an answer.

    The architect emits ONLY a short verdict (~20-50 tokens at 6.75 t/s → ~6s).
    Returns None if OK, or "WRONG: <corrections>" if incorrect.

    Args:
        question: Original user question.
        answer: The answer to review.
        primitives: LLM primitives for inference.
        worker_digests: Optional TOON-encodable worker digests.
        context_digest: Optional compact context summary.

    Returns:
        None if answer is OK, or "WRONG: ..." string if corrections needed.
    """
    prompt = build_review_verdict_prompt(
        question, answer,
        context_digest=context_digest,
        worker_digests=worker_digests,
    )
    try:
        result = primitives.llm_call(
            prompt,
            role="architect_general",
            n_tokens=80,  # Hard cap — verdict only
        )
        text = result.strip()
        if text.upper().startswith("OK"):
            return None
        return text  # "WRONG: <corrections>"
    except Exception:
        return None  # On error, don't block — return original answer


def _fast_revise(
    question: str,
    original_answer: str,
    corrections: str,
    primitives: "LLMPrimitives",
) -> str:
    """Fast worker expands architect's corrections into full answer.

    Uses worker_explore (port 8082, 44 t/s) — the fastest model in the stack.
    7B is sufficient since the architect already specified exactly what to fix.

    Args:
        question: Original user question.
        original_answer: The answer to revise.
        corrections: Architect's correction notes.
        primitives: LLM primitives for inference.

    Returns:
        Revised answer, or original if revision fails.
    """
    prompt = build_revision_prompt(question, original_answer, corrections)
    try:
        result = primitives.llm_call(
            prompt,
            role="worker_explore",
            n_tokens=2000,
        )
        return result.strip() or original_answer
    except Exception:
        return original_answer  # Fallback to original on error


# ── Architect Plan Review Gate ─────────────────────────────────────────────


def _needs_plan_review(
    task_ir: dict,
    routing_decision: list,
    state: "AppState",
) -> bool:
    """Determine whether the plan needs architect review before execution.

    Bypass conditions (skip review when any is true):
    1. TaskComplexity is TRIVIAL or SIMPLE
    2. TaskComplexity is COMPLEX (architect already owns plan)
    3. Single-step plan (no multi-step coordination to review)
    4. Architect is already the actor (no self-review)
    5. Phase B: Q-value >= 0.6 for task class
    6. Phase C: 90% skip (stochastic)
    7. Feature flag disabled (checked by caller)

    Args:
        task_ir: TaskIR dict with objective, task_type.
        routing_decision: List of roles selected for routing.
        state: Application state.

    Returns:
        True if architect plan review should run.
    """
    import random
    from src.proactive_delegation import classify_task_complexity, TaskComplexity

    objective = task_ir.get("objective", "")
    complexity, _signals = classify_task_complexity(objective)

    # Bypass 1+2: Only review MODERATE complexity
    if complexity != TaskComplexity.MODERATE:
        return False

    # Bypass 3: Single-step plans don't need coordination review
    plan = task_ir.get("plan", {})
    steps = plan.get("steps", [])
    if len(steps) <= 1:
        # No explicit plan steps yet — check routing for multi-role indication
        # For chat requests, routing_decision is typically 1 role, but plan
        # review is still useful if complexity is MODERATE
        pass  # Allow review for MODERATE tasks even without explicit steps

    # Bypass 4: Don't self-review architect
    if routing_decision and "architect" in str(routing_decision[0]):
        return False

    # Phase-dependent gating
    phase = state.plan_review_phase

    # Bypass 6: Phase C — 90% stochastic skip
    if phase == "C":
        if random.random() > 0.10:
            return False

    # Bypass 5: Phase B — Q-value gating
    if phase == "B" and state.hybrid_router:
        try:
            retriever = state.hybrid_router.retriever
            results = retriever.retrieve_for_routing(task_ir)
            if results:
                avg_q = sum(r.q_value for r in results) / len(results)
                if avg_q >= 0.6:
                    return False
        except Exception:
            pass  # On error, allow review

    return True


def _architect_plan_review(
    task_ir: dict,
    routing_decision: list,
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
) -> "PlanReviewResult | None":
    """Execute architect plan review and return result.

    Non-blocking: returns None on timeout or error.

    Args:
        task_ir: TaskIR dict.
        routing_decision: Current routing decision.
        primitives: LLM primitives for calling architect.
        state: Application state.
        task_id: Current task ID.

    Returns:
        PlanReviewResult or None on failure.
    """
    import logging
    from src.proactive_delegation import ArchitectReviewService

    log = logging.getLogger(__name__)

    objective = task_ir.get("objective", "")
    task_type = task_ir.get("task_type", "chat")

    # Construct plan steps from routing decision (minimal for chat requests)
    plan = task_ir.get("plan", {})
    plan_steps = plan.get("steps", [])

    # If no explicit plan, synthesize from routing_decision
    if not plan_steps and routing_decision:
        plan_steps = [
            {
                "id": f"S{i+1}",
                "actor": str(role),
                "action": objective[:50],
                "outputs": [],
            }
            for i, role in enumerate(routing_decision)
        ]

    if not plan_steps:
        return None

    review_service = ArchitectReviewService(primitives)
    result = review_service.review_plan(
        objective=objective,
        task_type=task_type,
        plan_steps=plan_steps,
    )

    if result:
        log.info(
            "Plan review: decision=%s score=%.2f feedback=%s",
            result.decision,
            result.score,
            result.feedback[:60],
        )

    return result


def _apply_plan_review(
    routing_decision: list,
    review: "PlanReviewResult",
) -> list:
    """Apply architect's plan review corrections to routing decision.

    Handles 'reroute' patches that change which specialist handles a step.

    Args:
        routing_decision: Current routing decision list.
        review: Architect's plan review result.

    Returns:
        Updated routing decision (may be unchanged if no reroute patches).
    """
    if not review.patches:
        return routing_decision

    updated = list(routing_decision)

    for patch in review.patches:
        op = patch.get("op", "")
        if op == "reroute":
            new_role = patch.get("v", "")
            step_id = patch.get("step", "")
            if new_role:
                # Map step index to routing decision index
                # S1 → index 0, S2 → index 1, etc.
                try:
                    idx = int(step_id.replace("S", "")) - 1
                    if 0 <= idx < len(updated):
                        updated[idx] = new_role
                    elif idx == 0 and len(updated) >= 1:
                        updated[0] = new_role
                except (ValueError, IndexError):
                    # If step_id isn't parseable, reroute the first step
                    if updated:
                        updated[0] = new_role

    return updated


def _store_plan_review_episode(
    state: "AppState",
    task_id: str,
    task_ir: dict,
    review: "PlanReviewResult",
) -> None:
    """Store plan review result as MemRL episode and progress log entry.

    Architect corrections become high-quality training signals:
    - review.score mapped to reward: score * 2 - 1 (0-1 → -1..+1)
    - Action: plan:{role1},{role2} (the routing decision)

    Args:
        state: Application state.
        task_id: Current task ID.
        task_ir: TaskIR dict.
        review: Architect's plan review result.
    """
    # Log to progress JSONL
    if state.progress_logger:
        from orchestration.repl_memory.progress_logger import ProgressEntry, EventType
        state.progress_logger.log(
            ProgressEntry(
                event_type=EventType.PLAN_REVIEWED,
                task_id=task_id,
                agent_role="architect_general",
                data={
                    "decision": review.decision,
                    "score": review.score,
                    "feedback": review.feedback[:100],
                    "patches": review.patches[:5],
                },
                outcome="success" if review.is_ok else "corrected",
            )
        )

    # Update plan review stats (thread-safe via GIL for dict mutations)
    stats = state._plan_review_stats
    stats["total_reviews"] = stats.get("total_reviews", 0) + 1
    if review.is_ok:
        stats["approved"] = stats.get("approved", 0) + 1
    else:
        stats["corrected"] = stats.get("corrected", 0) + 1

    # Recompute phase
    state.plan_review_phase = _compute_plan_review_phase(stats)

    # Store as MemRL episode for Q-learning (expert demonstration)
    if state.q_scorer and state.hybrid_router:
        try:
            reward = review.score * 2 - 1  # Map 0-1 to -1..+1
            state.q_scorer.score_external_result(
                task_description=task_ir.get("objective", "")[:200],
                action=f"plan_review:{review.decision}",
                reward=reward,
                context={
                    "task_type": task_ir.get("task_type", "chat"),
                    "review_decision": review.decision,
                    "review_feedback": review.feedback[:100],
                    "source": "plan_review",
                },
            )
        except Exception:
            pass  # MemRL storage is non-critical


def _compute_plan_review_phase(stats: dict) -> str:
    """Compute current plan review phase from statistics.

    Phase A (bootstrap): < 50 reviews or low Q-values
    Phase B (supervised fade): mean Q >= 0.7, min Q >= 0.5
    Phase C (spot-check): min Q >= 0.7 and >= 100 reviews

    Args:
        stats: Plan review statistics dict.

    Returns:
        Phase string: "A", "B", or "C".
    """
    total = stats.get("total_reviews", 0)
    if total < 50:
        return "A"

    q_vals = stats.get("task_class_q_values", {})
    if not q_vals:
        return "A"

    values = list(q_vals.values())
    mean_q = sum(values) / len(values)
    min_q = min(values)

    if min_q >= 0.7 and total >= 100:
        return "C"
    if mean_q >= 0.7 and min_q >= 0.5:
        return "B"
    return "A"

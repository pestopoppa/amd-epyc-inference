"""Pipeline stage 7.5: Proactive parallel delegation.

Decomposes COMPLEX tasks into parallel-executable steps via architect,
then delegates via ProactiveDelegator for wave-based execution.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import re as _re
import time

from src.api.models import ChatRequest, ChatResponse
from src.constants import TASK_IR_OBJECTIVE_LEN
from src.api.routes.chat_utils import RoutingResult
from src.api.services.memrl import score_completed_task
from src.api.structured_logging import task_extra
from src.features import features
from src.llm_primitives import LLMPrimitives
from src.task_ir import canonicalize_task_ir

log = logging.getLogger(__name__)


def _should_inline_plan_call_for_test(primitives: LLMPrimitives) -> bool:
    """Use inline call only in test/mocked contexts to avoid teardown hangs."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    llm_call = getattr(primitives, "llm_call", None)
    return type(llm_call).__module__.startswith("unittest.mock")


def _parse_plan_steps(raw: str) -> list[dict]:
    """Parse architect JSON output into validated plan step dicts.

    Tolerant of markdown fences, trailing commas, and minor formatting issues.
    Returns empty list on parse failure (caller falls through to standard flow).
    """
    text = raw.strip()

    # Strip markdown code fences if present
    text = _re.sub(r"^```(?:json)?\s*", "", text)
    text = _re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Fix trailing commas before ] (common LLM quirk)
    text = _re.sub(r",\s*]", "]", text)

    try:
        steps = _json.loads(text)
    except _json.JSONDecodeError:
        return []

    if not isinstance(steps, list):
        return []

    # Validate each step has required fields
    valid_steps = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if "id" not in step or "action" not in step:
            continue
        # Ensure defaults
        step.setdefault("actor", "worker")
        step.setdefault("depends_on", [])
        step.setdefault("outputs", [])
        valid_steps.append(step)

    return valid_steps


async def _execute_proactive(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
) -> ChatResponse | None:
    """Proactive parallel delegation for COMPLEX tasks.

    When parallel_execution feature is enabled and the task is classified as
    COMPLEX, asks the architect to decompose it into parallel-executable steps,
    then delegates via ProactiveDelegator for wave-based parallel execution.

    Returns None to fall through to standard flow if:
    - Feature not enabled
    - Task not COMPLEX
    - Architect already selected (avoids double-entry with _execute_delegated)
    - Plan parsing fails or produces < 2 steps
    """
    if not (features().parallel_execution and request.real_mode):
        return None

    from src.proactive_delegation import classify_task_complexity, TaskComplexity

    complexity, _signals = classify_task_complexity(request.prompt)
    if complexity != TaskComplexity.COMPLEX:
        return None

    # Avoid double-entry: if architect was already selected by routing, let
    # _execute_delegated() handle it (sequential TOON delegation path)
    initial_role = routing.routing_decision[0] if routing.routing_decision else "frontdoor"
    if str(initial_role) in ("architect_general", "architect_coding"):
        return None

    log.info(
        "Proactive delegation: COMPLEX task detected, requesting plan from architect",
        extra=task_extra(task_id=routing.task_id, stage="execute", mode="proactive"),
    )

    # Ask architect to decompose into parallel steps
    from src.prompt_builders import build_task_decomposition_prompt

    plan_prompt = build_task_decomposition_prompt(
        request.prompt,
        request.context or "",
    )

    try:
        if _should_inline_plan_call_for_test(primitives):
            plan_json_str = primitives.llm_call(
                plan_prompt,
                role="architect_general",
                n_tokens=256,
            )
        else:
            # Keep model I/O off the event loop in production runtime paths.
            plan_json_str = await asyncio.to_thread(
                primitives.llm_call,
                plan_prompt,
                role="architect_general",
                n_tokens=256,
            )
    except Exception as e:
        log.warning(
            "Proactive delegation: architect plan call failed: %s",
            e,
            extra=task_extra(
                task_id=routing.task_id,
                stage="execute",
                mode="proactive",
                error_type=type(e).__name__,
            ),
        )
        return None

    steps = _parse_plan_steps(plan_json_str)
    if not steps or len(steps) < 2:
        log.info(
            "Proactive delegation: plan has %d steps (need >= 2), falling through",
            len(steps),
            extra=task_extra(task_id=routing.task_id, stage="execute", mode="proactive"),
        )
        return None

    # Build TaskIR from parsed steps
    task_ir = canonicalize_task_ir({
        "task_id": routing.task_id,
        "task_type": routing.task_ir.get("task_type", "chat"),
        "objective": request.prompt[:TASK_IR_OBJECTIVE_LEN],
        "plan": {"steps": steps},
        "context_preview": request.context or "",
    })

    from src.proactive_delegation import ProactiveDelegator

    delegator = ProactiveDelegator(
        registry=state.registry,
        primitives=primitives,
        progress_logger=state.progress_logger,
        hybrid_router=state.hybrid_router,
    )

    try:
        result = await delegator.delegate(task_ir)
    except Exception as e:
        log.warning(
            "Proactive delegation: execution failed: %s",
            e,
            extra=task_extra(
                task_id=routing.task_id,
                stage="execute",
                mode="proactive",
                error_type=type(e).__name__,
            ),
        )
        return None

    answer = result.aggregated_output.strip() if result.aggregated_output else ""
    if not answer:
        return None

    elapsed = time.perf_counter() - start_time
    n_subtasks = len(result.subtask_results)
    state.increment_request(mock_mode=False, turns=1 + n_subtasks)

    if state.progress_logger:
        state.progress_logger.log_task_completed(
            task_id=routing.task_id,
            success=result.all_approved,
            details=f"Proactive delegation: {n_subtasks} subtasks, {elapsed:.3f}s",
            completion_meta={
                "producer_role": "proactive_delegation",
                "delegation_lineage": result.roles_used or ["architect_general"],
                "final_answer_role": (result.roles_used or ["architect_general"])[-1],
            },
        )
        score_completed_task(
            state,
            routing.task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )

    cache_stats = primitives.get_cache_stats() if primitives._backends else None
    delegation_events = getattr(result, "delegation_events", [])
    delegation_success = result.all_approved if delegation_events else None
    return ChatResponse(
        answer=answer,
        turns=1 + n_subtasks,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=True,
        cache_stats=cache_stats,
        routed_to="proactive_delegation",
        role_history=result.roles_used or ["architect_general"],
        routing_strategy="proactive",
        mode="proactive",
        tokens_generated=primitives.total_tokens_generated,
        formalization_applied=routing.formalization_applied,
        tools_used=0,
        delegation_events=delegation_events,
        delegation_success=delegation_success,
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
    )

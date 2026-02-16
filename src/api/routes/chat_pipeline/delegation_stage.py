"""Pipeline stage 7: Architect delegation mode.

Handles architect → specialist delegation using TOON parsing and
multi-loop execution via _architect_delegated_answer.
"""

from __future__ import annotations

import logging
import time

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat_delegation import _architect_delegated_answer
from src.api.routes.chat_utils import RoutingResult, _truncate_looped_answer
from src.api.services.memrl import score_completed_task
from src.api.structured_logging import task_extra
from src.features import features
from src.llm_primitives import LLMPrimitives

log = logging.getLogger(__name__)


def _execute_delegated(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
    initial_role,
    execution_mode: str,
) -> ChatResponse | None:
    """Handle architect delegation mode. Returns None if delegation fails (fall through)."""
    is_architect = str(initial_role) in ("architect_general", "architect_coding")
    # allow_delegation can override the feature flag per-request
    delegation_allowed = (
        request.allow_delegation if request.allow_delegation is not None
        else features().architect_delegation
    )
    use_delegation = (
        is_architect and delegation_allowed
    ) or execution_mode == "delegated"

    if not (use_delegation and request.real_mode):
        return None

    log.info(
        "Delegated mode for %s (prompt: %d chars)",
        initial_role,
        len(request.prompt),
        extra=task_extra(
            task_id=routing.task_id,
            role=str(initial_role),
            stage="execute",
            mode="delegated",
            prompt_len=len(request.prompt),
        ),
    )

    try:
        answer, delegation_stats = _architect_delegated_answer(
            question=request.prompt,
            context=request.context or "",
            primitives=primitives,
            state=state,
            architect_role=str(initial_role) if is_architect else "architect_general",
            max_loops=3,
            force_response_on_cap=True,
        )
        answer = answer.strip() if answer else ""
    except Exception as e:
        log.warning(
            "Delegation failed (%s), falling back to direct",
            e,
            extra=task_extra(
                task_id=routing.task_id,
                role=str(initial_role),
                stage="execute",
                mode="delegated",
                error_type=type(e).__name__,
            ),
        )
        return None

    if not answer:
        return None

    if not answer.startswith("[ERROR"):
        answer = _truncate_looped_answer(answer, request.prompt)

    elapsed = time.perf_counter() - start_time
    loops = delegation_stats.get("loops", 0)
    state.increment_request(mock_mode=False, turns=1 + loops)
    if state.progress_logger:
        phases_log = ", ".join(
            f"{p['phase']}{p.get('loop', '?')}={p.get('ms', '?')}ms"
            for p in delegation_stats.get("phases", [])
        )
        state.progress_logger.log_task_completed(
            task_id=routing.task_id,
            success=True,
            details=f"Delegated mode ({initial_role}), {elapsed:.3f}s, {phases_log}",
            completion_meta={
                "producer_role": str(initial_role),
                "delegation_lineage": [str(initial_role)]
                + [
                    p.get("delegate_to", "")
                    for p in delegation_stats.get("phases", [])
                    if p.get("phase") == "B"
                ],
                "final_answer_role": str(initial_role),
            },
        )
        score_completed_task(
            state,
            routing.task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )

    cache_stats = primitives.get_cache_stats() if primitives._backends else None
    delegation_events = delegation_stats.get("delegation_events", [])
    delegation_success = None
    if delegation_events:
        delegation_success = any(e.get("success") for e in delegation_events)

    return ChatResponse(
        answer=answer,
        turns=1 + loops,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=True,
        cache_stats=cache_stats,
        routed_to=str(initial_role),
        role_history=[str(initial_role)]
        + [
            p.get("delegate_to", "")
            for p in delegation_stats.get("phases", [])
            if p.get("phase") == "B"
        ],
        routing_strategy="delegated",
        mode="delegated",
        tokens_generated=primitives.total_tokens_generated,
        formalization_applied=routing.formalization_applied,
        tools_used=delegation_stats.get("tools_used", 0),
        tools_called=delegation_stats.get("tools_called", []),
        tool_timings=delegation_stats.get("tool_timings", []),
        delegation_events=delegation_events,
        delegation_success=delegation_success,
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
    )

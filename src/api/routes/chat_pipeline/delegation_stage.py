"""Pipeline stage 7: Architect delegation mode.

Handles architect → specialist delegation using TOON parsing and
multi-loop execution via _architect_delegated_answer.
"""

from __future__ import annotations

import logging
import time
from collections import Counter

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat_delegation import _architect_delegated_answer
from src.api.routes.chat_utils import RoutingResult, _truncate_looped_answer
from src.api.services.memrl import score_completed_task
from src.api.structured_logging import task_extra
from src.features import features
from src.llm_primitives import LLMPrimitives

log = logging.getLogger(__name__)


def _delegation_diagnostics(
    architect_role: str,
    role_history: list[str],
    delegation_events: list[dict],
    delegation_stats: dict,
) -> dict:
    """Build compact loop diagnostics for delegated-mode observability."""
    edge_counts = Counter(
        (e.get("from_role", ""), e.get("to_role", ""))
        for e in delegation_events
        if isinstance(e, dict)
    )
    repeated_edges = {
        f"{src}->{dst}": n
        for (src, dst), n in edge_counts.items()
        if src and dst and n > 1
    }
    role_counts = Counter(role_history)
    repeated_roles = {r: n for r, n in role_counts.items() if n > 1}
    infer = [
        (e.get("inference_meta") or {})
        for e in delegation_events
        if isinstance(e, dict)
    ]
    infer = [m for m in infer if m]
    avg_prompt_ms = None
    avg_gen_ms = None
    if infer:
        p = [float(m.get("prompt_ms", 0) or 0) for m in infer]
        g = [float(m.get("gen_ms", 0) or 0) for m in infer]
        avg_prompt_ms = round(sum(p) / max(len(p), 1), 1)
        avg_gen_ms = round(sum(g) / max(len(g), 1), 1)
    return {
        "architect_role": architect_role,
        "loops": int(delegation_stats.get("loops", 0) or 0),
        "phases_count": len(delegation_stats.get("phases", []) or []),
        "break_reason": str(delegation_stats.get("break_reason", "") or ""),
        "cap_reached": bool(delegation_stats.get("cap_reached", False)),
        "effective_max_loops": int(delegation_stats.get("effective_max_loops", 0) or 0),
        "reentrant_depth": int(delegation_stats.get("reentrant_depth", 0) or 0),
        "repeated_edges": repeated_edges,
        "repeated_roles": repeated_roles,
        "report_handles_count": len(delegation_stats.get("report_handles", []) or []),
        "report_handles": delegation_stats.get("report_handles", [])[:4],
        "delegation_inference_hops": len(infer),
        "avg_prompt_ms": avg_prompt_ms,
        "avg_gen_ms": avg_gen_ms,
    }


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
        err_text = str(e).lower()
        if (
            "lock timeout" in err_text
            or "cancelled" in err_text
            or "deadline exceeded" in err_text
            or "timed out" in err_text
            or "timeout" in err_text
        ):
            elapsed = time.perf_counter() - start_time
            diag = {
                "architect_role": str(initial_role),
                "loops": 0,
                "phases_count": 0,
                "break_reason": "pre_delegation_lock_timeout",
                "cap_reached": False,
                "effective_max_loops": 3,
                "reentrant_depth": 0,
                "repeated_edges": {},
                "repeated_roles": {},
                "report_handles_count": 0,
                "report_handles": [],
                "delegation_inference_hops": 0,
                "avg_prompt_ms": None,
                "avg_gen_ms": None,
            }
            return ChatResponse(
                answer=f"[ERROR: Delegated inference timed out/cancelled before execution: {e}]",
                turns=1,
                tokens_used=primitives.total_tokens_generated,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=True,
                cache_stats=primitives.get_cache_stats() if primitives._backends else None,
                routed_to=str(initial_role),
                role_history=[str(initial_role)],
                routing_strategy="delegated",
                mode="delegated",
                tokens_generated=primitives.total_tokens_generated,
                formalization_applied=routing.formalization_applied,
                tools_used=0,
                tools_called=[],
                tool_timings=[],
                delegation_events=[],
                delegation_diagnostics=diag,
                delegation_success=False,
                prompt_eval_ms=primitives.total_prompt_eval_ms,
                generation_ms=primitives.total_generation_ms,
                predicted_tps=primitives._last_predicted_tps,
                http_overhead_ms=primitives.total_http_overhead_ms,
                skills_retrieved=len(routing.skill_ids),
                skill_ids=routing.skill_ids,
            )
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
    role_history = [str(initial_role)] + [
        p.get("delegate_to", "")
        for p in delegation_stats.get("phases", [])
        if p.get("phase") == "B"
    ]
    delegation_diag = _delegation_diagnostics(
        architect_role=str(initial_role),
        role_history=role_history,
        delegation_events=delegation_events,
        delegation_stats=delegation_stats,
    )
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
        role_history=role_history,
        routing_strategy="delegated",
        mode="delegated",
        tokens_generated=primitives.total_tokens_generated,
        formalization_applied=routing.formalization_applied,
        tools_used=delegation_stats.get("tools_used", 0),
        tools_called=delegation_stats.get("tools_called", []),
        tool_timings=delegation_stats.get("tool_timings", []),
        delegation_events=delegation_events,
        delegation_diagnostics=delegation_diag,
        delegation_success=delegation_success,
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
    )

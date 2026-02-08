"""Pipeline stages: shared helpers, mock mode, react mode, error annotation.

Large stage functions have been split into focused modules:
    vision_stage.py      — Stage 6: _execute_vision()
    delegation_stage.py  — Stage 7: _execute_delegated()
    proactive_stage.py   — Stage 7.5: _execute_proactive(), _parse_plan_steps()
    direct_stage.py      — Stage 9: _execute_direct()

This file retains:
    _quality_escalate()  — Shared quality-check-then-escalate helper
    _execute_mock()      — Stage 4: Mock mode
    _execute_react()     — Stage 8: ReAct tool loop
    _annotate_error()    — Error annotation
"""

from __future__ import annotations

import logging
import time

from src.api.models import ChatRequest, ChatResponse
from src.api.services.memrl import score_completed_task
from src.features import features
from src.llm_primitives import LLMPrimitives
from src.roles import Role

from src.repl_environment import REPLEnvironment
from src.api.routes.chat_review import (
    _detect_output_quality_issue,
)
from src.api.routes.chat_utils import (
    RoutingResult,
    _truncate_looped_answer,
)
from src.api.structured_logging import task_extra

log = logging.getLogger(__name__)


# ── Shared helpers ──────────────────────────────────────────────────────


def _quality_escalate(
    answer: str,
    prompt: str,
    primitives: "LLMPrimitives",
    initial_role,
    *,
    allow_escalation: bool = True,
) -> tuple[str, "Role"]:
    """Detect quality issue and escalate to coder_escalation if needed.

    Returns (answer, role) — either unchanged or with escalated answer and role.
    """
    if not allow_escalation:
        return answer, initial_role
    if not (answer and not answer.startswith("[ERROR") and features().generation_monitor):
        return answer, initial_role
    quality_issue = _detect_output_quality_issue(answer)
    if not quality_issue:
        return answer, initial_role
    try:
        escalated = primitives.llm_call(
            prompt, role="coder_escalation", n_tokens=2048, skip_suffix=True,
        )
        if escalated.strip():
            return escalated.strip(), Role.CODER_ESCALATION
    except Exception as exc:
        log.debug("Quality escalation failed: %s", exc)
    return answer, initial_role


# ── Stage 4: Mock mode ──────────────────────────────────────────────────


def _execute_mock(
    request: ChatRequest,
    routing: RoutingResult,
    state,
    start_time: float,
) -> ChatResponse:
    """Handle mock mode requests with simulated response."""
    turns = 1
    answer = f"[MOCK] Processed prompt: {request.prompt[:100]}..."

    if request.context:
        answer += f" (with {len(request.context)} chars of context)"

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=True, turns=turns)

    if state.progress_logger:
        state.progress_logger.log_task_completed(
            task_id=routing.task_id,
            success=True,
            details=f"Mock response in {elapsed:.3f}s",
        )
        score_completed_task(state, routing.task_id)

    return ChatResponse(
        answer=answer,
        turns=turns,
        tokens_used=0,
        elapsed_seconds=elapsed,
        mock_mode=True,
        real_mode=False,
        cache_stats=None,
        mode="mock",
    )


# ── Stage 8: ReAct mode ─────────────────────────────────────────────────


def _execute_react(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
    initial_role,
) -> ChatResponse | None:
    """Handle ReAct tool loop mode. Returns None if react fails (fall through)."""
    if not (request.real_mode):
        return None

    log.info(
        "ReAct mode for %s (prompt: %d chars)",
        initial_role,
        len(request.prompt),
        extra=task_extra(
            task_id=routing.task_id,
            role=str(initial_role),
            stage="execute",
            mode="react",
            prompt_len=len(request.prompt),
        ),
    )

    react_tools_used = 0
    react_tools_called: list[str] = []
    react_tool_timings: list[dict] = []
    _react_registry = state.tool_registry if hasattr(state, "tool_registry") else None
    try:
        # React mode unified into REPL with structured_mode=True
        react_repl = REPLEnvironment(
            context=f"{request.prompt}\n\nContext:\n{request.context}" if request.context else request.prompt,
            llm_primitives=primitives,
            tool_registry=_react_registry,
            role=str(initial_role),
            structured_mode=True,
        )
        from src.prompt_builders import build_root_lm_prompt, extract_code_from_response, auto_wrap_final
        answer = ""
        react_last_output = ""
        react_last_error = ""
        for _turn in range(5):
            repl_state = react_repl.get_state()
            react_prompt = build_root_lm_prompt(
                state=repl_state,
                original_prompt=request.prompt,
                last_output=react_last_output,
                last_error=react_last_error,
                turn=_turn,
            )
            code = primitives.llm_call(
                react_prompt,
                role=str(initial_role),
            )
            code = extract_code_from_response(code)
            code = auto_wrap_final(code)
            result = react_repl.execute(code)
            if result.get("final"):
                answer = result["final"]
                break
            react_last_output = result.get("output", "")
            react_last_error = result.get("error", "")
        else:
            answer = react_repl.get_state()
        react_tools_used = react_repl._tool_invocations
        if react_repl.tool_registry and hasattr(react_repl.tool_registry, "get_invocation_log"):
            react_tool_timings = [
                {"tool_name": inv.tool_name, "elapsed_ms": inv.elapsed_ms, "success": inv.success}
                for inv in react_repl.tool_registry.get_invocation_log()
            ]
            react_tools_called = [inv.tool_name for inv in react_repl.tool_registry.get_invocation_log()]
        react_tools_used = max(
            react_tools_used,
            len(react_tools_called),
            len(react_tool_timings),
        )
        answer = answer.strip()
    except Exception as e:
        log.warning(
            "ReAct mode failed (%s), falling back to direct",
            e,
            extra=task_extra(
                task_id=routing.task_id,
                role=str(initial_role),
                stage="execute",
                mode="react",
                error_type=type(e).__name__,
            ),
        )
        return None

    if not answer:
        return None

    # Post-processing: truncation, quality check
    answer = _truncate_looped_answer(answer, request.prompt)

    answer, initial_role = _quality_escalate(
        answer,
        request.prompt,
        primitives,
        initial_role,
        allow_escalation=not bool(request.force_role),
    )

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=False, turns=1)
    if state.progress_logger:
        state.progress_logger.log_task_completed(
            task_id=routing.task_id,
            success=True,
            details=f"ReAct mode ({initial_role}), {elapsed:.3f}s",
        )
        score_completed_task(state, routing.task_id)

    cache_stats = primitives.get_cache_stats() if primitives._backends else None
    return ChatResponse(
        answer=answer,
        turns=1,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=True,
        cache_stats=cache_stats,
        routed_to=str(initial_role),
        role_history=[str(initial_role)],
        routing_strategy="react",
        mode="react",
        tokens_generated=primitives.total_tokens_generated,
        formalization_applied=routing.formalization_applied,
        tools_used=react_tools_used,
        tools_called=react_tools_called,
        tool_timings=react_tool_timings,
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
    )


# ── Error annotation ────────────────────────────────────────────────────


def _annotate_error(response: ChatResponse) -> ChatResponse:
    """Detect error patterns in answer and set error_code/error_detail.

    Phase 1b KV cache bug mitigation: instead of silently returning HTTP 200
    with an error string, we set structured error fields that the endpoint
    wrapper can use to return appropriate HTTP status codes.
    """
    if not response.answer:
        return response

    answer = response.answer

    # Timeout / backend failure patterns
    if answer.startswith("[ERROR:") and (
        "timed out" in answer.lower() or "timeout" in answer.lower()
    ):
        response.error_code = 504
        response.error_detail = answer
    elif answer.startswith("[ERROR:") and (
        "backend" in answer.lower() or "failed" in answer.lower()
    ):
        response.error_code = 502
        response.error_detail = answer
    elif answer.startswith("[ERROR:"):
        response.error_code = 500
        response.error_detail = answer
    elif answer.startswith("[FAILED:"):
        response.error_code = 500
        response.error_detail = answer

    return response

"""SSE streaming adapter — reuses pipeline stages for the streaming endpoint.

Provides generate_stream() which wraps pipeline stages 1-3 and 5 with SSE
event emission, eliminating the routing/preprocessing/init duplication between
_handle_chat() and chat_stream().

Gated behind features().unified_streaming flag.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncGenerator

from src.api.models import ChatRequest
from src.api.services.memrl import score_completed_task
from src.escalation import (
    EscalationAction,
    EscalationContext,
    EscalationPolicy,
)
from src.llm_primitives import LLMPrimitives
from src.prompt_builders import (
    auto_wrap_final,
    build_escalation_prompt,
    build_root_lm_prompt,
    build_routing_context,
    classify_error,
    extract_code_from_response,
)
from src.repl_environment import REPLEnvironment
from src.roles import Role
from src.sse_utils import (
    done_event,
    error_event,
    final_event,
    thinking_event,
    token_event,
    turn_end_event,
    turn_start_event,
)

from src.api.routes.chat_pipeline.routing import (
    _init_primitives,
    _plan_review_gate,
    _preprocess,
    _route_request,
)
from src.api.routes.chat_review import (
    _architect_verdict,
    _fast_revise,
    _should_review,
)
from src.api.routes.chat_utils import _resolve_answer

log = logging.getLogger(__name__)


# ── Mock-mode streaming ───────────────────────────────────────────────


async def _stream_mock(
    request: ChatRequest,
    routing,
    state,
    start_time: float,
) -> AsyncGenerator[dict, None]:
    """Yield SSE events for mock mode requests."""
    yield turn_start_event(turn=1, role=str(Role.FRONTDOOR))

    if request.thinking_budget > 0:
        for step in [
            "Analyzing the user's request...",
            f"Request type: {request.prompt[:30].split()[0] if request.prompt else 'unknown'}",
            "Determining appropriate response strategy...",
            "Preparing response...",
        ]:
            yield thinking_event(step)

    if request.permission_mode == "plan":
        analysis = f"[PLAN MODE] Would process: {request.prompt[:100]}..."
        yield token_event(analysis)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        yield turn_end_event(tokens=len(analysis), elapsed_ms=elapsed_ms)
        if state.progress_logger:
            state.progress_logger.log_task_completed(
                routing.task_id, success=True, details="Plan mode"
            )
            score_completed_task(state, routing.task_id)
        yield done_event()
        return

    mock_response = f"[MOCK] Processed: {request.prompt[:50]}..."
    for char in mock_response:
        yield token_event(char)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    yield turn_end_event(tokens=len(mock_response), elapsed_ms=elapsed_ms)

    if state.progress_logger:
        state.progress_logger.log_task_completed(
            routing.task_id, success=True, details="Mock stream"
        )
        score_completed_task(state, routing.task_id)
    yield done_event()


# ── Real-mode REPL streaming ──────────────────────────────────────────


async def _stream_repl(
    request: ChatRequest,
    routing,
    primitives: LLMPrimitives,
    state,
    start_time: float,
) -> AsyncGenerator[dict, None]:
    """Yield SSE events for real-mode REPL loop with escalation support."""
    task_id = routing.task_id

    combined_context = request.prompt
    if request.context:
        combined_context += f"\n\nContext:\n{request.context}"

    repl = REPLEnvironment(
        context=combined_context,
        llm_primitives=primitives,
        tool_registry=state.tool_registry,
        script_registry=state.script_registry,
        role=request.role or Role.FRONTDOOR,
        retriever=state.hybrid_router.retriever if state.hybrid_router else None,
        hybrid_router=state.hybrid_router,
    )

    last_output = ""
    last_error = ""
    result = None

    current_role = request.role or Role.FRONTDOOR
    consecutive_failures = 0
    role_history = [current_role]
    escalation_prompt = ""

    for turn in range(request.max_turns):
        turn_start_time_inner = time.perf_counter()
        yield turn_start_event(turn=turn + 1, role=str(current_role))

        repl_state = repl.get_state()
        if escalation_prompt:
            root_prompt = escalation_prompt
            escalation_prompt = ""
        else:
            routing_ctx = ""
            if turn == 0 and state.hybrid_router:
                routing_ctx = build_routing_context(
                    role=current_role,
                    hybrid_router=state.hybrid_router,
                    task_description=request.prompt,
                )
            root_prompt = build_root_lm_prompt(
                state=repl_state,
                original_prompt=request.prompt,
                last_output=last_output,
                last_error=last_error,
                turn=turn,
                routing_context=routing_ctx,
            )

        try:
            code = primitives.llm_call(root_prompt, role=current_role, n_tokens=1024)
        except Exception as e:
            if state.progress_logger:
                state.progress_logger.log_task_completed(
                    task_id, success=False, details=f"Root LM failed: {e}"
                )
                score_completed_task(state, task_id)
            yield error_event(f"Root LM call failed: {e}")
            yield done_event()
            return

        code = extract_code_from_response(code)
        code = auto_wrap_final(code)
        for line in code.split("\n"):
            yield token_event(line + "\n")

        result = repl.execute(code)

        # Model-initiated routing
        if repl.artifacts.get("_escalation_requested"):
            target = repl.artifacts.pop("_escalation_target", None)
            reason = repl.artifacts.pop("_escalation_reason", "Model requested")
            repl.artifacts.pop("_escalation_requested", None)

            new_role = None
            if target:
                resolved = Role.from_string(target)
                if resolved:
                    new_role = str(resolved)
            else:
                esc_ctx = EscalationContext(
                    current_role=current_role,
                    error_category="early_abort",
                    error_message=reason,
                    failure_count=1,
                    task_id=task_id,
                )
                esc_decision = EscalationPolicy().decide(esc_ctx)
                if esc_decision.should_escalate and esc_decision.target_role:
                    new_role = str(esc_decision.target_role)

            if new_role and new_role != current_role:
                turn_elapsed_ms = int(
                    (time.perf_counter() - turn_start_time_inner) * 1000
                )
                yield turn_end_event(tokens=len(code), elapsed_ms=turn_elapsed_ms)

                current_role = new_role
                role_history.append(current_role)
                consecutive_failures = 0
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=EscalationContext(
                        current_role=role_history[-2],
                        error_message=reason,
                        error_category="early_abort",
                        task_id=task_id,
                    ),
                    decision=EscalationPolicy().decide(
                        EscalationContext(
                            current_role=role_history[-2],
                            error_category="early_abort",
                            error_message=reason,
                            task_id=task_id,
                        )
                    ),
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=role_history[-2],
                        to_tier=current_role,
                        reason=f"Model-initiated: {reason}",
                    )
                continue

        # Delegation logging
        if repl.artifacts.get("_delegations"):
            for deleg in repl.artifacts["_delegations"]:
                if state.progress_logger:
                    state.progress_logger.log_exploration(
                        task_id=task_id,
                        query=deleg.get("prompt_preview", ""),
                        strategy_used=f"delegate:{deleg.get('to_role', 'unknown')}",
                        success=deleg.get("success", False),
                    )
            repl.artifacts["_delegations"] = []

        turn_elapsed_ms = int((time.perf_counter() - turn_start_time_inner) * 1000)
        yield turn_end_event(tokens=len(code), elapsed_ms=turn_elapsed_ms)

        # Completion check
        if result.is_final:
            tool_outputs = repl.artifacts.get("_tool_outputs", [])
            stream_answer = _resolve_answer(result, tool_outputs=tool_outputs)

            if _should_review(state, task_id, current_role, stream_answer):
                verdict = _architect_verdict(
                    question=request.prompt,
                    answer=stream_answer,
                    primitives=primitives,
                )
                if verdict and verdict.upper().startswith("WRONG"):
                    corrections = (
                        verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
                    )
                    stream_answer = _fast_revise(
                        question=request.prompt,
                        original_answer=stream_answer,
                        corrections=corrections,
                        primitives=primitives,
                    )

            yield final_event(stream_answer)
            break

        # Error handling with escalation
        if result.error:
            consecutive_failures += 1
            last_error = result.error
            last_output = result.output

            error_category = classify_error(result.error)
            esc_ctx = EscalationContext(
                current_role=current_role,
                error_message=result.error,
                error_category=error_category.value,
                failure_count=consecutive_failures,
                task_id=task_id,
            )
            decision = EscalationPolicy().decide(esc_ctx)

            if decision.should_escalate and decision.target_role:
                current_role = str(decision.target_role)
                role_history.append(current_role)
                consecutive_failures = 0
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=esc_ctx,
                    decision=decision,
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=role_history[-2],
                        to_tier=current_role,
                        reason=f"{decision.reason} (failures: {consecutive_failures})",
                    )
            elif decision.action == EscalationAction.FAIL:
                yield error_event(f"[FAILED: {decision.reason}]")
                break
        else:
            consecutive_failures = 0
            last_error = ""
            last_output = result.output

    # Log completion
    if state.progress_logger:
        success = result is not None and result.is_final
        role_info = (
            f", roles: {' -> '.join(str(r) for r in role_history)}"
            if len(role_history) > 1
            else ""
        )
        state.progress_logger.log_task_completed(
            task_id,
            success=success,
            details=f"Stream complete{role_info}",
        )
        score_completed_task(state, task_id)
    yield done_event()


# ── Main entry point ──────────────────────────────────────────────────


async def generate_stream(
    request: ChatRequest,
    state,
) -> AsyncGenerator[dict, None]:
    """Yield SSE events using pipeline stages shared with _handle_chat().

    Reuses _route_request, _preprocess, _init_primitives, _plan_review_gate
    instead of reimplementing them inline. The REPL streaming loop remains
    separate because it must yield per-event (unlike the batch graph runner).
    """
    start_time = time.perf_counter()

    # Stage 1: Routing (shared with _handle_chat)
    routing = _route_request(request, state)

    # Stage 2: Preprocessing (shared with _handle_chat)
    _preprocess(request, state, routing)

    # Mock mode
    if routing.use_mock:
        async for event in _stream_mock(request, routing, state, start_time):
            yield event
        return

    # Stage 3: Backend init (shared with _handle_chat)
    try:
        primitives = _init_primitives(request, state)
    except Exception as e:
        if state.progress_logger:
            state.progress_logger.log_task_completed(
                routing.task_id, success=False, details=str(e)
            )
            score_completed_task(state, routing.task_id)
        yield error_event(str(e))
        yield done_event()
        return

    # Stage 5: Plan review gate (shared with _handle_chat)
    _plan_review_gate(request, routing, primitives, state)

    # Real-mode REPL streaming
    async for event in _stream_repl(request, routing, primitives, state, start_time):
        yield event

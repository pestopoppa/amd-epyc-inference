"""Chat endpoints for the orchestrator API.

Thin orchestrator module containing only endpoint functions and the
_handle_chat() / chat_stream() pipelines. All helper functions have been
extracted into focused modules during Phase 1 decomposition:

- chat_utils.py      — Constants + utility functions
- chat_vision.py     — Vision pipeline (OCR, VL routing, ReAct VL)
- chat_summarization.py — Two-stage/three-stage context processing
- chat_review.py     — Architect review, quality gates, plan review
- chat_delegation.py — Architect delegation (TOON parsing, multi-loop)
- chat_routing.py    — Intent classification, mode selection, routing
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.dependencies import dep_app_state
from src.api.models import ChatRequest, ChatResponse, RewardRequest
from src.api.state import AppState
from src.config import get_config
from src.prompt_builders import (
    build_root_lm_prompt,
    build_routing_context,
    extract_code_from_response,
    auto_wrap_final,
    classify_error,
    build_escalation_prompt,
)
from src.api.services.memrl import (
    ensure_memrl_initialized,
    score_completed_task,
    store_external_reward,
)
from src.llm_primitives import LLMPrimitives
from src.repl_environment import REPLEnvironment
from src.escalation import (
    EscalationPolicy,
    EscalationContext,
    EscalationAction,
)
from src.roles import Role
from src.sse_utils import (
    create_sse_response,
    token_event,
    thinking_event,
    turn_start_event,
    turn_end_event,
    error_event,
    final_event,
    done_event,
)

# Decomposed modules (Phase 1 — extract-and-move, no behavior changes)
from src.api.routes.chat_utils import (
    _resolve_answer,
)
from src.api.routes.chat_review import (
    _should_review,
    _architect_verdict,
    _fast_revise,
)
from src.api.routes.chat_routing import (
    _select_mode,
)
from src.features import features

# Phase 1b: Pipeline stage functions (extracted from _handle_chat)
from src.api.routes.chat_pipeline import (
    _route_request,
    _preprocess,
    _init_primitives,
    _execute_mock,
    _execute_vision,
    _plan_review_gate,
    _execute_proactive,
    _execute_delegated,
    _execute_react,
    _execute_direct,
    _execute_repl,
    _annotate_error,
)


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    state: AppState = Depends(dep_app_state),
) -> ChatResponse:
    """Process a chat request through the orchestrator.

    Modes:
    - mock_mode=True (default): Returns simulated response, no real inference
    - real_mode=True: Uses RadixAttention caching with live llama-server instances
    - Neither: Uses legacy model server (if configured)

    The real_mode flag enables:
    - CachingBackend with prefix routing
    - Cache statistics in response
    - Full orchestration loop with Root LM (Phase 8)
    """
    # Track active requests for idle-time Q-scoring (thread-safe)
    state.increment_active()
    try:
        return await _handle_chat(request, state)
    finally:
        state.decrement_active()


@router.post("/chat/reward")
async def inject_reward(
    request: RewardRequest,
    state: AppState = Depends(dep_app_state),
):
    """Inject an external reward signal into MemRL.

    Used by orchestrator_eval.py to close the learning loop: after
    deterministic scoring of a benchmark answer, the score is fed back
    as a reward so the Q-scorer can learn from routing decisions.

    Supports precomputed embeddings via `embedding` field to avoid
    re-embedding the same task_description across multiple reward injections.
    """
    success = await asyncio.to_thread(
        store_external_reward,
        state,
        request.task_description,
        request.action,
        request.reward,
        request.context,
        request.embedding,
    )
    return {"success": success}


async def _handle_chat(request: ChatRequest, state: AppState) -> ChatResponse:
    """Thin dispatcher — routes through pipeline stages.

    Phase 1b restructure: each stage is a named function in chat_pipeline.py.
    The 1,091-line monolith is now ~80 lines of orchestration calling
    independently testable stage functions.

    Pipeline:
        1. _route_request()     → RoutingResult (role, strategy, timeout)
        2. _preprocess()        → input formalization (mutates request)
        3. _execute_mock()      → ChatResponse (mock mode early return)
        4. _init_primitives()   → LLMPrimitives (backend setup)
        5. _plan_review_gate()  → optional routing adjustment
        6. _execute_vision()    → ChatResponse | None (vision early return)
        7. Mode selection       → "direct" / "react" / "repl" / "delegated"
        8. Mode handler         → ChatResponse
        9. _annotate_error()    → set error_code/error_detail on failures
    """
    start_time = time.perf_counter()

    # Stage 1: Routing
    routing = _route_request(request, state)

    # Stage 2: Input formalization
    _preprocess(request, state, routing)

    # Stage 3: Mock mode (early return)
    if routing.use_mock:
        return _execute_mock(request, routing, state, start_time)

    # Stage 4: Initialize LLM backends
    primitives = _init_primitives(request, state)

    # Stage 5: Architect plan review gate
    _plan_review_gate(request, routing, primitives, state)

    # Stage 6: Vision pipeline (early return if image/file present)
    vision_result = await _execute_vision(request, routing, primitives, state, start_time)
    if vision_result is not None:
        return _annotate_error(vision_result)

    # Stage 6.5: Proactive delegation for COMPLEX tasks
    proactive_result = await _execute_proactive(
        request,
        routing,
        primitives,
        state,
        start_time,
    )
    if proactive_result is not None:
        return _annotate_error(proactive_result)

    # Stage 7: Mode selection
    initial_role = routing.routing_decision[0] if routing.routing_decision else Role.FRONTDOOR

    # Force REPL for vision-preprocessed requests — the document pipeline
    # populated routing.document_result, so we need DocumentREPLEnvironment
    # with a text model (frontdoor) to synthesize the answer.
    if routing.document_result is not None:
        execution_mode = "repl"
        initial_role = Role.FRONTDOOR
    elif request.force_mode and request.force_mode in ("direct", "react", "repl", "delegated"):
        execution_mode = request.force_mode
    else:
        execution_mode = _select_mode(request.prompt, request.context or "", state)

    # Stage 8: Execute selected mode (with fallthrough on failure)

    # 8a: Delegated mode (architect → specialist)
    # _execute_delegated checks delegation_allowed internally and returns None if not allowed
    result = await asyncio.to_thread(
        _execute_delegated,
        request,
        routing,
        primitives,
        state,
        start_time,
        initial_role,
        execution_mode,
    )
    if result is not None:
        return _annotate_error(result)

    # 8b: ReAct tool loop mode
    if execution_mode == "react":
        result = await asyncio.to_thread(
            _execute_react,
            request,
            routing,
            primitives,
            state,
            start_time,
            initial_role,
        )
        if result is not None:
            return _annotate_error(result)

    # 8c: Direct LLM call mode
    if execution_mode == "direct" and request.real_mode:
        return _annotate_error(
            await asyncio.to_thread(
                _execute_direct,
                request,
                routing,
                primitives,
                state,
                start_time,
                initial_role,
            )
        )

    # 8d: REPL orchestration mode (default fallback)
    return _annotate_error(
        await _execute_repl(request, routing, primitives, state, start_time, initial_role)
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    state: AppState = Depends(dep_app_state),
) -> StreamingResponse:
    """SSE streaming endpoint with routing metadata.

    Streams events using standardized SSE format (via sse_utils):
    - turn_start: {type: "turn_start", turn: N, role: "..."}
    - thinking: {type: "thinking", content: "..."} (when thinking_budget > 0)
    - token: {type: "token", content: "..."}
    - tool: {type: "tool", name: "...", args: {...}, result: ...}
    - permission_request: {type: "permission_request", id: "...", tool: "...", args: {...}}
    - file: {type: "file", path: "...", content: "...", action: "create"|"modify"}
    - turn_end: {type: "turn_end", tokens: N, elapsed_ms: N}
    - error: {type: "error", message: "..."}
    - done: [DONE] when complete

    Parameters:
    - thinking_budget: Token budget for internal reasoning (0=disabled)
    - permission_mode: "normal", "auto-accept", or "plan"

    Note: Uses sse-starlette when available (via feature flag), otherwise
    falls back to manual SSE formatting for backward compatibility.
    """
    # Unified streaming path — reuses pipeline stages from _handle_chat()
    if features().unified_streaming:
        from src.api.routes.chat_pipeline.stream_adapter import generate_stream

        return create_sse_response(generate_stream(request, state))

    # Legacy inline streaming path (preserved until unified_streaming is validated)
    # Generate task ID for MemRL tracking (outside generator for closure)
    task_id = f"stream-{uuid.uuid4().hex[:8]}"

    async def generate() -> AsyncGenerator[dict, None]:
        start_time = time.perf_counter()
        use_mock = request.mock_mode and not request.real_mode

        # Construct task_ir and log start (MemRL integration)
        task_ir = {
            "task_type": "chat_stream",
            "objective": request.prompt[:200],
            "priority": "interactive",
        }
        if state.progress_logger:
            state.progress_logger.log_task_started(
                task_id=task_id,
                task_ir=task_ir,
                routing_decision=[request.role],
                routing_strategy="mock" if use_mock else "rules",
            )

        # Mock mode
        if use_mock:
            # Emit turn start
            yield turn_start_event(turn=1, role=str(Role.FRONTDOOR))

            # Emit thinking events if thinking_budget > 0 (Claude Code parity)
            if request.thinking_budget > 0:
                thinking_steps = [
                    "Analyzing the user's request...",
                    f"Request type: {request.prompt[:30].split()[0] if request.prompt else 'unknown'}",
                    "Determining appropriate response strategy...",
                    "Preparing response...",
                ]
                for step in thinking_steps:
                    yield thinking_event(step)

            # Check permission mode - in plan mode, only emit analysis
            if request.permission_mode == "plan":
                analysis = f"[PLAN MODE] Would process: {request.prompt[:100]}..."
                yield token_event(analysis)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                yield turn_end_event(tokens=len(analysis), elapsed_ms=elapsed_ms)
                # Log completion (MemRL)
                if state.progress_logger:
                    state.progress_logger.log_task_completed(
                        task_id, success=True, details="Plan mode"
                    )
                    score_completed_task(state, task_id)
                yield done_event()
                return

            # Simulate streaming tokens
            mock_response = f"[MOCK] Processed: {request.prompt[:50]}..."
            for char in mock_response:
                yield token_event(char)

            # Emit turn end
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            yield turn_end_event(tokens=len(mock_response), elapsed_ms=elapsed_ms)

            # Log completion (MemRL)
            if state.progress_logger:
                state.progress_logger.log_task_completed(
                    task_id, success=True, details="Mock stream"
                )
                score_completed_task(state, task_id)
            yield done_event()
            return

        # Real mode - initialize MemRL components on first real use (lazy loading)
        ensure_memrl_initialized(state)

        server_urls = request.server_urls or get_config().server_urls.as_dict()
        try:
            primitives = LLMPrimitives(
                mock_mode=False, server_urls=server_urls, registry=state.registry
            )
        except Exception as e:
            # Log failure (MemRL)
            if state.progress_logger:
                state.progress_logger.log_task_completed(task_id, success=False, details=str(e))
                score_completed_task(state, task_id)
            yield error_event(str(e))
            yield done_event()
            return

        # Create REPL
        combined_context = request.prompt
        if request.context:
            combined_context += f"\n\nContext:\n{request.context}"

        repl = REPLEnvironment(
            context=combined_context,
            llm_primitives=primitives,
            tool_registry=state.tool_registry,
            script_registry=state.script_registry,
            role=request.role or Role.FRONTDOOR,
            # MemRL components for model self-routing
            retriever=state.hybrid_router.retriever if state.hybrid_router else None,
            hybrid_router=state.hybrid_router,
        )

        # Root LM loop with streaming + escalation support
        last_output = ""
        last_error = ""
        result = None

        # Escalation tracking (parity with main endpoint)
        current_role = request.role or Role.FRONTDOOR
        consecutive_failures = 0
        role_history = [current_role]
        escalation_prompt = ""

        for turn in range(request.max_turns):
            turn_start_time_inner = time.perf_counter()

            # Emit turn start with current role
            yield turn_start_event(turn=turn + 1, role=str(current_role))

            # Get state and build prompt
            repl_state = repl.get_state()
            if escalation_prompt:
                root_prompt = escalation_prompt
                escalation_prompt = ""
            else:
                # Inject routing context on turn 0
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

            # Call Root LM with current role
            try:
                code = primitives.llm_call(root_prompt, role=current_role, n_tokens=1024)
            except Exception as e:
                # Log failure (MemRL)
                if state.progress_logger:
                    state.progress_logger.log_task_completed(
                        task_id, success=False, details=f"Root LM failed: {e}"
                    )
                    score_completed_task(state, task_id)
                yield error_event(f"Root LM call failed: {e}")
                yield done_event()
                return

            # Stream the generated code tokens
            code = extract_code_from_response(code)
            # Auto-wrap in FINAL() if code looks like a complete answer
            code = auto_wrap_final(code)
            for line in code.split("\n"):
                yield token_event(line + "\n")

            # Execute in REPL
            result = repl.execute(code)

            # Check model-initiated routing artifacts
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
                    # Emit turn end before role switch
                    turn_elapsed_ms = int((time.perf_counter() - turn_start_time_inner) * 1000)
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
                    continue  # Next turn with new role

            # Log delegation outcomes
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

            # Emit turn end
            turn_elapsed_ms = int((time.perf_counter() - turn_start_time_inner) * 1000)
            yield turn_end_event(tokens=len(code), elapsed_ms=turn_elapsed_ms)

            # Check for completion
            if result.is_final:
                tool_outputs = repl.artifacts.get("_tool_outputs", [])
                stream_answer = _resolve_answer(result, tool_outputs=tool_outputs)

                # MemRL-informed quality review gate (blocking, streaming parity)
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

            # Handle errors with escalation policy
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

        # Log completion (MemRL) - success if we got a final answer
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

    # Use SSE utilities for response (handles sse-starlette vs manual fallback)
    return create_sse_response(generate())

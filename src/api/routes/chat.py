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
import logging
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

log = logging.getLogger(__name__)

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

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
from src.prompt_builders.builder import build_corpus_context
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
    tool_start_event,
    tool_end_event,
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

if TYPE_CHECKING:
    from src.api.routes.chat_utils import RoutingResult

# Phase 1b: Pipeline stage functions (extracted from _handle_chat)
from src.api.routes.chat_pipeline import (
    _route_request,
    _preprocess,
    _init_primitives,
    _execute_mock,
    _execute_vision,
    _execute_vision_multimodal,
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
        response = await _handle_chat(request, state)
        # Return appropriate HTTP status instead of silent 200 OK on failure
        if response.error_code:
            headers = {}
            if response.error_code == 503:
                headers["Retry-After"] = "30"
            return JSONResponse(
                status_code=response.error_code,
                content=response.model_dump(),
                headers=headers or None,
            )
        return response
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


async def _try_cheap_first(
    request: ChatRequest,
    routing: "RoutingResult",
    primitives: "LLMPrimitives",
    state: AppState,
    start_time: float,
    initial_role,
    execution_mode: str,
) -> ChatResponse | None:
    """Speculative pre-filter: try answering with cheap 7B model first.

    Returns ChatResponse if the cheap answer passes quality gate,
    or None to fall through to the normal pipeline.

    The cheap attempt is a speculative optimization layer ABOVE the existing
    delegation/escalation chain. On failure, the 7B's output can be passed
    as context to the specialist, giving the expensive model a head start.
    """
    import logging

    log = logging.getLogger(__name__)

    cfg = get_config().chat
    if not cfg.try_cheap_first_enabled:
        return None

    # Skip for forced modes, vision, delegation, or already-cheap roles
    cheap_role = cfg.try_cheap_first_role
    if request.force_mode or request.force_role:
        return None
    if str(initial_role) in {"worker_explore", "worker_math", "worker_vision"}:
        return None  # Already cheap
    if execution_mode == "delegated":
        return None

    # Phase B/C: check Q-value before attempting
    if cfg.try_cheap_first_phase in ("B", "C"):
        if hasattr(state, "hybrid_router") and state.hybrid_router is not None:
            try:
                task_ir = {
                    "task_type": "chat",
                    "objective": request.prompt[:200],
                }
                results = state.hybrid_router.retriever.retrieve_for_routing(task_ir)
                # Check if worker_explore Q-value is above threshold
                worker_q = 0.0
                for r in results:
                    if r.memory.metadata.get("role") == cheap_role:
                        worker_q = max(worker_q, r.q_value)
                if worker_q < cfg.try_cheap_first_q_threshold:
                    return None  # MemRL says cheap won't work for this task class
            except Exception:
                pass  # Fall through to Phase A behavior

    # Attempt cheap answer
    prompt = request.prompt
    if request.context:
        prompt = f"{request.context}\n\n{request.prompt}"

    try:
        from src.api.routes.chat_utils import QWEN_STOP

        answer = primitives.llm_call(
            prompt,
            role=cheap_role,
            n_tokens=cfg.try_cheap_first_max_tokens,
            skip_suffix=True,
            stop_sequences=["\n\n\n", QWEN_STOP],
        )
        answer = answer.strip()
    except Exception as e:
        log.debug("Cheap-first attempt failed: %s", e)
        return None

    # Quality gate: reject empty, very short, or error-like answers
    if not answer or len(answer) < 20:
        return None
    if answer.startswith("[ERROR"):
        return None

    # Check for obvious quality issues
    from src.api.routes.chat_review import _detect_output_quality_issue

    quality_issue = _detect_output_quality_issue(answer)
    if quality_issue:
        log.debug("Cheap-first quality gate failed: %s", quality_issue)
        return None

    # Passed quality gate — return cheap answer
    elapsed = time.perf_counter() - start_time
    log.info(
        "Try-cheap-first PASSED: %s answered in %.1fs (phase %s)",
        cheap_role, elapsed, cfg.try_cheap_first_phase,
    )

    return ChatResponse(
        answer=answer,
        turns=1,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=True,
        routed_to=cheap_role,
        role_history=[cheap_role],
        routing_strategy=f"cheap_first:{routing.routing_strategy}",
        generation_ms=elapsed * 1000,
        mode="direct",
        cheap_first_attempted=True,
        cheap_first_passed=True,
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
    )


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

    vision_roles = {"worker_vision", "vision_escalation"}
    forced_mode = request.force_mode if request.force_mode in ("direct", "react", "repl", "delegated") else None

    # Vision-preprocessed requests need REPL document context unless an explicit
    # forced mode is set. For multi-file/document workflows we keep synthesis on
    # frontdoor; for pure image workflows preserve vision role if selected/forced.
    if routing.document_result is not None:
        execution_mode = forced_mode or "repl"
        if request.files:
            initial_role = Role.FRONTDOOR
        elif request.force_role in vision_roles:
            initial_role = request.force_role
        elif str(initial_role) not in vision_roles:
            initial_role = Role.FRONTDOOR
    elif forced_mode:
        execution_mode = forced_mode
    else:
        execution_mode = _select_mode(request.prompt, request.context or "", state)

    # Stage 7.5: Vision multimodal handler
    # Text-only paths (_execute_direct, _execute_repl) discard image data.
    # When a vision role has image data, route through the VL handler instead.
    if str(initial_role) in vision_roles and (request.image_path or request.image_base64):
        vision_mm = await _execute_vision_multimodal(
            request, routing, primitives, state, start_time, initial_role, execution_mode,
        )
        if vision_mm is not None:
            return _annotate_error(vision_mm)

    # Stage 7.9: Try-cheap-first speculative pre-filter.
    # Attempts the task with the cheapest HOT model (7B, 44 t/s) before
    # routing to expensive specialists. On quality gate pass, returns the
    # cheap answer (2-3x faster). On fail, falls through to normal pipeline.
    cheap_result = await _try_cheap_first(
        request, routing, primitives, state, start_time, initial_role, execution_mode,
    )
    if cheap_result is not None:
        return _annotate_error(cheap_result)

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
                mock_mode=False,
                server_urls=server_urls,
                registry=state.registry,
                health_tracker=state.health_tracker,
                admission_controller=getattr(state, "admission", None),
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
        prev_tool_count = 0

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
                corpus_ctx = ""
                if turn == 0:
                    if state.hybrid_router:
                        routing_ctx = build_routing_context(
                            role=current_role,
                            hybrid_router=state.hybrid_router,
                            task_description=request.prompt,
                        )
                    corpus_ctx = build_corpus_context(
                        role=current_role,
                        task_description=request.prompt,
                    )
                root_prompt = build_root_lm_prompt(
                    state=repl_state,
                    original_prompt=request.prompt,
                    last_output=last_output,
                    last_error=last_error,
                    turn=turn,
                    routing_context=routing_ctx,
                    corpus_context=corpus_ctx,
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

            # Emit tool events for any tool invocations in this turn
            if repl.tool_registry:
                inv_log = repl.tool_registry.get_invocation_log()
                for inv in inv_log[prev_tool_count:]:
                    yield tool_start_event(inv.tool_name)
                    yield tool_end_event(inv.tool_name, int(inv.elapsed_ms), inv.success)
                prev_tool_count = len(inv_log)

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

                if decision.should_think_harder and decision.config_override:
                    cot_prefix = decision.config_override.get("cot_prefix", "")
                    if cot_prefix:
                        escalation_prompt = cot_prefix + root_prompt
                    log.info("Think-harder in legacy stream: %s", decision.reason)
                elif decision.should_escalate and decision.target_role:
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

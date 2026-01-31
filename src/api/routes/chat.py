"""Chat endpoints for the orchestrator API.

Thin orchestrator module containing only endpoint functions and the
_handle_chat() / chat_stream() pipelines. All helper functions have been
extracted into focused modules during Phase 1 decomposition:

- chat_utils.py      — Constants + utility functions
- chat_vision.py     — Vision pipeline (OCR, VL routing, ReAct VL)
- chat_summarization.py — Two-stage/three-stage context processing
- chat_review.py     — Architect review, quality gates, plan review
- chat_react.py      — ReAct tool loop (Thought/Action/Observation)
- chat_delegation.py — Architect delegation (TOON parsing, multi-loop)
- chat_routing.py    — Intent classification, mode selection, routing
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.models import ChatRequest, ChatResponse
from src.api.state import get_state
from src.prompt_builders import (
    build_root_lm_prompt,
    build_long_context_exploration_prompt,
    build_routing_context,
    extract_code_from_response,
    auto_wrap_final,
    classify_error,
    build_escalation_prompt,
)
from src.api.services.memrl import (
    ensure_memrl_initialized,
    score_completed_task,
)
from src.features import features
from src.llm_primitives import LLMPrimitives
from src.repl_environment import REPLEnvironment
from src.escalation import (
    EscalationPolicy,
    EscalationContext,
    EscalationAction,
)
from src.generation_monitor import GenerationMonitor, MonitorConfig
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
    TWO_STAGE_CONFIG,
    QWEN_STOP,
    LONG_CONTEXT_CONFIG,
    _resolve_answer,
    _strip_tool_outputs,
    _truncate_looped_answer,
    _should_formalize,
    _formalize_output,
)
from src.api.routes.chat_vision import (
    _handle_vision_request,
    _vision_react_mode_answer,
    _handle_multi_file_vision,
)
from src.api.routes.chat_summarization import (
    _should_use_two_stage,
    _run_two_stage_summarization,
)
from src.api.routes.chat_review import (
    _detect_output_quality_issue,
    _should_review,
    _architect_verdict,
    _fast_revise,
    _needs_plan_review,
    _architect_plan_review,
    _apply_plan_review,
    _store_plan_review_episode,
)
from src.api.routes.chat_react import (
    _react_mode_answer,
)
from src.api.routes.chat_delegation import (
    _architect_delegated_answer,
)
from src.api.routes.chat_routing import (
    _select_mode,
    _classify_and_route,
)


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
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
    state = get_state()

    # Track active requests for idle-time Q-scoring (thread-safe)
    state.increment_active()
    try:
        return await _handle_chat(request)
    finally:
        state.decrement_active()


async def _handle_chat(request: ChatRequest) -> ChatResponse:
    """Internal handler for chat requests."""
    state = get_state()
    start_time = time.perf_counter()

    # Generate task ID for MemRL tracking
    task_id = f"chat-{uuid.uuid4().hex[:8]}"

    # Construct task_ir for logging
    task_ir = {
        "task_type": "chat",
        "objective": request.prompt[:200],
        "priority": "interactive",
    }

    # Determine mode (real_mode takes precedence over mock_mode)
    use_mock = request.mock_mode and not request.real_mode

    # Initialize MemRL early for real_mode to enable HybridRouter
    if request.real_mode and not use_mock:
        ensure_memrl_initialized(state)

    # Determine routing using HybridRouter if available, otherwise rules
    if use_mock:
        routing_decision = [request.role or Role.FRONTDOOR]
        routing_strategy = "mock"
    elif request.force_role:
        # Forced role bypasses ALL routing (for comparative seeding/testing)
        routing_decision = [request.force_role]
        routing_strategy = "forced"
    elif request.role and request.role not in ("", "frontdoor"):
        # Explicit specialist role requested — honor it
        routing_decision = [request.role]
        routing_strategy = "explicit"
    elif state.hybrid_router and request.real_mode:
        # Use learned routing for real-mode requests
        routing_decision, routing_strategy = state.hybrid_router.route(task_ir)
    else:
        # Proactive intent classification — route to specialist based on prompt
        has_image = bool(request.image_path or request.image_base64)
        classified_role, routing_strategy = _classify_and_route(
            request.prompt, request.context or "", has_image=has_image,
        )
        routing_decision = [classified_role]

    # Phase 3: Failure graph veto — if a specialist has high failure risk,
    # revert to frontdoor as safety layer. This catches sudden specialist
    # degradation that gradual Q-value updates might miss.
    if (
        state.failure_graph
        and routing_decision
        and str(routing_decision[0]) != str(Role.FRONTDOOR)
        and routing_strategy not in ("mock", "forced")
    ):
        try:
            risk = state.failure_graph.get_failure_risk(str(routing_decision[0]))
            if risk > 0.5:
                import logging as _veto_log
                _veto_log.getLogger(__name__).warning(
                    f"Failure veto: {routing_decision[0]} risk={risk:.2f} > 0.5, "
                    f"reverting to frontdoor"
                )
                routing_decision = [str(Role.FRONTDOOR)]
                routing_strategy = "failure_vetoed"
        except Exception:
            pass  # Veto check is non-critical

    # Log task start (MemRL integration)
    if state.progress_logger:
        state.progress_logger.log_task_started(
            task_id=task_id,
            task_ir=task_ir,
            routing_decision=routing_decision,
            routing_strategy=routing_strategy,
        )

    # Phase 4: Input formalization preprocessing
    # If enabled and prompt qualifies, extract formal specification
    # before specialist handles the request.
    formalization_applied = False
    if (
        features().input_formalizer
        and request.real_mode
        and not use_mock
        and routing_strategy not in ("mock",)
    ):
        from src.formalizer import should_formalize_input, formalize_prompt, inject_formalization
        should_fml, problem_hint = should_formalize_input(request.prompt)
        if should_fml:
            fml_result = formalize_prompt(
                request.prompt, problem_hint, state.registry
            )
            if fml_result.success:
                request.context = inject_formalization(
                    request.prompt, request.context or "", fml_result.ir_json
                )
                formalization_applied = True
                import logging as _fml_log
                _fml_log.getLogger(__name__).info(
                    "Input formalization: %s (%.1fs, %s)",
                    problem_hint,
                    fml_result.elapsed_seconds,
                    fml_result.model_role,
                )

    if use_mock:
        # Mock mode: simulate orchestration
        turns = 1
        answer = f"[MOCK] Processed prompt: {request.prompt[:100]}..."

        if request.context:
            answer += f" (with {len(request.context)} chars of context)"

        elapsed = time.perf_counter() - start_time
        state.increment_request(mock_mode=True, turns=turns)

        # Log task completion (MemRL integration)
        if state.progress_logger:
            state.progress_logger.log_task_completed(
                task_id=task_id,
                success=True,
                details=f"Mock response in {elapsed:.3f}s",
            )
            score_completed_task(state, task_id)

        return ChatResponse(
            answer=answer,
            turns=turns,
            tokens_used=0,
            elapsed_seconds=elapsed,
            mock_mode=True,
            real_mode=False,
            cache_stats=None,
        )

    # Real mode: use RadixAttention with CachingBackend
    if request.real_mode:
        # MemRL already initialized earlier for HybridRouter routing

        # Get server URLs from request or use defaults
        server_urls = request.server_urls or LLMPrimitives.DEFAULT_SERVER_URLS

        # Reuse shared LLMPrimitives instance for connection pooling.
        # Each request resets per-request counters but keeps persistent
        # httpx connections (~6x latency reduction from keepalive).
        if (
            hasattr(state, '_real_primitives')
            and state._real_primitives is not None
            and not request.server_urls  # Custom URLs require fresh instance
        ):
            primitives = state._real_primitives
            primitives.reset_counters()
        else:
            try:
                primitives = LLMPrimitives(
                    mock_mode=False,
                    server_urls=server_urls,
                    registry=state.registry,
                )
                if not request.server_urls:
                    state._real_primitives = primitives
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to initialize real mode backends: {e}",
                )

        # Forward per-request cache_prompt override to primitives
        primitives.cache_prompt = request.cache_prompt

        # Verify at least one backend is available
        if not primitives._backends:
            raise HTTPException(
                status_code=503,
                detail="No backends available. Ensure llama-server is running on configured ports.",
            )

    else:
        # Legacy mode: use ModelServer (if configured)
        primitives = LLMPrimitives(mock_mode=False, registry=state.registry)

        if primitives.model_server is None:
            raise HTTPException(
                status_code=503,
                detail="Real inference not available: no model server configured",
            )

    # ── Architect plan review gate ───────────────────────────────────────
    # Synchronous architect review of the frontdoor's tentative plan.
    # Non-blocking: returns None on timeout/error, proceeds without review.
    plan_review_result = None
    if (
        request.real_mode
        and features().plan_review
        and _needs_plan_review(task_ir, routing_decision, state)
    ):
        plan_review_result = _architect_plan_review(
            task_ir, routing_decision, primitives, state, task_id
        )
        if plan_review_result and plan_review_result.decision != "ok":
            routing_decision = _apply_plan_review(routing_decision, plan_review_result)
        if plan_review_result:
            _store_plan_review_episode(state, task_id, task_ir, plan_review_result)

    # Vision routing: when image data or files are present, route through vision pipeline
    # instead of standard text-only orchestration.
    # Supports specialist routing: force_role selects VL server, force_mode selects
    # direct (OCR pre-chaining + VL) vs react (VL model decides when to call OCR).
    has_vision_input = (request.image_path or request.image_base64 or request.files)
    if request.real_mode and has_vision_input:
        import logging as _vl_log
        _vl_logger = _vl_log.getLogger(__name__)

        # Determine vision server and mode from routing decision
        vision_roles = {"worker_vision", "vision_escalation"}
        forced_role = request.force_role or ""
        forced_mode = request.force_mode or ""
        routed_to_vision = "worker_vision"  # default

        # Map forced role to server constraint
        force_server = None
        if forced_role in vision_roles:
            force_server = forced_role
            routed_to_vision = forced_role

        # Determine execution mode for vision
        # worker_vision supports: direct, react
        # vision_escalation supports: direct only (0% agentic)
        vision_exec_mode = forced_mode if forced_mode in ("direct", "react") else "direct"
        if forced_role == "vision_escalation" and vision_exec_mode == "react":
            _vl_logger.warning("vision_escalation cannot do react mode, forcing direct")
            vision_exec_mode = "direct"

        vision_tools_used = 0
        vision_tools_called: list[str] = []
        try:
            if request.files and len(request.files) > 0:
                answer = await _handle_multi_file_vision(request, primitives, state, task_id)
            elif vision_exec_mode == "react" and force_server in (None, "worker_vision"):
                # Vision ReAct: VL model decides when to invoke OCR
                import base64 as _b64

                image_b64 = request.image_base64
                if not image_b64 and request.image_path:
                    from pathlib import Path
                    image_b64 = _b64.b64encode(Path(request.image_path).read_bytes()).decode("utf-8")

                # Detect MIME type
                mime_type = "image/jpeg"
                try:
                    raw = _b64.b64decode(image_b64[:32])
                    if raw[:4] == b'\x89PNG':
                        mime_type = "image/png"
                    elif raw[:4] == b'RIFF':
                        mime_type = "image/webp"
                except Exception:
                    pass

                vl_port = 8086  # worker_vision
                _vl_logger.info(f"Vision ReAct mode on port {vl_port}")
                answer, vision_tools_used, vision_tools_called = await _vision_react_mode_answer(
                    prompt=request.prompt,
                    image_b64=image_b64,
                    mime_type=mime_type,
                    context=request.context or "",
                    vl_port=vl_port,
                    max_turns=5,
                )
                routed_to_vision = "worker_vision"
            else:
                # Direct mode: OCR pre-chaining + VL (existing path)
                # OCR is always attempted even in direct mode
                answer = await _handle_vision_request(
                    request, primitives, state, task_id,
                    force_server=force_server,
                )
                vision_tools_used = 1  # OCR pre-processing counts as a tool call
                vision_tools_called = ["ocr_extract"]
                if force_server:
                    routed_to_vision = force_server

            elapsed = time.perf_counter() - start_time
            state.increment_request(mock_mode=False, turns=1)

            if state.progress_logger:
                mode_label = f"vision/{vision_exec_mode}"
                state.progress_logger.log_task_completed(
                    task_id=task_id,
                    success=True,
                    details=f"{mode_label} via {routed_to_vision}, {elapsed:.3f}s",
                )
                score_completed_task(state, task_id)

            return ChatResponse(
                answer=answer,
                turns=1,
                tokens_used=0,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=True,
                cache_stats=None,
                routed_to=routed_to_vision,
                role_history=[routed_to_vision],
                routing_strategy=routing_strategy,
                tokens_generated=0,
                formalization_applied=False,
                tools_used=vision_tools_used,
                tools_called=vision_tools_called,
                prompt_eval_ms=primitives.total_prompt_eval_ms,
                generation_ms=primitives.total_generation_ms,
                predicted_tps=primitives._last_predicted_tps,
                http_overhead_ms=primitives.total_http_overhead_ms,
            )
        except Exception as e:
            import logging
            logging.warning(f"Vision pipeline failed: {type(e).__name__}: {e}")
            # Make image info available to REPL as fallback context
            if request.image_path:
                request.context = (request.context or "") + (
                    f"\n\n[IMAGE: {request.image_path} — Vision pipeline unavailable. "
                    f"Use image analysis tools if available, otherwise note the limitation.]"
                )
            # Fall through to standard orchestration

    # ── Direct-answer mode ──────────────────────────────────────────────
    # For tasks that don't need file access or tool exploration, call the
    # model directly without the REPL Python-code wrapper.  This preserves
    # the model's native instruction-following quality (e.g. 11/11 on
    # instruction_precision vs 2/11 through REPL).
    initial_role = routing_decision[0] if routing_decision else Role.FRONTDOOR

    # Three-way mode selection: direct → react → repl
    # Uses MemRL route_with_mode() when available, heuristic fallback otherwise.
    if request.force_mode and request.force_mode in ("direct", "react", "repl", "delegated"):
        execution_mode = request.force_mode
    else:
        execution_mode = _select_mode(request.prompt, request.context or "", state)

    # ── Architect delegation mode ─────────────────────────────────────
    # Architect formulates investigation briefs, fast specialist runs tool loops.
    is_architect = str(initial_role) in ("architect_general", "architect_coding")
    use_delegation = (
        (is_architect and features().architect_delegation)
        or execution_mode == "delegated"
    )

    if use_delegation and request.real_mode:
        import logging as _deleg_log
        _deleg_log.getLogger(__name__).info(
            f"Delegated mode for {initial_role} (prompt: {len(request.prompt)} chars)"
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
            import logging as _dl
            _dl.getLogger(__name__).warning(f"Delegation failed ({e}), falling back to direct")
            answer = None
            delegation_stats = {}

        if answer:
            # If specialist produced the document (architect said "Approved"),
            # specialist_output is already set as the answer.
            if answer and not answer.startswith("[ERROR"):
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
                    task_id=task_id,
                    success=True,
                    details=f"Delegated mode ({initial_role}), {elapsed:.3f}s, {phases_log}",
                )
                score_completed_task(state, task_id)

            cache_stats = primitives.get_cache_stats() if primitives._backends else None
            return ChatResponse(
                answer=answer,
                turns=1 + loops,
                tokens_used=primitives.total_tokens_generated,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=True,
                cache_stats=cache_stats,
                routed_to=str(initial_role),
                role_history=[str(initial_role)] + [
                    p.get("delegate_to", "")
                    for p in delegation_stats.get("phases", [])
                    if p.get("phase") == "B"
                ],
                routing_strategy="delegated",
                tokens_generated=primitives.total_tokens_generated,
                formalization_applied=formalization_applied,
                tools_used=delegation_stats.get("tools_used", 0),
                tools_called=delegation_stats.get("tools_called", []),
                prompt_eval_ms=primitives.total_prompt_eval_ms,
                generation_ms=primitives.total_generation_ms,
                predicted_tps=primitives._last_predicted_tps,
                http_overhead_ms=primitives.total_http_overhead_ms,
            )

    # ── ReAct tool loop mode ───────────────────────────────────────────
    # For direct-mode prompts that need tool access (search, calculate, date).
    # Uses ReAct Thought/Action/Observation loop with whitelisted read-only tools.
    if execution_mode == "react" and request.real_mode:
        import logging as _react_log
        _react_log.getLogger(__name__).info(
            f"ReAct mode for {initial_role} (prompt: {len(request.prompt)} chars)"
        )

        react_tools_used = 0
        react_tools_called: list[str] = []
        try:
            answer, react_tools_used, react_tools_called = _react_mode_answer(
                prompt=request.prompt,
                context=request.context or "",
                primitives=primitives,
                role=str(initial_role),
                tool_registry=state.tool_registry if hasattr(state, 'tool_registry') else None,
                max_turns=5,
            )
            answer = answer.strip()
        except Exception as e:
            import logging as _rl
            _rl.getLogger(__name__).warning(f"ReAct mode failed ({e}), falling back to direct")
            answer = None

        if answer:
            # Apply post-processing: truncation, quality check, review gate
            answer = _truncate_looped_answer(answer, request.prompt)

            if answer and not answer.startswith("[ERROR") and features().generation_monitor:
                quality_issue = _detect_output_quality_issue(answer)
                if quality_issue:
                    try:
                        escalated = primitives.llm_call(
                            request.prompt,
                            role="coder_escalation",
                            n_tokens=2048,
                            skip_suffix=True,
                        )
                        if escalated.strip():
                            answer = escalated.strip()
                            initial_role = Role.CODER_ESCALATION
                    except Exception:
                        pass

            elapsed = time.perf_counter() - start_time
            state.increment_request(mock_mode=False, turns=1)
            if state.progress_logger:
                state.progress_logger.log_task_completed(
                    task_id=task_id,
                    success=True,
                    details=f"ReAct mode ({initial_role}), {elapsed:.3f}s",
                )
                score_completed_task(state, task_id)

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
                tokens_generated=primitives.total_tokens_generated,
                formalization_applied=formalization_applied,
                tools_used=react_tools_used,
                tools_called=react_tools_called,
                prompt_eval_ms=primitives.total_prompt_eval_ms,
                generation_ms=primitives.total_generation_ms,
                predicted_tps=primitives._last_predicted_tps,
                http_overhead_ms=primitives.total_http_overhead_ms,
            )

    if execution_mode == "direct" and request.real_mode:
        import logging as _log
        _log.getLogger(__name__).info(
            f"Direct-answer mode for {initial_role} (prompt: {len(request.prompt)} chars)"
        )

        # Build a clean prompt — just the user's question + context, no REPL wrapper
        # NO preamble — any system instruction (even soft) degrades model quality.
        # Wave 1 scored 100% thinking with zero preamble.
        # NOTE: No date injection here. Factual context (date, time, etc.) should
        # come from tools (get_current_date in tool_registry.yaml), not prompt
        # hacking. Direct-answer mode needs a ReAct tool loop (Wave 4) to access
        # tools like web_search, calculate, get_current_date.
        direct_prompt = request.prompt
        if request.context:
            direct_prompt = f"{request.context}\n\n{request.prompt}"

        try:
            answer = primitives.llm_call(
                direct_prompt,
                role=str(initial_role),
                n_tokens=2048,
                skip_suffix=True,  # No registry suffix — it forces elaboration
                stop_sequences=["\n\n\n", QWEN_STOP],  # Triple-newline = end of response (anti-loop)
            )
            answer = answer.strip()
        except Exception as e:
            # Retry once on transient LLM backend failures
            import logging as _retry_log
            _retry_log.getLogger(__name__).warning(
                f"Direct LLM call failed ({e}), retrying once..."
            )
            try:
                answer = primitives.llm_call(
                    direct_prompt,
                    role=str(initial_role),
                    n_tokens=4096,
                    skip_suffix=True,
                    stop_sequences=["\n\n\n", QWEN_STOP],
                )
                answer = answer.strip()
            except Exception as e2:
                answer = f"[ERROR: Direct LLM call failed after retry: {e2}]"

        # ── Defense-in-depth: truncate if model loops back to prompt ──
        if answer and not answer.startswith("[ERROR"):
            answer = _truncate_looped_answer(answer, direct_prompt)

        # ── Output formalizer: enforce format constraints ─────────────
        # If the prompt specifies format constraints (word count, JSON, list),
        # reformat the answer to comply. Runs BEFORE quality check so the
        # check sees the finalized output.
        if answer and not answer.startswith("[ERROR"):
            should_fmt, fmt_spec = _should_formalize(request.prompt)
            if should_fmt:
                answer = _formalize_output(answer, request.prompt, fmt_spec, primitives)

        # ── Entropy-inspired output quality check ─────────────────────
        # Post-hoc detection of confused/repetitive output. If the
        # frontdoor model produced garbled output, escalate to a
        # stronger model (coder_primary 32B or architect_general 235B).
        # This is SAFE routing: only triggers on detected failure
        # (unlike keyword routing which caused Wave 2 regressions).
        if answer and not answer.startswith("[ERROR") and features().generation_monitor:
            quality_issue = _detect_output_quality_issue(answer)
            if quality_issue:
                _log.getLogger(__name__).info(
                    f"Output quality issue detected ({quality_issue}), "
                    f"escalating from {initial_role} to coder_escalation"
                )
                try:
                    escalated_answer = primitives.llm_call(
                        direct_prompt,
                        role="coder_escalation",
                        n_tokens=2048,
                        skip_suffix=True,
                    )
                    if escalated_answer.strip():
                        answer = escalated_answer.strip()
                        initial_role = Role.CODER_ESCALATION
                except Exception:
                    pass  # Keep original answer if escalation fails

        # MemRL-informed quality review gate
        if answer and not answer.startswith("[ERROR") and _should_review(
            state, task_id, initial_role, answer
        ):
            verdict = _architect_verdict(
                question=request.prompt,
                answer=answer,
                primitives=primitives,
            )
            if verdict and verdict.upper().startswith("WRONG"):
                corrections = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
                answer = _fast_revise(
                    question=request.prompt,
                    original_answer=answer,
                    corrections=corrections,
                    primitives=primitives,
                )

        elapsed = time.perf_counter() - start_time
        state.increment_request(mock_mode=False, turns=1)
        success = not answer.startswith("[ERROR")

        if state.progress_logger:
            state.progress_logger.log_task_completed(
                task_id=task_id,
                success=success,
                details=f"Direct answer mode ({initial_role}), {elapsed:.3f}s",
            )
            score_completed_task(state, task_id)

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
            routing_strategy=routing_strategy,
            tokens_generated=primitives.total_tokens_generated,
            formalization_applied=formalization_applied,
            tools_used=0,
            prompt_eval_ms=primitives.total_prompt_eval_ms,
            generation_ms=primitives.total_generation_ms,
            predicted_tps=primitives._last_predicted_tps,
            http_overhead_ms=primitives.total_http_overhead_ms,
        )

    # ── REPL orchestration mode ───────────────────────────────────────
    # For tasks needing file exploration, tool access, or large context.
    # Create REPL environment
    combined_context = request.prompt
    if request.context:
        combined_context += f"\n\nContext:\n{request.context}"

    repl = REPLEnvironment(
        context=combined_context,
        llm_primitives=primitives,
        tool_registry=state.tool_registry,
        script_registry=state.script_registry,
        role=initial_role,
        progress_logger=state.progress_logger,
        task_id=task_id,
        # MemRL components for model self-routing
        retriever=state.hybrid_router.retriever if state.hybrid_router else None,
        hybrid_router=state.hybrid_router,
    )

    # Check for two-stage summarization opportunity
    if request.real_mode and _should_use_two_stage(
        prompt=request.prompt,
        context=request.context,
    ):
        try:
            answer, two_stage_stats = await _run_two_stage_summarization(
                prompt=request.prompt,
                context=request.context or "",
                primitives=primitives,
                state=state,
                task_id=task_id,
            )

            elapsed = time.perf_counter() - start_time
            state.increment_request(mock_mode=False, turns=2)  # Count as 2 turns

            # Log task completion
            if state.progress_logger:
                cache_info = "cache hit" if two_stage_stats.get("cache_hit") else "cache miss"
                state.progress_logger.log_task_completed(
                    task_id=task_id,
                    success=True,
                    details=f"Two-stage summarization ({cache_info}), {elapsed:.3f}s",
                )
                score_completed_task(state, task_id)

            cache_stats = primitives.get_cache_stats() if primitives._backends else None

            return ChatResponse(
                answer=answer,
                turns=2,
                tokens_used=primitives.total_tokens_generated,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=True,
                cache_stats=cache_stats,
                routed_to=str(initial_role),
                role_history=[str(initial_role), str(TWO_STAGE_CONFIG["stage2_role"])],
                routing_strategy=routing_strategy,
                tokens_generated=primitives.total_tokens_generated,
                formalization_applied=formalization_applied,
                tools_used=0,
                prompt_eval_ms=primitives.total_prompt_eval_ms,
                generation_ms=primitives.total_generation_ms,
                predicted_tps=primitives._last_predicted_tps,
                http_overhead_ms=primitives.total_http_overhead_ms,
            )
        except Exception as e:
            # Fall back to standard orchestration on two-stage failure
            import logging
            logging.warning(f"Two-stage summarization failed: {type(e).__name__}: {e}")
            # Continue to normal loop below

    # Detect long context → use REPL-based exploration strategy
    # Instead of dumping full context into one model, the frontdoor uses
    # peek/grep/summarize_chunks to explore and synthesize.
    context_chars = len(combined_context)
    use_long_context_exploration = (
        LONG_CONTEXT_CONFIG["enabled"]
        and request.real_mode
        and context_chars > LONG_CONTEXT_CONFIG["threshold_chars"]
    )

    if use_long_context_exploration:
        import logging
        logging.info(
            f"Long context detected ({context_chars:,} chars). "
            f"Using REPL exploration strategy."
        )

    # Run Root LM orchestration loop with escalation support
    turns = 0
    answer = ""
    last_output = ""
    last_error = ""

    # Escalation tracking
    current_role = initial_role
    consecutive_failures = 0
    role_history = [current_role]
    escalation_prompt = ""  # Set when escalating

    # Long context exploration allows more turns
    max_turns = (
        LONG_CONTEXT_CONFIG["max_turns"]
        if use_long_context_exploration
        else request.max_turns
    )

    for turn in range(max_turns):
        turns += 1

        # 1. Get current REPL state
        repl_state = repl.get_state()

        # 2. Build prompt - use escalation prompt if we just escalated
        if escalation_prompt:
            root_prompt = escalation_prompt
            escalation_prompt = ""  # Clear after use
        elif use_long_context_exploration and turn == 0:
            # First turn with long context: use exploration-aware prompt
            root_prompt = build_long_context_exploration_prompt(
                original_prompt=request.prompt,
                context_chars=context_chars,
                state=repl_state,
            )
        else:
            # Inject routing context on turn 0 (MemRL intelligence)
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

        # 3. Call Root LM (current role) to generate Python code
        # Use generation monitoring for early failure detection if feature enabled
        f = features()
        generation_aborted = False
        abort_reason = ""

        try:
            if f.generation_monitor and not request.mock_mode:
                # Create monitor with tier-appropriate config
                monitor_config = MonitorConfig.for_tier(current_role)
                monitor = GenerationMonitor(
                    config=monitor_config,
                    mock_mode=request.mock_mode,
                )
                llm_result = primitives.llm_call_monitored(
                    root_prompt,
                    role=current_role,
                    monitor=monitor,
                )
                code = llm_result.text
                generation_aborted = llm_result.aborted
                abort_reason = llm_result.abort_reason
            else:
                # Standard call without monitoring
                code = primitives.llm_call(
                    root_prompt,
                    role=current_role,
                    n_tokens=1024,
                )
        except Exception as e:
            # If LLM call fails, return error
            answer = f"[ERROR: {current_role} LM call failed: {e}]"
            break

        # Handle early abort from generation monitoring
        if generation_aborted:
            # Treat as early failure detection - escalate immediately
            escalation_ctx = EscalationContext(
                current_role=current_role,
                error_message=f"Generation aborted: {abort_reason}",
                error_category="early_abort",
                failure_count=1,  # Treat as first failure to trigger escalation
                task_id=task_id,
            )
            policy = EscalationPolicy()
            decision = policy.decide(escalation_ctx)

            if decision.should_escalate and decision.target_role:
                # Record failure in graph for the role that failed
                if state.failure_graph:
                    try:
                        state.failure_graph.record_failure(
                            memory_id=task_id,
                            symptoms=["early_abort", abort_reason[:100]],
                            description=f"{role_history[-1]} failed: {abort_reason[:200]}",
                            severity=3,
                        )
                    except Exception:
                        pass  # Failure recording is non-critical
                current_role = str(decision.target_role)
                role_history.append(current_role)
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=escalation_ctx,
                    decision=decision,
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=role_history[-2],
                        to_tier=current_role,
                        reason=f"Early abort: {abort_reason}",
                    )
                continue  # Skip to next turn with escalated role
            else:
                # Can't escalate further - try to use partial output
                pass  # Continue with partial code

        # Extract code from response (handle markdown code blocks)
        code = extract_code_from_response(code)
        # Auto-wrap in FINAL() if code looks like a complete answer
        code = auto_wrap_final(code)

        # 4. Execute code in REPL
        result = repl.execute(code)

        # 4a. Check model-initiated routing (escalation/delegation artifacts)
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
                # No specific target — use standard escalation chain
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
                    decision=EscalationPolicy().decide(EscalationContext(
                        current_role=role_history[-2],
                        error_category="early_abort",
                        error_message=reason,
                        task_id=task_id,
                    )),
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=role_history[-2],
                        to_tier=current_role,
                        reason=f"Model-initiated: {reason}",
                    )
                continue  # Next turn with new role

        # 4b. Log delegation outcomes (MemRL learning)
        if repl.artifacts.get("_delegations"):
            for deleg in repl.artifacts["_delegations"]:
                if state.progress_logger:
                    state.progress_logger.log_exploration(
                        task_id=task_id,
                        query=deleg.get("prompt_preview", ""),
                        strategy_used=f"delegate:{deleg.get('to_role', 'unknown')}",
                        success=deleg.get("success", False),
                    )
            repl.artifacts["_delegations"] = []  # Clear after logging

        # 5. Check for FINAL() completion
        if result.is_final:
            tool_outputs = repl.artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(result, tool_outputs=tool_outputs)
            consecutive_failures = 0  # Success resets failure count

            # MemRL-informed quality review gate (blocking)
            if request.real_mode and _should_review(state, task_id, current_role, answer):
                import logging
                logging.info(f"Review gate triggered for {current_role} (task {task_id})")
                verdict = _architect_verdict(
                    question=request.prompt,
                    answer=answer,
                    primitives=primitives,
                )
                if verdict and verdict.upper().startswith("WRONG"):
                    corrections = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
                    logging.info(f"Review verdict: WRONG — revising ({corrections[:80]})")
                    answer = _fast_revise(
                        question=request.prompt,
                        original_answer=answer,
                        corrections=corrections,
                        primitives=primitives,
                    )
                else:
                    logging.info("Review verdict: OK")

            break

        # 6. Handle errors with EscalationPolicy for escalation decisions
        if result.error:
            consecutive_failures += 1
            last_error = result.error
            last_output = result.output

            # Consult EscalationPolicy for escalation decision (unified module)
            error_category = classify_error(result.error)
            escalation_ctx = EscalationContext(
                current_role=current_role,
                error_message=result.error,
                error_category=error_category.value,  # Pass as string for compatibility
                failure_count=consecutive_failures,
                task_id=task_id,
            )
            policy = EscalationPolicy()
            decision = policy.decide(escalation_ctx)

            # Log escalation decision (for escalate actions)
            if decision.should_escalate and state.progress_logger:
                state.progress_logger.log_escalation(
                    task_id=task_id,
                    from_tier=current_role,
                    to_tier=str(decision.target_role) if decision.target_role else current_role,
                    reason=f"{decision.reason} (failures: {consecutive_failures})",
                )

            # Act on routing decision
            if decision.should_escalate and decision.target_role:
                # Record failure in graph for the role that triggered escalation
                if state.failure_graph:
                    try:
                        state.failure_graph.record_failure(
                            memory_id=task_id,
                            symptoms=[error_category.value, last_error[:100]],
                            description=f"{current_role} failed: {last_error[:200]}",
                            severity=min(consecutive_failures + 2, 5),
                        )
                    except Exception:
                        pass  # Failure recording is non-critical
                # Switch to higher-tier role
                current_role = str(decision.target_role)
                role_history.append(current_role)
                consecutive_failures = 0  # Reset for new role
                # Build escalation prompt with failure context
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=escalation_ctx,
                    decision=decision,
                )
            elif decision.action == EscalationAction.EXPLORE:
                # Terminal role — fall back to REPL exploration
                # Switch to exploration prompt so the model uses
                # peek/grep/summarize_chunks instead of raw LLM calls
                consecutive_failures = 0
                escalation_prompt = build_long_context_exploration_prompt(
                    original_prompt=request.prompt,
                    context_chars=len(repl.context),
                    state=repl_state,
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=current_role,
                        to_tier=f"{current_role}+explore",
                        reason="Terminal role: switching to REPL exploration",
                    )
            elif decision.action == EscalationAction.FAIL:
                # Max retries/escalations reached - stop with error
                answer = f"[FAILED: {decision.reason}]"
                break
        else:
            # Success - reset failure count but keep role
            consecutive_failures = 0
            last_error = ""
            last_output = result.output

    # If max turns reached without FINAL()
    if not answer:
        # Try to extract substantive content from last_output, stripping tool outputs
        tool_outputs = repl.artifacts.get("_tool_outputs", [])
        cleaned_output = _strip_tool_outputs(last_output, tool_outputs) if last_output else ""

        if cleaned_output and len(cleaned_output) > 20:
            answer = f"[Max turns ({max_turns}) reached without FINAL()]\n\n{cleaned_output}"
        elif last_output:
            answer = f"[Max turns ({max_turns}) reached without FINAL()]\n\nLast output:\n{last_output}"
        else:
            answer = f"[Max turns ({max_turns}) reached without FINAL()]"

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=False, turns=turns)

    # Get cache stats if using real_mode with RadixAttention
    cache_stats = None
    if request.real_mode and primitives._backends:
        cache_stats = primitives.get_cache_stats()

    # Log exploration telemetry (tool usage, function counts)
    success = not answer.startswith("[ERROR") and not answer.startswith("[Max turns")
    repl.log_exploration_completed(success=success, result=answer)

    # Log task completion (MemRL integration)
    if state.progress_logger:
        # Include role history if escalation occurred
        role_info = f", roles: {' -> '.join(role_history)}" if len(role_history) > 1 else ""
        state.progress_logger.log_task_completed(
            task_id=task_id,
            success=success,
            details=f"Real inference: {turns} turns, {elapsed:.3f}s{role_info}",
        )
        score_completed_task(state, task_id)

    return ChatResponse(
        answer=answer,
        turns=turns,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=request.real_mode,
        cache_stats=cache_stats,
        routed_to=str(current_role),
        role_history=[str(r) for r in role_history],
        routing_strategy=routing_strategy,
        tokens_generated=primitives.total_tokens_generated,
        formalization_applied=formalization_applied,
        tools_used=repl._tool_invocations,
        tools_called=(
            [inv.tool_name for inv in repl.tool_registry.get_invocation_log()]
            if repl.tool_registry else []
        ),
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
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
    state = get_state()

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
                    state.progress_logger.log_task_completed(task_id, success=True, details="Plan mode")
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
                state.progress_logger.log_task_completed(task_id, success=True, details="Mock stream")
                score_completed_task(state, task_id)
            yield done_event()
            return

        # Real mode - initialize MemRL components on first real use (lazy loading)
        ensure_memrl_initialized(state)

        server_urls = request.server_urls or LLMPrimitives.DEFAULT_SERVER_URLS
        try:
            primitives = LLMPrimitives(mock_mode=False, server_urls=server_urls, registry=state.registry)
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
                    state.progress_logger.log_task_completed(task_id, success=False, details=f"Root LM failed: {e}")
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
                        decision=EscalationPolicy().decide(EscalationContext(
                            current_role=role_history[-2],
                            error_category="early_abort",
                            error_message=reason,
                            task_id=task_id,
                        )),
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
                        corrections = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
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
            role_info = f", roles: {' -> '.join(str(r) for r in role_history)}" if len(role_history) > 1 else ""
            state.progress_logger.log_task_completed(
                task_id, success=success,
                details=f"Stream complete{role_info}",
            )
            score_completed_task(state, task_id)
        yield done_event()

    # Use SSE utilities for response (handles sse-starlette vs manual fallback)
    return create_sse_response(generate())

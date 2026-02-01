"""Pipeline stage functions for _handle_chat() (Phase 1b extraction).

Each function corresponds to a named stage in the chat processing pipeline.
Extracted from the 1,091-line _handle_chat() to enable independent testing,
explicit failure signaling, and role-specific timeouts.

Stage functions:
    _route_request()     — Routing decision, task_id, MemRL logging
    _preprocess()        — Input formalization
    _init_primitives()   — LLMPrimitives setup + backend validation
    _execute_mock()      — Mock mode response
    _execute_vision()    — Vision pipeline (OCR, VL, multi-file)
    _execute_delegated() — Architect delegation mode
    _execute_react()     — ReAct tool loop mode
    _execute_direct()    — Direct LLM call mode
    _execute_repl()      — REPL orchestration mode
    _annotate_error()    — Detect error answers → set error_code/error_detail
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import HTTPException

from src.api.models import ChatRequest, ChatResponse
from src.api.services.memrl import ensure_memrl_initialized, score_completed_task
from src.escalation import EscalationAction, EscalationContext, EscalationPolicy
from src.features import features
from src.generation_monitor import GenerationMonitor, MonitorConfig
from src.llm_primitives import LLMPrimitives
from src.prompt_builders import (
    auto_wrap_final,
    build_escalation_prompt,
    build_long_context_exploration_prompt,
    build_root_lm_prompt,
    build_routing_context,
    classify_error,
    extract_code_from_response,
)
from src.repl_environment import REPLEnvironment
from src.roles import Role

from src.api.routes.chat_delegation import _architect_delegated_answer
from src.api.routes.chat_react import _react_mode_answer
from src.api.routes.chat_review import (
    _apply_plan_review,
    _architect_plan_review,
    _architect_verdict,
    _detect_output_quality_issue,
    _fast_revise,
    _needs_plan_review,
    _should_review,
    _store_plan_review_episode,
)
from src.api.routes.chat_routing import _classify_and_route, _select_mode
from src.api.routes.chat_summarization import (
    _run_two_stage_summarization,
    _should_use_two_stage,
)
from src.api.routes.chat_utils import (
    LONG_CONTEXT_CONFIG,
    QWEN_STOP,
    TWO_STAGE_CONFIG,
    RoutingResult,
    _formalize_output,
    _resolve_answer,
    _should_formalize,
    _strip_tool_outputs,
    _truncate_looped_answer,
)
from src.api.routes.chat_vision import (  # noqa: F401 — kept for backward compat
    _handle_multi_file_vision,
    _handle_vision_request,
    _vision_react_mode_answer,
)

if TYPE_CHECKING:
    pass

from src.api.structured_logging import task_extra

log = logging.getLogger(__name__)


# ── Stage 1: Routing ────────────────────────────────────────────────────


def _route_request(request: ChatRequest, state) -> RoutingResult:
    """Determine routing decision, strategy, and task metadata.

    Produces a RoutingResult that captures all routing decisions made
    before execution begins. Includes failure graph veto and MemRL logging.
    """
    import uuid

    task_id = f"chat-{uuid.uuid4().hex[:8]}"
    task_ir = {
        "task_type": "chat",
        "objective": request.prompt[:200],
        "priority": "interactive",
    }

    use_mock = request.mock_mode and not request.real_mode

    # Initialize MemRL early for real_mode to enable HybridRouter
    if request.real_mode and not use_mock:
        ensure_memrl_initialized(state)

    # Determine routing using HybridRouter if available, otherwise rules
    if use_mock:
        routing_decision = [request.role or Role.FRONTDOOR]
        routing_strategy = "mock"
    elif request.force_role:
        routing_decision = [request.force_role]
        routing_strategy = "forced"
    elif request.role and request.role not in ("", "frontdoor"):
        routing_decision = [request.role]
        routing_strategy = "explicit"
    elif state.hybrid_router and request.real_mode:
        routing_decision, routing_strategy = state.hybrid_router.route(task_ir)
    else:
        has_image = bool(request.image_path or request.image_base64)
        classified_role, routing_strategy = _classify_and_route(
            request.prompt, request.context or "", has_image=has_image,
        )
        routing_decision = [classified_role]

    # Failure graph veto — revert high-risk specialists to frontdoor
    if (
        state.failure_graph
        and routing_decision
        and str(routing_decision[0]) != str(Role.FRONTDOOR)
        and routing_strategy not in ("mock", "forced")
    ):
        try:
            risk = state.failure_graph.get_failure_risk(str(routing_decision[0]))
            if risk > 0.5:
                log.warning(
                    "Failure veto: %s risk=%.2f > 0.5, reverting to frontdoor",
                    routing_decision[0], risk,
                    extra=task_extra(task_id=task_id, role=str(routing_decision[0]),
                                     stage="routing", strategy="failure_vetoed"),
                )
                routing_decision = [str(Role.FRONTDOOR)]
                routing_strategy = "failure_vetoed"
        except Exception:
            pass

    # Log task start (MemRL integration)
    if state.progress_logger:
        state.progress_logger.log_task_started(
            task_id=task_id,
            task_ir=task_ir,
            routing_decision=routing_decision,
            routing_strategy=routing_strategy,
        )

    # Compute role-specific timeout
    role_str = str(routing_decision[0]) if routing_decision else str(Role.FRONTDOOR)
    from src.api.routes.chat_utils import ROLE_TIMEOUTS, DEFAULT_TIMEOUT_S
    timeout_s = ROLE_TIMEOUTS.get(role_str, DEFAULT_TIMEOUT_S)

    return RoutingResult(
        task_id=task_id,
        task_ir=task_ir,
        use_mock=use_mock,
        routing_decision=routing_decision,
        routing_strategy=routing_strategy,
        timeout_s=timeout_s,
    )


# ── Stage 2: Preprocessing ──────────────────────────────────────────────


def _preprocess(request: ChatRequest, state, routing: RoutingResult) -> None:
    """Apply input formalization if enabled. Mutates request.context and routing."""
    if (
        features().input_formalizer
        and request.real_mode
        and not routing.use_mock
        and routing.routing_strategy not in ("mock",)
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
                routing.formalization_applied = True
                log.info(
                    "Input formalization: %s (%.1fs, %s)",
                    problem_hint,
                    fml_result.elapsed_seconds,
                    fml_result.model_role,
                    extra=task_extra(task_id=routing.task_id, stage="preprocess",
                                     latency_ms=fml_result.elapsed_seconds * 1000),
                )


# ── Stage 3: Backend initialization ─────────────────────────────────────


def _init_primitives(request: ChatRequest, state) -> LLMPrimitives:
    """Initialize LLM backends for real inference.

    Reuses shared LLMPrimitives instance for connection pooling when possible.
    Raises HTTPException(503) if backends unavailable.
    """
    if request.real_mode:
        server_urls = request.server_urls or LLMPrimitives.DEFAULT_SERVER_URLS

        if (
            hasattr(state, '_real_primitives')
            and state._real_primitives is not None
            and not request.server_urls
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

        primitives.cache_prompt = request.cache_prompt

        if not primitives._backends:
            raise HTTPException(
                status_code=503,
                detail="No backends available. Ensure llama-server is running on configured ports.",
            )
    else:
        primitives = LLMPrimitives(mock_mode=False, registry=state.registry)
        if primitives.model_server is None:
            raise HTTPException(
                status_code=503,
                detail="Real inference not available: no model server configured",
            )

    return primitives


# ── Stage 4: Mock mode ──────────────────────────────────────────────────


def _execute_mock(
    request: ChatRequest, routing: RoutingResult, state, start_time: float,
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


# ── Stage 5: Plan review gate ───────────────────────────────────────────


def _plan_review_gate(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
) -> list | None:
    """Run architect plan review if applicable. Returns modified routing_decision or None."""
    plan_review_result = None
    if (
        request.real_mode
        and features().plan_review
        and _needs_plan_review(routing.task_ir, routing.routing_decision, state)
    ):
        plan_review_result = _architect_plan_review(
            routing.task_ir, routing.routing_decision, primitives, state, routing.task_id
        )
        if plan_review_result and plan_review_result.decision != "ok":
            routing.routing_decision = _apply_plan_review(
                routing.routing_decision, plan_review_result
            )
        if plan_review_result:
            _store_plan_review_episode(state, routing.task_id, routing.task_ir, plan_review_result)
    return plan_review_result


# ── Stage 6: Vision pipeline ────────────────────────────────────────────


async def _execute_vision(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
) -> ChatResponse | None:
    """Preprocess vision inputs through the document pipeline.

    Instead of answering directly, this stage runs DocumentPreprocessor to
    extract text, sections, and figures from the image/document. The result
    is stored on routing.document_result so that _execute_repl() can use
    DocumentREPLEnvironment with structured document access tools.

    Returns None to fall through to REPL mode (the normal path).
    Only returns ChatResponse on unrecoverable errors.
    """
    has_vision_input = request.image_path or request.image_base64 or request.files
    if not (request.real_mode and has_vision_input):
        return None

    from src.services.document_preprocessor import (
        DocumentPreprocessor,
        PreprocessingConfig,
    )

    config = PreprocessingConfig(
        extract_figures=True,
        describe_figures=True,
    )
    preprocessor = DocumentPreprocessor(config=config)

    try:
        if request.image_path:
            log.info(
                "Vision preprocessing file: %s", request.image_path,
                extra=task_extra(task_id=routing.task_id, stage="execute", mode="vision_preprocess"),
            )
            result = await preprocessor.preprocess_file(request.image_path)
        elif request.image_base64:
            log.info(
                "Vision preprocessing base64 image",
                extra=task_extra(task_id=routing.task_id, stage="execute", mode="vision_preprocess"),
            )
            task_ir = {"inputs": [{"type": "base64", "value": request.image_base64}]}
            result = await preprocessor.preprocess(task_ir)
        elif request.files:
            log.info(
                "Vision preprocessing %d files", len(request.files),
                extra=task_extra(task_id=routing.task_id, stage="execute", mode="vision_preprocess"),
            )
            task_ir = {"inputs": [{"type": "path", "value": f} for f in request.files]}
            result = await preprocessor.preprocess(task_ir)
        else:
            result = None

        if result and result.success and result.document_result:
            routing.document_result = result
            log.info(
                "Document preprocessing succeeded: %d sections, %d figures",
                len(result.document_result.sections),
                len(result.document_result.figures),
                extra=task_extra(task_id=routing.task_id, stage="execute",
                                 mode="vision_preprocess"),
            )
            return None  # Fall through to REPL with document context

        # Preprocessing returned but without usable document result
        warn_msg = result.error if result else "unknown"
        log.warning(
            "Document preprocessing failed: %s", warn_msg,
            extra=task_extra(task_id=routing.task_id, stage="execute",
                             mode="vision_preprocess", error_type="preprocess_failed"),
        )

    except Exception as e:
        log.warning(
            "Vision preprocessing exception: %s: %s", type(e).__name__, e,
            extra=task_extra(task_id=routing.task_id, stage="execute",
                             mode="vision_preprocess", error_type=type(e).__name__),
        )

    # Preprocessing failed — inject context note and fall through to text modes
    image_ref = request.image_path or "(base64 image)"
    request.context = (request.context or "") + (
        f"\n\n[IMAGE: {image_ref} — Document pipeline failed. "
        f"Answering without OCR context.]"
    )
    return None  # Fall through to standard orchestration


# ── Stage 7: Delegated mode ─────────────────────────────────────────────


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
    use_delegation = (
        (is_architect and features().architect_delegation)
        or execution_mode == "delegated"
    )

    if not (use_delegation and request.real_mode):
        return None

    log.info(
        "Delegated mode for %s (prompt: %d chars)", initial_role, len(request.prompt),
        extra=task_extra(task_id=routing.task_id, role=str(initial_role),
                         stage="execute", mode="delegated", prompt_len=len(request.prompt)),
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
            "Delegation failed (%s), falling back to direct", e,
            extra=task_extra(task_id=routing.task_id, role=str(initial_role),
                             stage="execute", mode="delegated", error_type=type(e).__name__),
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
        )
        score_completed_task(state, routing.task_id)

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
        mode="delegated",
        tokens_generated=primitives.total_tokens_generated,
        formalization_applied=routing.formalization_applied,
        tools_used=delegation_stats.get("tools_used", 0),
        tools_called=delegation_stats.get("tools_called", []),
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
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
        "ReAct mode for %s (prompt: %d chars)", initial_role, len(request.prompt),
        extra=task_extra(task_id=routing.task_id, role=str(initial_role),
                         stage="execute", mode="react", prompt_len=len(request.prompt)),
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
        log.warning(
            "ReAct mode failed (%s), falling back to direct", e,
            extra=task_extra(task_id=routing.task_id, role=str(initial_role),
                             stage="execute", mode="react", error_type=type(e).__name__),
        )
        return None

    if not answer:
        return None

    # Post-processing: truncation, quality check
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
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
    )


# ── Stage 9: Direct mode ────────────────────────────────────────────────


def _execute_direct(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
    initial_role,
) -> ChatResponse:
    """Handle direct LLM call mode (no REPL wrapper)."""
    log.info(
        "Direct-answer mode for %s (prompt: %d chars)", initial_role, len(request.prompt),
        extra=task_extra(task_id=routing.task_id, role=str(initial_role),
                         stage="execute", mode="direct", prompt_len=len(request.prompt)),
    )

    direct_prompt = request.prompt
    if request.context:
        direct_prompt = f"{request.context}\n\n{request.prompt}"

    try:
        answer = primitives.llm_call(
            direct_prompt,
            role=str(initial_role),
            n_tokens=2048,
            skip_suffix=True,
            stop_sequences=["\n\n\n", QWEN_STOP],
        )
        answer = answer.strip()
    except Exception as e:
        log.warning(
            "Direct LLM call failed (%s), retrying once...", e,
            extra=task_extra(task_id=routing.task_id, role=str(initial_role),
                             stage="execute", mode="direct", error_type=type(e).__name__),
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

    # Defense-in-depth: truncate if model loops back to prompt
    if answer and not answer.startswith("[ERROR"):
        answer = _truncate_looped_answer(answer, direct_prompt)

    # Output formalizer: enforce format constraints
    if answer and not answer.startswith("[ERROR"):
        should_fmt, fmt_spec = _should_formalize(request.prompt)
        if should_fmt:
            answer = _formalize_output(answer, request.prompt, fmt_spec, primitives)

    # Entropy-inspired output quality check
    if answer and not answer.startswith("[ERROR") and features().generation_monitor:
        quality_issue = _detect_output_quality_issue(answer)
        if quality_issue:
            log.info(
                "Output quality issue detected (%s), escalating from %s to coder_escalation",
                quality_issue, initial_role,
                extra=task_extra(task_id=routing.task_id, role=str(initial_role),
                                 stage="quality_check", error_type=quality_issue),
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
                pass

    # MemRL-informed quality review gate
    if answer and not answer.startswith("[ERROR") and _should_review(
        state, routing.task_id, initial_role, answer
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
            task_id=routing.task_id,
            success=success,
            details=f"Direct answer mode ({initial_role}), {elapsed:.3f}s",
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
        routing_strategy=routing.routing_strategy,
        mode="direct",
        tokens_generated=primitives.total_tokens_generated,
        formalization_applied=routing.formalization_applied,
        tools_used=0,
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
    )


# ── Stage 10: REPL orchestration mode ───────────────────────────────────


async def _execute_repl(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
    initial_role,
) -> ChatResponse:
    """Handle REPL orchestration mode (file exploration, tool access, large context)."""
    task_id = routing.task_id
    routing_strategy = routing.routing_strategy
    formalization_applied = routing.formalization_applied

    # Create REPL environment — use DocumentREPLEnvironment when document
    # preprocessing produced structured results (sections, figures, etc.)
    combined_context = request.prompt
    if request.context:
        combined_context += f"\n\nContext:\n{request.context}"

    if routing.document_result and routing.document_result.document_result:
        from src.repl_document import DocumentREPLEnvironment, DocumentContext

        doc_result = routing.document_result.document_result
        doc_context = DocumentContext.from_document_result(doc_result)

        # Use searchable text as REPL context, prepend the user's question
        doc_text = doc_result.to_searchable_text()
        combined_context = f"Question: {request.prompt}\n\n{doc_text}"

        repl = DocumentREPLEnvironment(
            context=combined_context,
            document_context=doc_context,
            llm_primitives=primitives,
            tool_registry=state.tool_registry,
            script_registry=state.script_registry,
            role=initial_role,
            progress_logger=state.progress_logger,
            task_id=task_id,
            retriever=state.hybrid_router.retriever if state.hybrid_router else None,
            hybrid_router=state.hybrid_router,
        )
        log.info(
            "Using DocumentREPLEnvironment: %d sections, %d figures",
            len(doc_context.sections), len(doc_context.figures),
            extra=task_extra(task_id=task_id, stage="execute", mode="repl_document"),
        )
    else:
        repl = REPLEnvironment(
            context=combined_context,
            llm_primitives=primitives,
            tool_registry=state.tool_registry,
            script_registry=state.script_registry,
            role=initial_role,
            progress_logger=state.progress_logger,
            task_id=task_id,
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
            state.increment_request(mock_mode=False, turns=2)

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
                mode="repl",
                tokens_generated=primitives.total_tokens_generated,
                formalization_applied=formalization_applied,
                tools_used=0,
                prompt_eval_ms=primitives.total_prompt_eval_ms,
                generation_ms=primitives.total_generation_ms,
                predicted_tps=primitives._last_predicted_tps,
                http_overhead_ms=primitives.total_http_overhead_ms,
            )
        except Exception as e:
            logging.warning(f"Two-stage summarization failed: {type(e).__name__}: {e}")

    # Detect long context → use REPL-based exploration strategy
    context_chars = len(combined_context)
    use_long_context_exploration = (
        LONG_CONTEXT_CONFIG["enabled"]
        and request.real_mode
        and context_chars > LONG_CONTEXT_CONFIG["threshold_chars"]
    )

    if use_long_context_exploration:
        logging.info(
            f"Long context detected ({context_chars:,} chars). "
            f"Using REPL exploration strategy."
        )

    # Run Root LM orchestration loop with escalation support
    turns = 0
    answer = ""
    last_output = ""
    last_error = ""

    current_role = initial_role
    consecutive_failures = 0
    role_history = [current_role]
    escalation_prompt = ""

    max_turns = (
        LONG_CONTEXT_CONFIG["max_turns"]
        if use_long_context_exploration
        else request.max_turns
    )

    for turn in range(max_turns):
        turns += 1

        # 1. Get current REPL state
        repl_state = repl.get_state()

        # 2. Build prompt
        if escalation_prompt:
            root_prompt = escalation_prompt
            escalation_prompt = ""
        elif use_long_context_exploration and turn == 0:
            root_prompt = build_long_context_exploration_prompt(
                original_prompt=request.prompt,
                context_chars=context_chars,
                state=repl_state,
            )
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

        # 3. Call Root LM with generation monitoring
        f = features()
        generation_aborted = False
        abort_reason = ""

        try:
            if f.generation_monitor and not request.mock_mode:
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
                code = primitives.llm_call(
                    root_prompt,
                    role=current_role,
                    n_tokens=1024,
                )
        except Exception as e:
            answer = f"[ERROR: {current_role} LM call failed: {e}]"
            break

        # Handle early abort from generation monitoring
        if generation_aborted:
            escalation_ctx = EscalationContext(
                current_role=current_role,
                error_message=f"Generation aborted: {abort_reason}",
                error_category="early_abort",
                failure_count=1,
                task_id=task_id,
            )
            policy = EscalationPolicy()
            decision = policy.decide(escalation_ctx)

            if decision.should_escalate and decision.target_role:
                if state.failure_graph:
                    try:
                        state.failure_graph.record_failure(
                            memory_id=task_id,
                            symptoms=["early_abort", abort_reason[:100]],
                            description=f"{role_history[-1]} failed: {abort_reason[:200]}",
                            severity=3,
                        )
                    except Exception:
                        pass
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
                continue
            else:
                pass  # Continue with partial code

        # Extract code from response
        code = extract_code_from_response(code)
        code = auto_wrap_final(code)

        # 4. Execute code in REPL
        result = repl.execute(code)

        # 4a. Check model-initiated routing
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
                continue

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
            repl.artifacts["_delegations"] = []

        # 5. Check for FINAL() completion
        if result.is_final:
            tool_outputs = repl.artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(result, tool_outputs=tool_outputs)
            consecutive_failures = 0

            # MemRL-informed quality review gate
            if request.real_mode and _should_review(state, task_id, current_role, answer):
                log.info(
                    "Review gate triggered for %s (task %s)", current_role, task_id,
                    extra=task_extra(task_id=task_id, role=current_role, stage="review"),
                )
                verdict = _architect_verdict(
                    question=request.prompt,
                    answer=answer,
                    primitives=primitives,
                )
                if verdict and verdict.upper().startswith("WRONG"):
                    corrections = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
                    log.info(
                        "Review verdict: WRONG — revising (%.80s)", corrections,
                        extra=task_extra(task_id=task_id, role=current_role, stage="review"),
                    )
                    answer = _fast_revise(
                        question=request.prompt,
                        original_answer=answer,
                        corrections=corrections,
                        primitives=primitives,
                    )
                else:
                    log.info(
                        "Review verdict: OK",
                        extra=task_extra(task_id=task_id, role=current_role, stage="review"),
                    )

            break

        # 6. Handle errors with EscalationPolicy
        if result.error:
            consecutive_failures += 1
            last_error = result.error
            last_output = result.output

            error_category = classify_error(result.error)
            escalation_ctx = EscalationContext(
                current_role=current_role,
                error_message=result.error,
                error_category=error_category.value,
                failure_count=consecutive_failures,
                task_id=task_id,
            )
            policy = EscalationPolicy()
            decision = policy.decide(escalation_ctx)

            if decision.should_escalate and state.progress_logger:
                state.progress_logger.log_escalation(
                    task_id=task_id,
                    from_tier=current_role,
                    to_tier=str(decision.target_role) if decision.target_role else current_role,
                    reason=f"{decision.reason} (failures: {consecutive_failures})",
                )

            if decision.should_escalate and decision.target_role:
                if state.failure_graph:
                    try:
                        state.failure_graph.record_failure(
                            memory_id=task_id,
                            symptoms=[error_category.value, last_error[:100]],
                            description=f"{current_role} failed: {last_error[:200]}",
                            severity=min(consecutive_failures + 2, 5),
                        )
                    except Exception:
                        pass
                current_role = str(decision.target_role)
                role_history.append(current_role)
                consecutive_failures = 0
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=escalation_ctx,
                    decision=decision,
                )
            elif decision.action == EscalationAction.EXPLORE:
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
                answer = f"[FAILED: {decision.reason}]"
                break
        else:
            consecutive_failures = 0
            last_error = ""
            last_output = result.output

    # If max turns reached without FINAL()
    if not answer:
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

    cache_stats = None
    if request.real_mode and primitives._backends:
        cache_stats = primitives.get_cache_stats()

    success = not answer.startswith("[ERROR") and not answer.startswith("[Max turns")
    repl.log_exploration_completed(success=success, result=answer)

    if state.progress_logger:
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
        mode="repl",
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
    if answer.startswith("[ERROR:") and ("timed out" in answer.lower() or "timeout" in answer.lower()):
        response.error_code = 504
        response.error_detail = answer
    elif answer.startswith("[ERROR:") and ("backend" in answer.lower() or "failed" in answer.lower()):
        response.error_code = 502
        response.error_detail = answer
    elif answer.startswith("[ERROR:"):
        response.error_code = 500
        response.error_detail = answer
    elif answer.startswith("[FAILED:"):
        response.error_code = 500
        response.error_detail = answer

    return response

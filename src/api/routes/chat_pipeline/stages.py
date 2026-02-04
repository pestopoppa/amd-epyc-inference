"""Pipeline stages 4, 6-9 + error annotation.

Contains mock mode, vision preprocessing, delegated mode, proactive
parallel delegation, ReAct tool loop, direct LLM call, and error annotation.
"""

from __future__ import annotations

import json as _json
import logging
import re as _re
import time

from src.api.models import ChatRequest, ChatResponse
from src.api.services.memrl import score_completed_task
from src.features import features
from src.llm_primitives import LLMPrimitives
from src.roles import Role

from src.api.routes.chat_delegation import _architect_delegated_answer
from src.api.routes.chat_react import _react_mode_answer
from src.api.routes.chat_review import (
    _architect_verdict,
    _detect_output_quality_issue,
    _fast_revise,
    _should_review,
)
from src.api.routes.chat_utils import (
    QWEN_STOP,
    RoutingResult,
    _formalize_output,
    _should_formalize,
    _truncate_looped_answer,
)
from src.api.structured_logging import task_extra

log = logging.getLogger(__name__)


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
                "Vision preprocessing file: %s",
                request.image_path,
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
            )
            result = await preprocessor.preprocess_file(request.image_path)
        elif request.image_base64:
            log.info(
                "Vision preprocessing base64 image",
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
            )
            task_ir = {"inputs": [{"type": "base64", "value": request.image_base64}]}
            result = await preprocessor.preprocess(task_ir)
        elif request.files:
            log.info(
                "Vision preprocessing %d files",
                len(request.files),
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
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
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
            )
            return None  # Fall through to REPL with document context

        # Preprocessing returned but without usable document result
        warn_msg = result.error if result else "unknown"
        log.warning(
            "Document preprocessing failed: %s",
            warn_msg,
            extra=task_extra(
                task_id=routing.task_id,
                stage="execute",
                mode="vision_preprocess",
                error_type="preprocess_failed",
            ),
        )

    except Exception as e:
        log.warning(
            "Vision preprocessing exception: %s: %s",
            type(e).__name__,
            e,
            extra=task_extra(
                task_id=routing.task_id,
                stage="execute",
                mode="vision_preprocess",
                error_type=type(e).__name__,
            ),
        )

    # Preprocessing failed — inject context note and fall through to text modes
    image_ref = request.image_path or "(base64 image)"
    request.context = (request.context or "") + (
        f"\n\n[IMAGE: {image_ref} — Document pipeline failed. Answering without OCR context.]"
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
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
    )


# ── Stage 7.5: Proactive parallel delegation ──────────────────────────


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
        plan_json_str = primitives.llm_call(
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
    task_ir = {
        "task_id": routing.task_id,
        "task_type": routing.task_ir.get("task_type", "chat"),
        "objective": request.prompt[:200],
        "plan": {"steps": steps},
    }

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
        )
        score_completed_task(state, routing.task_id)

    cache_stats = primitives.get_cache_stats() if primitives._backends else None
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
    try:
        answer, react_tools_used, react_tools_called = _react_mode_answer(
            prompt=request.prompt,
            context=request.context or "",
            primitives=primitives,
            role=str(initial_role),
            tool_registry=state.tool_registry if hasattr(state, "tool_registry") else None,
            max_turns=5,
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
            except Exception as exc:
                log.debug("ReAct coder escalation failed: %s", exc)

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
        "Direct-answer mode for %s (prompt: %d chars)",
        initial_role,
        len(request.prompt),
        extra=task_extra(
            task_id=routing.task_id,
            role=str(initial_role),
            stage="execute",
            mode="direct",
            prompt_len=len(request.prompt),
        ),
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
            "Direct LLM call failed (%s), retrying once...",
            e,
            extra=task_extra(
                task_id=routing.task_id,
                role=str(initial_role),
                stage="execute",
                mode="direct",
                error_type=type(e).__name__,
            ),
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
                quality_issue,
                initial_role,
                extra=task_extra(
                    task_id=routing.task_id,
                    role=str(initial_role),
                    stage="quality_check",
                    error_type=quality_issue,
                ),
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
            except Exception as exc:
                log.debug("Direct mode coder escalation failed: %s", exc)

    # MemRL-informed quality review gate
    if (
        answer
        and not answer.startswith("[ERROR")
        and _should_review(state, routing.task_id, initial_role, answer)
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

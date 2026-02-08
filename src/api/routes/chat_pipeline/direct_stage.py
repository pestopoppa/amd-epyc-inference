"""Pipeline stage 9: Direct LLM call mode.

Handles direct LLM calls without REPL wrapper. Includes output
formalizer, quality escalation, and MemRL-informed review gate.
"""

from __future__ import annotations

import logging
import time

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat_review import (
    _architect_verdict,
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
from src.api.services.memrl import score_completed_task
from src.api.structured_logging import task_extra
from src.llm_primitives import LLMPrimitives

from src.api.routes.chat_pipeline.stages import _quality_escalate

log = logging.getLogger(__name__)


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
    answer, initial_role = _quality_escalate(
        answer,
        direct_prompt,
        primitives,
        initial_role,
        allow_escalation=not bool(request.force_role),
    )

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

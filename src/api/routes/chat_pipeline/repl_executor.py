"""Pipeline stage 10: REPL orchestration mode.

The largest pipeline stage — manages the multi-turn REPL loop with
escalation support, generation monitoring, two-stage summarization,
long-context exploration, and quality review gates.
"""

from __future__ import annotations

import asyncio
import logging
import time

from src.api.models import ChatRequest, ChatResponse
from src.api.services.memrl import score_completed_task
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

from src.api.routes.chat_review import (
    _architect_verdict,
    _fast_revise,
    _should_review,
)
from src.api.routes.chat_summarization import (
    _run_two_stage_summarization,
    _should_use_two_stage,
)
from src.api.routes.chat_utils import (
    LONG_CONTEXT_CONFIG,
    TWO_STAGE_CONFIG,
    RoutingResult,
    _resolve_answer,
    _strip_tool_outputs,
)
from src.api.structured_logging import task_extra

log = logging.getLogger(__name__)


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
                    except Exception as exc:
                        log.debug("failure_graph.record_failure (abort) failed: %s", exc)
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

        # 4. Execute code in REPL (in thread to avoid blocking event loop —
        #    repl.execute() can call blocking I/O like requests.get in _web_fetch).
        #    signal.alarm() timeouts silently fail in non-main threads, so we use
        #    asyncio.wait_for as the authoritative timeout mechanism.
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(repl.execute, code),
                timeout=repl.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            from src.repl_environment.types import ExecutionResult
            result = ExecutionResult(
                output="",
                is_final=False,
                error=f"REPL execution timed out after {repl.config.timeout_seconds}s",
            )

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
                    except Exception as exc:
                        log.debug("failure_graph.record_failure (error) failed: %s", exc)
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

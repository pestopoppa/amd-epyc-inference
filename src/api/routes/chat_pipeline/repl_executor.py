"""Pipeline stage 10: REPL orchestration mode.

The largest pipeline stage — manages the multi-turn REPL loop with
escalation support, generation monitoring, two-stage summarization,
long-context exploration, and quality review gates.

The inner escalation loop is now driven by the pydantic-graph orchestration
graph (src.graph), replacing the manual for-loop. Bug fixes included:
- escalation_count is incremented on every escalation
- failure_graph.record_failure() is called on every error
- hypothesis_graph.add_evidence() is called on task outcomes
- Hardcoded EscalationPolicy() fallbacks are eliminated
"""

from __future__ import annotations

import asyncio
import logging
import time

from src.api.models import ChatRequest, ChatResponse
from src.api.services.memrl import score_completed_task
from src.graph import run_task, GraphConfig, TaskDeps, TaskState
from src.llm_primitives import LLMPrimitives
from src.repl_environment import REPLEnvironment

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
    _strip_tool_outputs,
)
from src.api.structured_logging import task_extra

log = logging.getLogger(__name__)


# ── Stage 10: REPL orchestration mode ───────────────────────────────────


def _tools_success(answer: str, tool_outputs: list, tool_invocations: int) -> bool | None:
    """Infer whether tool outputs influenced the final answer."""
    if tool_invocations <= 0:
        return None
    if not tool_outputs or not answer:
        return None
    answer_text = answer.lower()
    for output in tool_outputs:
        text = str(output).strip()
        if not text:
            continue
        if text[:200].lower() in answer_text:
            return True
    return False


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
            len(doc_context.sections),
            len(doc_context.figures),
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
                score_completed_task(
                    state,
                    task_id,
                    force_role=request.force_role,
                    real_mode=request.real_mode,
                )

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
            f"Long context detected ({context_chars:,} chars). Using REPL exploration strategy."
        )

    max_turns = (
        LONG_CONTEXT_CONFIG["max_turns"] if use_long_context_exploration else request.max_turns
    )

    # ── Run orchestration graph ──────────────────────────────────────
    # The graph replaces the manual for-loop with typed node transitions.
    # Bug fixes: escalation_count incremented, failure_graph.record_failure()
    # called, hypothesis_graph.add_evidence() called.

    graph_config = GraphConfig(max_turns=max_turns)
    try:
        graph_config = GraphConfig.from_config()
        graph_config.max_turns = max_turns
    except Exception:
        pass  # Use defaults if config unavailable

    task_state = TaskState(
        task_id=task_id,
        prompt=request.prompt,
        context=combined_context,
        current_role=initial_role,
        role_history=[str(initial_role)],
        max_turns=max_turns,
        tool_required=routing.tool_required,
        tool_hint=routing.tool_hint,
    )

    task_deps = TaskDeps(
        primitives=primitives,
        repl=repl,
        failure_graph=state.failure_graph if hasattr(state, "failure_graph") else None,
        hypothesis_graph=state.hypothesis_graph if hasattr(state, "hypothesis_graph") else None,
        config=graph_config,
        progress_logger=state.progress_logger,
        session_store=state.session_store if hasattr(state, "session_store") else None,
    )

    graph_result = await run_task(task_state, task_deps, start_role=initial_role)

    answer = graph_result.answer
    turns = graph_result.turns
    role_history = graph_result.role_history or [str(initial_role)]
    current_role = role_history[-1] if role_history else str(initial_role)
    delegation_events = graph_result.delegation_events

    # ── Post-graph processing ────────────────────────────────────────

    # Quality review gate (skip when force_role is set —
    # seeding/eval calls should not trigger expensive architect reviews)
    if (
        graph_result.success
        and request.real_mode
        and not request.force_role
        and _should_review(state, task_id, current_role, answer)
    ):
        log.info(
            "Review gate triggered for %s (task %s)",
            current_role,
            task_id,
            extra=task_extra(task_id=task_id, role=current_role, stage="review"),
        )
        verdict = await asyncio.to_thread(
            _architect_verdict,
            question=request.prompt,
            answer=answer,
            primitives=primitives,
        )
        if verdict and verdict.upper().startswith("WRONG"):
            corrections = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
            log.info(
                "Review verdict: WRONG — revising (%.80s)",
                corrections,
                extra=task_extra(task_id=task_id, role=current_role, stage="review"),
            )
            answer = await asyncio.to_thread(
                _fast_revise,
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

    # If max turns reached without FINAL() and graph returned empty
    if not answer:
        # Try rescue: extract answer from last LLM output before generating error
        last_output = task_state.last_output
        if last_output:
            from src.graph.nodes import _rescue_from_last_output

            rescued = _rescue_from_last_output(last_output)
            if rescued:
                log.info("Post-graph rescue: %r", rescued[:100])
                answer = rescued

    if not answer:
        last_output = task_state.last_output
        tool_outputs = repl.artifacts.get("_tool_outputs", [])
        cleaned_output = _strip_tool_outputs(last_output, tool_outputs) if last_output else ""

        if cleaned_output and len(cleaned_output) > 20:
            answer = f"[Max turns ({max_turns}) reached without FINAL()]\n\n{cleaned_output}"
        elif last_output:
            answer = (
                f"[Max turns ({max_turns}) reached without FINAL()]\n\nLast output:\n{last_output}"
            )
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
        score_completed_task(
            state,
            task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )

    tool_outputs = repl.artifacts.get("_tool_outputs", [])
    tools_success = _tools_success(answer, tool_outputs, repl._tool_invocations)
    delegation_success = None
    if delegation_events:
        delegation_success = any(e.get("success") for e in delegation_events)

    invocation_log = (
        repl.tool_registry.get_invocation_log()
        if repl.tool_registry
        else []
    )
    tools_called = [inv.tool_name for inv in invocation_log]
    tool_timings = [
        {"tool_name": inv.tool_name, "elapsed_ms": inv.elapsed_ms, "success": inv.success}
        for inv in invocation_log
    ]
    tools_used = max(repl._tool_invocations, len(tools_called), len(tool_timings))

    # Detect parallel tool usage from invocation log
    parallel_tools = False
    if len(tools_called) >= 2:
        read_only = {"peek", "grep", "list_dir", "file_info", "list_tools",
                     "recall", "list_findings", "registry_lookup", "my_role",
                     "route_advice", "context_len", "benchmark_compare"}
        parallel_tools = all(t in read_only for t in tools_called) and len(tools_called) >= 2

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
        tools_used=tools_used,
        tools_called=tools_called,
        tool_timings=tool_timings,
        delegation_events=delegation_events,
        tools_success=tools_success,
        delegation_success=delegation_success,
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
        # Orchestrator intelligence diagnostics
        think_harder_attempted=task_state.think_harder_attempted,
        think_harder_succeeded=task_state.think_harder_succeeded,
        grammar_enforced=task_state.grammar_enforced,
        parallel_tools_used=parallel_tools,
        cache_affinity_bonus=task_state.cache_affinity_bonus,
    )

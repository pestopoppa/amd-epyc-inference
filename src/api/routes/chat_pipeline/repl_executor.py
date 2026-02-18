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
import uuid
from collections import Counter
from datetime import datetime, timezone

from src.api.models import ChatRequest, ChatResponse
from src.api.services.memrl import score_completed_task
from src.graph import run_task, GraphConfig, TaskDeps, TaskState
from src.llm_primitives import LLMPrimitives
from src.constants import TOOL_OUTPUT_MATCH_LEN
from src.repl_environment import REPLEnvironment
from src.session.models import Checkpoint

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


def _tools_success(
    answer: str,
    tool_outputs: list,
    tool_invocations: int,
    invocation_log: list | None = None,
) -> bool | None:
    """Infer whether tool outputs influenced the final answer."""
    if tool_invocations <= 0:
        return None
    if not answer:
        return None

    if tool_outputs:
        answer_text = answer.lower()
        for output in tool_outputs:
            text = str(output).strip()
            if not text:
                continue
            if text[:TOOL_OUTPUT_MATCH_LEN].lower() in answer_text:
                return True
        return False

    # Deferred mode: _tool_outputs may be intentionally empty.
    if invocation_log:
        successes = [bool(getattr(inv, "success", False)) for inv in invocation_log]
        if successes:
            return any(successes)
    return False


def _build_delegation_diagnostics(
    role_history: list[str],
    delegation_events: list[dict],
) -> dict:
    edge_counts = Counter(
        (e.get("from_role", ""), e.get("to_role", ""))
        for e in delegation_events
        if isinstance(e, dict)
    )
    repeated_edges = {
        f"{src}->{dst}": n
        for (src, dst), n in edge_counts.items()
        if src and dst and n > 1
    }
    role_counts = Counter(role_history)
    repeated_roles = {r: n for r, n in role_counts.items() if n > 1}
    return {
        "role_chain_len": len(role_history),
        "delegation_events_count": len(delegation_events),
        "repeated_edges": repeated_edges,
        "repeated_roles": repeated_roles,
    }


def _build_tool_chain_summary(invocation_log: list, chain_exec_log: list[dict] | None = None) -> list[dict]:
    """Group chained tool invocations by chain_id for response diagnostics."""
    chains: dict[str, dict] = {}
    for inv in invocation_log:
        chain_id = getattr(inv, "chain_id", None)
        if not chain_id:
            continue
        entry = chains.setdefault(
            chain_id,
            {
                "chain_id": chain_id,
                "caller_type": getattr(inv, "caller_type", "chain"),
                "tools": [],
                "elapsed_ms": 0.0,
                "success": True,
            },
        )
        entry["tools"].append(getattr(inv, "tool_name", ""))
        entry["elapsed_ms"] += float(getattr(inv, "elapsed_ms", 0.0) or 0.0)
        entry["success"] = bool(entry["success"] and getattr(inv, "success", False))

    if chain_exec_log:
        for meta in chain_exec_log:
            if not isinstance(meta, dict):
                continue
            cid = str(meta.get("chain_id", "")).strip()
            if not cid:
                continue
            entry = chains.setdefault(
                cid,
                {
                    "chain_id": cid,
                    "caller_type": "chain",
                    "tools": [],
                    "elapsed_ms": 0.0,
                    "success": True,
                },
            )
            for k in (
                "mode_requested",
                "mode_used",
                "fallback_to_seq",
                "parallel_mutations_enabled",
                "waves",
                "steps",
            ):
                if k in meta:
                    entry[k] = meta[k]

    summaries = list(chains.values())
    summaries.sort(key=lambda c: c["chain_id"])
    return summaries


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

    # Inject SkillBank skill context if available (SkillRL §3.2)
    if routing.skill_context:
        combined_context = f"{routing.skill_context}\n\n{combined_context}"

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

    # Phase 3: cross-request globals restore (opt-in via session_id).
    session_id = getattr(request, "session_id", None)
    session_store = state.session_store if hasattr(state, "session_store") else None
    session_persistence: dict[str, object] = {
        "session_id": session_id,
        "restore_attempted": bool(session_id and session_store),
        "restore_found_checkpoint": False,
        "restore_success": False,
        "restored_globals": 0,
        "skipped_globals": [],
        "restore_error": None,
        "checkpoint_saved": False,
        "checkpoint_id": None,
        "saved_globals": 0,
        "save_error": None,
    }
    if session_id and session_store:
        try:
            checkpoint = session_store.get_latest_checkpoint(session_id)
            if checkpoint:
                session_persistence["restore_found_checkpoint"] = True
            if checkpoint and checkpoint.user_globals:
                repl.restore(checkpoint.to_dict())
                session_persistence["restore_success"] = True
                session_persistence["restored_globals"] = len(checkpoint.user_globals)
                session_persistence["skipped_globals"] = list(checkpoint.skipped_user_globals or [])
                log.info(
                    "Restored %d globals from session %s",
                    len(checkpoint.user_globals),
                    session_id[:8],
                    extra=task_extra(task_id=task_id, stage="execute", mode="repl_restore"),
                )
        except Exception as e:
            session_persistence["restore_error"] = str(e)
            log.warning(
                "Session globals restore failed for %s: %s",
                session_id[:8],
                e,
                extra=task_extra(task_id=task_id, stage="execute", mode="repl_restore"),
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
                    completion_meta={
                        "producer_role": str(TWO_STAGE_CONFIG["stage2_role"]),
                        "delegation_lineage": [str(initial_role), str(TWO_STAGE_CONFIG["stage2_role"])],
                        "final_answer_role": str(TWO_STAGE_CONFIG["stage2_role"]),
                    },
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
                skills_retrieved=len(routing.skill_ids),
                skill_ids=routing.skill_ids,
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
            from src.graph.helpers import _rescue_from_last_output

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
            completion_meta={
                "producer_role": current_role,
                "delegation_lineage": role_history,
                "final_answer_role": current_role,
                "workspace_version": (task_state.workspace_state or {}).get("version", 0),
                "workspace_decisions": len((task_state.workspace_state or {}).get("decisions", [])),
            },
        )
        score_completed_task(
            state,
            task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )

    invocation_log = (
        repl.tool_registry.get_invocation_log()
        if repl.tool_registry
        else []
    )
    tool_outputs = repl.artifacts.get("_tool_outputs", [])
    tools_success = _tools_success(
        answer,
        tool_outputs,
        repl._tool_invocations,
        invocation_log=invocation_log,
    )
    delegation_success = None
    if delegation_events:
        delegation_success = any(e.get("success") for e in delegation_events)
    delegation_diag = _build_delegation_diagnostics(role_history, delegation_events)

    tools_called = [inv.tool_name for inv in invocation_log]
    tool_timings = [
        {"tool_name": inv.tool_name, "elapsed_ms": inv.elapsed_ms, "success": inv.success}
        for inv in invocation_log
    ]
    chain_exec_log = []
    if hasattr(repl, "get_chain_execution_log"):
        try:
            chain_exec_log = repl.get_chain_execution_log()
        except Exception:
            chain_exec_log = []
    tool_chains = _build_tool_chain_summary(invocation_log, chain_exec_log=chain_exec_log)
    tools_used = max(repl._tool_invocations, len(tools_called), len(tool_timings))

    # Detect parallel tool usage from invocation log
    parallel_tools = False
    if len(tools_called) >= 2:
        read_only = set()
        if hasattr(repl, "_get_read_only_tools"):
            try:
                read_only = repl._get_read_only_tools()
            except Exception:
                read_only = set()
        if not read_only and repl.tool_registry and hasattr(repl.tool_registry, "get_read_only_tools"):
            try:
                read_only = set(repl.tool_registry.get_read_only_tools())
            except Exception:
                read_only = set()
        parallel_tools = all(t in read_only for t in tools_called) and len(tools_called) >= 2

    if session_id and session_store:
        try:
            session = session_store.get_session(session_id)
            if session:
                repl_checkpoint = repl.checkpoint()
                checkpoint = Checkpoint(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    created_at=datetime.now(timezone.utc),
                    context_hash="sha256:chat_pipeline",
                    artifacts=repl_checkpoint.get("artifacts", {}),
                    execution_count=repl_checkpoint.get("execution_count", 0),
                    exploration_calls=repl_checkpoint.get("exploration_calls", 0),
                    message_count=turns,
                    trigger="chat_request",
                    user_globals=repl_checkpoint.get("user_globals", {}),
                    variable_lineage=repl_checkpoint.get("variable_lineage", {}),
                    skipped_user_globals=repl_checkpoint.get("skipped_user_globals", []),
                )
                session_store.save_checkpoint(checkpoint)
                session_persistence["checkpoint_saved"] = True
                session_persistence["checkpoint_id"] = checkpoint.id
                session_persistence["saved_globals"] = len(checkpoint.user_globals)
            else:
                session_persistence["save_error"] = "session_not_found"
        except Exception as e:
            session_persistence["save_error"] = str(e)
            log.warning(
                "Session checkpoint save failed for %s: %s",
                session_id[:8],
                e,
                extra=task_extra(task_id=task_id, stage="execute", mode="repl_checkpoint"),
            )

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
        tool_chains=tool_chains,
        delegation_events=delegation_events,
        delegation_diagnostics=delegation_diag,
        tools_success=tools_success,
        delegation_success=delegation_success,
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
        # Orchestrator intelligence diagnostics
        think_harder_attempted=task_state.think_harder_attempted,
        think_harder_succeeded=task_state.think_harder_succeeded,
        think_harder_expected_roi=float(task_state.artifacts.get("think_harder_expected_roi", 0.0)),
        grammar_enforced=task_state.grammar_enforced,
        parallel_tools_used=parallel_tools,
        cache_affinity_bonus=task_state.cache_affinity_bonus,
        # SkillBank integration
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
        session_persistence=session_persistence,
    )

"""Orchestration graph nodes — typed escalation flow.

Each node class maps to one or more orchestration roles. The ``run()``
return-type Union encodes which transitions are valid, enforced at type
check time.  Shared helpers extract reusable logic from the old
``repl_executor.py`` manual loop.

Bug fixes included in this migration:
- ``state.escalation_count`` is incremented on every escalation.
- ``deps.failure_graph.record_failure()`` is called on every error.
- ``deps.hypothesis_graph.add_evidence()`` is called on task outcomes.
- Hardcoded ``EscalationPolicy()`` fallbacks are eliminated.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Union

from pydantic_graph import BaseNode, End, GraphRunContext

from src.escalation import ErrorCategory
from src.roles import Role

from src.graph.state import (
    TaskDeps,
    TaskResult,
    TaskState,
)

log = logging.getLogger(__name__)

# Type aliases
Ctx = GraphRunContext[TaskState, TaskDeps]


# ── Shared helpers ─────────────────────────────────────────────────────


def _classify_error(error_message: str) -> ErrorCategory:
    """Classify an error message into an ErrorCategory.

    Extracted from prompt_builders.code_utils.classify_error for
    dependency isolation — nodes should not import prompt_builders.
    """
    lower = error_message.lower()

    if any(kw in lower for kw in ("timeout", "timed out", "deadline")):
        return ErrorCategory.TIMEOUT
    if any(kw in lower for kw in ("json", "schema", "validation", "jsonschema")):
        return ErrorCategory.SCHEMA
    if any(kw in lower for kw in ("format", "style", "lint", "ruff", "markdown")):
        return ErrorCategory.FORMAT
    if any(
        kw in lower
        for kw in ("abort", "generation aborted", "early_abort", "early abort")
    ):
        return ErrorCategory.EARLY_ABORT
    if any(
        kw in lower
        for kw in ("backend", "connection", "unreachable", "infrastructure", "502", "503")
    ):
        return ErrorCategory.INFRASTRUCTURE
    if any(
        kw in lower
        for kw in ("syntax", "type error", "typeerror", "nameerror", "import error", "test fail")
    ):
        return ErrorCategory.CODE
    if any(kw in lower for kw in ("wrong", "incorrect", "assertion", "logic")):
        return ErrorCategory.LOGIC

    return ErrorCategory.UNKNOWN


def _record_failure(ctx: Ctx, error_category: ErrorCategory, error_msg: str) -> None:
    """Record failure in the FailureGraph (anti-memory).

    FIX: This was never called in the old repl_executor.py.
    """
    fg = ctx.deps.failure_graph
    if fg is None:
        return
    try:
        fg.record_failure(
            memory_id=ctx.state.task_id,
            symptoms=[error_category.value, error_msg[:100]],
            description=f"{ctx.state.current_role} failed: {error_msg[:200]}",
            severity=min(ctx.state.consecutive_failures + 2, 5),
        )
    except Exception as exc:
        log.debug("failure_graph.record_failure failed: %s", exc)


def _record_mitigation(ctx: Ctx, from_role: str, to_role: str) -> None:
    """Record a successful mitigation in the FailureGraph.

    FIX: This was never called in the old code.
    """
    fg = ctx.deps.failure_graph
    if fg is None:
        return
    try:
        fg.record_mitigation(
            memory_id=ctx.state.task_id,
            description=f"Escalation from {from_role} to {to_role} succeeded",
        )
    except Exception as exc:
        log.debug("failure_graph.record_mitigation failed: %s", exc)


def _add_evidence(ctx: Ctx, outcome: str, delta: float) -> None:
    """Record evidence in the HypothesisGraph.

    FIX: This was never called in the old code.
    """
    hg = ctx.deps.hypothesis_graph
    if hg is None:
        return
    try:
        hg.add_evidence(
            hypothesis_id=ctx.state.task_id,
            evidence=f"{ctx.state.current_role}:{outcome}",
            delta=delta,
        )
    except Exception as exc:
        log.debug("hypothesis_graph.add_evidence failed: %s", exc)


def _log_escalation(ctx: Ctx, from_role: str, to_role: str, reason: str) -> None:
    """Log an escalation event via progress logger."""
    pl = ctx.deps.progress_logger
    if pl is None:
        return
    try:
        pl.log_escalation(
            task_id=ctx.state.task_id,
            from_tier=from_role,
            to_tier=to_role,
            reason=reason,
        )
    except Exception as exc:
        log.debug("progress_logger.log_escalation failed: %s", exc)


async def _execute_turn(ctx: Ctx, role: Role | str) -> tuple[str, str | None, bool, dict]:
    """Execute one LLM → REPL turn.

    Returns:
        (code_output, error_or_none, is_final, artifacts)
    """
    state = ctx.state
    deps = ctx.deps
    state.turns += 1
    log.debug("_execute_turn: turn=%d, role=%s", state.turns, role)

    if deps.primitives is None or deps.repl is None:
        return "", "No LLM primitives or REPL configured", False, {}

    # Build prompt
    if state.escalation_prompt:
        prompt = state.escalation_prompt
        state.escalation_prompt = ""
    else:
        from src.prompt_builders import build_root_lm_prompt

        repl_state = deps.repl.get_state()
        prompt = build_root_lm_prompt(
            state=repl_state,
            original_prompt=state.prompt,
            last_output=state.last_output,
            last_error=state.last_error,
            turn=state.turns - 1,
        )

    # LLM call
    try:
        code = await asyncio.to_thread(
            deps.primitives.llm_call,
            prompt,
            role=str(role),
            n_tokens=1024,
        )
    except Exception as e:
        return "", f"LLM call failed: {e}", False, {}

    # Extract and wrap code
    from src.prompt_builders import extract_code_from_response, auto_wrap_final

    code = extract_code_from_response(code)
    code = auto_wrap_final(code)

    # REPL execution
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(deps.repl.execute, code),
            timeout=deps.repl.config.timeout_seconds,
        )
    except asyncio.TimeoutError:
        from src.repl_environment.types import ExecutionResult

        result = ExecutionResult(
            output="",
            is_final=False,
            error=f"REPL execution timed out after {deps.repl.config.timeout_seconds}s",
        )

    artifacts = dict(deps.repl.artifacts) if hasattr(deps.repl, "artifacts") else {}
    # Prefer final_answer when is_final=True (FINAL() captures the answer
    # in final_answer, not in output)
    output = result.output
    if result.is_final and hasattr(result, "final_answer") and result.final_answer:
        output = result.final_answer
    log.debug(
        "_execute_turn: output=%r, error=%r, is_final=%s, code=%r",
        output[:200] if output else "",
        result.error[:200] if result.error else None,
        result.is_final,
        code[:200] if code else "",
    )
    return output, result.error if result.error else None, result.is_final, artifacts


def _should_escalate(
    ctx: Ctx,
    error_category: ErrorCategory,
    next_tier: Role | None,
) -> bool:
    """Determine if we should escalate (vs retry or fail)."""
    cfg = ctx.deps.config
    state = ctx.state

    # Format/schema errors never escalate
    if error_category in cfg.no_escalate_categories:
        return False

    # No target to escalate to
    if next_tier is None:
        return False

    # Max escalations reached
    if state.escalation_count >= cfg.max_escalations:
        return False

    # Retries exhausted → escalate
    return state.consecutive_failures >= cfg.max_retries


def _should_retry(ctx: Ctx, error_category: ErrorCategory) -> bool:
    """Determine if we should retry with the same role."""
    cfg = ctx.deps.config
    return ctx.state.consecutive_failures < cfg.max_retries


def _timeout_skip(ctx: Ctx, error_msg: str) -> bool:
    """Check if a timeout error should result in a SKIP (optional gate)."""
    # For now, check if the error mentions an optional gate
    cfg = ctx.deps.config
    for gate in cfg.optional_gates:
        if gate in error_msg.lower():
            return True
    return False


def _make_end_result(ctx: Ctx, answer: str, success: bool) -> End[TaskResult]:
    """Create an End node with a TaskResult."""
    repl = ctx.deps.repl
    tool_outputs = []
    tools_used = 0
    if repl and hasattr(repl, "artifacts"):
        tool_outputs = repl.artifacts.get("_tool_outputs", [])
    if repl and hasattr(repl, "_tool_invocations"):
        tools_used = repl._tool_invocations

    # Record outcome evidence
    _add_evidence(ctx, "success" if success else "failure", 0.5 if success else -0.5)

    return End(
        TaskResult(
            answer=answer,
            success=success,
            role_history=list(ctx.state.role_history),
            tool_outputs=tool_outputs,
            tools_used=tools_used,
            turns=ctx.state.turns,
            delegation_events=list(ctx.state.delegation_events),
        )
    )


def _resolve_answer(output: str, tool_outputs: list) -> str:
    """Extract the best answer from REPL output and tool outputs.

    Simplified version that doesn't depend on chat_utils internals.
    The full answer resolution (with final_answer handling, stub detection,
    tool output stripping) happens in the repl_executor wrapper.
    """
    if output and output.strip():
        return output.strip()
    if tool_outputs:
        return "\n".join(str(t) for t in tool_outputs if t)
    return ""


# ── Node classes ───────────────────────────────────────────────────────


@dataclass
class FrontdoorNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Entry node for unclassified/frontdoor requests.

    Escalates to CoderNode on failure.
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["FrontdoorNode", "CoderNode", "WorkerNode", End[TaskResult]]:
        state = ctx.state

        if state.turns >= state.max_turns:
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        output, error, is_final, artifacts = await _execute_turn(ctx, Role.FRONTDOOR)
        state.artifacts.update(artifacts)

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            return _make_end_result(ctx, answer, True)

        if error:
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if error_cat == ErrorCategory.EARLY_ABORT:
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.CODER_PRIMARY)
                _log_escalation(ctx, from_role, str(Role.CODER_PRIMARY), f"Early abort: {error[:100]}")
                return CoderNode()

            if _should_escalate(ctx, error_cat, Role.CODER_PRIMARY):
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.CODER_PRIMARY)
                _log_escalation(
                    ctx, from_role, str(Role.CODER_PRIMARY),
                    f"Escalating after {state.consecutive_failures} failures",
                )
                return CoderNode()

            if _should_retry(ctx, error_cat):
                return FrontdoorNode()

            return _make_end_result(ctx, f"[FAILED: {error}]", False)

        state.consecutive_failures = 0
        state.last_error = ""
        state.last_output = output
        return FrontdoorNode()


@dataclass
class WorkerNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Worker node for all WORKER_* roles.

    Escalates to CoderNode on failure.
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["WorkerNode", "CoderNode", End[TaskResult]]:
        state = ctx.state

        if state.turns >= state.max_turns:
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        output, error, is_final, artifacts = await _execute_turn(ctx, state.current_role)
        state.artifacts.update(artifacts)

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            return _make_end_result(ctx, answer, True)

        if error:
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if error_cat == ErrorCategory.EARLY_ABORT:
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.CODER_PRIMARY)
                _log_escalation(ctx, from_role, str(Role.CODER_PRIMARY), f"Early abort: {error[:100]}")
                return CoderNode()

            if _should_escalate(ctx, error_cat, Role.CODER_PRIMARY):
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.CODER_PRIMARY)
                _log_escalation(
                    ctx, from_role, str(Role.CODER_PRIMARY),
                    f"Escalating after {state.consecutive_failures} failures",
                )
                return CoderNode()

            if _should_retry(ctx, error_cat):
                return WorkerNode()

            return _make_end_result(ctx, f"[FAILED: {error}]", False)

        state.consecutive_failures = 0
        state.last_error = ""
        state.last_output = output
        return WorkerNode()


@dataclass
class CoderNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Coder node for CODER_PRIMARY and THINKING_REASONING roles.

    Escalates to ArchitectNode on failure.
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["CoderNode", "ArchitectNode", End[TaskResult]]:
        state = ctx.state

        if state.turns >= state.max_turns:
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        output, error, is_final, artifacts = await _execute_turn(ctx, state.current_role)
        state.artifacts.update(artifacts)

        # Check model-initiated escalation
        if artifacts.get("_escalation_requested"):
            artifacts.pop("_escalation_target", None)
            reason = artifacts.pop("_escalation_reason", "Model requested")
            artifacts.pop("_escalation_requested", None)

            state.escalation_count += 1
            state.consecutive_failures = 0
            from_role = str(state.current_role)
            state.record_role(Role.ARCHITECT_GENERAL)
            _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL), f"Model-initiated: {reason}")
            return ArchitectNode()

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            # Record mitigation if we got here via escalation
            if state.escalation_count > 0:
                _record_mitigation(ctx, state.role_history[-2] if len(state.role_history) > 1 else "unknown", str(state.current_role))
            return _make_end_result(ctx, answer, True)

        if error:
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if error_cat == ErrorCategory.EARLY_ABORT:
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.ARCHITECT_GENERAL)
                _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL), f"Early abort: {error[:100]}")
                return ArchitectNode()

            if _should_escalate(ctx, error_cat, Role.ARCHITECT_GENERAL):
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.ARCHITECT_GENERAL)
                _log_escalation(
                    ctx, from_role, str(Role.ARCHITECT_GENERAL),
                    f"Escalating after {state.consecutive_failures} failures",
                )
                return ArchitectNode()

            if _should_retry(ctx, error_cat):
                return CoderNode()

            return _make_end_result(ctx, f"[FAILED: {error}]", False)

        state.consecutive_failures = 0
        state.last_error = ""
        state.last_output = output
        return CoderNode()


@dataclass
class CoderEscalationNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Escalation coder node for CODER_ESCALATION role.

    Escalates to ArchitectCodingNode (parallel coding chain).
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["CoderEscalationNode", "ArchitectCodingNode", End[TaskResult]]:
        state = ctx.state

        if state.turns >= state.max_turns:
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        output, error, is_final, artifacts = await _execute_turn(ctx, Role.CODER_ESCALATION)
        state.artifacts.update(artifacts)

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            if state.escalation_count > 0:
                _record_mitigation(ctx, state.role_history[-2] if len(state.role_history) > 1 else "unknown", str(state.current_role))
            return _make_end_result(ctx, answer, True)

        if error:
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if error_cat == ErrorCategory.EARLY_ABORT:
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.ARCHITECT_CODING)
                _log_escalation(ctx, from_role, str(Role.ARCHITECT_CODING), f"Early abort: {error[:100]}")
                return ArchitectCodingNode()

            if _should_escalate(ctx, error_cat, Role.ARCHITECT_CODING):
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.ARCHITECT_CODING)
                _log_escalation(
                    ctx, from_role, str(Role.ARCHITECT_CODING),
                    f"Escalating after {state.consecutive_failures} failures",
                )
                return ArchitectCodingNode()

            if _should_retry(ctx, error_cat):
                return CoderEscalationNode()

            return _make_end_result(ctx, f"[FAILED: {error}]", False)

        state.consecutive_failures = 0
        state.last_error = ""
        state.last_output = output
        return CoderEscalationNode()


@dataclass
class IngestNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Ingest node for INGEST_LONG_CONTEXT role (SSM path, no spec).

    Escalates to ArchitectNode on failure.
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["IngestNode", "ArchitectNode", End[TaskResult]]:
        state = ctx.state

        if state.turns >= state.max_turns:
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        output, error, is_final, artifacts = await _execute_turn(
            ctx, Role.INGEST_LONG_CONTEXT
        )
        state.artifacts.update(artifacts)

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            return _make_end_result(ctx, answer, True)

        if error:
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if error_cat == ErrorCategory.EARLY_ABORT:
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.ARCHITECT_GENERAL)
                _log_escalation(ctx, from_role, str(Role.ARCHITECT_GENERAL), f"Early abort: {error[:100]}")
                return ArchitectNode()

            if _should_escalate(ctx, error_cat, Role.ARCHITECT_GENERAL):
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.ARCHITECT_GENERAL)
                _log_escalation(
                    ctx, from_role, str(Role.ARCHITECT_GENERAL),
                    f"Escalating after {state.consecutive_failures} failures",
                )
                return ArchitectNode()

            if _should_retry(ctx, error_cat):
                return IngestNode()

            return _make_end_result(ctx, f"[FAILED: {error}]", False)

        state.consecutive_failures = 0
        state.last_error = ""
        state.last_output = output
        return IngestNode()


@dataclass
class ArchitectNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Architect node for ARCHITECT_GENERAL role.

    Terminal — no further escalation. Falls back to EXPLORE on repeated failure.
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["ArchitectNode", End[TaskResult]]:
        state = ctx.state

        if state.turns >= state.max_turns:
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        output, error, is_final, artifacts = await _execute_turn(
            ctx, Role.ARCHITECT_GENERAL
        )
        state.artifacts.update(artifacts)

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            if state.escalation_count > 0:
                _record_mitigation(ctx, state.role_history[-2] if len(state.role_history) > 1 else "unknown", str(state.current_role))
            return _make_end_result(ctx, answer, True)

        if error:
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if _should_retry(ctx, error_cat):
                return ArchitectNode()

            # Terminal — EXPLORE fallback
            _add_evidence(ctx, "explore_fallback", -0.3)
            return _make_end_result(
                ctx,
                f"[FAILED: Terminal role {state.current_role}: {error}]",
                False,
            )

        state.consecutive_failures = 0
        state.last_error = ""
        state.last_output = output
        return ArchitectNode()


@dataclass
class ArchitectCodingNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Architect coding node for ARCHITECT_CODING role.

    Terminal — no further escalation.
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["ArchitectCodingNode", End[TaskResult]]:
        state = ctx.state

        if state.turns >= state.max_turns:
            return _make_end_result(
                ctx, f"[Max turns ({state.max_turns}) reached]", False
            )

        output, error, is_final, artifacts = await _execute_turn(
            ctx, Role.ARCHITECT_CODING
        )
        state.artifacts.update(artifacts)

        if is_final:
            tool_outputs = artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(output, tool_outputs)
            if state.escalation_count > 0:
                _record_mitigation(ctx, state.role_history[-2] if len(state.role_history) > 1 else "unknown", str(state.current_role))
            return _make_end_result(ctx, answer, True)

        if error:
            state.consecutive_failures += 1
            state.last_error = error
            state.last_output = output
            error_cat = _classify_error(error)
            _record_failure(ctx, error_cat, error)

            if _should_retry(ctx, error_cat):
                return ArchitectCodingNode()

            _add_evidence(ctx, "explore_fallback", -0.3)
            return _make_end_result(
                ctx,
                f"[FAILED: Terminal role {state.current_role}: {error}]",
                False,
            )

        state.consecutive_failures = 0
        state.last_error = ""
        state.last_output = output
        return ArchitectCodingNode()


# ── Node selection helper ──────────────────────────────────────────────

# Maps initial roles to their starting node class.
_ROLE_TO_NODE: dict[Role, type] = {
    Role.FRONTDOOR: FrontdoorNode,
    Role.WORKER_GENERAL: WorkerNode,
    Role.WORKER_MATH: WorkerNode,
    Role.WORKER_SUMMARIZE: WorkerNode,
    Role.WORKER_VISION: WorkerNode,
    Role.TOOLRUNNER: WorkerNode,
    Role.CODER_PRIMARY: CoderNode,
    Role.THINKING_REASONING: CoderNode,
    Role.CODER_ESCALATION: CoderEscalationNode,
    Role.INGEST_LONG_CONTEXT: IngestNode,
    Role.ARCHITECT_GENERAL: ArchitectNode,
    Role.ARCHITECT_CODING: ArchitectCodingNode,
}


def select_start_node(role: Role | str) -> BaseNode:
    """Select the graph start node class for a given role."""
    if isinstance(role, str):
        role = Role.from_string(role) or Role.FRONTDOOR

    node_cls = _ROLE_TO_NODE.get(role, FrontdoorNode)
    return node_cls()

"""Orchestration graph nodes — typed escalation flow.

Each node class maps to one or more orchestration roles. The ``run()``
return-type Union encodes which transitions are valid, enforced at type
check time.  Shared helpers live in ``src.graph.helpers``.

Bug fixes included in this migration:
- ``state.escalation_count`` is incremented on every escalation.
- ``deps.failure_graph.record_failure()`` is called on every error.
- ``deps.hypothesis_graph.add_evidence()`` is called on task outcomes.
- Hardcoded ``EscalationPolicy()`` fallbacks are eliminated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

from pydantic_graph import BaseNode, End

from src.escalation import ErrorCategory
from src.roles import Role

from src.graph.state import (
    TaskDeps,
    TaskResult,
    TaskState,
)

# Import all helpers from the extracted module
from src.graph.helpers import (  # noqa: F401 — re-exported for backward compat
    Ctx,
    MAX_CONSECUTIVE_NUDGES,
    _add_evidence,
    _check_approval_gate,
    _classify_error,
    _detect_role_cycle,
    _execute_turn,
    _extract_final_from_raw,
    _extract_prose_answer,
    _is_comment_only,
    _log_escalation,
    _make_end_result,
    _maybe_compact_context,
    _record_failure,
    _record_mitigation,
    _rescue_from_last_output,
    _resolve_answer,
    _should_escalate,
    _should_retry,
    _should_think_harder,
    _timeout_skip,
)

log = logging.getLogger(__name__)



# ── Node classes ───────────────────────────────────────────────────────


@dataclass
class FrontdoorNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    """Entry node for unclassified/frontdoor requests.

    Escalates to CoderNode on failure (via escalation map: FRONTDOOR → CODER_PRIMARY).
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["FrontdoorNode", "CoderNode", "WorkerNode", End[TaskResult]]:
        state = ctx.state

        if state.turns >= state.max_turns:
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (frontdoor): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
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

            # Think-harder: same model with CoT + 2x tokens before escalating
            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = {
                    "n_tokens": 4096,
                    "cot_prefix": "Think step by step before answering.\n\n",
                    "temperature": 0.5,
                }
                log.info("Think-harder triggered at %s (attempt before escalation)", state.current_role)
                return FrontdoorNode()

            if _should_escalate(ctx, error_cat, Role.CODER_PRIMARY):
                # If think-harder was attempted and we still escalate, it failed
                if state.think_harder_attempted:
                    state.think_harder_succeeded = False
                state.escalation_count += 1
                state.consecutive_failures = 0
                state.think_harder_attempted = False  # Reset for next role
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

        state.last_output = output
        # If think-harder was attempted and we got a successful turn, it worked
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return FrontdoorNode()
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
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
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (worker): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
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

            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = {
                    "n_tokens": 4096,
                    "cot_prefix": "Think step by step before answering.\n\n",
                    "temperature": 0.5,
                }
                log.info("Think-harder triggered at %s", state.current_role)
                return WorkerNode()

            if _should_escalate(ctx, error_cat, Role.CODER_PRIMARY):
                if state.think_harder_attempted:
                    state.think_harder_succeeded = False
                state.escalation_count += 1
                state.consecutive_failures = 0
                state.think_harder_attempted = False
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
        state.last_output = output
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return WorkerNode()
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
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
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (coder): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
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

            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = {
                    "n_tokens": 4096,
                    "cot_prefix": "Think step by step before answering.\n\n",
                    "temperature": 0.5,
                }
                log.info("Think-harder triggered at %s", state.current_role)
                return CoderNode()

            if _should_escalate(ctx, error_cat, Role.ARCHITECT_GENERAL):
                if state.think_harder_attempted:
                    state.think_harder_succeeded = False
                state.escalation_count += 1
                state.consecutive_failures = 0
                state.think_harder_attempted = False
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
        state.last_output = output
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return CoderNode()
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
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
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (coder_escalation): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
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

            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = {
                    "n_tokens": 4096,
                    "cot_prefix": "Think step by step before answering.\n\n",
                    "temperature": 0.5,
                }
                log.info("Think-harder triggered at %s", state.current_role)
                return CoderEscalationNode()

            if _should_escalate(ctx, error_cat, Role.ARCHITECT_CODING):
                if state.think_harder_attempted:
                    state.think_harder_succeeded = False
                state.escalation_count += 1
                state.consecutive_failures = 0
                state.think_harder_attempted = False
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
        state.last_output = output
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return CoderEscalationNode()
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
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
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (ingest): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
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

            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = {
                    "n_tokens": 4096,
                    "cot_prefix": "Think step by step before answering.\n\n",
                    "temperature": 0.5,
                }
                log.info("Think-harder triggered at %s", state.current_role)
                return IngestNode()

            if _should_escalate(ctx, error_cat, Role.ARCHITECT_GENERAL):
                if state.think_harder_attempted:
                    state.think_harder_succeeded = False
                state.escalation_count += 1
                state.consecutive_failures = 0
                state.think_harder_attempted = False
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
        state.last_output = output
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return IngestNode()
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
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
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (architect): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
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

            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = {
                    "n_tokens": 4096,
                    "cot_prefix": "Think step by step before answering.\n\n",
                    "temperature": 0.5,
                }
                log.info("Think-harder triggered at %s (terminal role)", state.current_role)
                return ArchitectNode()

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
        state.last_output = output
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return ArchitectNode()
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
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
            rescued = _rescue_from_last_output(state.last_output)
            if rescued:
                log.info("Max-turns rescue (architect_coding): %r", rescued[:100])
                return _make_end_result(ctx, rescued, True)
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

            if _should_think_harder(ctx, error_cat):
                state.think_harder_attempted = True
                state.think_harder_config = {
                    "n_tokens": 4096,
                    "cot_prefix": "Think step by step before answering.\n\n",
                    "temperature": 0.5,
                }
                log.info("Think-harder triggered at %s (terminal role)", state.current_role)
                return ArchitectCodingNode()

            if _should_retry(ctx, error_cat):
                return ArchitectCodingNode()

            _add_evidence(ctx, "explore_fallback", -0.3)
            return _make_end_result(
                ctx,
                f"[FAILED: Terminal role {state.current_role}: {error}]",
                False,
            )

        state.consecutive_failures = 0
        state.last_output = output
        if state.think_harder_attempted and state.think_harder_succeeded is None:
            state.think_harder_succeeded = True
        nudge = artifacts.get("_nudge")
        if nudge:
            state.consecutive_nudges += 1
            state.last_error = nudge
            if state.consecutive_nudges >= MAX_CONSECUTIVE_NUDGES:
                log.warning("Max nudges (%d) reached at %s, promoting to error", state.consecutive_nudges, state.current_role)
                state.consecutive_failures += 1
                state.consecutive_nudges = 0
                return ArchitectCodingNode()
        else:
            state.consecutive_failures = 0
            state.consecutive_nudges = 0
            state.last_error = ""
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

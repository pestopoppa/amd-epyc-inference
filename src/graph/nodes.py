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
import re
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


# ── REPL tap — separate file for REPL execution visibility ────────────

_REPL_TAP_PATH = "/mnt/raid0/llm/tmp/repl_tap.log"
_repl_tap_lock = __import__("threading").Lock()


def _tap_write_repl_exec(code: str, turn: int) -> None:
    """Write REPL execution input to the REPL tap file (separate from inference tap)."""
    try:
        preview = code[:800]
        if len(code) > 800:
            preview += f"\n... [{len(code) - 800} chars truncated]"
        text = (
            f"[turn {turn}] $ python3 <<'CODE'\n"
            f"{preview}\n"
            f"CODE\n"
        )
        with _repl_tap_lock:
            with open(_REPL_TAP_PATH, "a") as f:
                f.write(text)
                f.flush()
    except Exception:
        pass


def _tap_write_repl_result(
    output: str, error: str | None, is_final: bool, turn: int,
) -> None:
    """Write REPL execution result to the REPL tap file."""
    try:
        parts: list[str] = []
        if is_final:
            parts.append(f"[turn {turn}] FINAL")
        if output:
            out_preview = output[:500]
            if len(output) > 500:
                out_preview += f"\n... [{len(output) - 500} chars truncated]"
            parts.append(out_preview)
        elif not error:
            parts.append(f"[turn {turn}] (no output)")
        if error:
            err_preview = error[:300]
            if len(error) > 300:
                err_preview += f"\n... [{len(error) - 300} chars truncated]"
            parts.append(f"[turn {turn}] ERROR:\n{err_preview}")
        parts.append("")  # trailing newline
        text = "\n".join(parts)
        with _repl_tap_lock:
            with open(_REPL_TAP_PATH, "a") as f:
                f.write(text)
                f.flush()
    except Exception:
        pass


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


async def _maybe_compact_context(ctx: Ctx) -> None:
    """Compact old context entries if conversation is long (OpenClaw pattern).

    Trigger: turns > 5 AND context > 12000 chars AND primitives available.
    Uses worker_summarize role (44 t/s) for cheap, fast compression.
    """
    from src.features import features as _get_features

    if not _get_features().session_compaction:
        return

    state = ctx.state
    if state.turns <= 5 or len(state.context) <= 12000:
        return
    if ctx.deps.primitives is None:
        return

    try:
        # Summarize old context, keep recent material
        old_context = state.context
        # Keep last 3000 chars verbatim
        keep_verbatim = old_context[-3000:] if len(old_context) > 3000 else old_context
        to_summarize = old_context[: -3000] if len(old_context) > 3000 else ""

        if not to_summarize.strip():
            return

        summary_prompt = (
            "Summarize the following conversation context into a concise paragraph "
            "preserving all key facts, decisions, and error messages:\n\n"
            f"{to_summarize[:8000]}"
        )

        summary = await asyncio.to_thread(
            ctx.deps.primitives.llm_call,
            summary_prompt,
            role="worker_summarize",
        )

        state.context = f"[Compacted context]\n{summary}\n\n[Recent]\n{keep_verbatim}"
        state.compaction_count += 1
        log.info(
            "Session compaction #%d: %d → %d chars",
            state.compaction_count,
            len(old_context),
            len(state.context),
        )
    except Exception as exc:
        log.debug("Session compaction failed (non-fatal): %s", exc)


_FINAL_RE = re.compile(
    r"""FINAL\(\s*(?:'{3}(.+?)'{3}|"{3}(.+?)"{3}|["'](.+?)["']|(\S+?))\s*\)""",
    re.DOTALL,
)


def _is_comment_only(code: str) -> bool:
    """Return True if code has no executable lines (all comments/blank)."""
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return False
    return True


def _extract_final_from_raw(text: str) -> str | None:
    """Extract answer from FINAL("answer") in raw LLM output.

    Used as rescue when REPL execution fails but the model DID produce
    a FINAL() call.  Returns None if no FINAL() found.
    """
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1) or m.group(2) or m.group(3) or m.group(4) or ""
    return None


# Patterns for extracting answers from prose (no FINAL(), no code blocks).
# Ordered most-specific first.  Captures the first non-whitespace token after
# the trigger phrase so e.g. "The answer is: D" → "D".
_PROSE_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"[Tt]he\s+(?:correct\s+)?answer\s+is[:\s]+|"
    r"[Aa]nswer[:\s]+|"
    r"[Tt]herefore[,:\s]+(?:the\s+answer\s+is[:\s]+)?|"
    r"[Ss]o\s+the\s+answer\s+is[:\s]+|"
    r"[Ss]o,?\s+I\s+will\s+go\s+with[:\s]+|"
    r"I(?:'ll|\s+will)\s+go\s+with[:\s]+|"
    r"I\s+(?:choose|select|pick)[:\s]+|"
    r"[Mm]y\s+answer\s+is[:\s]+|"
    r"[Tt]he\s+correct\s+(?:option|choice)\s+is[:\s]+"
    r")([A-Za-z0-9][A-Za-z0-9.)]*)",
)


def _extract_prose_answer(text: str) -> str | None:
    """Extract answer from prose LLM output that lacks FINAL().

    Catches common patterns like "The answer is D", "I will go with D",
    "Answer: B", etc.  Falls back to a bare MCQ letter on its own line.
    Returns None if no clear answer pattern found.
    """
    m = _PROSE_ANSWER_RE.search(text)
    if m:
        answer = m.group(1).rstrip(".)").strip()
        if answer:
            return answer
    # Fallback: bare MCQ letter on its own line (e.g. just "D")
    bare = re.search(r"(?:^|\n)\s*([A-D])\s*(?:\n|$)", text)
    if bare:
        return bare.group(1)
    return None


async def _execute_turn(ctx: Ctx, role: Role | str) -> tuple[str, str | None, bool, dict]:
    """Execute one LLM → REPL turn.

    Returns:
        (code_output, error_or_none, is_final, artifacts)
    """
    state = ctx.state
    deps = ctx.deps
    state.turns += 1
    log.debug("_execute_turn: turn=%d, role=%s", state.turns, role)

    # Session compaction before execution
    await _maybe_compact_context(ctx)

    if deps.primitives is None or deps.repl is None:
        return "", "No LLM primitives or REPL configured", False, {}

    # Build prompt
    if state.escalation_prompt:
        prompt = state.escalation_prompt
        state.escalation_prompt = ""
    else:
        from src.prompt_builders.builder import PromptBuilder
        from src.prompt_builders.types import PromptConfig, PromptStyle

        repl_state = deps.repl.get_state()
        builder = PromptBuilder(PromptConfig(style=PromptStyle.MINIMAL))
        prompt = builder.build_root_lm_prompt(
            state=repl_state,
            original_prompt=state.prompt,
            last_output=state.last_output,
            last_error=state.last_error,
            turn=state.turns - 1,
        )

    # LLM call — stop at first code block close to prevent repetition loops.
    # REPL expects one action per turn; without this, the model can generate
    # FINAL("X") then repeat the same code block hundreds of tokens.
    # Early-stop streaming: abort generation the moment FINAL(...) is
    # detected so the model doesn't keep reasoning after the answer.
    if deps.primitives is not None:
        deps.primitives._early_stop_check = lambda text: bool(_FINAL_RE.search(text))
    try:
        code = await asyncio.to_thread(
            deps.primitives.llm_call,
            prompt,
            role=str(role),
            stop_sequences=["\n```\n"],
        )
    except Exception as e:
        return "", f"LLM call failed: {e}", False, {}
    finally:
        if deps.primitives is not None:
            deps.primitives._early_stop_check = None

    # Save raw LLM output for FINAL() rescue before code extraction
    raw_llm_output = code

    # Extract and wrap code
    from src.prompt_builders import extract_code_from_response, auto_wrap_final

    code = extract_code_from_response(code)
    code = auto_wrap_final(code)

    # Prose answer rescue: model answered in prose (e.g. "The answer is D")
    # without producing FINAL() or code blocks.  Extract the answer from
    # the raw output and synthesize FINAL() to avoid an infinite REPL loop.
    if "FINAL(" not in code:
        prose_answer = _extract_prose_answer(raw_llm_output)
        if prose_answer is not None:
            log.info(
                "Prose answer rescue (turn %d): extracted %r from raw output",
                state.turns, prose_answer[:100],
            )
            code = f'FINAL("{prose_answer}")'

    # Comment-only guard: model reasoned in comments without executable code.
    # Try to rescue the answer from the comments before nudging.
    if _is_comment_only(code):
        comment_text = "\n".join(
            ln.strip().lstrip("#").strip()
            for ln in code.split("\n") if ln.strip().startswith("#")
        )
        prose_answer = _extract_prose_answer(comment_text)
        if prose_answer:
            log.info(
                "Comment-only rescue (turn %d): extracted %r from comments",
                state.turns, prose_answer[:50],
            )
            code = f'FINAL("{prose_answer}")'
            # Fall through to execute FINAL()
        else:
            log.info("Comment-only code detected (turn %d), nudging model", state.turns)
            state.last_error = (
                "Your output was all comments — no executable code ran. "
                "You already reasoned through the problem. Call FINAL now with the actual value — e.g. FINAL(\"B\") or FINAL(42)."
            )
            return "", state.last_error, False, {}

    # Comment-ratio guard: model is reasoning in comments with minimal
    # executable code (e.g. a bare `for` loop full of `# thinking...`).
    # This wastes turns without progress.  Nudge toward file-based workflow.
    code_lines = [ln for ln in code.split("\n") if ln.strip()]
    if code_lines:
        comment_lines = sum(1 for ln in code_lines if ln.strip().startswith("#"))
        ratio = comment_lines / len(code_lines)
        if ratio > 0.6 and len(code_lines) > 5 and "FINAL(" not in code:
            # Try to extract the answer the model already reasoned to
            comment_text = "\n".join(
                ln.strip().lstrip("#").strip()
                for ln in code_lines if ln.strip().startswith("#")
            )
            prose_answer = _extract_prose_answer(comment_text)
            if prose_answer:
                log.info(
                    "Comment-ratio rescue (turn %d, %.0f%% comments): extracted %r",
                    state.turns, ratio * 100, prose_answer[:50],
                )
                code = f'FINAL("{prose_answer}")'
                # Fall through to execute FINAL()
            else:
                log.info(
                    "High comment ratio (%.0f%%, turn %d), nudging to commit",
                    ratio * 100, state.turns,
                )
                state.last_error = (
                    f"Your code is {int(ratio*100)}% comments — you already reasoned through the problem. "
                    "STOP re-deriving. Call FINAL now with the value you reached — e.g. FINAL(\"B\") or FINAL(42). "
                    "Do NOT start over. Do NOT re-explain."
                )
                return "", state.last_error, False, {}

    # Write code to inference tap so the TUI shows what's being executed
    _tap_write_repl_exec(code, state.turns)

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

    # Write execution result to inference tap
    _tap_write_repl_result(result.output, result.error, result.is_final, state.turns)

    # FINAL() rescue: if REPL execution failed but the model DID write
    # FINAL("answer") in its output, extract the answer directly.
    # This prevents escalation when code before FINAL() has errors.
    if not result.is_final and result.error:
        final_rescue = _extract_final_from_raw(raw_llm_output)
        if final_rescue is not None:
            log.info("FINAL() rescue: extracted %r from raw output (REPL error: %s)",
                     final_rescue[:100], result.error[:80])
            from src.repl_environment.types import ExecutionResult

            result = ExecutionResult(
                output="",
                is_final=True,
                final_answer=final_rescue,
            )

    # No-output guard: code ran successfully but produced no output, no error,
    # and no FINAL().  Typical case: model generated a class/function definition
    # that runs silently.  Without feedback the model repeats indefinitely.
    if not result.is_final and not result.error and not result.output:
        log.info("Silent execution detected (turn %d), nudging model", state.turns)
        nudge = (
            "Your code ran but produced no output and did not call FINAL(). "
            "You must call FINAL with the actual computed value — e.g. FINAL(\"B\") or FINAL(42). "
            "If the task asks for code, call FINAL with the complete program text as a string."
        )
        return "", nudge, False, {}

    # Status-message guard: model called FINAL() with a status phrase
    # instead of the actual answer (e.g. "Code execution complete.",
    # "Done", "Function implemented and tested successfully").  Reject and nudge.
    if result.is_final and hasattr(result, "final_answer") and result.final_answer:
        _fa = result.final_answer.strip().rstrip(".!").lower()
        _STATUS_PHRASES = {
            "code execution complete", "execution complete",
            "done", "complete", "completed", "implemented",
            "implementation complete", "finished", "success",
            "task complete", "task completed", "code complete",
            # Template placeholder echoes — model copied the example
            # instead of substituting the actual value
            "answer", "your answer", "your_answer",
            "your answer here", "your_answer_here",
            "result", "the answer", "the result",
        }
        # Keyword detection: catch longer status messages that aren't in the
        # exact set (e.g. "Function implemented and tested successfully").
        # Only flag if the answer has NO code-like content.
        _STATUS_KEYWORDS = {"implemented", "completed", "successfully", "finished", "executed"}
        _CODE_MARKERS = {"def ", "class ", "import ", "return ", "print(", "for ", "while ", "if ", "= "}
        _has_status_kw = any(kw in _fa for kw in _STATUS_KEYWORDS)
        _has_code = any(m in result.final_answer for m in _CODE_MARKERS)
        if _fa in _STATUS_PHRASES or (_has_status_kw and not _has_code and len(_fa.split()) < 12):
            log.info("Status-message FINAL rejected (turn %d): %r", state.turns, result.final_answer)
            nudge = (
                f'FINAL("{result.final_answer}") is a status message, not an answer. '
                "FINAL must contain the actual answer or complete program text. "
                "If the task asks for code, call FINAL(your_code_as_string)."
            )
            return "", nudge, False, {}

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


def _check_approval_gate(
    ctx: Ctx,
    from_role: str,
    to_role: str,
    reason: str,
) -> bool:
    """Check approval gate before escalation. Returns True if approved."""
    from src.features import features as _get_features

    if not _get_features().approval_gates:
        return True

    from src.graph.approval_gate import request_approval_for_escalation, ApprovalDecision

    decision = request_approval_for_escalation(ctx, from_role, to_role, reason)
    return decision == ApprovalDecision.APPROVE


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

    Escalates to CoderEscalationNode on failure (port 8081, different model).
    """

    async def run(
        self, ctx: Ctx
    ) -> Union["FrontdoorNode", "CoderEscalationNode", "WorkerNode", End[TaskResult]]:
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
                state.record_role(Role.CODER_ESCALATION)
                _log_escalation(ctx, from_role, str(Role.CODER_ESCALATION), f"Early abort: {error[:100]}")
                return CoderEscalationNode()

            if _should_escalate(ctx, error_cat, Role.CODER_ESCALATION):
                state.escalation_count += 1
                state.consecutive_failures = 0
                from_role = str(state.current_role)
                state.record_role(Role.CODER_ESCALATION)
                _log_escalation(
                    ctx, from_role, str(Role.CODER_ESCALATION),
                    f"Escalating after {state.consecutive_failures} failures",
                )
                return CoderEscalationNode()

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

"""Architect delegation for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: architect decision parsing (TOON/JSON/text),
multi-loop delegation pipeline where architect formulates
investigation briefs and specialists execute via ReAct or REPL.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, TYPE_CHECKING

import hashlib as _hashlib
import threading

from src.constants import (
    DELEGATION_BRIEF_KEY_LEN,
    DELEGATION_MAX_SAME_TARGET,
    DELEGATION_MAX_TOTAL_TOKENS,
)
from src.delegation_reports import store_report
from src.exceptions import InferenceError
from src.repl_environment import REPLEnvironment

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives


# Valid delegation targets for architect briefs
_VALID_DELEGATE_ROLES = frozenset(
    {
        "coder_escalation",
        "worker_coder",
        "worker_summarize",
        "worker_explore",
        "worker_general",
        "worker_math",
        "worker_vision",
        "vision_escalation",
    }
)

# Thread-local delegation depth counter to detect re-entrance
# (specialist escalating back to architect_coding starts a fresh loop counter)
_delegation_local = threading.local()


def _get_delegation_depth() -> int:
    return getattr(_delegation_local, "depth", 0)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Invalid %s=%r, using default %d", name, raw, default)
        return default


def _delegation_specialist_turn_token_cap(
    delegate_mode: str,
    question: str,
    brief: str,
    delegate_to: str,
) -> int:
    """Task-aware specialist turn cap to reduce over-generation latency."""
    base = _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_TURN_N_TOKENS", 256)
    q = f"{question}\n{brief}".lower()
    summary_signals = ("summarize", "summary", "extract key", "bullet")
    coding_signals = (
        "implement", "write code", "class ", "function", "refactor", "patch",
        "multi-file", "api", "middleware", "algorithm",
    )
    if delegate_to == "worker_summarize" or any(s in q for s in summary_signals):
        base = min(base, _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_TURN_N_TOKENS_SUMMARY", 192))
    elif any(s in q for s in coding_signals):
        base = max(base, _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_TURN_N_TOKENS_CODE", 320))
    else:
        base = min(base, _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_TURN_N_TOKENS_DEFAULT", 224))
    return max(96, base)


def _delegation_forced_synthesis_token_cap() -> int:
    """Bound cap-synthesis generation so loop recovery cannot stall."""
    cap = _env_int("ORCHESTRATOR_DELEGATION_FORCED_SYNTHESIS_N_TOKENS", 128)
    return max(64, cap)


def _delegation_specialist_max_turns(delegate_mode: str) -> int:
    """Max specialist REPL turns per delegation round."""
    if delegate_mode == "react":
        return max(1, _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_MAX_TURNS_REACT", 3))
    return max(1, _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_MAX_TURNS_REPL", 4))


def _delegation_specialist_max_seconds() -> float:
    """Wall-clock budget for one specialist delegation round."""
    return float(max(10, _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_MAX_SECONDS", 45)))


def _delegation_total_max_seconds() -> float:
    """End-to-end wall-clock budget for architect delegated flow."""
    return float(max(20, _env_int("ORCHESTRATOR_DELEGATION_TOTAL_MAX_SECONDS", 110)))


def _skip_synthesis_on_timeout() -> bool:
    """Return latest specialist report directly when timeout guards fire."""
    raw = os.environ.get("ORCHESTRATOR_DELEGATION_SKIP_SYNTHESIS_ON_TIMEOUT", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _delegation_trace_enabled() -> bool:
    raw = os.environ.get("ORCHESTRATOR_DELEGATION_TRACE", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _delegation_summarize_enabled() -> bool:
    raw = os.environ.get("ORCHESTRATOR_DELEGATION_SUMMARIZE_LONG_REPORTS", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _delegation_summarize_threshold_chars() -> int:
    return max(800, _env_int("ORCHESTRATOR_DELEGATION_SUMMARIZE_REPORT_CHARS", 2800))


def _delegation_summarize_n_tokens() -> int:
    return max(96, _env_int("ORCHESTRATOR_DELEGATION_SUMMARIZE_N_TOKENS", 220))


def _delegation_specialist_question_chars() -> int:
    return max(600, _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_QUESTION_CHARS", 2200))


def _delegation_specialist_brief_chars() -> int:
    return max(240, _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_BRIEF_CHARS", 700))


def _delegation_specialist_context_chars() -> int:
    return max(0, _env_int("ORCHESTRATOR_DELEGATION_SPECIALIST_CONTEXT_CHARS", 800))


def _delegation_specialist_use_corpus_context() -> bool:
    raw = os.environ.get("ORCHESTRATOR_DELEGATION_SPECIALIST_CORPUS_CONTEXT", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _delegation_compact_specialist_prompt_enabled() -> bool:
    raw = os.environ.get("ORCHESTRATOR_DELEGATION_COMPACT_SPECIALIST_PROMPT", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _trim_block(text: str, max_chars: int) -> str:
    body = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(body) <= max_chars:
        return body
    return body[:max_chars].rstrip() + "..."


def _delegation_report_handle_enabled() -> bool:
    raw = os.environ.get("ORCHESTRATOR_DELEGATION_REPORT_HANDLES", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _delegation_report_handle_threshold_chars() -> int:
    return max(1200, _env_int("ORCHESTRATOR_DELEGATION_REPORT_HANDLE_CHARS", 2600))


def _store_report_handle(report: str, delegate_to: str) -> dict[str, str] | None:
    text = (report or "").strip()
    if not text:
        return None
    try:
        return store_report(text, delegate_to)
    except Exception as exc:
        if _delegation_trace_enabled():
            log.warning("Failed to persist delegation report handle: %s", exc)
        return None


def _to_report_handle_text(handle: dict[str, str], summary: str) -> str:
    return (
        f"[REPORT_HANDLE id={handle.get('id')} chars={handle.get('chars')} "
        f"sha16={handle.get('sha16')}]\n"
        f"Use fetch_report('{handle.get('id')}') for full content.\n\n"
        f"Summary:\n{(summary or '').strip()}"
    )


def _build_compact_specialist_prompt(
    delegate_to: str,
    question: str,
    brief: str,
    turn: int,
    last_output: str,
    last_error: str,
) -> str:
    """Compact specialist prompt for delegated mode to reduce prefill cost."""
    prompt = (
        f"You are {delegate_to}. Execute the delegated coding task quickly.\n\n"
        f"User question:\n{question}\n\n"
        f"Architect guidance:\n{brief}\n\n"
        "Output Python code only when computation is required. "
        "If you already have a complete implementation/report, output it directly. "
        "If executing in REPL, end with FINAL(answer) when possible.\n"
    )
    if turn > 0 and (last_output or last_error):
        prompt += "\nPrevious turn signals:\n"
        if last_output:
            prompt += f"- output: {_trim_block(last_output, 600)}\n"
        if last_error:
            prompt += f"- error: {_trim_block(last_error, 400)}\n"
    return prompt


def _maybe_summarize_specialist_report(
    report: str,
    question: str,
    primitives: "LLMPrimitives",
    *,
    force: bool = False,
) -> str:
    """Summarize oversized specialist reports via worker_summarize."""
    text = (report or "").strip()
    if not text:
        return report
    if not _delegation_summarize_enabled():
        return report
    if not force and len(text) < _delegation_summarize_threshold_chars():
        return report
    prompt = (
        "Summarize the specialist report for the architect. Keep only actionable "
        "implementation details and final recommendation. Max 12 bullets, no fluff.\n\n"
        f"Question:\n{question[:1200]}\n\n"
        f"Specialist report:\n{text[:12000]}"
    )
    try:
        summarized = primitives.llm_call(
            prompt,
            role="worker_summarize",
            skip_suffix=True,
            n_tokens=_delegation_summarize_n_tokens(),
        )
        summarized = (summarized or "").strip()
        if summarized:
            if _delegation_trace_enabled():
                log.warning(
                    "Delegation summarize: report_chars=%d -> summary_chars=%d",
                    len(text),
                    len(summarized),
                )
            return summarized
    except Exception as exc:
        if _delegation_trace_enabled():
            log.warning("Delegation summarize failed, keeping original report: %s", exc)
    return report


def _compress_report_for_loop(
    report: str,
    question: str,
    primitives: "LLMPrimitives",
    delegate_to: str,
) -> tuple[str, dict[str, str] | None]:
    """Persist long reports and return compact handle+summary text."""
    text = (report or "").strip()
    if not text:
        return report, None
    if not _delegation_report_handle_enabled():
        return _maybe_summarize_specialist_report(text, question, primitives), None
    if len(text) < _delegation_report_handle_threshold_chars():
        return _maybe_summarize_specialist_report(text, question, primitives), None
    handle = _store_report_handle(text, delegate_to)
    if handle is None:
        return _maybe_summarize_specialist_report(text, question, primitives), None
    summary = _maybe_summarize_specialist_report(text, question, primitives, force=True)
    return _to_report_handle_text(handle, summary), handle


def _strip_think(text: str) -> str:
    """Strip complete and incomplete <think> blocks.

    During streaming, models may produce ``<think>I should delegate with
    I|brief:...`` without closing the tag.  The incomplete block must be
    stripped so that deliberation about delegation isn't mistaken for an
    actual TOON decision.
    """
    # 1. Complete blocks
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2. Trailing incomplete block (opened but never closed)
    result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL)
    return result


def _extract_toon_decision(text: str) -> str | None:
    """Extract D|answer or I|brief:...|to:... from anywhere in model output.

    The model often embeds TOON decisions mid-sentence after reasoning:
      "The answer is C. Decision: D|CTo confirm this..."
    This function extracts "D|C" from that mess.

    Strategy:
      1. MCQ shortcut: D| followed by single letter [A-D] not followed by alpha
      2. Own-line: D|... on its own line
      3. General: D| followed by text until newline (take first sentence)
      4. I| delegation patterns
    """
    # Template-echo blocklist: model echoed the placeholder instead of
    # substituting an actual answer.  Return None so the caller falls
    # through to prose rescue or raw-output handling.
    _TEMPLATE_ECHOES = {"answer", "<answer>", "the answer", "your answer"}

    # 1. MCQ: D|X where X is A-D, followed by non-alpha or end
    mcq = re.search(r"D\|([A-D])(?=[^a-zA-Z]|$)", text)
    if mcq:
        return "D|" + mcq.group(1)

    # 1b. D|I| hybrid: architect emits "D|I|brief:...|to:role" — strip
    # the leading D| and treat as delegation.  4+ sightings across batches.
    hybrid = re.search(r"D\|(I\|.+?)(?:\n|$)", text)
    if hybrid:
        return hybrid.group(1).strip()

    # 2. Own line: D|... on its own line
    own_line = re.search(r"^D\|(.+)$", text, re.MULTILINE)
    if own_line:
        val = own_line.group(1).strip()
        if val.lower() in _TEMPLATE_ECHOES:
            return None
        return "D|" + val

    # 3. General D|: take text until period+space, newline, or next D|
    general = re.search(r"D\|(.+?)(?:\.\s|\n|D\||$)", text)
    if general:
        answer = general.group(1).strip().rstrip(".")
        if answer and answer.lower() not in _TEMPLATE_ECHOES:
            return "D|" + answer

    # 4. I| delegation — accept with or without "brief:" prefix.
    # Models sometimes emit I|description|to:role without "brief:".
    invest = re.search(r"I\|(brief:.+?)(?:\n|$)", text, re.IGNORECASE)
    if invest:
        return "I|" + invest.group(1).strip()

    # 4b. Lenient I|: no "brief:" prefix but has "|to:" somewhere
    invest_lenient = re.search(r"I\|(.+?\|to:\w+)", text)
    if invest_lenient:
        raw_val = invest_lenient.group(1).strip()
        # Normalize: prepend "brief:" if missing
        if not raw_val.lower().startswith("brief:"):
            raw_val = "brief:" + raw_val
        return "I|" + raw_val

    return None


def _parse_architect_decision(response: str) -> dict:
    """Parse architect's TOON-encoded decision.

    Handles:
    - TOON direct: ``D|<answer>``
    - TOON investigate: ``I|brief:<text>|to:<role>`` (default mode=react)
    - TOON investigate+mode: ``I|brief:<text>|to:<role>|mode:repl``
    - JSON: ``{"mode":"direct","answer":"..."}`` or ``{"mode":"investigate",...}``
    - Markdown-wrapped JSON: ```json {...} ```
    - Bare text fallback: treated as direct answer

    Args:
        response: Raw architect response text.

    Returns:
        Dict with keys: mode ("direct"/"investigate"), answer, brief,
        delegate_to, delegate_mode ("react"/"repl").
    """
    text = response.strip()

    # ── TOON: D|<answer> ──
    if text.startswith("D|"):
        raw_answer = text[2:].strip()
        # Guard: model emitted D| then started reasoning instead of answering.
        # If the "answer" is suspiciously long, try to rescue an MCQ letter
        # from the first line or from the reasoning body.
        if len(raw_answer) > 50:
            # Try MCQ letter on the same line as D|
            first_line = raw_answer.split("\n", 1)[0].strip()
            mcq_match = re.match(r"^([A-D])(?:[^a-zA-Z]|$)", first_line)
            if mcq_match:
                raw_answer = mcq_match.group(1)
            else:
                # Try to find a clear MCQ answer in the reasoning.
                # Patterns (checked in priority order):
                #   "Answer: B", "Correct Answer: B"
                #   "answer is A", "answer would be A", "answer should be A"
                #   "option A seems", "option A with"
                rescue = re.search(
                    r"(?:the\s+)?(?:correct\s+)?answer\s*(?:is|would\s+be|should\s+be|:)\s*([A-D])(?=[^a-zA-Z]|$)",
                    raw_answer,
                    re.IGNORECASE,
                )
                if not rescue:
                    rescue = re.search(
                        r"\boption\s+([A-D])(?:\s+(?:seems|with|is|looks)\b|[^a-zA-Z]|$)",
                        raw_answer,
                        re.IGNORECASE,
                    )
                if not rescue:
                    # Last resort: find the last D|X (MCQ) in the reasoning.
                    # Handles models that emit empty D| first, then reason,
                    # then conclude with D|B at the end.
                    last_toon = list(re.finditer(
                        r"D\|([A-D])(?=[^a-zA-Z]|$)", raw_answer
                    ))
                    if last_toon:
                        rescue = last_toon[-1]
                if rescue:
                    raw_answer = rescue.group(1).upper()
                # else: keep raw_answer as-is (best effort)
        return {
            "mode": "direct",
            "answer": raw_answer,
            "brief": "",
            "delegate_to": "",
            "delegate_mode": "react",
        }

    # ── TOON: I|brief:...|to:...[|mode:...] ──
    if text.startswith("I|"):
        parts_str = text[2:]
        fields: dict[str, str] = {}
        for segment in parts_str.split("|"):
            if ":" in segment:
                key, _, val = segment.partition(":")
                fields[key.strip().lower()] = val.strip()

        brief = fields.get("brief", parts_str)
        delegate_to = fields.get("to", "coder_escalation")
        delegate_mode = fields.get("mode", "react")

        # Clamp to valid role
        if delegate_to not in _VALID_DELEGATE_ROLES:
            delegate_to = "coder_escalation"
        # Clamp to valid mode
        if delegate_mode not in ("react", "repl"):
            delegate_mode = "react"

        return {
            "mode": "investigate",
            "answer": "",
            "brief": brief,
            "delegate_to": delegate_to,
            "delegate_mode": delegate_mode,
        }

    # ── JSON (possibly markdown-wrapped) ──
    import re as _re

    json_match = _re.search(r"```(?:json)?\s*\n?(.*?)```", text, _re.DOTALL)
    json_text = json_match.group(1).strip() if json_match else text

    # Try JSON parse
    try:
        obj = json.loads(json_text)
        if isinstance(obj, dict):
            mode = obj.get("mode", "direct")
            if mode == "investigate":
                delegate_to = obj.get("delegate_to", obj.get("to", "coder_escalation"))
                if delegate_to not in _VALID_DELEGATE_ROLES:
                    delegate_to = "coder_escalation"
                delegate_mode = obj.get("delegate_mode", obj.get("mode_detail", "react"))
                if delegate_mode not in ("react", "repl"):
                    delegate_mode = "react"
                return {
                    "mode": "investigate",
                    "answer": "",
                    "brief": obj.get("brief", ""),
                    "delegate_to": delegate_to,
                    "delegate_mode": delegate_mode,
                }
            return {
                "mode": "direct",
                "answer": obj.get("answer", json_text),
                "brief": "",
                "delegate_to": "",
                "delegate_mode": "react",
            }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # ── Bare text fallback — treat as direct answer ──
    return {
        "mode": "direct",
        "answer": text,
        "brief": "",
        "delegate_to": "",
        "delegate_mode": "react",
    }


# Full budget for computation turns (code execution in mini-REPL)
_ARCHITECT_TOKEN_BUDGET: dict[str, int] = {
    "architect_general": 768,
    "architect_coding": 512,
}

# Tight budget for the routing decision (D|answer or I|brief:...|to:role).
# architect_coding uses <think> tags so 500 visible tokens suffices.
# architect_general (Qwen3-235B) reasons in plain text, exhausting 500 tokens
# before emitting D|.  Give it 1500 so ~1000 goes to reasoning + 500 to answer.
_ARCHITECT_DECISION_BUDGET: dict[str, int] = {
    "architect_general": 512,
    "architect_coding": 192,
}


def _architect_decision_token_budget(role: str) -> int:
    """Token budget for architect routing decision (turn 0)."""
    default = _ARCHITECT_DECISION_BUDGET.get(role, 256)
    return max(64, _env_int("ORCHESTRATOR_DELEGATION_ARCHITECT_DECISION_N_TOKENS", default))


def _architect_compute_token_budget(role: str) -> int:
    """Token budget for architect computation follow-up turns."""
    default = _ARCHITECT_TOKEN_BUDGET.get(role, 512)
    return max(128, _env_int("ORCHESTRATOR_DELEGATION_ARCHITECT_COMPUTE_N_TOKENS", default))


def _classify_failure_reason(exc: Exception) -> str:
    """Map inference failure text to a stable delegated break_reason."""
    text = str(exc).lower()
    if "lock timeout" in text:
        return "pre_delegation_lock_timeout"
    if "deadline exceeded" in text:
        return "deadline_exceeded"
    if "cancelled" in text or "canceled" in text:
        return "request_cancelled"
    if "timed out" in text or "timeout" in text:
        return "request_timeout"
    return "pre_delegation_architect_error"


def _run_architect_decision(
    prompt_text: str,
    question: str,
    primitives: "LLMPrimitives",
    architect_role: str,
    tool_registry: "Any | None",
) -> tuple[str | None, int, int]:
    """Run architect's mini-REPL decision loop (Phase A).

    Returns:
        (arch_response, computation_turns, tool_invocations, failure_reason)
    """
    from src.graph.helpers import _extract_prose_answer
    from src.prompt_builders import extract_code_from_response

    # Mini-REPL loop: architect can compute before deciding
    arch_repl = REPLEnvironment(
        context=question,
        llm_primitives=primitives,
        tool_registry=tool_registry,
        role=architect_role,
    )
    arch_response = None
    arch_last_output = ""
    raw = ""

    for _aturn in range(4):  # Max 4 computation turns
        if _aturn == 0:
            full_prompt = prompt_text
        else:
            full_prompt = (
                f"{prompt_text}\n\n"
                f"Computation result:\n{arch_last_output}\n\n"
                f"Now make your decision (D|answer or I|brief:...|to:role):"
            )
        # Turn 0 = routing decision (tight budget).
        # Turns 1+ = computation follow-up (full budget).
        n_tok = (
            _architect_decision_token_budget(architect_role)
            if _aturn == 0
            else _architect_compute_token_budget(architect_role)
        )
        # Early-stop streaming: abort generation once a complete TOON
        # decision line is detected.  Saves 3000+ tokens of post-decision
        # rambling on the architect model.
        # Uses _strip_think to also handle incomplete <think> blocks so
        # that deliberation *about* delegating isn't mistaken for a real
        # TOON decision.
        # IMPORTANT: Free-form D| answers must wait for a trailing newline
        # before firing.  The old pattern `D\|.+` fired after a single
        # token (e.g. "D|The" → answer="The"), truncating multi-word
        # answers.  MCQ answers (D|A through D|D) still fire immediately
        # via the lookahead shortcut.
        _toon_re = re.compile(r"D\|[A-D](?=[^a-zA-Z]|$)|D\|[^\n]+\n|I\|.+\|to:\w+")
        primitives._early_stop_check = lambda text: bool(
            _toon_re.search(_strip_think(text))
        )
        try:
            raw = primitives.llm_call(
                full_prompt,
                role=architect_role,
                skip_suffix=True,
                n_tokens=n_tok,
            )
        except (InferenceError, ConnectionError, TimeoutError, OSError) as e:
            log.warning(f"Architect call failed (turn {_aturn}): {e}")
            return None, _aturn + 1, arch_repl._tool_invocations, _classify_failure_reason(e)
        except Exception as e:
            log.warning(f"Architect call failed unexpectedly (turn {_aturn}): {e}")
            return None, _aturn + 1, arch_repl._tool_invocations, _classify_failure_reason(e)
        finally:
            primitives._early_stop_check = None

        # Strip <think>...</think> tags (reasoning models), including
        # incomplete trailing blocks so mid-thought I|/D| aren't parsed.
        stripped = _strip_think(raw).strip()
        # llm_call may return "[ERROR: ...]" strings instead of raising.
        # Treat these as Phase-A failures so break_reason is populated.
        if stripped.startswith("[ERROR:"):
            return (
                None,
                _aturn + 1,
                arch_repl._tool_invocations,
                _classify_failure_reason(RuntimeError(stripped)),
            )
        # Extract TOON decision from anywhere in the response.
        arch_response = _extract_toon_decision(stripped)
        if arch_response:
            break
        # Check start-of-response for JSON (fallback TOON format).
        # Require minimum length to avoid accepting truncated fragments
        # like '{"' that the model emits when confused by the prompt.
        if (stripped.startswith("{") or stripped.startswith("```")) and len(stripped) > 5:
            arch_response = stripped
            break

        # Turn 0 must be a routing decision (D| or I|).  If the
        # architect wrote prose or code instead, treat it as a direct
        # answer.  DO NOT enter the computation loop — the architect
        # models are slow (6-10 t/s) and 3 computation turns of 3375-
        # 5150 tokens each can burn 10-26 minutes on a single question.
        if _aturn == 0:
            # Prose answer rescue: extract from "The answer is X"
            prose = _extract_prose_answer(stripped)
            if prose:
                arch_response = f"D|{prose}"
                log.info("Architect turn 0: no TOON, prose rescue -> D|%s", prose[:50])
            else:
                # Avoid double-wrapping: if stripped already starts with
                # D| or I| (but _extract_toon_decision couldn't parse it,
                # e.g. bare "D|\n<reasoning>"), pass it through directly
                # so _parse_architect_decision's long-answer rescue can
                # extract the MCQ letter from the reasoning body.
                if stripped and re.match(r"^[DI]\|", stripped):
                    arch_response = stripped
                else:
                    arch_response = f"D|{stripped}" if stripped else raw
                log.info("Architect turn 0: no TOON, treating raw output as D|answer (%d chars)", len(stripped or raw))
            break

        # Turns 1+: treat as Python code — execute in mini-REPL
        code = extract_code_from_response(raw)
        if not code.strip():
            # No code extracted, treat raw output as direct answer
            arch_response = raw
            break
        # Comment-only guard: architect is reasoning in comments, not computing
        if all(
            not ln.strip() or ln.strip().startswith("#")
            for ln in code.split("\n")
        ):
            log.info("Architect computation turn %d: comment-only, treating as direct answer", _aturn)
            arch_response = raw
            break
        result = arch_repl.execute(code)
        if result.is_final:
            arch_response = result.final_answer or ""
            break
        arch_last_output = result.output or result.error or ""
        log.info(f"Architect computation turn {_aturn}: {len(arch_last_output)} chars output")
    else:
        # Exhausted computation turns, use last raw response
        arch_response = raw if raw else f"D|{arch_last_output}"

    return arch_response, _aturn + 1, arch_repl._tool_invocations, None


def _apply_decision_guards(
    decision: dict,
    question: str,
    loop: int,
    primitives: "LLMPrimitives",
    architect_role: str,
) -> dict:
    """Apply guard clauses to architect decision (MCQ misroute, short-answer, coding task).

    Returns:
        Potentially modified decision dict.
    """
    # ── MCQ misroute guard ──
    # If the question is multiple-choice (has A/B/C/D options) and the
    # architect tries to delegate, force a direct answer.  Specialists
    # cannot reason about factual/science MCQ — delegation just wastes
    # 50-300s and usually returns a wrong answer.
    if decision["mode"] == "investigate" and loop == 0:
        _mcq_re = re.compile(
            r"(?:^|\n)\s*[A-D]\s*[).\]]",  # A) or A. or A]
            re.MULTILINE,
        )
        if _mcq_re.search(question):
            log.warning(
                "MCQ misroute blocked: architect tried to delegate factual MCQ "
                "(brief=%s), forcing direct answer",
                decision["brief"][:80],
            )
            # Re-prompt the architect with a forced direct-answer instruction
            force_prompt = (
                f"This is a multiple-choice question. You MUST answer directly.\n"
                f"Respond with D| followed by the letter (A, B, C, or D). No delegation.\n"
                f"Do NOT explain your reasoning. Output ONLY the decision line.\n\n"
                f"Question: {question[:2000]}\n\n"
                f"Answer with the letter only (A, B, C, or D).\n\nDecision:"
            )
            try:
                forced_raw = primitives.llm_call(
                    force_prompt,
                    role=architect_role,
                    skip_suffix=True,
                    n_tokens=128,
                )
                forced_stripped = _strip_think(forced_raw).strip()
                forced_decision = _extract_toon_decision(forced_stripped)
                if forced_decision and forced_decision.startswith("D|"):
                    decision = _parse_architect_decision(forced_decision)
                    log.info("MCQ misroute recovered: architect answered D|%s", decision["answer"])
                else:
                    # Last resort: extract any single letter A-D.
                    # Strip D|/I| prefix first to avoid matching the
                    # protocol marker as an MCQ letter.
                    _cleaned = re.sub(r"^[DI]\|", "", forced_stripped).strip()
                    letter_match = re.search(r"\b([A-D])\b", _cleaned)
                    if letter_match:
                        decision = {"mode": "direct", "answer": letter_match.group(1),
                                    "brief": "", "delegate_to": "", "delegate_mode": "react"}
                        log.info("MCQ misroute recovered (letter extract): D|%s", decision["answer"])
            except Exception as exc:
                log.warning("MCQ misroute re-prompt failed: %s", exc)

    # ── Short-answer delegation guard ──
    # If the architect wants to delegate but the brief is essentially a
    # computed answer (short, numeric, or a factual statement), force
    # direct answer.  This catches: architect solves "soda bottle costs
    # $1.50" in <think>, then delegates "compute the cost" to coder who
    # has nothing to add.  The coder burns 50-300s round-tripping the
    # answer the architect already has.
    if decision["mode"] == "investigate" and loop == 0:
        brief = decision["brief"]
        _code_delegate = decision["delegate_to"] == "coder_escalation"
        _code_signals_in_q = any(
            sig in question for sig in (
                "INPUT FORMAT", "OUTPUT FORMAT", "SAMPLE INPUT",
                "USACO", "Codeforces", "Write a Python", "def ",
                "```python",
            )
        )
        # If delegating to coder but the question is NOT a coding task,
        # the architect is misrouting a factual/math question.
        if _code_delegate and not _code_signals_in_q:
            # Check if the brief looks like a computed answer rather
            # than a genuine implementation task.
            brief_words = brief.split()
            brief_is_short = len(brief_words) < 15
            brief_has_number = bool(re.search(r"\d+\.?\d*", brief))
            if brief_is_short and brief_has_number:
                log.warning(
                    "Short-answer delegation blocked: architect delegated "
                    "D|%s to %s for non-code question, forcing direct. "
                    "Brief: %s",
                    brief[:30],
                    decision["delegate_to"],
                    brief[:80],
                )
                # Extract the numeric answer from the brief
                number_match = re.search(r"[\d]+\.?\d*", brief)
                forced_answer = number_match.group(0) if number_match else brief
                decision = {
                    "mode": "direct",
                    "answer": forced_answer,
                    "brief": "",
                    "delegate_to": "",
                    "delegate_mode": "react",
                }

    # ── Coding task direct-answer guard ──
    # If the question asks for code (CP, LeetCode, implementation tasks)
    # and the architect gives a short direct answer instead of delegating,
    # force delegation to coder.  The scorer expects runnable code, not a
    # numeric value like "4" or "-1".
    if decision["mode"] == "direct" and loop == 0:
        _code_signals = (
            "INPUT FORMAT", "OUTPUT FORMAT", "SAMPLE INPUT", "SAMPLE OUTPUT",
            "reads from stdin", "writes to stdout", "USACO", "Codeforces",
            "Write a Python solution",
            "Write a Python function",
            "def ", "```python",
            "Include proper type hints",
            "handle edge cases",
        )
        if any(sig in question for sig in _code_signals):
            short_answer = decision["answer"].strip()
            # Only intercept short answers (not full programs)
            if len(short_answer) < 50 and not short_answer.startswith(
                ("import", "def ", "class ")
            ):
                log.warning(
                    "Code direct-answer blocked: architect answered D|%s for coding "
                    "question, forcing delegation to coder_escalation",
                    short_answer[:30],
                )
                # Don't leak the architect's numeric guess to the
                # coder — it causes hardcoded FINAL(N) instead of
                # a general solution.
                hint = "" if re.fullmatch(r"-?\d+\.?\d*", short_answer.strip()) else f" {short_answer}"
                decision = {
                    "mode": "investigate",
                    "answer": "",
                    "brief": f"Implement a complete Python solution that reads from stdin and writes to stdout.{hint}",
                    "delegate_to": "coder_escalation",
                    "delegate_mode": "repl",
                }

    return decision


def _run_specialist_loop(
    question: str,
    context: str,
    brief: str,
    delegate_to: str,
    delegate_mode: str,
    primitives: "LLMPrimitives",
    tool_registry: "Any | None",
    time_budget_s: float | None = None,
) -> tuple[str, int, list[str], list[dict], bool, bool, dict[str, Any]]:
    """Run specialist delegation execution loop (Phase B).

    Returns:
        (report, tool_invocations, tools_called, tool_timings)
    """
    from src.prompt_builders import (
        build_root_lm_prompt,
        extract_code_from_response,
        auto_wrap_final,
    )
    from src.prompt_builders.builder import build_corpus_context
    import hashlib

    tools_called: list[str] = []
    phase_tool_timings: list[dict] = []
    timed_out = False
    report_rescued = False
    infer_meta_last: dict[str, Any] = {}

    # Both react and repl modes use the same REPL loop
    # (react = structured_mode=True, repl = structured_mode=False)
    structured = delegate_mode == "react"
    max_delegate_turns = _delegation_specialist_max_turns(delegate_mode)
    specialist_turn_cap = _delegation_specialist_turn_token_cap(
        delegate_mode=delegate_mode,
        question=question,
        brief=brief,
        delegate_to=delegate_to,
    )
    specialist_time_budget_s = (
        min(_delegation_specialist_max_seconds(), float(time_budget_s))
        if time_budget_s is not None
        else _delegation_specialist_max_seconds()
    )
    q_for_specialist = _trim_block(question, _delegation_specialist_question_chars())
    brief_for_specialist = _trim_block(brief, _delegation_specialist_brief_chars())
    ctx_for_specialist = _trim_block(context, _delegation_specialist_context_chars())
    specialist_started = time.perf_counter()
    try:
        deleg_repl = REPLEnvironment(
            context=(
                f"{brief_for_specialist}\n\nContext:\n{ctx_for_specialist}"
                if ctx_for_specialist
                else brief_for_specialist
            ),
            llm_primitives=primitives,
            tool_registry=tool_registry,
            role=delegate_to,
            structured_mode=structured,
        )
        deleg_last_output = ""
        deleg_last_error = ""
        _prev_code_hash = ""  # dedup guard
        # Build specialist task: full question + architect's brief as
        # design guidance.  Previously only the brief was passed, so the
        # specialist never saw the actual problem (constraints, examples,
        # method signatures).
        specialist_task = (
            f"{q_for_specialist}\n\n"
            f"## Architect guidance\n{brief_for_specialist}"
        )
        # Retrieve corpus context once for the delegation (turn 0 only)
        corpus_ctx = (
            build_corpus_context(role=delegate_to, task_description=q_for_specialist)
            if _delegation_specialist_use_corpus_context()
            else ""
        )

        for _turn in range(max_delegate_turns):
            elapsed_s = time.perf_counter() - specialist_started
            if elapsed_s >= specialist_time_budget_s:
                log.warning(
                    "Specialist loop timeout: role=%s mode=%s turns=%d elapsed=%.1fs budget=%.1fs",
                    delegate_to, delegate_mode, _turn, elapsed_s, specialist_time_budget_s,
                )
                timed_out = True
                report = (deleg_last_output or "").strip()
                if report.lower() in {"", "[no output]", "observation: [no output]"}:
                    report = (
                        f"[Delegation timeout after {_turn} turn(s), {elapsed_s:.1f}s. "
                        f"Specialist role={delegate_to}, mode={delegate_mode}. "
                        f"Brief={brief[:160]}]"
                    )
                break
            if _delegation_compact_specialist_prompt_enabled():
                deleg_prompt = _build_compact_specialist_prompt(
                    delegate_to=delegate_to,
                    question=q_for_specialist,
                    brief=brief_for_specialist,
                    turn=_turn,
                    last_output=deleg_last_output,
                    last_error=deleg_last_error,
                )
            else:
                repl_state = deleg_repl.get_state()
                deleg_prompt = build_root_lm_prompt(
                    state=repl_state,
                    original_prompt=specialist_task,
                    last_output=deleg_last_output,
                    last_error=deleg_last_error,
                    turn=_turn,
                    corpus_context=corpus_ctx if _turn == 0 else "",
                )
            _final_re = re.compile(
                r"""FINAL\(\s*(?:'{3}(.+?)'{3}|"{3}(.+?)"{3}|["'](.+?)["']|(\S+?))\s*\)""",
                re.DOTALL,
            )
            primitives._early_stop_check = lambda text: bool(_final_re.search(text))
            llm_started = time.perf_counter()
            try:
                code = primitives.llm_call(
                    deleg_prompt,
                    role=delegate_to,
                    stop_sequences=["\n```\n"],
                    n_tokens=specialist_turn_cap,
                )
            finally:
                primitives._early_stop_check = None
            llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000
            infer_meta_last = dict(getattr(primitives, "_last_inference_meta", {}) or {})
            if infer_meta_last:
                infer_meta_last["llm_elapsed_ms"] = round(llm_elapsed_ms, 1)
            raw_deleg_output = code
            if _delegation_trace_enabled():
                log.warning(
                    "Delegation trace turn=%d role=%s mode=%s prompt_chars=%d raw_chars=%d llm_ms=%.1f infer=%s",
                    _turn,
                    delegate_to,
                    delegate_mode,
                    len(deleg_prompt),
                    len(raw_deleg_output or ""),
                    llm_elapsed_ms,
                    infer_meta_last,
                )
            code = extract_code_from_response(code)
            code = auto_wrap_final(code)
            if _delegation_trace_enabled():
                log.warning(
                    "Delegation trace turn=%d extracted_code_chars=%d has_final=%s",
                    _turn,
                    len(code or ""),
                    "FINAL(" in (code or ""),
                )
            # Dedup guard: if coder generates identical code twice in a
            # row, inject an error to break the loop instead of wasting
            # another turn on the same silent execution.
            _code_hash = hashlib.sha256(code.encode()).hexdigest()
            if _code_hash == _prev_code_hash:
                deleg_last_error = (
                    "You generated the exact same code as last turn. "
                    "It didn't work because FINAL() was never reached at top level. "
                    "Do NOT wrap code in main(). Write flat top-level code ending with FINAL(answer)."
                )
                deleg_last_output = ""
                _prev_code_hash = _code_hash
                continue
            _prev_code_hash = _code_hash
            # Prose report rescue: specialist answered in prose without
            # code or FINAL().  In delegation, the specialist's prose IS
            # the investigation report — return it to the architect for
            # synthesis instead of discarding it or looping.
            if "FINAL(" not in code and not any(
                ln.strip() and not ln.strip().startswith("#")
                and any(ln.strip().startswith(kw) for kw in (
                    "def ", "class ", "for ", "while ", "if ", "try:",
                    "print(", "result =", "answer =",
                ))
                for ln in code.split("\n")
            ):
                # No executable code found — treat raw prose as report
                report = raw_deleg_output.strip()
                if report:
                    report_rescued = True
                    log.info(
                        "Delegation prose report (turn %d): %d chars returned to architect",
                        _turn, len(report),
                    )
                    break
            # Code-report rescue: specialist returned substantial code/design
            # text but omitted FINAL(). For delegation we need a report, not
            # necessarily an executed terminal value. Accept it directly to
            # avoid repeated 30s generation loops that only attempt FINAL().
            if "FINAL(" not in code:
                non_comment_lines = [
                    ln for ln in code.split("\n")
                    if ln.strip() and not ln.strip().startswith("#")
                ]
                if len(non_comment_lines) >= 6 or len(code.strip()) >= 400:
                    report = raw_deleg_output.strip() or code.strip()
                    if report:
                        report_rescued = True
                        log.info(
                            "Delegation code report rescue (turn %d): %d chars returned to architect",
                            _turn, len(report),
                        )
                        break
            # Comment-only guard
            if all(
                not ln.strip() or ln.strip().startswith("#")
                for ln in code.split("\n")
            ):
                deleg_last_error = (
                    "Your output was all comments — no executable code ran. "
                    "Write Python code that computes the answer and call FINAL(answer)."
                )
                deleg_last_output = ""
                continue
            exec_started = time.perf_counter()
            result = deleg_repl.execute(code)
            exec_elapsed_ms = (time.perf_counter() - exec_started) * 1000
            if _delegation_trace_enabled():
                log.warning(
                    "Delegation trace turn=%d exec_ms=%.1f is_final=%s output_chars=%d error_chars=%d",
                    _turn,
                    exec_elapsed_ms,
                    bool(getattr(result, "is_final", False)),
                    len((getattr(result, "output", "") or "")),
                    len((getattr(result, "error", "") or "")),
                )
            if result.is_final:
                report = result.final_answer or ""
                # Specialist already reached a terminal answer; return directly
                # instead of spending another architect synthesis turn.
                report_rescued = True
                break
            deleg_last_output = result.output or ""
            deleg_last_error = result.error or ""
        else:
            log.warning(
                "Specialist loop turn cap reached: role=%s mode=%s turns=%d",
                delegate_to, delegate_mode, max_delegate_turns,
            )
            report = deleg_repl.get_state()
        if deleg_repl.tool_registry:
            for inv in deleg_repl.tool_registry.get_invocation_log():
                tools_called.append(inv.tool_name)
                phase_tool_timings.append(
                    {"tool_name": inv.tool_name, "elapsed_ms": inv.elapsed_ms, "success": inv.success}
                )
    except (InferenceError, ConnectionError, TimeoutError, OSError) as e:
        report = f"[Delegation failed: {e}]"
        err_text = str(e).lower()
        timed_out = any(
            marker in err_text
            for marker in ("timeout", "timed out", "deadline", "lock timeout", "cancel")
        )
        return report, 0, tools_called, phase_tool_timings, timed_out, report_rescued, infer_meta_last
    except Exception as e:
        report = f"[Delegation failed (unexpected): {e}]"
        return report, 0, tools_called, phase_tool_timings, timed_out, report_rescued, infer_meta_last

    return (
        report,
        deleg_repl._tool_invocations,
        tools_called,
        phase_tool_timings,
        timed_out,
        report_rescued,
        infer_meta_last,
    )


def _architect_delegated_answer(
    question: str,
    context: str,
    primitives: "LLMPrimitives",
    state: "Any",
    architect_role: str = "architect_general",
    max_loops: int = 3,
    force_response_on_cap: bool = True,
) -> tuple[str, dict]:
    """Multi-loop architect delegation: architect decides, specialist investigates.

    Loop flow:
      1. Architect decides (answer directly or delegate)
      2. If delegate: specialist runs ReAct or REPL loop
      3. Architect synthesizes from report (or requests more)
      4. Repeat up to max_loops

    Args:
        question: The user's question.
        context: Optional context text.
        primitives: LLM primitives for inference.
        state: Application state (has tool_registry, etc.).
        architect_role: Architect role to use.
        max_loops: Maximum delegation loops.
        force_response_on_cap: If True, force architect to answer when cap reached.
            If False, return partial with needs_input flag (production mode).

    Returns:
        Tuple of (answer_text, stats_dict).
    """
    total_tools = 0
    all_tools_called: list[str] = []
    stats: dict = {
        "loops": 0,
        "phases": [],
        "specialist_output": "",
        "needs_input": False,
        "tools_used": 0,
        "delegation_events": [],
        "tool_timings": [],
        "cap_reached": False,
        "break_reason": "",
        "effective_max_loops": max_loops,
        "reentrant_depth": 0,
        "report_handles": [],
    }

    # Optional TOON encoding for context
    toon_context = context
    if context:
        try:
            from src.services.toon_encoder import encode, is_available

            if is_available() and len(context) > 100:
                toon_context = encode({"ctx": context[:3000]})
        except Exception as exc:
            log.debug("TOON encoding failed, using raw context: %s", exc)

    tool_registry = getattr(state, "tool_registry", None)
    reports: list[str] = []
    previous_brief_keys: set[str] = set()  # Semantic dedup: hash(brief + delegate_to)
    delegate_history: list[str] = []  # Track delegation targets for repetition guard

    # Re-entrance guard: if specialist escalates back to architect, reduce max_loops
    depth = _get_delegation_depth()
    if depth > 0:
        effective_max_loops = min(max_loops, 1)
        log.warning(
            "Re-entrant delegation detected (depth=%d), reducing max_loops %d -> %d",
            depth, max_loops, effective_max_loops,
        )
    else:
        effective_max_loops = max_loops
    stats["effective_max_loops"] = effective_max_loops
    stats["reentrant_depth"] = depth

    _delegation_local.depth = depth + 1
    try:
        return _architect_delegated_answer_inner(
            question, context, primitives, state, architect_role,
            effective_max_loops, force_response_on_cap, toon_context,
            tool_registry, reports, previous_brief_keys, delegate_history,
            total_tools, all_tools_called, stats,
        )
    finally:
        _delegation_local.depth = depth


def _architect_delegated_answer_inner(
    question: str,
    context: str,
    primitives: "LLMPrimitives",
    state: "Any",
    architect_role: str,
    max_loops: int,
    force_response_on_cap: bool,
    toon_context: str,
    tool_registry: "Any | None",
    reports: list[str],
    previous_brief_keys: set[str],
    delegate_history: list[str],
    total_tools: int,
    all_tools_called: list[str],
    stats: dict,
) -> tuple[str, dict]:
    """Inner loop extracted to keep try/finally clean in the outer function."""
    from src.prompt_builders import (
        build_architect_investigate_prompt,
        build_architect_synthesis_prompt,
    )
    cumulative_delegate_tokens = 0
    orchestration_started = time.perf_counter()
    total_budget_s = _delegation_total_max_seconds()
    stats.setdefault("report_handles", [])

    for loop in range(max_loops + 1):  # +1 for initial decision
        total_elapsed_s = time.perf_counter() - orchestration_started
        if total_elapsed_s >= total_budget_s:
            log.warning(
                "Delegation total timeout: elapsed=%.1fs budget=%.1fs",
                total_elapsed_s, total_budget_s,
            )
            stats["break_reason"] = "wall_clock_budget"
            break
        phase_start = time.perf_counter()

        # ── Phase A: Architect call with computation REPL ──
        if loop == 0:
            prompt_text = build_architect_investigate_prompt(question, toon_context)
        else:
            report = reports[-1] if reports else ""
            prompt_text = build_architect_synthesis_prompt(
                question,
                report,
                loop,
                max_loops,
            )

        decision_result = _run_architect_decision(
            prompt_text, question, primitives, architect_role, tool_registry,
        )
        if len(decision_result) == 3:
            # Backward-compatible path for tests/patches that still mock the legacy tuple.
            arch_response, computation_turns, arch_tools = decision_result
            phase_a_failure_reason = None
        else:
            arch_response, computation_turns, arch_tools, phase_a_failure_reason = decision_result
        total_tools += arch_tools

        if arch_response is None:
            # Early error return
            stats["break_reason"] = phase_a_failure_reason or "pre_delegation_architect_error"
            stats["phases"].append(
                {
                    "loop": loop,
                    "phase": "A",
                    "ms": round((time.perf_counter() - phase_start) * 1000),
                    "decision": "error",
                    "computation_turns": computation_turns,
                }
            )
            if reports:
                return reports[-1], stats
            return "[ERROR: Architect delegation failed]", stats

        phase_a_ms = (time.perf_counter() - phase_start) * 1000
        decision = _parse_architect_decision(arch_response)
        decision_answer = str(decision.get("answer", "") or "").strip()
        if decision["mode"] == "direct" and decision_answer.startswith("[ERROR:"):
            stats["break_reason"] = _classify_failure_reason(RuntimeError(decision_answer))
            stats["phases"].append(
                {
                    "loop": loop,
                    "phase": "A",
                    "ms": round(phase_a_ms),
                    "decision": "error",
                    "computation_turns": computation_turns,
                }
            )
            if reports:
                return reports[-1], stats
            return "[ERROR: Architect delegation failed]", stats
        stats["phases"].append(
            {
                "loop": loop,
                "phase": "A",
                "ms": round(phase_a_ms),
                "decision": decision["mode"],
                "computation_turns": computation_turns,
            }
        )

        log.info(f"Architect loop {loop}: {decision['mode']} ({phase_a_ms:.0f}ms, {computation_turns} turns)")

        decision = _apply_decision_guards(decision, question, loop, primitives, architect_role)

        # ── Direct/final answer ──
        if decision["mode"] == "direct":
            answer = decision["answer"]
            # If architect just says "Approved" and we have specialist output,
            # the specialist's document IS the final answer
            if answer.lower().strip().startswith("approved") and stats["specialist_output"]:
                answer = stats["specialist_output"]
            stats["loops"] = loop
            stats["tools_used"] = max(
                total_tools,
                len(all_tools_called),
                len(stats.get("tool_timings", [])),
            )
            stats["tools_called"] = all_tools_called
            return answer, stats

        # ── Last-loop guard: skip specialist, go to forced synthesis ──
        # Running the specialist on the final iteration wastes hundreds of
        # seconds and the result is thrown away by forced synthesis anyway.
        if loop >= max_loops:
            log.info("Architect still wants to investigate on last loop (%d), forcing synthesis", loop)
            stats["break_reason"] = "max_loops"
            break

        # ── Phase B: Specialist execution ──
        brief = decision["brief"]
        delegate_to = decision["delegate_to"]
        delegate_mode = decision["delegate_mode"]

        # ── Loop guard 1: Semantic dedup ──
        # Hash brief + delegate_to as combined key. Architect rephrasing
        # the same brief to a different target still gets caught because
        # we also check target repetition separately (guard 3).
        brief_key = _hashlib.sha256(
            f"{brief.strip().lower()[:DELEGATION_BRIEF_KEY_LEN]}|{delegate_to}".encode()
        ).hexdigest()
        if brief_key in previous_brief_keys:
            log.warning(
                "Semantic dedup: duplicate brief+target (loop %d, target=%s), forcing synthesis",
                loop, delegate_to,
            )
            stats["break_reason"] = "semantic_dedup"
            break
        previous_brief_keys.add(brief_key)

        # ── Loop guard 2: Max cumulative tokens ──
        if cumulative_delegate_tokens > DELEGATION_MAX_TOTAL_TOKENS:
            log.warning(
                "Token budget exceeded: %d > %d tokens across delegation loops, forcing synthesis",
                cumulative_delegate_tokens, DELEGATION_MAX_TOTAL_TOKENS,
            )
            stats["break_reason"] = "token_budget"
            break

        # ── Loop guard 3: Role repetition ──
        # If architect delegates to the same role N consecutive times, force synthesis.
        delegate_history.append(delegate_to)
        if len(delegate_history) >= DELEGATION_MAX_SAME_TARGET:
            recent = delegate_history[-DELEGATION_MAX_SAME_TARGET:]
            if all(r == delegate_to for r in recent):
                log.warning(
                    "Role repetition guard: %s delegated %d consecutive times, forcing synthesis",
                    delegate_to, DELEGATION_MAX_SAME_TARGET,
                )
                stats["break_reason"] = "role_repetition"
                break

        log.info(f"Delegating to {delegate_to} (mode={delegate_mode}): {brief[:100]}...")

        phase_b_start = time.perf_counter()
        tokens_before = primitives.total_tokens_generated
        remaining_budget_s = total_budget_s - (time.perf_counter() - orchestration_started)
        specialist_budget_s = max(10.0, remaining_budget_s - 5.0)

        report, deleg_tools, deleg_tools_called, phase_tool_timings, specialist_timed_out, report_rescued, specialist_infer_meta = _run_specialist_loop(
            question,
            context,
            brief,
            delegate_to,
            delegate_mode,
            primitives,
            tool_registry,
            time_budget_s=specialist_budget_s,
        )
        total_tools += deleg_tools
        all_tools_called.extend(deleg_tools_called)
        compressed_report, report_handle = _compress_report_for_loop(
            report, question, primitives, delegate_to,
        )
        report = compressed_report
        if report_handle:
            stats["report_handles"].append(report_handle)

        phase_b_ms = (time.perf_counter() - phase_b_start) * 1000
        delegate_tokens = primitives.total_tokens_generated - tokens_before
        cumulative_delegate_tokens += delegate_tokens
        reports.append(report)
        stats["specialist_output"] = report
        stats["phases"].append(
            {
                "loop": loop,
                "phase": "B",
                "ms": round(phase_b_ms),
                "delegate_to": delegate_to,
                "delegate_mode": delegate_mode,
            }
        )
        stats["tool_timings"].extend(phase_tool_timings)
        # Delegation telemetry
        report_text = report or ""
        failed_prefixes = (
            "[ERROR",
            "[Delegation failed",
            "[Delegation timeout",
            "[Investigation failed",
            "[REPL delegation failed",
        )
        success = bool(report_text) and not report_text.startswith(failed_prefixes)
        stats["delegation_events"].append(
            {
                "from_role": architect_role,
                "to_role": delegate_to,
                "task_summary": brief[:DELEGATION_BRIEF_KEY_LEN],
                "success": success,
                "elapsed_ms": round(phase_b_ms),
                "tokens_generated": delegate_tokens,
                "inference_meta": {
                    "transport": specialist_infer_meta.get("transport"),
                    "completion_reason": specialist_infer_meta.get("completion_reason"),
                    "prompt_ms": specialist_infer_meta.get("prompt_ms"),
                    "gen_ms": specialist_infer_meta.get("gen_ms"),
                    "first_token_ms": specialist_infer_meta.get("first_token_ms"),
                    "chunks": specialist_infer_meta.get("stream_chunks"),
                    "tokens": specialist_infer_meta.get("tokens"),
                    "llm_elapsed_ms": specialist_infer_meta.get("llm_elapsed_ms"),
                },
            }
        )

        log.info(f"Specialist {delegate_to} done ({phase_b_ms:.0f}ms, {len(report)} chars)")

        # Specialist timeout is a strong signal of lock pressure or decode stall.
        # Force synthesis immediately instead of re-delegating into another stall.
        if specialist_timed_out:
            stats["break_reason"] = "specialist_timeout"
            break
        # If specialist already produced a substantial report (prose/code rescue),
        # return it directly to avoid an expensive architect synthesis hop.
        if report_rescued:
            stats["break_reason"] = "specialist_report"
            stats["loops"] = loop + 1
            stats["tools_used"] = max(
                total_tools,
                len(all_tools_called),
                len(stats.get("tool_timings", [])),
            )
            stats["tools_called"] = all_tools_called
            return report, stats

    # ── Cap reached ──
    stats["loops"] = max_loops
    stats["tools_used"] = max(
        total_tools,
        len(all_tools_called),
        len(stats.get("tool_timings", [])),
    )
    stats["tools_called"] = all_tools_called

    if force_response_on_cap:
        stats["cap_reached"] = True
        if not stats.get("break_reason"):
            stats["break_reason"] = "forced_synthesis"
        if (
            _skip_synthesis_on_timeout()
            and stats.get("break_reason") in {"specialist_timeout", "wall_clock_budget"}
            and reports
        ):
            # Timeout-triggered synthesis can itself stall on a saturated specialist/
            # architect path. Returning the latest report prevents request timeouts.
            log.warning(
                "Skipping forced synthesis due to timeout break_reason=%s, returning latest report",
                stats.get("break_reason"),
            )
            return reports[-1], stats
        # Force architect to synthesize with whatever we have
        log.warning(f"Architect delegation capped at {max_loops} loops, forcing synthesis")
        forced_prompt = (
            f"You MUST answer now. Synthesize from all available information.\n\n"
            f"Question: {question[:2000]}\n\n"
            f"Investigation reports:\n" + "\n---\n".join(reports[-3:])  # Last 3 reports
        )
        try:
            answer = primitives.llm_call(
                forced_prompt,
                role=architect_role,
                skip_suffix=True,
                n_tokens=_delegation_forced_synthesis_token_cap(),
            )
            return answer.strip(), stats
        except Exception as exc:
            log.debug("Forced synthesis failed, returning last report: %s", exc)
            # Last resort: return the latest specialist report
            return reports[-1] if reports else "[ERROR: Delegation exhausted]", stats
    else:
        # Production mode: signal that user input needed
        stats["needs_input"] = True
        partial = reports[-1] if reports else ""
        return partial, stats

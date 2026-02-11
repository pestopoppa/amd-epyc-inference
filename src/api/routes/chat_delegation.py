"""Architect delegation for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: architect decision parsing (TOON/JSON/text),
multi-loop delegation pipeline where architect formulates
investigation briefs and specialists execute via ReAct or REPL.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, TYPE_CHECKING

from src.repl_environment import REPLEnvironment

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives


# Valid delegation targets for architect briefs
_VALID_DELEGATE_ROLES = frozenset(
    {
        "coder_primary",
        "coder_escalation",
        "worker_explore",
        "worker_general",
        "worker_math",
    }
)


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
    "architect_general": 3375,   # 6.75 t/s × 500s
    "architect_coding": 5150,    # 10.3 t/s × 500s
}

# Tight budget for the routing decision (D|answer or I|brief:...|to:role).
# architect_coding uses <think> tags so 500 visible tokens suffices.
# architect_general (Qwen3-235B) reasons in plain text, exhausting 500 tokens
# before emitting D|.  Give it 1500 so ~1000 goes to reasoning + 500 to answer.
_ARCHITECT_DECISION_BUDGET: dict[str, int] = {
    "architect_general": 1500,
    "architect_coding": 500,
}


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
    from src.prompt_builders import (
        build_architect_investigate_prompt,
        build_architect_synthesis_prompt,
        build_root_lm_prompt,
    )

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
    previous_briefs: list[str] = []  # Track briefs to detect zero-progress loops

    for loop in range(max_loops + 1):  # +1 for initial decision
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

        # Mini-REPL loop: architect can compute before deciding
        arch_repl = REPLEnvironment(
            context=question,
            llm_primitives=primitives,
            tool_registry=tool_registry,
            role=architect_role,
        )
        arch_response = None
        arch_last_output = ""
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
                _ARCHITECT_DECISION_BUDGET.get(architect_role, 500)
                if _aturn == 0
                else _ARCHITECT_TOKEN_BUDGET.get(architect_role, 3375)
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
            except Exception as e:
                log.warning(f"Architect call failed (loop {loop}, turn {_aturn}): {e}")
                if reports:
                    return reports[-1], stats
                return f"[ERROR: Architect delegation failed: {e}]", stats
            finally:
                primitives._early_stop_check = None

            # Strip <think>...</think> tags (reasoning models), including
            # incomplete trailing blocks so mid-thought I|/D| aren't parsed.
            stripped = _strip_think(raw).strip()
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
                from src.graph.nodes import _extract_prose_answer
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
            from src.prompt_builders import extract_code_from_response
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

        total_tools += arch_repl._tool_invocations

        phase_a_ms = (time.perf_counter() - phase_start) * 1000
        decision = _parse_architect_decision(arch_response)
        stats["phases"].append(
            {
                "loop": loop,
                "phase": "A",
                "ms": round(phase_a_ms),
                "decision": decision["mode"],
                "computation_turns": _aturn + 1,
            }
        )

        log.info(f"Architect loop {loop}: {decision['mode']} ({phase_a_ms:.0f}ms, {_aturn+1} turns)")

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
                    f"Respond with D| followed by the letter (A, B, C, or D). No delegation.\n\n"
                    f"Question: {question[:2000]}\n\nDecision:"
                )
                try:
                    forced_raw = primitives.llm_call(
                        force_prompt,
                        role=architect_role,
                        skip_suffix=True,
                        n_tokens=50,
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
            _code_delegate = decision["delegate_to"] in ("coder_escalation", "coder_primary")
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
            break

        # ── Phase B: Specialist execution ──
        brief = decision["brief"]
        delegate_to = decision["delegate_to"]
        delegate_mode = decision["delegate_mode"]

        # Zero-progress guard: if the architect re-delegates with the same
        # brief, it's looping without making progress.  Break immediately.
        brief_key = brief.strip().lower()[:200]
        if brief_key in previous_briefs:
            log.warning("Architect re-delegated with duplicate brief (loop %d), breaking zero-progress loop", loop)
            break
        previous_briefs.append(brief_key)

        log.info(f"Delegating to {delegate_to} (mode={delegate_mode}): {brief[:100]}...")

        phase_b_start = time.perf_counter()
        tokens_before = primitives.total_tokens_generated
        phase_tool_timings: list[dict] = []

        # Both react and repl modes use the same REPL loop
        # (react = structured_mode=True, repl = structured_mode=False)
        structured = delegate_mode == "react"
        max_delegate_turns = 8 if structured else 10
        try:
            deleg_repl = REPLEnvironment(
                context=f"{brief}\n\nContext:\n{context}" if context else brief,
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
                f"{question}\n\n"
                f"## Architect guidance\n{brief}"
            )
            for _turn in range(max_delegate_turns):
                repl_state = deleg_repl.get_state()
                deleg_prompt = build_root_lm_prompt(
                    state=repl_state,
                    original_prompt=specialist_task,
                    last_output=deleg_last_output,
                    last_error=deleg_last_error,
                    turn=_turn,
                )
                _final_re = re.compile(
                    r"""FINAL\(\s*(?:'{3}(.+?)'{3}|"{3}(.+?)"{3}|["'](.+?)["']|(\S+?))\s*\)""",
                    re.DOTALL,
                )
                primitives._early_stop_check = lambda text: bool(_final_re.search(text))
                try:
                    code = primitives.llm_call(
                        deleg_prompt,
                        role=delegate_to,
                        stop_sequences=["\n```\n"],
                    )
                finally:
                    primitives._early_stop_check = None
                raw_deleg_output = code
                from src.prompt_builders import extract_code_from_response, auto_wrap_final
                code = extract_code_from_response(code)
                code = auto_wrap_final(code)
                # Dedup guard: if coder generates identical code twice in a
                # row, inject an error to break the loop instead of wasting
                # another turn on the same silent execution.
                import hashlib
                _code_hash = hashlib.md5(code.encode()).hexdigest()
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
                        log.info(
                            "Delegation prose report (turn %d): %d chars returned to architect",
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
                result = deleg_repl.execute(code)
                if result.is_final:
                    report = result.final_answer or ""
                    break
                deleg_last_output = result.output or ""
                deleg_last_error = result.error or ""
            else:
                report = deleg_repl.get_state()
            total_tools += deleg_repl._tool_invocations
            if deleg_repl.tool_registry:
                for inv in deleg_repl.tool_registry.get_invocation_log():
                    all_tools_called.append(inv.tool_name)
                    phase_tool_timings.append(
                        {"tool_name": inv.tool_name, "elapsed_ms": inv.elapsed_ms, "success": inv.success}
                    )
        except Exception as e:
            report = f"[Delegation failed: {e}]"

        phase_b_ms = (time.perf_counter() - phase_b_start) * 1000
        delegate_tokens = primitives.total_tokens_generated - tokens_before
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
        failed_prefixes = ("[Investigation failed", "[REPL delegation failed")
        success = bool(report_text) and not report_text.startswith(failed_prefixes)
        stats["delegation_events"].append(
            {
                "from_role": architect_role,
                "to_role": delegate_to,
                "task_summary": brief[:200],
                "success": success,
                "elapsed_ms": round(phase_b_ms),
                "tokens_generated": delegate_tokens,
            }
        )

        log.info(f"Specialist {delegate_to} done ({phase_b_ms:.0f}ms, {len(report)} chars)")

    # ── Cap reached ──
    stats["loops"] = max_loops
    stats["tools_used"] = max(
        total_tools,
        len(all_tools_called),
        len(stats.get("tool_timings", [])),
    )
    stats["tools_called"] = all_tools_called

    if force_response_on_cap:
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

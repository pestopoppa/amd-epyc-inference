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
        return {
            "mode": "direct",
            "answer": text[2:].strip(),
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
                fields[key.strip()] = val.strip()

        brief = fields.get("brief", parts_str)
        delegate_to = fields.get("to", "coder_primary")
        delegate_mode = fields.get("mode", "react")

        # Clamp to valid role
        if delegate_to not in _VALID_DELEGATE_ROLES:
            delegate_to = "coder_primary"
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
                delegate_to = obj.get("delegate_to", obj.get("to", "coder_primary"))
                if delegate_to not in _VALID_DELEGATE_ROLES:
                    delegate_to = "coder_primary"
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


_ARCHITECT_TOKEN_BUDGET: dict[str, int] = {
    "architect_general": 3375,   # 6.75 t/s × 500s
    "architect_coding": 5150,    # 10.3 t/s × 500s
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
            try:
                raw = primitives.llm_call(
                    full_prompt,
                    role=architect_role,
                    skip_suffix=True,
                    n_tokens=_ARCHITECT_TOKEN_BUDGET.get(architect_role, 3375),
                )
            except Exception as e:
                log.warning(f"Architect call failed (loop {loop}, turn {_aturn}): {e}")
                if reports:
                    return reports[-1], stats
                return f"[ERROR: Architect delegation failed: {e}]", stats

            # Strip <think>...</think> tags (reasoning models)
            stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            # Search for TOON decision anywhere in the response
            # (model may prefix with reasoning before D|answer)
            decision_match = re.search(r"^(D\|.+)$", stripped, re.MULTILINE)
            if not decision_match:
                decision_match = re.search(r"^(I\|.+)$", stripped, re.MULTILINE)
            if decision_match:
                arch_response = decision_match.group(1).strip()
                break
            # Check start-of-response for TOON/JSON (backward compat)
            if stripped.startswith("D|") or stripped.startswith("I|"):
                arch_response = stripped
                break
            if stripped.startswith("{") or stripped.startswith("```"):
                arch_response = stripped
                break

            # Otherwise treat as Python code — execute it
            from src.prompt_builders import extract_code_from_response
            code = extract_code_from_response(raw)
            if not code.strip():
                # No code extracted, treat raw output as direct answer
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

        # ── Direct/final answer ──
        if decision["mode"] == "direct":
            answer = decision["answer"]
            # If architect just says "Approved" and we have specialist output,
            # the specialist's document IS the final answer
            if answer.lower().strip() in ("approved", "approved.") and stats["specialist_output"]:
                answer = stats["specialist_output"]
            stats["loops"] = loop
            stats["tools_used"] = max(
                total_tools,
                len(all_tools_called),
                len(stats.get("tool_timings", [])),
            )
            stats["tools_called"] = all_tools_called
            return answer, stats

        # ── Phase B: Specialist execution ──
        brief = decision["brief"]
        delegate_to = decision["delegate_to"]
        delegate_mode = decision["delegate_mode"]

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
            for _turn in range(max_delegate_turns):
                repl_state = deleg_repl.get_state()
                deleg_prompt = build_root_lm_prompt(
                    state=repl_state,
                    original_prompt=brief,
                    last_output=deleg_last_output,
                    last_error=deleg_last_error,
                    turn=_turn,
                )
                code = primitives.llm_call(
                    deleg_prompt,
                    role=delegate_to,
                )
                from src.prompt_builders import extract_code_from_response, auto_wrap_final
                code = extract_code_from_response(code)
                code = auto_wrap_final(code)
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

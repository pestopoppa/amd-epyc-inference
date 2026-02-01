"""Architect delegation for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: architect decision parsing (TOON/JSON/text),
multi-loop delegation pipeline where architect formulates
investigation briefs and specialists execute via ReAct or REPL.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TYPE_CHECKING

from src.api.routes.chat_react import _react_mode_answer
from src.repl_environment import REPLEnvironment

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives


# Valid delegation targets for architect briefs
_VALID_DELEGATE_ROLES = frozenset({
    "coder_primary", "coder_escalation", "worker_explore",
    "worker_general", "worker_math",
})


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
    )

    total_tools = 0
    all_tools_called: list[str] = []
    stats: dict = {
        "loops": 0,
        "phases": [],
        "specialist_output": "",
        "needs_input": False,
        "tools_used": 0,
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

        # ── Phase A: Architect call ──
        if loop == 0:
            prompt_text = build_architect_investigate_prompt(question, toon_context)
            n_tokens = 512
        else:
            report = reports[-1] if reports else ""
            prompt_text = build_architect_synthesis_prompt(
                question, report, loop, max_loops,
            )
            n_tokens = 2048  # Synthesis needs more room

        try:
            arch_response = primitives.llm_call(
                prompt_text,
                role=architect_role,
                n_tokens=n_tokens,
                skip_suffix=True,
            )
        except Exception as e:
            log.warning(f"Architect call failed (loop {loop}): {e}")
            # If we have reports, synthesize from what we have
            if reports:
                return reports[-1], stats
            return f"[ERROR: Architect delegation failed: {e}]", stats

        phase_a_ms = (time.perf_counter() - phase_start) * 1000
        decision = _parse_architect_decision(arch_response)
        stats["phases"].append({
            "loop": loop,
            "phase": "A",
            "ms": round(phase_a_ms),
            "decision": decision["mode"],
        })

        log.info(
            f"Architect loop {loop}: {decision['mode']} "
            f"({phase_a_ms:.0f}ms)"
        )

        # ── Direct/final answer ──
        if decision["mode"] == "direct":
            answer = decision["answer"]
            # If architect just says "Approved" and we have specialist output,
            # the specialist's document IS the final answer
            if (
                answer.lower().strip() in ("approved", "approved.")
                and stats["specialist_output"]
            ):
                answer = stats["specialist_output"]
            stats["loops"] = loop
            stats["tools_used"] = total_tools
            stats["tools_called"] = all_tools_called
            return answer, stats

        # ── Phase B: Specialist execution ──
        brief = decision["brief"]
        delegate_to = decision["delegate_to"]
        delegate_mode = decision["delegate_mode"]

        log.info(
            f"Delegating to {delegate_to} (mode={delegate_mode}): "
            f"{brief[:100]}..."
        )

        phase_b_start = time.perf_counter()

        if delegate_mode == "react":
            # ReAct investigation loop
            try:
                report, phase_tools, phase_tool_names = _react_mode_answer(
                    prompt=brief,
                    context=context,
                    primitives=primitives,
                    role=delegate_to,
                    tool_registry=tool_registry,
                    max_turns=8,
                )
                total_tools += phase_tools
                all_tools_called.extend(phase_tool_names)
            except Exception as e:
                report = f"[Investigation failed: {e}]"
        else:
            # REPL mode for drafting — use REPL environment
            try:
                deleg_repl = REPLEnvironment(
                    context=f"{brief}\n\nContext:\n{context}" if context else brief,
                    llm_primitives=primitives,
                    tool_registry=tool_registry,
                    role=delegate_to,
                )
                # Run REPL for up to 10 turns
                for _turn in range(10):
                    code = primitives.llm_call(
                        deleg_repl.get_prompt(question=brief),
                        role=delegate_to,
                        n_tokens=2048,
                    )
                    result = deleg_repl.execute(code)
                    if result.get("final"):
                        report = result["final"]
                        break
                else:
                    report = deleg_repl.get_state()
                total_tools += deleg_repl._tool_invocations
                if deleg_repl.tool_registry:
                    all_tools_called.extend(
                        inv.tool_name for inv in deleg_repl.tool_registry.get_invocation_log()
                    )
            except Exception as e:
                report = f"[REPL delegation failed: {e}]"

        phase_b_ms = (time.perf_counter() - phase_b_start) * 1000
        reports.append(report)
        stats["specialist_output"] = report
        stats["phases"].append({
            "loop": loop,
            "phase": "B",
            "ms": round(phase_b_ms),
            "delegate_to": delegate_to,
            "delegate_mode": delegate_mode,
        })

        log.info(
            f"Specialist {delegate_to} done ({phase_b_ms:.0f}ms, "
            f"{len(report)} chars)"
        )

    # ── Cap reached ──
    stats["loops"] = max_loops
    stats["tools_used"] = total_tools
    stats["tools_called"] = all_tools_called

    if force_response_on_cap:
        # Force architect to synthesize with whatever we have
        log.warning(f"Architect delegation capped at {max_loops} loops, forcing synthesis")
        forced_prompt = (
            f"You MUST answer now. Synthesize from all available information.\n\n"
            f"Question: {question[:2000]}\n\n"
            f"Investigation reports:\n"
            + "\n---\n".join(reports[-3:])  # Last 3 reports
        )
        try:
            answer = primitives.llm_call(
                forced_prompt,
                role=architect_role,
                n_tokens=2048,
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

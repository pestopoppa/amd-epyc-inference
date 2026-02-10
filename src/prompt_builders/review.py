"""Architect-facing prompts: quality review, plan review, and delegation."""

from __future__ import annotations

from typing import Any

from src.prompt_builders.resolver import resolve_prompt


# ── Fallback Constants ──────────────────────────────────────────────────────

_REVIEW_VERDICT_FALLBACK = """Judge this answer. Respond with ONLY one line:
- "OK" if correct and complete
- "WRONG: <what to fix>" if incorrect (max 30 words)
{digest_section}
Q: {question}
A: {answer}

Verdict:"""

_REVISION_FALLBACK = """Rewrite this answer applying the corrections below.
Keep the same style and depth. Only change what the corrections require.

Question: {question}

Original answer: {original}

Corrections: {corrections}

Revised answer:"""

_PLAN_REVIEW_FALLBACK = """Review plan. Reply JSON ONLY:
{{"d":"ok|reorder|drop|add|reroute","s":0.0-1.0,"f":"<15 words","p":[]}}

d=decision, s=confidence, f=feedback, p=patches (optional)
Patch format: {{"step":"S1","op":"reroute|drop|add|reorder","v":"new_value"}}

Task: {objective}
Type: {task_type}
Plan:
{steps_section}

Verdict:"""

_ARCHITECT_INVESTIGATE_FALLBACK = """You are a software architect. You design solutions; a coding specialist implements them.

REPL: available for quick math/estimation ONLY. Never write full programs.

OUTPUT FORMAT: Reply with EXACTLY ONE decision line. No explanation before or after.
- Direct answer: D|<answer>
- Delegate to specialist: I|brief:<spec>|to:<role>

Rules:
- For factual/reasoning/multiple-choice: D|answer IMMEDIATELY. No elaboration.
  NEVER delegate factual questions to ANY role — specialists cannot look up facts. Answer directly.
- For quick math: compute in REPL, then D|answer
- For code/algorithms/implementation: I|brief:<your design>|to:coder_escalation
- For investigation/search: I|brief:<plan>|to:worker_explore
- Valid roles: coder_escalation, worker_explore, worker_summarize, worker_vision, vision_escalation

CRITICAL: Output the decision line ONLY. Stop generating after D|answer or I|brief:...|to:role. Do NOT explain your reasoning, justify your choice, or add any text after the decision.
{context_section}
Question: {question}

Decision:"""

_ARCHITECT_SYNTHESIS_FALLBACK = """The specialist has investigated and reported back. Extract the answer from their report.

Respond ONLY with:
D|<the answer>
{investigate_option}
If the report contains a complete correct answer, respond with:
D|Approved

If the report contains a clear answer (e.g. a single letter or short value), trust it and respond D| followed by that answer. Do NOT substitute your own reasoning when the specialist gave a definitive answer.

If the report is truly empty or an error, use your own reasoning to answer. Do NOT re-delegate.

Question: {question}

Specialist Report:
{report}

Decision:"""


# ── Quality Review Prompts ──────────────────────────────────────────────────


def build_review_verdict_prompt(
    question: str,
    answer: str,
    context_digest: str = "",
    worker_digests: list[dict] | None = None,
) -> str:
    """Build architect verdict prompt — forces hyper-concise output.

    Uses TOON encoding for worker_digests (uniform array of section summaries)
    to minimize tokens sent to architect. TOON achieves 40-65% reduction on
    structured arrays vs JSON.

    Args:
        question: Original user question (truncated to 300 chars).
        answer: The answer to review (truncated to 1500 chars).
        context_digest: Optional compact text digest for context-dependent claims.
        worker_digests: Optional list of worker digest dicts for TOON encoding.

    Returns:
        Prompt string for architect verdict.
    """
    digest_section = ""
    if worker_digests:
        # TOON-encode structured worker digests (uniform array → 40-65% savings)
        try:
            from src.services.toon_encoder import encode, is_available

            if is_available():
                digest_section = f"\nEvidence:\n{encode(worker_digests)}\n"
            else:
                import json

                digest_section = f"\nEvidence:\n{json.dumps(worker_digests)}\n"
        except Exception:
            import json

            digest_section = f"\nEvidence:\n{json.dumps(worker_digests)}\n"
    elif context_digest:
        digest_section = f"\nContext: {context_digest[:800]}\n"

    return resolve_prompt(
        "review_verdict", _REVIEW_VERDICT_FALLBACK,
        digest_section=digest_section,
        question=question[:300],
        answer=answer[:1500],
    )


def build_revision_prompt(question: str, original: str, corrections: str) -> str:
    """Build fast model revision prompt — expands architect corrections.

    Args:
        question: Original user question.
        original: The original answer to revise.
        corrections: Architect's correction notes.

    Returns:
        Prompt string for the revision model.
    """
    return resolve_prompt(
        "revision", _REVISION_FALLBACK,
        question=question[:300],
        original=original[:1500],
        corrections=corrections,
    )


# ── Plan Review Prompts ───────────────────────────────────────────────────


def build_plan_review_prompt(
    objective: str,
    task_type: str,
    plan_steps: list[dict[str, Any]],
) -> str:
    """Build architect plan review prompt — forces hyper-concise JSON output.

    The architect reviews the frontdoor's tentative plan and can confirm,
    reroute steps, reorder, drop, or add missing steps.

    Args:
        objective: Task objective (truncated to 200 chars).
        task_type: Task type (e.g., "code", "chat").
        plan_steps: List of plan step dicts with id, actor/role, action, deps.

    Returns:
        Prompt string for architect plan review (~100-120 tokens input).
    """
    # Format steps compactly: S1:coder:Implement handler->output.py
    step_lines = []
    for step in plan_steps[:8]:  # Cap at 8 steps
        step_id = step.get("id", "S?")
        actor = step.get("actor", step.get("role", "worker"))
        action = step.get("action", "")[:50]
        outputs = step.get("outputs", step.get("out", []))
        out_str = ",".join(str(o) for o in outputs[:2]) if outputs else ""
        deps = step.get("deps", step.get("inputs", []))
        dep_str = f"({','.join(str(d) for d in deps[:2])})" if deps else ""

        line = f"{step_id}:{actor}:{action}"
        if out_str:
            line += f"->{out_str}"
        if dep_str:
            line += dep_str
        step_lines.append(line)

    steps_block = "\n".join(step_lines)

    return resolve_prompt(
        "plan_review", _PLAN_REVIEW_FALLBACK,
        objective=objective[:200],
        task_type=task_type,
        steps_section=steps_block,
    )


# ── Architect Delegation Prompts ─────────────────────────────────────────


def build_architect_investigate_prompt(
    question: str,
    context: str = "",
) -> str:
    """Build prompt asking architect to decide: answer directly or delegate investigation.

    The architect emits TOON-encoded decisions:
    - Direct: ``D|<answer>``
    - Investigate (ReAct): ``I|brief:<text>|to:<role>``
    - Investigate (REPL/draft): ``I|brief:<text>|to:<role>|mode:repl``
    Falls back to JSON if architect ignores TOON instruction.

    Args:
        question: The user's question.
        context: Optional TOON-encoded or plain-text context.

    Returns:
        Prompt string for architect.
    """
    context_section = ""
    if context:
        context_section = f"\nContext (TOON-encoded for efficiency):\n{context[:3000]}\n"

    return resolve_prompt(
        "architect_investigate", _ARCHITECT_INVESTIGATE_FALLBACK,
        context_section=context_section,
        question=question[:2000],
    )


def _specialist_clearly_failed(report: str) -> bool:
    """Return True if the specialist report is empty or an explicit error."""
    if not report or not report.strip():
        return True
    prefixes = ("[ERROR", "[FAILED", "[Investigation failed", "[Delegation failed")
    return report.strip().startswith(prefixes)


def build_architect_synthesis_prompt(
    question: str,
    report: str,
    loop_num: int,
    max_loops: int,
) -> str:
    """Build prompt for architect to synthesize from investigation report.

    The architect extracts/approves the specialist's answer.  Re-delegation
    is only offered when the specialist clearly failed (empty output or
    error prefix).  This prevents pointless loops where the architect
    repeatedly delegates without progress.

    Args:
        question: The original user question.
        report: The specialist's investigation report.
        loop_num: Current loop number (1-indexed).
        max_loops: Maximum allowed loops.

    Returns:
        Prompt string for architect synthesis.
    """
    # Only offer re-delegation if specialist clearly failed AND loops remain
    can_reinvestigate = (
        loop_num < max_loops and _specialist_clearly_failed(report)
    )
    if can_reinvestigate:
        investigate_option = (
            f"\nThe specialist FAILED. If you need to retry (loop {loop_num}/{max_loops}), respond with:\n"
            "I|brief:<what to investigate differently>|to:coder_escalation\n"
        )
    else:
        investigate_option = (
            "\nDo NOT delegate or request further investigation. Answer now.\n"
        )

    return resolve_prompt(
        "architect_synthesis", _ARCHITECT_SYNTHESIS_FALLBACK,
        investigate_option=investigate_option,
        question=question[:2000],
        report=report[:6000],
    )

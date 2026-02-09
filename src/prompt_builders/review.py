"""Architect-facing prompts: quality review, plan review, and delegation."""

from __future__ import annotations

from typing import Any


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

    return f"""Judge this answer. Respond with ONLY one line:
- "OK" if correct and complete
- "WRONG: <what to fix>" if incorrect (max 30 words)
{digest_section}
Q: {question[:300]}
A: {answer[:1500]}

Verdict:"""


def build_revision_prompt(question: str, original: str, corrections: str) -> str:
    """Build fast model revision prompt — expands architect corrections.

    Args:
        question: Original user question.
        original: The original answer to revise.
        corrections: Architect's correction notes.

    Returns:
        Prompt string for the revision model.
    """
    return f"""Rewrite this answer applying the corrections below.
Keep the same style and depth. Only change what the corrections require.

Question: {question[:300]}

Original answer: {original[:1500]}

Corrections: {corrections}

Revised answer:"""


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

    return f"""Review plan. Reply JSON ONLY:
{{"d":"ok|reorder|drop|add|reroute","s":0.0-1.0,"f":"<15 words","p":[]}}

d=decision, s=confidence, f=feedback, p=patches (optional)
Patch format: {{"step":"S1","op":"reroute|drop|add|reorder","v":"new_value"}}

Task: {objective[:200]}
Type: {task_type}
Plan:
{steps_block}

Verdict:"""


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

    return f"""You are an architect deciding how to answer a question.

You have a Python REPL available. If you need to compute something (math, numerical estimation, symbolic manipulation), output Python code and the result will be returned to you. Then make your decision based on the computed result. Available: math, numpy, scipy, statistics, itertools, fractions, decimal.

When ready to decide, output your decision ON ITS OWN LINE:

D|<your answer>
I|brief:<plan>|to:coder_escalation

Nothing else on the decision line. Stop generating after D|answer.

Rules:
- Be concise and essential. Give the answer, not an essay. No preamble, no restating the question — unless explicitly asked for elaboration.
- After reaching a conclusion, COMMIT to it. Do not second-guess, revise, or restart your analysis. Your first well-reasoned answer is your final answer.
- For computation-heavy questions: compute first, then D|answer
- For investigation/search: delegate with I| and a detailed plan
- List ALL steps in the brief — you get one report per loop
- "to:" must be a valid role: coder_escalation, worker_explore, worker_summarize, worker_vision, vision_escalation
{context_section}
Question: {question[:2000]}

Decision:"""


def build_architect_synthesis_prompt(
    question: str,
    report: str,
    loop_num: int,
    max_loops: int,
) -> str:
    """Build prompt for architect to synthesize from investigation report.

    The architect reviews the specialist's report and either:
    - Emits a final answer: ``D|<answer>``
    - Requests another investigation: ``I|brief:...|to:<role>``

    Args:
        question: The original user question.
        report: The specialist's investigation report.
        loop_num: Current loop number (1-indexed).
        max_loops: Maximum allowed loops.

    Returns:
        Prompt string for architect synthesis.
    """
    can_investigate = loop_num < max_loops
    investigate_option = ""
    if can_investigate:
        investigate_option = (
            f"\nIf you need MORE investigation (loop {loop_num}/{max_loops}), respond with:\n"
            "I|brief:<what else to investigate>|to:coder_escalation\n"
        )

    return f"""You are an architect synthesizing an answer from an investigation report.

Review the report below and provide a final answer. Be concise — give the answer directly, no preamble. Commit to your best answer; do not second-guess or restart analysis.

Respond with:
D|<your final answer>
{investigate_option}
If the specialist produced a complete document/code and it looks correct, respond with:
D|Approved

Question: {question[:2000]}

Investigation Report:
{report[:6000]}

Decision:"""

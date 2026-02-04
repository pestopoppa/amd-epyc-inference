#!/usr/bin/env python3
"""Input formalizer for complex prompt preprocessing.

Detects prompts that benefit from formal specification extraction
(optimization, proofs, algorithms, ambiguous requirements) and runs
a cold-tier MathSmith-8B model to produce a FormalizationIR JSON.

The formal spec is injected into the specialist's context — NOT replacing
the original prompt — so the specialist sees both the natural language
request and the structured specification.

Usage:
    from src.formalizer import should_formalize_input, formalize_prompt, inject_formalization

    should, hint = should_formalize_input(prompt)
    if should:
        result = formalize_prompt(prompt, hint, registry)
        if result.success:
            context = inject_formalization(prompt, context, result.ir_json)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.registry_loader import RegistryLoader

from src.config import _registry_timeout

log = logging.getLogger(__name__)

# Default formalizer timeout from registry
_FORMALIZER_TIMEOUT = int(_registry_timeout("tools", "formalizer", 60))

# ---------------------------------------------------------------------------
# 4.1a: Keyword detection heuristics
# ---------------------------------------------------------------------------

# Pattern groups: (compiled regex, problem_type_hint)
_OPTIMIZATION_RE = re.compile(
    r"\b(optimiz\w*|minimiz\w*|maximiz\w*|constraint|feasib\w*|objective\s+function"
    r"|linear\s+program\w*|convex|pareto|slack\s+variable)\b",
    re.IGNORECASE,
)

_PROOF_RE = re.compile(
    r"\b(prove|proof|verify|invariant|theorem|lemma|corollary"
    r"|induction|contradiction|qed)\b",
    re.IGNORECASE,
)

_ALGORITHM_RE = re.compile(
    r"\b(algorithm|data\s+structure|complexity|O\(|amortiz\w*|asymptotic"
    r"|concurrent|distributed|lock-free|race\s+condition|deadlock"
    r"|topological|dynamic\s+program\w*|greedy|backtrack\w*)\b",
    re.IGNORECASE,
)

_VAGUE_PREFIX_RE = re.compile(
    r"^(build|create|design|implement|develop|make)\s+(a|an|the)\s+",
    re.IGNORECASE,
)

# Minimum word count for vague-spec detection (short prompts are usually clear)
_VAGUE_MIN_WORDS = 50


def should_formalize_input(prompt: str) -> tuple[bool, str]:
    """Decide whether a prompt would benefit from formal specification extraction.

    Uses conservative keyword heuristics — false negatives (missed formalization)
    are acceptable; false positives add latency but don't hurt quality.

    Args:
        prompt: Raw user prompt text.

    Returns:
        Tuple of (should_formalize, problem_type_hint).
        problem_type_hint is one of: optimization, proof, algorithm, ambiguous_spec, or "".
    """
    if not prompt or len(prompt.strip()) < 20:
        return (False, "")

    if _OPTIMIZATION_RE.search(prompt):
        return (True, "optimization")

    if _PROOF_RE.search(prompt):
        return (True, "proof")

    if _ALGORITHM_RE.search(prompt):
        return (True, "algorithm")

    # Vague specification detection: starts with "build a..." and is long
    # but doesn't contain specific technical terms that would indicate clarity
    word_count = len(prompt.split())
    if word_count >= _VAGUE_MIN_WORDS and _VAGUE_PREFIX_RE.match(prompt):
        return (True, "ambiguous_spec")

    return (False, "")


# ---------------------------------------------------------------------------
# 4.1b: Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FormalizationResult:
    """Result of formalizer invocation."""

    success: bool
    ir_json: dict[str, Any] | None = None  # Parsed FormalizationIR
    raw_output: str = ""  # Raw model output
    elapsed_seconds: float = 0.0
    model_role: str = ""  # Which formalizer role was used
    error: str = ""  # Error message on failure


# ---------------------------------------------------------------------------
# 4.1c: Formalizer invocation
# ---------------------------------------------------------------------------

_FORMALIZER_SYSTEM = """\
You are a formal specification extractor. Given a problem description, \
output a JSON object with these fields:
- "problem_type": one of "optimization", "constraint_satisfaction", "proof", \
"algorithm", "validation", "search", "tool_orchestration", "architecture", "workflow"
- "variables": array of {"name": string, "type": string, "constraints": [string]}
- "constraints": array of constraint strings
- "objective": objective function or goal (string or null)
- "edge_cases": array of {"input": string, "expected": string}
- "acceptance_criteria": array of criterion strings

Output ONLY valid JSON. No explanation, no markdown fences."""


def _build_formalizer_prompt(prompt: str, problem_type_hint: str) -> str:
    """Build the prompt sent to the formalizer model."""
    hint_line = f"\nProblem type hint: {problem_type_hint}" if problem_type_hint else ""
    return f"{_FORMALIZER_SYSTEM}\n\nProblem:{hint_line}\n{prompt}\n\nJSON:"


def _parse_formalizer_output(raw: str) -> dict[str, Any] | None:
    """Try to parse formalizer output as JSON.

    Handles common failure modes: markdown fences, leading text, trailing garbage.
    """
    text = raw.strip()
    if not text:
        return None

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    # Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "problem_type" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in output
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            obj = json.loads(text[brace_start : brace_end + 1])
            if isinstance(obj, dict) and "problem_type" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return None


def formalize_prompt(
    prompt: str,
    problem_type_hint: str,
    registry: "RegistryLoader",
    timeout: int = _FORMALIZER_TIMEOUT,
) -> FormalizationResult:
    """Run formalizer model to extract formal specification from prompt.

    Tries "formalizer" role first (MathSmith-8B Q8_0), falls back to
    "formalizer_q4" (MathSmith-8B Q4_K_M) on failure.

    Args:
        prompt: The user's raw prompt.
        problem_type_hint: Detected problem type (from should_formalize_input).
        registry: RegistryLoader for command generation.
        timeout: Subprocess timeout in seconds (from registry).

    Returns:
        FormalizationResult with parsed IR or error details.
    """
    formalizer_prompt = _build_formalizer_prompt(prompt, problem_type_hint)
    roles_to_try = ["formalizer", "formalizer_q4"]

    for role_name in roles_to_try:
        try:
            registry.get_role(role_name)
        except KeyError:
            log.debug("Formalizer role '%s' not in registry, skipping", role_name)
            continue

        start = time.monotonic()
        try:
            cmd = registry.generate_command(
                role_name,
                prompt=formalizer_prompt,
                n_tokens=1024,
            )
            log.info("Running formalizer (%s): %s", role_name, cmd[:120])

            import shlex

            proc = subprocess.run(
                shlex.split(cmd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.monotonic() - start
            raw = proc.stdout or ""

            if proc.returncode != 0:
                log.warning(
                    "Formalizer %s exited %d (%.1fs): %s",
                    role_name,
                    proc.returncode,
                    elapsed,
                    (proc.stderr or "")[:200],
                )
                continue

            ir_json = _parse_formalizer_output(raw)
            if ir_json is not None:
                return FormalizationResult(
                    success=True,
                    ir_json=ir_json,
                    raw_output=raw,
                    elapsed_seconds=elapsed,
                    model_role=role_name,
                )
            else:
                log.warning(
                    "Formalizer %s output not valid JSON (%.1fs), trying next",
                    role_name,
                    elapsed,
                )

        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            log.warning("Formalizer %s timed out after %.1fs", role_name, elapsed)
        except Exception as e:
            elapsed = time.monotonic() - start
            log.warning("Formalizer %s error: %s (%.1fs)", role_name, e, elapsed)

    # All attempts failed
    return FormalizationResult(
        success=False,
        error="All formalizer attempts failed",
        elapsed_seconds=0.0,
    )


# ---------------------------------------------------------------------------
# 4.1d: Context injection
# ---------------------------------------------------------------------------


def inject_formalization(
    prompt: str,
    context: str,
    ir: dict[str, Any],
) -> str:
    """Augment context with formal specification block.

    Appends a [FORMAL SPECIFICATION] block to the existing context.
    The specialist sees both the original prompt and the formal spec.

    Args:
        prompt: Original user prompt (unused but reserved for future use).
        context: Existing context string.
        ir: Parsed FormalizationIR dict.

    Returns:
        Augmented context string.
    """
    parts = ["[FORMAL SPECIFICATION]"]

    problem_type = ir.get("problem_type", "unknown")
    parts.append(f"Problem type: {problem_type}")

    # Variables
    variables = ir.get("variables", [])
    if variables:
        var_lines = []
        for v in variables:
            name = v.get("name", "?")
            vtype = v.get("type", "?")
            constraints = v.get("constraints", [])
            cstr = f" ({', '.join(constraints)})" if constraints else ""
            var_lines.append(f"  - {name}: {vtype}{cstr}")
        parts.append("Variables:\n" + "\n".join(var_lines))

    # Constraints
    constraints = ir.get("constraints", [])
    if constraints:
        parts.append("Constraints:\n" + "\n".join(f"  - {c}" for c in constraints))

    # Objective
    objective = ir.get("objective")
    if objective:
        parts.append(f"Objective: {objective}")

    # Edge cases
    edge_cases = ir.get("edge_cases", [])
    if edge_cases:
        ec_lines = []
        for ec in edge_cases:
            inp = ec.get("input", "?")
            exp = ec.get("expected", "?")
            ec_lines.append(f"  - Input: {inp} → Expected: {exp}")
        parts.append("Edge cases:\n" + "\n".join(ec_lines))

    # Acceptance criteria
    criteria = ir.get("acceptance_criteria", [])
    if criteria:
        parts.append("Acceptance criteria:\n" + "\n".join(f"  - {c}" for c in criteria))

    parts.append("[/FORMAL SPECIFICATION]")

    spec_block = "\n".join(parts)

    if context:
        return context + "\n\n" + spec_block
    return spec_block

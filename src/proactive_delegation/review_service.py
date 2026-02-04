"""Architect review and aggregation services."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, TYPE_CHECKING

from src.config import _registry_timeout
from src.proactive_delegation.types import (
    ArchitectReview,
    PlanReviewResult,
    ReviewDecision,
    SubtaskResult,
)

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives

logger = logging.getLogger(__name__)

# Default review timeout from registry
_REVIEW_TIMEOUT = float(_registry_timeout("external", "review_service", 15))


class AggregationService:
    """Service for combining outputs from multiple specialists.

    Strategies:
    - concatenate: Simple concatenation with headers
    - merge_code: Merge code outputs intelligently
    - structured: Combine into structured JSON
    """

    def aggregate(
        self,
        results: list[SubtaskResult],
        strategy: str = "concatenate",
    ) -> str:
        """Aggregate multiple specialist outputs into final result."""
        if not results:
            return ""

        if strategy == "concatenate":
            return self._aggregate_concatenate(results)
        elif strategy == "merge_code":
            return self._aggregate_merge_code(results)
        elif strategy == "structured":
            return self._aggregate_structured(results)
        else:
            logger.warning(f"Unknown aggregation strategy '{strategy}', using concatenate")
            return self._aggregate_concatenate(results)

    def _aggregate_concatenate(self, results: list[SubtaskResult]) -> str:
        """Simple concatenation with section headers."""
        sections = []
        for result in results:
            if result.success and result.output:
                header = f"## {result.subtask_id} ({result.role})"
                sections.append(f"{header}\n\n{result.output}")
        return "\n\n---\n\n".join(sections)

    def _aggregate_merge_code(self, results: list[SubtaskResult]) -> str:
        """Merge code outputs, handling imports and dependencies."""
        imports = set()
        code_blocks = []

        for result in results:
            if not result.success or not result.output:
                continue

            lines = result.output.split("\n")
            current_block = []

            for line in lines:
                # Extract imports
                if line.startswith("import ") or line.startswith("from "):
                    imports.add(line)
                else:
                    current_block.append(line)

            if current_block:
                code_blocks.append(
                    f"# From {result.subtask_id} ({result.role})\n" + "\n".join(current_block)
                )

        # Combine imports at top, then code blocks
        output_parts = []
        if imports:
            output_parts.append("\n".join(sorted(imports)))
        if code_blocks:
            output_parts.append("\n\n".join(code_blocks))

        return "\n\n".join(output_parts)

    def _aggregate_structured(self, results: list[SubtaskResult]) -> str:
        """Combine into structured JSON output."""
        structured = {
            "results": [
                {
                    "subtask_id": r.subtask_id,
                    "role": r.role,
                    "success": r.success,
                    "output": r.output,
                    "error": r.error,
                }
                for r in results
            ],
            "summary": {
                "total": len(results),
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
            },
        }
        return json.dumps(structured, indent=2)


class ArchitectReviewService:
    """Service for architect to review specialist outputs.

    The architect evaluates outputs against the original spec and provides
    feedback for iteration or approval.

    IMPORTANT: Prompts are designed for minimal token output from expensive
    architect models (Qwen3-235B, Qwen3-Coder-480B).
    """

    # Concise review prompt - minimize architect output tokens
    REVIEW_PROMPT_TEMPLATE = """Review specialist output. Be BRIEF.

Objective: {objective}
Subtask: {action}
Output (truncated):
{output}

Reply JSON only (no explanation):
{{"d":"approve|changes|escalate|reject","s":0.0-1.0,"f":"<10 words","c":["fix1"]}}

d=decision, s=score, f=feedback, c=changes (optional, max 3 items)"""

    # Even more compact for simple approve/reject decisions
    QUICK_REVIEW_PROMPT = """Review: {action}
Output: {output_preview}
Reply: {{"d":"approve|changes","s":0.0-1.0,"f":"<5 words"}}"""

    # TaskIR generation prompt (for future use when frontdoor queries architect)
    TASKIR_GENERATION_PROMPT = """Break down task into subtasks. Be MINIMAL.

Task: {objective}

Reply JSON only:
{{"steps":[{{"id":"S1","actor":"coder|worker|math","action":"<10 words","out":["file.py"]}}]}}

Rules:
- Max 5 steps
- actor: coder (code), worker (docs/tests), math (proofs)
- action: imperative, <10 words
- out: expected output files"""

    # Max tokens for architect responses (expensive model)
    MAX_REVIEW_TOKENS = 128
    MAX_TASKIR_TOKENS = 256

    def __init__(
        self,
        primitives: "LLMPrimitives",
        architect_role: str = "architect_general",
    ):
        """Initialize the review service."""
        self.primitives = primitives
        self.architect_role = architect_role

    def review(
        self,
        spec: dict[str, Any],
        subtask: dict[str, Any],
        output: str,
        quick_mode: bool = False,
    ) -> ArchitectReview:
        """Have architect review a specialist's output."""
        subtask_id = subtask.get("id", "unknown")
        action = subtask.get("action", "")

        # Truncate output aggressively to save input tokens
        output_truncated = output[:500] + "..." if len(output) > 500 else output

        # Build concise review prompt - only include objective, not full spec
        objective = spec.get("objective", "")[:200]

        if quick_mode:
            prompt = self.QUICK_REVIEW_PROMPT.format(
                action=action[:50],
                output_preview=output[:200],
            )
        else:
            prompt = self.REVIEW_PROMPT_TEMPLATE.format(
                objective=objective,
                action=action,
                output=output_truncated,
            )

        try:
            # Call architect with strict token limit
            response = self.primitives.llm_call(
                prompt,
                role=self.architect_role,
                n_tokens=self.MAX_REVIEW_TOKENS,
            )

            # Parse abbreviated JSON response
            review_data = self._parse_review_response(response)

            # Map abbreviated keys to full names
            decision_str = review_data.get("d", review_data.get("decision", "changes"))
            # Normalize decision string
            if decision_str == "changes":
                decision_str = "request_changes"
            decision = ReviewDecision(decision_str)

            return ArchitectReview(
                subtask_id=subtask_id,
                decision=decision,
                feedback=review_data.get("f", review_data.get("feedback", "")),
                score=float(review_data.get("s", review_data.get("score", 0.5))),
                suggested_changes=review_data.get("c", review_data.get("suggested_changes", [])),
                approved_output=output if decision == ReviewDecision.APPROVE else None,
            )

        except Exception as e:
            logger.warning(f"Architect review failed: {e}", exc_info=True)
            # Default to request_changes on failure
            return ArchitectReview(
                subtask_id=subtask_id,
                decision=ReviewDecision.REQUEST_CHANGES,
                feedback=f"Review failed: {e}",
                score=0.3,
            )

    # Max tokens for plan review responses
    MAX_PLAN_REVIEW_TOKENS = 128

    def review_plan(
        self,
        objective: str,
        task_type: str,
        plan_steps: list[dict[str, Any]],
        timeout_seconds: float = _REVIEW_TIMEOUT,
    ) -> PlanReviewResult | None:
        """Have architect review a plan before specialist execution.

        Returns PlanReviewResult on success, None on timeout/error.
        Non-blocking: never prevents request completion.
        """
        from src.prompt_builders import build_plan_review_prompt

        prompt = build_plan_review_prompt(objective, task_type, plan_steps)

        try:
            response = self.primitives.llm_call(
                prompt,
                role=self.architect_role,
                n_tokens=self.MAX_PLAN_REVIEW_TOKENS,
            )

            # Parse abbreviated JSON response
            review_data = self._parse_review_response(response)

            decision = review_data.get("d", review_data.get("decision", "ok"))
            # Normalize valid decisions
            valid_decisions = {"ok", "reorder", "drop", "add", "reroute"}
            if decision not in valid_decisions:
                decision = "ok"

            return PlanReviewResult(
                decision=decision,
                score=float(review_data.get("s", review_data.get("score", 0.5))),
                feedback=review_data.get("f", review_data.get("feedback", "")),
                patches=review_data.get("p", review_data.get("patches", [])),
                raw_response=response[:200],
            )

        except Exception as e:
            logger.warning(f"Plan review failed: {e}", exc_info=True)
            return None  # Non-blocking -- proceed without review

    def generate_taskir(self, objective: str) -> dict[str, Any]:
        """Have architect generate minimal TaskIR for an objective."""
        prompt = self.TASKIR_GENERATION_PROMPT.format(
            objective=objective[:300],  # Truncate long objectives
        )

        try:
            response = self.primitives.llm_call(
                prompt,
                role=self.architect_role,
                n_tokens=self.MAX_TASKIR_TOKENS,
            )

            taskir_data = self._parse_review_response(response)

            # Normalize abbreviated format to full TaskIR
            steps = taskir_data.get("steps", [])
            normalized_steps = []
            for i, step in enumerate(steps):
                normalized_steps.append(
                    {
                        "id": step.get("id", f"S{i + 1}"),
                        "actor": step.get("actor", "worker"),
                        "action": step.get("action", ""),
                        "outputs": step.get("out", step.get("outputs", [])),
                        "inputs": step.get("in", step.get("inputs", [])),
                    }
                )

            return {
                "task_id": f"arch-{uuid.uuid4().hex[:8]}",
                "objective": objective,
                "plan": {"steps": normalized_steps},
            }

        except Exception as e:
            logger.warning(f"TaskIR generation failed: {e}", exc_info=True)
            # Return single-step fallback
            return {
                "task_id": f"arch-{uuid.uuid4().hex[:8]}",
                "objective": objective,
                "plan": {
                    "steps": [
                        {
                            "id": "S1",
                            "actor": "coder",
                            "action": objective[:50],
                            "outputs": ["output.txt"],
                        }
                    ]
                },
            }

    def _parse_review_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from architect response."""
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Could not parse review response: {response[:200]}")
            return {"decision": "request_changes", "feedback": "Parse error"}

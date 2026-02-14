"""
Distillation prompt templates for skill extraction.

Three prompt types matching SkillRL §3.1:
1. Success distillation — strategic routing patterns
2. Failure lessons — structured failure analysis (failure point, flawed reasoning,
   correct alternative, prevention principle)
3. Escalation patterns — transferable reasoning from architect to worker

Templates are stored as module constants with {trajectories_json} and
optional {failure_graph_summary} placeholders.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

SUCCESS_DISTILLATION_PROMPT = """\
# Task: Distill Routing Skills from Successful Trajectories

You are analyzing successful task trajectories from an LLM orchestration system
that routes tasks between model tiers (7B workers, 30B front door, 32B coder,
235B architect, 480B architect-coding). Extract reusable strategic patterns
that explain WHY these routing decisions worked.

## Trajectories

{trajectories_json}

## Instructions

For each cluster of similar successful patterns, produce a skill record as JSON:

```json
{{
  "title": "3-7 word imperative title",
  "skill_type": "routing",
  "principle": "1-3 sentence actionable strategy. Be specific about model tiers, task types, and conditions.",
  "when_to_apply": "Precise applicability conditions. Reference task_type, complexity indicators, or context fields.",
  "task_types": ["list", "of", "applicable", "task_types"],
  "source_outcome": "success"
}}
```

## Rules

1. Each skill must be ACTIONABLE — a routing decision the system can directly apply
2. Merge similar trajectories into ONE skill (compress, don't enumerate)
3. Preserve specific model names, tier names, and port numbers from the trajectories
4. Reference concrete thresholds from Q-values when available
5. Produce 3-8 skills per batch (fewer, more general is better than many narrow ones)
6. Do NOT produce skills for trivially obvious patterns
7. Return all skills inside a single ```json [...] ``` fenced block
"""

FAILURE_LESSON_PROMPT = """\
# Task: Extract Failure Lessons from Failed Trajectories

Analyze these failed task trajectories and extract structured failure lessons.

## Trajectories

{trajectories_json}

## Existing FailureGraph Context

{failure_graph_summary}

## Instructions

For each failure pattern, produce a failure lesson as JSON:

```json
{{
  "title": "3-7 word title describing the anti-pattern",
  "skill_type": "failure_lesson",
  "principle": "Structure as: FAILURE POINT: [what went wrong]. FLAWED REASONING: [why the system made this choice]. CORRECT ALTERNATIVE: [what should have happened]. PREVENTION: [actionable rule to avoid recurrence].",
  "when_to_apply": "Conditions that indicate this failure pattern is about to recur",
  "task_types": ["applicable", "types"],
  "source_outcome": "failure"
}}
```

## Rules

1. Each lesson must identify the ROOT CAUSE, not just the symptom
2. The PREVENTION field must be a concrete, testable rule
3. Cross-reference with the FailureGraph summary — avoid duplicating known mitigations
4. If a failure has an existing mitigation with success_rate > 0.8, skip it
5. Produce 2-5 lessons per batch
6. Return all skills inside a single ```json [...] ``` fenced block
"""

ESCALATION_PATTERN_PROMPT = """\
# Task: Extract Escalation Patterns

Analyze trajectories where tasks were escalated from a lower tier to a higher tier.
Identify patterns that could allow the lower tier to handle these tasks directly.

## Escalated Trajectories

{trajectories_json}

## Instructions

For each escalation pattern, produce a skill as JSON:

```json
{{
  "title": "Imperative title for avoiding this escalation",
  "skill_type": "escalation",
  "principle": "What the lower-tier model should do differently to handle this without escalation. Reference specific reasoning strategies the architect used.",
  "when_to_apply": "Task characteristics that currently trigger escalation but could be handled locally",
  "task_types": ["applicable", "types"],
  "source_outcome": "success"
}}
```

## Rules

1. Focus on TRANSFERABLE reasoning — strategies the 7B/30B model can actually execute
2. Don't suggest escalation avoidance for genuinely complex tasks (architecture, novel design)
3. Prioritize high-frequency escalation patterns (most impact from preventing common escalations)
4. Produce 1-4 skills per batch
5. Return all skills inside a single ```json [...] ``` fenced block
"""


def build_success_prompt(trajectories: List[Dict[str, Any]]) -> str:
    """Build a success distillation prompt with trajectory data."""
    return SUCCESS_DISTILLATION_PROMPT.format(
        trajectories_json=json.dumps(trajectories, indent=2, default=str),
    )


def build_failure_prompt(
    trajectories: List[Dict[str, Any]],
    failure_graph_summary: str = "No FailureGraph context available.",
) -> str:
    """Build a failure lesson prompt with trajectory data and FailureGraph context."""
    return FAILURE_LESSON_PROMPT.format(
        trajectories_json=json.dumps(trajectories, indent=2, default=str),
        failure_graph_summary=failure_graph_summary,
    )


def build_escalation_prompt(trajectories: List[Dict[str, Any]]) -> str:
    """Build an escalation pattern prompt with trajectory data."""
    return ESCALATION_PATTERN_PROMPT.format(
        trajectories_json=json.dumps(trajectories, indent=2, default=str),
    )

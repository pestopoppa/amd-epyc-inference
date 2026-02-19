"""Claude meta-agent integration for memory design evolution.

MetaAgentWorkflow builds reflection prompts from archive history,
parses Claude's proposed candidates, evaluates them via replay,
and generates comparison reports for human review.

Dual interface:
- CLI: `python3 -m orchestration.repl_memory.replay.meta_agent --days 14`
- Library: `workflow.build_reflection_prompt()` → Claude → `workflow.parse_candidates(response)`
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..progress_logger import ProgressReader
from ..q_scorer import ScoringConfig
from ..retriever import RetrievalConfig
from .candidates import DesignArchive, DesignCandidate
from .engine import ReplayEngine
from .metrics import ReplayMetrics
from .trajectory import TrajectoryExtractor

logger = logging.getLogger(__name__)

# Prompt template location
PROMPT_TEMPLATE_PATH = Path(
    "/mnt/raid0/llm/claude/orchestration/prompts/meta_agent_reflect.md"
)


class MetaAgentWorkflow:
    """Orchestrates the meta-learning loop for memory design evolution.

    1. Build a reflection prompt with archive history + trajectory stats
    2. Parse Claude's proposed candidate configs
    3. Evaluate candidates via replay engine
    4. Generate comparison report
    5. Recommend promotion if candidate beats baseline by >5% on regret-optimized objective
    """

    def __init__(
        self,
        archive: Optional[DesignArchive] = None,
        reader: Optional[ProgressReader] = None,
        replay_engine: Optional[ReplayEngine] = None,
    ):
        self.archive = archive or DesignArchive()
        self.reader = reader or ProgressReader()
        self.engine = replay_engine or ReplayEngine()

    def build_reflection_prompt(
        self,
        days: int = 14,
        max_trajectories: int = 1000,
    ) -> str:
        """Build a prompt for Claude to propose new design candidates.

        Assembles current production config, archive history, and recent
        trajectory statistics into a reflection prompt.

        Args:
            days: Days of trajectory history to summarize.
            max_trajectories: Max trajectories for stats.

        Returns:
            Formatted prompt string for Claude.
        """
        # Load prompt template
        template = _load_prompt_template()

        # Current production config
        baseline = DesignCandidate.default()
        config_section = _format_config(baseline)

        # Archive summary
        archive_section = self._format_archive_summary()

        # Recent trajectory stats
        extractor = TrajectoryExtractor(self.reader)
        trajectories = extractor.extract_complete(days=days, max_trajectories=max_trajectories)
        stats_section = _format_trajectory_stats(trajectories)

        # Assemble prompt
        prompt = template.format(
            config=config_section,
            archive=archive_section,
            stats=stats_section,
        )
        return prompt

    def parse_candidates(
        self,
        claude_response: str,
        parent_id: Optional[str] = None,
    ) -> List[DesignCandidate]:
        """Parse candidate configs from Claude's response.

        Expects JSON blocks with retrieval_config and scoring_config fields.

        Args:
            claude_response: Claude's text response.
            parent_id: Parent candidate ID for lineage.

        Returns:
            List of validated DesignCandidate objects.
        """
        candidates = []

        # Extract JSON blocks from response
        json_blocks = re.findall(r"```json\s*\n(.*?)\n```", claude_response, re.DOTALL)
        if not json_blocks:
            # Try bare JSON objects
            json_blocks = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", claude_response)

        for block in json_blocks:
            try:
                data = json.loads(block)
                candidate = self._parse_single_candidate(data, parent_id)
                if candidate:
                    candidates.append(candidate)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("Failed to parse candidate block: %s", e)
                continue

        logger.info("Parsed %d candidates from Claude response", len(candidates))
        return candidates

    def evaluate_candidates(
        self,
        candidates: List[DesignCandidate],
        trajectories: Optional[List] = None,
        days: int = 14,
        max_trajectories: int = 1000,
    ) -> List[Tuple[DesignCandidate, ReplayMetrics]]:
        """Evaluate candidates via replay engine.

        Args:
            candidates: Candidates to evaluate.
            trajectories: Pre-extracted trajectories (or None to extract).
            days: Days of logs for trajectory extraction.
            max_trajectories: Max trajectories.

        Returns:
            List of (candidate, metrics) tuples.
        """
        if trajectories is None:
            extractor = TrajectoryExtractor(self.reader)
            trajectories = extractor.extract_complete(days=days, max_trajectories=max_trajectories)

        results = []
        for candidate in candidates:
            logger.info("Evaluating candidate %s: %s", candidate.candidate_id, candidate.notes)
            metrics = self.engine.run_with_metrics(
                retrieval_config=candidate.retrieval_config,
                scoring_config=candidate.scoring_config,
                trajectories=trajectories,
                candidate_id=candidate.candidate_id,
            )
            self.archive.store_result(candidate, metrics)
            results.append((candidate, metrics))

        return results

    def recommend_promotion(
        self,
        results: List[Tuple[DesignCandidate, ReplayMetrics]],
    ) -> Optional[DesignCandidate]:
        """Recommend best candidate if it beats baseline by >5% on RM-softmax objective.

        Args:
            results: Evaluated candidates with metrics.

        Returns:
            Best candidate if it beats baseline by >5%, else None.
        """
        baseline_result = self.archive.get_baseline()
        if not baseline_result:
            # No baseline — recommend the best candidate
            if results:
                best = max(results, key=lambda r: r[1].rm_softmax_score)
                return best[0]
            return None

        _, baseline_metrics = baseline_result
        best = max(results, key=lambda r: r[1].rm_softmax_score)
        best_candidate, best_metrics = best

        if baseline_metrics.rm_softmax_score != 0:
            improvement = (
                (best_metrics.rm_softmax_score - baseline_metrics.rm_softmax_score)
                / abs(baseline_metrics.rm_softmax_score)
            )
            if improvement > 0.05:  # >5% improvement
                logger.info(
                    "Candidate %s beats baseline by %.1f%% on rm_softmax_score — recommending promotion",
                    best_candidate.candidate_id, improvement * 100,
                )
                return best_candidate

        logger.info("No candidate beats baseline by >5%% on rm_softmax_score — no promotion recommended")
        return None

    def generate_report(
        self,
        results: List[Tuple[DesignCandidate, ReplayMetrics]],
    ) -> str:
        """Generate a markdown comparison report.

        Args:
            results: Evaluated candidates with metrics.

        Returns:
            Markdown-formatted report string.
        """
        baseline_result = self.archive.get_baseline()
        baseline_metrics = baseline_result[1] if baseline_result else None

        lines = [
            "# Replay Evaluation Report",
            "",
            f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Candidates evaluated**: {len(results)}",
            "",
            "## Results",
            "",
            (
                "| Candidate | Notes | Routing Acc | Esc Prec | Utility | RM-Softmax | "
                "Regret p95 | Cum Reward | Cost Eff | vs Baseline |"
            ),
            (
                "|-----------|-------|-------------|----------|---------|------------|------------|"
                "------------|----------|-------------|"
            ),
        ]

        for candidate, metrics in results:
            delta_str = "—"
            if baseline_metrics and baseline_metrics.rm_softmax_score != 0:
                delta = (
                    (metrics.rm_softmax_score - baseline_metrics.rm_softmax_score)
                    / abs(baseline_metrics.rm_softmax_score) * 100
                )
                delta_str = f"{delta:+.1f}%"

            lines.append(
                f"| `{candidate.candidate_id[:8]}` | {candidate.notes[:30]} "
                f"| {metrics.routing_accuracy:.1%} "
                f"| {metrics.escalation_precision:.1%} "
                f"| {metrics.utility_score:.3f} "
                f"| {metrics.rm_softmax_score:.3f} "
                f"| {metrics.regret_p95:.3f} "
                f"| {metrics.cumulative_reward:.1f} "
                f"| {metrics.cost_efficiency:.3f} "
                f"| {delta_str} |"
            )

        # Recommendation
        promotion = self.recommend_promotion(results)
        lines.extend(["", "## Recommendation", ""])
        if promotion:
            lines.append(f"**Promote `{promotion.candidate_id[:8]}`**: {promotion.notes}")
            lines.append("")
            lines.append("Config changes:")
            lines.append(f"- semantic_k: {promotion.retrieval_config.semantic_k}")
            lines.append(f"- q_weight: {promotion.retrieval_config.q_weight}")
            lines.append(f"- cost_lambda: {promotion.retrieval_config.cost_lambda}")
            lines.append(
                f"- confidence_estimator: {promotion.retrieval_config.confidence_estimator}"
            )
            lines.append(
                f"- confidence_threshold: {promotion.retrieval_config.confidence_threshold}"
            )
            lines.append(
                "- calibrated_confidence_threshold: "
                f"{promotion.retrieval_config.calibrated_confidence_threshold}"
            )
            lines.append(f"- conformal_margin: {promotion.retrieval_config.conformal_margin}")
            lines.append(f"- risk_budget_id: {promotion.retrieval_config.risk_budget_id}")
            lines.append(
                f"- risk_gate_min_samples: {promotion.retrieval_config.risk_gate_min_samples}"
            )
            lines.append(
                f"- risk_abstain_target_role: "
                f"{promotion.retrieval_config.risk_abstain_target_role}"
            )
            lines.append(
                f"- risk_gate_rollout_ratio: {promotion.retrieval_config.risk_gate_rollout_ratio}"
            )
            lines.append(
                f"- risk_gate_kill_switch: {promotion.retrieval_config.risk_gate_kill_switch}"
            )
            lines.append(
                "- risk_budget_guardrail_min_events: "
                f"{promotion.retrieval_config.risk_budget_guardrail_min_events}"
            )
            lines.append(
                "- risk_budget_guardrail_max_abstain_rate: "
                f"{promotion.retrieval_config.risk_budget_guardrail_max_abstain_rate}"
            )
            lines.append(f"- prior_strength: {promotion.retrieval_config.prior_strength}")
            lines.append(
                f"- warm_probability_hit: {promotion.retrieval_config.warm_probability_hit}"
            )
            lines.append(
                f"- warm_probability_miss: {promotion.retrieval_config.warm_probability_miss}"
            )
            lines.append(f"- learning_rate: {promotion.scoring_config.learning_rate}")
            lines.append(f"- cost_penalty_lambda: {promotion.scoring_config.cost_penalty_lambda}")
        else:
            lines.append(
                "No candidate beats baseline by >5% on regret-optimized objective."
                " Keep current production config."
            )

        return "\n".join(lines)

    def _format_archive_summary(self) -> str:
        """Format archive history for the reflection prompt."""
        sample = self.archive.sample_for_reflection(n=5)
        if not sample:
            return "No prior candidates in archive."

        lines = ["| Candidate | Routing Acc | RM-Softmax | Notes |",
                 "|-----------|-------------|------------|-------|"]
        for candidate, metrics in sample:
            lines.append(
                f"| `{candidate.candidate_id[:8]}` "
                f"| {metrics.routing_accuracy:.1%} "
                f"| {metrics.rm_softmax_score:.3f} "
                f"| {candidate.notes[:40]} |"
            )
        return "\n".join(lines)

    def _parse_single_candidate(
        self,
        data: Dict[str, Any],
        parent_id: Optional[str],
    ) -> Optional[DesignCandidate]:
        """Parse and validate a single candidate from JSON data."""
        try:
            ret_data = data.get("retrieval_config", {})
            scr_data = data.get("scoring_config", {})

            ret_config = RetrievalConfig(**{
                k: _clamp(k, v) for k, v in ret_data.items()
                if k in RetrievalConfig.__dataclass_fields__
            })
            scr_config = ScoringConfig(**{
                k: v for k, v in scr_data.items()
                if k in ScoringConfig.__dataclass_fields__
            })

            return DesignCandidate(
                candidate_id=str(uuid.uuid4()),
                parent_id=parent_id,
                retrieval_config=ret_config,
                scoring_config=scr_config,
                notes=data.get("notes", ""),
                created_at=datetime.now(timezone.utc),
            )
        except (TypeError, ValueError) as e:
            logger.warning("Invalid candidate data: %s", e)
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Acceptable parameter ranges for validation
_PARAM_RANGES = {
    "semantic_k": (5, 100),
    "min_similarity": (0.1, 0.9),
    "min_q_value": (0.0, 0.9),
    "q_weight": (0.3, 1.0),
    "cost_lambda": (0.0, 1.0),
    "top_n": (1, 20),
    "confidence_threshold": (0.3, 0.95),
    "confidence_trim_ratio": (0.0, 0.4),
    "confidence_min_neighbors": (1, 20),
    "calibrated_confidence_threshold": (0.3, 0.99),
    "conformal_margin": (0.0, 0.2),
    "risk_gate_min_samples": (1, 50),
    "risk_gate_rollout_ratio": (0.0, 1.0),
    "risk_budget_guardrail_min_events": (1, 10000),
    "risk_budget_guardrail_max_abstain_rate": (0.0, 1.0),
    "prior_strength": (0.0, 1.0),
    "warm_probability_hit": (0.5, 1.0),
    "warm_probability_miss": (0.0, 0.5),
    "warm_cost_fallback_s": (0.1, 30.0),
    "cold_cost_fallback_s": (0.1, 60.0),
}


def _clamp(key: str, value: Any) -> Any:
    """Clamp a parameter value to its acceptable range."""
    if key in _PARAM_RANGES and isinstance(value, (int, float)):
        lo, hi = _PARAM_RANGES[key]
        return type(value)(max(lo, min(hi, value)))
    return value


def _load_prompt_template() -> str:
    """Load the reflection prompt template."""
    if PROMPT_TEMPLATE_PATH.exists():
        return PROMPT_TEMPLATE_PATH.read_text()
    # Fallback inline template
    return (
        "# Memory Design Reflection\n\n"
        "## Current Production Config\n{config}\n\n"
        "## Archive History\n{archive}\n\n"
        "## Recent Trajectory Statistics\n{stats}\n\n"
        "## Task\n"
        "Propose 2-3 new DesignCandidate configs as JSON blocks.\n"
        "Each should have `retrieval_config`, `scoring_config`, and `notes` fields.\n"
    )


def _format_config(candidate: DesignCandidate) -> str:
    """Format a candidate's config for display."""
    rc = candidate.retrieval_config
    sc = candidate.scoring_config
    return (
        f"- semantic_k: {rc.semantic_k}\n"
        f"- min_similarity: {rc.min_similarity}\n"
        f"- q_weight: {rc.q_weight}\n"
        f"- cost_lambda: {rc.cost_lambda}\n"
        f"- confidence_threshold: {rc.confidence_threshold}\n"
        f"- confidence_estimator: {rc.confidence_estimator}\n"
        f"- confidence_trim_ratio: {rc.confidence_trim_ratio}\n"
        f"- confidence_min_neighbors: {rc.confidence_min_neighbors}\n"
        f"- calibrated_confidence_threshold: {rc.calibrated_confidence_threshold}\n"
        f"- conformal_margin: {rc.conformal_margin}\n"
        f"- risk_control_enabled: {rc.risk_control_enabled}\n"
        f"- risk_budget_id: {rc.risk_budget_id}\n"
        f"- risk_gate_min_samples: {rc.risk_gate_min_samples}\n"
        f"- risk_abstain_target_role: {rc.risk_abstain_target_role}\n"
        f"- risk_gate_rollout_ratio: {rc.risk_gate_rollout_ratio}\n"
        f"- risk_gate_kill_switch: {rc.risk_gate_kill_switch}\n"
        f"- risk_budget_guardrail_min_events: {rc.risk_budget_guardrail_min_events}\n"
        f"- risk_budget_guardrail_max_abstain_rate: {rc.risk_budget_guardrail_max_abstain_rate}\n"
        f"- prior_strength: {rc.prior_strength}\n"
        f"- warm_probability_hit: {rc.warm_probability_hit}\n"
        f"- warm_probability_miss: {rc.warm_probability_miss}\n"
        f"- warm_cost_fallback_s: {rc.warm_cost_fallback_s}\n"
        f"- cold_cost_fallback_s: {rc.cold_cost_fallback_s}\n"
        f"- learning_rate: {sc.learning_rate}\n"
        f"- success_reward: {sc.success_reward}\n"
        f"- failure_reward: {sc.failure_reward}\n"
        f"- cost_penalty_lambda: {sc.cost_penalty_lambda}\n"
    )


def _format_trajectory_stats(trajectories: list) -> str:
    """Format trajectory statistics for the prompt."""
    if not trajectories:
        return "No trajectories available."

    total = len(trajectories)
    outcomes = {}
    types = {}
    for t in trajectories:
        outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1
        types[t.task_type] = types.get(t.task_type, 0) + 1

    lines = [
        f"- Total trajectories: {total}",
        f"- Outcomes: {outcomes}",
        f"- Task types: {types}",
    ]
    escalated = sum(1 for t in trajectories if t.escalations)
    lines.append(f"- Escalation rate: {escalated/total:.1%}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for meta-agent workflow."""
    parser = argparse.ArgumentParser(description="Replay evaluation meta-agent")
    parser.add_argument("--days", type=int, default=14, help="Days of log history")
    parser.add_argument("--max-trajectories", type=int, default=1000)
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=DEFAULT_ARCHIVE_PATH,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    workflow = MetaAgentWorkflow(
        archive=DesignArchive(args.archive_path),
    )

    # Build prompt
    prompt = workflow.build_reflection_prompt(
        days=args.days,
        max_trajectories=args.max_trajectories,
    )
    print("=== REFLECTION PROMPT ===")
    print(prompt)
    print("========================")
    print()
    print("Paste this prompt to Claude, then provide the response as input.")
    print("Press Ctrl+D when done:")

    # Read Claude's response from stdin
    try:
        response = sys.stdin.read()
    except KeyboardInterrupt:
        print("\nAborted.")
        return

    if not response.strip():
        print("No response provided.")
        return

    # Parse and evaluate
    candidates = workflow.parse_candidates(response)
    if not candidates:
        print("No valid candidates parsed from response.")
        return

    print(f"\nParsed {len(candidates)} candidates. Evaluating...")

    extractor = TrajectoryExtractor(workflow.reader)
    trajectories = extractor.extract_complete(
        days=args.days, max_trajectories=args.max_trajectories,
    )

    results = workflow.evaluate_candidates(candidates, trajectories)

    # Generate report
    report = workflow.generate_report(results)
    print("\n" + report)


# Allow `from .candidates import DEFAULT_ARCHIVE_PATH` for CLI
DEFAULT_ARCHIVE_PATH = DesignArchive.__init__.__defaults__[0]  # type: ignore[attr-defined]

if __name__ == "__main__":
    main()

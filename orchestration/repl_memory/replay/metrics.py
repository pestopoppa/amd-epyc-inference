"""Replay evaluation metrics for comparing memory design candidates.

ReplayMetrics captures the performance of a DesignCandidate over a set of
replayed trajectories — routing accuracy, escalation prediction, Q-convergence,
cumulative reward, and cost efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ReplayMetrics:
    """Metrics from replaying trajectories with a given design candidate."""

    candidate_id: str
    num_trajectories: int
    num_complete: int

    # Routing
    routing_accuracy: float = 0.0  # % match with successful actual route
    routing_accuracy_by_type: Dict[str, float] = field(default_factory=dict)

    # Escalation prediction
    escalation_precision: float = 0.0  # low-confidence predicted actual escalation
    escalation_recall: float = 0.0  # fraction of actual escalations predicted

    # Q-value convergence
    q_convergence_step: int = 0  # step where running Q std < 0.05

    # Rewards
    cumulative_reward: float = 0.0
    avg_reward: float = 0.0

    # Cost
    cost_efficiency: float = 0.0  # reward / weighted tier cost

    # Tier breakdown
    tier_usage: Dict[str, int] = field(default_factory=dict)

    # Timing
    replay_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "candidate_id": self.candidate_id,
            "num_trajectories": self.num_trajectories,
            "num_complete": self.num_complete,
            "routing_accuracy": self.routing_accuracy,
            "routing_accuracy_by_type": self.routing_accuracy_by_type,
            "escalation_precision": self.escalation_precision,
            "escalation_recall": self.escalation_recall,
            "q_convergence_step": self.q_convergence_step,
            "cumulative_reward": self.cumulative_reward,
            "avg_reward": self.avg_reward,
            "cost_efficiency": self.cost_efficiency,
            "tier_usage": self.tier_usage,
            "replay_duration_seconds": self.replay_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReplayMetrics:
        """Deserialize from dict."""
        return cls(
            candidate_id=data["candidate_id"],
            num_trajectories=data["num_trajectories"],
            num_complete=data["num_complete"],
            routing_accuracy=data.get("routing_accuracy", 0.0),
            routing_accuracy_by_type=data.get("routing_accuracy_by_type", {}),
            escalation_precision=data.get("escalation_precision", 0.0),
            escalation_recall=data.get("escalation_recall", 0.0),
            q_convergence_step=data.get("q_convergence_step", 0),
            cumulative_reward=data.get("cumulative_reward", 0.0),
            avg_reward=data.get("avg_reward", 0.0),
            cost_efficiency=data.get("cost_efficiency", 0.0),
            tier_usage=data.get("tier_usage", {}),
            replay_duration_seconds=data.get("replay_duration_seconds", 0.0),
        )

    def compare(self, baseline: ReplayMetrics) -> Dict[str, Dict[str, float]]:
        """Compare this candidate's metrics against a baseline.

        Returns dict of {metric_name: {"delta": absolute, "pct_change": relative}}.
        """
        result: Dict[str, Dict[str, float]] = {}
        for metric in (
            "routing_accuracy",
            "escalation_precision",
            "escalation_recall",
            "cumulative_reward",
            "avg_reward",
            "cost_efficiency",
        ):
            self_val = getattr(self, metric)
            base_val = getattr(baseline, metric)
            delta = self_val - base_val
            pct = (delta / base_val * 100) if base_val != 0 else 0.0
            result[metric] = {"delta": round(delta, 4), "pct_change": round(pct, 2)}
        return result

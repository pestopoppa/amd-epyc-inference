"""Replay engine for offline evaluation of memory design candidates.

Creates an isolated EpisodicStore per candidate, replays trajectories
chronologically, and collects per-step results for metrics computation.

The engine uses pre-computed embeddings (no live embedder calls) and
deterministic replay order for reproducible evaluations.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..episodic_store import EpisodicStore
from ..progress_logger import EventType
from ..q_scorer import QScorer, ScoringConfig
from ..retriever import RetrievalConfig, TwoPhaseRetriever
from .metrics import ReplayMetrics
from .trajectory import Trajectory

logger = logging.getLogger(__name__)

# Replay temp directory on RAID array
REPLAY_TMP_DIR = Path("/mnt/raid0/llm/tmp/replay")


@dataclass
class ReplayStepResult:
    """Result from replaying a single trajectory."""

    trajectory_id: str
    candidate_action: Optional[str]  # What candidate's config would have routed
    actual_action: str  # What actually happened
    routing_match: bool  # candidate == actual
    q_value_after: float  # Q-value after this step
    reward: float  # Computed reward
    escalation_predicted: bool  # Candidate predicted escalation


class NullEmbedder:
    """Safety guard: raises if the replay engine tries to call the embedder.

    Replay must use pre-computed embeddings from TrajectoryExtractor.
    """

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"NullEmbedder.{name}() called — replay engine must use "
            "pre-computed embeddings, not live embedding calls"
        )


class ReplayEngine:
    """Offline replay engine for evaluating memory design candidates.

    Creates an isolated EpisodicStore per candidate, replays trajectories
    in chronological order, and collects routing accuracy / reward data.
    """

    def __init__(
        self,
        tmp_dir: Path = REPLAY_TMP_DIR,
        embedding_dim: int = 1024,
    ):
        self.tmp_dir = tmp_dir
        self.embedding_dim = embedding_dim

    def run(
        self,
        retrieval_config: RetrievalConfig,
        scoring_config: ScoringConfig,
        trajectories: List[Trajectory],
        candidate_id: Optional[str] = None,
    ) -> List[ReplayStepResult]:
        """Replay trajectories with given configs and return per-step results.

        Args:
            retrieval_config: Retrieval parameters for this candidate.
            scoring_config: Scoring parameters for this candidate.
            trajectories: Pre-sorted trajectories (by started_at).
            candidate_id: Optional ID for temp dir naming.

        Returns:
            List of ReplayStepResult, one per trajectory.
        """
        cid = candidate_id or str(uuid.uuid4())[:8]
        store_dir = self.tmp_dir / cid
        store_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._replay(retrieval_config, scoring_config, trajectories, store_dir)
        finally:
            # Cleanup isolated store
            try:
                shutil.rmtree(store_dir, ignore_errors=True)
            except Exception as e:
                logger.warning("Failed to clean up replay dir %s: %s", store_dir, e)

    def run_with_metrics(
        self,
        retrieval_config: RetrievalConfig,
        scoring_config: ScoringConfig,
        trajectories: List[Trajectory],
        candidate_id: Optional[str] = None,
    ) -> ReplayMetrics:
        """Run replay and compute aggregate metrics.

        Args:
            retrieval_config: Retrieval parameters for this candidate.
            scoring_config: Scoring parameters for this candidate.
            trajectories: Pre-sorted trajectories (by started_at).
            candidate_id: Optional ID for this candidate.

        Returns:
            Aggregate ReplayMetrics.
        """
        cid = candidate_id or str(uuid.uuid4())[:8]
        start_time = time.monotonic()
        results = self.run(retrieval_config, scoring_config, trajectories, cid)
        elapsed = time.monotonic() - start_time
        return self._compute_metrics(cid, trajectories, results, elapsed)

    def _replay(
        self,
        retrieval_config: RetrievalConfig,
        scoring_config: ScoringConfig,
        trajectories: List[Trajectory],
        store_dir: Path,
    ) -> List[ReplayStepResult]:
        """Core replay loop."""
        # Create isolated store
        store = EpisodicStore(
            db_path=store_dir,
            embedding_dim=self.embedding_dim,
            use_faiss=True,
        )
        null_embedder = NullEmbedder()
        retriever = TwoPhaseRetriever(
            store=store,
            embedder=null_embedder,
            config=retrieval_config,
        )

        # Build a scorer stub for reward computation
        scorer = QScorer.__new__(QScorer)
        scorer.config = scoring_config

        results: List[ReplayStepResult] = []

        for trajectory in trajectories:
            step_result = self._replay_step(
                trajectory, store, retriever, scorer,
            )
            results.append(step_result)

        # Cleanup
        store.close()
        return results

    def _replay_step(
        self,
        trajectory: Trajectory,
        store: EpisodicStore,
        retriever: TwoPhaseRetriever,
        scorer: QScorer,
    ) -> ReplayStepResult:
        """Replay a single trajectory step.

        1. Use pre-computed embedding to query the retriever
        2. Determine what the candidate config would have routed
        3. Compute reward from actual outcome
        4. Store the experience in the isolated store
        5. Return the step result
        """
        embedding = trajectory.embedding
        candidate_action: Optional[str] = None
        escalation_predicted = False

        if embedding is not None:
            # Query retriever with raw embedding (bypass embedder)
            candidates = store.retrieve_by_similarity(
                embedding,
                k=retriever.config.semantic_k,
                action_type="routing",
                min_q_value=retriever.config.min_q_value,
            )
            if candidates:
                candidate_action = candidates[0].action
                # Check if candidate predicts low confidence (escalation signal)
                if candidates[0].q_value < retriever.config.confidence_threshold:
                    escalation_predicted = True

        routing_match = (candidate_action == trajectory.routing_decision)

        # Compute reward using the candidate's scoring config
        reward = scorer._compute_reward(
            task_outcome=trajectory.outcome_entry or _make_fake_outcome(trajectory.outcome),
            gate_results=trajectory.gate_entries,
            escalations=trajectory.escalation_entries,
            plan_reviews=trajectory.plan_review_entries if trajectory.plan_review_entries else None,
            cost_metrics=trajectory.cost_metrics or None,
        )

        # Store this experience in the isolated store
        q_value = 0.5 + reward * 0.5  # Map [-1,1] reward to [0,1] Q-value
        q_value = max(0.0, min(1.0, q_value))

        if embedding is not None:
            context = {
                "task_type": trajectory.task_type,
                "objective": trajectory.objective,
                "role": trajectory.routing_decision,
            }
            store.store(
                embedding=embedding,
                action=trajectory.routing_decision,
                action_type="routing",
                context=context,
                outcome=trajectory.outcome,
                initial_q=q_value,
            )

        return ReplayStepResult(
            trajectory_id=trajectory.task_id,
            candidate_action=candidate_action,
            actual_action=trajectory.routing_decision,
            routing_match=routing_match,
            q_value_after=q_value,
            reward=reward,
            escalation_predicted=escalation_predicted,
        )

    def _compute_metrics(
        self,
        candidate_id: str,
        trajectories: List[Trajectory],
        results: List[ReplayStepResult],
        elapsed: float,
    ) -> ReplayMetrics:
        """Compute aggregate metrics from replay results."""
        if not results:
            return ReplayMetrics(
                candidate_id=candidate_id,
                num_trajectories=0,
                num_complete=0,
                replay_duration_seconds=elapsed,
            )

        n = len(results)
        matches = sum(1 for r in results if r.routing_match)
        routing_accuracy = matches / n if n > 0 else 0.0

        # Per-type accuracy
        by_type: Dict[str, List[bool]] = {}
        tier_usage: Dict[str, int] = {}
        for traj, res in zip(trajectories, results):
            by_type.setdefault(traj.task_type, []).append(res.routing_match)
            tier_usage[res.actual_action] = tier_usage.get(res.actual_action, 0) + 1

        accuracy_by_type = {}
        for task_type, matches_list in by_type.items():
            accuracy_by_type[task_type] = sum(matches_list) / len(matches_list) if matches_list else 0.0

        # Escalation metrics
        actual_escalations = [t for t in trajectories if len(t.escalations) > 0]
        predicted_escalations = [r for r in results if r.escalation_predicted]

        # Precision: of those predicted as escalation, how many actually escalated
        esc_true_pos = sum(
            1 for t, r in zip(trajectories, results)
            if r.escalation_predicted and len(t.escalations) > 0
        )
        esc_precision = esc_true_pos / len(predicted_escalations) if predicted_escalations else 0.0
        esc_recall = esc_true_pos / len(actual_escalations) if actual_escalations else 0.0

        # Q-convergence: step where running std of Q-values < 0.05
        q_values = [r.q_value_after for r in results]
        q_convergence = self._find_convergence_step(q_values, threshold=0.05, window=10)

        # Rewards
        rewards = [r.reward for r in results]
        cumulative = sum(rewards)
        avg = cumulative / n if n > 0 else 0.0

        # Cost efficiency: reward per weighted tier cost
        # Simple proxy: reward / num_trajectories (normalized)
        cost_efficiency = avg  # Can be refined with actual tier costs later

        return ReplayMetrics(
            candidate_id=candidate_id,
            num_trajectories=n,
            num_complete=n,
            routing_accuracy=routing_accuracy,
            routing_accuracy_by_type=accuracy_by_type,
            escalation_precision=esc_precision,
            escalation_recall=esc_recall,
            q_convergence_step=q_convergence,
            cumulative_reward=cumulative,
            avg_reward=avg,
            cost_efficiency=cost_efficiency,
            tier_usage=tier_usage,
            replay_duration_seconds=elapsed,
        )

    @staticmethod
    def _find_convergence_step(
        values: List[float],
        threshold: float = 0.05,
        window: int = 10,
    ) -> int:
        """Find the step where running std drops below threshold."""
        if len(values) < window:
            return len(values)

        for i in range(window, len(values)):
            window_vals = values[i - window : i]
            std = np.std(window_vals)
            if std < threshold:
                return i
        return len(values)


def _make_fake_outcome(outcome: str):
    """Create a minimal fake ProgressEntry for reward computation."""
    from ..progress_logger import ProgressEntry, EventType
    evt = EventType.TASK_COMPLETED if outcome != "failure" else EventType.TASK_FAILED
    return ProgressEntry(
        event_type=evt,
        task_id="replay",
        outcome=outcome,
    )

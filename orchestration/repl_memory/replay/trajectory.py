"""Trajectory extraction from progress logs for offline replay evaluation.

Reads structured JSONL progress logs, groups events by task_id, and builds
complete Trajectory objects suitable for replay evaluation of memory configs.

Only complete trajectories (task_started + routing_decision + task_completed/failed)
are included. Incomplete trajectories are logged as warnings and excluded.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..progress_logger import EventType, ProgressEntry, ProgressReader

logger = logging.getLogger(__name__)


def _sort_key_tz(t: "Trajectory") -> datetime:
    """Sort key that normalizes tz-naive datetimes to UTC for safe comparison."""
    dt = t.started_at
    if dt is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class Trajectory:
    """A complete task trajectory extracted from progress logs."""

    task_id: str
    task_type: str
    objective: str
    routing_decision: str  # The action chosen by the router
    outcome: str  # "success", "failure", "partial"
    cost_metrics: Dict[str, Any] = field(default_factory=dict)
    escalations: List[str] = field(default_factory=list)
    gate_results: List[Dict[str, Any]] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None  # Pre-computed 1024-dim
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Raw entries for replay engine access
    outcome_entry: Optional[ProgressEntry] = None
    gate_entries: List[ProgressEntry] = field(default_factory=list)
    escalation_entries: List[ProgressEntry] = field(default_factory=list)
    plan_review_entries: List[ProgressEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (embedding excluded — stored separately)."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "objective": self.objective,
            "routing_decision": self.routing_decision,
            "outcome": self.outcome,
            "cost_metrics": self.cost_metrics,
            "escalations": self.escalations,
            "gate_results": self.gate_results,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> Trajectory:
        """Deserialize from dict."""
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            objective=data["objective"],
            routing_decision=data["routing_decision"],
            outcome=data["outcome"],
            cost_metrics=data.get("cost_metrics", {}),
            escalations=data.get("escalations", []),
            gate_results=data.get("gate_results", []),
            embedding=embedding,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )


class TrajectoryExtractor:
    """Extract complete trajectories from progress logs.

    Groups progress log entries by task_id, filters for completeness,
    and builds Trajectory objects with optional pre-computed embeddings.
    """

    def __init__(
        self,
        reader: ProgressReader,
        embedder: Optional[Any] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.reader = reader
        self.embedder = embedder
        self.cache_dir = cache_dir or Path("/mnt/raid0/llm/claude/cache/replay")

    def extract_complete(
        self,
        days: int = 25,
        max_trajectories: int = 1000,
        seed: int = 42,
    ) -> List[Trajectory]:
        """Extract complete trajectories from the last N days.

        Args:
            days: Number of days of logs to read.
            max_trajectories: Max trajectories to return (0 = all).
                Default 1000 uses stratified sampling by task_type.
            seed: Random seed for reproducible sampling.

        Returns:
            List of complete Trajectory objects sorted by started_at.
        """
        entries = self.reader.read_recent(days=days)
        if not entries:
            logger.warning("No progress entries found for last %d days", days)
            return []

        # Group by task_id
        groups: Dict[str, List[ProgressEntry]] = {}
        for entry in entries:
            if entry.task_id:
                groups.setdefault(entry.task_id, []).append(entry)

        # Build trajectories from complete groups
        trajectories = []
        incomplete_count = 0

        for task_id, task_entries in groups.items():
            trajectory = self._build_trajectory(task_id, task_entries)
            if trajectory is not None:
                trajectories.append(trajectory)
            else:
                incomplete_count += 1

        if incomplete_count > 0:
            logger.warning(
                "Skipped %d incomplete trajectories (missing required events)",
                incomplete_count,
            )

        # Sort by started_at (normalize tz-naive to UTC to avoid comparison errors)
        trajectories.sort(key=_sort_key_tz)

        # Stratified sampling if needed
        if max_trajectories > 0 and len(trajectories) > max_trajectories:
            trajectories = self._stratified_sample(trajectories, max_trajectories, seed)

        logger.info(
            "Extracted %d complete trajectories from %d days (%d task groups)",
            len(trajectories),
            days,
            len(groups),
        )

        return trajectories

    def precompute_embeddings(
        self,
        trajectories: List[Trajectory],
        cache_key: str = "default",
    ) -> List[Trajectory]:
        """Pre-compute embeddings for trajectories using TaskEmbedder.

        Args:
            trajectories: Trajectories to embed.
            cache_key: Cache file identifier (for invalidation).

        Returns:
            Same trajectories with embedding field populated.
        """
        if self.embedder is None:
            logger.warning("No embedder configured — skipping embedding pre-computation")
            return trajectories

        cache_path = self.cache_dir / f"embeddings_{cache_key}.npz"

        # Try loading from cache
        cached = self._load_embedding_cache(cache_path, trajectories)
        if cached is not None:
            return cached

        # Compute embeddings in batch
        texts = []
        for t in trajectories:
            context = f"{t.task_type}: {t.objective}"
            texts.append(context)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.embedder, "embed_batch"):
            embeddings = self.embedder.embed_batch(texts)
            for i, t in enumerate(trajectories):
                t.embedding = embeddings[i]
        else:
            for i, t in enumerate(trajectories):
                t.embedding = self.embedder.embed_text(texts[i])

        # Save cache
        self._save_embedding_cache(cache_path, trajectories)

        return trajectories

    def _build_trajectory(
        self,
        task_id: str,
        entries: List[ProgressEntry],
    ) -> Optional[Trajectory]:
        """Build a Trajectory from a group of entries for one task.

        Returns None if the trajectory is incomplete.
        """
        task_started = None
        task_completed = None
        routing_decision = None
        escalations = []
        gate_results = []
        plan_reviews = []
        cost_metrics: Dict[str, Any] = {}

        for entry in entries:
            if entry.event_type == EventType.TASK_STARTED:
                task_started = entry
            elif entry.event_type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED):
                task_completed = entry
            elif entry.event_type == EventType.ROUTING_DECISION:
                routing_decision = entry
            elif entry.event_type in (
                EventType.ESCALATION_TRIGGERED,
                EventType.ESCALATION_RESOLVED,
                EventType.ESCALATION_FAILED,
            ):
                escalations.append(entry)
            elif entry.event_type in (EventType.GATE_PASSED, EventType.GATE_FAILED):
                gate_results.append(entry)
            elif entry.event_type == EventType.PLAN_REVIEWED:
                plan_reviews.append(entry)

        # Completeness check
        if not task_started or not task_completed or not routing_decision:
            return None

        # Extract fields from data payloads
        task_type = task_started.data.get("task_type", "unknown")
        objective = task_started.data.get("objective", task_started.data.get("prompt", ""))
        routing_data = routing_decision.data or {}
        action = routing_data.get("action", routing_data.get("chosen_action"))
        if not action:
            routing_list = routing_data.get("routing", [])
            if isinstance(routing_list, list) and routing_list:
                action = ",".join(str(r) for r in routing_list)
            elif isinstance(routing_list, str):
                action = routing_list
            else:
                action = routing_data.get("role", "unknown")

        # Outcome from the completion entry
        outcome = task_completed.outcome or "unknown"

        # Cost metrics from completion data
        if task_completed.data:
            for key in ("tokens_generated", "elapsed_seconds", "generation_ms", "role"):
                if key in task_completed.data:
                    cost_metrics[key] = task_completed.data[key]
            # Optional teacher/delegation telemetry keys.
            for key in (
                "regret",
                "speedup_vs_teacher",
                "pass_teacher",
                "pass_chosen",
                "producer_role",
                "final_answer_role",
                "delegation_lineage",
            ):
                if key in task_completed.data:
                    cost_metrics[key] = task_completed.data[key]

        gate_dicts = []
        for g in gate_results:
            gate_dicts.append({
                "gate": g.data.get("gate_name", "unknown"),
                "passed": g.event_type == EventType.GATE_PASSED,
            })

        escalation_list = []
        for e in escalations:
            escalation_list.append(e.data.get("target_role", e.data.get("action", "unknown")))

        return Trajectory(
            task_id=task_id,
            task_type=task_type,
            objective=objective,
            routing_decision=action,
            outcome=outcome,
            cost_metrics=cost_metrics,
            escalations=escalation_list,
            gate_results=gate_dicts,
            started_at=task_started.timestamp,
            completed_at=task_completed.timestamp,
            outcome_entry=task_completed,
            gate_entries=gate_results,
            escalation_entries=escalations,
            plan_review_entries=plan_reviews,
        )

    def _stratified_sample(
        self,
        trajectories: List[Trajectory],
        max_count: int,
        seed: int,
    ) -> List[Trajectory]:
        """Stratified sample proportional to task_type distribution."""
        rng = np.random.default_rng(seed)

        # Group by task_type
        by_type: Dict[str, List[Trajectory]] = {}
        for t in trajectories:
            by_type.setdefault(t.task_type, []).append(t)

        total = len(trajectories)
        sampled: List[Trajectory] = []

        for task_type, group in by_type.items():
            # Proportional allocation (at least 1 per type)
            n = max(1, int(len(group) / total * max_count))
            n = min(n, len(group))
            indices = rng.choice(len(group), size=n, replace=False)
            for idx in indices:
                sampled.append(group[idx])

        # If we overshot due to rounding, trim; if under, add more from largest groups
        if len(sampled) > max_count:
            indices = rng.choice(len(sampled), size=max_count, replace=False)
            sampled = [sampled[i] for i in sorted(indices)]
        elif len(sampled) < max_count:
            remaining = [t for t in trajectories if t not in sampled]
            deficit = max_count - len(sampled)
            if remaining:
                extra_indices = rng.choice(
                    len(remaining), size=min(deficit, len(remaining)), replace=False
                )
                for idx in extra_indices:
                    sampled.append(remaining[idx])

        # Re-sort by started_at
        sampled.sort(key=_sort_key_tz)
        return sampled

    def _load_embedding_cache(
        self,
        cache_path: Path,
        trajectories: List[Trajectory],
    ) -> Optional[List[Trajectory]]:
        """Load embeddings from npz cache if valid."""
        if not cache_path.exists():
            return None

        try:
            data = np.load(cache_path, allow_pickle=False)
            task_ids = list(data["task_ids"]) if "task_ids" in data else []
            embeddings = data["embeddings"] if "embeddings" in data else None

            if embeddings is None or len(task_ids) != len(trajectories):
                return None

            # Verify task_id alignment
            traj_ids = [t.task_id for t in trajectories]
            if task_ids != traj_ids:
                return None

            for i, t in enumerate(trajectories):
                t.embedding = embeddings[i]

            logger.info("Loaded %d cached embeddings from %s", len(trajectories), cache_path)
            return trajectories
        except Exception:
            logger.debug("Failed to load embedding cache from %s", cache_path)
            return None

    def _save_embedding_cache(
        self,
        cache_path: Path,
        trajectories: List[Trajectory],
    ) -> None:
        """Save embeddings to npz cache."""
        try:
            task_ids = np.array([t.task_id for t in trajectories], dtype=object)
            embeddings = np.stack([
                t.embedding for t in trajectories if t.embedding is not None
            ])
            np.savez_compressed(cache_path, task_ids=task_ids, embeddings=embeddings)
            logger.info("Saved %d embeddings to cache %s", len(trajectories), cache_path)
        except Exception as e:
            logger.warning("Failed to save embedding cache: %s", e)

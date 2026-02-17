"""Warm-start protocol for model swap resilience.

When a model serving a role is swapped (e.g., Qwen2.5-Coder-32B → Qwen3-Coder-30B),
existing memories may have stale Q-values that reflected the old model's capabilities.
WarmStartProtocol detects model swaps and resets Q-values for affected memories,
doubling the learning rate during a warmup period to re-learn quickly.

RoleConfig maps roles to model-specific retrieval/scoring parameters.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..episodic_store import EpisodicStore
from ..q_scorer import ScoringConfig
from ..retriever import RetrievalConfig

logger = logging.getLogger(__name__)


@dataclass
class RoleConfig:
    """Role-specific memory configuration."""

    role: str
    model_id: str  # e.g. "qwen2.5-coder-32b-q4km"
    retrieval_config: RetrievalConfig
    scoring_config: ScoringConfig

    @classmethod
    def default_for_role(cls, role: str, model_id: str) -> RoleConfig:
        """Create a RoleConfig with global defaults for a given role."""
        return cls(
            role=role,
            model_id=model_id,
            retrieval_config=RetrievalConfig(),
            scoring_config=ScoringConfig(),
        )


@dataclass
class WarmStartStats:
    """Statistics from a warm-start execution."""

    memories_reset: int
    warmup_tasks_remaining: int
    model_id_old: str
    model_id_new: str


class WarmStartProtocol:
    """Detect and handle model swaps for a given role.

    When a model swap is detected, Q-values for affected memories are
    reset to 0.5 (neutral) and the model_id is updated. The learning
    rate is doubled during warmup (first 50 tasks) to re-learn quickly.
    """

    WARMUP_TASKS: int = 50
    RESET_Q_VALUE: float = 0.5

    @staticmethod
    def detect_model_swap(
        role: str,
        current_model_id: str,
        store: EpisodicStore,
    ) -> bool:
        """Detect if a model swap has occurred for a role.

        Checks if the majority of routing memories for this role have a
        different model_id than the current one.

        Args:
            role: The role to check (e.g. "coder_escalation").
            current_model_id: The model ID currently serving this role.
            store: The episodic store to query.

        Returns:
            True if a swap is detected (majority of memories are from a different model).
        """
        try:
            with sqlite3.connect(store.sqlite_path) as conn:
                # Count memories for this role with a different model_id
                total = conn.execute(
                    """
                    SELECT COUNT(*) FROM memories
                    WHERE action_type = 'routing'
                    AND action = ?
                    AND model_id IS NOT NULL
                    """,
                    (role,),
                ).fetchone()[0]

                if total == 0:
                    return False

                different = conn.execute(
                    """
                    SELECT COUNT(*) FROM memories
                    WHERE action_type = 'routing'
                    AND action = ?
                    AND model_id IS NOT NULL
                    AND model_id != ?
                    """,
                    (role, current_model_id),
                ).fetchone()[0]

            return different > total / 2
        except (sqlite3.OperationalError, AttributeError) as e:
            logger.warning("Failed to detect model swap for role %s: %s", role, e)
            return False

    @staticmethod
    def execute_warm_start(
        role: str,
        new_model_id: str,
        store: EpisodicStore,
    ) -> WarmStartStats:
        """Execute warm-start protocol for a model swap.

        1. Reset Q-values to 0.5 for memories with a different model_id
        2. Update model_id to the new model on reset memories
        3. Return statistics about the warm-start

        Args:
            role: The role being swapped.
            new_model_id: The new model ID.
            store: The episodic store to update.

        Returns:
            WarmStartStats with counts of affected memories.
        """
        old_model_id = "unknown"
        memories_reset = 0

        try:
            with sqlite3.connect(store.sqlite_path) as conn:
                # Find the old model_id
                row = conn.execute(
                    """
                    SELECT model_id FROM memories
                    WHERE action_type = 'routing'
                    AND action = ?
                    AND model_id IS NOT NULL
                    AND model_id != ?
                    LIMIT 1
                    """,
                    (role, new_model_id),
                ).fetchone()
                if row:
                    old_model_id = row[0]

                # Reset Q-values and update model_id
                cursor = conn.execute(
                    """
                    UPDATE memories
                    SET q_value = ?,
                        model_id = ?,
                        updated_at = datetime('now')
                    WHERE action_type = 'routing'
                    AND action = ?
                    AND (model_id IS NULL OR model_id != ?)
                    """,
                    (WarmStartProtocol.RESET_Q_VALUE, new_model_id, role, new_model_id),
                )
                memories_reset = cursor.rowcount
                conn.commit()

        except (sqlite3.OperationalError, AttributeError) as e:
            logger.error("Failed to execute warm-start for role %s: %s", role, e)
            return WarmStartStats(
                memories_reset=0,
                warmup_tasks_remaining=WarmStartProtocol.WARMUP_TASKS,
                model_id_old=old_model_id,
                model_id_new=new_model_id,
            )

        logger.info(
            "Warm-start for role %s: reset %d memories from %s → %s",
            role, memories_reset, old_model_id, new_model_id,
        )

        return WarmStartStats(
            memories_reset=memories_reset,
            warmup_tasks_remaining=WarmStartProtocol.WARMUP_TASKS,
            model_id_old=old_model_id,
            model_id_new=new_model_id,
        )

    @staticmethod
    def is_warmup_active(
        role: str,
        store: EpisodicStore,
        model_id: Optional[str] = None,
    ) -> bool:
        """Check if warmup is still active for a role.

        Returns True if fewer than WARMUP_TASKS memories have been scored
        since the last warm-start (i.e., memories with the current model_id
        and update_count > 0).

        Args:
            role: The role to check.
            store: The episodic store.
            model_id: The current model_id to check against.

        Returns:
            True if warmup is still active.
        """
        if model_id is None:
            return False

        try:
            with sqlite3.connect(store.sqlite_path) as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) FROM memories
                    WHERE action_type = 'routing'
                    AND action = ?
                    AND model_id = ?
                    AND update_count > 0
                    """,
                    (role, model_id),
                ).fetchone()
                scored = row[0] if row else 0
            return scored < WarmStartProtocol.WARMUP_TASKS
        except (sqlite3.OperationalError, AttributeError):
            return False

    @staticmethod
    def get_warmup_scoring_config(
        base_config: ScoringConfig,
    ) -> ScoringConfig:
        """Return a ScoringConfig with doubled learning rate for warmup.

        Args:
            base_config: The base scoring config.

        Returns:
            A new ScoringConfig with 2x learning_rate.
        """
        return ScoringConfig(
            learning_rate=base_config.learning_rate * 2.0,
            success_reward=base_config.success_reward,
            failure_reward=base_config.failure_reward,
            partial_reward=base_config.partial_reward,
            temporal_decay_rate=base_config.temporal_decay_rate,
            use_claude_judge=base_config.use_claude_judge,
            judge_model_path=base_config.judge_model_path,
            judge_binary=base_config.judge_binary,
            cost_penalty_lambda=base_config.cost_penalty_lambda,
            baseline_tps_by_role=base_config.baseline_tps_by_role,
            baseline_quality_by_role=base_config.baseline_quality_by_role,
            memory_cost_by_role=base_config.memory_cost_by_role,
            cost_lambda_quality_gap=base_config.cost_lambda_quality_gap,
            cost_lambda_memory=base_config.cost_lambda_memory,
            min_score_interval_seconds=base_config.min_score_interval_seconds,
            batch_size=base_config.batch_size,
        )

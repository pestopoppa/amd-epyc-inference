"""PARL-inspired staged reward shaping for MemRL Q-value updates.

Anneals an exploration bonus into the reward signal so that early
task outcomes encourage trying diverse (action, task_type) combos
while later outcomes converge to pure exploitation of the base reward.

Formula:
    staged_reward = lambda(step) * exploration_bonus + (1 - lambda(step)) * base_reward

Where:
    lambda(step)      = initial_lambda * (1 - step / anneal_steps), clamped >= min_lambda
    exploration_bonus  = 1 / sqrt(N + 1),  N = count of prior observations for this combo

Reference:
    Kimi K2.5 / PARL (2025): Staged reward transitions from exploration
    to exploitation as the system accumulates experience.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .episodic_store import EpisodicStore

logger = logging.getLogger(__name__)


@dataclass
class StagedConfig:
    """Configuration for staged reward annealing.

    Attributes:
        initial_lambda: Starting weight for exploration bonus (default 0.3).
        anneal_steps: Number of scoring steps to reach min_lambda.
        min_lambda: Floor for annealing coefficient (default 0.0 = pure exploit).
    """

    initial_lambda: float = 0.3
    anneal_steps: int = 50
    min_lambda: float = 0.0


class StagedQScorer:
    """PARL-inspired staged reward shaping for MemRL.

    Maintains a global step counter that drives the annealing schedule.
    Each call to ``compute_staged_reward`` increments the counter and
    blends an exploration bonus with the base reward according to the
    current lambda value.

    The exploration bonus is ``1/sqrt(N+1)`` where N is the number of
    times the system has previously observed a particular (action, task_type)
    combination. This rewards trying under-explored routes early on and
    fades to zero as lambda anneals.

    Usage:
        scorer = StagedQScorer()
        shaped = scorer.compute_staged_reward(base_reward, action, task_type, store)
    """

    def __init__(self, config: StagedConfig | None = None) -> None:
        self.config = config or StagedConfig()
        self._global_step: int = 0

    @property
    def global_step(self) -> int:
        """Current global step count."""
        return self._global_step

    @property
    def current_lambda(self) -> float:
        """Current annealing coefficient (decreases over time).

        Returns initial_lambda at step 0, linearly decays to min_lambda
        at anneal_steps, then stays at min_lambda.
        """
        if self._global_step >= self.config.anneal_steps:
            return self.config.min_lambda
        return self.config.initial_lambda * (
            1 - self._global_step / self.config.anneal_steps
        )

    def compute_staged_reward(
        self,
        base_reward: float,
        action: str,
        task_type: str,
        store: "EpisodicStore",
    ) -> float:
        """Compute annealed reward with exploration bonus.

        Early training (high lambda): shifts reward toward exploration
        bonus for under-explored (action, task_type) combos.

        Late training (lambda -> 0): returns base_reward unchanged.

        Args:
            base_reward: Raw reward from QScorer._compute_reward().
            action: Action string (e.g. comma-joined routing roles).
            task_type: Task type from TaskIR context.
            store: EpisodicStore for count_by_combo() lookup.

        Returns:
            Shaped reward clamped to [-1.0, 1.0].
        """
        n = store.count_by_combo(action, task_type)
        exploration_bonus = 1.0 / sqrt(n + 1)

        lam = self.current_lambda
        staged = lam * exploration_bonus + (1 - lam) * base_reward

        self._global_step += 1
        return max(-1.0, min(1.0, staged))

    def reset(self) -> None:
        """Reset step counter (useful for tests)."""
        self._global_step = 0

#!/usr/bin/env python3
"""Real-time generation health monitoring for early failure detection.

This module monitors token-by-token generation metrics to predict failures
early and enable compute-saving aborts. Implements training-free heuristics
based on entropy, repetition, and perplexity trends.

Literature References:
- Gnosis (arXiv:2512.20578): Early detection at 40% generation
- ZIP-RC (arXiv:2512.01457): Per-token failure prediction
- KnowLoop (arXiv:2406.00430): Entropy-based uncertainty estimation

Usage:
    from src.generation_monitor import GenerationMonitor, MonitorConfig

    # Real mode with logits
    monitor = GenerationMonitor()
    for token, logits in generate():
        health = monitor.update(token, logits)
        if monitor.should_abort()[0]:
            break

    # Mock mode for testing
    monitor = GenerationMonitor(mock_mode=True)
    monitor.set_mock_scenario("high_entropy")
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class PerplexityTrend(str, Enum):
    """Perplexity trend over generation window."""

    STABLE = "stable"
    RISING = "rising"
    FALLING = "falling"
    SPIKING = "spiking"


class AbortReason(str, Enum):
    """Reasons for aborting generation early."""

    NONE = "none"
    HIGH_ENTROPY = "high_entropy"
    ENTROPY_SPIKE = "entropy_spike"
    HIGH_REPETITION = "high_repetition"
    RISING_PERPLEXITY = "rising_perplexity"
    RUNAWAY_LENGTH = "runaway_length"
    COMBINED_SIGNALS = "combined_signals"


def _monitor_cfg():
    from src.config import get_config
    return get_config().monitor


@dataclass
class MonitorConfig:
    """Configuration for generation monitoring.

    Thresholds can be adjusted per-task or per-tier. Higher tiers
    (architect) get more relaxed thresholds since they handle
    harder tasks with natural uncertainty.

    Base defaults sourced from centralized config (src.config).
    Tier-specific overrides in for_tier() are algorithmic and stay hardcoded.

    Attributes:
        min_tokens_before_abort: Minimum tokens before abort is allowed.
        entropy_threshold: Sustained entropy above this triggers abort.
        entropy_spike_threshold: Single-token entropy jump threshold.
        repetition_threshold: Fraction of repeated n-grams (0-1).
        perplexity_window: Rolling window size for perplexity trend.
        max_length_multiplier: Abort if >N× median task length.
        entropy_sustained_count: Tokens of high entropy before abort.
        ngram_size: N-gram size for repetition detection.
        combined_threshold: Weighted score for combined signals.
    """

    min_tokens_before_abort: int = field(default_factory=lambda: _monitor_cfg().min_tokens_before_abort)
    entropy_threshold: float = field(default_factory=lambda: _monitor_cfg().entropy_threshold)
    entropy_spike_threshold: float = field(default_factory=lambda: _monitor_cfg().entropy_spike_threshold)
    repetition_threshold: float = field(default_factory=lambda: _monitor_cfg().repetition_threshold)
    perplexity_window: int = field(default_factory=lambda: _monitor_cfg().perplexity_window)
    max_length_multiplier: float = field(default_factory=lambda: _monitor_cfg().max_length_multiplier)
    entropy_sustained_count: int = field(default_factory=lambda: _monitor_cfg().entropy_sustained_count)
    ngram_size: int = field(default_factory=lambda: _monitor_cfg().ngram_size)
    combined_threshold: float = field(default_factory=lambda: _monitor_cfg().combined_threshold)

    @classmethod
    def for_tier(cls, tier: str) -> MonitorConfig:
        """Get tier-specific configuration from centralized config.

        Higher tiers get more relaxed thresholds since they handle
        harder tasks with natural uncertainty.

        Args:
            tier: One of "worker", "coder", "architect", "ingest".

        Returns:
            MonitorConfig with tier-appropriate thresholds.
        """
        cfg = _monitor_cfg()
        overrides = cfg.tier_overrides.get(tier, {})
        if not overrides:
            return cls()
        return cls(**{k: v for k, v in overrides.items()
                      if k in cls.__dataclass_fields__})

    @classmethod
    def for_task(cls, task_type: str) -> MonitorConfig:
        """Get task-specific configuration from centralized config.

        Args:
            task_type: One of "code", "reasoning", "general".

        Returns:
            MonitorConfig with task-appropriate thresholds.
        """
        cfg = _monitor_cfg()
        overrides = cfg.task_overrides.get(task_type, {})
        if not overrides:
            return cls()
        return cls(**{k: v for k, v in overrides.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class GenerationHealth:
    """Real-time health metrics during generation.

    Attributes:
        tokens_generated: Total tokens generated so far.
        avg_token_entropy: Rolling average entropy.
        current_entropy: Most recent token's entropy.
        max_entropy_spike: Largest single-token entropy jump.
        repetition_ratio: Fraction of repeated n-grams (0-1).
        perplexity_trend: Trend direction over recent window.
        rolling_perplexity: Current rolling perplexity value.
        estimated_failure_prob: Combined heuristic score (0-1).
        abort_reason: Reason if abort recommended, else NONE.
    """

    tokens_generated: int = 0
    avg_token_entropy: float = 0.0
    current_entropy: float = 0.0
    max_entropy_spike: float = 0.0
    repetition_ratio: float = 0.0
    perplexity_trend: PerplexityTrend = PerplexityTrend.STABLE
    rolling_perplexity: float = 0.0
    estimated_failure_prob: float = 0.0
    abort_reason: AbortReason = AbortReason.NONE


class GenerationMonitor:
    """Monitor generation health and signal early abort.

    Tracks per-token metrics during generation to detect likely
    failures early. Supports both real inference (with logits) and
    mock mode for testing abort logic.

    The monitor is stateful - create a new instance for each generation.
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
        mock_mode: bool = False,
        expected_length: int | None = None,
    ):
        """Initialize the generation monitor.

        Args:
            config: Monitoring configuration (thresholds, windows).
            mock_mode: If True, use synthetic metrics for testing.
            expected_length: Expected generation length (for runaway detection).
        """
        self.config = config if config is not None else MonitorConfig()
        self.mock_mode = mock_mode
        self.expected_length = expected_length

        # Tracking state
        self._token_ids: list[int] = []
        self._entropies: list[float] = []
        self._perplexities: list[float] = []
        self._high_entropy_count: int = 0

        # Mock mode state
        self._mock_scenario: str = "normal"
        self._mock_step: int = 0

        # Cached health for efficiency
        self._last_health: GenerationHealth = GenerationHealth()

    def update(
        self,
        token_id: int,
        logits: Sequence[float] | None = None,
    ) -> GenerationHealth:
        """Update metrics with new token.

        Called per-token during generation. In mock mode, logits
        can be None and synthetic metrics will be used.

        Args:
            token_id: The generated token ID.
            logits: Raw logits from the model (optional in mock mode).

        Returns:
            Current GenerationHealth with all metrics.
        """
        self._token_ids.append(token_id)

        if self.mock_mode:
            health = self._mock_update()
        else:
            if logits is None:
                raise ValueError("logits required in real mode")
            health = self._real_update(token_id, logits)

        self._last_health = health
        return health

    def should_abort(self) -> tuple[bool, AbortReason]:
        """Check if generation should be aborted early.

        Returns:
            Tuple of (should_abort, reason).
        """
        health = self._last_health

        # Don't abort too early
        if health.tokens_generated < self.config.min_tokens_before_abort:
            return False, AbortReason.NONE

        # Check individual signals
        if health.abort_reason != AbortReason.NONE:
            return True, health.abort_reason

        # Check combined threshold
        if health.estimated_failure_prob >= self.config.combined_threshold:
            return True, AbortReason.COMBINED_SIGNALS

        return False, AbortReason.NONE

    def get_health(self) -> GenerationHealth:
        """Get the current health metrics.

        Returns:
            Most recent GenerationHealth.
        """
        return self._last_health

    def reset(self) -> None:
        """Reset monitor state for new generation."""
        self._token_ids.clear()
        self._entropies.clear()
        self._perplexities.clear()
        self._high_entropy_count = 0
        self._mock_step = 0
        self._last_health = GenerationHealth()

    def set_mock_scenario(self, scenario: str) -> None:
        """Set the mock scenario for testing.

        Args:
            scenario: One of "normal", "high_entropy", "entropy_spike",
                     "high_repetition", "rising_perplexity", "runaway".
        """
        if not self.mock_mode:
            raise RuntimeError("set_mock_scenario only valid in mock_mode")
        self._mock_scenario = scenario
        self._mock_step = 0

    def _real_update(
        self,
        token_id: int,
        logits: Sequence[float],
    ) -> GenerationHealth:
        """Update with real logits.

        Args:
            token_id: The generated token ID.
            logits: Raw logits from the model.

        Returns:
            Updated GenerationHealth.
        """
        # Compute entropy from logits
        entropy = self._compute_entropy(logits)
        self._entropies.append(entropy)

        # Compute perplexity
        perplexity = math.exp(entropy)
        self._perplexities.append(perplexity)

        # Track entropy spikes
        prev_entropy = self._entropies[-2] if len(self._entropies) > 1 else entropy
        entropy_spike = entropy - prev_entropy

        # Track sustained high entropy
        if entropy > self.config.entropy_threshold:
            self._high_entropy_count += 1
        else:
            self._high_entropy_count = 0

        # Compute repetition ratio
        repetition_ratio = self._compute_repetition_ratio()

        # Compute perplexity trend
        perplexity_trend = self._compute_perplexity_trend()

        # Compute rolling averages
        avg_entropy = sum(self._entropies) / len(self._entropies)
        rolling_perplexity = (
            sum(self._perplexities[-self.config.perplexity_window :])
            / min(len(self._perplexities), self.config.perplexity_window)
        )

        # Determine abort reason (if any)
        abort_reason = self._check_abort_conditions(
            entropy=entropy,
            entropy_spike=entropy_spike,
            repetition_ratio=repetition_ratio,
            perplexity_trend=perplexity_trend,
        )

        # Compute combined failure probability
        failure_prob = self._compute_failure_probability(
            entropy=entropy,
            entropy_spike=entropy_spike,
            repetition_ratio=repetition_ratio,
            perplexity_trend=perplexity_trend,
        )

        return GenerationHealth(
            tokens_generated=len(self._token_ids),
            avg_token_entropy=avg_entropy,
            current_entropy=entropy,
            max_entropy_spike=max(entropy_spike, self._last_health.max_entropy_spike),
            repetition_ratio=repetition_ratio,
            perplexity_trend=perplexity_trend,
            rolling_perplexity=rolling_perplexity,
            estimated_failure_prob=failure_prob,
            abort_reason=abort_reason,
        )

    def _mock_update(self) -> GenerationHealth:
        """Generate synthetic metrics for testing.

        Returns:
            Synthetic GenerationHealth based on mock scenario.
        """
        self._mock_step += 1
        n = len(self._token_ids)

        # Default values
        entropy = 2.0
        entropy_spike = 0.1
        repetition = 0.05
        perplexity_trend = PerplexityTrend.STABLE
        abort_reason = AbortReason.NONE

        if self._mock_scenario == "high_entropy":
            # Entropy gradually rises above threshold
            entropy = 2.0 + (n * 0.05)
            if entropy > self.config.entropy_threshold:
                self._high_entropy_count += 1
                if self._high_entropy_count >= self.config.entropy_sustained_count:
                    abort_reason = AbortReason.HIGH_ENTROPY

        elif self._mock_scenario == "entropy_spike":
            # Normal until sudden spike at token 60
            if n == 60:
                entropy = 7.0
                entropy_spike = 5.0
                abort_reason = AbortReason.ENTROPY_SPIKE
            else:
                entropy = 2.0 + (n * 0.01)

        elif self._mock_scenario == "high_repetition":
            # Repetition gradually increases
            repetition = min(0.05 + (n * 0.005), 0.8)
            if repetition > self.config.repetition_threshold:
                abort_reason = AbortReason.HIGH_REPETITION

        elif self._mock_scenario == "rising_perplexity":
            # Perplexity monotonically rises
            entropy = 2.0 + (n * 0.03)
            perplexity_trend = PerplexityTrend.RISING
            if n > self.config.perplexity_window * 2:
                abort_reason = AbortReason.RISING_PERPLEXITY

        elif self._mock_scenario == "runaway":
            # Simulate runaway generation
            if self.expected_length and n > self.expected_length * self.config.max_length_multiplier:
                abort_reason = AbortReason.RUNAWAY_LENGTH

        # Track for averages
        self._entropies.append(entropy)
        self._perplexities.append(math.exp(entropy))

        # Compute failure probability
        failure_prob = self._compute_failure_probability(
            entropy=entropy,
            entropy_spike=entropy_spike,
            repetition_ratio=repetition,
            perplexity_trend=perplexity_trend,
        )

        return GenerationHealth(
            tokens_generated=n,
            avg_token_entropy=sum(self._entropies) / len(self._entropies),
            current_entropy=entropy,
            max_entropy_spike=max(entropy_spike, self._last_health.max_entropy_spike),
            repetition_ratio=repetition,
            perplexity_trend=perplexity_trend,
            rolling_perplexity=math.exp(entropy),
            estimated_failure_prob=failure_prob,
            abort_reason=abort_reason,
        )

    def _compute_entropy(self, logits: Sequence[float]) -> float:
        """Compute Shannon entropy from logits.

        Args:
            logits: Raw logits from the model.

        Returns:
            Entropy value (in nats).
        """
        # Softmax to get probabilities
        max_logit = max(logits)
        exp_logits = [math.exp(x - max_logit) for x in logits]
        sum_exp = sum(exp_logits)
        probs = [x / sum_exp for x in exp_logits]

        # Shannon entropy: -sum(p * log(p))
        entropy = 0.0
        for p in probs:
            if p > 1e-10:
                entropy -= p * math.log(p)

        return entropy

    def _compute_repetition_ratio(self) -> float:
        """Compute ratio of repeated n-grams.

        Returns:
            Fraction of tokens that are part of repeated n-grams (0-1).
        """
        if len(self._token_ids) < self.config.ngram_size * 2:
            return 0.0

        # Extract n-grams
        ngrams = []
        for i in range(len(self._token_ids) - self.config.ngram_size + 1):
            ngram = tuple(self._token_ids[i : i + self.config.ngram_size])
            ngrams.append(ngram)

        if not ngrams:
            return 0.0

        # Count frequencies
        counter = Counter(ngrams)
        repeated_count = sum(count for count in counter.values() if count > 1)

        return repeated_count / len(ngrams)

    def _compute_perplexity_trend(self) -> PerplexityTrend:
        """Determine perplexity trend over recent window.

        Returns:
            PerplexityTrend enum value.
        """
        window = self.config.perplexity_window
        if len(self._perplexities) < window:
            return PerplexityTrend.STABLE

        recent = self._perplexities[-window:]

        # Check for spikes first
        mean_ppl = sum(recent) / len(recent)
        max_ppl = max(recent)
        if max_ppl > mean_ppl * 2:
            return PerplexityTrend.SPIKING

        # Count increasing and decreasing transitions
        increasing_count = sum(
            1 for i in range(1, len(recent)) if recent[i] > recent[i - 1]
        )
        decreasing_count = sum(
            1 for i in range(1, len(recent)) if recent[i] < recent[i - 1]
        )

        # Determine trend direction
        # If mostly stable (neither mostly increasing nor decreasing), return STABLE
        if increasing_count > window * 0.7:
            return PerplexityTrend.RISING
        elif decreasing_count > window * 0.7:
            return PerplexityTrend.FALLING
        else:
            return PerplexityTrend.STABLE

    def _check_abort_conditions(
        self,
        entropy: float,
        entropy_spike: float,
        repetition_ratio: float,
        perplexity_trend: PerplexityTrend,
    ) -> AbortReason:
        """Check individual abort conditions.

        Args:
            entropy: Current token entropy.
            entropy_spike: Entropy change from previous token.
            repetition_ratio: Current repetition ratio.
            perplexity_trend: Current perplexity trend.

        Returns:
            AbortReason if condition met, else NONE.
        """
        # Check sustained high entropy
        if self._high_entropy_count >= self.config.entropy_sustained_count:
            return AbortReason.HIGH_ENTROPY

        # Check entropy spike
        if entropy_spike > self.config.entropy_spike_threshold:
            return AbortReason.ENTROPY_SPIKE

        # Check repetition
        if repetition_ratio > self.config.repetition_threshold:
            return AbortReason.HIGH_REPETITION

        # Check runaway length
        if self.expected_length:
            max_length = self.expected_length * self.config.max_length_multiplier
            if len(self._token_ids) > max_length:
                return AbortReason.RUNAWAY_LENGTH

        # Check rising perplexity (only after sufficient window)
        if (
            perplexity_trend == PerplexityTrend.RISING
            and len(self._perplexities) > self.config.perplexity_window * 2
        ):
            return AbortReason.RISING_PERPLEXITY

        return AbortReason.NONE

    def _compute_failure_probability(
        self,
        entropy: float,
        entropy_spike: float,
        repetition_ratio: float,
        perplexity_trend: PerplexityTrend,
    ) -> float:
        """Compute combined failure probability from signals.

        Weighted combination of normalized signals. Weights based on
        empirical effectiveness from literature.

        Args:
            entropy: Current token entropy.
            entropy_spike: Entropy change from previous token.
            repetition_ratio: Current repetition ratio.
            perplexity_trend: Current perplexity trend.

        Returns:
            Estimated failure probability (0-1).
        """
        # Normalize signals to 0-1 range
        entropy_score = min(entropy / (self.config.entropy_threshold * 1.5), 1.0)
        spike_score = min(entropy_spike / (self.config.entropy_spike_threshold * 1.5), 1.0)
        repetition_score = min(repetition_ratio / self.config.repetition_threshold, 1.0)

        trend_score = 0.0
        if perplexity_trend == PerplexityTrend.RISING:
            trend_score = 0.5
        elif perplexity_trend == PerplexityTrend.SPIKING:
            trend_score = 0.8

        # Weighted combination
        # Weights from empirical studies: entropy most predictive
        weights = {
            "entropy": 0.35,
            "spike": 0.25,
            "repetition": 0.25,
            "trend": 0.15,
        }

        combined = (
            weights["entropy"] * entropy_score
            + weights["spike"] * spike_score
            + weights["repetition"] * repetition_score
            + weights["trend"] * trend_score
        )

        return min(combined, 1.0)

    def get_summary(self) -> dict:
        """Get a summary of the monitoring session.

        Returns:
            Dict with key metrics and decisions.
        """
        health = self._last_health
        should_abort, reason = self.should_abort()

        return {
            "tokens_generated": health.tokens_generated,
            "avg_entropy": round(health.avg_token_entropy, 3),
            "max_entropy_spike": round(health.max_entropy_spike, 3),
            "repetition_ratio": round(health.repetition_ratio, 3),
            "perplexity_trend": health.perplexity_trend.value,
            "failure_probability": round(health.estimated_failure_prob, 3),
            "should_abort": should_abort,
            "abort_reason": reason.value,
            "config": {
                "min_tokens": self.config.min_tokens_before_abort,
                "entropy_threshold": self.config.entropy_threshold,
                "repetition_threshold": self.config.repetition_threshold,
            },
        }

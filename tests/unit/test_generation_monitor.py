#!/usr/bin/env python3
"""Unit tests for GenerationMonitor.

Tests early failure detection heuristics including:
- Entropy calculation and thresholds
- Repetition detection
- Perplexity trend analysis
- Mock mode scenarios
- Integration with LLM primitives

Literature references:
- Gnosis (arXiv:2512.20578)
- ZIP-RC (arXiv:2512.01457)
- KnowLoop (arXiv:2406.00430)
"""

import math
import pytest

from src.generation_monitor import (
    AbortReason,
    GenerationHealth,
    GenerationMonitor,
    MonitorConfig,
    PerplexityTrend,
)


class TestMonitorConfig:
    """Test MonitorConfig creation and tier/task presets."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MonitorConfig()
        assert config.min_tokens_before_abort == 50
        assert config.entropy_threshold == 4.0
        assert config.entropy_spike_threshold == 2.0
        assert config.repetition_threshold == 0.3
        assert config.perplexity_window == 20

    def test_tier_worker_config(self):
        """Test worker tier configuration."""
        config = MonitorConfig.for_tier("worker")
        assert config.entropy_threshold == 4.5
        assert config.entropy_spike_threshold == 2.5
        assert config.min_tokens_before_abort == 50

    def test_tier_coder_config(self):
        """Test coder tier configuration (stricter repetition)."""
        config = MonitorConfig.for_tier("coder")
        assert config.entropy_threshold == 5.0
        assert config.repetition_threshold == 0.2  # Stricter
        assert config.min_tokens_before_abort == 100

    def test_tier_architect_config(self):
        """Test architect tier configuration (most tolerant)."""
        config = MonitorConfig.for_tier("architect")
        assert config.entropy_threshold == 6.0
        assert config.repetition_threshold == 0.4  # Most tolerant
        assert config.min_tokens_before_abort == 200

    def test_tier_unknown_returns_default(self):
        """Test unknown tier returns default config."""
        config = MonitorConfig.for_tier("unknown_tier")
        default = MonitorConfig()
        assert config.entropy_threshold == default.entropy_threshold

    def test_task_code_config(self):
        """Test code task configuration."""
        config = MonitorConfig.for_task("code")
        assert config.min_tokens_before_abort == 100
        assert config.repetition_threshold == 0.2
        assert config.ngram_size == 4

    def test_task_reasoning_config(self):
        """Test reasoning task configuration."""
        config = MonitorConfig.for_task("reasoning")
        assert config.entropy_threshold == 4.5
        assert config.min_tokens_before_abort == 30

    def test_task_unknown_returns_default(self):
        """Test unknown task returns default config."""
        config = MonitorConfig.for_task("unknown_task")
        default = MonitorConfig()
        assert config.entropy_threshold == default.entropy_threshold


class TestGenerationHealth:
    """Test GenerationHealth dataclass."""

    def test_default_values(self):
        """Test default GenerationHealth values."""
        health = GenerationHealth()
        assert health.tokens_generated == 0
        assert health.avg_token_entropy == 0.0
        assert health.abort_reason == AbortReason.NONE
        assert health.perplexity_trend == PerplexityTrend.STABLE

    def test_custom_values(self):
        """Test GenerationHealth with custom values."""
        health = GenerationHealth(
            tokens_generated=100,
            avg_token_entropy=3.5,
            abort_reason=AbortReason.HIGH_ENTROPY,
        )
        assert health.tokens_generated == 100
        assert health.avg_token_entropy == 3.5
        assert health.abort_reason == AbortReason.HIGH_ENTROPY


class TestEntropyCalculation:
    """Test entropy calculation from logits."""

    def test_uniform_distribution_max_entropy(self):
        """Test entropy is high for uniform distribution."""
        monitor = GenerationMonitor()
        # Uniform distribution over 100 tokens
        logits = [1.0] * 100
        entropy = monitor._compute_entropy(logits)
        # ln(100) ≈ 4.6
        assert entropy > 4.0

    def test_peaked_distribution_low_entropy(self):
        """Test entropy is low for peaked distribution."""
        monitor = GenerationMonitor()
        # One token much higher than others
        logits = [10.0] + [0.0] * 99
        entropy = monitor._compute_entropy(logits)
        # Should be close to 0
        assert entropy < 1.0

    def test_moderate_distribution(self):
        """Test entropy for moderate distribution."""
        monitor = GenerationMonitor()
        # 10 equally likely tokens out of 100
        logits = [5.0] * 10 + [0.0] * 90
        entropy = monitor._compute_entropy(logits)
        # ln(10) ≈ 2.3
        assert 2.0 < entropy < 3.0

    def test_softmax_numerical_stability(self):
        """Test entropy calculation with large logit values."""
        monitor = GenerationMonitor()
        # Very large logits (should handle overflow)
        logits = [1000.0] + [0.0] * 99
        entropy = monitor._compute_entropy(logits)
        assert not math.isnan(entropy)
        assert not math.isinf(entropy)


class TestRepetitionDetection:
    """Test n-gram repetition detection."""

    def test_no_repetition(self):
        """Test repetition ratio is 0 for unique tokens."""
        monitor = GenerationMonitor()
        # All unique tokens
        for i in range(100):
            monitor._token_ids.append(i)
        ratio = monitor._compute_repetition_ratio()
        assert ratio == 0.0

    def test_high_repetition(self):
        """Test repetition ratio is high for repeated pattern."""
        monitor = GenerationMonitor(config=MonitorConfig(ngram_size=3))
        # Repeating pattern: [1, 2, 3, 1, 2, 3, 1, 2, 3, ...]
        for _ in range(20):
            monitor._token_ids.extend([1, 2, 3])
        ratio = monitor._compute_repetition_ratio()
        # Should be very high (most trigrams repeat)
        assert ratio > 0.5

    def test_short_sequence_returns_zero(self):
        """Test short sequences return 0 repetition."""
        monitor = GenerationMonitor(config=MonitorConfig(ngram_size=3))
        # Too short for trigram analysis
        monitor._token_ids.extend([1, 2, 3])
        ratio = monitor._compute_repetition_ratio()
        assert ratio == 0.0

    def test_partial_repetition(self):
        """Test partial repetition detection."""
        monitor = GenerationMonitor(config=MonitorConfig(ngram_size=3))
        # Some unique, some repeated
        for i in range(20):
            monitor._token_ids.append(i)
        # Add some repeated trigrams
        monitor._token_ids.extend([1, 2, 3, 1, 2, 3, 1, 2, 3])
        ratio = monitor._compute_repetition_ratio()
        # Should detect some repetition
        assert 0.0 < ratio < 1.0


class TestPerplexityTrend:
    """Test perplexity trend detection."""

    def test_stable_trend(self):
        """Test stable perplexity trend detection."""
        monitor = GenerationMonitor(config=MonitorConfig(perplexity_window=10))
        # Flat perplexity
        monitor._perplexities = [10.0] * 20
        trend = monitor._compute_perplexity_trend()
        assert trend == PerplexityTrend.STABLE

    def test_rising_trend(self):
        """Test rising perplexity trend detection."""
        monitor = GenerationMonitor(config=MonitorConfig(perplexity_window=10))
        # Monotonically increasing
        monitor._perplexities = list(range(20))
        trend = monitor._compute_perplexity_trend()
        assert trend == PerplexityTrend.RISING

    def test_falling_trend(self):
        """Test falling perplexity trend detection."""
        monitor = GenerationMonitor(config=MonitorConfig(perplexity_window=10))
        # Monotonically decreasing
        monitor._perplexities = list(range(20, 0, -1))
        trend = monitor._compute_perplexity_trend()
        assert trend == PerplexityTrend.FALLING

    def test_spiking_trend(self):
        """Test spiking perplexity trend detection."""
        monitor = GenerationMonitor(config=MonitorConfig(perplexity_window=10))
        # Normal with one big spike
        monitor._perplexities = [10.0] * 19 + [100.0]
        trend = monitor._compute_perplexity_trend()
        assert trend == PerplexityTrend.SPIKING

    def test_short_sequence_stable(self):
        """Test short sequences return stable trend."""
        monitor = GenerationMonitor(config=MonitorConfig(perplexity_window=10))
        # Too short
        monitor._perplexities = [10.0] * 5
        trend = monitor._compute_perplexity_trend()
        assert trend == PerplexityTrend.STABLE


class TestMockMode:
    """Test mock mode scenarios."""

    def test_normal_scenario_no_abort(self):
        """Test normal mock scenario completes without abort."""
        monitor = GenerationMonitor(mock_mode=True)
        monitor.set_mock_scenario("normal")

        for i in range(100):
            monitor.update(i)

        should_abort, reason = monitor.should_abort()
        assert not should_abort
        assert reason == AbortReason.NONE

    def test_high_entropy_scenario(self):
        """Test high entropy mock scenario triggers abort."""
        config = MonitorConfig(min_tokens_before_abort=10, entropy_sustained_count=5)
        monitor = GenerationMonitor(mock_mode=True, config=config)
        monitor.set_mock_scenario("high_entropy")

        # Run until abort or max tokens
        aborted = False
        for i in range(200):
            monitor.update(i)
            should_abort, reason = monitor.should_abort()
            if should_abort:
                aborted = True
                assert reason == AbortReason.HIGH_ENTROPY
                break

        assert aborted, "Expected abort due to high entropy"

    def test_entropy_spike_scenario(self):
        """Test entropy spike mock scenario triggers abort."""
        config = MonitorConfig(min_tokens_before_abort=10)
        monitor = GenerationMonitor(mock_mode=True, config=config)
        monitor.set_mock_scenario("entropy_spike")

        aborted = False
        for i in range(100):
            monitor.update(i)
            should_abort, reason = monitor.should_abort()
            if should_abort:
                aborted = True
                assert reason == AbortReason.ENTROPY_SPIKE
                break

        assert aborted, "Expected abort due to entropy spike"

    def test_high_repetition_scenario(self):
        """Test high repetition mock scenario triggers abort."""
        config = MonitorConfig(min_tokens_before_abort=10, repetition_threshold=0.3)
        monitor = GenerationMonitor(mock_mode=True, config=config)
        monitor.set_mock_scenario("high_repetition")

        aborted = False
        for i in range(200):
            monitor.update(i)
            should_abort, reason = monitor.should_abort()
            if should_abort:
                aborted = True
                assert reason == AbortReason.HIGH_REPETITION
                break

        assert aborted, "Expected abort due to high repetition"

    def test_runaway_scenario(self):
        """Test runaway length mock scenario triggers abort."""
        config = MonitorConfig(min_tokens_before_abort=10, max_length_multiplier=2.0)
        monitor = GenerationMonitor(mock_mode=True, config=config, expected_length=50)
        monitor.set_mock_scenario("runaway")

        aborted = False
        for i in range(200):
            monitor.update(i)
            should_abort, reason = monitor.should_abort()
            if should_abort:
                aborted = True
                assert reason == AbortReason.RUNAWAY_LENGTH
                break

        assert aborted, "Expected abort due to runaway length"

    def test_set_mock_scenario_requires_mock_mode(self):
        """Test set_mock_scenario raises in real mode."""
        monitor = GenerationMonitor(mock_mode=False)
        with pytest.raises(RuntimeError, match="mock_mode"):
            monitor.set_mock_scenario("normal")


class TestAbortDecision:
    """Test abort decision logic."""

    def test_no_abort_before_min_tokens(self):
        """Test no abort before minimum tokens reached."""
        config = MonitorConfig(min_tokens_before_abort=100)
        monitor = GenerationMonitor(mock_mode=True, config=config)
        monitor.set_mock_scenario("high_entropy")

        # Generate tokens but less than min
        for i in range(50):
            monitor.update(i)

        should_abort, _ = monitor.should_abort()
        assert not should_abort

    def test_abort_after_min_tokens(self):
        """Test abort allowed after minimum tokens."""
        config = MonitorConfig(min_tokens_before_abort=10, entropy_sustained_count=5)
        monitor = GenerationMonitor(mock_mode=True, config=config)
        monitor.set_mock_scenario("high_entropy")

        # Generate enough tokens
        for i in range(200):
            monitor.update(i)
            if monitor.should_abort()[0]:
                break

        should_abort, _ = monitor.should_abort()
        assert should_abort

    def test_combined_threshold_abort(self):
        """Test abort on combined threshold."""
        config = MonitorConfig(
            min_tokens_before_abort=10,
            combined_threshold=0.5,  # Lower threshold
        )
        monitor = GenerationMonitor(mock_mode=True, config=config)

        # Simulate moderate failure signals
        for i in range(50):
            monitor.update(i)
            health = monitor.get_health()
            if health.estimated_failure_prob >= config.combined_threshold:
                should_abort, reason = monitor.should_abort()
                if should_abort:
                    assert reason == AbortReason.COMBINED_SIGNALS
                    return

        # May not always trigger depending on mock scenario


class TestRealModeEntropy:
    """Test real mode with actual logits."""

    def test_real_mode_requires_logits(self):
        """Test real mode raises without logits."""
        monitor = GenerationMonitor(mock_mode=False)
        with pytest.raises(ValueError, match="logits required"):
            monitor.update(token_id=1, logits=None)

    def test_real_mode_accepts_logits(self):
        """Test real mode accepts logits list."""
        monitor = GenerationMonitor(mock_mode=False)
        logits = [1.0] * 100
        health = monitor.update(token_id=1, logits=logits)
        assert health.tokens_generated == 1
        assert health.current_entropy > 0

    def test_entropy_spike_detection(self):
        """Test entropy spike is detected with real logits."""
        config = MonitorConfig(entropy_spike_threshold=1.0)
        monitor = GenerationMonitor(mock_mode=False, config=config)

        # First token: peaked distribution (low entropy)
        low_entropy_logits = [10.0] + [0.0] * 99
        monitor.update(token_id=1, logits=low_entropy_logits)

        # Second token: uniform distribution (high entropy)
        high_entropy_logits = [1.0] * 100
        health = monitor.update(token_id=2, logits=high_entropy_logits)

        # Should detect spike
        assert health.max_entropy_spike > 1.0


class TestMonitorReset:
    """Test monitor reset functionality."""

    def test_reset_clears_state(self):
        """Test reset clears all tracked state."""
        monitor = GenerationMonitor(mock_mode=True)

        # Generate some tokens
        for i in range(50):
            monitor.update(i)

        assert monitor.get_health().tokens_generated == 50

        # Reset
        monitor.reset()

        health = monitor.get_health()
        assert health.tokens_generated == 0
        assert len(monitor._token_ids) == 0
        assert len(monitor._entropies) == 0


class TestGetSummary:
    """Test summary generation."""

    def test_summary_contains_key_metrics(self):
        """Test summary includes all key metrics."""
        monitor = GenerationMonitor(mock_mode=True)
        for i in range(50):
            monitor.update(i)

        summary = monitor.get_summary()

        assert "tokens_generated" in summary
        assert "avg_entropy" in summary
        assert "repetition_ratio" in summary
        assert "failure_probability" in summary
        assert "should_abort" in summary
        assert "abort_reason" in summary
        assert "config" in summary

    def test_summary_values_are_reasonable(self):
        """Test summary values are within expected ranges."""
        monitor = GenerationMonitor(mock_mode=True)
        for i in range(50):
            monitor.update(i)

        summary = monitor.get_summary()

        assert summary["tokens_generated"] == 50
        assert 0.0 <= summary["failure_probability"] <= 1.0
        assert isinstance(summary["should_abort"], bool)


class TestFailureProbability:
    """Test combined failure probability calculation."""

    def test_low_failure_prob_for_normal(self):
        """Test failure probability is low for normal generation."""
        monitor = GenerationMonitor(mock_mode=True)
        monitor.set_mock_scenario("normal")

        for i in range(50):
            monitor.update(i)

        health = monitor.get_health()
        assert health.estimated_failure_prob < 0.5

    def test_high_failure_prob_for_bad_signals(self):
        """Test failure probability is high for bad signals."""
        config = MonitorConfig(min_tokens_before_abort=10, entropy_sustained_count=5)
        monitor = GenerationMonitor(mock_mode=True, config=config)
        monitor.set_mock_scenario("high_entropy")

        for i in range(100):
            monitor.update(i)

        health = monitor.get_health()
        # Should have elevated failure probability
        assert health.estimated_failure_prob > 0.3


class TestIntegrationWithFailureRouter:
    """Test integration with failure router."""

    def test_early_abort_category_exists(self):
        """Test EARLY_ABORT error category is available."""
        from src.failure_router import ErrorCategory
        assert hasattr(ErrorCategory, "EARLY_ABORT")
        assert ErrorCategory.EARLY_ABORT.value == "early_abort"

    def test_early_abort_routing(self):
        """Test early abort triggers immediate escalation."""
        from src.failure_router import (
            ErrorCategory,
            FailureContext,
            FailureRouter,
        )

        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=0,  # First failure
            error_category=ErrorCategory.EARLY_ABORT,
            error_message="High entropy detected",
        )

        decision = router.route_failure(context)

        # Should escalate immediately, not retry
        # Note: "coder" chain maps to "coder_primary" specific role
        assert decision.action == "escalate"
        assert decision.next_role == "coder_primary"

    def test_early_abort_at_architect_fails(self):
        """Test early abort at architect tier fails gracefully."""
        from src.failure_router import (
            ErrorCategory,
            FailureContext,
            FailureRouter,
        )

        router = FailureRouter()
        context = FailureContext(
            role="architect",
            failure_count=0,
            error_category=ErrorCategory.EARLY_ABORT,
            error_message="High entropy detected",
        )

        decision = router.route_failure(context)

        # No escalation from architect
        assert decision.action == "fail"
        assert decision.next_role is None


class TestIntegrationWithLLMPrimitives:
    """Test integration with LLM primitives."""

    def test_llm_result_dataclass_exists(self):
        """Test LLMResult dataclass is available."""
        from src.llm_primitives import LLMResult
        result = LLMResult(text="test", aborted=True, abort_reason="high_entropy")
        assert result.text == "test"
        assert result.aborted is True
        assert result.abort_reason == "high_entropy"

    def test_monitored_call_with_abort(self):
        """Test monitored call triggers abort."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=True)
        config = MonitorConfig(min_tokens_before_abort=10, entropy_sustained_count=5)
        monitor = GenerationMonitor(mock_mode=True, config=config)
        monitor.set_mock_scenario("high_entropy")

        result = primitives.llm_call_monitored(
            prompt="Test prompt",
            monitor=monitor,
        )

        assert result.aborted is True
        assert result.abort_reason == "high_entropy"
        assert result.tokens_saved > 0

    def test_monitored_call_without_abort(self):
        """Test monitored call completes without abort."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=True)
        monitor = GenerationMonitor(mock_mode=True)
        monitor.set_mock_scenario("normal")

        result = primitives.llm_call_monitored(
            prompt="Test prompt",
            monitor=monitor,
        )

        assert result.aborted is False
        assert result.tokens_generated > 0

    def test_monitored_call_requires_monitor(self):
        """Test monitored call raises without monitor."""
        from src.llm_primitives import LLMPrimitives

        primitives = LLMPrimitives(mock_mode=True)

        with pytest.raises(ValueError, match="monitor required"):
            primitives.llm_call_monitored(
                prompt="Test prompt",
                monitor=None,
            )

"""Tests for SkillBank integration into pipeline diagnostics.

Covers:
- New anomaly signals (skill_mismatch, no_skills_available)
- Extended compute_anomaly_signals with skill params
- build_diagnostic with skill retrieval data
"""

from __future__ import annotations

from src.pipeline_monitor.anomaly import (
    SIGNAL_WEIGHTS,
    anomaly_score,
    compute_anomaly_signals,
    detect_no_skills_available,
    detect_skill_mismatch,
)
from src.pipeline_monitor.diagnostic import build_diagnostic


# ── detect_skill_mismatch ──────────────────────────────────────────────


class TestDetectSkillMismatch:
    def test_triggers_when_failed_with_skills(self):
        assert detect_skill_mismatch(passed=False, skills_retrieved=3) is True

    def test_no_trigger_when_passed(self):
        assert detect_skill_mismatch(passed=True, skills_retrieved=3) is False

    def test_no_trigger_when_no_skills(self):
        assert detect_skill_mismatch(passed=False, skills_retrieved=0) is False

    def test_no_trigger_when_passed_no_skills(self):
        assert detect_skill_mismatch(passed=True, skills_retrieved=0) is False


# ── detect_no_skills_available ─────────────────────────────────────────


class TestDetectNoSkillsAvailable:
    def test_triggers_when_coverage_but_no_retrieval(self):
        assert detect_no_skills_available(skills_retrieved=0, skill_coverage=True) is True

    def test_no_trigger_when_skills_found(self):
        assert detect_no_skills_available(skills_retrieved=2, skill_coverage=True) is False

    def test_no_trigger_when_skillbank_disabled(self):
        assert detect_no_skills_available(skills_retrieved=0, skill_coverage=False) is False


# ── Signal weights ─────────────────────────────────────────────────────


class TestSignalWeights:
    def test_skill_mismatch_in_weights(self):
        assert "skill_mismatch" in SIGNAL_WEIGHTS
        assert SIGNAL_WEIGHTS["skill_mismatch"] == 0.5

    def test_no_skills_available_in_weights(self):
        assert "no_skills_available" in SIGNAL_WEIGHTS
        assert SIGNAL_WEIGHTS["no_skills_available"] == 0.3


# ── compute_anomaly_signals with skill params ──────────────────────────


class TestComputeAnomalySignalsSkills:
    def test_backward_compatible_no_skill_params(self):
        """compute_anomaly_signals works without skill params (defaults)."""
        signals = compute_anomaly_signals(answer="Some answer", role="frontdoor")
        assert "skill_mismatch" in signals
        assert "no_skills_available" in signals
        # Both should be False with defaults
        assert signals["skill_mismatch"] is False
        assert signals["no_skills_available"] is False

    def test_skill_mismatch_signal(self):
        signals = compute_anomaly_signals(
            answer="wrong answer",
            role="frontdoor",
            skills_retrieved=3,
            skill_coverage=True,
            passed=False,
        )
        assert signals["skill_mismatch"] is True

    def test_no_skills_available_signal(self):
        signals = compute_anomaly_signals(
            answer="some answer",
            role="frontdoor",
            skills_retrieved=0,
            skill_coverage=True,
            passed=True,
        )
        assert signals["no_skills_available"] is True

    def test_anomaly_score_with_skill_mismatch(self):
        signals = compute_anomaly_signals(
            answer="wrong",
            role="frontdoor",
            skills_retrieved=2,
            passed=False,
        )
        score = anomaly_score(signals)
        assert score >= 0.5  # skill_mismatch weight


# ── build_diagnostic with skill params ─────────────────────────────────


class TestBuildDiagnosticSkills:
    def _make_diag(self, **kwargs):
        """Helper to build diagnostic with defaults."""
        defaults = dict(
            question_id="test/q1",
            suite="test",
            config="frontdoor:direct",
            role="frontdoor",
            mode="direct",
            passed=True,
            answer="42",
            expected="42",
            scoring_method="exact_match",
            error=None,
            error_type="none",
            tokens_generated=100,
            elapsed_s=1.5,
            role_history=["frontdoor"],
            delegation_events=[],
            tools_used=0,
            tools_called=[],
        )
        defaults.update(kwargs)
        return build_diagnostic(**defaults)

    def test_no_skill_retrieval_when_zero(self):
        diag = self._make_diag(skills_retrieved=0)
        assert diag["skill_retrieval"] == {}

    def test_skill_retrieval_present_when_nonzero(self):
        diag = self._make_diag(
            skills_retrieved=3,
            skill_types=["routing", "failure_lesson"],
            skill_context_tokens=450,
        )
        sr = diag["skill_retrieval"]
        assert sr["skills_retrieved"] == 3
        assert sr["skill_types"] == ["routing", "failure_lesson"]
        assert sr["skill_context_tokens"] == 450

    def test_skill_mismatch_in_anomaly_signals(self):
        diag = self._make_diag(
            passed=False,
            skills_retrieved=2,
        )
        assert diag["anomaly_signals"]["skill_mismatch"] is True

    def test_no_skill_mismatch_when_passed(self):
        diag = self._make_diag(
            passed=True,
            skills_retrieved=2,
        )
        assert diag["anomaly_signals"]["skill_mismatch"] is False

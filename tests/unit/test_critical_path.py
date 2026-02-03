"""Tests for critical path computation, step timing extraction, and report fields.

Covers:
- compute_critical_path(): empty, single, linear, diamond, independent, complex DAGs
- Slack computation: non-critical steps have positive slack
- Parallelism ratio: total_work / wall_clock
- extract_step_timings(): basic extraction, missing deps, order preservation
- CriticalPathReport: serialization
"""

from __future__ import annotations

import pytest

from src.metrics.critical_path import (
    CriticalPathReport,
    StepTiming,
    compute_critical_path,
)


# ── compute_critical_path ────────────────────────────────────────────────


class TestComputeCriticalPath:
    """Critical path DAG computation."""

    def test_empty_timings(self):
        """Empty input returns empty report."""
        report = compute_critical_path([])
        assert report.critical_path_steps == []
        assert report.critical_path_seconds == 0.0
        assert report.total_work_seconds == 0.0

    def test_single_step(self):
        """Single step is the entire critical path."""
        timings = [StepTiming("S1", 5.0)]
        report = compute_critical_path(timings, wall_clock_seconds=5.0)
        assert report.critical_path_steps == ["S1"]
        assert report.critical_path_seconds == 5.0
        assert report.total_work_seconds == 5.0
        assert report.step_slack == {"S1": 0.0}

    def test_linear_chain(self):
        """S1 → S2 → S3: critical path = sum of all."""
        timings = [
            StepTiming("S1", 2.0),
            StepTiming("S2", 3.0, depends_on=("S1",)),
            StepTiming("S3", 1.0, depends_on=("S2",)),
        ]
        report = compute_critical_path(timings, wall_clock_seconds=6.0)
        assert report.critical_path_steps == ["S1", "S2", "S3"]
        assert report.critical_path_seconds == 6.0
        assert report.total_work_seconds == 6.0
        # All steps on critical path → zero slack
        for sid in ["S1", "S2", "S3"]:
            assert report.step_slack[sid] == 0.0

    def test_diamond_dag(self):
        """S1 → {S2, S3} → S4: critical path through longer branch."""
        timings = [
            StepTiming("S1", 1.0),
            StepTiming("S2", 4.0, depends_on=("S1",)),
            StepTiming("S3", 2.0, depends_on=("S1",)),
            StepTiming("S4", 1.0, depends_on=("S2", "S3")),
        ]
        report = compute_critical_path(timings, wall_clock_seconds=6.0)
        # Critical path: S1 → S2 → S4 (1 + 4 + 1 = 6)
        assert report.critical_path_steps == ["S1", "S2", "S4"]
        assert report.critical_path_seconds == 6.0
        assert report.total_work_seconds == 8.0
        # S3: EF=3, LF=5 (S4 starts at 5), slack = 5-3 = 2
        assert report.step_slack["S3"] == pytest.approx(2.0)
        # S1, S2, S4 are on critical path → zero slack
        assert report.step_slack["S1"] == 0.0
        assert report.step_slack["S2"] == 0.0
        assert report.step_slack["S4"] == 0.0

    def test_independent_steps(self):
        """No dependencies: critical path = longest single step."""
        timings = [
            StepTiming("S1", 2.0),
            StepTiming("S2", 5.0),
            StepTiming("S3", 3.0),
        ]
        report = compute_critical_path(timings, wall_clock_seconds=5.0)
        assert report.critical_path_steps == ["S2"]
        assert report.critical_path_seconds == 5.0
        assert report.total_work_seconds == 10.0
        assert report.step_slack["S1"] == pytest.approx(3.0)
        assert report.step_slack["S2"] == 0.0
        assert report.step_slack["S3"] == pytest.approx(2.0)

    def test_slack_computation(self):
        """Non-critical steps have positive slack."""
        # S1 → S2 (critical path = 3)
        # S3 independent (elapsed = 1, slack = 2)
        timings = [
            StepTiming("S1", 1.0),
            StepTiming("S2", 2.0, depends_on=("S1",)),
            StepTiming("S3", 1.0),
        ]
        report = compute_critical_path(timings)
        assert report.critical_path_seconds == 3.0
        assert report.step_slack["S3"] == pytest.approx(2.0)
        assert report.step_slack["S1"] == 0.0
        assert report.step_slack["S2"] == 0.0

    def test_parallelism_ratio(self):
        """Parallelism ratio > 1 when parallel execution saves time."""
        # Two independent 5s steps → total_work=10, wall_clock=5
        timings = [
            StepTiming("S1", 5.0),
            StepTiming("S2", 5.0),
        ]
        report = compute_critical_path(timings, wall_clock_seconds=5.0)
        assert report.parallelism_ratio == pytest.approx(2.0)

    def test_parallelism_ratio_no_wall_clock(self):
        """Without wall_clock, uses critical path as denominator."""
        timings = [
            StepTiming("S1", 5.0),
            StepTiming("S2", 5.0),
        ]
        report = compute_critical_path(timings)
        # total_work=10, critical_path=5 → ratio=2.0
        assert report.parallelism_ratio == pytest.approx(2.0)

    def test_complex_dag(self):
        """6-step DAG with mixed dependencies.

        DAG:
            S1 (2s) → S2 (3s) → S5 (1s) → S6 (2s)
            S1 (2s) → S3 (1s) → S5 (1s)
            S4 (4s) → S6 (2s)

        Critical path: S4 (4) → S6 (2) = 6?  or  S1→S2→S5→S6 = 8?
        S1→S2→S5→S6: 2+3+1+2 = 8
        S4→S6: 4+2 = 6
        S1→S3→S5→S6: 2+1+1+2 = 6

        Critical path = S1→S2→S5→S6 = 8s
        """
        timings = [
            StepTiming("S1", 2.0),
            StepTiming("S2", 3.0, depends_on=("S1",)),
            StepTiming("S3", 1.0, depends_on=("S1",)),
            StepTiming("S4", 4.0),
            StepTiming("S5", 1.0, depends_on=("S2", "S3")),
            StepTiming("S6", 2.0, depends_on=("S5", "S4")),
        ]
        report = compute_critical_path(timings, wall_clock_seconds=8.0)
        assert report.critical_path_steps == ["S1", "S2", "S5", "S6"]
        assert report.critical_path_seconds == 8.0
        assert report.total_work_seconds == 13.0
        # S3: EF=3, LF=5 (LS[S5]=5), slack = 5-3 = 2
        assert report.step_slack["S3"] == pytest.approx(2.0)
        # S4: EF=4, LF=6 (LS[S6]=6), slack = 6-4 = 2
        assert report.step_slack["S4"] == pytest.approx(2.0)

    def test_circular_dependency_raises(self):
        """Circular dependencies are detected."""
        timings = [
            StepTiming("S1", 1.0, depends_on=("S2",)),
            StepTiming("S2", 1.0, depends_on=("S1",)),
        ]
        with pytest.raises(ValueError, match="Circular dependency"):
            compute_critical_path(timings)

    def test_unknown_dependency_raises(self):
        """Reference to nonexistent step raises ValueError."""
        timings = [
            StepTiming("S1", 1.0, depends_on=("S99",)),
        ]
        with pytest.raises(ValueError, match="unknown step"):
            compute_critical_path(timings)


# ── extract_step_timings ─────────────────────────────────────────────────


class TestExtractStepTimings:
    """extract_step_timings() utility function."""

    def _make_result(self, subtask_id: str, elapsed: float):
        from src.proactive_delegation import SubtaskResult

        return SubtaskResult(
            subtask_id=subtask_id,
            role="worker_general",
            output="done",
            success=True,
            elapsed_seconds=elapsed,
        )

    def test_basic_extraction(self):
        from src.parallel_step_executor import extract_step_timings

        results = [
            self._make_result("S1", 2.0),
            self._make_result("S2", 3.0),
        ]
        steps = [
            {"id": "S1"},
            {"id": "S2", "depends_on": ["S1"]},
        ]
        timings = extract_step_timings(results, steps)
        assert len(timings) == 2
        assert timings[0].step_id == "S1"
        assert timings[0].elapsed_seconds == 2.0
        assert timings[0].depends_on == ()
        assert timings[1].step_id == "S2"
        assert timings[1].elapsed_seconds == 3.0
        assert timings[1].depends_on == ("S1",)

    def test_missing_deps(self):
        """Steps without depends_on get empty tuple."""
        from src.parallel_step_executor import extract_step_timings

        results = [self._make_result("S1", 1.0)]
        steps = [{"id": "S1"}]  # No depends_on key
        timings = extract_step_timings(results, steps)
        assert timings[0].depends_on == ()

    def test_preserves_order(self):
        """Output order matches result order."""
        from src.parallel_step_executor import extract_step_timings

        results = [
            self._make_result("S3", 1.0),
            self._make_result("S1", 2.0),
            self._make_result("S2", 3.0),
        ]
        steps = [
            {"id": "S1"},
            {"id": "S2", "depends_on": ["S1"]},
            {"id": "S3"},
        ]
        timings = extract_step_timings(results, steps)
        assert [t.step_id for t in timings] == ["S3", "S1", "S2"]


# ── CriticalPathReport ───────────────────────────────────────────────────


class TestCriticalPathReport:
    """CriticalPathReport serialization."""

    def test_to_dict(self):
        report = CriticalPathReport(
            critical_path_steps=["S1", "S2"],
            critical_path_seconds=5.123456,
            total_work_seconds=8.654321,
            wall_clock_seconds=5.5,
            parallelism_ratio=1.573,
            step_slack={"S1": 0.0, "S2": 0.0, "S3": 2.123456},
        )
        d = report.to_dict()
        assert d["critical_path_steps"] == ["S1", "S2"]
        assert d["critical_path_seconds"] == 5.123
        assert d["total_work_seconds"] == 8.654
        assert d["wall_clock_seconds"] == 5.5
        assert d["parallelism_ratio"] == 1.573
        assert d["step_slack"]["S3"] == 2.123

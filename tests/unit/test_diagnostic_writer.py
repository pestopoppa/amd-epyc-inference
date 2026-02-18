"""Tests for pipeline_monitor diagnostic record builder and JSONL writer."""

from __future__ import annotations

import json
from pathlib import Path

from src.pipeline_monitor.diagnostic import build_diagnostic, append_diagnostic


class TestBuildDiagnostic:
    def test_basic_structure(self):
        diag = build_diagnostic(
            question_id="thinking/t1_q1",
            suite="thinking",
            config="SELF:direct",
            role="frontdoor",
            mode="direct",
            passed=True,
            answer="C",
            expected="C",
            scoring_method="multiple_choice",
            error=None,
            error_type="none",
            tokens_generated=50,
            elapsed_s=3.5,
            role_history=["frontdoor"],
            delegation_events=[],
            tools_used=0,
            tools_called=[],
        )
        assert diag["question_id"] == "thinking/t1_q1"
        assert diag["suite"] == "thinking"
        assert diag["passed"] is True
        assert diag["anomaly_signals"] is not None
        assert isinstance(diag["anomaly_score"], float)
        assert "ts" in diag
        assert diag["delegation_diagnostics"] == {}

    def test_delegation_diagnostics_included(self):
        diag = build_diagnostic(
            question_id="test",
            suite="coding",
            config="ARCHITECT:delegated",
            role="architect_coding",
            mode="delegated",
            passed=False,
            answer="",
            expected="",
            scoring_method="code_execution",
            error="timeout",
            error_type="timeout",
            tokens_generated=999,
            elapsed_s=90.0,
            role_history=["architect_coding", "coder_escalation"],
            delegation_events=[{"to_role": "coder_escalation"}],
            delegation_diagnostics={"break_reason": "role_repetition", "effective_max_loops": 2},
            tools_used=1,
            tools_called=["delegate"],
        )
        assert diag["delegation_diagnostics"]["break_reason"] == "role_repetition"

    def test_anomaly_signals_populated(self):
        diag = build_diagnostic(
            question_id="test",
            suite="thinking",
            config="ARCHITECT",
            role="architect_general",
            mode="delegated",
            passed=False,
            answer="After careful analysis of the quantum mechanics problem I believe the correct answer is B based on first principles",
            expected="C",
            scoring_method="exact_match",
            error=None,
            error_type="none",
            tokens_generated=100,
            elapsed_s=5.0,
            role_history=["architect_general"],
            delegation_events=[],
            tools_used=0,
            tools_called=[],
        )
        assert diag["anomaly_signals"]["format_violation"] is True
        assert diag["anomaly_score"] >= 1.0

    def test_tap_offset_included(self):
        diag = build_diagnostic(
            question_id="test",
            suite="thinking",
            config="SELF:direct",
            role="frontdoor",
            mode="direct",
            passed=True,
            answer="C",
            expected="C",
            scoring_method="multiple_choice",
            error=None,
            error_type="none",
            tokens_generated=10,
            elapsed_s=1.0,
            role_history=["frontdoor"],
            delegation_events=[],
            tools_used=0,
            tools_called=[],
            tap_offset_bytes=1024,
            tap_length_bytes=4096,
        )
        assert diag["tap_offset_bytes"] == 1024
        assert diag["tap_length_bytes"] == 4096


class TestAppendDiagnostic:
    def test_writes_valid_jsonl(self, tmp_path: Path):
        path = tmp_path / "test_diag.jsonl"
        diag = build_diagnostic(
            question_id="thinking/t1_q1",
            suite="thinking",
            config="SELF:direct",
            role="frontdoor",
            mode="direct",
            passed=True,
            answer="C",
            expected="C",
            scoring_method="multiple_choice",
            error=None,
            error_type="none",
            tokens_generated=10,
            elapsed_s=1.0,
            role_history=["frontdoor"],
            delegation_events=[],
            tools_used=0,
            tools_called=[],
        )
        append_diagnostic(diag, path=path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["question_id"] == "thinking/t1_q1"

    def test_appends_multiple(self, tmp_path: Path):
        path = tmp_path / "test_diag.jsonl"
        for i in range(3):
            diag = build_diagnostic(
                question_id=f"q{i}",
                suite="thinking",
                config="SELF:direct",
                role="frontdoor",
                mode="direct",
                passed=True,
                answer="C",
                expected="C",
                scoring_method="multiple_choice",
                error=None,
                error_type="none",
                tokens_generated=10,
                elapsed_s=1.0,
                role_history=["frontdoor"],
                delegation_events=[],
                tools_used=0,
                tools_called=[],
            )
            append_diagnostic(diag, path=path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            loaded = json.loads(line)
            assert loaded["question_id"] == f"q{i}"

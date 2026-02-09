"""Tests for pipeline_monitor change audit log."""

from __future__ import annotations

import json
from pathlib import Path

from src.pipeline_monitor.change_log import ChangeLog, _extract_modified_files, diff_snapshots


class TestExtractModifiedFiles:
    def test_single_file(self):
        diff = "diff --git a/src/foo.py b/src/foo.py\n--- a/src/foo.py\n+++ b/src/foo.py\n"
        assert _extract_modified_files(diff) == ["src/foo.py"]

    def test_multiple_files(self):
        diff = (
            "diff --git a/src/foo.py b/src/foo.py\n"
            "diff --git a/orchestration/prompts/arch.md b/orchestration/prompts/arch.md\n"
        )
        files = _extract_modified_files(diff)
        assert len(files) == 2
        assert "src/foo.py" in files
        assert "orchestration/prompts/arch.md" in files

    def test_empty_diff(self):
        assert _extract_modified_files("") == []


class TestChangeLog:
    def test_record_writes_valid_jsonl(self, tmp_path: Path):
        log_path = tmp_path / "changes.jsonl"
        cl = ChangeLog(log_path=log_path)

        cl.record(
            session_id="sess-123",
            batch_id=1,
            batch=[{"question_id": "q1", "anomaly_signals": {"format_violation": True}}],
            claude_response={"result": "Fixed the prompt format"},
            git_diff="diff --git a/prompts/arch.md b/prompts/arch.md\n",
            commit_sha="abc1234",
        )

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["session_id"] == "sess-123"
        assert entry["batch_id"] == 1
        assert entry["git_commit_sha"] == "abc1234"
        assert "q1" in entry["questions_analyzed"]
        assert "format_violation" in entry["anomalies_seen"].get("q1", {})

    def test_multiple_records_append(self, tmp_path: Path):
        log_path = tmp_path / "changes.jsonl"
        cl = ChangeLog(log_path=log_path)

        for i in range(3):
            cl.record(
                session_id="sess-123",
                batch_id=i,
                batch=[{"question_id": f"q{i}", "anomaly_signals": {}}],
                claude_response={"result": f"batch {i}"},
                git_diff="",
            )

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_anomalies_filtered(self, tmp_path: Path):
        log_path = tmp_path / "changes.jsonl"
        cl = ChangeLog(log_path=log_path)

        cl.record(
            session_id="sess",
            batch_id=0,
            batch=[
                {"question_id": "q1", "anomaly_signals": {"format_violation": True, "near_empty": False}},
                {"question_id": "q2", "anomaly_signals": {"format_violation": False, "near_empty": False}},
            ],
            claude_response={"result": "ok"},
            git_diff="",
        )

        entry = json.loads(log_path.read_text().strip())
        # q1 has active anomaly, q2 does not
        assert "q1" in entry["anomalies_seen"]
        assert "q2" not in entry["anomalies_seen"]


class TestDiffSnapshots:
    def test_no_changes(self):
        before = {"a.py": "abc", "b.py": "def"}
        after = {"a.py": "abc", "b.py": "def"}
        assert diff_snapshots(before, after) == []

    def test_file_changed(self):
        before = {"a.py": "abc", "b.py": "def"}
        after = {"a.py": "abc", "b.py": "xyz"}
        assert diff_snapshots(before, after) == ["b.py"]

    def test_file_added(self):
        before = {"a.py": "abc"}
        after = {"a.py": "abc", "b.py": "new"}
        assert diff_snapshots(before, after) == ["b.py"]

    def test_file_removed(self):
        before = {"a.py": "abc", "b.py": "def"}
        after = {"a.py": "abc"}
        assert diff_snapshots(before, after) == ["b.py"]

    def test_multiple_changes(self):
        before = {"a.py": "v1", "b.py": "v1", "c.py": "v1"}
        after = {"a.py": "v2", "b.py": "v1", "c.py": "v2"}
        assert diff_snapshots(before, after) == ["a.py", "c.py"]

    def test_empty_snapshots(self):
        assert diff_snapshots({}, {}) == []

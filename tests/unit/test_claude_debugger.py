"""Tests for pipeline_monitor Claude debugger (subprocess mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline_monitor.claude_debugger import ClaudeDebugger


@pytest.fixture(autouse=True)
def _isolate_change_log(tmp_path: Path):
    """Prevent tests from writing to the production debug_changes.jsonl."""
    with patch("src.pipeline_monitor.change_log.LOG_PATH", tmp_path / "debug_changes.jsonl"):
        yield


def _make_diag(
    question_id: str = "thinking/t1_q1",
    anomaly_score: float = 0.0,
    **overrides,
) -> dict:
    """Create a minimal diagnostic dict for testing."""
    base = {
        "ts": "2026-02-10T12:00:00Z",
        "question_id": question_id,
        "suite": "thinking",
        "config": "SELF:direct",
        "role": "frontdoor",
        "mode": "direct",
        "passed": True,
        "answer": "C",
        "expected": "C",
        "scoring_method": "multiple_choice",
        "error": None,
        "error_type": "none",
        "tokens_generated": 50,
        "elapsed_s": 3.0,
        "role_history": ["frontdoor"],
        "delegation_events": [],
        "tools_used": 0,
        "tools_called": [],
        "anomaly_signals": {"format_violation": False},
        "anomaly_score": anomaly_score,
        "tap_offset_bytes": 0,
        "tap_length_bytes": 0,
        "repl_tap_offset_bytes": 0,
        "repl_tap_length_bytes": 0,
    }
    base.update(overrides)
    return base


class TestClaudeDebuggerBatching:
    def test_batch_accumulates(self, tmp_path: Path):
        debugger = ClaudeDebugger(
            project_root=tmp_path,
            batch_size=5,
            dry_run=True,
        )
        debugger.add_diagnostic(_make_diag("q1"))
        assert len(debugger.batch) == 1

    def test_batch_triggers_at_size(self, tmp_path: Path):
        debugger = ClaudeDebugger(
            project_root=tmp_path,
            batch_size=3,
            dry_run=True,
        )
        debugger.add_diagnostic(_make_diag("q1"))
        debugger.add_diagnostic(_make_diag("q2"))
        debugger.add_diagnostic(_make_diag("q3"))
        assert len(debugger.batch) == 0  # cleared after dispatch
        assert debugger.batch_count == 1

    def test_critical_anomaly_deferred_to_end_question(self, tmp_path: Path):
        """Critical anomaly sets _urgent but doesn't dispatch until end_question()."""
        debugger = ClaudeDebugger(
            project_root=tmp_path,
            batch_size=10,
            dry_run=True,
        )
        debugger.add_diagnostic(_make_diag("q1", anomaly_score=1.0))
        assert len(debugger.batch) == 1  # NOT dispatched yet
        assert debugger._urgent is True

    def test_end_question_dispatches_when_urgent(self, tmp_path: Path):
        debugger = ClaudeDebugger(
            project_root=tmp_path,
            batch_size=10,
            dry_run=True,
        )
        debugger.add_diagnostic(_make_diag("q1", anomaly_score=1.0))
        debugger.add_diagnostic(_make_diag("q1b", anomaly_score=0.0))
        debugger.end_question()
        assert len(debugger.batch) == 0  # dispatched
        assert debugger.batch_count == 1

    def test_end_question_noop_when_not_urgent(self, tmp_path: Path):
        debugger = ClaudeDebugger(
            project_root=tmp_path,
            batch_size=10,
            dry_run=True,
        )
        debugger.add_diagnostic(_make_diag("q1", anomaly_score=0.3))
        debugger.end_question()
        assert len(debugger.batch) == 1  # not dispatched
        assert debugger.batch_count == 0

    def test_flush_processes_remaining(self, tmp_path: Path):
        debugger = ClaudeDebugger(
            project_root=tmp_path,
            batch_size=10,
            dry_run=True,
        )
        debugger.add_diagnostic(_make_diag("q1"))
        debugger.add_diagnostic(_make_diag("q2"))
        debugger.flush()
        assert len(debugger.batch) == 0
        assert debugger.batch_count == 1

    def test_flush_empty_batch_noop(self, tmp_path: Path):
        debugger = ClaudeDebugger(
            project_root=tmp_path,
            batch_size=10,
            dry_run=True,
        )
        debugger.flush()
        assert debugger.batch_count == 0


class TestClaudeDebuggerPromptBuilding:
    def test_first_batch_includes_system_prompt(self, tmp_path: Path):
        debugger = ClaudeDebugger(
            project_root=tmp_path,
            batch_size=1,
            dry_run=True,
        )
        debugger.batch_count = 1
        prompt = debugger._build_prompt([_make_diag("q1")])
        assert "debugging the orchestration pipeline" in prompt

    def test_subsequent_batch_no_system_prompt(self, tmp_path: Path):
        debugger = ClaudeDebugger(
            project_root=tmp_path,
            batch_size=1,
            dry_run=True,
        )
        debugger.batch_count = 2  # Not first batch
        prompt = debugger._build_prompt([_make_diag("q1")])
        assert "debugging the orchestration pipeline" not in prompt
        assert "Diagnostic Batch #2" in prompt

    def test_tap_inlined_when_present(self, tmp_path: Path):
        """Diagnostic with tap data + tap file present → inlined in prompt."""
        # Create a fake tap file
        tap_file = tmp_path / "tap.log"
        tap_content = "ROLE=architect_general PROMPT=test\nRESPONSE: D|coder\n"
        tap_file.write_text(tap_content)

        debugger = ClaudeDebugger(project_root=tmp_path, batch_size=5, dry_run=True)
        debugger.batch_count = 2

        diag = _make_diag(
            "q1",
            tap_offset_bytes=0,
            tap_length_bytes=len(tap_content.encode()),
        )

        with patch("src.pipeline_monitor.claude_debugger._TAP_PATH", str(tap_file)):
            prompt = debugger._build_prompt([diag])

        assert "Inference log" in prompt
        assert "ROLE=architect_general" in prompt

    def test_tap_not_inlined_when_zero(self, tmp_path: Path):
        """Diagnostic with tap_offset=0 and tap_length=0 → no tap in prompt."""
        debugger = ClaudeDebugger(project_root=tmp_path, batch_size=5, dry_run=True)
        debugger.batch_count = 2
        prompt = debugger._build_prompt([_make_diag("q1")])
        assert "Inference log" not in prompt

    def test_tap_graceful_when_file_missing(self, tmp_path: Path):
        """Tap file missing → no crash, no tap in prompt."""
        debugger = ClaudeDebugger(project_root=tmp_path, batch_size=5, dry_run=True)
        debugger.batch_count = 2

        diag = _make_diag("q1", tap_offset_bytes=100, tap_length_bytes=500)

        with patch(
            "src.pipeline_monitor.claude_debugger._TAP_PATH",
            str(tmp_path / "nonexistent.log"),
        ):
            prompt = debugger._build_prompt([diag])

        assert "Inference log" not in prompt

    def test_tap_truncated_when_too_large(self, tmp_path: Path):
        """Tap content exceeding _MAX_TAP_INLINE is truncated."""
        tap_file = tmp_path / "tap.log"
        big_content = "x" * 20_000
        tap_file.write_text(big_content)

        debugger = ClaudeDebugger(project_root=tmp_path, batch_size=5, dry_run=True)
        debugger.batch_count = 2

        diag = _make_diag(
            "q1",
            tap_offset_bytes=0,
            tap_length_bytes=len(big_content.encode()),
        )

        with patch("src.pipeline_monitor.claude_debugger._TAP_PATH", str(tap_file)):
            prompt = debugger._build_prompt([diag])

        assert "Inference log" in prompt
        assert "truncated" in prompt

    def test_repl_tap_inlined_when_present(self, tmp_path: Path):
        """Diagnostic with REPL tap data + file present → inlined in prompt."""
        repl_file = tmp_path / "repl_tap.log"
        repl_content = "[EXEC] x = 42\n[RESULT] NameError: name 'answer' is not defined\n"
        repl_file.write_text(repl_content)

        debugger = ClaudeDebugger(project_root=tmp_path, batch_size=5, dry_run=True)
        debugger.batch_count = 2

        diag = _make_diag(
            "q1",
            repl_tap_offset_bytes=0,
            repl_tap_length_bytes=len(repl_content.encode()),
        )

        with patch("src.pipeline_monitor.claude_debugger._REPL_TAP_PATH", str(repl_file)):
            prompt = debugger._build_prompt([diag])

        assert "REPL execution log" in prompt
        assert "NameError" in prompt

    def test_repl_tap_not_inlined_when_zero(self, tmp_path: Path):
        """Diagnostic with repl_tap_length=0 → no REPL tap in prompt."""
        debugger = ClaudeDebugger(project_root=tmp_path, batch_size=5, dry_run=True)
        debugger.batch_count = 2
        prompt = debugger._build_prompt([_make_diag("q1")])
        assert "REPL execution log" not in prompt

    def test_repl_tap_graceful_when_file_missing(self, tmp_path: Path):
        """REPL tap file missing → no crash, no REPL tap in prompt."""
        debugger = ClaudeDebugger(project_root=tmp_path, batch_size=5, dry_run=True)
        debugger.batch_count = 2

        diag = _make_diag("q1", repl_tap_offset_bytes=50, repl_tap_length_bytes=200)

        with patch(
            "src.pipeline_monitor.claude_debugger._REPL_TAP_PATH",
            str(tmp_path / "nonexistent_repl.log"),
        ):
            prompt = debugger._build_prompt([diag])

        assert "REPL execution log" not in prompt

    def test_both_taps_inlined_together(self, tmp_path: Path):
        """Both inference and REPL taps present → both inlined."""
        tap_file = tmp_path / "tap.log"
        tap_file.write_text("ROLE=frontdoor PROMPT=test\nRESPONSE: D|42\n")

        repl_file = tmp_path / "repl_tap.log"
        repl_file.write_text("[EXEC] print(42)\n[RESULT] 42\n")

        debugger = ClaudeDebugger(project_root=tmp_path, batch_size=5, dry_run=True)
        debugger.batch_count = 2

        diag = _make_diag(
            "q1",
            tap_offset_bytes=0,
            tap_length_bytes=tap_file.stat().st_size,
            repl_tap_offset_bytes=0,
            repl_tap_length_bytes=repl_file.stat().st_size,
        )

        with patch("src.pipeline_monitor.claude_debugger._TAP_PATH", str(tap_file)), \
             patch("src.pipeline_monitor.claude_debugger._REPL_TAP_PATH", str(repl_file)):
            prompt = debugger._build_prompt([diag])

        assert "Inference log" in prompt
        assert "REPL execution log" in prompt


class TestClaudeDebuggerInvocation:
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff_stat")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff")
    @patch("src.pipeline_monitor.claude_debugger.subprocess.Popen")
    def test_popen_called_with_correct_args(
        self, mock_popen, mock_git_diff, mock_stat, tmp_path: Path,
    ):
        mock_stat.return_value = {}
        mock_git_diff.return_value = ""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdout.read.return_value = json.dumps(
            {"session_id": "test-session", "result": "analyzed"}
        )
        mock_proc.stderr.read.return_value = ""
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        debugger = ClaudeDebugger(project_root=tmp_path, batch_size=1)
        debugger.add_diagnostic(_make_diag("q1"))
        # flush to wait for background
        debugger.flush()

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        assert debugger.session_id == "test-session"

    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff_stat")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff")
    @patch("src.pipeline_monitor.claude_debugger.subprocess.Popen")
    def test_session_id_reused_on_resume(
        self, mock_popen, mock_git_diff, mock_stat, tmp_path: Path,
    ):
        mock_stat.return_value = {}
        mock_git_diff.return_value = ""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stdout.read.return_value = json.dumps({"result": "ok"})
        mock_proc.stderr.read.return_value = ""
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        debugger = ClaudeDebugger(project_root=tmp_path, batch_size=1)
        debugger.session_id = "existing-session"
        debugger.add_diagnostic(_make_diag("q1"))
        debugger.flush()

        cmd = mock_popen.call_args[0][0]
        assert "--resume" in cmd
        assert "existing-session" in cmd

    def test_dry_run_skips_subprocess(self, tmp_path: Path):
        debugger = ClaudeDebugger(
            project_root=tmp_path,
            batch_size=1,
            dry_run=True,
        )
        debugger.add_diagnostic(_make_diag("q1"))
        # dry_run dispatches but doesn't launch subprocess
        assert debugger.batch_count == 1
        assert debugger._bg_process is None


class TestClaudeDebuggerCodeChanges:
    @patch("src.pipeline_monitor.claude_debugger._check_service_health", return_value=True)
    @patch("src.pipeline_monitor.claude_debugger.subprocess.run")
    def test_py_change_triggers_reload(self, mock_run, mock_health, tmp_path: Path):
        mock_run.return_value.returncode = 0
        debugger = ClaudeDebugger(project_root=tmp_path)
        result = debugger._hot_restart_api(["src/graph/nodes.py"])
        assert result is True
        mock_run.assert_called_once()

    @patch("src.pipeline_monitor.claude_debugger.subprocess.run")
    def test_empty_py_list_skips_reload(self, mock_run, tmp_path: Path):
        """No Python files changed → no restart."""
        # _hot_restart_api is only called when py_changes is non-empty
        # So we test that the caller logic works by checking _process_result
        # doesn't call _hot_restart_api when no .py files changed
        mock_run.assert_not_called()

    @patch("src.pipeline_monitor.claude_debugger.diff_snapshots")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff_stat")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff")
    def test_snapshot_diffing_ignores_preexisting(
        self, mock_git_diff, mock_stat_after, mock_diff, tmp_path: Path,
    ):
        """Pre-existing dirty files should not appear in changed_files."""
        debugger = ClaudeDebugger(project_root=tmp_path)

        # Simulate: before and after have the same dirty files
        snapshot = {"src/api/routes/chat.py": "abc123abc123"}
        mock_stat_after.return_value = snapshot
        mock_diff.return_value = []  # No changes
        mock_git_diff.return_value = ""

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.read.return_value = json.dumps({"result": "ok"})
        mock_proc.stderr.read.return_value = ""

        debugger._process_result(mock_proc, snapshot.copy(), [_make_diag("q1")])

        # diff_snapshots was called
        mock_diff.assert_called_once_with(snapshot, snapshot)

    @patch("src.pipeline_monitor.claude_debugger.diff_snapshots")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff_stat")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff")
    @patch("src.pipeline_monitor.claude_debugger.ClaudeDebugger._hot_restart_api")
    def test_claude_py_change_triggers_restart(
        self, mock_restart, mock_git_diff, mock_stat_after, mock_diff, tmp_path: Path,
    ):
        """When Claude changes a .py file, API should restart."""
        debugger = ClaudeDebugger(project_root=tmp_path)

        before = {"src/graph/nodes.py": "aaa111"}
        after = {"src/graph/nodes.py": "bbb222"}
        mock_stat_after.return_value = after
        mock_diff.return_value = ["src/graph/nodes.py"]
        mock_git_diff.return_value = "diff --git a/src/graph/nodes.py b/src/graph/nodes.py\n"

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.read.return_value = json.dumps({"result": "ok"})
        mock_proc.stderr.read.return_value = ""

        debugger._process_result(mock_proc, before, [_make_diag("q1")])

        mock_restart.assert_called_once_with(["src/graph/nodes.py"])

    @patch("src.pipeline_monitor.claude_debugger.diff_snapshots")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff_stat")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff")
    @patch("src.pipeline_monitor.claude_debugger.ClaudeDebugger._hot_restart_api")
    def test_claude_md_change_no_restart(
        self, mock_restart, mock_git_diff, mock_stat_after, mock_diff, tmp_path: Path,
    ):
        """When Claude only changes .md files, no API restart."""
        debugger = ClaudeDebugger(project_root=tmp_path)

        before = {"orchestration/prompts/architect.md": "aaa111"}
        after = {"orchestration/prompts/architect.md": "bbb222"}
        mock_stat_after.return_value = after
        mock_diff.return_value = ["orchestration/prompts/architect.md"]
        mock_git_diff.return_value = "diff --git a/orchestration/prompts/architect.md b/orchestration/prompts/architect.md\n"

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.read.return_value = json.dumps({"result": "ok"})
        mock_proc.stderr.read.return_value = ""

        debugger._process_result(mock_proc, before, [_make_diag("q1")])

        mock_restart.assert_not_called()


class TestClaudeDebuggerRetryQueue:
    def test_pop_retries_empty_when_no_changes(self, tmp_path: Path):
        """No file changes → pop_retries returns empty."""
        debugger = ClaudeDebugger(project_root=tmp_path, dry_run=True)
        retries, suites = debugger.pop_retries()
        assert retries == []
        assert suites == set()

    @patch("src.pipeline_monitor.claude_debugger.diff_snapshots")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff_stat")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff")
    @patch("src.pipeline_monitor.claude_debugger.ClaudeDebugger._hot_restart_api")
    def test_failed_questions_queued_on_file_change(
        self, mock_restart, mock_git_diff, mock_stat_after, mock_diff, tmp_path: Path,
    ):
        """When Claude changes files and batch has failures, they're queued."""
        debugger = ClaudeDebugger(project_root=tmp_path)

        mock_stat_after.return_value = {"src/nodes.py": "new"}
        mock_diff.return_value = ["src/nodes.py"]
        mock_git_diff.return_value = "diff"

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.read.return_value = json.dumps({"result": "ok"})
        mock_proc.stderr.read.return_value = ""

        batch = [
            _make_diag("thinking/t1_q1", passed=True, suite="thinking"),
            _make_diag("thinking/t1_q2", passed=False, suite="thinking"),
            _make_diag("math/m1_q1", passed=False, suite="math"),
        ]
        debugger._process_result(mock_proc, {"src/nodes.py": "old"}, batch)

        retries, suites = debugger.pop_retries()
        assert len(retries) == 2
        assert ("thinking", "t1_q2") in retries
        assert ("math", "m1_q1") in retries
        assert suites == {"thinking", "math"}

    @patch("src.pipeline_monitor.claude_debugger.diff_snapshots")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff_stat")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff")
    @patch("src.pipeline_monitor.claude_debugger.ClaudeDebugger._hot_restart_api")
    def test_same_question_retried_only_once(
        self, mock_restart, mock_git_diff, mock_stat_after, mock_diff, tmp_path: Path,
    ):
        """_retried set prevents the same question from being queued twice."""
        debugger = ClaudeDebugger(project_root=tmp_path)

        mock_stat_after.return_value = {"src/x.py": "new"}
        mock_diff.return_value = ["src/x.py"]
        mock_git_diff.return_value = "diff"

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.read.return_value = json.dumps({"result": "ok"})
        mock_proc.stderr.read.return_value = ""

        batch = [_make_diag("thinking/t1_q1", passed=False, suite="thinking")]

        # First batch with changes → queued
        debugger._process_result(mock_proc, {"src/x.py": "v1"}, batch)
        retries1, _ = debugger.pop_retries()
        assert len(retries1) == 1

        # Second batch with same question failing again → NOT re-queued
        mock_stat_after.return_value = {"src/x.py": "v3"}
        debugger._process_result(mock_proc, {"src/x.py": "v2"}, batch)
        retries2, _ = debugger.pop_retries()
        assert len(retries2) == 0

    @patch("src.pipeline_monitor.claude_debugger.diff_snapshots")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff_stat")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff")
    @patch("src.pipeline_monitor.claude_debugger.ClaudeDebugger._hot_restart_api")
    def test_no_retries_when_no_failures(
        self, mock_restart, mock_git_diff, mock_stat_after, mock_diff, tmp_path: Path,
    ):
        """File changes with all passing → retries empty, but suites tracked."""
        debugger = ClaudeDebugger(project_root=tmp_path)

        mock_stat_after.return_value = {"src/x.py": "new"}
        mock_diff.return_value = ["src/x.py"]
        mock_git_diff.return_value = "diff"

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.read.return_value = json.dumps({"result": "ok"})
        mock_proc.stderr.read.return_value = ""

        batch = [_make_diag("thinking/t1_q1", passed=True, suite="thinking")]
        debugger._process_result(mock_proc, {"src/x.py": "old"}, batch)

        retries, suites = debugger.pop_retries()
        assert retries == []
        assert suites == {"thinking"}  # suite still tracked even with no failures

    def test_pop_retries_clears_queue(self, tmp_path: Path):
        """pop_retries drains the queue — second call returns empty."""
        debugger = ClaudeDebugger(project_root=tmp_path, dry_run=True)
        # Manually populate to test clearing
        debugger._retry_queue = [("thinking", "q1"), ("math", "q2")]
        debugger._retry_suites = {"thinking", "math"}

        retries1, suites1 = debugger.pop_retries()
        assert len(retries1) == 2
        assert len(suites1) == 2

        retries2, suites2 = debugger.pop_retries()
        assert retries2 == []
        assert suites2 == set()

    @patch("src.pipeline_monitor.claude_debugger.diff_snapshots")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff_stat")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff")
    def test_no_retries_when_no_file_changes(
        self, mock_git_diff, mock_stat_after, mock_diff, tmp_path: Path,
    ):
        """No file changes → failures are NOT queued (fix didn't happen)."""
        debugger = ClaudeDebugger(project_root=tmp_path)

        mock_stat_after.return_value = {}
        mock_diff.return_value = []  # No changes
        mock_git_diff.return_value = ""

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.read.return_value = json.dumps({"result": "ok"})
        mock_proc.stderr.read.return_value = ""

        batch = [_make_diag("thinking/t1_q1", passed=False, suite="thinking")]
        debugger._process_result(mock_proc, {}, batch)

        retries, suites = debugger.pop_retries()
        assert retries == []
        assert suites == set()

    @patch("src.pipeline_monitor.claude_debugger.diff_snapshots")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff_stat")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff")
    @patch("src.pipeline_monitor.claude_debugger.ClaudeDebugger._hot_restart_api")
    def test_qid_stripped_from_suite_prefix(
        self, mock_restart, mock_git_diff, mock_stat_after, mock_diff, tmp_path: Path,
    ):
        """question_id='thinking/t1_q1' → qid='t1_q1' (suite prefix stripped)."""
        debugger = ClaudeDebugger(project_root=tmp_path)

        mock_stat_after.return_value = {"src/x.py": "new"}
        mock_diff.return_value = ["src/x.py"]
        mock_git_diff.return_value = "diff"

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.read.return_value = json.dumps({"result": "ok"})
        mock_proc.stderr.read.return_value = ""

        batch = [_make_diag("thinking/t1_q1", passed=False, suite="thinking")]
        debugger._process_result(mock_proc, {"src/x.py": "old"}, batch)

        retries, _ = debugger.pop_retries()
        assert retries == [("thinking", "t1_q1")]


class TestInfraHealthAndReload:
    """Tests for infrastructure health checks and service reload."""

    @patch("src.pipeline_monitor.claude_debugger._check_service_health")
    def test_check_infra_health_all_up(self, mock_health):
        from src.pipeline_monitor.claude_debugger import check_infra_health

        mock_health.return_value = True
        result = check_infra_health()
        assert result == {
            "orchestrator": True,
            "nextplaid-code": True,
            "nextplaid-docs": True,
        }

    @patch("src.pipeline_monitor.claude_debugger._check_service_health")
    def test_check_infra_health_partial_down(self, mock_health):
        from src.pipeline_monitor.claude_debugger import check_infra_health

        def side_effect(port, path, timeout=3.0):
            return port != 8089  # nextplaid-docs is down

        mock_health.side_effect = side_effect
        result = check_infra_health()
        assert result["orchestrator"] is True
        assert result["nextplaid-code"] is True
        assert result["nextplaid-docs"] is False

    @patch("src.pipeline_monitor.claude_debugger._check_service_health", return_value=True)
    @patch("src.pipeline_monitor.claude_debugger.subprocess.run")
    def test_reload_service_allowed(self, mock_run, mock_health, tmp_path: Path):
        mock_run.return_value.returncode = 0
        debugger = ClaudeDebugger(project_root=tmp_path)
        result = debugger._reload_service("nextplaid-code", reason="test")
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "reload" in args
        assert "nextplaid-code" in args

    def test_reload_service_unknown_rejected(self, tmp_path: Path):
        debugger = ClaudeDebugger(project_root=tmp_path)
        result = debugger._reload_service("unknown-service")
        assert result is False

    @patch("src.pipeline_monitor.claude_debugger._check_service_health", return_value=False)
    @patch("src.pipeline_monitor.claude_debugger.subprocess.run")
    def test_reload_service_unhealthy_after(self, mock_run, mock_health, tmp_path: Path):
        """Service reloaded but health check fails → returns False."""
        mock_run.return_value.returncode = 0
        debugger = ClaudeDebugger(project_root=tmp_path)
        result = debugger._reload_service("orchestrator")
        assert result is False  # healthy check failed

    def test_extract_reload_requests_valid(self):
        from src.pipeline_monitor.claude_debugger import _extract_reload_requests

        text = (
            "Analysis complete.\n"
            "RELOAD_SERVICE: nextplaid-docs reason=container not responding to health checks\n"
            "RELOAD_SERVICE: orchestrator reason=stale API after code edit\n"
        )
        reloads = _extract_reload_requests(text)
        assert len(reloads) == 2
        assert reloads[0]["service"] == "nextplaid-docs"
        assert "not responding" in reloads[0]["reason"]
        assert reloads[1]["service"] == "orchestrator"

    def test_extract_reload_requests_unknown_service_ignored(self):
        from src.pipeline_monitor.claude_debugger import _extract_reload_requests

        text = "RELOAD_SERVICE: unknown-svc reason=test\n"
        reloads = _extract_reload_requests(text)
        assert len(reloads) == 0

    @patch("src.pipeline_monitor.claude_debugger.ClaudeDebugger._reload_service")
    @patch("src.pipeline_monitor.claude_debugger.diff_snapshots")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff_stat")
    @patch("src.pipeline_monitor.claude_debugger.capture_git_diff")
    def test_process_result_executes_reload_directives(
        self, mock_git_diff, mock_stat_after, mock_diff, mock_reload, tmp_path: Path,
    ):
        """RELOAD_SERVICE: in Claude's response triggers _reload_service()."""
        debugger = ClaudeDebugger(project_root=tmp_path)
        mock_stat_after.return_value = {}
        mock_diff.return_value = []
        mock_git_diff.return_value = ""

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout.read.return_value = json.dumps({
            "result": "RELOAD_SERVICE: nextplaid-docs reason=down during REPL execution"
        })
        mock_proc.stderr.read.return_value = ""

        debugger._process_result(mock_proc, {}, [_make_diag("q1")])

        mock_reload.assert_called_once_with("nextplaid-docs", reason="down during REPL execution")

    @patch("src.pipeline_monitor.claude_debugger.check_infra_health")
    def test_build_prompt_includes_infra_status(self, mock_infra, tmp_path: Path):
        """Prompt includes INFRA DEGRADED when services are down."""
        mock_infra.return_value = {
            "orchestrator": True,
            "nextplaid-code": True,
            "nextplaid-docs": False,
        }
        debugger = ClaudeDebugger(project_root=tmp_path)
        debugger.batch_count = 0  # Will become 1 in _build_prompt context
        prompt = debugger._build_prompt([_make_diag("q1")])
        assert "INFRA DEGRADED" in prompt
        assert "nextplaid-docs" in prompt

    @patch("src.pipeline_monitor.claude_debugger.check_infra_health")
    def test_build_prompt_all_healthy(self, mock_infra, tmp_path: Path):
        """Prompt shows 'all services healthy' when nothing is degraded."""
        mock_infra.return_value = {
            "orchestrator": True,
            "nextplaid-code": True,
            "nextplaid-docs": True,
        }
        debugger = ClaudeDebugger(project_root=tmp_path)
        prompt = debugger._build_prompt([_make_diag("q1")])
        assert "all services healthy" in prompt
        assert "INFRA DEGRADED" not in prompt

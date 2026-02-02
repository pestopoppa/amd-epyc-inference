#!/usr/bin/env python3
"""Unit tests for REPL file mutation tools (_FileMutationMixin).

Tests cover: _log_append, _file_write_safe, _prepare_patch,
_list_patches, _apply_approved_patch, _reject_patch.

Patch tools are tested by calling mixin methods directly on REPLEnvironment
since they are not exposed in REPL globals.
"""

import glob
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.repl_environment import REPLConfig, REPLEnvironment, ExecutionResult


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def repl():
    """Create a REPLEnvironment for testing."""
    return REPLEnvironment(context="test context")


@pytest.fixture
def tmp_write_dir(tmp_path):
    """Provide a writable temp dir under /mnt/raid0/llm/tmp for path validation."""
    d = Path("/mnt/raid0/llm/tmp/test_file_mutation")
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def patches_base(tmp_path):
    """Create a patches directory structure for patch tool tests."""
    base = Path("/mnt/raid0/llm/claude/orchestration/patches")
    for sub in ("pending", "approved", "rejected"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    yield base


# ── log_append tests ─────────────────────────────────────────────────────

class TestLogAppend:
    """Test _log_append() method."""

    def test_append_writes_with_timestamp(self, repl):
        log_file = "/mnt/raid0/llm/claude/logs/test_mutation.log"
        try:
            if os.path.exists(log_file):
                os.remove(log_file)

            result = repl._log_append("test_mutation.log", "Hello from test")

            assert "Appended" in result
            with open(log_file) as f:
                content = f.read()
            assert "Hello from test" in content
            assert "[" in content  # timestamp bracket
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)

    def test_append_to_blocked_path(self, repl):
        """Log name that would resolve outside allowed paths."""
        # Path validation catches traversal
        result = repl._log_append("../../etc/passwd", "bad")
        assert "[ERROR:" in result

    def test_append_increments_exploration_calls(self, repl):
        before = repl._exploration_calls
        log_file = "/mnt/raid0/llm/claude/logs/test_counter.log"
        try:
            repl._log_append("test_counter.log", "msg")
            assert repl._exploration_calls == before + 1
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)


# ── file_write_safe tests ───────────────────────────────────────────────

class TestFileWriteSafe:
    """Test _file_write_safe() method."""

    def test_write_content(self, repl, tmp_write_dir):
        path = str(tmp_write_dir / "hello.txt")
        result = repl._file_write_safe(path, "Hello World", backup=False)
        assert "Wrote 11 bytes" in result
        assert Path(path).read_text() == "Hello World"

    def test_write_creates_parent_dirs(self, repl):
        deep = "/mnt/raid0/llm/tmp/test_deep/a/b/c/file.txt"
        try:
            result = repl._file_write_safe(deep, "nested", backup=False)
            assert "Wrote" in result
            assert Path(deep).read_text() == "nested"
        finally:
            shutil.rmtree("/mnt/raid0/llm/tmp/test_deep", ignore_errors=True)

    def test_write_with_backup(self, repl, tmp_write_dir):
        path = str(tmp_write_dir / "backup_test.txt")
        Path(path).write_text("original")

        result = repl._file_write_safe(path, "updated", backup=True)
        assert "Wrote" in result

        # Check backup created
        backups = glob.glob(f"{path}.bak.*")
        assert len(backups) == 1
        assert Path(backups[0]).read_text() == "original"
        assert Path(path).read_text() == "updated"

    def test_write_blocked_path(self, repl):
        result = repl._file_write_safe("/etc/passwd", "bad")
        assert "[ERROR:" in result

    def test_write_logs_exploration_event(self, repl, tmp_write_dir):
        path = str(tmp_write_dir / "event.txt")
        repl._file_write_safe(path, "data", backup=False)
        log = repl.get_exploration_log()
        assert any("file_write_safe" in str(e) for e in log.events)


# ── _prepare_patch tests ────────────────────────────────────────────────

class TestPreparePatch:
    """Test _prepare_patch() method (called directly, not via REPL globals)."""

    @patch("subprocess.run")
    def test_prepare_patch_creates_file(self, mock_run, repl, patches_base):
        mock_run.return_value = MagicMock(
            stdout="diff --git a/file.py b/file.py\n+new line\n"
        )

        result = repl._prepare_patch(["src/file.py"], "add feature")
        assert "Patch created:" in result
        assert "pending" in result

        # Verify patch file was created
        pending = list((patches_base / "pending").glob("*.patch"))
        assert len(pending) >= 1

        # Verify metadata header
        content = pending[-1].read_text()
        assert "# Patch: add feature" in content
        assert "PENDING APPROVAL" in content

        # Cleanup
        for p in pending:
            p.unlink()

    @patch("subprocess.run")
    def test_prepare_patch_no_changes(self, mock_run, repl):
        mock_run.return_value = MagicMock(stdout="")

        result = repl._prepare_patch(["src/file.py"], "nothing")
        assert "[INFO: No changes" in result

    def test_prepare_patch_increments_counter(self, repl):
        before = repl._exploration_calls
        with patch("subprocess.run", return_value=MagicMock(stdout="")):
            repl._prepare_patch(["f.py"], "x")
        assert repl._exploration_calls == before + 1


# ── _list_patches tests ─────────────────────────────────────────────────

class TestListPatches:
    """Test _list_patches() method."""

    def test_list_patches_empty(self, repl, patches_base):
        # Clear pending
        for p in (patches_base / "pending").glob("*.patch"):
            p.unlink()

        result = repl._list_patches("pending")
        assert "[INFO: No pending patches" in result

    def test_list_patches_finds_existing(self, repl, patches_base):
        # Create a test patch file
        patch_file = patches_base / "pending" / "test_list.patch"
        patch_file.write_text(
            "# Patch: test description\n"
            "# Created: 2026-01-01T00:00:00\n"
            "# Files: a.py, b.py\n"
            "# Status: PENDING APPROVAL\n"
        )
        try:
            result = repl._list_patches("pending")
            assert "test description" in result
            assert "a.py" in result
        finally:
            patch_file.unlink(missing_ok=True)

    def test_list_patches_all_statuses(self, repl, patches_base):
        """Test listing all statuses."""
        for status in ("pending", "approved", "rejected"):
            (patches_base / status / f"test_{status}.patch").write_text(
                f"# Patch: {status} patch\n# Created: 2026-01-01\n"
            )
        try:
            result = repl._list_patches("all")
            assert "pending patch" in result
            assert "approved patch" in result
            assert "rejected patch" in result
        finally:
            for status in ("pending", "approved", "rejected"):
                (patches_base / status / f"test_{status}.patch").unlink(missing_ok=True)


# ── _apply_approved_patch tests ──────────────────────────────────────────

class TestApplyApprovedPatch:
    """Test _apply_approved_patch() method."""

    def test_apply_nonexistent_patch(self, repl, patches_base):
        result = repl._apply_approved_patch("nonexistent.patch")
        assert "[ERROR: Patch not found" in result

    @patch("subprocess.run")
    def test_apply_patch_dry_run_failure(self, mock_run, repl, patches_base):
        # Create a pending patch
        patch_file = patches_base / "pending" / "fail.patch"
        patch_file.write_text("# Patch: fail\ndiff content\n")

        mock_run.return_value = MagicMock(returncode=1, stderr="conflict")
        try:
            result = repl._apply_approved_patch("fail.patch")
            assert "[ERROR:" in result
            assert "conflict" in result
        finally:
            patch_file.unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_apply_patch_success(self, mock_run, repl, patches_base):
        # Create a pending patch
        patch_file = patches_base / "pending" / "good.patch"
        patch_file.write_text("# Patch: good change\ndiff content\n")

        # First call = dry run (success), second call = apply (success)
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        try:
            result = repl._apply_approved_patch("good.patch")
            assert "applied successfully" in result
            assert (patches_base / "approved" / "good.patch").exists()
            assert not (patches_base / "pending" / "good.patch").exists()
        finally:
            (patches_base / "approved" / "good.patch").unlink(missing_ok=True)


# ── _reject_patch tests ─────────────────────────────────────────────────

class TestRejectPatch:
    """Test _reject_patch() method."""

    def test_reject_nonexistent_patch(self, repl, patches_base):
        result = repl._reject_patch("nonexistent.patch", "bad code")
        assert "[ERROR: Patch not found" in result

    def test_reject_patch_success(self, repl, patches_base):
        # Create a pending patch
        patch_file = patches_base / "pending" / "reject_me.patch"
        patch_file.write_text("# Patch: reject me\ndiff content\n")

        try:
            result = repl._reject_patch("reject_me.patch", "Does not meet standards")
            assert "rejected" in result.lower()
            assert (patches_base / "rejected" / "reject_me.patch").exists()
            assert not (patches_base / "pending" / "reject_me.patch").exists()

            # Verify rejection reason appended
            content = (patches_base / "rejected" / "reject_me.patch").read_text()
            assert "Does not meet standards" in content
        finally:
            (patches_base / "rejected" / "reject_me.patch").unlink(missing_ok=True)

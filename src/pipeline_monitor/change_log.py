"""Change audit log for Claude-in-the-loop debugging.

Records every debugging batch: what anomalies were seen, what Claude
reasoned, what files were modified. Enables rewind (git revert) and
meta-steering (user reads log, adds constraints to next session).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LOG_PATH = Path("/mnt/raid0/llm/claude/logs/debug_changes.jsonl")


def _extract_modified_files(git_diff: str) -> list[str]:
    """Extract file paths from git diff output."""
    files: list[str] = []
    for line in git_diff.split("\n"):
        if line.startswith("diff --git"):
            # "diff --git a/path b/path" → "path"
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                files.append(parts[1])
    return files


def capture_git_diff(project_root: Path) -> str:
    """Run git diff and return the output."""
    try:
        result = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=30,
        )
        return result.stdout
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"[change_log] git diff failed: {e}")
        return ""


def capture_git_diff_stat(project_root: Path) -> dict[str, str]:
    """Snapshot file modification state: path → short sha of content.

    Used before/after Claude invocation to detect which files Claude
    actually changed (vs pre-existing dirty working tree files).
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=15,
        )
        stat: dict[str, str] = {}
        for fname in result.stdout.strip().split("\n"):
            fname = fname.strip()
            if not fname:
                continue
            fpath = project_root / fname
            if fpath.is_file():
                # Hash the first 8KB for fast fingerprinting
                try:
                    content = fpath.read_bytes()[:8192]
                    import hashlib
                    stat[fname] = hashlib.sha1(content).hexdigest()[:12]
                except OSError:
                    stat[fname] = "unreadable"
            else:
                stat[fname] = "missing"
        return stat
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"[change_log] git diff --name-only failed: {e}")
        return {}


def diff_snapshots(
    before: dict[str, str], after: dict[str, str],
) -> list[str]:
    """Return file paths that changed between two snapshots."""
    changed: list[str] = []
    all_files = set(before) | set(after)
    for f in sorted(all_files):
        b = before.get(f)
        a = after.get(f)
        if b != a:
            changed.append(f)
    return changed


def commit_changes(
    project_root: Path,
    batch_id: int,
    summary: str,
) -> str | None:
    """Auto-commit changes with a descriptive message. Returns commit SHA or None."""
    msg = f"debug: batch {batch_id} — {summary}"
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(project_root),
            timeout=30,
            check=True,
        )
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=30,
        )
        if result.returncode != 0:
            return None
        # Extract SHA from "... [branch SHA] message"
        sha_match = re.search(r"\[[\w/-]+ ([a-f0-9]+)\]", result.stdout)
        return sha_match.group(1) if sha_match else None
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as e:
        logger.warning(f"[change_log] auto-commit failed: {e}")
        return None


class ChangeLog:
    """Audit trail for all debugging changes. Enables rewind and meta-steering."""

    def __init__(self, log_path: Path | None = None):
        self.log_path = log_path or LOG_PATH

    def record(
        self,
        session_id: str | None,
        batch_id: int,
        batch: list[dict],
        claude_response: dict[str, Any],
        git_diff: str,
        commit_sha: str | None = None,
    ) -> None:
        """Record one debugging batch with full context."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "batch_id": batch_id,
            "questions_analyzed": [d.get("question_id", "") for d in batch],
            "anomalies_seen": {
                d.get("question_id", ""): d.get("anomaly_signals", {})
                for d in batch
                if any(d.get("anomaly_signals", {}).values())
            },
            "claude_reasoning": claude_response.get("result", ""),
            "files_modified": _extract_modified_files(git_diff),
            "git_diff": git_diff,
            "git_commit_sha": commit_sha,
        }
        self._append(entry)

    def _append(self, entry: dict[str, Any]) -> None:
        """Append entry to the JSONL file with fcntl locking."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, default=str) + "\n"
        with open(self.log_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

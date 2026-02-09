"""Claude-in-the-loop debugger for seeding pipeline.

Manages a persistent Claude Code session that receives batches of
diagnostic records, analyzes anomalies, and applies fixes (prompt
hot-swaps, code patches). Session persists across batches, accumulating
context about the system.

Key design decisions (2026-02-09 rewrite):
- invoke_claude() runs as a background subprocess (Popen) so seeding
  continues while Claude analyzes.  Results are collected on the next
  add_diagnostic() call or on flush().
- Critical anomalies (score >= 1.0) no longer bypass batching.  They
  set a flag so the batch is sent as soon as all configs for the
  current question are submitted (end-of-question flush).
- Git diff comparison uses before/after file-content snapshots so
  pre-existing dirty files don't trigger spurious API restarts.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any

from src.pipeline_monitor.change_log import (
    ChangeLog,
    capture_git_diff,
    capture_git_diff_stat,
    commit_changes,
    diff_snapshots,
)

logger = logging.getLogger(__name__)

STACK_SCRIPT = Path("/mnt/raid0/llm/claude/scripts/server/orchestrator_stack.py")

# System prompt sent on the first invocation of each session.
DEBUGGER_SYSTEM_PROMPT = textwrap.dedent("""\
You are debugging the orchestration pipeline for a local LLM inference system.
You will receive batches of diagnostic records from seeding evaluation runs.

For each batch:
1. Scan anomaly_signals for flagged issues
2. Read the full answer text to diagnose the root cause
3. If needed, read the inference tap section for deep context:
   python scripts/benchmark/read_tap_section.py --offset {tap_offset_bytes} --length {tap_length_bytes}
4. Apply fixes:
   - Prompt issues: Edit orchestration/prompts/{relevant_prompt}.md
   - Code issues: Edit src/ files (note: requires API restart)
5. Report what you fixed and why

Key files:
- orchestration/prompts/*.md — hot-swappable, edits take effect on next request
- orchestration/prompts/roles/*.md — per-role system prompts
- src/graph/nodes.py — REPL loop, escalation, defenses
- src/api/routes/chat_delegation.py — architect delegation parsing
- src/prompt_builders/resolver.py — prompt resolution (uncached file reads)

Known bug classes: repetition loops, comment-only code, template echo (D| AND I| both output),\
 self-doubt loops, format violations, think-tag leaks, delegation format errors.

IMPORTANT: Only edit files when you're confident in the fix. For uncertain cases, describe the\
 issue and proposed fix but don't apply it.
""")


class ClaudeDebugger:
    """Manages a persistent Claude Code session for pipeline debugging.

    Batches diagnostic records and invokes Claude when the batch is full.
    Critical anomalies flag the batch as urgent but don't bypass batching —
    use end_question() after all configs for a question are submitted to
    trigger an early flush when urgent.
    """

    def __init__(
        self,
        project_root: Path,
        batch_size: int = 5,
        anomaly_threshold: float = 0.3,
        auto_commit: bool = False,
        dry_run: bool = False,
    ):
        self.project_root = project_root
        self.session_id: str | None = None
        self.batch: list[dict] = []
        self.batch_size = batch_size
        self.anomaly_threshold = anomaly_threshold
        self.auto_commit = auto_commit
        self.dry_run = dry_run
        self.batch_count = 0
        self.fixes_applied: list[dict] = []
        self.change_log = ChangeLog()
        self._urgent = False  # set when a critical anomaly is added

        # Background invocation state
        self._bg_process: subprocess.Popen | None = None
        self._bg_snapshot: dict[str, str] = {}
        self._bg_batch: list[dict] = []
        self._bg_lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────

    def add_diagnostic(self, diag: dict) -> None:
        """Add a diagnostic record.

        Does NOT block.  If the batch is full, dispatches Claude in the
        background.  If score >= 1.0, marks the batch as urgent (flushed
        at end_question()).
        """
        # Collect any finished background results first
        self._collect_background()

        self.batch.append(diag)
        score = diag.get("anomaly_score", 0.0)

        if score >= 1.0:
            self._urgent = True

        if len(self.batch) >= self.batch_size:
            self._dispatch()

    def end_question(self) -> None:
        """Called after all configs for one question are submitted.

        If the batch contains urgent diagnostics, dispatch now.
        """
        self._collect_background()
        if self._urgent and self.batch:
            self._dispatch()

    def flush(self) -> None:
        """Process remaining batch + wait for background. Called at end of run."""
        if self.batch:
            self._dispatch()
        self._wait_background()

    # ── Background dispatch ─────────────────────────────────────────

    def _dispatch(self) -> None:
        """Send the current batch to Claude in the background."""
        # Wait for any prior invocation to finish first
        self._wait_background()

        self.batch_count += 1
        self._urgent = False
        prompt = self._build_prompt(self.batch)
        batch_copy = list(self.batch)
        self.batch.clear()

        if self.dry_run:
            logger.info(
                f"[DEBUG DRY-RUN] Would invoke Claude with batch {self.batch_count} "
                f"({len(batch_copy)} diagnostics)"
            )
            self._log_dry_run(prompt)
            return

        # Snapshot file state BEFORE invoking Claude
        snapshot_before = capture_git_diff_stat(self.project_root)

        cmd = [
            "claude",
            "-p", prompt,
            "--output-format", "json",
            "--allowedTools", "Read,Edit,Bash,Grep,Glob",
        ]
        if self.session_id:
            cmd.extend(["--resume", self.session_id])

        logger.info(
            f"[DEBUG] Dispatching Claude (batch {self.batch_count}, "
            f"{len(batch_copy)} diagnostics) in background..."
        )

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.project_root),
            )
        except OSError as e:
            logger.error(f"[DEBUG] Failed to launch claude: {e}")
            return

        with self._bg_lock:
            self._bg_process = proc
            self._bg_snapshot = snapshot_before
            self._bg_batch = batch_copy

    def _collect_background(self) -> None:
        """Non-blocking check: if background Claude is done, process results."""
        with self._bg_lock:
            proc = self._bg_process
            if proc is None:
                return
            ret = proc.poll()
            if ret is None:
                return  # still running
            # Process is done
            self._bg_process = None
            snapshot_before = self._bg_snapshot
            batch_copy = self._bg_batch
            self._bg_snapshot = {}
            self._bg_batch = []

        self._process_result(proc, snapshot_before, batch_copy)

    def _wait_background(self) -> None:
        """Block until background Claude is done, then process results."""
        with self._bg_lock:
            proc = self._bg_process
            if proc is None:
                return
            self._bg_process = None
            snapshot_before = self._bg_snapshot
            batch_copy = self._bg_batch
            self._bg_snapshot = {}
            self._bg_batch = []

        # Wait with timeout
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            logger.warning("[DEBUG] Claude invocation timed out (300s), killing")
            proc.kill()
            proc.wait(timeout=10)
            return

        self._process_result(proc, snapshot_before, batch_copy)

    def _process_result(
        self,
        proc: subprocess.Popen,
        snapshot_before: dict[str, str],
        batch_copy: list[dict],
    ) -> None:
        """Process completed Claude invocation results."""
        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""

        if proc.returncode != 0:
            logger.warning(
                f"[DEBUG] Claude exited with code {proc.returncode}: "
                f"{stderr[:500]}"
            )

        # Parse response
        response: dict[str, Any] = {}
        if stdout.strip():
            try:
                response = json.loads(stdout)
            except json.JSONDecodeError:
                logger.warning("[DEBUG] Claude returned non-JSON output")
                response = {"result": stdout[:2000]}

        # Capture session_id from first invocation
        if not self.session_id and "session_id" in response:
            self.session_id = response["session_id"]
            logger.info(f"[DEBUG] Session ID: {self.session_id}")

        # Detect changes by comparing before/after snapshots
        snapshot_after = capture_git_diff_stat(self.project_root)
        changed_files = diff_snapshots(snapshot_before, snapshot_after)

        if changed_files:
            logger.info(f"[DEBUG] Claude changed {len(changed_files)} files: {changed_files}")

            commit_sha = None
            if self.auto_commit:
                summary = self._summarize_batch_from(batch_copy)
                commit_sha = commit_changes(
                    self.project_root, self.batch_count, summary,
                )
                if commit_sha:
                    logger.info(f"[DEBUG] Auto-committed: {commit_sha}")

            # Log with full git diff for the change log (for rewind)
            git_diff = capture_git_diff(self.project_root)
            self.change_log.record(
                session_id=self.session_id,
                batch_id=self.batch_count,
                batch=batch_copy,
                claude_response=response,
                git_diff=git_diff,
                commit_sha=commit_sha,
            )

            # Hot-restart API only if Claude's changes include .py files
            py_changes = [f for f in changed_files if f.endswith(".py")]
            if py_changes:
                self._hot_restart_api(py_changes)
        else:
            # Log even when no changes (for audit trail)
            self.change_log.record(
                session_id=self.session_id,
                batch_id=self.batch_count,
                batch=batch_copy,
                claude_response=response,
                git_diff="",
                commit_sha=None,
            )

    # ── Helpers ──────────────────────────────────────────────────────

    def _hot_restart_api(self, py_files: list[str]) -> bool:
        """Hot-restart the API if Python files were changed by Claude."""
        logger.warning(f"[DEBUG] Code changes detected in: {py_files}")
        logger.warning("[DEBUG] Hot-restarting orchestrator API...")

        try:
            subprocess.run(
                [sys.executable, str(STACK_SCRIPT), "reload", "orchestrator"],
                cwd=str(self.project_root),
                timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.error(f"[DEBUG] API reload failed: {e}")
            return False

        return True

    def _build_prompt(self, batch: list[dict]) -> str:
        """Build the prompt for Claude with diagnostic batch."""
        parts: list[str] = []

        # Include system prompt on first invocation only
        if self.batch_count == 1:
            parts.append(DEBUGGER_SYSTEM_PROMPT)
            parts.append("---")

        parts.append(f"## Diagnostic Batch #{self.batch_count}")
        parts.append(f"**{len(batch)} answers analyzed**\n")

        for i, diag in enumerate(batch, 1):
            triggered = [
                name for name, active in diag.get("anomaly_signals", {}).items()
                if active
            ]
            anomaly_str = ", ".join(triggered) if triggered else "none"
            passed_str = "PASS" if diag.get("passed") else "FAIL"

            parts.append(f"### Answer {i}: {diag.get('question_id', '?')} [{passed_str}]")
            parts.append(f"- **Config**: {diag.get('config', '?')}")
            parts.append(f"- **Role**: {diag.get('role', '?')} ({diag.get('mode', '?')})")
            parts.append(f"- **Expected**: {diag.get('expected', '?')}")
            parts.append(f"- **Scoring**: {diag.get('scoring_method', '?')}")
            parts.append(f"- **Tokens**: {diag.get('tokens_generated', 0)} in {diag.get('elapsed_s', 0):.1f}s")
            parts.append(f"- **Error**: {diag.get('error', 'none')} ({diag.get('error_type', 'none')})")
            parts.append(f"- **Anomalies**: {anomaly_str} (score={diag.get('anomaly_score', 0):.2f})")
            parts.append(f"- **Role history**: {' → '.join(diag.get('role_history', []))}")
            parts.append(f"- **Tools**: {diag.get('tools_used', 0)} ({', '.join(diag.get('tools_called', []))})")

            if diag.get("tap_offset_bytes") and diag.get("tap_length_bytes"):
                parts.append(
                    f"- **Tap**: offset={diag['tap_offset_bytes']}, "
                    f"length={diag['tap_length_bytes']}"
                )

            # Include answer text (truncated for very long answers)
            answer = diag.get("answer", "")
            if len(answer) > 2000:
                answer = answer[:2000] + f"\n... [{len(answer) - 2000} chars truncated]"
            parts.append(f"\n**Answer text:**\n```\n{answer}\n```\n")

        parts.append("---")
        parts.append(
            "Analyze the anomalies above. If you can identify a root cause "
            "and a confident fix, apply it. Otherwise describe the issue and "
            "proposed fix without editing files."
        )

        return "\n".join(parts)

    def _summarize_batch_from(self, batch: list[dict]) -> str:
        """One-line summary from a batch copy (for commit messages)."""
        questions = [d.get("question_id", "?") for d in batch]
        anomalies = set()
        for d in batch:
            for name, active in d.get("anomaly_signals", {}).items():
                if active:
                    anomalies.add(name)
        q_str = ", ".join(questions[:3])
        if len(questions) > 3:
            q_str += f" +{len(questions) - 3} more"
        a_str = ", ".join(sorted(anomalies)) if anomalies else "no anomalies"
        return f"{q_str} | {a_str}"

    def _log_dry_run(self, prompt: str) -> None:
        """Log the prompt that would be sent in dry-run mode."""
        preview = prompt[:500]
        if len(prompt) > 500:
            preview += f"\n... [{len(prompt) - 500} chars truncated]"
        logger.info(f"[DEBUG DRY-RUN] Prompt preview:\n{preview}")

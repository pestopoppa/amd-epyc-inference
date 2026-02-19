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
import socket
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any

from src.prompt_builders.resolver import resolve_prompt
from src.pipeline_monitor.change_log import (
    ChangeLog,
    capture_git_diff,
    capture_git_diff_stat,
    commit_changes,
    diff_snapshots,
)

logger = logging.getLogger(__name__)

STACK_SCRIPT = Path("/mnt/raid0/llm/claude/scripts/server/orchestrator_stack.py")
_RETRY_QUEUE_PATH = Path("/mnt/raid0/llm/claude/logs/retry_queue.jsonl")
_PROPOSED_SIGNALS_PATH = Path("/mnt/raid0/llm/claude/logs/proposed_signals.jsonl")

# Services the debugger is allowed to reload via orchestrator_stack.py.
# Keys = service names recognized by the stack script.
# Values = health check endpoints (port, path).
RELOADABLE_SERVICES: dict[str, tuple[int, str]] = {
    "orchestrator": (8000, "/health"),
    "nextplaid-code": (8088, "/health"),
    "nextplaid-docs": (8089, "/health"),
}

_TAP_PATH = "/mnt/raid0/llm/tmp/inference_tap.log"
_REPL_TAP_PATH = "/mnt/raid0/llm/tmp/repl_tap.log"
_MAX_TAP_INLINE = 12_000  # chars — ~3000 tokens, fits in Claude's context
_MAX_REPL_TAP_INLINE = 4_000  # chars — REPL output is more compact


def _read_tap_inline(offset: int, length: int, path: str | None = None, max_chars: int = _MAX_TAP_INLINE) -> str:
    """Read tap section and truncate if too large."""
    if path is None:
        path = _TAP_PATH
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(min(length, max_chars * 4))  # UTF-8 worst case
        text = data.decode("utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [{length - max_chars} bytes truncated]"
        return text
    except (FileNotFoundError, OSError):
        return ""


import re as _re

_NEW_SIGNAL_RE = _re.compile(
    r"NEW_SIGNAL:\s*name=(?P<name>\w+)\s+"
    r"weight=(?P<weight>[\d.]+)\s+"
    r"description=(?P<desc>.+?)\n"
    r"detector=(?P<detector>.+?)\n"
    r"evidence=(?P<evidence>.+)",
    _re.MULTILINE,
)

_RELOAD_SERVICE_RE = _re.compile(
    r"RELOAD_SERVICE:\s*(?P<service>\S+)(?:\s+reason=(?P<reason>.+))?",
)


def _extract_proposed_signals(text: str) -> list[dict]:
    """Parse NEW_SIGNAL: proposals from Claude's response text."""
    proposals = []
    for m in _NEW_SIGNAL_RE.finditer(text):
        proposals.append({
            "name": m.group("name"),
            "weight": float(m.group("weight")),
            "description": m.group("desc").strip(),
            "detector": m.group("detector").strip(),
            "evidence": [e.strip() for e in m.group("evidence").split(",")],
        })
    return proposals


def _extract_reload_requests(text: str) -> list[dict[str, str]]:
    """Parse RELOAD_SERVICE: directives from Claude's response text."""
    reloads = []
    for m in _RELOAD_SERVICE_RE.finditer(text):
        service = m.group("service")
        if service in RELOADABLE_SERVICES:
            reloads.append({
                "service": service,
                "reason": (m.group("reason") or "").strip(),
            })
        else:
            logger.warning(
                f"[DEBUG] Claude requested reload of unknown service '{service}' "
                f"(allowed: {list(RELOADABLE_SERVICES)})"
            )
    return reloads


def _check_service_health(port: int, path: str = "/health", timeout: float = 3.0) -> bool:
    """Quick TCP+HTTP health check for a service."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            if s.connect_ex(("localhost", port)) != 0:
                return False
    except OSError:
        return False
    # Port is open — try HTTP health endpoint
    try:
        import urllib.request
        req = urllib.request.Request(f"http://localhost:{port}{path}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_infra_health() -> dict[str, bool]:
    """Check health of all reloadable services. Returns {name: healthy}."""
    status = {}
    for name, (port, path) in RELOADABLE_SERVICES.items():
        status[name] = _check_service_health(port, path)
    return status


def _persist_proposed_signals(
    proposals: list[dict], batch_id: int, session_id: str | None,
    path: Path | None = None,
) -> None:
    """Append proposed signals to the discovery log."""
    if not proposals:
        return
    from datetime import datetime, timezone
    target = path or _PROPOSED_SIGNALS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a") as f:
        for p in proposals:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "batch_id": batch_id,
                "session_id": session_id,
                **p,
            }
            f.write(json.dumps(entry) + "\n")


# System prompt sent on the first invocation of each session.
# Fallback used when orchestration/prompts/debugger_system.md is missing.
# The .md file is the primary source (hot-swappable without restart).
_DEBUGGER_SYSTEM_FALLBACK = textwrap.dedent("""\
You are debugging the orchestration pipeline for a local LLM inference system.
You will receive batches of diagnostic records from seeding evaluation runs.
Scan anomaly_signals, read the answer and inference logs, diagnose root causes, apply fixes.
Edit orchestration/prompts/*.md for prompt issues, src/ for code issues.
rules.md uses few-shot examples — only edit to add/improve examples, never add rules.
For infra issues, use RELOAD_SERVICE: orchestrator|nextplaid-code|nextplaid-docs reason=...
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
        retry_path: Path | None = None,
        replay_context: bool = False,
        retrieval_overrides: dict[str, Any] | None = None,
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

        # Retry queue: failed questions pending post-fix verification.
        # Persisted to disk so script restarts don't lose pending retries.
        self._retry_queue: list[tuple[str, str]] = []   # (suite, qid)
        self._retry_suites: set[str] = set()             # suites affected by fix
        self._retried: set[tuple[str, str]] = set()      # prevent infinite retry loops
        self._retry_path = retry_path or (project_root / "logs" / "retry_queue.jsonl")
        self._load_persisted_retries()

        # Replay evaluation context (loaded lazily on first prompt build)
        self._replay_context_enabled = replay_context
        self._replay_summary: str | None = None
        self._retrieval_overrides = retrieval_overrides or {}

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

    def pop_retries(self) -> tuple[list[tuple[str, str]], set[str]]:
        """Return (failed_qids, affected_suites) and clear queue.

        Called by the eval loop after end_question() to check whether
        a post-fix mini regression suite should run.

        Uses _wait_background() (blocking) instead of _collect_background()
        to ensure Claude's analysis is complete before checking for retries.
        Without blocking, the background process is almost always still
        running when this is called, resulting in empty retry queues.
        """
        self._wait_background()
        retries = list(self._retry_queue)
        suites = set(self._retry_suites)
        self._retry_queue.clear()
        self._retry_suites.clear()
        self._clear_persisted_retries()
        return retries, suites

    # ── Retry persistence ────────────────────────────────────────────

    def _persist_retries(self) -> None:
        """Append pending retries to disk so script restarts don't lose them."""
        if not self._retry_queue:
            return
        self._retry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._retry_path, "a") as f:
            for suite, qid in self._retry_queue:
                entry = {"suite": suite, "qid": qid}
                f.write(json.dumps(entry) + "\n")

    def _load_persisted_retries(self) -> None:
        """Load any retries persisted by a previous session."""
        if not self._retry_path.is_file():
            return
        try:
            with open(self._retry_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    key = (entry["suite"], entry["qid"])
                    if key not in self._retried:
                        self._retry_queue.append(key)
                        self._retry_suites.add(entry["suite"])
            if self._retry_queue:
                logger.info(
                    f"[DEBUG] Loaded {len(self._retry_queue)} persisted "
                    f"retries from previous session"
                )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"[DEBUG] Failed to load persisted retries: {e}")

    def _clear_persisted_retries(self) -> None:
        """Remove the persisted retry file after retries are consumed."""
        try:
            self._retry_path.unlink(missing_ok=True)
        except OSError:
            pass

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

        # Extract any proposed new anomaly signals from Claude's response
        result_text = response.get("result", "")
        proposals = _extract_proposed_signals(result_text)
        if proposals:
            logger.info(
                f"[DEBUG] Claude proposed {len(proposals)} new anomaly signal(s): "
                f"{[p['name'] for p in proposals]}"
            )
            _persist_proposed_signals(
                proposals, self.batch_count, self.session_id,
                path=self.project_root / "logs" / "proposed_signals.jsonl",
            )

        # Execute any RELOAD_SERVICE: directives from Claude
        reload_requests = _extract_reload_requests(result_text)
        for req in reload_requests:
            self._reload_service(req["service"], reason=req.get("reason", ""))

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

            # Queue failed questions for post-fix verification
            failed_qids: set[tuple[str, str]] = set()
            affected_suites: set[str] = set()
            for diag in batch_copy:
                suite = diag.get("suite", "")
                affected_suites.add(suite)
                if not diag.get("passed", True):
                    raw_qid = diag.get("question_id", "")
                    qid = raw_qid.split("/", 1)[1] if "/" in raw_qid else raw_qid
                    key = (suite, qid)
                    if key not in self._retried:
                        failed_qids.add(key)

            for key in failed_qids:
                self._retried.add(key)
                self._retry_queue.append(key)
            self._retry_suites.update(affected_suites)

            if failed_qids:
                self._persist_retries()
                logger.info(
                    f"[DEBUG] Queued {len(failed_qids)} failed questions for retry "
                    f"(suites: {affected_suites})"
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

    def _reload_service(self, service_name: str, reason: str = "") -> bool:
        """Reload a service via orchestrator_stack.py.

        Allowed services are defined in RELOADABLE_SERVICES.
        Returns True if the service came back healthy after reload.
        """
        if service_name not in RELOADABLE_SERVICES:
            logger.error(
                f"[DEBUG] Cannot reload '{service_name}' — "
                f"not in allowed list: {list(RELOADABLE_SERVICES)}"
            )
            return False

        reason_str = f" (reason: {reason})" if reason else ""
        logger.warning(f"[DEBUG] Reloading service '{service_name}'{reason_str}")

        try:
            result = subprocess.run(
                [sys.executable, str(STACK_SCRIPT), "reload", service_name],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.error(
                    f"[DEBUG] Reload of '{service_name}' failed (exit {result.returncode}): "
                    f"{result.stderr[:300]}"
                )
                return False
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.error(f"[DEBUG] Reload of '{service_name}' failed: {e}")
            return False

        # Verify health after reload
        port, path = RELOADABLE_SERVICES[service_name]
        if _check_service_health(port, path, timeout=5.0):
            logger.info(f"[DEBUG] Service '{service_name}' healthy after reload")
            return True
        else:
            logger.warning(f"[DEBUG] Service '{service_name}' NOT healthy after reload")
            return False

    def _hot_restart_api(self, py_files: list[str]) -> bool:
        """Hot-restart the API if Python files were changed by Claude."""
        logger.warning(f"[DEBUG] Code changes detected in: {py_files}")
        return self._reload_service("orchestrator", reason=f"code changes: {py_files}")

    def _get_replay_summary(self) -> str:
        """Build a one-time summary of historical memory performance via replay harness.

        Loads lazily on first call. Returns empty string if replay unavailable.
        """
        if self._replay_summary is not None:
            return self._replay_summary
        self._replay_summary = ""
        if not self._replay_context_enabled:
            return ""
        try:
            from orchestration.repl_memory.replay.trajectory import TrajectoryExtractor
            from orchestration.repl_memory.replay.engine import ReplayEngine
            from orchestration.repl_memory.retriever import RetrievalConfig
            from orchestration.repl_memory.q_scorer import ScoringConfig
            from orchestration.repl_memory.progress_logger import ProgressReader

            retrieval_config = RetrievalConfig(**{
                k: v for k, v in self._retrieval_overrides.items()
                if k in RetrievalConfig.__dataclass_fields__ and v is not None
            })

            reader = ProgressReader()
            extractor = TrajectoryExtractor(reader)
            trajectories = extractor.extract_complete(days=14, max_trajectories=500)
            if not trajectories:
                self._replay_summary = ""
                return ""

            # Try skill-aware replay if SkillBank available
            skill_metrics = None
            try:
                from orchestration.repl_memory.replay.skill_replay import (
                    SkillAwareReplayEngine, SkillBankConfig,
                )
                from orchestration.repl_memory.skill_bank import SkillBank

                skill_db = Path("orchestration/repl_memory/sessions/skills.db")
                if skill_db.exists():
                    sb = SkillBank(db_path=skill_db)
                    engine = SkillAwareReplayEngine(skill_bank=sb)
                    skill_metrics = engine.run_with_skill_metrics(
                        retrieval_config, ScoringConfig(), SkillBankConfig(),
                        trajectories, "debugger_skill_baseline",
                    )
                    metrics = skill_metrics.base_metrics
            except ImportError:
                pass

            if skill_metrics is None:
                engine = ReplayEngine()
                metrics = engine.run_with_metrics(
                    retrieval_config, ScoringConfig(), trajectories, "debugger_baseline",
                )

            by_type = ", ".join(
                f"{t}: {a:.0%}" for t, a in metrics.routing_accuracy_by_type.items()
            )
            tier = ", ".join(
                f"{t}: {n}" for t, n in sorted(
                    metrics.tier_usage.items(), key=lambda x: -x[1],
                )[:5]
            )
            skill_lines = ""
            if skill_metrics is not None:
                skill_lines = (
                    f"- **Skills**: {skill_metrics.total_skills_retrieved} retrieved, "
                    f"coverage={skill_metrics.skill_coverage:.1%}, "
                    f"avg/step={skill_metrics.avg_skills_per_step:.1f}\n"
                )
                # Skill health via EvolutionMonitor
                try:
                    from orchestration.repl_memory.skill_evolution import EvolutionMonitor
                    monitor = EvolutionMonitor(sb)
                    evo_summary = monitor.get_evolution_summary()
                    skill_lines += (
                        f"- **Skill health**: {evo_summary.get('total', 0)} total, "
                        f"{evo_summary.get('active', 0)} active, "
                        f"{evo_summary.get('deprecated', 0)} deprecated\n"
                    )
                except Exception:
                    pass

            self._replay_summary = (
                f"\n## MemRL Replay Context (last 14 days)\n"
                f"- **Trajectories**: {metrics.num_trajectories} "
                f"({metrics.num_complete} complete)\n"
                f"- **Routing accuracy**: {metrics.routing_accuracy:.1%} "
                f"(by type: {by_type or 'N/A'})\n"
                f"- **Avg reward**: {metrics.avg_reward:.3f}, "
                f"cumulative: {metrics.cumulative_reward:.1f}\n"
                f"- **Q convergence**: step {metrics.q_convergence_step}\n"
                f"- **Tier usage**: {tier or 'N/A'}\n"
                f"- **Escalation**: precision={metrics.escalation_precision:.0%}, "
                f"recall={metrics.escalation_recall:.0%}\n"
                f"- **Calibration**: ECE={metrics.ece_global:.3f}, "
                f"Brier={metrics.brier_global:.3f}, "
                f"coverage={metrics.conformal_coverage:.1%}, "
                f"risk={metrics.conformal_risk:.1%}\n"
                f"- **Retrieval overrides**: {self._retrieval_overrides or 'default'}\n"
                f"{skill_lines}"
            )
            logger.info(
                f"[DEBUG] Replay context loaded: {metrics.num_trajectories} trajectories, "
                f"routing accuracy {metrics.routing_accuracy:.1%}"
            )
        except Exception as e:
            logger.warning(f"[DEBUG] Replay context unavailable: {e}")
            self._replay_summary = ""
        return self._replay_summary

    def _build_prompt(self, batch: list[dict]) -> str:
        """Build the prompt for Claude with diagnostic batch."""
        parts: list[str] = []

        # Include system prompt on first invocation only
        if self.batch_count == 1:
            parts.append(resolve_prompt("debugger_system", _DEBUGGER_SYSTEM_FALLBACK))
            # Append replay harness summary if available
            replay_ctx = self._get_replay_summary()
            if replay_ctx:
                parts.append(replay_ctx)
            parts.append("---")

        parts.append(f"## Diagnostic Batch #{self.batch_count}")
        parts.append(f"**{len(batch)} answers analyzed**\n")

        # Include infrastructure health status
        infra = check_infra_health()
        degraded = [name for name, ok in infra.items() if not ok]
        if degraded:
            parts.append(f"**INFRA DEGRADED**: {', '.join(degraded)} not healthy")
            parts.append(
                "Use `RELOAD_SERVICE: <name>` to restart. "
                "See Reloadable Services in system prompt.\n"
            )
        else:
            parts.append("**Infra**: all services healthy\n")

        # Suite-level failure detection
        suite_stats: dict[str, dict[str, int]] = {}
        for diag in batch:
            s = diag.get("suite", "unknown")
            suite_stats.setdefault(s, {"total": 0, "failed": 0})
            suite_stats[s]["total"] += 1
            if not diag.get("passed", True):
                suite_stats[s]["failed"] += 1
        for s, st in suite_stats.items():
            if st["failed"] == st["total"] and st["total"] >= 2:
                parts.append(
                    f"**SUITE-LEVEL FAILURE**: {s} — {st['failed']}/{st['total']} failed. "
                    "Investigate tools/scorer, not individual answers."
                )
        parts.append("")

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
            deleg_diag = diag.get("delegation_diagnostics", {}) or {}
            if deleg_diag:
                br = deleg_diag.get("break_reason", "n/a")
                loops = deleg_diag.get("effective_max_loops", "n/a")
                rep = deleg_diag.get("repeated_targets", {})
                infer_hops = deleg_diag.get("delegation_inference_hops", "n/a")
                avg_prompt_ms = deleg_diag.get("avg_prompt_ms", "n/a")
                avg_gen_ms = deleg_diag.get("avg_gen_ms", "n/a")
                report_handles = deleg_diag.get("report_handles_count", "n/a")
                handle_ids = []
                for h in deleg_diag.get("report_handles", []) or []:
                    if isinstance(h, dict) and h.get("id"):
                        handle_ids.append(str(h["id"]))
                parts.append(
                    f"- **Delegation diagnostics**: break_reason={br}, "
                    f"max_loops={loops}, repeated_targets={rep}, "
                    f"infer_hops={infer_hops}, avg_prompt_ms={avg_prompt_ms}, "
                    f"avg_gen_ms={avg_gen_ms}, report_handles={report_handles}, "
                    f"report_ids={','.join(handle_ids[:3]) if handle_ids else 'none'}"
                )
                phases = deleg_diag.get("phases", []) or []
                if phases:
                    parts.append("  - delegation_timeline:")
                    for phase in phases[:12]:
                        if not isinstance(phase, dict):
                            continue
                        p_loop = phase.get("loop", "?")
                        p_name = phase.get("phase", "?")
                        p_decision = phase.get("decision", "")
                        p_target = phase.get("delegate_to", "")
                        p_mode = phase.get("delegate_mode", "")
                        p_ms = phase.get("ms", "n/a")
                        p_turns = phase.get("computation_turns", "")
                        p_parts = [f"loop={p_loop}", f"phase={p_name}", f"elapsed_ms={p_ms}"]
                        if p_decision:
                            p_parts.append(f"decision={p_decision}")
                        if p_target:
                            p_parts.append(f"target={p_target}")
                        if p_mode:
                            p_parts.append(f"delegate_mode={p_mode}")
                        if p_turns != "":
                            p_parts.append(f"turns={p_turns}")
                        parts.append(f"    - {' '.join(p_parts)}")
            parts.append(f"- **Tools**: {diag.get('tools_used', 0)} ({', '.join(diag.get('tools_called', []))})")
            tool_chains = diag.get("tool_chains", []) or []
            if tool_chains:
                parts.append(f"- **Tool chains**: {len(tool_chains)}")
                for ch in tool_chains[:5]:
                    if not isinstance(ch, dict):
                        continue
                    cid = ch.get("chain_id", "?")
                    tools = ch.get("tools", []) or []
                    tool_list = ",".join(str(t) for t in tools[:4]) if tools else "none"
                    if len(tools) > 4:
                        tool_list += f"+{len(tools)-4}"
                    mode_req = ch.get("mode_requested", "n/a")
                    mode_used = ch.get("mode_used", "n/a")
                    waves = ch.get("waves", "n/a")
                    fallback = ch.get("fallback_to_seq", "n/a")
                    par_mut = ch.get("parallel_mutations_enabled", "n/a")
                    success = ch.get("success", "n/a")
                    parts.append(
                        f"  - chain={cid} tools={tool_list} mode={mode_req}->{mode_used} "
                        f"waves={waves} fallback={fallback} parallel_mutations={par_mut} success={success}"
                    )
                    wave_timeline = ch.get("wave_timeline", []) or []
                    if wave_timeline:
                        parts.append("    - wave_timeline:")
                        for w in wave_timeline[:8]:
                            if not isinstance(w, dict):
                                continue
                            w_idx = w.get("wave_index", "?")
                            w_tools = w.get("tools", []) or []
                            w_tool_list = ",".join(str(t) for t in w_tools[:4]) if w_tools else "none"
                            if len(w_tools) > 4:
                                w_tool_list += f"+{len(w_tools)-4}"
                            w_mode = w.get("mode_used", mode_used)
                            w_ms = w.get("elapsed_ms", "n/a")
                            w_fallback = w.get("fallback_to_seq", fallback)
                            w_par = w.get("parallel_mutations_enabled", par_mut)
                            parts.append(
                                f"    - wave#{w_idx} tools={w_tool_list} mode={w_mode} "
                                f"elapsed_ms={w_ms} fallback={w_fallback} parallel_mutations={w_par}"
                            )

            # Orchestrator intelligence tunables (only show when relevant)
            cost_dims = diag.get("cost_dimensions", {})
            if cost_dims:
                dims_str = ", ".join(f"{k}={v:.3f}" for k, v in cost_dims.items())
                parts.append(f"- **Cost dimensions**: {dims_str}")
            if diag.get("think_harder_attempted"):
                th_result = "succeeded" if diag.get("think_harder_succeeded") else "failed→escalated"
                parts.append(f"- **Think-harder**: attempted, {th_result}")
            if diag.get("cheap_first_attempted"):
                cf_result = "passed quality gate" if diag.get("cheap_first_passed") else "failed→normal pipeline"
                parts.append(f"- **Cheap-first**: attempted, {cf_result}")
            if diag.get("grammar_enforced"):
                parts.append("- **Grammar**: GBNF-constrained generation active")
            if diag.get("parallel_tools_used"):
                parts.append("- **Parallel tools**: read-only tools executed in parallel")
            affinity = diag.get("cache_affinity_bonus", 0.0)
            if affinity > 0:
                parts.append(f"- **Cache affinity**: +{affinity:.0%} bonus applied")
            th_roi = diag.get("think_harder_expected_roi", 0.0)
            if th_roi > 0:
                parts.append(f"- **Think-harder ROI**: {th_roi:.3f} (EMA marginal utility)")
            # Context window management signals
            if diag.get("compaction_triggered"):
                saved = diag.get("compaction_tokens_saved", 0)
                parts.append(f"- **Compaction**: triggered, saved {saved} tokens")
            cleared = diag.get("tool_results_cleared", 0)
            if cleared > 0:
                parts.append(f"- **Tool output clearing**: {cleared} stale blocks removed")
            budget = diag.get("budget_diagnostics", {})
            if budget:
                deadline_ms = budget.get("deadline_remaining_ms", "n/a")
                clamped = budget.get("timeout_clamped", False)
                exhausted = budget.get("budget_exhausted", False)
                budget_parts = [f"deadline_remaining_ms={deadline_ms}"]
                if clamped:
                    budget_parts.append("timeout_clamped=true")
                if exhausted:
                    budget_parts.append("BUDGET_EXHAUSTED")
                parts.append(f"- **Budget**: {', '.join(budget_parts)}")

            skill_data = diag.get("skill_retrieval", {})
            if skill_data:
                parts.append(
                    f"- **Skills**: {skill_data.get('skills_retrieved', 0)} retrieved "
                    f"({', '.join(skill_data.get('skill_types', []))}), "
                    f"~{skill_data.get('skill_context_tokens', 0)} tokens injected"
                )

            tap_off = diag.get("tap_offset_bytes", 0)
            tap_len = diag.get("tap_length_bytes", 0)
            if tap_len > 0:
                tap_text = _read_tap_inline(tap_off, tap_len)
                if tap_text:
                    parts.append(f"\n**Inference log** ({tap_len} bytes):\n```\n{tap_text}\n```")

            repl_off = diag.get("repl_tap_offset_bytes", 0)
            repl_len = diag.get("repl_tap_length_bytes", 0)
            if repl_len > 0:
                repl_text = _read_tap_inline(repl_off, repl_len, _REPL_TAP_PATH, _MAX_REPL_TAP_INLINE)
                if repl_text:
                    parts.append(f"\n**REPL execution log** ({repl_len} bytes):\n```\n{repl_text}\n```")

            # Include answer text (truncated for very long answers)
            answer = diag.get("answer", "")
            if len(answer) > 2000:
                answer = answer[:2000] + f"\n... [{len(answer) - 2000} chars truncated]"
            parts.append(f"\n**Answer text:**\n```\n{answer}\n```\n")

        parts.append("---")
        parts.append(
            "Analyze the anomalies above. For systemic failures (same suite 100% fail), "
            "investigate tools and scoring before blaming models. Apply fixes — prompt "
            "edits are instant, code edits auto-restart the API. The retry queue will verify."
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

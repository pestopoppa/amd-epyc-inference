"""
ProgressLogger: Lightweight structured logging for orchestration events.

All tiers publish progress logs that the Q-scorer agent periodically processes.
This implements the "lab book" pattern from the async Q-scoring architecture.

Log format is JSONL (one JSON object per line) for efficient streaming reads.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default log path (on RAID array, or fallback to workspace for devcontainer)
_RAID_LOG_PATH = Path("/mnt/raid0/llm/claude/logs/progress")
_WORKSPACE_LOG_PATH = Path("/workspace/logs/progress")

# Use RAID path if available, otherwise fallback to workspace
DEFAULT_LOG_PATH = _RAID_LOG_PATH if _RAID_LOG_PATH.parent.exists() else _WORKSPACE_LOG_PATH


def _get_fallback_log_dir() -> Path:
    """Get a fallback log directory when default paths aren't writable.

    Returns a temp directory that will be cleaned up on process exit.
    Used in CI environments where neither RAID nor workspace paths exist.
    """
    import tempfile

    return Path(tempfile.mkdtemp(prefix="progress_logs_"))


class EventType(str, Enum):
    """Types of orchestration events."""

    # Task lifecycle
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    # Routing events
    ROUTING_DECISION = "routing_decision"
    ROUTING_FALLBACK = "routing_fallback"

    # Escalation events
    ESCALATION_TRIGGERED = "escalation_triggered"
    ESCALATION_RESOLVED = "escalation_resolved"
    ESCALATION_FAILED = "escalation_failed"

    # Gate events
    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"

    # REPL exploration events
    EXPLORATION_STARTED = "exploration_started"
    EXPLORATION_STRATEGY = "exploration_strategy"
    EXPLORATION_COMPLETED = "exploration_completed"

    # Formalizer events
    FORMALIZER_INVOKED = "formalizer_invoked"

    # Plan review events (architect-in-the-loop)
    PLAN_REVIEWED = "plan_reviewed"

    # Q-scoring events (from Q-scorer agent)
    Q_VALUE_UPDATED = "q_value_updated"
    MEMORY_STORED = "memory_stored"

    # Session lifecycle events (for session persistence)
    SESSION_CREATED = "session_created"
    SESSION_RESUMED = "session_resumed"
    SESSION_CHECKPOINTED = "session_checkpointed"
    SESSION_ARCHIVED = "session_archived"
    SESSION_FINDING_ADDED = "session_finding_added"


@dataclass
class ProgressEntry:
    """A single progress log entry."""

    event_type: EventType
    task_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_tier: Optional[str] = None  # "A", "B1", "B2", "B3", "C", "D"
    agent_role: Optional[str] = None  # "frontdoor", "coder", etc.

    # Event-specific data
    data: Dict[str, Any] = field(default_factory=dict)

    # Linking to memory
    memory_id: Optional[str] = None  # If this event is linked to a memory entry

    # Outcome (for completed events)
    outcome: Optional[str] = None  # "success", "failure", "partial"
    outcome_details: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        d = {
            "event_type": self.event_type.value,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_tier": self.agent_tier,
            "agent_role": self.agent_role,
            "data": self.data,
            "memory_id": self.memory_id,
            "outcome": self.outcome,
            "outcome_details": self.outcome_details,
        }
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str: str) -> ProgressEntry:
        """Deserialize from JSON string."""
        d = json.loads(json_str)
        return cls(
            event_type=EventType(d["event_type"]),
            task_id=d["task_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            agent_tier=d.get("agent_tier"),
            agent_role=d.get("agent_role"),
            data=d.get("data", {}),
            memory_id=d.get("memory_id"),
            outcome=d.get("outcome"),
            outcome_details=d.get("outcome_details"),
        )


class ProgressLogger:
    """
    Append-only progress logger for orchestration events.

    Log files are organized by date:
    - progress/2026-01-13.jsonl
    - progress/2026-01-14.jsonl
    - ...

    Each line is a JSON object (JSONL format).
    """

    def __init__(
        self,
        log_dir: Path = DEFAULT_LOG_PATH,
        buffer_size: int = 10,  # Flush after N entries
    ):
        self.log_dir = log_dir
        self.buffer_size = buffer_size
        self._buffer: List[ProgressEntry] = []
        self._disabled = False

        # Ensure log directory exists (fall back to temp dir if not writable)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fall back to temp directory (e.g., in CI environment)
            self.log_dir = _get_fallback_log_dir()
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                # Can't create any directory - disable logging
                self._disabled = True

    def _get_log_path(self, dt: datetime) -> Path:
        """Get log file path for a given datetime."""
        date_str = dt.strftime("%Y-%m-%d")
        return self.log_dir / f"{date_str}.jsonl"

    def log(self, entry: ProgressEntry) -> None:
        """
        Log a progress entry.

        Args:
            entry: Progress entry to log
        """
        if self._disabled:
            return

        self._buffer.append(entry)

        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered entries to disk."""
        if self._disabled or not self._buffer:
            return

        # Group entries by date
        by_date: Dict[str, List[ProgressEntry]] = {}
        for entry in self._buffer:
            date_key = entry.timestamp.strftime("%Y-%m-%d")
            if date_key not in by_date:
                by_date[date_key] = []
            by_date[date_key].append(entry)

        # Write to files
        for date_key, entries in by_date.items():
            log_path = self.log_dir / f"{date_key}.jsonl"
            with open(log_path, "a") as f:
                for entry in entries:
                    f.write(entry.to_json() + "\n")

        self._buffer.clear()

    def log_task_started(
        self,
        task_id: str,
        task_ir: Dict[str, Any],
        routing_decision: List[str],
        routing_strategy: str,  # "learned" or "rules"
    ) -> None:
        """Log task start with routing decision."""
        self.log(
            ProgressEntry(
                event_type=EventType.TASK_STARTED,
                task_id=task_id,
                data={
                    "task_type": task_ir.get("task_type"),
                    "objective": task_ir.get("objective", "")[:200],
                    "priority": task_ir.get("priority"),
                },
            )
        )

        self.log(
            ProgressEntry(
                event_type=EventType.ROUTING_DECISION,
                task_id=task_id,
                data={
                    "routing": routing_decision,
                    "strategy": routing_strategy,
                },
            )
        )

    def log_task_completed(
        self,
        task_id: str,
        success: bool,
        details: Optional[str] = None,
    ) -> None:
        """Log task completion."""
        self.log(
            ProgressEntry(
                event_type=EventType.TASK_COMPLETED if success else EventType.TASK_FAILED,
                task_id=task_id,
                outcome="success" if success else "failure",
                outcome_details=details,
            )
        )

    def log_gate_result(
        self,
        task_id: str,
        gate_name: str,
        passed: bool,
        agent_tier: str,
        agent_role: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Log gate pass/fail."""
        self.log(
            ProgressEntry(
                event_type=EventType.GATE_PASSED if passed else EventType.GATE_FAILED,
                task_id=task_id,
                agent_tier=agent_tier,
                agent_role=agent_role,
                data={
                    "gate_name": gate_name,
                    "error_message": error_message[:500] if error_message else None,
                },
                outcome="success" if passed else "failure",
            )
        )

    def log_escalation(
        self,
        task_id: str,
        from_tier: str,
        to_tier: str,
        reason: str,
        memory_id: Optional[str] = None,
    ) -> None:
        """Log escalation event."""
        self.log(
            ProgressEntry(
                event_type=EventType.ESCALATION_TRIGGERED,
                task_id=task_id,
                agent_tier=from_tier,
                data={
                    "from_tier": from_tier,
                    "to_tier": to_tier,
                    "reason": reason,
                },
                memory_id=memory_id,
            )
        )

    def log_escalation_outcome(
        self,
        task_id: str,
        resolved: bool,
        details: Optional[str] = None,
    ) -> None:
        """Log escalation resolution."""
        self.log(
            ProgressEntry(
                event_type=(
                    EventType.ESCALATION_RESOLVED if resolved else EventType.ESCALATION_FAILED
                ),
                task_id=task_id,
                outcome="success" if resolved else "failure",
                outcome_details=details,
            )
        )

    def log_exploration(
        self,
        task_id: str,
        query: str,
        strategy_used: str,
        tokens_spent: int,
        success: bool,
        function_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Log REPL exploration event with tool usage breakdown."""
        data: Dict[str, Any] = {
            "query": query[:200],
            "strategy": strategy_used,
            "tokens_spent": tokens_spent,
        }
        if function_counts:
            data["function_counts"] = function_counts
        self.log(
            ProgressEntry(
                event_type=EventType.EXPLORATION_COMPLETED,
                task_id=task_id,
                data=data,
                outcome="success" if success else "failure",
            )
        )

    def log_formalizer_invocation(
        self,
        task_id: str,
        service: str,
        endpoint: str,
        pages: int = 0,
        elapsed_sec: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Log document formalizer / OCR service invocation."""
        self.log(
            ProgressEntry(
                event_type=EventType.FORMALIZER_INVOKED,
                task_id=task_id,
                data={
                    "service": service,
                    "endpoint": endpoint,
                    "pages": pages,
                    "elapsed_sec": round(elapsed_sec, 2),
                },
                outcome="success" if success else "failure",
                outcome_details=error,
            )
        )

    def log_memory_update(
        self,
        memory_id: str,
        old_q: float,
        new_q: float,
        reward: float,
        task_id: str,
    ) -> None:
        """Log Q-value update (from Q-scorer agent)."""
        self.log(
            ProgressEntry(
                event_type=EventType.Q_VALUE_UPDATED,
                task_id=task_id,
                memory_id=memory_id,
                data={
                    "old_q": old_q,
                    "new_q": new_q,
                    "reward": reward,
                },
            )
        )

    # =========================================================================
    # Session lifecycle events
    # =========================================================================

    def log_session_created(
        self,
        session_id: str,
        task_id: str,
        name: Optional[str] = None,
        project: Optional[str] = None,
    ) -> None:
        """Log session creation event.

        Args:
            session_id: The session UUID.
            task_id: Initial task ID for MemRL lineage.
            name: Optional session name.
            project: Optional project identifier.
        """
        self.log(
            ProgressEntry(
                event_type=EventType.SESSION_CREATED,
                task_id=task_id,
                data={
                    "session_id": session_id,
                    "name": name,
                    "project": project,
                },
            )
        )

    def log_session_resumed(
        self,
        session_id: str,
        task_id: str,
        previous_task_id: str,
        resume_count: int,
        message_count: int,
        document_changes: int = 0,
    ) -> None:
        """Log session resume event.

        Args:
            session_id: The session UUID.
            task_id: New forked task ID (e.g., "original__r1").
            previous_task_id: Task ID before resume.
            resume_count: Total resume count for this session.
            message_count: Messages in session at resume time.
            document_changes: Number of source documents that changed.
        """
        self.log(
            ProgressEntry(
                event_type=EventType.SESSION_RESUMED,
                task_id=task_id,
                data={
                    "session_id": session_id,
                    "previous_task_id": previous_task_id,
                    "resume_count": resume_count,
                    "message_count": message_count,
                    "document_changes": document_changes,
                },
            )
        )

    def log_session_checkpointed(
        self,
        session_id: str,
        task_id: str,
        checkpoint_id: str,
        trigger: str,
        message_count: int,
        findings_synced: int = 0,
    ) -> None:
        """Log session checkpoint event.

        Args:
            session_id: The session UUID.
            task_id: Current task ID.
            checkpoint_id: The checkpoint UUID.
            trigger: What triggered checkpoint ("turns", "idle", "explicit").
            message_count: Messages at checkpoint time.
            findings_synced: Number of findings synced to store.
        """
        self.log(
            ProgressEntry(
                event_type=EventType.SESSION_CHECKPOINTED,
                task_id=task_id,
                data={
                    "session_id": session_id,
                    "checkpoint_id": checkpoint_id,
                    "trigger": trigger,
                    "message_count": message_count,
                    "findings_synced": findings_synced,
                },
            )
        )

    def log_session_archived(
        self,
        session_id: str,
        task_id: str,
        message_count: int,
        findings_count: int,
        summary_generated: bool = False,
    ) -> None:
        """Log session archive event.

        Args:
            session_id: The session UUID.
            task_id: Final task ID.
            message_count: Total messages in session.
            findings_count: Total findings in session.
            summary_generated: Whether LLM summary was generated.
        """
        self.log(
            ProgressEntry(
                event_type=EventType.SESSION_ARCHIVED,
                task_id=task_id,
                data={
                    "session_id": session_id,
                    "message_count": message_count,
                    "findings_count": findings_count,
                    "summary_generated": summary_generated,
                },
                outcome="success",
            )
        )

    def log_session_finding(
        self,
        session_id: str,
        task_id: str,
        finding_id: str,
        source: str,
        confidence: float,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log finding added to session.

        Args:
            session_id: The session UUID.
            task_id: Current task ID.
            finding_id: The finding UUID.
            source: Finding source ("user_marked", "heuristic", "llm_extracted").
            confidence: Finding confidence score (0-1).
            tags: Optional finding tags.
        """
        self.log(
            ProgressEntry(
                event_type=EventType.SESSION_FINDING_ADDED,
                task_id=task_id,
                data={
                    "session_id": session_id,
                    "finding_id": finding_id,
                    "source": source,
                    "confidence": confidence,
                    "tags": tags or [],
                },
            )
        )

    def __del__(self):
        """Flush on destruction."""
        self.flush()


class ProgressReader:
    """
    Reader for progress logs.

    Used by Q-scorer agent to process completed tasks.
    """

    def __init__(self, log_dir: Path = DEFAULT_LOG_PATH):
        self.log_dir = log_dir

    def read_date(self, date_str: str) -> List[ProgressEntry]:
        """Read all entries for a specific date."""
        log_path = self.log_dir / f"{date_str}.jsonl"
        if not log_path.exists():
            return []

        entries = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(ProgressEntry.from_json(line))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Skip malformed entries
                        continue
        return entries

    def read_recent(self, days: int = 7) -> List[ProgressEntry]:
        """Read entries from the last N days."""
        entries = []
        today = datetime.utcnow()

        for i in range(days):
            date = today - __import__("datetime").timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            entries.extend(self.read_date(date_str))

        return entries

    def get_unscored_tasks(self, days: int = 7) -> List[str]:
        """
        Find task IDs that have completed but not been Q-scored.

        Looks for TASK_COMPLETED without corresponding Q_VALUE_UPDATED.
        """
        entries = self.read_recent(days)

        # Track completed and scored tasks
        completed_tasks = set()
        scored_memory_ids = set()

        for entry in entries:
            if entry.event_type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED):
                completed_tasks.add(entry.task_id)
            elif entry.event_type == EventType.Q_VALUE_UPDATED:
                scored_memory_ids.add(entry.memory_id)

        # Find tasks whose routing memories haven't been scored
        unscored = []
        for entry in entries:
            if entry.event_type == EventType.ROUTING_DECISION:
                if entry.task_id in completed_tasks:
                    if entry.memory_id and entry.memory_id not in scored_memory_ids:
                        unscored.append(entry.task_id)
                    elif not entry.memory_id:
                        # No memory_id means it was rule-based - still need to score
                        unscored.append(entry.task_id)

        return list(set(unscored))

    def get_task_trajectory(self, task_id: str) -> List[ProgressEntry]:
        """Get all entries for a specific task."""
        # Search recent logs for this task
        entries = self.read_recent(days=30)
        return [e for e in entries if e.task_id == task_id]

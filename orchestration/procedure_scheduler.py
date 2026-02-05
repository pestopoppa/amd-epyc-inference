#!/usr/bin/env python3
"""Procedure scheduler for executing procedures with dependency tracking.

This module provides the ProcedureScheduler for scheduling and executing
procedures with proper dependency resolution, state management, and
execution context handling.

Usage:
    from orchestration.procedure_scheduler import ProcedureScheduler
    from orchestration.procedure_registry import ProcedureRegistry

    registry = ProcedureRegistry()
    scheduler = ProcedureScheduler(registry)

    # Schedule a single procedure
    job_id = scheduler.schedule("benchmark_model", model_path="/path/to/model.gguf")

    # Schedule with dependencies
    job_id = scheduler.schedule("update_registry", depends_on=["benchmark_model_job"])

    # Execute all scheduled procedures
    results = scheduler.run_all()
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from orchestration.procedure_registry import ProcedureRegistry, ProcedureResult


class JobStatus(Enum):
    """Status of a scheduled job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"  # Waiting on dependencies


@dataclass
class ScheduledJob:
    """A scheduled procedure execution."""

    job_id: str
    procedure_id: str
    inputs: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: ProcedureResult | None = None
    error: str | None = None
    priority: int = 0  # Higher = more urgent
    role: str | None = None
    retry_count: int = 0
    max_retries: int = 0


@dataclass
class SchedulerState:
    """Persistent state of the scheduler."""

    jobs: dict[str, ScheduledJob] = field(default_factory=dict)
    completed_jobs: list[str] = field(default_factory=list)
    failed_jobs: list[str] = field(default_factory=list)
    total_executed: int = 0
    total_succeeded: int = 0
    total_failed: int = 0


class ProcedureScheduler:
    """Scheduler for executing procedures with dependency tracking.

    Features:
    - Dependency resolution (topological sort)
    - Priority-based execution
    - State persistence
    - Retry handling
    - Concurrent execution (optional)
    """

    def __init__(
        self,
        registry: ProcedureRegistry,
        state_path: Path | str | None = None,
        max_concurrent: int = 1,
        persist_state: bool = True,
    ):
        """Initialize the scheduler.

        Args:
            registry: ProcedureRegistry instance.
            state_path: Path to persist scheduler state.
            max_concurrent: Maximum concurrent procedure executions.
            persist_state: Whether to persist state between runs.
        """
        self.registry = registry
        self.state_path = Path(state_path or "/mnt/raid0/llm/claude/orchestration/procedures/state/scheduler.json")
        self.max_concurrent = max_concurrent
        self.persist_state = persist_state

        # State
        self.state = SchedulerState()
        self._lock = threading.RLock()

        # Load persisted state
        if persist_state and self.state_path.exists():
            self._load_state()

    def schedule(
        self,
        procedure_id: str,
        depends_on: list[str] | None = None,
        priority: int = 0,
        role: str | None = None,
        max_retries: int = 0,
        **inputs: Any,
    ) -> str:
        """Schedule a procedure for execution.

        Args:
            procedure_id: ID of the procedure to execute.
            depends_on: List of job IDs that must complete first.
            priority: Execution priority (higher = more urgent).
            role: Role executing the procedure.
            max_retries: Number of retries on failure.
            **inputs: Input parameters for the procedure.

        Returns:
            Job ID for tracking.

        Raises:
            ValueError: If procedure doesn't exist.
        """
        # Validate procedure exists
        procedure = self.registry.get(procedure_id)
        if procedure is None:
            raise ValueError(f"Procedure not found: {procedure_id}")

        job_id = f"{procedure_id}_{uuid.uuid4().hex[:8]}"

        job = ScheduledJob(
            job_id=job_id,
            procedure_id=procedure_id,
            inputs=inputs,
            depends_on=depends_on or [],
            priority=priority,
            role=role,
            max_retries=max_retries,
        )

        with self._lock:
            self.state.jobs[job_id] = job

            # Mark as blocked if has dependencies
            if job.depends_on:
                job.status = JobStatus.BLOCKED

            if self.persist_state:
                self._save_state()

        return job_id

    def cancel(self, job_id: str) -> bool:
        """Cancel a scheduled job.

        Args:
            job_id: ID of job to cancel.

        Returns:
            True if cancelled, False if job not found or already completed.
        """
        with self._lock:
            job = self.state.jobs.get(job_id)
            if job is None:
                return False

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                return False

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()

            if self.persist_state:
                self._save_state()

            return True

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        """Get the status of a scheduled job.

        Args:
            job_id: Job ID to query.

        Returns:
            Job status dict or None if not found.
        """
        with self._lock:
            job = self.state.jobs.get(job_id)
            if job is None:
                return None

            return {
                "job_id": job.job_id,
                "procedure_id": job.procedure_id,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "depends_on": job.depends_on,
                "error": job.error,
                "result_success": job.result.success if job.result else None,
            }

    def list_jobs(
        self,
        status: JobStatus | None = None,
        procedure_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List scheduled jobs with optional filtering.

        Args:
            status: Filter by status.
            procedure_id: Filter by procedure ID.

        Returns:
            List of job status dicts.
        """
        with self._lock:
            jobs = []
            for job in self.state.jobs.values():
                if status and job.status != status:
                    continue
                if procedure_id and job.procedure_id != procedure_id:
                    continue

                jobs.append({
                    "job_id": job.job_id,
                    "procedure_id": job.procedure_id,
                    "status": job.status.value,
                    "priority": job.priority,
                    "created_at": job.created_at.isoformat(),
                })

            return sorted(jobs, key=lambda j: (-j["priority"], j["created_at"]))

    def get_ready_jobs(self) -> list[ScheduledJob]:
        """Get jobs that are ready to execute (no unmet dependencies).

        Returns:
            List of jobs ready to run, sorted by priority.
        """
        with self._lock:
            ready = []
            completed_ids = set(self.state.completed_jobs)

            for job in self.state.jobs.values():
                if job.status not in (JobStatus.PENDING, JobStatus.BLOCKED):
                    continue

                # Check if all dependencies are satisfied
                deps_satisfied = all(
                    dep in completed_ids or
                    self.state.jobs.get(dep, ScheduledJob("", "", {})).status == JobStatus.COMPLETED
                    for dep in job.depends_on
                )

                if deps_satisfied:
                    ready.append(job)

            # Sort by priority (descending) then creation time
            return sorted(ready, key=lambda j: (-j.priority, j.created_at))

    def run_one(self, job_id: str | None = None) -> ProcedureResult | None:
        """Execute a single job.

        Args:
            job_id: Specific job to run, or None to run next ready job.

        Returns:
            ProcedureResult or None if no job available.
        """
        with self._lock:
            if job_id:
                job = self.state.jobs.get(job_id)
                if job is None:
                    return None
            else:
                ready = self.get_ready_jobs()
                if not ready:
                    return None
                job = ready[0]

            # Mark as running
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

        # Execute outside lock
        try:
            result = self.registry.execute(
                job.procedure_id,
                role=job.role,
                **job.inputs,
            )

            with self._lock:
                job.result = result
                job.completed_at = datetime.now()

                if result.success:
                    job.status = JobStatus.COMPLETED
                    self.state.completed_jobs.append(job.job_id)
                    self.state.total_succeeded += 1
                else:
                    # Check retry
                    if job.retry_count < job.max_retries:
                        job.retry_count += 1
                        job.status = JobStatus.PENDING
                        job.error = f"Retry {job.retry_count}/{job.max_retries}: {result.error}"
                    else:
                        job.status = JobStatus.FAILED
                        job.error = result.error
                        self.state.failed_jobs.append(job.job_id)
                        self.state.total_failed += 1

                self.state.total_executed += 1

                if self.persist_state:
                    self._save_state()

            return result

        except Exception as e:
            with self._lock:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = str(e)
                self.state.failed_jobs.append(job.job_id)
                self.state.total_failed += 1
                self.state.total_executed += 1

                if self.persist_state:
                    self._save_state()

            return None

    def run_all(self, timeout_seconds: int = 3600) -> list[ProcedureResult]:
        """Execute all pending jobs in dependency order.

        Args:
            timeout_seconds: Maximum time to run.

        Returns:
            List of results from executed procedures.
        """
        results = []
        start_time = time.time()

        while True:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                break

            # Get ready jobs
            ready = self.get_ready_jobs()
            if not ready:
                # Check if there are still pending/blocked jobs
                with self._lock:
                    pending = [
                        j for j in self.state.jobs.values()
                        if j.status in (JobStatus.PENDING, JobStatus.BLOCKED)
                    ]
                if not pending:
                    break

                # Still waiting on dependencies
                time.sleep(0.1)
                continue

            # Run next job
            result = self.run_one()
            if result:
                results.append(result)

        return results

    def clear_completed(self, before: datetime | None = None) -> int:
        """Clear completed jobs from state.

        Args:
            before: Only clear jobs completed before this time.

        Returns:
            Number of jobs cleared.
        """
        with self._lock:
            to_remove = []
            for job_id, job in self.state.jobs.items():
                if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    continue
                if before and job.completed_at and job.completed_at > before:
                    continue
                to_remove.append(job_id)

            for job_id in to_remove:
                del self.state.jobs[job_id]

            if self.persist_state:
                self._save_state()

            return len(to_remove)

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dict with execution statistics.
        """
        with self._lock:
            status_counts = {}
            for job in self.state.jobs.values():
                status = job.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "total_jobs": len(self.state.jobs),
                "total_executed": self.state.total_executed,
                "total_succeeded": self.state.total_succeeded,
                "total_failed": self.state.total_failed,
                "success_rate": (
                    self.state.total_succeeded / self.state.total_executed
                    if self.state.total_executed > 0 else 0
                ),
                "by_status": status_counts,
            }

    def _load_state(self) -> None:
        """Load state from disk."""
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct jobs
            for job_data in data.get("jobs", {}).values():
                job = ScheduledJob(
                    job_id=job_data["job_id"],
                    procedure_id=job_data["procedure_id"],
                    inputs=job_data.get("inputs", {}),
                    status=JobStatus(job_data.get("status", "pending")),
                    depends_on=job_data.get("depends_on", []),
                    created_at=datetime.fromisoformat(job_data["created_at"]),
                    priority=job_data.get("priority", 0),
                    role=job_data.get("role"),
                )
                if job_data.get("started_at"):
                    job.started_at = datetime.fromisoformat(job_data["started_at"])
                if job_data.get("completed_at"):
                    job.completed_at = datetime.fromisoformat(job_data["completed_at"])
                job.error = job_data.get("error")
                self.state.jobs[job.job_id] = job

            self.state.completed_jobs = data.get("completed_jobs", [])
            self.state.failed_jobs = data.get("failed_jobs", [])
            self.state.total_executed = data.get("total_executed", 0)
            self.state.total_succeeded = data.get("total_succeeded", 0)
            self.state.total_failed = data.get("total_failed", 0)

        except Exception as e:
            # Start fresh if load fails
            logger.debug("Failed to load scheduler state, starting fresh: %s", e)
            self.state = SchedulerState()

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize jobs
            jobs_data = {}
            for job_id, job in self.state.jobs.items():
                jobs_data[job_id] = {
                    "job_id": job.job_id,
                    "procedure_id": job.procedure_id,
                    "inputs": job.inputs,
                    "status": job.status.value,
                    "depends_on": job.depends_on,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "priority": job.priority,
                    "role": job.role,
                    "error": job.error,
                }

            data = {
                "jobs": jobs_data,
                "completed_jobs": self.state.completed_jobs,
                "failed_jobs": self.state.failed_jobs,
                "total_executed": self.state.total_executed,
                "total_succeeded": self.state.total_succeeded,
                "total_failed": self.state.total_failed,
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.debug("Failed to save scheduler state: %s", e)  # Best effort persistence

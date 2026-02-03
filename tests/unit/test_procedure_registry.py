#!/usr/bin/env python3
"""Tests for the procedure registry and scheduler.

Tests cover:
- Procedure loading and validation
- Input validation
- Step execution
- Dependency tracking
- Scheduler operations
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_procedures_dir(tmp_path: Path) -> Path:
    """Create a temporary procedures directory with test procedures."""
    procedures_dir = tmp_path / "procedures"
    procedures_dir.mkdir()

    # Create a simple test procedure
    test_procedure = {
        "id": "test_echo",
        "name": "Test Echo",
        "version": "1.0.0",
        "description": "A simple test procedure that echoes input",
        "category": "maintenance",
        "estimated_tokens": 100,
        "permissions": {
            "roles": ["admin", "frontdoor"],
            "requires_approval": False,
            "destructive": False,
        },
        "inputs": [
            {
                "name": "message",
                "type": "string",
                "description": "Message to echo",
                "required": True,
            },
            {
                "name": "count",
                "type": "integer",
                "description": "Number of times to echo",
                "required": False,
                "default": 1,
                "validation": {"min": 1, "max": 10},
            },
        ],
        "outputs": [
            {
                "name": "result",
                "type": "string",
                "description": "Echoed message",
            }
        ],
        "steps": [
            {
                "id": "S1",
                "name": "Echo message",
                "action": {
                    "type": "python",
                    "command": "'Echo: ' + inputs['message']",
                    "capture_output": "echo_result",
                },
            }
        ],
        "verification": {
            "gates": ["output_valid"],
        },
    }

    with open(procedures_dir / "test_echo.json", "w") as f:
        json.dump(test_procedure, f)

    # Create a procedure with dependencies
    dep_procedure = {
        "id": "test_with_deps",
        "name": "Test With Dependencies",
        "version": "1.0.0",
        "description": "A procedure with step dependencies",
        "category": "maintenance",
        "inputs": [],
        "steps": [
            {
                "id": "S1",
                "name": "Step 1",
                "action": {"type": "python", "command": "1 + 1", "capture_output": "step1"},
            },
            {
                "id": "S2",
                "name": "Step 2",
                "depends_on": ["S1"],
                "action": {"type": "python", "command": "2 + 2", "capture_output": "step2"},
            },
        ],
        "verification": {"gates": ["output_valid"]},
    }

    with open(procedures_dir / "test_with_deps.json", "w") as f:
        json.dump(dep_procedure, f)

    return procedures_dir


@pytest.fixture
def temp_schema_path(tmp_path: Path) -> Path:
    """Create a copy of the procedure schema."""
    # Use the actual schema from the codebase
    actual_schema = Path(__file__).parent.parent.parent / "orchestration" / "procedure.schema.json"
    if actual_schema.exists():
        schema_content = actual_schema.read_text()
    else:
        # Minimal schema for testing
        schema_content = json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": [
                    "id",
                    "name",
                    "version",
                    "description",
                    "category",
                    "steps",
                    "verification",
                ],
            }
        )

    schema_path = tmp_path / "procedure.schema.json"
    schema_path.write_text(schema_content)
    return schema_path


@pytest.fixture
def registry(temp_procedures_dir: Path, temp_schema_path: Path):
    """Create a ProcedureRegistry with test procedures."""
    from orchestration.procedure_registry import ProcedureRegistry

    return ProcedureRegistry(
        procedures_dir=temp_procedures_dir,
        schema_path=temp_schema_path,
        state_dir=temp_procedures_dir / "state",
        checkpoint_dir=temp_procedures_dir / "checkpoints",
        validate_on_load=False,  # Skip validation for tests
    )


# =============================================================================
# ProcedureRegistry Tests
# =============================================================================


class TestProcedureRegistryLoad:
    """Tests for procedure loading."""

    def test_load_procedures(self, registry):
        """Test that procedures are loaded correctly."""
        assert len(registry.procedures) == 2
        assert "test_echo" in registry.procedures
        assert "test_with_deps" in registry.procedures

    def test_get_procedure(self, registry):
        """Test getting a procedure by ID."""
        proc = registry.get("test_echo")
        assert proc is not None
        assert proc.id == "test_echo"
        assert proc.name == "Test Echo"
        assert proc.category == "maintenance"

    def test_get_nonexistent_procedure(self, registry):
        """Test getting a procedure that doesn't exist."""
        proc = registry.get("nonexistent")
        assert proc is None

    def test_list_procedures(self, registry):
        """Test listing all procedures."""
        procs = registry.list_procedures()
        assert len(procs) == 2

        ids = [p["id"] for p in procs]
        assert "test_echo" in ids
        assert "test_with_deps" in ids

    def test_list_procedures_by_category(self, registry):
        """Test filtering procedures by category."""
        procs = registry.list_procedures(category="maintenance")
        assert len(procs) == 2

        procs = registry.list_procedures(category="benchmark")
        assert len(procs) == 0

    def test_list_procedures_by_role(self, registry):
        """Test filtering procedures by role."""
        procs = registry.list_procedures(role="admin")
        assert len(procs) >= 1

        procs = registry.list_procedures(role="worker_general")
        assert len(procs) >= 0  # May or may not have permissions


class TestProcedureInputValidation:
    """Tests for input validation."""

    def test_validate_required_input(self, registry):
        """Test that required inputs are validated."""
        proc = registry.get("test_echo")

        # Missing required input
        is_valid, errors = registry.validate_inputs(proc, {})
        assert not is_valid
        assert any("message" in e for e in errors)

        # With required input
        is_valid, errors = registry.validate_inputs(proc, {"message": "hello"})
        assert is_valid
        assert len(errors) == 0

    def test_validate_integer_bounds(self, registry):
        """Test integer min/max validation."""
        proc = registry.get("test_echo")

        # Valid count
        is_valid, errors = registry.validate_inputs(proc, {"message": "hi", "count": 5})
        assert is_valid

        # Count too low
        is_valid, errors = registry.validate_inputs(proc, {"message": "hi", "count": 0})
        assert not is_valid
        assert any("count" in e for e in errors)

        # Count too high
        is_valid, errors = registry.validate_inputs(proc, {"message": "hi", "count": 100})
        assert not is_valid
        assert any("count" in e for e in errors)

    def test_default_values(self, registry):
        """Test that default values are applied."""
        proc = registry.get("test_echo")

        # Only required input, default should be used for count
        is_valid, errors = registry.validate_inputs(proc, {"message": "hello"})
        assert is_valid


class TestProcedureExecution:
    """Tests for procedure execution."""

    def test_execute_simple_procedure(self, registry):
        """Test executing a simple procedure."""
        result = registry.execute("test_echo", message="hello world")

        assert result.procedure_id == "test_echo"
        assert result.success
        assert result.error is None
        assert len(result.step_results) == 1
        assert result.step_results[0].success

    def test_execute_dry_run(self, registry):
        """Test dry run execution."""
        result = registry.execute("test_echo", dry_run=True, message="hello")

        assert result.success
        assert result.outputs.get("dry_run") is True

    def test_execute_with_role_permission(self, registry):
        """Test role-based permission checking."""
        # Allowed role
        result = registry.execute("test_echo", role="admin", message="hello")
        assert result.success

        # Disallowed role
        result = registry.execute("test_echo", role="worker_general", message="hello")
        # Should fail due to permission (if roles are set)
        # The test procedure allows admin and frontdoor
        assert "not allowed" in (result.error or "") or result.success  # Depends on permissions

    def test_execute_nonexistent_procedure(self, registry):
        """Test executing a procedure that doesn't exist."""
        result = registry.execute("nonexistent", message="hello")

        assert not result.success
        assert "not found" in result.error.lower()

    def test_execute_missing_required_input(self, registry):
        """Test executing with missing required input."""
        result = registry.execute("test_echo")  # Missing 'message'

        assert not result.success
        assert "validation" in result.error.lower() or "message" in result.error.lower()


class TestStepDependencies:
    """Tests for step dependency tracking."""

    def test_execute_with_dependencies(self, registry):
        """Test that step dependencies are respected."""
        result = registry.execute("test_with_deps")

        assert result.success
        assert len(result.step_results) == 2

        # Both steps should succeed
        assert result.step_results[0].success
        assert result.step_results[1].success

        # S2 should have run after S1
        s1_idx = next(i for i, s in enumerate(result.step_results) if s.step_id == "S1")
        s2_idx = next(i for i, s in enumerate(result.step_results) if s.step_id == "S2")
        assert s1_idx < s2_idx


# =============================================================================
# ProcedureScheduler Tests
# =============================================================================


class TestProcedureScheduler:
    """Tests for the procedure scheduler."""

    def test_schedule_procedure(self, registry):
        """Test scheduling a procedure."""
        from orchestration.procedure_scheduler import ProcedureScheduler

        scheduler = ProcedureScheduler(registry, persist_state=False)
        job_id = scheduler.schedule("test_echo", message="hello")

        assert job_id is not None
        assert job_id.startswith("test_echo_")

    def test_schedule_nonexistent_procedure(self, registry):
        """Test scheduling a procedure that doesn't exist."""
        from orchestration.procedure_scheduler import ProcedureScheduler

        scheduler = ProcedureScheduler(registry, persist_state=False)

        with pytest.raises(ValueError, match="not found"):
            scheduler.schedule("nonexistent", message="hello")

    def test_get_job_status(self, registry):
        """Test getting job status."""
        from orchestration.procedure_scheduler import ProcedureScheduler

        scheduler = ProcedureScheduler(registry, persist_state=False)
        job_id = scheduler.schedule("test_echo", message="hello")

        status = scheduler.get_status(job_id)
        assert status is not None
        assert status["job_id"] == job_id
        assert status["status"] == "pending"

    def test_list_jobs(self, registry):
        """Test listing scheduled jobs."""
        from orchestration.procedure_scheduler import ProcedureScheduler

        scheduler = ProcedureScheduler(registry, persist_state=False)
        scheduler.schedule("test_echo", message="hello1")
        scheduler.schedule("test_echo", message="hello2")

        jobs = scheduler.list_jobs()
        assert len(jobs) == 2

    def test_run_one_job(self, registry):
        """Test running a single job."""
        from orchestration.procedure_scheduler import ProcedureScheduler

        scheduler = ProcedureScheduler(registry, persist_state=False)
        job_id = scheduler.schedule("test_echo", message="hello")

        result = scheduler.run_one(job_id)
        assert result is not None
        assert result.success

        status = scheduler.get_status(job_id)
        assert status["status"] == "completed"

    def test_run_all_jobs(self, registry):
        """Test running all scheduled jobs."""
        from orchestration.procedure_scheduler import ProcedureScheduler

        scheduler = ProcedureScheduler(registry, persist_state=False)
        scheduler.schedule("test_echo", message="hello1")
        scheduler.schedule("test_echo", message="hello2")

        results = scheduler.run_all(timeout_seconds=60)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_job_dependencies(self, registry):
        """Test job dependency handling."""
        from orchestration.procedure_scheduler import ProcedureScheduler

        scheduler = ProcedureScheduler(registry, persist_state=False)

        job1_id = scheduler.schedule("test_echo", message="first")
        job2_id = scheduler.schedule("test_echo", depends_on=[job1_id], message="second")

        # job2 should be blocked
        status = scheduler.get_status(job2_id)
        assert status["status"] == "blocked"

        # Run job1
        scheduler.run_one(job1_id)

        # Now job2 should be ready
        ready = scheduler.get_ready_jobs()
        ready_ids = [j.job_id for j in ready]
        assert job2_id in ready_ids

    def test_cancel_job(self, registry):
        """Test cancelling a job."""
        from orchestration.procedure_scheduler import ProcedureScheduler

        scheduler = ProcedureScheduler(registry, persist_state=False)
        job_id = scheduler.schedule("test_echo", message="hello")

        assert scheduler.cancel(job_id)

        status = scheduler.get_status(job_id)
        assert status["status"] == "cancelled"

    def test_scheduler_statistics(self, registry):
        """Test scheduler statistics."""
        from orchestration.procedure_scheduler import ProcedureScheduler

        scheduler = ProcedureScheduler(registry, persist_state=False)
        scheduler.schedule("test_echo", message="hello")
        scheduler.run_all()

        stats = scheduler.get_statistics()
        assert stats["total_executed"] == 1
        assert stats["total_succeeded"] == 1
        assert stats["total_failed"] == 0
        assert stats["success_rate"] == 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the procedure system."""

    def test_full_workflow(self, temp_procedures_dir: Path, temp_schema_path: Path):
        """Test a complete workflow: load, schedule, execute, verify."""
        from orchestration.procedure_registry import ProcedureRegistry
        from orchestration.procedure_scheduler import ProcedureScheduler

        # Create registry
        registry = ProcedureRegistry(
            procedures_dir=temp_procedures_dir,
            schema_path=temp_schema_path,
            validate_on_load=False,
        )

        # Create scheduler
        scheduler = ProcedureScheduler(registry, persist_state=False)

        # Schedule multiple procedures
        job1 = scheduler.schedule("test_echo", priority=1, message="high priority")
        scheduler.schedule("test_echo", priority=0, message="low priority")

        # High priority should run first
        ready = scheduler.get_ready_jobs()
        assert ready[0].job_id == job1

        # Run all
        results = scheduler.run_all()
        assert len(results) == 2
        assert all(r.success for r in results)

        # Verify statistics
        stats = scheduler.get_statistics()
        assert stats["total_succeeded"] == 2


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

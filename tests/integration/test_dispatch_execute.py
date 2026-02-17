"""Integration tests for the dispatch-execute flow.

These tests verify the full pipeline from TaskIR input through
Dispatcher to Executor, ensuring all components work together.
"""

from pathlib import Path

import pytest
import yaml

from src.context_manager import ContextManager
from src.dispatcher import Dispatcher
from src.executor import Executor, ExecutorConfig, StepStatus
from src.model_server import ModelServer
from src.registry_loader import RegistryLoader


@pytest.fixture
def test_registry(tmp_path: Path) -> RegistryLoader:
    """Create a test registry with mock models.

    Uses production role names (coder_escalation, worker_general, etc.) to match
    the dispatcher's role mapping.
    """
    registry = {
        "runtime_defaults": {
            "model_base_path": str(tmp_path),
            "threads": 96,
            "context_length": 8192,
        },
        "roles": {
            "frontdoor": {
                "tier": "A",
                "description": "Front door agent",
                "model": {
                    "name": "frontdoor-model",
                    "path": "frontdoor.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 1.0,
                },
                "acceleration": {"type": "none"},
                "performance": {"baseline_tps": 40.0},
                "memory": {"residency": "hot"},
            },
            "coder_escalation": {
                "tier": "B",
                "description": "Code generation",
                "model": {
                    "name": "coder-model",
                    "path": "coder.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 10.0,
                },
                "acceleration": {"type": "none"},
                "performance": {"baseline_tps": 30.0},
                "memory": {"residency": "hot"},
            },
            "worker_general": {
                "tier": "C",
                "description": "General worker / reviewer",
                "model": {
                    "name": "worker-model",
                    "path": "worker.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 5.0,
                },
                "acceleration": {"type": "none"},
                "performance": {"baseline_tps": 50.0},
                "memory": {"residency": "warm"},
            },
        },
        "command_templates": {
            "baseline": "echo 'test output for {role}'",
        },
    }

    # Create mock model files
    for role_config in registry["roles"].values():
        model_path = tmp_path / role_config["model"]["path"]
        model_path.touch()

    registry_path = tmp_path / "registry.yaml"
    with registry_path.open("w") as f:
        yaml.dump(registry, f)

    return RegistryLoader(registry_path)


@pytest.fixture
def dispatcher(test_registry: RegistryLoader) -> Dispatcher:
    """Create a dispatcher with test registry."""
    return Dispatcher(registry=test_registry, validate_paths=False)


@pytest.fixture
def executor(test_registry: RegistryLoader) -> Executor:
    """Create an executor with dry run config."""
    server = ModelServer(registry=test_registry)
    config = ExecutorConfig(dry_run=True)
    return Executor(model_server=server, config=config, registry=test_registry)


class TestDispatchExecuteFlow:
    """Integration tests for dispatch-execute flow."""

    def test_simple_single_step_task(self, dispatcher: Dispatcher, executor: Executor):
        """Test a simple task with one step."""
        task_ir = {
            "task_id": "simple-task",
            "task_type": "code",
            "priority": "interactive",
            "objective": "Write a hello world function",
            "agents": [{"role": "coder"}],
            "plan": {
                "steps": [
                    {
                        "id": "S1",
                        "actor": "coder",
                        "action": "Write a hello world function in Python",
                        "inputs": [],
                        "outputs": ["hello.py"],
                        "depends_on": [],
                    }
                ]
            },
        }

        # Dispatch
        dispatch_result = dispatcher.dispatch(task_ir)
        assert len(dispatch_result.steps) == 1
        assert dispatch_result.steps[0].step_id == "S1"

        # Execute
        execution_result = executor.execute(dispatch_result)
        assert execution_result.status == StepStatus.COMPLETED
        assert execution_result.successful_steps == 1
        assert execution_result.failed_steps == 0

    def test_multi_step_sequential_task(self, dispatcher: Dispatcher, executor: Executor):
        """Test a task with multiple sequential steps."""
        task_ir = {
            "task_id": "sequential-task",
            "task_type": "code",
            "priority": "interactive",
            "objective": "Create and test a function",
            "agents": [{"role": "coder"}, {"role": "reviewer"}],
            "plan": {
                "steps": [
                    {
                        "id": "S1",
                        "actor": "coder",
                        "action": "Write a function",
                        "inputs": [],
                        "outputs": ["function.py"],
                        "depends_on": [],
                    },
                    {
                        "id": "S2",
                        "actor": "coder",
                        "action": "Write tests for the function",
                        "inputs": ["function.py"],
                        "outputs": ["test_function.py"],
                        "depends_on": ["S1"],
                    },
                    {
                        "id": "S3",
                        "actor": "reviewer",
                        "action": "Review the code and tests",
                        "inputs": ["function.py", "test_function.py"],
                        "outputs": ["review.md"],
                        "depends_on": ["S2"],
                    },
                ]
            },
        }

        # Dispatch
        dispatch_result = dispatcher.dispatch(task_ir)
        assert len(dispatch_result.steps) == 3

        # Execute
        execution_result = executor.execute(dispatch_result)
        assert execution_result.status == StepStatus.COMPLETED
        assert execution_result.successful_steps == 3

        # Verify step order
        s1 = execution_result.steps["S1"]
        s2 = execution_result.steps["S2"]
        s3 = execution_result.steps["S3"]

        assert s1.completed_at <= s2.started_at
        assert s2.completed_at <= s3.started_at

    def test_parallel_steps_task(self, dispatcher: Dispatcher, executor: Executor):
        """Test a task with parallel execution groups."""
        task_ir = {
            "task_id": "parallel-task",
            "task_type": "code",
            "priority": "batch",
            "objective": "Create multiple components",
            "agents": [{"role": "coder"}, {"role": "worker"}],
            "plan": {
                "steps": [
                    {
                        "id": "S1",
                        "actor": "coder",
                        "action": "Write component A",
                        "inputs": [],
                        "outputs": ["component_a.py"],
                        "depends_on": [],
                        "parallel_group": "components",
                    },
                    {
                        "id": "S2",
                        "actor": "coder",
                        "action": "Write component B",
                        "inputs": [],
                        "outputs": ["component_b.py"],
                        "depends_on": [],
                        "parallel_group": "components",
                    },
                    {
                        "id": "S3",
                        "actor": "worker",
                        "action": "Integrate components",
                        "inputs": ["component_a.py", "component_b.py"],
                        "outputs": ["main.py"],
                        "depends_on": ["S1", "S2"],
                    },
                ]
            },
        }

        # Dispatch
        dispatch_result = dispatcher.dispatch(task_ir)
        assert len(dispatch_result.steps) == 3

        # Execute
        execution_result = executor.execute(dispatch_result)
        assert execution_result.status == StepStatus.COMPLETED
        assert execution_result.successful_steps == 3

        # Verify S3 started after both S1 and S2 completed
        s1 = execution_result.steps["S1"]
        s2 = execution_result.steps["S2"]
        s3 = execution_result.steps["S3"]

        assert s1.completed_at <= s3.started_at
        assert s2.completed_at <= s3.started_at

    def test_unknown_actor_fallback(self, dispatcher: Dispatcher, executor: Executor):
        """Test that unknown actors fall back to worker_general."""
        task_ir = {
            "task_id": "unknown-actor-task",
            "task_type": "code",
            "priority": "interactive",
            "objective": "Test unknown actor handling",
            "agents": [{"role": "coder"}, {"role": "unknown_role"}],
            "plan": {
                "steps": [
                    {
                        "id": "S1",
                        "actor": "coder",
                        "action": "Write code",
                        "inputs": [],
                        "outputs": ["code.py"],
                        "depends_on": [],
                    },
                    {
                        "id": "S2",
                        "actor": "unknown_role",
                        "action": "Do something",
                        "inputs": ["code.py"],
                        "outputs": ["result.txt"],
                        "depends_on": ["S1"],
                    },
                ]
            },
        }

        # Dispatch - should have warnings about unknown role fallback
        dispatch_result = dispatcher.dispatch(task_ir)
        assert any("unknown_role" in w.lower() for w in dispatch_result.warnings)

        # Execute - both steps should complete (unknown falls back to worker_general)
        execution_result = executor.execute(dispatch_result)

        # Both steps should complete via fallback
        assert execution_result.steps["S1"].status == StepStatus.COMPLETED
        assert execution_result.steps["S2"].status == StepStatus.COMPLETED


class TestDispatchExecuteWithContext:
    """Tests for context passing in dispatch-execute flow."""

    def test_context_passed_between_steps(
        self, dispatcher: Dispatcher, test_registry: RegistryLoader
    ):
        """Test that context is passed between steps."""
        server = ModelServer(registry=test_registry)
        config = ExecutorConfig(dry_run=True)
        executor = Executor(model_server=server, config=config, registry=test_registry)

        task_ir = {
            "task_id": "context-task",
            "task_type": "code",
            "priority": "interactive",
            "objective": "Test context passing",
            "agents": [{"role": "coder"}],
            "plan": {
                "steps": [
                    {
                        "id": "S1",
                        "actor": "coder",
                        "action": "Generate initial output",
                        "inputs": [],
                        "outputs": ["initial_output"],
                        "depends_on": [],
                    },
                    {
                        "id": "S2",
                        "actor": "coder",
                        "action": "Use initial output",
                        "inputs": ["initial_output"],
                        "outputs": ["final_output"],
                        "depends_on": ["S1"],
                    },
                ]
            },
        }

        dispatch_result = dispatcher.dispatch(task_ir)
        executor.execute(dispatch_result)

        # Check that context was populated
        context = executor.get_context()
        assert context.has("initial_output")
        assert context.has("final_output")

    def test_context_manager_integration(
        self, dispatcher: Dispatcher, test_registry: RegistryLoader
    ):
        """Test Context Manager integration with execution flow."""
        # Create components
        server = ModelServer(registry=test_registry)
        config = ExecutorConfig(dry_run=True)
        executor = Executor(model_server=server, config=config, registry=test_registry)
        ctx = ContextManager()

        task_ir = {
            "task_id": "ctx-manager-task",
            "task_type": "code",
            "priority": "interactive",
            "objective": "Test context manager",
            "agents": [{"role": "coder"}],
            "plan": {
                "steps": [
                    {
                        "id": "S1",
                        "actor": "coder",
                        "action": "Generate code",
                        "inputs": [],
                        "outputs": ["code.py"],
                        "depends_on": [],
                    }
                ]
            },
        }

        # Execute
        dispatch_result = dispatcher.dispatch(task_ir)
        executor.execute(dispatch_result)

        # Transfer executor context to ContextManager
        for key, value in executor.get_context().items():
            ctx.set(key, value, step_id="S1")

        # Verify context manager has the data
        assert ctx.has("code.py")
        entry = ctx.get_entry("code.py")
        assert entry.step_id == "S1"


class TestDispatchResultMetadata:
    """Tests for dispatch result metadata."""

    def test_dispatch_result_has_roles_used(self, dispatcher: Dispatcher):
        """Test that dispatch result includes roles used."""
        task_ir = {
            "task_id": "roles-task",
            "task_type": "code",
            "priority": "interactive",
            "objective": "Test roles tracking",
            "agents": [{"role": "coder"}, {"role": "worker"}],
            "plan": {
                "steps": [
                    {
                        "id": "S1",
                        "actor": "coder",
                        "action": "Write code",
                        "inputs": [],
                        "outputs": [],
                        "depends_on": [],
                    },
                    {
                        "id": "S2",
                        "actor": "worker",
                        "action": "Review code",
                        "inputs": [],
                        "outputs": [],
                        "depends_on": ["S1"],
                    },
                ]
            },
        }

        dispatch_result = dispatcher.dispatch(task_ir)

        # Dispatcher maps "coder" -> "coder_escalation", "worker" -> "worker_general"
        assert "coder_escalation" in dispatch_result.roles_used
        assert "worker_general" in dispatch_result.roles_used

    def test_dispatch_result_has_timestamp(self, dispatcher: Dispatcher):
        """Test that dispatch result has a timestamp."""
        task_ir = {
            "task_id": "timestamp-task",
            "task_type": "code",
            "priority": "interactive",
            "objective": "Test timestamp",
            "agents": [{"role": "coder"}],
            "plan": {
                "steps": [
                    {
                        "id": "S1",
                        "actor": "coder",
                        "action": "Write code",
                        "inputs": [],
                        "outputs": [],
                        "depends_on": [],
                    }
                ]
            },
        }

        dispatch_result = dispatcher.dispatch(task_ir)

        # Timestamp is an ISO format string
        assert dispatch_result.timestamp is not None
        assert len(dispatch_result.timestamp) > 0


class TestExecutionTiming:
    """Tests for execution timing and performance."""

    def test_execution_has_timing_info(self, dispatcher: Dispatcher, executor: Executor):
        """Test that execution results include timing information."""
        task_ir = {
            "task_id": "timing-task",
            "task_type": "code",
            "priority": "interactive",
            "objective": "Test timing",
            "agents": [{"role": "coder"}],
            "plan": {
                "steps": [
                    {
                        "id": "S1",
                        "actor": "coder",
                        "action": "Write code",
                        "inputs": [],
                        "outputs": [],
                        "depends_on": [],
                    }
                ]
            },
        }

        dispatch_result = dispatcher.dispatch(task_ir)
        execution_result = executor.execute(dispatch_result)

        # Check overall timing
        assert execution_result.started_at is not None
        assert execution_result.completed_at is not None
        assert execution_result.elapsed_time >= 0

        # Check step timing
        step_result = execution_result.steps["S1"]
        assert step_result.started_at is not None
        assert step_result.completed_at is not None
        assert step_result.elapsed_time >= 0

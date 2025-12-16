"""Unit tests for executor module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.dispatcher import DispatchResult, StepExecution
from src.executor import (
    ExecutionResult,
    Executor,
    ExecutorConfig,
    StepResult,
    StepStatus,
)
from src.model_server import ModelServer
from src.registry_loader import RegistryLoader, RoleConfig


@pytest.fixture
def minimal_registry(tmp_path: Path) -> RegistryLoader:
    """Create a minimal valid registry."""
    registry = {
        "runtime_defaults": {
            "model_base_path": str(tmp_path),
            "threads": 96,
            "context_length": 8192,
        },
        "roles": {
            "coder": {
                "tier": "C",
                "description": "Test coder role",
                "model": {
                    "name": "test-model",
                    "path": "test-model.gguf",
                    "quant": "Q4_K_M",
                    "size_gb": 1.0,
                },
                "acceleration": {"type": "none"},
                "performance": {"baseline_tps": 10.0},
                "memory": {"residency": "hot"},
            },
        },
        "command_templates": {
            "baseline": "echo 'test output'",
        },
    }

    (tmp_path / "test-model.gguf").touch()

    registry_path = tmp_path / "registry.yaml"
    with registry_path.open("w") as f:
        yaml.dump(registry, f)

    return RegistryLoader(registry_path)


@pytest.fixture
def mock_model_server(minimal_registry: RegistryLoader) -> ModelServer:
    """Create a mock model server."""
    server = ModelServer(registry=minimal_registry)
    return server


@pytest.fixture
def sample_dispatch_result() -> DispatchResult:
    """Create a sample dispatch result for testing."""
    role_config = MagicMock(spec=RoleConfig)
    role_config.name = "coder"

    steps = [
        StepExecution(
            step_id="S1",
            actor="coder",
            action="Write a hello world function",
            inputs=[],
            outputs=["hello.py"],
            depends_on=[],
            role_config=role_config,
            command="echo 'def hello(): pass'",
        ),
        StepExecution(
            step_id="S2",
            actor="coder",
            action="Write tests for hello world",
            inputs=["hello.py"],
            outputs=["test_hello.py"],
            depends_on=["S1"],
            role_config=role_config,
            command="echo 'def test_hello(): pass'",
        ),
    ]

    return DispatchResult(
        task_id="test-task",
        steps=steps,
        warnings=[],
        errors=[],
        timestamp=1702700000.0,
        roles_used=["coder"],
    )


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.WAITING.value == "waiting"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_creation(self):
        """Test creating a step result."""
        result = StepResult(
            step_id="S1",
            status=StepStatus.COMPLETED,
            output="Hello world",
        )

        assert result.step_id == "S1"
        assert result.status == StepStatus.COMPLETED
        assert result.output == "Hello world"
        assert result.artifacts == []
        assert result.error_message is None

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        result = StepResult(
            step_id="S1",
            status=StepStatus.COMPLETED,
            started_at=100.0,
            completed_at=105.5,
        )

        assert result.elapsed_time == 5.5

    def test_elapsed_time_no_times(self):
        """Test elapsed time with missing times."""
        result = StepResult(
            step_id="S1",
            status=StepStatus.PENDING,
        )

        assert result.elapsed_time == 0.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = StepResult(
            step_id="S1",
            status=StepStatus.COMPLETED,
            output="Hello world",
            artifacts=["file.py"],
            started_at=100.0,
            completed_at=105.0,
        )

        d = result.to_dict()
        assert d["step_id"] == "S1"
        assert d["status"] == "completed"
        assert d["output"] == "Hello world"
        assert d["artifacts"] == ["file.py"]
        assert d["elapsed_time"] == 5.0

    def test_to_dict_truncates_long_output(self):
        """Test that long outputs are truncated."""
        long_output = "x" * 1000
        result = StepResult(
            step_id="S1",
            status=StepStatus.COMPLETED,
            output=long_output,
        )

        d = result.to_dict()
        assert len(d["output"]) == 500


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_execution_result_creation(self):
        """Test creating an execution result."""
        result = ExecutionResult(
            task_id="test-task",
            status=StepStatus.COMPLETED,
        )

        assert result.task_id == "test-task"
        assert result.status == StepStatus.COMPLETED
        assert result.steps == {}
        assert result.warnings == []
        assert result.errors == []

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        result = ExecutionResult(
            task_id="test-task",
            status=StepStatus.COMPLETED,
            started_at=100.0,
            completed_at=110.0,
        )

        assert result.elapsed_time == 10.0

    def test_successful_steps_count(self):
        """Test counting successful steps."""
        result = ExecutionResult(
            task_id="test-task",
            status=StepStatus.COMPLETED,
            steps={
                "S1": StepResult(step_id="S1", status=StepStatus.COMPLETED),
                "S2": StepResult(step_id="S2", status=StepStatus.COMPLETED),
                "S3": StepResult(step_id="S3", status=StepStatus.FAILED),
            },
        )

        assert result.successful_steps == 2

    def test_failed_steps_count(self):
        """Test counting failed steps."""
        result = ExecutionResult(
            task_id="test-task",
            status=StepStatus.FAILED,
            steps={
                "S1": StepResult(step_id="S1", status=StepStatus.COMPLETED),
                "S2": StepResult(step_id="S2", status=StepStatus.FAILED),
                "S3": StepResult(step_id="S3", status=StepStatus.FAILED),
            },
        )

        assert result.failed_steps == 2

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = ExecutionResult(
            task_id="test-task",
            status=StepStatus.COMPLETED,
            started_at=100.0,
            completed_at=110.0,
            warnings=["warning1"],
            errors=["error1"],
        )

        d = result.to_dict()
        assert d["task_id"] == "test-task"
        assert d["status"] == "completed"
        assert d["elapsed_time"] == 10.0
        assert d["warnings"] == ["warning1"]
        assert d["errors"] == ["error1"]


class TestExecutorConfig:
    """Tests for ExecutorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExecutorConfig()

        assert config.max_parallel_workers == 2
        assert config.step_timeout == 300
        assert config.retry_failed_steps is False
        assert config.max_retries == 1
        assert config.dry_run is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExecutorConfig(
            max_parallel_workers=4,
            step_timeout=600,
            dry_run=True,
        )

        assert config.max_parallel_workers == 4
        assert config.step_timeout == 600
        assert config.dry_run is True


class TestExecutor:
    """Tests for Executor class."""

    def test_executor_creation(self, mock_model_server: ModelServer):
        """Test creating an executor."""
        executor = Executor(model_server=mock_model_server)

        assert executor.server is not None
        assert executor.config is not None

    def test_executor_with_config(self, mock_model_server: ModelServer):
        """Test creating executor with custom config."""
        config = ExecutorConfig(dry_run=True)
        executor = Executor(model_server=mock_model_server, config=config)

        assert executor.config.dry_run is True

    def test_dry_run_execution(
        self,
        mock_model_server: ModelServer,
        sample_dispatch_result: DispatchResult,
    ):
        """Test dry run execution."""
        config = ExecutorConfig(dry_run=True)
        executor = Executor(model_server=mock_model_server, config=config)

        result = executor.execute(sample_dispatch_result)

        assert result.status == StepStatus.COMPLETED
        assert result.successful_steps == 2
        assert result.failed_steps == 0
        assert "[DRY RUN]" in result.steps["S1"].output
        assert "[DRY RUN]" in result.steps["S2"].output

    def test_dependency_resolution(
        self,
        mock_model_server: ModelServer,
        sample_dispatch_result: DispatchResult,
    ):
        """Test that dependencies are respected."""
        config = ExecutorConfig(dry_run=True)
        executor = Executor(model_server=mock_model_server, config=config)

        result = executor.execute(sample_dispatch_result)

        # S1 should complete before S2
        s1_completed = result.steps["S1"].completed_at
        s2_started = result.steps["S2"].started_at

        assert s1_completed is not None
        assert s2_started is not None
        assert s1_completed <= s2_started

    def test_skipped_on_failed_dependency(
        self,
        mock_model_server: ModelServer,
    ):
        """Test that steps are skipped when dependencies fail."""
        # Create a dispatch result with a step that will fail
        role_config = MagicMock(spec=RoleConfig)
        role_config.name = "coder"

        steps = [
            StepExecution(
                step_id="S1",
                actor="coder",
                action="Failing step",
                inputs=[],
                outputs=[],
                depends_on=[],
                role_config=None,  # No role config = will be skipped
                command="",
            ),
            StepExecution(
                step_id="S2",
                actor="coder",
                action="Dependent step",
                inputs=[],
                outputs=[],
                depends_on=["S1"],
                role_config=role_config,
                command="echo 'test'",
            ),
        ]

        dispatch_result = DispatchResult(
            task_id="test-task",
            steps=steps,
            warnings=[],
            errors=[],
            timestamp=1702700000.0,
            roles_used=["coder"],
        )

        config = ExecutorConfig(dry_run=True)
        executor = Executor(model_server=mock_model_server, config=config)

        result = executor.execute(dispatch_result)

        # S1 should be skipped (no model)
        assert result.steps["S1"].status == StepStatus.SKIPPED
        # S2 should be skipped due to failed dependency
        assert result.steps["S2"].status == StepStatus.SKIPPED

    def test_context_passing(
        self,
        mock_model_server: ModelServer,
    ):
        """Test that outputs are passed to dependent steps."""
        config = ExecutorConfig(dry_run=True)
        executor = Executor(model_server=mock_model_server, config=config)

        role_config = MagicMock(spec=RoleConfig)
        role_config.name = "coder"

        steps = [
            StepExecution(
                step_id="S1",
                actor="coder",
                action="Create file",
                inputs=[],
                outputs=["output1"],
                depends_on=[],
                role_config=role_config,
                command="echo 'content'",
            ),
        ]

        dispatch_result = DispatchResult(
            task_id="test-task",
            steps=steps,
            warnings=[],
            errors=[],
            timestamp=1702700000.0,
            roles_used=["coder"],
        )

        executor.execute(dispatch_result)
        context = executor.get_context()

        # In dry run mode, context should have the dry run output
        assert context.has("output1")
        assert "[DRY RUN]" in context.get("output1")

    def test_clear_context(self, mock_model_server: ModelServer):
        """Test clearing the context."""
        executor = Executor(model_server=mock_model_server)
        executor.context.set("test_key", "test_value")

        executor.clear_context()

        assert executor.context.count() == 0

    def test_warnings_errors_propagated(
        self,
        mock_model_server: ModelServer,
    ):
        """Test that warnings and errors from dispatch are propagated."""
        role_config = MagicMock(spec=RoleConfig)
        role_config.name = "coder"

        dispatch_result = DispatchResult(
            task_id="test-task",
            steps=[
                StepExecution(
                    step_id="S1",
                    actor="coder",
                    action="Test",
                    inputs=[],
                    outputs=[],
                    depends_on=[],
                    role_config=role_config,
                    command="echo 'test'",
                ),
            ],
            warnings=["dispatch warning"],
            errors=["dispatch error"],
            timestamp=1702700000.0,
            roles_used=["coder"],
        )

        config = ExecutorConfig(dry_run=True)
        executor = Executor(model_server=mock_model_server, config=config)

        result = executor.execute(dispatch_result)

        assert "dispatch warning" in result.warnings
        assert "dispatch error" in result.errors


class TestExecutorFindReadySteps:
    """Tests for _find_ready_steps method."""

    def test_find_ready_no_deps(self, mock_model_server: ModelServer):
        """Test finding steps with no dependencies."""
        executor = Executor(model_server=mock_model_server)

        role_config = MagicMock(spec=RoleConfig)
        steps = [
            StepExecution(
                step_id="S1",
                actor="coder",
                action="Test",
                inputs=[],
                outputs=[],
                depends_on=[],
                role_config=role_config,
                command="",
            ),
            StepExecution(
                step_id="S2",
                actor="coder",
                action="Test",
                inputs=[],
                outputs=[],
                depends_on=[],
                role_config=role_config,
                command="",
            ),
        ]

        completed: set[str] = set()
        results = {
            "S1": StepResult(step_id="S1", status=StepStatus.PENDING),
            "S2": StepResult(step_id="S2", status=StepStatus.PENDING),
        }

        ready = executor._find_ready_steps(steps, completed, results)

        assert len(ready) == 2
        assert ready[0].step_id == "S1"
        assert ready[1].step_id == "S2"

    def test_find_ready_with_deps(self, mock_model_server: ModelServer):
        """Test finding steps with dependencies."""
        executor = Executor(model_server=mock_model_server)

        role_config = MagicMock(spec=RoleConfig)
        steps = [
            StepExecution(
                step_id="S1",
                actor="coder",
                action="Test",
                inputs=[],
                outputs=[],
                depends_on=[],
                role_config=role_config,
                command="",
            ),
            StepExecution(
                step_id="S2",
                actor="coder",
                action="Test",
                inputs=[],
                outputs=[],
                depends_on=["S1"],
                role_config=role_config,
                command="",
            ),
        ]

        completed: set[str] = set()
        results = {
            "S1": StepResult(step_id="S1", status=StepStatus.PENDING),
            "S2": StepResult(step_id="S2", status=StepStatus.PENDING),
        }

        ready = executor._find_ready_steps(steps, completed, results)

        # Only S1 should be ready (S2 depends on S1)
        assert len(ready) == 1
        assert ready[0].step_id == "S1"

    def test_find_ready_after_completion(self, mock_model_server: ModelServer):
        """Test finding steps after dependency completes."""
        executor = Executor(model_server=mock_model_server)

        role_config = MagicMock(spec=RoleConfig)
        steps = [
            StepExecution(
                step_id="S1",
                actor="coder",
                action="Test",
                inputs=[],
                outputs=[],
                depends_on=[],
                role_config=role_config,
                command="",
            ),
            StepExecution(
                step_id="S2",
                actor="coder",
                action="Test",
                inputs=[],
                outputs=[],
                depends_on=["S1"],
                role_config=role_config,
                command="",
            ),
        ]

        completed = {"S1"}
        results = {
            "S1": StepResult(step_id="S1", status=StepStatus.COMPLETED),
            "S2": StepResult(step_id="S2", status=StepStatus.PENDING),
        }

        ready = executor._find_ready_steps(steps, completed, results)

        # Only S2 should be ready (S1 already completed)
        assert len(ready) == 1
        assert ready[0].step_id == "S2"


class TestExecutorGroupByParallel:
    """Tests for _group_by_parallel method."""

    def test_group_by_parallel(self, mock_model_server: ModelServer):
        """Test grouping steps by parallel group."""
        executor = Executor(model_server=mock_model_server)

        role_config = MagicMock(spec=RoleConfig)
        steps = [
            StepExecution(
                step_id="S1",
                actor="coder",
                action="Test",
                inputs=[],
                outputs=[],
                depends_on=[],
                role_config=role_config,
                command="",
                parallel_group="group1",
            ),
            StepExecution(
                step_id="S2",
                actor="coder",
                action="Test",
                inputs=[],
                outputs=[],
                depends_on=[],
                role_config=role_config,
                command="",
                parallel_group="group1",
            ),
            StepExecution(
                step_id="S3",
                actor="coder",
                action="Test",
                inputs=[],
                outputs=[],
                depends_on=[],
                role_config=role_config,
                command="",
                parallel_group="group2",
            ),
        ]

        groups = executor._group_by_parallel(steps)

        assert len(groups) == 2
        assert len(groups["group1"]) == 2
        assert len(groups["group2"]) == 1


class TestExecutorBuildPrompt:
    """Tests for _build_prompt method."""

    def test_build_prompt_basic(self, mock_model_server: ModelServer):
        """Test building a basic prompt."""
        executor = Executor(model_server=mock_model_server)

        role_config = MagicMock(spec=RoleConfig)
        step = StepExecution(
            step_id="S1",
            actor="coder",
            action="Write a hello world function",
            inputs=[],
            outputs=["hello.py"],
            depends_on=[],
            role_config=role_config,
            command="",
        )

        prompt = executor._build_prompt(step)

        assert "Write a hello world function" in prompt
        assert "hello.py" in prompt

    def test_build_prompt_with_context(self, mock_model_server: ModelServer):
        """Test building prompt with context from previous step."""
        executor = Executor(model_server=mock_model_server)
        executor.context.set("previous_output", "Previous step result", step_id="S1")

        role_config = MagicMock(spec=RoleConfig)
        step = StepExecution(
            step_id="S2",
            actor="coder",
            action="Use the previous output",
            inputs=["previous_output"],
            outputs=[],
            depends_on=["S1"],
            role_config=role_config,
            command="",
        )

        prompt = executor._build_prompt(step)

        assert "Use the previous output" in prompt
        assert "previous_output" in prompt
        assert "Previous step result" in prompt

    def test_build_prompt_truncates_long_context(
        self, mock_model_server: ModelServer
    ):
        """Test that long context is truncated."""
        executor = Executor(model_server=mock_model_server)
        executor.context.set("long_output", "x" * 5000, step_id="S1")

        role_config = MagicMock(spec=RoleConfig)
        step = StepExecution(
            step_id="S2",
            actor="coder",
            action="Test",
            inputs=["long_output"],
            outputs=[],
            depends_on=[],
            role_config=role_config,
            command="",
        )

        prompt = executor._build_prompt(step)

        assert "truncated" in prompt

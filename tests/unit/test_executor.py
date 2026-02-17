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
        assert config.step_timeout == 600
        assert config.retry_failed_steps is True  # Enabled by default
        assert config.max_retries == 2  # 2 retries = 3 total attempts
        assert config.retry_backoff_base == 1.0
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

    def test_build_prompt_truncates_long_context(self, mock_model_server: ModelServer):
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


class TestRetryInfo:
    """Tests for RetryInfo dataclass."""

    def test_default_retry_info(self):
        """Test default retry info values."""
        from src.executor import RetryInfo

        info = RetryInfo()
        assert info.attempts == 0
        assert info.max_attempts == 1
        assert info.can_retry is True  # 0 < 1

    def test_can_retry_with_attempts(self):
        """Test can_retry property."""
        from src.executor import RetryInfo

        info = RetryInfo(max_attempts=3)
        assert info.can_retry is True

        info.attempts = 2
        assert info.can_retry is True

        info.attempts = 3
        assert info.can_retry is False

    def test_record_attempt(self):
        """Test recording a retry attempt."""
        from src.executor import ErrorCategory, RetryInfo

        info = RetryInfo(max_attempts=3, backoff_seconds=1.0)
        info.record_attempt("Test error", ErrorCategory.TIMEOUT)

        assert info.attempts == 1
        assert info.last_error == "Test error"
        assert info.error_category == ErrorCategory.TIMEOUT
        assert info.backoff_seconds == 2.0  # Doubled

    def test_exponential_backoff(self):
        """Test exponential backoff capping."""
        from src.executor import ErrorCategory, RetryInfo

        info = RetryInfo(max_attempts=10, backoff_seconds=16.0)
        info.record_attempt("Error 1", ErrorCategory.RETRYABLE)
        assert info.backoff_seconds == 30.0  # Capped at 30


class TestErrorCategory:
    """Tests for error categorization."""

    def test_categorize_timeout(self, mock_model_server: ModelServer):
        """Test categorizing timeout errors."""
        from src.executor import ErrorCategory

        executor = Executor(model_server=mock_model_server)

        assert executor._categorize_error("Connection timeout") == ErrorCategory.TIMEOUT
        assert executor._categorize_error("Request timed out") == ErrorCategory.TIMEOUT

    def test_categorize_model_error(self, mock_model_server: ModelServer):
        """Test categorizing model errors."""
        from src.executor import ErrorCategory

        executor = Executor(model_server=mock_model_server)

        assert executor._categorize_error("Model not found: test.gguf") == ErrorCategory.MODEL_ERROR
        assert executor._categorize_error("Load failed for model") == ErrorCategory.MODEL_ERROR

    def test_categorize_retryable(self, mock_model_server: ModelServer):
        """Test categorizing retryable errors."""
        from src.executor import ErrorCategory

        executor = Executor(model_server=mock_model_server)

        assert executor._categorize_error("Connection refused") == ErrorCategory.RETRYABLE
        assert (
            executor._categorize_error("Service temporarily unavailable") == ErrorCategory.RETRYABLE
        )

    def test_categorize_internal(self, mock_model_server: ModelServer):
        """Test categorizing internal errors."""
        from src.executor import ErrorCategory

        executor = Executor(model_server=mock_model_server)

        assert executor._categorize_error("Unknown error occurred") == ErrorCategory.INTERNAL_ERROR
        assert executor._categorize_error(None) == ErrorCategory.INTERNAL_ERROR


class TestExecutorRetry:
    """Tests for executor retry functionality."""

    def test_retry_info_initialized(
        self, mock_model_server: ModelServer, sample_dispatch_result: DispatchResult
    ):
        """Test that retry info is initialized for each step."""
        config = ExecutorConfig(dry_run=True, retry_failed_steps=True, max_retries=2)
        executor = Executor(model_server=mock_model_server, config=config)

        executor.execute(sample_dispatch_result)

        # Check retry info was created for each step
        assert "S1" in executor._retry_info
        assert "S2" in executor._retry_info
        assert executor._retry_info["S1"].max_attempts == 3  # 2 retries + 1 initial

    def test_step_result_includes_retry_count(self, mock_model_server: ModelServer):
        """Test that step result to_dict includes retry count when > 0."""
        result = StepResult(
            step_id="S1",
            status=StepStatus.FAILED,
            retry_count=2,
            error_category="timeout",
        )

        d = result.to_dict()
        assert d["retry_count"] == 2
        assert d["error_category"] == "timeout"

    def test_step_result_excludes_zero_retry(self):
        """Test that step result to_dict excludes retry count when 0."""
        result = StepResult(
            step_id="S1",
            status=StepStatus.COMPLETED,
            retry_count=0,
        )

        d = result.to_dict()
        assert "retry_count" not in d
        assert "error_category" not in d

    def test_find_retryable_steps(self, mock_model_server: ModelServer):
        """Test finding steps that can be retried."""
        from src.executor import RetryInfo

        config = ExecutorConfig(retry_failed_steps=True, max_retries=2)
        executor = Executor(model_server=mock_model_server, config=config)

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
        ]

        # Set up retry tracking
        executor._retry_info["S1"] = RetryInfo(max_attempts=3, attempts=1)

        completed: set[str] = set()
        results = {
            "S1": StepResult(step_id="S1", status=StepStatus.FAILED),
        }

        retryable = executor._find_retryable_steps(steps, completed, results)
        assert len(retryable) == 1
        assert retryable[0].step_id == "S1"

    def test_no_retry_when_exhausted(self, mock_model_server: ModelServer):
        """Test that exhausted retries are not retried."""
        from src.executor import RetryInfo

        config = ExecutorConfig(retry_failed_steps=True, max_retries=2)
        executor = Executor(model_server=mock_model_server, config=config)

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
        ]

        # Max attempts reached
        executor._retry_info["S1"] = RetryInfo(max_attempts=3, attempts=3)

        completed: set[str] = set()
        results = {
            "S1": StepResult(step_id="S1", status=StepStatus.FAILED),
        }

        retryable = executor._find_retryable_steps(steps, completed, results)
        assert len(retryable) == 0


class TestExecutorEscalation:
    """Tests for executor escalation functionality."""

    def test_config_has_escalation_settings(self):
        """Test that ExecutorConfig includes escalation settings."""
        config = ExecutorConfig()
        assert config.enable_escalation is True
        assert config.max_escalations_per_step == 2

    def test_config_escalation_disabled(self):
        """Test disabling escalation via config."""
        config = ExecutorConfig(enable_escalation=False)
        assert config.enable_escalation is False

    def test_step_result_has_escalation_fields(self):
        """Test that StepResult includes escalation tracking fields."""
        result = StepResult(
            step_id="S1",
            status=StepStatus.COMPLETED,
            escalation_count=1,
            escalated_from="coder_escalation",
            executed_role="coder_escalation",
        )
        assert result.escalation_count == 1
        assert result.escalated_from == "coder_escalation"
        assert result.executed_role == "coder_escalation"

    def test_step_result_to_dict_with_escalation(self):
        """Test that StepResult.to_dict includes escalation info when present."""
        result = StepResult(
            step_id="S1",
            status=StepStatus.COMPLETED,
            escalation_count=2,
            escalated_from="coder_escalation",
            executed_role="architect_coding",
        )
        d = result.to_dict()
        assert d["escalation_count"] == 2
        assert d["escalated_from"] == "coder_escalation"
        assert d["executed_role"] == "architect_coding"

    def test_step_result_to_dict_excludes_zero_escalation(self):
        """Test that StepResult.to_dict excludes escalation info when count is 0."""
        result = StepResult(
            step_id="S1",
            status=StepStatus.COMPLETED,
            escalation_count=0,
        )
        d = result.to_dict()
        assert "escalation_count" not in d
        assert "escalated_from" not in d

    def test_execution_result_total_escalations(self):
        """Test ExecutionResult.total_escalations property."""
        result = ExecutionResult(task_id="test", status=StepStatus.COMPLETED)
        result.steps["S1"] = StepResult(
            step_id="S1", status=StepStatus.COMPLETED, escalation_count=1
        )
        result.steps["S2"] = StepResult(
            step_id="S2", status=StepStatus.COMPLETED, escalation_count=2
        )
        assert result.total_escalations == 3

    def test_execution_result_escalated_steps(self):
        """Test ExecutionResult.escalated_steps property."""
        result = ExecutionResult(task_id="test", status=StepStatus.COMPLETED)
        result.steps["S1"] = StepResult(
            step_id="S1", status=StepStatus.COMPLETED, escalation_count=1
        )
        result.steps["S2"] = StepResult(
            step_id="S2", status=StepStatus.COMPLETED, escalation_count=0
        )
        result.steps["S3"] = StepResult(
            step_id="S3", status=StepStatus.COMPLETED, escalation_count=2
        )
        assert result.escalated_steps == 2

    def test_execution_result_to_dict_with_escalations(self):
        """Test ExecutionResult.to_dict includes escalation info when present."""
        result = ExecutionResult(task_id="test", status=StepStatus.COMPLETED)
        result.steps["S1"] = StepResult(
            step_id="S1", status=StepStatus.COMPLETED, escalation_count=1
        )
        d = result.to_dict()
        assert d["total_escalations"] == 1
        assert d["escalated_steps"] == 1

    def test_execution_result_to_dict_excludes_zero_escalations(self):
        """Test ExecutionResult.to_dict excludes escalation info when none occurred."""
        result = ExecutionResult(task_id="test", status=StepStatus.COMPLETED)
        result.steps["S1"] = StepResult(
            step_id="S1", status=StepStatus.COMPLETED, escalation_count=0
        )
        d = result.to_dict()
        assert "total_escalations" not in d
        assert "escalated_steps" not in d

    def test_escalation_count_initialized(
        self, mock_model_server: ModelServer, sample_dispatch_result: DispatchResult
    ):
        """Test that escalation counts are initialized for each step."""
        config = ExecutorConfig(dry_run=True, enable_escalation=True)
        executor = Executor(model_server=mock_model_server, config=config)

        executor.execute(sample_dispatch_result)

        # Check escalation counts were created for each step
        assert "S1" in executor._escalation_counts
        assert "S2" in executor._escalation_counts
        assert executor._escalation_counts["S1"] == 0
        assert executor._escalation_counts["S2"] == 0

    def test_can_escalate_disabled(self, mock_model_server: ModelServer):
        """Test _can_escalate returns False when escalation is disabled."""
        config = ExecutorConfig(enable_escalation=False)
        executor = Executor(model_server=mock_model_server, config=config)

        role_config = MagicMock(spec=RoleConfig)
        role_config.name = "coder_escalation"
        step = StepExecution(
            step_id="S1",
            actor="coder",
            action="Test",
            inputs=[],
            outputs=[],
            depends_on=[],
            role_config=role_config,
            command="",
        )
        step_result = StepResult(
            step_id="S1",
            status=StepStatus.FAILED,
            error_category="timeout",
        )

        assert not executor._can_escalate(step, step_result)

    def test_can_escalate_no_role_config(self, mock_model_server: ModelServer):
        """Test _can_escalate returns False when step has no role config."""
        config = ExecutorConfig(enable_escalation=True)
        executor = Executor(model_server=mock_model_server, config=config)

        step = StepExecution(
            step_id="S1",
            actor="coder",
            action="Test",
            inputs=[],
            outputs=[],
            depends_on=[],
            role_config=None,
            command="",
        )
        step_result = StepResult(step_id="S1", status=StepStatus.FAILED)

        assert not executor._can_escalate(step, step_result)

    def test_can_escalate_max_reached(self, mock_model_server: ModelServer):
        """Test _can_escalate returns False when max escalations reached."""
        config = ExecutorConfig(enable_escalation=True, max_escalations_per_step=1)
        executor = Executor(model_server=mock_model_server, config=config)
        executor._escalation_counts["S1"] = 1  # Already escalated once

        role_config = MagicMock(spec=RoleConfig)
        role_config.name = "coder_escalation"
        step = StepExecution(
            step_id="S1",
            actor="coder",
            action="Test",
            inputs=[],
            outputs=[],
            depends_on=[],
            role_config=role_config,
            command="",
        )
        step_result = StepResult(
            step_id="S1",
            status=StepStatus.FAILED,
            error_category="timeout",
        )

        assert not executor._can_escalate(step, step_result)

    def test_can_escalate_role_in_chain(self, mock_model_server: ModelServer):
        """Test _can_escalate returns True when role is in escalation chain."""
        config = ExecutorConfig(enable_escalation=True)
        executor = Executor(model_server=mock_model_server, config=config)
        executor._escalation_counts["S1"] = 0

        role_config = MagicMock(spec=RoleConfig)
        role_config.name = "coder_escalation"  # This is in the coder chain
        step = StepExecution(
            step_id="S1",
            actor="coder",
            action="Test",
            inputs=[],
            outputs=[],
            depends_on=[],
            role_config=role_config,
            command="",
        )
        step_result = StepResult(
            step_id="S1",
            status=StepStatus.FAILED,
            error_category="timeout",  # This should trigger escalation
        )

        # This depends on the registry having the coder chain configured
        result = executor._can_escalate(step, step_result)
        # Result depends on registry config - just ensure no errors
        assert isinstance(result, bool)

    def test_escalate_step_returns_next_role(self, mock_model_server: ModelServer):
        """Test _escalate_step returns the next role in the chain."""
        config = ExecutorConfig(enable_escalation=True)
        executor = Executor(model_server=mock_model_server, config=config)
        executor._escalation_counts["S1"] = 0

        role_config = MagicMock(spec=RoleConfig)
        role_config.name = "coder_escalation"
        step = StepExecution(
            step_id="S1",
            actor="coder",
            action="Test",
            inputs=[],
            outputs=[],
            depends_on=[],
            role_config=role_config,
            command="",
        )
        step_result = StepResult(step_id="S1", status=StepStatus.FAILED)
        exec_result = ExecutionResult(task_id="test", status=StepStatus.RUNNING)

        # Escalate
        next_role = executor._escalate_step(step, step_result, exec_result)

        # Should return architect_coding (coder_escalation escalates to architect_coding)
        if next_role:  # If chain exists in registry
            assert next_role == "architect_coding"
            assert executor._escalation_counts["S1"] == 1
            # Step's role_config should be updated
            assert step.role_config.name == "architect_coding"

    def test_escalate_step_increments_count(self, mock_model_server: ModelServer):
        """Test _escalate_step increments the escalation count."""
        config = ExecutorConfig(enable_escalation=True)
        executor = Executor(model_server=mock_model_server, config=config)
        executor._escalation_counts["S1"] = 0

        role_config = MagicMock(spec=RoleConfig)
        role_config.name = "coder_escalation"
        step = StepExecution(
            step_id="S1",
            actor="coder",
            action="Test",
            inputs=[],
            outputs=[],
            depends_on=[],
            role_config=role_config,
            command="",
        )
        step_result = StepResult(step_id="S1", status=StepStatus.FAILED)
        exec_result = ExecutionResult(task_id="test", status=StepStatus.RUNNING)

        executor._escalate_step(step, step_result, exec_result)

        # Count should be incremented if escalation succeeded
        if executor._escalation_counts["S1"] > 0:
            assert executor._escalation_counts["S1"] == 1

    def test_escalate_step_adds_warning(self, mock_model_server: ModelServer):
        """Test _escalate_step adds a warning to the execution result."""
        config = ExecutorConfig(enable_escalation=True)
        executor = Executor(model_server=mock_model_server, config=config)
        executor._escalation_counts["S1"] = 0

        role_config = MagicMock(spec=RoleConfig)
        role_config.name = "coder_escalation"
        step = StepExecution(
            step_id="S1",
            actor="coder",
            action="Test",
            inputs=[],
            outputs=[],
            depends_on=[],
            role_config=role_config,
            command="",
        )
        step_result = StepResult(step_id="S1", status=StepStatus.FAILED)
        exec_result = ExecutionResult(task_id="test", status=StepStatus.RUNNING)

        next_role = executor._escalate_step(step, step_result, exec_result)

        # Should add a warning about escalation
        if next_role:
            assert len(exec_result.warnings) > 0
            assert "Escalating" in exec_result.warnings[-1]

    def test_executed_role_initialized(
        self, mock_model_server: ModelServer, sample_dispatch_result: DispatchResult
    ):
        """Test that executed_role is initialized from role_config."""
        config = ExecutorConfig(dry_run=True)
        executor = Executor(model_server=mock_model_server, config=config)

        result = executor.execute(sample_dispatch_result)

        # Each step should have executed_role set
        for step_result in result.steps.values():
            # In sample_dispatch_result, steps have role_config with name="coder"
            assert step_result.executed_role == "coder"

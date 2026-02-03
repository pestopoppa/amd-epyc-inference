#!/usr/bin/env python3
"""Executor for hierarchical local-agent orchestration.

This module executes step plans from the Dispatcher, managing dependencies,
parallel execution, and result collection.

Usage:
    from src.executor import Executor
    from src.dispatcher import Dispatcher

    dispatcher = Dispatcher()
    result = dispatcher.dispatch(task_ir)

    executor = Executor()
    execution = executor.execute(result)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.context_manager import ContextConfig, ContextManager, ContextType
from src.dispatcher import DispatchResult, StepExecution
from src.model_server import (
    InferenceRequest,
    InferenceResult,
    ModelServer,
)
from src.registry_loader import RegistryLoader


class StepStatus(Enum):
    """Status of a step execution."""

    PENDING = "pending"
    WAITING = "waiting"  # Waiting for dependencies
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of executing a single step."""

    step_id: str
    status: StepStatus
    output: str = ""
    artifacts: list[str] = field(default_factory=list)
    inference_result: InferenceResult | None = None
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None
    retry_count: int = 0
    error_category: str | None = None
    escalation_count: int = 0
    escalated_from: str | None = None  # Role that was escalated from
    executed_role: str | None = None  # Role that actually executed the step

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "step_id": self.step_id,
            "status": self.status.value,
            "output": self.output[:500] if self.output else "",  # Truncate for summary
            "artifacts": self.artifacts,
            "elapsed_time": self.elapsed_time,
            "error_message": self.error_message,
        }
        if self.retry_count > 0:
            result["retry_count"] = self.retry_count
        if self.error_category:
            result["error_category"] = self.error_category
        if self.escalation_count > 0:
            result["escalation_count"] = self.escalation_count
            result["escalated_from"] = self.escalated_from
        if self.executed_role:
            result["executed_role"] = self.executed_role
        return result


@dataclass
class ExecutionResult:
    """Result of executing a full dispatch plan."""

    task_id: str
    status: StepStatus  # Overall status
    steps: dict[str, StepResult] = field(default_factory=dict)
    started_at: float | None = None
    completed_at: float | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def elapsed_time(self) -> float:
        """Calculate total elapsed time in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def successful_steps(self) -> int:
        """Count of successful steps."""
        return sum(1 for s in self.steps.values() if s.status == StepStatus.COMPLETED)

    @property
    def failed_steps(self) -> int:
        """Count of failed steps."""
        return sum(1 for s in self.steps.values() if s.status == StepStatus.FAILED)

    @property
    def total_escalations(self) -> int:
        """Count of total escalations across all steps."""
        return sum(s.escalation_count for s in self.steps.values())

    @property
    def escalated_steps(self) -> int:
        """Count of steps that were escalated at least once."""
        return sum(1 for s in self.steps.values() if s.escalation_count > 0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "task_id": self.task_id,
            "status": self.status.value,
            "elapsed_time": self.elapsed_time,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "warnings": self.warnings,
            "errors": self.errors,
        }
        # Only include escalation info if escalations occurred
        if self.total_escalations > 0:
            result["total_escalations"] = self.total_escalations
            result["escalated_steps"] = self.escalated_steps
        return result


class ExecutorError(Exception):
    """Error during execution."""

    pass


class ErrorCategory(Enum):
    """Categories of execution errors."""

    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"
    INFERENCE_ERROR = "inference_error"
    DEPENDENCY_ERROR = "dependency_error"
    INTERNAL_ERROR = "internal_error"
    RETRYABLE = "retryable"


@dataclass
class RetryInfo:
    """Information about retry attempts for a step."""

    attempts: int = 0
    max_attempts: int = 1
    last_error: str | None = None
    error_category: ErrorCategory | None = None
    backoff_seconds: float = 1.0

    @property
    def can_retry(self) -> bool:
        """Check if retry is possible."""
        return self.attempts < self.max_attempts

    def record_attempt(self, error: str, category: ErrorCategory) -> None:
        """Record a failed attempt."""
        self.attempts += 1
        self.last_error = error
        self.error_category = category
        # Exponential backoff with jitter
        self.backoff_seconds = min(30.0, self.backoff_seconds * 2)


def _exec_defaults():
    from src.config import get_config

    return get_config()


@dataclass
class ExecutorConfig:
    """Configuration for the Executor."""

    max_parallel_workers: int = 2
    step_timeout: int = field(default_factory=lambda: _exec_defaults().server.timeout)
    retry_failed_steps: bool = True  # Enable by default
    max_retries: int = field(default_factory=lambda: _exec_defaults().escalation.max_retries)
    retry_backoff_base: float = 1.0  # Base backoff in seconds
    dry_run: bool = False  # If True, don't actually run inference
    # Escalation settings
    enable_escalation: bool = True  # Escalate to more capable models on failure
    max_escalations_per_step: int = field(
        default_factory=lambda: _exec_defaults().escalation.max_escalations
    )


class Executor:
    """Execute step plans from the Dispatcher.

    The Executor takes a DispatchResult and runs each step in order,
    respecting dependencies and parallel execution groups.

    Uses ContextManager to track outputs between steps with:
    - Size limits and truncation
    - Type tracking (text, artifact, structured)
    - Prompt building for dependent steps
    """

    def __init__(
        self,
        model_server: ModelServer | None = None,
        config: ExecutorConfig | None = None,
        context_config: ContextConfig | None = None,
        registry: RegistryLoader | None = None,
    ):
        """Initialize the Executor.

        Args:
            model_server: Model server for inference. Created if None.
            config: Executor configuration.
            context_config: Context manager configuration.
            registry: Registry loader for escalation chains. Created if None.
        """
        self.server = model_server or ModelServer()
        self.config = config or ExecutorConfig()
        self.context = ContextManager(context_config)  # Rich context management
        # For escalation lookups - don't validate paths, allow missing in tests/CI
        self.registry = registry or RegistryLoader(validate_paths=False, allow_missing=True)
        self._retry_info: dict[str, RetryInfo] = {}  # Track retries per step
        self._escalation_counts: dict[str, int] = {}  # Track escalations per step

    def execute(self, dispatch_result: DispatchResult) -> ExecutionResult:
        """Execute a dispatch plan.

        Args:
            dispatch_result: Result from Dispatcher.dispatch()

        Returns:
            ExecutionResult with step results and overall status.
        """
        result = ExecutionResult(
            task_id=dispatch_result.task_id,
            status=StepStatus.RUNNING,
            started_at=time.time(),
        )

        # Initialize step results, retry info, and escalation tracking
        self._retry_info.clear()
        self._escalation_counts.clear()
        max_attempts = self.config.max_retries + 1 if self.config.retry_failed_steps else 1
        for step in dispatch_result.steps:
            result.steps[step.step_id] = StepResult(
                step_id=step.step_id,
                status=StepStatus.PENDING,
                executed_role=step.role_config.name if step.role_config else None,
            )
            self._retry_info[step.step_id] = RetryInfo(
                max_attempts=max_attempts,
                backoff_seconds=self.config.retry_backoff_base,
            )
            self._escalation_counts[step.step_id] = 0

        # Copy warnings/errors from dispatch
        result.warnings.extend(dispatch_result.warnings)
        result.errors.extend(dispatch_result.errors)

        # Track completed steps (success or exhausted retries)
        completed: set[str] = set()

        try:
            # Execute steps respecting dependencies
            while len(completed) < len(dispatch_result.steps):
                # Find steps ready to execute (including failed steps that can retry)
                ready_steps = self._find_ready_steps(dispatch_result.steps, completed, result.steps)

                # Also check for retryable failed steps
                retryable_steps = self._find_retryable_steps(
                    dispatch_result.steps, completed, result.steps
                )
                ready_steps.extend(retryable_steps)

                if not ready_steps:
                    # Check if we're stuck due to failed dependencies
                    pending = [s for s in dispatch_result.steps if s.step_id not in completed]
                    if pending:
                        for step in pending:
                            step_result = result.steps[step.step_id]
                            # Only mark as skipped if not already failed
                            if step_result.status != StepStatus.FAILED:
                                step_result.status = StepStatus.SKIPPED
                                step_result.error_message = "Skipped due to failed dependencies"
                                step_result.error_category = ErrorCategory.DEPENDENCY_ERROR.value
                            completed.add(step.step_id)
                    break

                # Group by parallel_group for concurrent execution
                parallel_groups = self._group_by_parallel(ready_steps)

                for group_steps in parallel_groups.values():
                    if len(group_steps) > 1 and self.config.max_parallel_workers > 1:
                        # Execute in parallel
                        self._execute_parallel(group_steps, result)
                    else:
                        # Execute sequentially
                        for step in group_steps:
                            self._execute_step_with_retry(step, result)

                    # Mark completed steps (success, skipped, or exhausted retries)
                    for step in group_steps:
                        step_result = result.steps[step.step_id]
                        retry_info = self._retry_info[step.step_id]
                        is_done = step_result.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
                        is_exhausted = (
                            step_result.status == StepStatus.FAILED and not retry_info.can_retry
                        )
                        if is_done or is_exhausted:
                            completed.add(step.step_id)
                        # If failed but can retry, don't mark as completed

        except Exception as e:
            result.errors.append(f"Execution error: {e}")
            result.status = StepStatus.FAILED

        # Determine overall status
        result.completed_at = time.time()
        if result.failed_steps > 0:
            result.status = StepStatus.FAILED
        elif result.successful_steps == len(dispatch_result.steps):
            result.status = StepStatus.COMPLETED
        else:
            result.status = StepStatus.COMPLETED  # Some skipped is OK

        return result

    def _find_ready_steps(
        self,
        steps: list[StepExecution],
        completed: set[str],
        results: dict[str, StepResult],
    ) -> list[StepExecution]:
        """Find steps whose dependencies are satisfied."""
        ready = []
        for step in steps:
            if step.step_id in completed:
                continue

            # Check if all dependencies are completed successfully
            deps_satisfied = True
            for dep_id in step.depends_on:
                if dep_id not in completed:
                    deps_satisfied = False
                    break
                # If dependency failed or was skipped, this step can't proceed
                if results[dep_id].status in (StepStatus.FAILED, StepStatus.SKIPPED):
                    deps_satisfied = False
                    break

            if deps_satisfied:
                ready.append(step)

        return ready

    def _find_retryable_steps(
        self,
        steps: list[StepExecution],
        completed: set[str],
        results: dict[str, StepResult],
    ) -> list[StepExecution]:
        """Find failed steps that can be retried."""
        retryable = []
        for step in steps:
            if step.step_id in completed:
                continue

            step_result = results.get(step.step_id)
            retry_info = self._retry_info.get(step.step_id)

            # Check if step failed and can retry
            if (
                step_result
                and step_result.status == StepStatus.FAILED
                and retry_info
                and retry_info.can_retry
            ):
                # Verify dependencies still satisfied
                deps_ok = all(
                    results.get(dep_id, StepResult(dep_id, StepStatus.PENDING)).status
                    == StepStatus.COMPLETED
                    for dep_id in step.depends_on
                )
                if deps_ok:
                    retryable.append(step)

        return retryable

    def _execute_step_with_retry(self, step: StepExecution, result: ExecutionResult) -> StepResult:
        """Execute a step with retry and escalation logic.

        Order of operations:
        1. Retry on failure (up to max_retries)
        2. Escalate to more capable model (if retries exhausted and enabled)
        3. Retry with escalated model

        Args:
            step: Step to execute.
            result: Overall execution result to update.

        Returns:
            StepResult for this step.
        """
        retry_info = self._retry_info.get(step.step_id)
        step_result = result.steps[step.step_id]
        original_role = step.role_config.name if step.role_config else None

        # Apply backoff if this is a retry
        if retry_info and retry_info.attempts > 0:
            backoff = retry_info.backoff_seconds
            result.warnings.append(
                f"Retrying step {step.step_id} after {backoff:.1f}s backoff "
                f"(attempt {retry_info.attempts + 1}/{retry_info.max_attempts})"
            )
            time.sleep(backoff)

        # Reset step status for retry
        if step_result.status == StepStatus.FAILED:
            step_result.status = StepStatus.PENDING
            step_result.error_message = None

        # Execute the step
        self._execute_step(step, result)

        # Update retry tracking on failure
        if step_result.status == StepStatus.FAILED and retry_info:
            error_category = self._categorize_error(step_result.error_message)
            retry_info.record_attempt(
                step_result.error_message or "Unknown error",
                error_category,
            )
            step_result.retry_count = retry_info.attempts
            step_result.error_category = error_category.value

            # Check if escalation is possible after exhausting retries
            if not retry_info.can_retry and self._can_escalate(step, step_result):
                escalated_role = self._escalate_step(step, step_result, result)
                if escalated_role:
                    # Track escalation
                    step_result.escalation_count = self._escalation_counts[step.step_id]
                    step_result.escalated_from = step_result.escalated_from or original_role
                    step_result.executed_role = escalated_role

                    # Reset retry info for the escalated model
                    max_attempts = (
                        self.config.max_retries + 1 if self.config.retry_failed_steps else 1
                    )
                    self._retry_info[step.step_id] = RetryInfo(
                        max_attempts=max_attempts,
                        backoff_seconds=self.config.retry_backoff_base,
                    )

                    # Re-execute with escalated model (recursive)
                    return self._execute_step_with_retry(step, result)

        return step_result

    def _can_escalate(self, step: StepExecution, step_result: StepResult) -> bool:
        """Check if a step can be escalated.

        Args:
            step: The step to check.
            step_result: Current result for the step.

        Returns:
            True if escalation is possible.
        """
        if not self.config.enable_escalation:
            return False

        if not step.role_config:
            return False

        current_role = step.role_config.name
        escalation_count = self._escalation_counts.get(step.step_id, 0)

        # Check max escalations per step
        if escalation_count >= self.config.max_escalations_per_step:
            return False

        # Check if escalation chain exists and has a next role
        chain = self.registry.get_chain_for_role(current_role)
        if not chain:
            return False

        # Check chain-specific max escalations
        if escalation_count >= chain.max_escalations:
            return False

        # Check if there's a next role in the chain
        next_role = chain.get_next_role(current_role)
        if not next_role:
            return False

        # Check if the error category triggers escalation
        if step_result.error_category:
            for trigger in chain.triggers:
                if (
                    "error_categories" in trigger
                    and step_result.error_category in trigger["error_categories"]
                ):
                    return True
        elif not chain.triggers:
            # Default: escalate on any failure if no triggers specified
            return True

        return False

    def _escalate_step(
        self, step: StepExecution, step_result: StepResult, result: ExecutionResult
    ) -> str | None:
        """Escalate a step to a more capable model.

        Args:
            step: The step to escalate.
            step_result: Current result for the step.
            result: Overall execution result for logging.

        Returns:
            The new role name if escalation succeeded, None otherwise.
        """
        if not step.role_config:
            return None

        current_role = step.role_config.name
        next_role = self.registry.get_escalation_target(current_role)

        if not next_role:
            return None

        # Get the role config for the escalated role
        next_role_config = self.registry.get_role(next_role)
        if not next_role_config:
            result.warnings.append(f"Escalation target '{next_role}' not found in registry")
            return None

        # Update the step's role config
        step.role_config = next_role_config

        # Increment escalation count
        self._escalation_counts[step.step_id] += 1

        result.warnings.append(
            f"Escalating step {step.step_id} from '{current_role}' to '{next_role}' "
            f"(escalation {self._escalation_counts[step.step_id]}/{self.config.max_escalations_per_step})"
        )

        return next_role

    def _categorize_error(self, error_message: str | None) -> ErrorCategory:
        """Categorize an error to determine if it's retryable.

        Args:
            error_message: The error message to categorize.

        Returns:
            ErrorCategory for the error.
        """
        if not error_message:
            return ErrorCategory.INTERNAL_ERROR

        msg_lower = error_message.lower()

        # Timeout errors
        if any(word in msg_lower for word in ["timeout", "timed out", "deadline"]):
            return ErrorCategory.TIMEOUT

        # Model errors (usually not retryable)
        if any(word in msg_lower for word in ["model not found", "invalid model", "load failed"]):
            return ErrorCategory.MODEL_ERROR

        # Inference errors (may be retryable)
        if any(word in msg_lower for word in ["inference", "generation", "output"]):
            return ErrorCategory.INFERENCE_ERROR

        # Generic retryable errors
        if any(
            word in msg_lower
            for word in [
                "connection",
                "network",
                "temporary",
                "unavailable",
                "busy",
                "overload",
                "retry",
            ]
        ):
            return ErrorCategory.RETRYABLE

        return ErrorCategory.INTERNAL_ERROR

    def _group_by_parallel(
        self, steps: list[StepExecution]
    ) -> dict[str | None, list[StepExecution]]:
        """Group steps by their parallel_group."""
        groups: dict[str | None, list[StepExecution]] = {}
        for step in steps:
            group = step.parallel_group
            if group not in groups:
                groups[group] = []
            groups[group].append(step)
        return groups

    def _execute_step(self, step: StepExecution, result: ExecutionResult) -> StepResult:
        """Execute a single step.

        Args:
            step: Step to execute.
            result: Overall execution result to update.

        Returns:
            StepResult for this step.
        """
        step_result = result.steps[step.step_id]
        step_result.status = StepStatus.RUNNING
        step_result.started_at = time.time()

        try:
            if not step.role_config:
                # No model assigned - skip
                step_result.status = StepStatus.SKIPPED
                step_result.error_message = "No model assigned to step"
            elif self.config.dry_run:
                # Dry run - just simulate
                step_result.output = f"[DRY RUN] Would execute: {step.action}"
                step_result.status = StepStatus.COMPLETED

                # Store outputs in context for dependent steps
                for output in step.outputs:
                    self.context.set(
                        key=output,
                        value=step_result.output,
                        step_id=step.step_id,
                        context_type=ContextType.TEXT,
                    )
            else:
                # Build prompt from step context
                prompt = self._build_prompt(step)

                # Run inference
                request = InferenceRequest(
                    role=step.role_config.name,
                    prompt=prompt,
                    n_tokens=512,
                    timeout=self.config.step_timeout,
                )

                inference_result = self.server.infer(request)
                step_result.inference_result = inference_result

                if inference_result.success:
                    step_result.output = inference_result.output
                    step_result.status = StepStatus.COMPLETED

                    # Store outputs in context for dependent steps
                    for output in step.outputs:
                        self.context.set(
                            key=output,
                            value=inference_result.output,
                            step_id=step.step_id,
                            context_type=ContextType.TEXT,
                            metadata={
                                "tokens": inference_result.tokens_generated,
                                "speed": inference_result.generation_speed,
                            },
                        )
                else:
                    step_result.status = StepStatus.FAILED
                    step_result.error_message = inference_result.error_message

        except Exception as e:
            step_result.status = StepStatus.FAILED
            step_result.error_message = str(e)
            result.errors.append(f"Step {step.step_id} failed: {e}")

        step_result.completed_at = time.time()
        return step_result

    def _execute_parallel(self, steps: list[StepExecution], result: ExecutionResult) -> None:
        """Execute multiple steps in parallel.

        Args:
            steps: Steps to execute concurrently.
            result: Overall execution result to update.
        """
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as pool:
            futures = {
                pool.submit(self._execute_step_with_retry, step, result): step for step in steps
            }

            for future in as_completed(futures):
                step = futures[future]
                try:
                    future.result()
                except Exception as e:
                    step_result = result.steps[step.step_id]
                    step_result.status = StepStatus.FAILED
                    step_result.error_message = str(e)
                    step_result.error_category = ErrorCategory.INTERNAL_ERROR.value
                    result.errors.append(f"Parallel step {step.step_id} failed: {e}")

    def _build_prompt(self, step: StepExecution) -> str:
        """Build a prompt for a step from its action and inputs.

        Uses ContextManager.build_prompt_context() for rich context formatting
        with size limits and proper handling of different content types.

        Args:
            step: Step to build prompt for.

        Returns:
            Prompt string for inference.
        """
        parts = [f"Task: {step.action}"]

        # Add inputs from context using ContextManager
        if step.inputs:
            context_str = self.context.build_prompt_context(
                input_keys=step.inputs,
                max_chars=4000,  # Leave room for task and outputs
            )
            if context_str:
                parts.append(f"\nInputs:{context_str}")
            else:
                # List unavailable inputs
                missing = [k for k in step.inputs if not self.context.has(k)]
                if missing:
                    parts.append(f"\nInputs (not available): {', '.join(missing)}")

        # Add expected outputs
        if step.outputs:
            parts.append(f"\nExpected outputs: {', '.join(step.outputs)}")

        return "\n".join(parts)

    def get_context(self) -> ContextManager:
        """Get the context manager.

        Returns:
            The ContextManager instance for direct access.
        """
        return self.context

    def get_context_dict(self) -> dict[str, Any]:
        """Get context as a simple dictionary.

        Returns:
            Dictionary of key-value pairs.
        """
        return dict(self.context.items())

    def clear_context(self) -> None:
        """Clear the execution context."""
        self.context.clear()


def main() -> int:
    """CLI entry point for testing."""
    import json

    from src.dispatcher import Dispatcher

    # Create a sample task
    task_ir = {
        "task_id": "test-execution",
        "task_type": "code",
        "priority": "interactive",
        "objective": "Test the executor",
        "agents": [{"role": "coder"}],
        "plan": {
            "steps": [
                {
                    "id": "S1",
                    "actor": "coder",
                    "action": "Write a hello world function",
                    "inputs": [],
                    "outputs": ["hello.py"],
                    "depends_on": [],
                },
                {
                    "id": "S2",
                    "actor": "coder",
                    "action": "Write tests for hello world",
                    "inputs": ["hello.py"],
                    "outputs": ["test_hello.py"],
                    "depends_on": ["S1"],
                },
            ],
        },
    }

    # Dispatch
    dispatcher = Dispatcher(validate_paths=False)
    dispatch_result = dispatcher.dispatch(task_ir)

    print("Dispatch Result:")
    print(json.dumps(dispatch_result.to_dict(), indent=2))

    # Execute (dry run)
    config = ExecutorConfig(dry_run=True)
    executor = Executor(config=config)
    execution_result = executor.execute(dispatch_result)

    print("\nExecution Result:")
    print(json.dumps(execution_result.to_dict(), indent=2))

    return 0 if execution_result.status == StepStatus.COMPLETED else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

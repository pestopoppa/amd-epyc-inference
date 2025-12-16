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

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "output": self.output[:500] if self.output else "",  # Truncate for summary
            "artifacts": self.artifacts,
            "elapsed_time": self.elapsed_time,
            "error_message": self.error_message,
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "elapsed_time": self.elapsed_time,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "warnings": self.warnings,
            "errors": self.errors,
        }


class ExecutorError(Exception):
    """Error during execution."""

    pass


@dataclass
class ExecutorConfig:
    """Configuration for the Executor."""

    max_parallel_workers: int = 2
    step_timeout: int = 300  # seconds
    retry_failed_steps: bool = False
    max_retries: int = 1
    dry_run: bool = False  # If True, don't actually run inference


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
    ):
        """Initialize the Executor.

        Args:
            model_server: Model server for inference. Created if None.
            config: Executor configuration.
            context_config: Context manager configuration.
        """
        self.server = model_server or ModelServer()
        self.config = config or ExecutorConfig()
        self.context = ContextManager(context_config)  # Rich context management

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

        # Initialize step results
        for step in dispatch_result.steps:
            result.steps[step.step_id] = StepResult(
                step_id=step.step_id,
                status=StepStatus.PENDING,
            )

        # Copy warnings/errors from dispatch
        result.warnings.extend(dispatch_result.warnings)
        result.errors.extend(dispatch_result.errors)

        # Track completed steps
        completed: set[str] = set()

        try:
            # Execute steps respecting dependencies
            while len(completed) < len(dispatch_result.steps):
                # Find steps ready to execute
                ready_steps = self._find_ready_steps(
                    dispatch_result.steps, completed, result.steps
                )

                if not ready_steps:
                    # Check if we're stuck due to failed dependencies
                    pending = [
                        s for s in dispatch_result.steps
                        if s.step_id not in completed
                    ]
                    if pending:
                        for step in pending:
                            result.steps[step.step_id].status = StepStatus.SKIPPED
                            result.steps[step.step_id].error_message = (
                                "Skipped due to failed dependencies"
                            )
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
                            self._execute_step(step, result)

                    # Mark as completed
                    for step in group_steps:
                        completed.add(step.step_id)

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

    def _execute_step(
        self, step: StepExecution, result: ExecutionResult
    ) -> StepResult:
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

    def _execute_parallel(
        self, steps: list[StepExecution], result: ExecutionResult
    ) -> None:
        """Execute multiple steps in parallel.

        Args:
            steps: Steps to execute concurrently.
            result: Overall execution result to update.
        """
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as pool:
            futures = {
                pool.submit(self._execute_step, step, result): step
                for step in steps
            }

            for future in as_completed(futures):
                step = futures[future]
                try:
                    future.result()
                except Exception as e:
                    result.steps[step.step_id].status = StepStatus.FAILED
                    result.steps[step.step_id].error_message = str(e)
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

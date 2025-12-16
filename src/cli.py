#!/usr/bin/env python3
"""CLI entry point for hierarchical local-agent orchestration.

This module provides a command-line interface for running the full
orchestration pipeline: Task → TaskIR → Dispatch → Execute → Result.

Usage:
    python -m src.cli "Write a Python function to sort a list"
    python -m src.cli --dry-run "Refactor the code in main.py"
    python -m src.cli --verbose "Analyze this algorithm"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dispatcher import Dispatcher
from src.executor import ExecutionResult, Executor, ExecutorConfig, StepStatus
from src.model_server import InferenceRequest, ModelServer
from src.registry_loader import RegistryLoader


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    dry_run: bool = False
    verbose: bool = False
    timeout: int = 300
    max_tokens: int = 512
    output_file: Path | None = None
    skip_frontdoor: bool = False  # Use provided TaskIR instead of generating


class Orchestrator:
    """Main orchestration pipeline.

    Coordinates the full flow:
    1. Front Door: Generate TaskIR from user prompt
    2. Dispatcher: Route TaskIR to models and create execution plan
    3. Executor: Run each step with actual inference
    4. Result: Collect and return outputs
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        """Initialize the orchestrator.

        Args:
            config: Orchestrator configuration.
        """
        self.config = config or OrchestratorConfig()
        self.registry = RegistryLoader(validate_paths=True)
        self.server = ModelServer(registry=self.registry)
        self.dispatcher = Dispatcher(registry=self.registry, validate_paths=False)
        self.executor = Executor(
            model_server=self.server,
            config=ExecutorConfig(
                dry_run=self.config.dry_run,
                step_timeout=self.config.timeout,
            ),
        )

    def run(self, user_prompt: str) -> dict[str, Any]:
        """Run the full orchestration pipeline.

        Args:
            user_prompt: User's task description.

        Returns:
            Dictionary with results from each stage.
        """
        results: dict[str, Any] = {
            "input": user_prompt,
            "started_at": time.time(),
            "stages": {},
        }

        try:
            # Stage 1: Front Door - Generate TaskIR
            if self.config.verbose:
                print("\n[1/3] Front Door: Generating TaskIR...")

            task_ir = self._generate_task_ir(user_prompt)
            results["stages"]["task_ir"] = task_ir

            if self.config.verbose:
                print(f"  Task ID: {task_ir.get('task_id', 'N/A')}")
                print(f"  Type: {task_ir.get('task_type', 'N/A')}")
                print(f"  Steps: {len(task_ir.get('plan', {}).get('steps', []))}")

            # Stage 2: Dispatcher - Create execution plan
            if self.config.verbose:
                print("\n[2/3] Dispatcher: Creating execution plan...")

            dispatch_result = self.dispatcher.dispatch(task_ir)
            results["stages"]["dispatch"] = dispatch_result.to_dict()

            if self.config.verbose:
                print(f"  Roles: {dispatch_result.roles_used}")
                print(f"  Steps: {len(dispatch_result.steps)}")
                if dispatch_result.warnings:
                    print(f"  Warnings: {dispatch_result.warnings}")

            # Stage 3: Executor - Run steps
            if self.config.verbose:
                mode = "DRY RUN" if self.config.dry_run else "LIVE"
                print(f"\n[3/3] Executor: Running steps ({mode})...")

            execution_result = self.executor.execute(dispatch_result)
            results["stages"]["execution"] = execution_result.to_dict()

            if self.config.verbose:
                print(f"  Status: {execution_result.status.value}")
                print(
                    f"  Completed: {execution_result.successful_steps}/{len(dispatch_result.steps)}"
                )
                print(f"  Time: {execution_result.elapsed_time:.2f}s")

            # Collect final output
            results["status"] = execution_result.status.value
            results["output"] = self._collect_output(execution_result)
            results["elapsed_time"] = time.time() - results["started_at"]

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            results["elapsed_time"] = time.time() - results["started_at"]

            if self.config.verbose:
                print(f"\nError: {e}")

        return results

    def _generate_task_ir(self, user_prompt: str) -> dict[str, Any]:
        """Generate TaskIR from user prompt using Front Door model.

        Args:
            user_prompt: User's task description.

        Returns:
            TaskIR dictionary.
        """
        if self.config.skip_frontdoor:
            # Return a simple TaskIR for the prompt
            return self._simple_task_ir(user_prompt)

        if self.config.dry_run:
            # In dry run, return a mock TaskIR
            return self._mock_task_ir(user_prompt)

        # Load front door prompt template
        prompt_path = Path(__file__).parent.parent / "orchestration" / "prompts" / "frontdoor.md"
        if prompt_path.exists():
            with prompt_path.open() as f:
                system_prompt = f.read()
        else:
            system_prompt = "You are a task planner. Parse the user request into a structured task."

        # Build full prompt
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}\n\nGenerate TaskIR JSON:"

        # Run inference on front door model
        request = InferenceRequest(
            role="frontdoor",
            prompt=full_prompt,
            n_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )

        result = self.server.infer(request)

        if not result.success:
            raise RuntimeError(f"Front door inference failed: {result.error_message}")

        # Parse TaskIR from output
        return self._parse_task_ir(result.output, user_prompt)

    def _parse_task_ir(self, output: str, user_prompt: str) -> dict[str, Any]:
        """Parse TaskIR JSON from model output.

        Args:
            output: Raw model output.
            user_prompt: Original user prompt (for fallback).

        Returns:
            Parsed TaskIR dictionary.
        """
        # Try to find JSON in output
        import re

        # Look for JSON block
        json_match = re.search(r"\{[\s\S]*\}", output)
        if json_match:
            try:
                task_ir = json.loads(json_match.group())
                # Validate required fields
                if "task_id" in task_ir and "plan" in task_ir:
                    return task_ir
            except json.JSONDecodeError:
                pass

        # Fallback to simple TaskIR
        return self._simple_task_ir(user_prompt)

    def _simple_task_ir(self, user_prompt: str) -> dict[str, Any]:
        """Create a simple single-step TaskIR.

        Args:
            user_prompt: User's task description.

        Returns:
            Simple TaskIR dictionary.
        """
        task_id = f"task-{int(time.time())}"
        return {
            "task_id": task_id,
            "task_type": "code",
            "priority": "interactive",
            "objective": user_prompt,
            "agents": [{"role": "coder"}],
            "plan": {
                "steps": [
                    {
                        "id": "S1",
                        "actor": "coder",
                        "action": user_prompt,
                        "inputs": [],
                        "outputs": ["result"],
                        "depends_on": [],
                    }
                ]
            },
        }

    def _mock_task_ir(self, user_prompt: str) -> dict[str, Any]:
        """Create a mock TaskIR for dry run mode.

        Args:
            user_prompt: User's task description.

        Returns:
            Mock TaskIR dictionary.
        """
        return self._simple_task_ir(user_prompt)

    def _collect_output(self, execution: ExecutionResult) -> str:
        """Collect final output from execution.

        Args:
            execution: Execution result.

        Returns:
            Combined output string.
        """
        outputs = []
        for step_id, step_result in execution.steps.items():
            if step_result.status == StepStatus.COMPLETED and step_result.output:
                outputs.append(f"=== {step_id} ===\n{step_result.output}")

        return "\n\n".join(outputs) if outputs else "(No output)"


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hierarchical Local-Agent Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli "Write a Python sort function"
  python -m src.cli --dry-run "Analyze this code"
  python -m src.cli -v "Implement binary search"
  python -m src.cli --simple "Hello world in Python"
        """,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Task description or prompt",
    )

    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Simulate execution without running models",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    parser.add_argument(
        "-s",
        "--simple",
        action="store_true",
        help="Skip Front Door, use simple single-step TaskIR",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write results to JSON file",
    )

    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=300,
        help="Timeout per step in seconds (default: 300)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )

    parser.add_argument(
        "--list-roles",
        action="store_true",
        help="List available model roles and exit",
    )

    args = parser.parse_args()

    # List roles mode
    if args.list_roles:
        registry = RegistryLoader(validate_paths=True)
        print("Available Roles:")
        print("=" * 60)
        for name, role in registry.roles.items():
            status = "✓" if Path(role.model.full_path).exists() else "✗"
            accel = role.acceleration.type
            tps = role.performance.optimized_tps or role.performance.baseline_tps or "N/A"
            print(f"  {status} {name:20} {role.model.name[:30]:30} {tps} t/s ({accel})")
        return 0

    # Require prompt for other modes
    if not args.prompt:
        parser.print_help()
        return 1

    # Configure orchestrator
    config = OrchestratorConfig(
        dry_run=args.dry_run,
        verbose=args.verbose,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        output_file=args.output,
        skip_frontdoor=args.simple,
    )

    # Run orchestration
    orchestrator = Orchestrator(config)

    if args.verbose:
        print("=" * 60)
        print("Hierarchical Local-Agent Orchestration")
        print("=" * 60)
        print(f"Prompt: {args.prompt[:50]}...")
        print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")

    results = orchestrator.run(args.prompt)

    # Output results
    if args.output:
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to: {args.output}")
    elif args.verbose:
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Status: {results.get('status', 'unknown')}")
        print(f"Time: {results.get('elapsed_time', 0):.2f}s")
        if results.get("error"):
            print(f"Error: {results['error']}")
        print("\nOutput:")
        print(results.get("output", "(No output)"))
    else:
        # Simple output mode
        if results.get("status") == "completed":
            print(results.get("output", ""))
        else:
            print(f"Error: {results.get('error', 'Unknown error')}", file=sys.stderr)
            return 1

    return 0 if results.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())

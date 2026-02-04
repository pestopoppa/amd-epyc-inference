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

from src.config import _registry_timeout
from src.dispatcher import Dispatcher
from src.executor import ExecutionResult, Executor, ExecutorConfig, StepStatus
from src.model_server import ModelServer
from src.registry_loader import RegistryLoader

# Get default timeout from registry (single source of truth)
_DEFAULT_TIMEOUT = int(_registry_timeout("server", "request", 600))


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator.

    Timeout default comes from model_registry.yaml (runtime_defaults.timeouts.server.request).
    """

    dry_run: bool = False
    verbose: bool = False
    timeout: int = _DEFAULT_TIMEOUT
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
        """Generate TaskIR from user prompt.

        Uses smart heuristics to detect multi-step tasks and generate
        appropriate TaskIR without requiring model JSON generation.

        Args:
            user_prompt: User's task description.

        Returns:
            TaskIR dictionary.
        """
        if self.config.skip_frontdoor:
            return self._simple_task_ir(user_prompt)

        if self.config.dry_run:
            return self._mock_task_ir(user_prompt)

        # Detect multi-step patterns in the prompt
        prompt_lower = user_prompt.lower()

        # Multi-step indicators
        multi_step_patterns = [
            (" then ", 2),
            (" and then ", 2),
            (" after that ", 2),
            (" followed by ", 2),
            (" also ", 2),
            (" and also ", 2),
            (" with tests", 2),
            (" and test", 2),
            (" write tests", 2),
        ]

        # Check for multi-step task
        for pattern, _min_steps in multi_step_patterns:
            if pattern in prompt_lower:
                return self._multi_step_task_ir(user_prompt)

        # Check for numbered steps (1. 2. 3. or step 1, step 2)
        import re

        if re.search(r"\b(step\s*\d|^\d+\.|first.*second|1\).*2\))", prompt_lower):
            return self._multi_step_task_ir(user_prompt)

        # Default to simple single-step TaskIR
        return self._simple_task_ir(user_prompt)

    def _multi_step_task_ir(self, user_prompt: str) -> dict[str, Any]:
        """Create a multi-step TaskIR by splitting the prompt.

        Args:
            user_prompt: User's task description.

        Returns:
            Multi-step TaskIR dictionary.
        """
        task_id = f"task-{int(time.time())}"

        # Split on common separators
        import re

        # Try to split on "then", "and then", etc.
        parts = re.split(
            r"\s+(?:and\s+)?then\s+|\s+after\s+that\s+|\s+followed\s+by\s+",
            user_prompt,
            flags=re.IGNORECASE,
        )

        # If no split, try "and also" or just ", and"
        if len(parts) == 1:
            parts = re.split(
                r"\s+and\s+also\s+|\s*,\s+and\s+|\s+also\s+",
                user_prompt,
                flags=re.IGNORECASE,
            )

        # Clean up parts
        parts = [p.strip() for p in parts if p.strip()]

        # If still single part, check for test-related suffix
        if len(parts) == 1:
            test_match = re.search(
                r"(.+?)\s+(?:with\s+tests?|and\s+(?:write\s+)?tests?)",
                user_prompt,
                flags=re.IGNORECASE,
            )
            if test_match:
                parts = [test_match.group(1), "Write tests for the above"]

        # Build steps
        steps = []
        for i, part in enumerate(parts):
            step_id = f"S{i + 1}"
            depends = [f"S{i}"] if i > 0 else []
            output_name = f"step{i + 1}_result" if i < len(parts) - 1 else "result"

            steps.append(
                {
                    "id": step_id,
                    "actor": "coder",
                    "action": part,
                    "inputs": [f"step{i}_result"] if i > 0 else [],
                    "outputs": [output_name],
                    "depends_on": depends,
                }
            )

        return {
            "task_id": task_id,
            "task_type": "code",
            "priority": "interactive",
            "objective": user_prompt,
            "agents": [{"role": "coder"}],
            "plan": {"steps": steps},
            "definition_of_done": [f"Complete all {len(steps)} steps"],
        }

    def _parse_task_ir(self, output: str, user_prompt: str) -> dict[str, Any]:
        """Parse TaskIR JSON from model output.

        Args:
            output: Raw model output.
            user_prompt: Original user prompt (for fallback).

        Returns:
            Parsed TaskIR dictionary.
        """
        import re

        # Try multiple JSON extraction strategies

        # Strategy 1: Find JSON block with balanced braces
        brace_count = 0
        json_start = -1
        json_end = -1

        for i, char in enumerate(output):
            if char == "{":
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and json_start >= 0:
                    json_end = i + 1
                    break

        if json_start >= 0 and json_end > json_start:
            json_str = output[json_start:json_end]
            try:
                task_ir = json.loads(json_str)
                # Validate required fields
                if "plan" in task_ir:
                    # Ensure task_id exists
                    if "task_id" not in task_ir:
                        task_ir["task_id"] = f"task-{int(time.time())}"
                    return task_ir
            except json.JSONDecodeError:
                pass

        # Strategy 2: Try regex for JSON object
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", output, re.DOTALL)
        if json_match:
            try:
                task_ir = json.loads(json_match.group())
                if "plan" in task_ir:
                    if "task_id" not in task_ir:
                        task_ir["task_id"] = f"task-{int(time.time())}"
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
        default=_DEFAULT_TIMEOUT,
        help=f"Timeout per step in seconds (default: {_DEFAULT_TIMEOUT} from registry)",
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

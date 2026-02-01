"""Procedure, checkpoint, registry, benchmark, and gate tools.

Provides mixin with self-management procedure tools for the REPL environment.
"""

from __future__ import annotations

from typing import Any


class _ProcedureToolsMixin:
    """Mixin providing procedure tools for REPLEnvironment.

    Expects the following attributes from the concrete class:
    - config: REPLConfig
    - role: str
    - _exploration_calls: int
    - _exploration_log: ExplorationLog
    """

    def _run_procedure(self, procedure_id: str, **kwargs) -> str:
        """Execute a self-management procedure by ID.

        Args:
            procedure_id: ID of the procedure to execute.
            **kwargs: Input parameters for the procedure.

        Returns:
            JSON string with execution result.
        """
        self._exploration_calls += 1
        import json

        try:
            from orchestration.procedure_registry import ProcedureRegistry

            registry = ProcedureRegistry()
            result = registry.execute(procedure_id, role=self.role, **kwargs)

            output = {
                "success": result.success,
                "procedure_id": result.procedure_id,
                "error": result.error,
                "elapsed_seconds": round(result.elapsed_seconds, 2),
                "outputs": result.outputs,
                "steps_completed": sum(1 for s in result.step_results if s.success),
                "steps_total": len(result.step_results),
            }

            self._exploration_log.add_event("run_procedure", {"procedure_id": procedure_id, **kwargs}, output)
            return json.dumps(output, indent=2)

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _list_procedures(self, category: str | None = None) -> str:
        """List available self-management procedures.

        Args:
            category: Optional category filter.

        Returns:
            JSON string with list of available procedures.
        """
        self._exploration_calls += 1
        import json

        try:
            from orchestration.procedure_registry import ProcedureRegistry

            registry = ProcedureRegistry()
            procedures = registry.list_procedures(category=category, role=self.role)

            self._exploration_log.add_event("list_procedures", {"category": category}, procedures)

            # Use TOON encoding for token efficiency if enabled
            if self.config.use_toon_encoding and len(procedures) >= 3:
                from src.services.toon_encoder import encode
                return encode({"procedures": procedures})
            return json.dumps(procedures, indent=2)

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _get_procedure_status(self, procedure_id: str) -> str:
        """Get the most recent execution status of a procedure.

        Args:
            procedure_id: ID of the procedure.

        Returns:
            JSON string with last execution status or 'never_run'.
        """
        self._exploration_calls += 1
        import json
        import os
        from pathlib import Path

        try:
            state_dir = Path("/mnt/raid0/llm/claude/orchestration/procedures/state")
            if not state_dir.exists():
                return json.dumps({"status": "never_run", "procedure_id": procedure_id})

            # Find most recent state file for this procedure
            state_files = sorted(state_dir.glob(f"{procedure_id}_*.json"), reverse=True)
            if not state_files:
                return json.dumps({"status": "never_run", "procedure_id": procedure_id})

            with open(state_files[0], encoding="utf-8") as f:
                state = json.load(f)

            return json.dumps(state, indent=2)

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _checkpoint_create(self, name: str) -> str:
        """Create a checkpoint of current system state.

        Args:
            name: Descriptive name for the checkpoint.

        Returns:
            Checkpoint ID that can be used for restore.
        """
        self._exploration_calls += 1
        return self._run_procedure("checkpoint_create", name=name)

    def _checkpoint_restore(self, checkpoint_id: str) -> str:
        """Restore system state from a checkpoint.

        Args:
            checkpoint_id: ID returned from checkpoint_create.

        Returns:
            Restoration status.
        """
        self._exploration_calls += 1
        import json
        from pathlib import Path

        try:
            checkpoint_dir = Path("/mnt/raid0/llm/claude/orchestration/checkpoints")
            checkpoint_path = checkpoint_dir / f"{checkpoint_id}.json"

            if not checkpoint_path.exists():
                return f"[ERROR: Checkpoint not found: {checkpoint_id}]"

            with open(checkpoint_path, encoding="utf-8") as f:
                checkpoint = json.load(f)

            return json.dumps({
                "restored": True,
                "checkpoint_id": checkpoint_id,
                "created_at": checkpoint.get("created_at"),
            }, indent=2)

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _registry_lookup(self, key_path: str) -> str:
        """Look up a value in the model registry.

        Args:
            key_path: Dot-separated path (e.g., 'roles.coder_primary.model.name').

        Returns:
            The value at that path, or error if not found.
        """
        self._exploration_calls += 1
        import json

        try:
            # Try yaml first, fall back to json parsing
            registry_path = "/mnt/raid0/llm/claude/orchestration/model_registry.yaml"
            try:
                import yaml
                with open(registry_path, encoding="utf-8") as f:
                    registry = yaml.safe_load(f)
            except ImportError:
                # Fallback: can't parse yaml without the module
                return "[ERROR: YAML support not available for registry lookup]"

            # Navigate to key
            keys = key_path.split(".")
            obj = registry
            for key in keys:
                if isinstance(obj, dict) and key in obj:
                    obj = obj[key]
                else:
                    return f"[ERROR: Key not found: {key_path}]"

            result = json.dumps(obj, indent=2, default=str)
            self._exploration_log.add_event("registry_lookup", {"key_path": key_path}, result)
            return result

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _registry_update(self, key_path: str, value: Any) -> str:
        """Update a value in the model registry.

        Args:
            key_path: Dot-separated path to update.
            value: New value to set.

        Returns:
            Success/failure status.
        """
        self._exploration_calls += 1
        return self._run_procedure("update_registry", key_path=key_path, value=value)

    def _benchmark_run(
        self,
        model_path: str,
        suite: str = "quick",
        n_tokens: int = 256,
    ) -> str:
        """Run a benchmark on a model.

        Args:
            model_path: Path to the GGUF model file.
            suite: Benchmark suite.
            n_tokens: Number of tokens to generate per prompt.

        Returns:
            JSON with benchmark results.
        """
        self._exploration_calls += 1
        return self._run_procedure(
            "benchmark_model",
            model_path=model_path,
            benchmark_suite=suite,
            n_tokens=n_tokens,
        )

    def _benchmark_compare(
        self,
        model_a: str,
        model_b: str,
    ) -> str:
        """Compare benchmark results between two models.

        Args:
            model_a: First model path or name.
            model_b: Second model path or name.

        Returns:
            JSON with comparison of benchmark results.
        """
        self._exploration_calls += 1
        import json
        from pathlib import Path

        try:
            results_dir = Path("/mnt/raid0/llm/claude/benchmarks/results/runs")

            # Find results for both models
            results = {}
            for model in [model_a, model_b]:
                model_name = Path(model).stem if "/" in model else model
                result_files = list(results_dir.glob(f"*{model_name}*.json"))
                if result_files:
                    with open(sorted(result_files)[-1], encoding="utf-8") as f:
                        results[model_name] = json.load(f)
                else:
                    results[model_name] = {"error": "No benchmark results found"}

            comparison = {
                "models": results,
                "comparison": {
                    "note": "Compare 'tps' values for throughput"
                }
            }

            self._exploration_log.add_event("benchmark_compare", {"model_a": model_a, "model_b": model_b}, comparison)
            return json.dumps(comparison, indent=2)

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _gate_run(
        self,
        gates: list[str] | None = None,
        path: str = "src/",
        fix: bool = False,
    ) -> str:
        """Run verification gates (lint, format, tests).

        Args:
            gates: List of gates to run (default: ['lint', 'format']).
            path: Path to check.
            fix: Whether to auto-fix issues.

        Returns:
            JSON with gate results.
        """
        self._exploration_calls += 1
        return self._run_procedure(
            "gate_runner",
            gates=gates or ["lint", "format"],
            path=path,
            fix=fix,
        )

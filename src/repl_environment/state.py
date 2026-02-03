"""State inspection, exploration tracking, and checkpoint/restore.

Provides mixin with: get_state, exploration log access, checkpoint, restore, reset.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from src.repl_environment.types import ExplorationEvent, ExplorationLog

if TYPE_CHECKING:
    from orchestration.repl_memory.retriever import TwoPhaseRetriever


class _StateMixin:
    """Mixin providing state inspection and persistence.

    Includes: get_state, get_exploration_log, get_grep_history, clear_grep_history,
    get_exploration_strategy, log_exploration_completed, suggest_exploration,
    checkpoint, restore, get_checkpoint_metadata, reset.

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig — environment configuration
        context: str — full input context
        artifacts: dict — collected artifacts
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
        _execution_count: int — number of execute() calls
        _final_answer: str | None — final answer if set
        _grep_hits_buffer: list — grep results buffer
        _findings_buffer: list — key findings buffer
        _globals: dict — restricted globals for REPL execution
        progress_logger: Any | None — progress tracking service
        task_id: str — task identifier for logging
        _build_globals: Callable[[], dict] — method to rebuild globals dict
    """

    def get_state(self) -> str:
        """Get a summary of current REPL state for the Root LM.

        Returns:
            String describing available variables and artifacts.
        """
        state_lines = [
            f"context: str ({len(self.context)} chars)",
            f"artifacts: {list(self.artifacts.keys()) if self.artifacts else '{}'}",
        ]

        # Show artifact previews
        for key, value in self.artifacts.items():
            preview = str(value)[:100]
            if len(str(value)) > 100:
                preview += "..."
            state_lines.append(f"  artifacts['{key}']: {preview}")

        return "\n".join(state_lines)

    def get_exploration_log(self) -> ExplorationLog:
        """Get the detailed exploration log.

        Returns:
            ExplorationLog containing all exploration events.
        """
        return self._exploration_log

    def get_grep_history(self) -> list[dict[str, Any]]:
        """Get grep hits buffer for two-stage summarization.

        Returns:
            List of grep hit records.
        """
        return self._grep_hits_buffer

    def clear_grep_history(self) -> None:
        """Clear the grep hits buffer.

        Call this when starting a new summarization task to avoid
        mixing grep hits from different documents.
        """
        self._grep_hits_buffer = []

    def get_exploration_strategy(self) -> dict[str, Any]:
        """Get a summary of the exploration strategy used.

        Returns:
            Dictionary with strategy summary including event counts and type.
        """
        return self._exploration_log.get_strategy_summary()

    def log_exploration_completed(
        self,
        success: bool,
        result: str = "",
    ) -> dict[str, Any]:
        """Log exploration completion to ProgressLogger.

        Args:
            success: Whether the task completed successfully.
            result: The final result (used for token efficiency calculation).

        Returns:
            Dictionary with the logged exploration data.
        """
        strategy = self.get_exploration_strategy()
        result_tokens = len(result) // 4  # Rough token estimate
        efficiency = self._exploration_log.get_token_efficiency(result_tokens)

        exploration_data = {
            "strategy": strategy,
            "efficiency": efficiency,
            "success": success,
        }

        # Log to ProgressLogger if available
        if self.progress_logger is not None:
            query_preview = self.context[:100] if self.context else ""
            self.progress_logger.log_exploration(
                task_id=self.task_id,
                query=query_preview,
                strategy_used=strategy.get("strategy_type", "unknown"),
                tokens_spent=strategy.get("total_tokens", 0),
                success=success,
                function_counts=strategy.get("function_counts"),
            )

        return exploration_data

    def suggest_exploration(
        self,
        task_description: str,
        retriever: TwoPhaseRetriever | None = None,
    ) -> list[str]:
        """Suggest exploration strategies based on similar past tasks.

        Args:
            task_description: Description of the current task.
            retriever: TwoPhaseRetriever from orchestration.repl_memory (optional).

        Returns:
            List of suggested exploration function calls as strings.
        """
        suggestions = []
        episodic_suggestions = []

        # If retriever available, query for similar successful exploration tasks
        if retriever is not None:
            try:
                context_preview = self.context[:500] if self.context else ""
                results = retriever.retrieve_for_exploration(
                    query=task_description,
                    context_preview=context_preview,
                )

                if results:
                    # Extract suggestions from successful similar tasks
                    for r in results[:3]:
                        # Only use high-quality memories (Q > 0.6, successful)
                        if r.q_value < 0.6 or r.memory.outcome != "success":
                            continue

                        context = r.memory.context or {}
                        strategy = context.get("exploration_strategy", {})
                        function_counts = strategy.get("function_counts", {})
                        strategy_type = strategy.get("strategy_type", "")

                        # Generate specific suggestions based on what worked
                        if function_counts.get("grep", 0) > 0:
                            episodic_suggestions.append(
                                f"grep('pattern')  # Similar task (q={r.q_value:.2f}) used grep"
                            )
                        if function_counts.get("llm_call", 0) > 0:
                            episodic_suggestions.append(
                                "llm_call('summarize key points')  # Similar task delegated effectively"
                            )
                        if strategy_type == "scan" and function_counts.get("peek", 0) > 0:
                            peek_count = function_counts["peek"]
                            episodic_suggestions.append(
                                f"# Scan strategy worked: {peek_count} peek() calls"
                            )

            except Exception:
                pass  # Silently ignore retrieval errors

        # Default suggestions based on context characteristics
        context_len = len(self.context)

        if context_len < 500:
            suggestions.append("peek(500)  # Context is short, read it all")
        elif context_len < 2000:
            suggestions.append("peek(1000)  # Scan the beginning")
        else:
            suggestions.append("peek(500)  # Preview context")
            suggestions.append("grep('keyword')  # Search for specific patterns")

        # Prepend episodic suggestions (learned patterns first)
        return episodic_suggestions + suggestions

    # =========================================================================
    # Checkpoint & Restore (for session persistence)
    # =========================================================================

    def checkpoint(self) -> dict[str, Any]:
        """Create a checkpoint of the current REPL state.

        Returns:
            Dict suitable for JSON serialization and later restore().
        """
        import json

        def is_json_serializable(value: Any) -> bool:
            """Check if a value can be JSON serialized."""
            try:
                json.dumps(value)
                return True
            except (TypeError, ValueError, OverflowError):
                return False

        def sanitize_value(value: Any) -> Any:
            """Sanitize a value for JSON serialization."""
            if is_json_serializable(value):
                return value
            # Mark as unserializable with type info
            return {
                "__unserializable__": True,
                "type": type(value).__name__,
                "repr": repr(value)[:100],  # Truncated repr for debugging
            }

        def sanitize_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
            """Sanitize artifacts dict, marking non-serializable values."""
            sanitized = {}
            for key, value in artifacts.items():
                if isinstance(value, dict):
                    # Recursively sanitize nested dicts
                    sanitized[key] = sanitize_artifacts(value)
                elif isinstance(value, list):
                    # Sanitize list items
                    sanitized[key] = [sanitize_value(item) for item in value]
                else:
                    sanitized[key] = sanitize_value(value)
            return sanitized

        # Sanitize exploration log events for serialization
        exploration_events = []
        for event in self._exploration_log.events:
            exploration_events.append(
                {
                    "function": event.function,
                    "args": sanitize_artifacts(event.args) if isinstance(event.args, dict) else {},
                    "result_size": event.result_size,
                    "timestamp": event.timestamp,
                    "token_estimate": event.token_estimate,
                }
            )

        return {
            "version": 1,  # Schema version for future compatibility
            "artifacts": sanitize_artifacts(self.artifacts),
            "execution_count": self._execution_count,
            "exploration_calls": self._exploration_calls,
            "exploration_tokens": self._exploration_log.total_exploration_tokens,
            "exploration_events": exploration_events,
            "grep_hits_buffer": self._grep_hits_buffer,
            "findings_buffer": self._findings_buffer,  # Key findings
            "context_length": len(self.context),  # For verification, not full context
            "task_id": self.task_id,
        }

    def restore(self, checkpoint: dict[str, Any]) -> None:
        """Restore REPL state from a checkpoint.

        Note: Non-serializable artifacts remain as marker dicts.
        The context is NOT restored - it should be passed to __init__.

        Args:
            checkpoint: Dict from a previous checkpoint() call.

        Raises:
            ValueError: If checkpoint format is invalid.
        """
        version = checkpoint.get("version", 0)
        if version != 1:
            raise ValueError(f"Unsupported checkpoint version: {version}")

        # Restore artifacts
        self.artifacts = checkpoint.get("artifacts", {})

        # Restore execution state
        self._execution_count = checkpoint.get("execution_count", 0)
        self._exploration_calls = checkpoint.get("exploration_calls", 0)

        # Restore exploration log
        self._exploration_log = ExplorationLog()
        self._exploration_log.total_exploration_tokens = checkpoint.get("exploration_tokens", 0)
        for event_data in checkpoint.get("exploration_events", []):
            event = ExplorationEvent(
                function=event_data.get("function", ""),
                args=event_data.get("args", {}),
                result_size=event_data.get("result_size", 0),
                timestamp=event_data.get("timestamp", 0.0),
                token_estimate=event_data.get("token_estimate", 0),
            )
            self._exploration_log.events.append(event)

        # Restore grep hits buffer
        self._grep_hits_buffer = checkpoint.get("grep_hits_buffer", [])

        # Restore findings buffer
        self._findings_buffer = checkpoint.get("findings_buffer", [])

        # Rebuild globals with restored artifacts
        self._globals = self._build_globals()

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get metadata about current state for checkpoint decision.

        Returns:
            Dict with state metadata.
        """
        return {
            "execution_count": self._execution_count,
            "exploration_calls": self._exploration_calls,
            "artifact_count": len(self.artifacts),
            "context_length": len(self.context),
            "grep_hits_count": len(self._grep_hits_buffer),
            "findings_count": len(self._findings_buffer),
        }

    def reset(self) -> None:
        """Reset the REPL state (clear artifacts, keep context)."""
        self.artifacts.clear()
        self._final_answer = None
        self._execution_count = 0
        self._exploration_calls = 0
        self._exploration_log = ExplorationLog()  # Reset exploration log
        self._grep_hits_buffer = []  # Clear grep history for two-stage pipeline
        self._findings_buffer = []  # Clear findings buffer
        self._globals = self._build_globals()

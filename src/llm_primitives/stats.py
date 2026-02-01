"""Statistics and logging utilities."""

from typing import Any

from .types import CallLogEntry


class StatsMixin:
    """Mixin for statistics and logging methods."""

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about LLM calls.

        Returns:
            Dict with call counts, token counts, cache stats, etc.
        """
        stats = {
            "total_calls": self.total_calls,
            "total_batch_calls": self.total_batch_calls,
            "total_tokens_generated": self.total_tokens_generated,
            "total_prompt_eval_ms": self.total_prompt_eval_ms,
            "total_generation_ms": self.total_generation_ms,
            "last_predicted_tps": self._last_predicted_tps,
            "call_log_size": len(self.call_log),
            "mock_mode": self.mock_mode,
            "max_recursion_depth_reached": self._max_recursion_depth_reached,
            "current_recursion_depth": self._recursion_depth,
        }

        # Add cache stats if using CachingBackend
        if self._backends:
            cache_stats = self.get_cache_stats()
            stats["cache_stats"] = cache_stats

            # Calculate aggregate hit rate
            total_routes = 0
            total_hits = 0
            for role_stats in cache_stats.values():
                total_routes += role_stats.get("router_total_routes", 0)
                total_hits += role_stats.get("router_hit_rate", 0) * role_stats.get("router_total_routes", 0)
            if total_routes > 0:
                stats["aggregate_cache_hit_rate"] = total_hits / total_routes

        return stats

    def get_recent_calls(self, n: int = 10) -> list[CallLogEntry]:
        """Get the most recent call log entries.

        Args:
            n: Number of entries to return.

        Returns:
            List of recent CallLogEntry objects.
        """
        return self.call_log[-n:]

    def clear_log(self) -> None:
        """Clear the call log."""
        self.call_log.clear()

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.total_calls = 0
        self.total_batch_calls = 0
        self.total_tokens_generated = 0
        self.total_prompt_eval_ms = 0.0
        self.total_generation_ms = 0.0
        self._last_predicted_tps = 0.0
        self.call_log.clear()
        self._max_recursion_depth_reached = 0
        # Note: _recursion_depth is not reset as it tracks current call stack

    def reset_counters(self) -> None:
        """Reset per-request counters for reuse across requests.

        Call this when reusing a shared LLMPrimitives instance across
        multiple API requests to get accurate per-request metrics.
        """
        self.call_log.clear()
        self.total_calls = 0
        self.total_batch_calls = 0
        self.total_tokens_generated = 0
        self.total_prompt_eval_ms = 0.0
        self.total_generation_ms = 0.0
        self.total_http_overhead_ms = 0.0
        self._last_predicted_tps = 0.0
        self._recursion_depth = 0
        self._max_recursion_depth_reached = 0
        self._current_query = None
        self._completed_queries.clear()

"""Cost tracking utilities for LLM primitives."""

from .types import QueryCost


class CostTrackingMixin:
    """Mixin for cost tracking methods."""

    def start_query(self, query_id: str) -> None:
        """Start tracking costs for a new query.

        Args:
            query_id: Unique identifier for this query (e.g., task_id).
        """
        # End any existing query first
        if self._current_query is not None:
            self.end_query()

        self._current_query = QueryCost(
            query_id=query_id,
            prompt_rate=self.config.default_prompt_rate,
            completion_rate=self.config.default_completion_rate,
        )

    def end_query(self) -> QueryCost | None:
        """End current query tracking and return the cost.

        Returns:
            QueryCost for the completed query, or None if no query active.
        """
        if self._current_query is None:
            return None

        query = self._current_query
        self._completed_queries.append(query)
        self._current_query = None
        return query

    def get_current_query_cost(self) -> QueryCost | None:
        """Get cost for the current query (if active).

        Returns:
            QueryCost for current query, or None if no query active.
        """
        return self._current_query

    def get_completed_queries(self, last_n: int | None = None) -> list[QueryCost]:
        """Get completed query costs.

        Args:
            last_n: Only return last N queries (None = all).

        Returns:
            List of QueryCost for completed queries.
        """
        if last_n is None:
            return list(self._completed_queries)
        return self._completed_queries[-last_n:]

    def get_total_cost(self) -> float:
        """Get total estimated cost for all completed queries.

        Returns:
            Total cost in dollars.
        """
        return sum(q.estimated_cost for q in self._completed_queries)

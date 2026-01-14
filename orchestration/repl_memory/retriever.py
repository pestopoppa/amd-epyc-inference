"""
TwoPhaseRetriever: MemRL-style two-phase retrieval for episodic memory.

Phase 1: Semantic filtering - retrieve k candidates by embedding similarity
Phase 2: Q-value ranking - rank candidates by learned utility

This separates "what's similar" from "what's useful" - the key insight from MemRL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embedder import TaskEmbedder
from .episodic_store import EpisodicStore, MemoryEntry


@dataclass
class RetrievalResult:
    """Result of a two-phase retrieval."""

    memory: MemoryEntry
    similarity: float  # Cosine similarity (0-1)
    q_value: float  # Learned utility (0-1)
    combined_score: float  # Weighted combination


@dataclass
class RetrievalConfig:
    """Configuration for two-phase retrieval."""

    # Phase 1: Semantic filtering
    semantic_k: int = 20  # Number of candidates to retrieve
    min_similarity: float = 0.3  # Minimum similarity threshold

    # Phase 2: Q-value ranking
    min_q_value: float = 0.3  # Minimum Q-value to consider
    q_weight: float = 0.7  # Weight of Q-value vs similarity (0-1)

    # Final selection
    top_n: int = 5  # Number of results to return

    # Confidence threshold for using learned routing
    confidence_threshold: float = 0.6  # Min combined score to trust


class TwoPhaseRetriever:
    """
    Two-phase retrieval system for episodic memory.

    Phase 1 (Semantic Filtering):
    - Embed the query
    - Retrieve top-k memories by cosine similarity
    - Filter by minimum similarity threshold

    Phase 2 (Q-Value Ranking):
    - Rank candidates by learned Q-value
    - Combine similarity and Q-value scores
    - Return top-n by combined score
    """

    def __init__(
        self,
        store: EpisodicStore,
        embedder: TaskEmbedder,
        config: Optional[RetrievalConfig] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.config = config or RetrievalConfig()

    def retrieve_for_routing(
        self,
        task_ir: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """
        Retrieve memories for task routing decision.

        Args:
            task_ir: TaskIR dictionary

        Returns:
            List of RetrievalResult sorted by combined score
        """
        embedding = self.embedder.embed_task_ir(task_ir)
        return self._retrieve(embedding, action_type="routing")

    def retrieve_for_escalation(
        self,
        failure_context: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """
        Retrieve memories for escalation decision.

        Args:
            failure_context: Failure context dictionary

        Returns:
            List of RetrievalResult sorted by combined score
        """
        embedding = self.embedder.embed_failure_context(failure_context)
        return self._retrieve(embedding, action_type="escalation")

    def retrieve_for_exploration(
        self,
        query: str,
        context_preview: str,
    ) -> List[RetrievalResult]:
        """
        Retrieve memories for REPL exploration strategy.

        Args:
            query: User query
            context_preview: Preview of context being explored

        Returns:
            List of RetrievalResult sorted by combined score
        """
        embedding = self.embedder.embed_exploration(query, context_preview)
        return self._retrieve(embedding, action_type="exploration")

    def _retrieve(
        self,
        embedding: np.ndarray,
        action_type: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Execute two-phase retrieval.

        Args:
            embedding: Query embedding
            action_type: Optional filter by action type

        Returns:
            List of RetrievalResult sorted by combined score
        """
        # Phase 1: Semantic filtering
        candidates = self.store.retrieve_by_similarity(
            embedding,
            k=self.config.semantic_k,
            action_type=action_type,
            min_q_value=self.config.min_q_value,
        )

        if not candidates:
            return []

        # Compute similarities for candidates
        query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        results = []

        for memory in candidates:
            mem_norm = memory.embedding / (np.linalg.norm(memory.embedding) + 1e-8)
            similarity = float(np.dot(query_norm, mem_norm))

            # Skip if below similarity threshold
            if similarity < self.config.min_similarity:
                continue

            # Phase 2: Combine similarity and Q-value
            combined = (
                self.config.q_weight * memory.q_value
                + (1 - self.config.q_weight) * similarity
            )

            results.append(
                RetrievalResult(
                    memory=memory,
                    similarity=similarity,
                    q_value=memory.q_value,
                    combined_score=combined,
                )
            )

        # Sort by combined score (descending)
        results.sort(key=lambda r: r.combined_score, reverse=True)

        # Return top-n
        return results[: self.config.top_n]

    def get_best_action(
        self,
        results: List[RetrievalResult],
    ) -> Optional[Tuple[str, float]]:
        """
        Get the best action from retrieval results if confidence is high enough.

        Args:
            results: Retrieval results

        Returns:
            (action, confidence) tuple or None if not confident
        """
        if not results:
            return None

        best = results[0]
        if best.combined_score >= self.config.confidence_threshold:
            return (best.memory.action, best.combined_score)

        return None

    def should_use_learned(
        self,
        results: List[RetrievalResult],
        min_samples: int = 3,
    ) -> bool:
        """
        Determine if we should use learned routing or fall back to rules.

        Criteria:
        - Have enough samples (min_samples)
        - Best result exceeds confidence threshold
        - Q-values are based on actual observations (update_count > 0)

        Args:
            results: Retrieval results
            min_samples: Minimum number of similar samples

        Returns:
            True if should use learned routing
        """
        if len(results) < min_samples:
            return False

        best = results[0]
        if best.combined_score < self.config.confidence_threshold:
            return False

        # Check that Q-values have been updated (not just initial values)
        observed_count = sum(1 for r in results if r.memory.update_count > 0)
        if observed_count < min_samples:
            return False

        return True


class HybridRouter:
    """
    Hybrid routing that combines learned and rule-based approaches.

    Uses learned routing when confident, falls back to rules otherwise.
    This implements the cold start strategy from the plan.
    """

    def __init__(
        self,
        retriever: TwoPhaseRetriever,
        rule_based_router: "RuleBasedRouter",  # Forward reference
    ):
        self.retriever = retriever
        self.rule_based = rule_based_router

    def route(self, task_ir: Dict[str, Any]) -> Tuple[List[str], str]:
        """
        Route a task using hybrid strategy.

        Args:
            task_ir: TaskIR dictionary

        Returns:
            (routing_decision, strategy_used) tuple
            strategy_used is "learned" or "rules"
        """
        # Try learned routing first
        results = self.retriever.retrieve_for_routing(task_ir)

        if self.retriever.should_use_learned(results):
            best_action = self.retriever.get_best_action(results)
            if best_action:
                action, confidence = best_action
                # Parse action as routing decision
                routing = self._parse_routing_action(action)
                return (routing, "learned")

        # Fall back to rule-based routing
        routing = self.rule_based.route(task_ir)
        return (routing, "rules")

    def _parse_routing_action(self, action: str) -> List[str]:
        """Parse stored action string to routing list."""
        # Actions are stored as comma-separated role names
        return [r.strip() for r in action.split(",")]


class RuleBasedRouter:
    """
    Rule-based router from model_registry.yaml.

    Used as fallback when learned routing is not confident.
    """

    def __init__(self, routing_hints: List[Dict[str, Any]]):
        """
        Initialize with routing hints from model_registry.yaml.

        Args:
            routing_hints: List of routing hint dictionaries
        """
        self.routing_hints = routing_hints

    def route(self, task_ir: Dict[str, Any]) -> List[str]:
        """
        Route using rule-based hints.

        Args:
            task_ir: TaskIR dictionary

        Returns:
            List of role names to use
        """
        task_type = task_ir.get("task_type", "chat")
        priority = task_ir.get("priority", "interactive")
        has_images = any(
            inp.get("type") == "image" for inp in task_ir.get("inputs", [])
        )

        # Check routing hints
        for hint in self.routing_hints:
            condition = hint.get("if", "")
            if self._evaluate_condition(
                condition, task_type, priority, has_images, task_ir
            ):
                return hint.get("use", ["frontdoor"])

        # Default routing
        return ["frontdoor"]

    def _evaluate_condition(
        self,
        condition: str,
        task_type: str,
        priority: str,
        has_images: bool,
        task_ir: Dict[str, Any],
    ) -> bool:
        """Evaluate a routing condition."""
        # Simple condition parsing (from model_registry.yaml format)
        if "task_type == 'code'" in condition and task_type == "code":
            return True
        if "task_type == 'ingest'" in condition and task_type == "ingest":
            return True
        if "task_type == 'doc'" in condition and task_type == "doc":
            if "priority == 'interactive'" in condition and priority == "interactive":
                return True
            elif "priority" not in condition:
                return True
        if "task_type == 'manage'" in condition and task_type == "manage":
            return True
        if "has_images == true" in condition and has_images:
            return True
        if "needs_math_reasoning == true" in condition:
            # Check for math-related keywords in objective
            objective = task_ir.get("objective", "").lower()
            if any(kw in objective for kw in ["math", "calculate", "prove", "theorem"]):
                return True
        return False

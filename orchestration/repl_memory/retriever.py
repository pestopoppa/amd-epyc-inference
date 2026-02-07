"""
TwoPhaseRetriever: MemRL-style two-phase retrieval for episodic memory.

Phase 1: Semantic filtering - retrieve k candidates by embedding similarity
Phase 2: Q-value ranking - rank candidates by learned utility
Phase 3 (optional): Graph-enhanced scoring with failure penalties and hypothesis confidence

This separates "what's similar" from "what's useful" - the key insight from MemRL.
Enhanced with failure anti-memory and hypothesis tracking from Graphiti-inspired design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .embedder import TaskEmbedder
from .episodic_store import EpisodicStore, MemoryEntry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .failure_graph import FailureGraph
    from .hypothesis_graph import HypothesisGraph


@dataclass
class RetrievalResult:
    """Result of a two-phase (or three-phase) retrieval."""

    memory: MemoryEntry
    similarity: float  # Cosine similarity (0-1)
    q_value: float  # Learned utility (0-1)
    combined_score: float  # Weighted combination

    # Graph-enhanced fields (optional)
    failure_penalty: float = 0.0  # Risk score from failure graph (0-1)
    hypothesis_confidence: float = 1.0  # Confidence from hypothesis graph (0-1)
    adjusted_score: float = 0.0  # Final score after graph adjustments
    warnings: List[str] = field(default_factory=list)  # Low-confidence warnings


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

    def retrieve_for_classification(
        self,
        prompt: str,
        classification_type: str = "routing",
    ) -> List[RetrievalResult]:
        """
        Retrieve memories for classification decision.

        Used by ClassificationRetriever to find similar classification exemplars
        and return the most confident classification based on Q-value weighted voting.

        Args:
            prompt: User prompt to classify.
            classification_type: Type of classification (routing, summarization, etc.).

        Returns:
            List of RetrievalResult sorted by combined score.
        """
        embedding = self.embedder.embed_classification_prompt(prompt, classification_type)
        return self._retrieve(embedding, action_type="classification")

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
            if memory.embedding is None:
                similarity = memory.similarity_score
            else:
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
        - At least some memories have been observed (Q-value != 0.5 default)

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

        # Check that Q-values are based on observations (not default 0.5)
        # Initial Q-values (0.5 + reward*0.5) are informative:
        # - Success → Q=1.0
        # - Failure → Q=0.25
        observed_count = sum(
            1 for r in results
            if r.memory.update_count > 0 or abs(r.memory.q_value - 0.5) > 0.1
        )
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

    def route_with_mode(
        self, task_ir: Dict[str, Any]
    ) -> Tuple[List[str], str, str]:
        """Route a task with mode selection (direct/react/repl).

        Extends route() to also return the recommended execution mode.
        Mode is parsed from action strings in format "role:mode" (colon-separated).
        Falls back to rule-based mode selection if no mode annotation found.

        Args:
            task_ir: TaskIR dictionary

        Returns:
            (routing_decision, strategy_used, mode) tuple
            mode is "direct", "react", or "repl"
        """
        # Try learned routing first
        results = self.retriever.retrieve_for_routing(task_ir)

        if self.retriever.should_use_learned(results):
            best_action = self.retriever.get_best_action(results)
            if best_action:
                action, confidence = best_action
                routing, mode = self._parse_routing_action_with_mode(action)
                return (routing, "learned", mode)

        # Fall back to rule-based routing (with mode)
        routing, mode = self.rule_based.route_with_mode(task_ir)
        return (routing, "rules", mode)

    def route_3way(
        self, task_ir: Dict[str, Any], cost_tiers: Optional[Dict[str, int]] = None
    ) -> Tuple[str, str, float]:
        """Route using 3-way action vocabulary with cost-adjusted decision.

        3-way actions:
        - SELF:direct - Frontdoor without tools
        - SELF:repl - Frontdoor with tools, no delegation
        - ARCHITECT - Architect with full delegation freedom
        - WORKER - (Not directly routed to, used via delegation)

        The Q-values represent P(success|action). Cost is applied at decision time:
            score = Q(action) / cost_tier

        Args:
            task_ir: TaskIR dictionary.
            cost_tiers: Optional cost tiers per action. Defaults to standard tiers.

        Returns:
            (action, strategy, confidence) tuple.
            action is one of: "SELF:direct", "SELF:repl", "ARCHITECT"
            strategy is "learned" or "rules"
        """
        # Default cost tiers (same as seeding_types.THREE_WAY_COST_TIER)
        if cost_tiers is None:
            cost_tiers = {
                "SELF:direct": 2,
                "SELF:repl": 2,
                "ARCHITECT": 4,
                "WORKER": 1,
            }

        # Retrieve learned Q-values
        results = self.retriever.retrieve_for_routing(task_ir)

        if self.retriever.should_use_learned(results):
            # Aggregate Q-values by 3-way category
            q_values: Dict[str, List[float]] = {
                "SELF:direct": [],
                "SELF:repl": [],
                "ARCHITECT": [],
            }

            for r in results:
                action = r.memory.action
                # Map old action format to 3-way categories
                if action in q_values:
                    # Already in 3-way format
                    q_values[action].append(r.q_value)
                elif action.startswith("frontdoor:direct"):
                    q_values["SELF:direct"].append(r.q_value)
                elif action.startswith("frontdoor:repl"):
                    q_values["SELF:repl"].append(r.q_value)
                elif action.startswith(("architect_", "ARCHITECT")):
                    q_values["ARCHITECT"].append(r.q_value)

            # Average Q-values per category
            avg_q = {}
            for action, values in q_values.items():
                if values:
                    avg_q[action] = sum(values) / len(values)
                else:
                    avg_q[action] = 0.5  # Default neutral

            # Apply cost-adjusted scoring
            scores = {}
            for action, q in avg_q.items():
                cost = cost_tiers.get(action, 2)
                scores[action] = q / cost

            # Select best action
            best_action = max(scores, key=scores.get)
            confidence = avg_q[best_action]

            return (best_action, "learned", confidence)

        # Fall back to rule-based routing
        # Determine 3-way action from task characteristics
        task_type = task_ir.get("task_type", "chat")
        objective = task_ir.get("objective", "").lower()
        context_length = task_ir.get("context_length", 0)

        # Heuristics for 3-way routing
        if task_type in ("architecture", "design", "complex"):
            return ("ARCHITECT", "rules", 0.5)
        elif context_length > 20000:
            return ("SELF:repl", "rules", 0.5)  # Large context needs tools
        elif any(kw in objective for kw in ["search", "file", "explore", "read"]):
            return ("SELF:repl", "rules", 0.5)
        else:
            return ("SELF:direct", "rules", 0.5)

    def _parse_routing_action(self, action: str) -> List[str]:
        """Parse stored action string to routing list."""
        # Actions are stored as comma-separated role names
        # Also handle "role:mode" format by stripping mode suffix
        roles = []
        for r in action.split(","):
            r = r.strip()
            if ":" in r:
                r = r.split(":")[0]  # Strip mode suffix
            roles.append(r)
        return roles

    def _parse_routing_action_with_mode(
        self, action: str
    ) -> Tuple[List[str], str]:
        """Parse stored action string to routing list and mode.

        Action format: "role1:mode,role2" — colon separates role from mode.
        Only the first role's mode is used. If no mode annotation, defaults
        to "direct".

        Args:
            action: Action string from episodic memory.

        Returns:
            (routing_list, mode) tuple.
        """
        roles = []
        mode = "direct"  # Default mode
        for i, r in enumerate(action.split(",")):
            r = r.strip()
            if ":" in r:
                role_part, mode_part = r.split(":", 1)
                roles.append(role_part)
                if i == 0:  # Mode from first role
                    mode = mode_part if mode_part in ("direct", "react", "repl") else "direct"
            else:
                roles.append(r)
        return roles, mode


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
            # Support both dict and RoutingHint dataclass
            if hasattr(hint, "condition"):
                condition = hint.condition
                use = hint.use
            else:
                condition = hint.get("if", "")
                use = hint.get("use", ["frontdoor"])
            if self._evaluate_condition(
                condition, task_type, priority, has_images, task_ir
            ):
                return use

        # Default routing
        return ["frontdoor"]

    def route_with_mode(
        self, task_ir: Dict[str, Any]
    ) -> Tuple[List[str], str]:
        """Route using rule-based hints with mode selection.

        Mode is selected based on task characteristics:
        - Large context → "repl" (needs peek/grep/summarize_chunks)
        - Tool-needing keywords → "react" (search, calculate, date)
        - Everything else → "direct" (best instruction-following quality)

        Args:
            task_ir: TaskIR dictionary

        Returns:
            (routing_list, mode) tuple
        """
        routing = self.route(task_ir)

        # Determine mode from task characteristics
        objective = task_ir.get("objective", "").lower()
        context_len = task_ir.get("context_length", 0)
        task_type = task_ir.get("task_type", "chat")

        # Large context → REPL for chunked exploration
        if context_len > 20000 or task_type == "ingest":
            return routing, "repl"

        # File operations → REPL
        file_indicators = [
            "read file", "list files", "explore", "scan",
            "write to", "save to", "open the file",
        ]
        if any(ind in objective for ind in file_indicators):
            return routing, "repl"

        # Tool-needing queries → ReAct
        react_indicators = [
            "search", "look up", "find information",
            "current date", "current time", "calculate",
            "search arxiv", "search papers", "wikipedia",
        ]
        if any(ind in objective for ind in react_indicators):
            return routing, "react"

        # Default → direct (best quality for instruction following)
        return routing, "direct"

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


class GraphEnhancedRetriever(TwoPhaseRetriever):
    """
    Graph-enhanced retriever with failure anti-memory and hypothesis tracking.

    Extends TwoPhaseRetriever with:
    - Failure graph penalty: Penalize actions linked to past failures
    - Hypothesis confidence: Warn on low-confidence action-task combinations
    - TTL caching: Graph lookups cached for 60s (80%+ cache hit rate expected)

    Scoring formula:
        adjusted_score = similarity × Q_value × (1 - failure_penalty) × hypothesis_confidence
    """

    def __init__(
        self,
        store: EpisodicStore,
        embedder: TaskEmbedder,
        failure_graph: Optional["FailureGraph"] = None,
        hypothesis_graph: Optional["HypothesisGraph"] = None,
        config: Optional[RetrievalConfig] = None,
        cache_ttl: int = 60,  # Cache TTL in seconds
        cache_maxsize: int = 500,  # Max cached items
    ):
        super().__init__(store, embedder, config)
        self.failure_graph = failure_graph
        self.hypothesis_graph = hypothesis_graph

        # TTL caches for graph lookups (5-20ms -> <1ms for cache hits)
        try:
            from cachetools import TTLCache
            self._failure_cache: Optional[TTLCache] = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
            self._confidence_cache: Optional[TTLCache] = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        except ImportError:
            # cachetools not installed - fall back to no caching
            self._failure_cache = None
            self._confidence_cache = None

    def _get_failure_penalty(self, action: str) -> float:
        """Get failure penalty with caching (5-20ms -> <1ms for cache hits)."""
        if self.failure_graph is None:
            return 0.0

        # Check cache first
        if self._failure_cache is not None:
            if action in self._failure_cache:
                return self._failure_cache[action]

        # Cache miss - fetch from graph
        try:
            penalty = self.failure_graph.get_failure_risk(action)
            if self._failure_cache is not None:
                self._failure_cache[action] = penalty
            return penalty
        except Exception:
            return 0.0  # Graceful degradation

    def _get_hypothesis_confidence(self, action: str, task_type: str) -> float:
        """Get hypothesis confidence with caching (5-20ms -> <1ms for cache hits)."""
        if self.hypothesis_graph is None or not task_type:
            return 1.0

        # Check cache first (key is action|task_type)
        cache_key = f"{action}|{task_type}"
        if self._confidence_cache is not None:
            if cache_key in self._confidence_cache:
                return self._confidence_cache[cache_key]

        # Cache miss - fetch from graph
        try:
            confidence = self.hypothesis_graph.get_confidence(action, task_type)
            if self._confidence_cache is not None:
                self._confidence_cache[cache_key] = confidence
            return confidence
        except Exception:
            return 1.0  # Graceful degradation

    def _retrieve(
        self,
        embedding: np.ndarray,
        action_type: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Execute three-phase retrieval with graph enhancement.

        Args:
            embedding: Query embedding
            action_type: Optional filter by action type
            task_type: Task type for hypothesis lookup

        Returns:
            List of RetrievalResult sorted by adjusted score
        """
        # Phase 1 & 2: Standard retrieval
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
            # Handle case where embedding might be None (FAISS optimization)
            if memory.embedding is None:
                similarity = memory.similarity_score  # Use pre-computed from FAISS
            else:
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

            # Phase 3: Graph-enhanced scoring (with caching)
            warnings = []

            # Get failure penalty with caching
            failure_penalty = self._get_failure_penalty(memory.action)

            # Get hypothesis confidence with caching
            hypothesis_confidence = self._get_hypothesis_confidence(
                memory.action, task_type or ""
            )

            # Add warnings for low confidence (not cached - rare case)
            if hypothesis_confidence < 0.2 and self.hypothesis_graph is not None and task_type:
                try:
                    graph_warnings = self.hypothesis_graph.get_low_confidence_warnings(
                        memory.action, task_type
                    )
                    warnings.extend(graph_warnings)
                except Exception:
                    pass  # Graceful degradation

            # Calculate adjusted score
            adjusted_score = combined * (1 - failure_penalty) * hypothesis_confidence

            results.append(
                RetrievalResult(
                    memory=memory,
                    similarity=similarity,
                    q_value=memory.q_value,
                    combined_score=combined,
                    failure_penalty=failure_penalty,
                    hypothesis_confidence=hypothesis_confidence,
                    adjusted_score=adjusted_score,
                    warnings=warnings,
                )
            )

        # Sort by adjusted score (descending)
        results.sort(key=lambda r: r.adjusted_score, reverse=True)

        # Return top-n
        return results[: self.config.top_n]

    def retrieve_for_routing(
        self,
        task_ir: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """Retrieve with graph enhancement for routing."""
        embedding = self.embedder.embed_task_ir(task_ir)
        task_type = task_ir.get("task_type", "general")
        return self._retrieve(embedding, action_type="routing", task_type=task_type)

    def retrieve_for_escalation(
        self,
        failure_context: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """Retrieve with graph enhancement for escalation."""
        embedding = self.embedder.embed_failure_context(failure_context)
        task_type = failure_context.get("task_type", "escalation")
        return self._retrieve(embedding, action_type="escalation", task_type=task_type)

    def retrieve_for_exploration(
        self,
        query: str,
        context_preview: str,
        task_type: str = "exploration",
    ) -> List[RetrievalResult]:
        """Retrieve with graph enhancement for exploration."""
        embedding = self.embedder.embed_exploration(query, context_preview)
        return self._retrieve(embedding, action_type="exploration", task_type=task_type)

    def get_best_action(
        self,
        results: List[RetrievalResult],
    ) -> Optional[Tuple[str, float, List[str]]]:
        """
        Get the best action with warnings.

        Args:
            results: Retrieval results

        Returns:
            (action, confidence, warnings) tuple or None if not confident
        """
        if not results:
            return None

        best = results[0]
        # Use adjusted_score for graph-enhanced, fall back to combined_score
        score = best.adjusted_score if best.adjusted_score > 0 else best.combined_score

        if score >= self.config.confidence_threshold:
            return (best.memory.action, score, best.warnings)

        return None

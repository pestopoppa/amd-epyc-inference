"""MemRL-backed classification retriever.

Uses TwoPhaseRetriever to find similar classification exemplars and
return the most confident classification based on Q-value weighted voting.

Falls back to keyword matching when:
- Not enough similar exemplars (< min_samples)
- Confidence below threshold
- MemRL components unavailable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from .types import ClassificationResult, RoutingDecision

if TYPE_CHECKING:
    from orchestration.repl_memory.retriever import RetrievalResult, TwoPhaseRetriever

logger = logging.getLogger(__name__)


@dataclass
class ClassificationConfig:
    """Configuration for classification retrieval."""

    min_samples: int = 3  # Minimum similar exemplars needed
    confidence_threshold: float = 0.6  # Minimum confidence to trust classification
    similarity_threshold: float = 0.4  # Minimum similarity for exemplars
    use_voting: bool = True  # Use weighted voting across exemplars


class ClassificationRetriever:
    """
    MemRL-backed classification using similarity to seeded exemplars.

    Classification strategy:
    1. Embed the prompt
    2. Retrieve similar classification exemplars from episodic memory
    3. Use Q-value weighted voting to determine classification
    4. Fall back to keyword matching if not confident

    The exemplars are seeded from classifier_config.yaml during system init.
    Each exemplar has:
    - prompt: Example prompt
    - action: Classification result (e.g., "coder_escalation", "summarization")
    - q_value: Learned confidence (0-1)
    """

    def __init__(
        self,
        retriever: "TwoPhaseRetriever",
        config: Optional[ClassificationConfig] = None,
    ):
        self.retriever = retriever
        self.config = config or ClassificationConfig()

    def classify_prompt(
        self,
        prompt: str,
        classification_type: str = "routing",
        fallback: Optional[Callable[[str], ClassificationResult]] = None,
    ) -> ClassificationResult:
        """
        Classify a prompt using MemRL similarity matching.

        Args:
            prompt: User prompt to classify.
            classification_type: Type of classification (routing, summarization, etc.).
            fallback: Optional fallback function for keyword matching.

        Returns:
            ClassificationResult with match details.
        """
        try:
            results = self.retriever.retrieve_for_classification(
                prompt, classification_type
            )
        except Exception as e:
            logger.debug("Classification retrieval failed: %s", e)
            if fallback:
                return fallback(prompt)
            return ClassificationResult(
                matched=False,
                matcher_name=classification_type,
                confidence=0.0,
                source="fallback",
            )

        # Check if we have enough confident samples
        if not self._should_use_learned(results):
            logger.debug(
                "Not enough confident samples for %s classification", classification_type
            )
            if fallback:
                return fallback(prompt)
            return ClassificationResult(
                matched=False,
                matcher_name=classification_type,
                confidence=0.0,
                source="fallback",
            )

        # Use Q-value weighted voting
        if self.config.use_voting:
            classification, confidence = self._vote_classification(results)
        else:
            # Just use the best match
            best = results[0]
            classification = best.memory.action
            confidence = best.combined_score

        return ClassificationResult(
            matched=True,
            matcher_name=classification_type,
            matched_keywords=[classification],
            confidence=confidence,
            source="memrl",
        )

    def classify_for_routing(
        self,
        prompt: str,
        context: str = "",
        has_image: bool = False,
        fallback: Optional[Callable[..., RoutingDecision]] = None,
    ) -> RoutingDecision:
        """
        Classify a prompt for routing decision.

        Args:
            prompt: User prompt.
            context: Optional context text.
            has_image: Whether the request includes an image.
            fallback: Optional fallback function for keyword routing.

        Returns:
            RoutingDecision with role, strategy, and confidence.
        """
        # Vision always routes to worker_vision
        if has_image:
            return RoutingDecision(
                role="worker_vision",
                strategy="classified",
                confidence=1.0,
            )

        try:
            results = self.retriever.retrieve_for_classification(prompt, "routing")
        except Exception as e:
            logger.debug("Routing classification failed: %s", e)
            if fallback:
                return fallback(prompt, context, has_image)
            return RoutingDecision(
                role="frontdoor",
                strategy="fallback",
                confidence=0.0,
            )

        if not self._should_use_learned(results):
            if fallback:
                return fallback(prompt, context, has_image)
            return RoutingDecision(
                role="frontdoor",
                strategy="rules",
                confidence=0.0,
            )

        # Aggregate votes by role
        role_votes: Dict[str, float] = {}
        for r in results:
            role = r.memory.action
            # Weight by combined score (similarity × Q-value)
            weight = r.combined_score
            role_votes[role] = role_votes.get(role, 0.0) + weight

        # Select role with highest weighted votes
        if role_votes:
            best_role = max(role_votes, key=role_votes.get)  # type: ignore
            confidence = role_votes[best_role] / sum(role_votes.values())
        else:
            best_role = "frontdoor"
            confidence = 0.0

        return RoutingDecision(
            role=best_role,
            strategy="learned",
            confidence=confidence,
            matched_keywords=[best_role],
        )

    def should_use_direct_mode(
        self,
        prompt: str,
        context: str = "",
        fallback: Optional[Callable[[str, str], bool]] = None,
    ) -> Tuple[bool, float]:
        """
        Determine if direct mode should be used via MemRL classification.

        Args:
            prompt: User prompt.
            context: Optional context text.
            fallback: Optional fallback function.

        Returns:
            (should_use_direct, confidence) tuple.
        """
        try:
            results = self.retriever.retrieve_for_classification(prompt, "mode")
        except Exception as e:
            logger.debug("Mode classification failed: %s", e)
            if fallback:
                return fallback(prompt, context), 0.0
            return True, 0.0  # Default to direct

        if not self._should_use_learned(results):
            if fallback:
                return fallback(prompt, context), 0.0
            return True, 0.0

        # Count votes for direct vs repl
        direct_score = 0.0
        repl_score = 0.0
        for r in results:
            action = r.memory.action.lower()
            weight = r.combined_score
            if "direct" in action:
                direct_score += weight
            elif "repl" in action or "react" in action:
                repl_score += weight

        total = direct_score + repl_score
        if total == 0:
            return True, 0.0

        use_direct = direct_score >= repl_score
        confidence = max(direct_score, repl_score) / total
        return use_direct, confidence

    def _should_use_learned(self, results: List["RetrievalResult"]) -> bool:
        """Determine if we should use learned classification."""
        if len(results) < self.config.min_samples:
            return False

        # Check that best result meets confidence threshold
        if results and results[0].combined_score < self.config.confidence_threshold:
            return False

        # Check that results have enough similarity
        similar_count = sum(
            1 for r in results if r.similarity >= self.config.similarity_threshold
        )
        if similar_count < self.config.min_samples:
            return False

        return True

    def _vote_classification(
        self, results: List["RetrievalResult"]
    ) -> Tuple[str, float]:
        """
        Use Q-value weighted voting to determine classification.

        Each exemplar votes for its classification with weight = combined_score.
        Returns the classification with highest total weight.
        """
        votes: Dict[str, float] = {}
        for r in results:
            classification = r.memory.action
            weight = r.combined_score
            votes[classification] = votes.get(classification, 0.0) + weight

        if not votes:
            return "", 0.0

        best = max(votes, key=votes.get)  # type: ignore
        total_weight = sum(votes.values())
        confidence = votes[best] / total_weight if total_weight > 0 else 0.0

        return best, confidence


# Singleton instance (lazy initialized)
_classification_retriever: Optional[ClassificationRetriever] = None


def get_classification_retriever() -> Optional[ClassificationRetriever]:
    """Get the singleton ClassificationRetriever instance.

    Returns None if MemRL components are not available.
    """
    global _classification_retriever
    if _classification_retriever is not None:
        return _classification_retriever

    try:
        from orchestration.repl_memory.embedder import TaskEmbedder
        from orchestration.repl_memory.episodic_store import EpisodicStore
        from orchestration.repl_memory.retriever import TwoPhaseRetriever

        # Initialize with default paths
        embedder = TaskEmbedder()
        store = EpisodicStore()
        retriever = TwoPhaseRetriever(store, embedder)
        _classification_retriever = ClassificationRetriever(retriever)
        logger.info("ClassificationRetriever initialized successfully")
        return _classification_retriever
    except Exception as e:
        logger.debug("Failed to initialize ClassificationRetriever: %s", e)
        return None

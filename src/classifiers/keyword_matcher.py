"""Config-driven keyword matching for classification.

Replaces hardcoded keyword lists throughout the codebase with
externalized configuration from classifier_config.yaml.
"""

from __future__ import annotations

import logging

from src.classifiers.types import ClassificationResult, MatcherConfig, RoutingDecision
from src.classifiers.config_loader import get_classifier_config

logger = logging.getLogger(__name__)


class KeywordMatcher:
    """Config-driven keyword matcher.

    Loads keyword lists from classifier_config.yaml and provides
    efficient substring matching with optional normalization.

    Usage:
        matcher = KeywordMatcher.from_config("summarization")
        if matcher.matches("Please summarize this document"):
            # Handle summarization

        # Or get detailed result
        result = matcher.classify("Give me a TL;DR")
        if result:
            print(f"Matched: {result.matched_keywords}")
    """

    def __init__(self, config: MatcherConfig):
        """Initialize with a matcher configuration.

        Args:
            config: MatcherConfig with keywords, patterns, and options.
        """
        self.config = config
        self._keywords = config.keywords or []
        self._patterns = config.patterns or []
        self._case_sensitive = config.case_sensitive
        self._normalize = config.normalize

    @classmethod
    def from_config(cls, matcher_name: str) -> "KeywordMatcher":
        """Create a matcher from the classifier config.

        Args:
            matcher_name: Name of the matcher in keyword_matchers section.

        Returns:
            Configured KeywordMatcher instance.

        Raises:
            KeyError: If matcher_name not found in config.
        """
        config = get_classifier_config()
        matchers = config.get("keyword_matchers", {})

        if matcher_name not in matchers:
            raise KeyError(f"Matcher '{matcher_name}' not found in classifier config")

        matcher_data = matchers[matcher_name]
        return cls(
            MatcherConfig(
                name=matcher_name,
                keywords=matcher_data.get("keywords", []),
                patterns=matcher_data.get("patterns", []),
                case_sensitive=matcher_data.get("case_sensitive", False),
                normalize=matcher_data.get("normalize", False),
            )
        )

    def _prepare_text(self, text: str) -> str:
        """Prepare text for matching based on configuration."""
        if self._normalize:
            return text.strip().rstrip(".").lower()
        elif not self._case_sensitive:
            return text.lower()
        return text

    def _prepare_keyword(self, keyword: str) -> str:
        """Prepare keyword for matching based on configuration."""
        if self._normalize or not self._case_sensitive:
            return keyword.lower()
        return keyword

    def matches(self, text: str) -> bool:
        """Check if text matches any keyword/pattern.

        Args:
            text: Text to check.

        Returns:
            True if any keyword/pattern matches.
        """
        prepared = self._prepare_text(text)

        # Check keywords (substring match)
        for kw in self._keywords:
            if self._prepare_keyword(kw) in prepared:
                return True

        # Check patterns (prefix match when normalized to avoid partial word matches)
        for pattern in self._patterns:
            prepared_pattern = self._prepare_keyword(pattern)
            if self._normalize:
                # For normalized patterns, use prefix match to catch "Analysis complete..."
                # but not "incomplete" matching "complete"
                if prepared.startswith(prepared_pattern):
                    return True
            elif prepared_pattern in prepared:
                return True

        return False

    def classify(self, text: str) -> ClassificationResult:
        """Classify text and return detailed result.

        Args:
            text: Text to classify.

        Returns:
            ClassificationResult with match details.
        """
        prepared = self._prepare_text(text)
        matched_keywords = []

        # Find all matching keywords
        for kw in self._keywords:
            if self._prepare_keyword(kw) in prepared:
                matched_keywords.append(kw)

        # Find all matching patterns (prefix match when normalized)
        for pattern in self._patterns:
            prepared_pattern = self._prepare_keyword(pattern)
            if self._normalize:
                # For normalized patterns, use prefix match to catch "Analysis complete..."
                # but not "incomplete" matching "complete"
                if prepared.startswith(prepared_pattern):
                    matched_keywords.append(pattern)
            elif prepared_pattern in prepared:
                matched_keywords.append(pattern)

        return ClassificationResult(
            matched=len(matched_keywords) > 0,
            matcher_name=self.config.name,
            matched_keywords=matched_keywords,
            confidence=1.0,
            source="keyword",
        )


# ============================================================================
# Convenience Functions (drop-in replacements for original functions)
# ============================================================================


def is_summarization_task(prompt: str) -> bool:
    """Detect if the prompt is a summarization task.

    Drop-in replacement for chat_summarization._is_summarization_task().

    Args:
        prompt: The user's prompt.

    Returns:
        True if this looks like a summarization request.
    """
    try:
        matcher = KeywordMatcher.from_config("summarization")
        return matcher.matches(prompt)
    except KeyError:
        # Fallback to hardcoded keywords if config missing
        keywords = [
            "summarize", "summary", "summarise", "summarisation",
            "executive summary", "overview", "key points", "main ideas",
            "tl;dr", "tldr", "synopsis",
        ]
        prompt_lower = prompt.lower()
        return any(kw in prompt_lower for kw in keywords)


def is_coding_task(prompt: str) -> bool:
    """Determine if a prompt is primarily a coding task.

    Drop-in replacement for chat_routing._is_coding_task().

    Args:
        prompt: The user's prompt.

    Returns:
        True if the task is coding-related.
    """
    try:
        matcher = KeywordMatcher.from_config("coding_task")
        return matcher.matches(prompt)
    except KeyError:
        # Fallback to hardcoded keywords if config missing
        keywords = [
            "implement", "code", "function", "class ", "method",
            "debug", "refactor", "bug", "error", "exception",
            "compile", "syntax", "algorithm", "data structure",
            "api", "endpoint", "database", "query", "sql",
            "test", "unit test", "integration",
        ]
        prompt_lower = prompt.lower()
        return any(kw in prompt_lower for kw in keywords)


def is_stub_final(text: str) -> bool:
    """Detect when FINAL() arg is a stub pointing to printed output.

    Drop-in replacement for chat_utils._is_stub_final().

    Args:
        text: The FINAL() argument text.

    Returns:
        True if the text is a stub pattern.
    """
    try:
        matcher = KeywordMatcher.from_config("stub_final")
        return matcher.matches(text)
    except KeyError:
        # Fallback to hardcoded patterns if config missing
        patterns = {
            "complete", "see above", "analysis complete",
            "estimation complete", "done", "finished",
            "see results above", "see output above",
            "see structured output above",
            "see integrated results above",
            "see the structured output above",
        }
        normalized = text.strip().rstrip(".").lower()
        return any(p in normalized for p in patterns)


def needs_structured_analysis(prompt: str) -> bool:
    """Detect if a vision prompt needs full structured analysis beyond OCR.

    Drop-in replacement for chat_vision._needs_structured_analysis().

    Args:
        prompt: The user's vision prompt.

    Returns:
        True if structured analysis should complement OCR.
    """
    try:
        matcher = KeywordMatcher.from_config("structured_analysis")
        return matcher.matches(prompt)
    except KeyError:
        # Fallback to hardcoded keywords if config missing
        keywords = [
            "analyze", "architecture", "diagram", "protocol",
            "economic model", "security audit", "security analysis",
            "whitepaper", "smart contract", "incentive",
            "trust assumption", "attack vector", "forensic",
            "entity extraction", "business relationship",
            "flow chart", "flowchart", "sequence diagram",
            "system design", "data flow", "state machine",
        ]
        prompt_lower = prompt.lower()
        return any(kw in prompt_lower for kw in keywords)


def _keyword_based_direct_mode(prompt: str, context: str, direct_config: dict) -> bool:
    """Keyword-based direct mode detection (fallback)."""
    # Keep REPL for large contexts
    threshold = direct_config.get("context_threshold", 20000)
    if context and len(context) > threshold:
        return False

    # Keep REPL when prompt explicitly needs file/tool operations
    repl_indicators = direct_config.get("repl_indicators", [
        "read the file", "list files", "list the files",
        "look at the file", "open the file", "read from",
        "write to", "save to", "execute", "run the", "run this",
        "search the codebase", "find in the", "grep for",
        "explore the", "scan the",
    ])

    prompt_lower = prompt.lower()
    if any(ind in prompt_lower for ind in repl_indicators):
        return False

    # Direct mode for everything else
    return True


def should_use_direct_mode(prompt: str, context: str = "") -> bool:
    """Decide if the prompt should bypass the REPL.

    Drop-in replacement for chat_routing._should_use_direct_mode().
    Uses MemRL classification when enabled, falls back to keyword matching.

    Args:
        prompt: The user's prompt.
        context: Optional context text.

    Returns:
        True if direct mode should be used.
    """
    config = get_classifier_config()
    direct_config = config.get("routing_classifiers", {}).get("direct_mode", {})

    # Check if MemRL is enabled for mode classification
    use_memrl = direct_config.get("use_memrl", False)

    if use_memrl:
        try:
            from src.classifiers.classification_retriever import get_classification_retriever

            retriever = get_classification_retriever()
            if retriever is not None:
                use_direct, confidence = retriever.should_use_direct_mode(
                    prompt,
                    context,
                    fallback=lambda p, c: _keyword_based_direct_mode(p, c, direct_config),
                )
                if confidence >= 0.5:  # Only trust if reasonably confident
                    logger.debug(
                        "MemRL mode classification: direct=%s, confidence=%.2f",
                        use_direct, confidence
                    )
                    return use_direct
        except Exception as e:
            logger.debug("MemRL mode classification failed, using keywords: %s", e)

    # Keyword-based fallback
    return _keyword_based_direct_mode(prompt, context, direct_config)


def _keyword_based_routing(
    prompt: str,
    routing_config: dict,
) -> RoutingDecision:
    """Keyword-based routing (fallback)."""
    from src.roles import Role

    categories = routing_config.get("categories", {})
    prompt_lower = prompt.lower()

    # Check each category in priority order
    for role_name, category_config in categories.items():
        keywords = category_config.get("keywords", [])
        matched = [kw for kw in keywords if kw.lower() in prompt_lower]
        if matched:
            # Map category name to Role enum
            role_map = {
                "coder_primary": str(Role.CODER_PRIMARY),
                "coder_escalation": str(Role.CODER_ESCALATION),
                "architect_general": str(Role.ARCHITECT_GENERAL),
            }
            role = role_map.get(role_name, str(Role.FRONTDOOR))
            return RoutingDecision(
                role=role,
                strategy="classified",
                confidence=1.0,
                matched_keywords=matched,
            )

    # Default: frontdoor
    return RoutingDecision(
        role=str(Role.FRONTDOOR),
        strategy="rules",
        confidence=1.0,
    )


def classify_and_route(
    prompt: str,
    context: str = "",
    has_image: bool = False,
) -> RoutingDecision:
    """Classify prompt intent and route to the best specialist.

    Drop-in replacement for chat_routing._classify_and_route().
    Uses MemRL classification when enabled, falls back to keyword matching.

    Args:
        prompt: The user's prompt.
        context: Optional context text.
        has_image: Whether the request includes an image.

    Returns:
        RoutingDecision with role, strategy, and matched keywords.
    """
    from src.features import features
    from src.roles import Role

    # Vision: has image data
    if has_image:
        return RoutingDecision(
            role="worker_vision",
            strategy="classified",
            confidence=1.0,
        )

    # Specialist routing (gated behind feature flag)
    if not features().specialist_routing:
        return RoutingDecision(
            role=str(Role.FRONTDOOR),
            strategy="rules",
            confidence=1.0,
        )

    config = get_classifier_config()
    routing_config = config.get("routing_classifiers", {}).get(
        "specialist_routing", {}
    )

    # Check if MemRL is enabled for routing classification
    use_memrl = routing_config.get("use_memrl", False)

    if use_memrl:
        try:
            from src.classifiers.classification_retriever import get_classification_retriever

            retriever = get_classification_retriever()
            if retriever is not None:
                decision = retriever.classify_for_routing(
                    prompt,
                    context,
                    has_image,
                    fallback=lambda p, c, i: _keyword_based_routing(p, routing_config),
                )
                if decision.confidence >= 0.5:  # Only trust if reasonably confident
                    logger.debug(
                        "MemRL routing: role=%s, confidence=%.2f",
                        decision.role, decision.confidence
                    )
                    return decision
        except Exception as e:
            logger.debug("MemRL routing failed, using keywords: %s", e)

    # Keyword-based fallback
    return _keyword_based_routing(prompt, routing_config)


def classify_and_route_tuple(
    prompt: str,
    context: str = "",
    has_image: bool = False,
) -> tuple[str, str]:
    """Classify and route, returning tuple for backward compatibility.

    This maintains the original function signature of _classify_and_route().

    Args:
        prompt: The user's prompt.
        context: Optional context text.
        has_image: Whether the request includes an image.

    Returns:
        Tuple of (role_name, routing_strategy).
    """
    decision = classify_and_route(prompt, context, has_image)
    return decision.role, decision.strategy

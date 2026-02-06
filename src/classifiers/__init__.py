"""Classifiers module for externalized keyword matching and routing.

This module provides config-driven classifiers that replace hardcoded
keyword matching functions throughout the codebase.

Usage:
    from src.classifiers import KeywordMatcher, get_classifier_config

    # Get a specific matcher
    matcher = KeywordMatcher.from_config("summarization")
    if matcher.matches("Please summarize this document"):
        # Handle summarization task

    # Or use convenience functions
    from src.classifiers import is_summarization_task, is_coding_task

    # Or use MemRL-backed classification (when enabled)
    from src.classifiers import get_classification_retriever
    retriever = get_classification_retriever()
    if retriever:
        decision = retriever.classify_for_routing(prompt, context)

Public API:
    - KeywordMatcher: Config-driven keyword matching
    - ClassificationResult: Typed result from classification
    - RoutingDecision: Typed result from routing classification
    - ClassificationRetriever: MemRL-backed classification
    - get_classification_retriever: Get singleton ClassificationRetriever
    - get_classifier_config: Get loaded classifier config
    - is_summarization_task: Check if prompt is summarization
    - is_coding_task: Check if prompt is coding-related
    - is_stub_final: Check if FINAL() arg is a stub
    - needs_structured_analysis: Check if vision prompt needs structured analysis
    - should_use_direct_mode: Check if prompt should bypass REPL
    - classify_and_route: Classify prompt and return routing decision
"""

from src.classifiers.types import ClassificationResult, RoutingDecision
from src.classifiers.config_loader import get_classifier_config
from src.classifiers.keyword_matcher import (
    KeywordMatcher,
    is_summarization_task,
    is_coding_task,
    is_stub_final,
    needs_structured_analysis,
    should_use_direct_mode,
    classify_and_route,
)

# Lazy import for MemRL components (may not be available)
def get_classification_retriever():
    """Get the singleton ClassificationRetriever instance.

    Returns None if MemRL components are not available.
    """
    try:
        from src.classifiers.classification_retriever import (
            get_classification_retriever as _get_retriever,
        )
        return _get_retriever()
    except ImportError:
        return None


__all__ = [
    "ClassificationResult",
    "RoutingDecision",
    "KeywordMatcher",
    "get_classifier_config",
    "get_classification_retriever",
    "is_summarization_task",
    "is_coding_task",
    "is_stub_final",
    "needs_structured_analysis",
    "should_use_direct_mode",
    "classify_and_route",
]

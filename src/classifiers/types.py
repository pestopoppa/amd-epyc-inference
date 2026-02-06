"""Type definitions for the classifiers module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClassificationResult:
    """Result from a classification operation.

    Attributes:
        matched: Whether the classification matched.
        matcher_name: Name of the matcher that produced this result.
        matched_keywords: Keywords that triggered the match.
        confidence: Confidence score (1.0 for keyword match, varies for MemRL).
        source: Source of the match ("keyword", "memrl", "fallback").
    """

    matched: bool
    matcher_name: str = ""
    matched_keywords: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "keyword"

    def __bool__(self) -> bool:
        """Allow using result directly in if statements."""
        return self.matched


@dataclass
class RoutingDecision:
    """Result from routing classification.

    Attributes:
        role: The selected role for routing.
        strategy: How the decision was made ("classified", "rules", "memrl").
        confidence: Confidence in the routing decision.
        matched_keywords: Keywords that influenced the decision.
    """

    role: str
    strategy: str = "rules"
    confidence: float = 1.0
    matched_keywords: list[str] = field(default_factory=list)


@dataclass
class MatcherConfig:
    """Configuration for a single keyword matcher.

    Loaded from classifier_config.yaml.
    """

    name: str
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    case_sensitive: bool = False
    normalize: bool = False  # For stub patterns: strip, lowercase, remove trailing .

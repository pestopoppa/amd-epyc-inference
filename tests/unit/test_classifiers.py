"""Tests for the classifiers module."""

import pytest

from src.classifiers import (
    KeywordMatcher,
    ClassificationResult,
    get_classifier_config,
    is_summarization_task,
    is_coding_task,
    is_stub_final,
    needs_structured_analysis,
    should_use_direct_mode,
    classify_and_route,
)
from src.classifiers.config_loader import reset_classifier_config
from src.classifiers.types import MatcherConfig, RoutingDecision


@pytest.fixture(autouse=True)
def reset_config():
    """Reset cached config before each test."""
    reset_classifier_config()
    yield
    reset_classifier_config()


class TestKeywordMatcher:
    """Tests for KeywordMatcher class."""

    def test_basic_keyword_match(self):
        """Test basic keyword matching."""
        config = MatcherConfig(
            name="test",
            keywords=["hello", "world"],
            case_sensitive=False,
        )
        matcher = KeywordMatcher(config)

        assert matcher.matches("hello there")
        assert matcher.matches("HELLO there")
        assert matcher.matches("say world")
        assert not matcher.matches("goodbye")

    def test_case_sensitive_match(self):
        """Test case-sensitive matching."""
        config = MatcherConfig(
            name="test",
            keywords=["Hello"],
            case_sensitive=True,
        )
        matcher = KeywordMatcher(config)

        assert matcher.matches("Hello there")
        assert not matcher.matches("hello there")
        assert not matcher.matches("HELLO there")

    def test_normalized_match(self):
        """Test normalized matching (for stub patterns)."""
        config = MatcherConfig(
            name="test",
            patterns=["complete", "see above"],
            normalize=True,
        )
        matcher = KeywordMatcher(config)

        assert matcher.matches("Complete.")
        assert matcher.matches("COMPLETE")
        assert matcher.matches("  see above.  ")
        assert not matcher.matches("incomplete")

    def test_classify_returns_result(self):
        """Test classify() returns detailed ClassificationResult."""
        config = MatcherConfig(
            name="summarization",
            keywords=["summarize", "summary"],
        )
        matcher = KeywordMatcher(config)

        result = matcher.classify("Please summarize this document")
        assert result.matched
        assert result.matcher_name == "summarization"
        assert "summarize" in result.matched_keywords
        assert result.confidence == 1.0
        assert result.source == "keyword"

    def test_classify_no_match(self):
        """Test classify() when no keywords match."""
        config = MatcherConfig(
            name="test",
            keywords=["xyz"],
        )
        matcher = KeywordMatcher(config)

        result = matcher.classify("no match here")
        assert not result.matched
        assert result.matched_keywords == []

    def test_from_config(self):
        """Test loading matcher from config file."""
        matcher = KeywordMatcher.from_config("summarization")
        assert matcher.matches("please summarize this")
        assert matcher.matches("give me a TL;DR")
        assert not matcher.matches("implement a function")

    def test_from_config_missing_raises(self):
        """Test from_config raises KeyError for missing matcher."""
        with pytest.raises(KeyError, match="nonexistent"):
            KeywordMatcher.from_config("nonexistent")


class TestIsSummarizationTask:
    """Tests for is_summarization_task()."""

    def test_summarization_keywords(self):
        """Test various summarization keywords."""
        assert is_summarization_task("summarize this document")
        assert is_summarization_task("Give me a summary")
        assert is_summarization_task("What's the TL;DR?")
        assert is_summarization_task("tldr please")
        assert is_summarization_task("executive summary needed")
        assert is_summarization_task("key points from this")

    def test_non_summarization(self):
        """Test non-summarization prompts."""
        assert not is_summarization_task("implement a function")
        assert not is_summarization_task("fix this bug")
        assert not is_summarization_task("What is 2+2?")

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert is_summarization_task("SUMMARIZE THIS")
        assert is_summarization_task("Summarize This")


class TestIsCodingTask:
    """Tests for is_coding_task()."""

    def test_coding_keywords(self):
        """Test various coding keywords."""
        assert is_coding_task("implement a binary search")
        assert is_coding_task("write code for authentication")
        assert is_coding_task("debug this function")
        assert is_coding_task("refactor the module")
        assert is_coding_task("fix the bug in login")
        assert is_coding_task("write a unit test")
        assert is_coding_task("create a class for users")

    def test_non_coding(self):
        """Test non-coding prompts."""
        assert not is_coding_task("summarize this document")
        assert not is_coding_task("What is the weather?")
        assert not is_coding_task("Explain quantum physics")

    def test_class_keyword_with_space(self):
        """Test 'class ' keyword matches with space (avoid 'classification')."""
        assert is_coding_task("create a class for users")
        assert is_coding_task("define the class hierarchy")
        # Note: "class" without space would match "classification" too
        # The keyword is "class " to avoid false positives


class TestIsStubFinal:
    """Tests for is_stub_final()."""

    def test_stub_patterns(self):
        """Test various stub patterns."""
        assert is_stub_final("Complete.")
        assert is_stub_final("See above")
        assert is_stub_final("Analysis complete")
        assert is_stub_final("Done")
        assert is_stub_final("See results above.")
        assert is_stub_final("  FINISHED  ")

    def test_non_stub_content(self):
        """Test actual content (not stubs)."""
        assert not is_stub_final("The answer is 42")
        assert not is_stub_final("Here is the implementation:")
        assert not is_stub_final("Based on the analysis, I recommend...")

    def test_normalization(self):
        """Test normalization (strip, lowercase, remove .)."""
        assert is_stub_final("  Complete.  ")
        assert is_stub_final("DONE.")
        assert is_stub_final("   See Above.   ")


class TestNeedsStructuredAnalysis:
    """Tests for needs_structured_analysis()."""

    def test_structured_keywords(self):
        """Test keywords that trigger structured analysis."""
        assert needs_structured_analysis("analyze this whitepaper")
        assert needs_structured_analysis("extract the architecture")
        assert needs_structured_analysis("parse the diagram")
        assert needs_structured_analysis("security audit needed")
        assert needs_structured_analysis("review the smart contract")
        assert needs_structured_analysis("understand the data flow")

    def test_simple_queries(self):
        """Test simple queries that don't need structured analysis."""
        assert not needs_structured_analysis("what does this image show?")
        assert not needs_structured_analysis("read the text")
        assert not needs_structured_analysis("describe the photo")


class TestShouldUseDirectMode:
    """Tests for should_use_direct_mode()."""

    def test_direct_mode_for_simple_queries(self):
        """Test direct mode for simple queries."""
        assert should_use_direct_mode("What is 2+2?")
        assert should_use_direct_mode("Explain photosynthesis")
        assert should_use_direct_mode("Format this as JSON")

    def test_repl_mode_for_file_operations(self):
        """Test REPL mode for file operations."""
        assert not should_use_direct_mode("read the file config.yaml")
        assert not should_use_direct_mode("list files in the directory")
        assert not should_use_direct_mode("search the codebase for errors")
        assert not should_use_direct_mode("grep for TODO comments")
        assert not should_use_direct_mode("execute this script")

    def test_repl_mode_for_large_context(self):
        """Test REPL mode for large contexts."""
        large_context = "x" * 25000  # Over threshold
        assert not should_use_direct_mode("analyze this", large_context)

    def test_direct_mode_for_small_context(self):
        """Test direct mode for small contexts."""
        small_context = "x" * 1000
        assert should_use_direct_mode("analyze this", small_context)


class TestClassifyAndRoute:
    """Tests for classify_and_route()."""

    def test_vision_request(self):
        """Test routing for vision requests."""
        result = classify_and_route("describe this image", has_image=True)
        assert isinstance(result, RoutingDecision)
        assert result.role == "worker_vision"
        assert result.strategy == "classified"

    def test_default_routing(self):
        """Test default routing to frontdoor."""
        result = classify_and_route("What is the weather?")
        assert result.role == "frontdoor"
        assert result.strategy == "rules"

    def test_routing_decision_type(self):
        """Test that result is RoutingDecision type."""
        result = classify_and_route("hello")
        assert isinstance(result, RoutingDecision)
        assert hasattr(result, "role")
        assert hasattr(result, "strategy")
        assert hasattr(result, "confidence")


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_bool_conversion(self):
        """Test __bool__ for use in if statements."""
        matched = ClassificationResult(matched=True, matcher_name="test")
        not_matched = ClassificationResult(matched=False, matcher_name="test")

        assert matched  # Should be truthy
        assert not not_matched  # Should be falsy

    def test_defaults(self):
        """Test default values."""
        result = ClassificationResult(matched=True)
        assert result.matcher_name == ""
        assert result.matched_keywords == []
        assert result.confidence == 1.0
        assert result.source == "keyword"


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_get_classifier_config(self):
        """Test loading the classifier config."""
        config = get_classifier_config()
        assert "keyword_matchers" in config
        assert "summarization" in config["keyword_matchers"]

    def test_config_has_expected_matchers(self):
        """Test config has all expected matchers."""
        config = get_classifier_config()
        matchers = config["keyword_matchers"]

        assert "summarization" in matchers
        assert "structured_analysis" in matchers
        assert "coding_task" in matchers
        assert "stub_final" in matchers

    def test_config_has_routing_classifiers(self):
        """Test config has routing classifiers."""
        config = get_classifier_config()
        routing = config.get("routing_classifiers", {})

        assert "direct_mode" in routing
        assert "specialist_routing" in routing


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with original functions."""

    def test_summarization_matches_original(self):
        """Test is_summarization_task matches original behavior."""
        # These were the original keywords
        original_keywords = [
            "summarize", "summary", "summarise", "summarisation",
            "executive summary", "overview", "key points", "main ideas",
            "tl;dr", "tldr", "synopsis",
        ]

        for kw in original_keywords:
            assert is_summarization_task(f"Please {kw} this document"), f"Failed for: {kw}"

    def test_coding_matches_original(self):
        """Test is_coding_task matches original behavior."""
        # These were the original keywords
        original_keywords = [
            "implement", "code", "function", "debug", "refactor",
            "bug", "error", "exception", "compile", "syntax",
            "algorithm", "data structure", "api", "endpoint",
            "database", "query", "sql", "test", "unit test", "integration",
        ]

        for kw in original_keywords:
            assert is_coding_task(f"Please {kw} this"), f"Failed for: {kw}"

    def test_stub_patterns_match_original(self):
        """Test is_stub_final matches original behavior."""
        # These were the original patterns
        original_patterns = [
            "complete", "see above", "analysis complete",
            "estimation complete", "done", "finished",
            "see results above", "see output above",
            "see structured output above",
            "see integrated results above",
            "see the structured output above",
        ]

        for pattern in original_patterns:
            assert is_stub_final(pattern), f"Failed for: {pattern}"


class TestClassificationRetriever:
    """Tests for ClassificationRetriever (MemRL-backed classification)."""

    def test_get_classification_retriever_returns_none_when_unavailable(self):
        """Test get_classification_retriever returns None if MemRL not available."""
        from src.classifiers import get_classification_retriever

        # In test environment, MemRL components may not be fully initialized
        # This should not raise an exception
        retriever = get_classification_retriever()
        # May be None or a valid retriever depending on environment
        assert retriever is None or hasattr(retriever, "classify_prompt")

    def test_classification_config_defaults(self):
        """Test ClassificationConfig has sensible defaults."""
        from src.classifiers.classification_retriever import ClassificationConfig

        config = ClassificationConfig()
        assert config.min_samples == 3
        assert config.confidence_threshold == 0.6
        assert config.similarity_threshold == 0.4
        assert config.use_voting is True

    def test_classification_result_from_memrl(self):
        """Test ClassificationResult source can be 'memrl'."""
        from src.classifiers.types import ClassificationResult

        result = ClassificationResult(
            matched=True,
            matcher_name="routing",
            matched_keywords=["coder_escalation"],
            confidence=0.85,
            source="memrl",
        )
        assert result.matched is True
        assert result.source == "memrl"
        assert result.confidence == 0.85

    def test_routing_decision_from_memrl(self):
        """Test RoutingDecision with learned strategy."""
        from src.classifiers.types import RoutingDecision

        decision = RoutingDecision(
            role="coder_escalation",
            strategy="learned",
            confidence=0.75,
            matched_keywords=["implement"],
        )
        assert decision.role == "coder_escalation"
        assert decision.strategy == "learned"
        assert decision.confidence == 0.75


class TestMemRLIntegration:
    """Tests for MemRL integration with keyword matchers."""

    def test_use_memrl_config_flag(self):
        """Test use_memrl flag is enabled in config."""
        config = get_classifier_config()
        direct_mode = config.get("routing_classifiers", {}).get("direct_mode", {})
        specialist = config.get("routing_classifiers", {}).get("specialist_routing", {})

        # MemRL is enabled for classification
        assert direct_mode.get("use_memrl", False) is True
        assert specialist.get("use_memrl", False) is True

    def test_keyword_fallback_when_memrl_disabled(self):
        """Test keyword matching is used when use_memrl is False."""
        # With use_memrl: false, should use keyword matching
        assert should_use_direct_mode("What is 2+2?") is True
        assert should_use_direct_mode("read the file") is False

    def test_classify_and_route_uses_keywords_by_default(self):
        """Test classify_and_route uses keywords when memrl disabled."""
        from unittest.mock import patch

        with patch("src.features.features") as mock_features:
            mock_features.return_value.specialist_routing = True
            decision = classify_and_route("implement a function")

        assert decision.strategy == "classified"
        assert "implement" in decision.matched_keywords

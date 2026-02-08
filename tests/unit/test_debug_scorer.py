#!/usr/bin/env python3
"""Tests for debug_scorer — deterministic scoring functions."""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "benchmark"))

from debug_scorer import (
    score_answer,
    score_batch,
    _extract_answer,
    _extract_code_block,
    _is_valid_json,
    _normalize_text,
    _score_exact_match,
    _score_f1,
    _score_multiple_choice,
    _score_programmatic,
    _score_substring,
)


# ---------------------------------------------------------------------------
# score_answer (dispatch + think-tag stripping)
# ---------------------------------------------------------------------------

class TestScoreAnswer:
    def test_exact_match_pass(self):
        assert score_answer("The answer is #### 42", "42", "exact_match") is True

    def test_exact_match_fail(self):
        assert score_answer("The answer is #### 99", "42", "exact_match") is False

    def test_empty_answer(self):
        assert score_answer("", "42", "exact_match") is False

    def test_whitespace_only(self):
        assert score_answer("   ", "42", "exact_match") is False

    def test_think_tag_stripped(self):
        answer = "<think>reasoning here</think>The answer is #### 42"
        assert score_answer(answer, "42", "exact_match") is True

    def test_think_tag_only_returns_false(self):
        assert score_answer("<think>only thinking</think>", "42", "exact_match") is False

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown scoring method"):
            score_answer("x", "x", "nonexistent_method")


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_numeric_match(self):
        assert _score_exact_match("#### 42", "42", {}) is True

    def test_numeric_with_comma(self):
        assert _score_exact_match("#### 1,234", "1234", {}) is True

    def test_float_match(self):
        assert _score_exact_match("#### 3.14", "3.14", {}) is True

    def test_fallback_to_last_line(self):
        assert _score_exact_match("reasoning\n42", "42", {}) is True

    def test_custom_pattern(self):
        config = {"extract_pattern": r"ANSWER:\s*(\S+)"}
        assert _score_exact_match("ANSWER: yes", "yes", config) is True

    def test_case_insensitive_default(self):
        assert _score_exact_match("#### Yes", "yes", {}) is True


# ---------------------------------------------------------------------------
# multiple_choice
# ---------------------------------------------------------------------------

class TestMultipleChoice:
    def test_answer_is_pattern(self):
        assert _score_multiple_choice("The answer is B", "B", {}) is True

    def test_letter_at_end(self):
        assert _score_multiple_choice("I choose A", "A", {}) is True

    def test_bold_letter(self):
        assert _score_multiple_choice("The correct answer is **C**", "C", {}) is True

    def test_wrong_letter(self):
        assert _score_multiple_choice("Answer: A", "B", {}) is False

    def test_invalid_expected(self):
        assert _score_multiple_choice("A", "42", {}) is False

    def test_standalone_letter(self):
        assert _score_multiple_choice("D", "D", {}) is True


# ---------------------------------------------------------------------------
# substring
# ---------------------------------------------------------------------------

class TestSubstring:
    def test_case_insensitive(self):
        assert _score_substring("The Capital is Paris", "paris", {}) is True

    def test_case_sensitive(self):
        config = {"case_sensitive": True}
        assert _score_substring("The Capital is Paris", "paris", config) is False
        assert _score_substring("The Capital is Paris", "Paris", config) is True

    def test_not_found(self):
        assert _score_substring("Hello world", "goodbye", {}) is False


# ---------------------------------------------------------------------------
# f1
# ---------------------------------------------------------------------------

class TestF1:
    def test_perfect_match(self):
        assert _score_f1("Paris", "Paris", {}) is True

    def test_partial_overlap(self):
        # "the city of paris" vs "paris france" — some overlap
        assert _score_f1("the city of paris", "paris france", {"threshold": 0.3}) is True

    def test_no_overlap(self):
        assert _score_f1("hello world", "goodbye moon", {}) is False

    def test_empty_gold(self):
        # Empty gold tokens — prediction must also be empty
        assert _score_f1("something", "", {}) is False


# ---------------------------------------------------------------------------
# programmatic (IFEval-style)
# ---------------------------------------------------------------------------

class TestProgrammatic:
    def test_word_count_min(self):
        long_answer = " ".join(["word"] * 50)
        assert _score_programmatic(long_answer, "", {"verifier": "word_count_min", "threshold": 10}) is True

    def test_word_count_min_fail(self):
        assert _score_programmatic("short", "", {"verifier": "word_count_min", "threshold": 10}) is False

    def test_contains_keyword(self):
        assert _score_programmatic("The quick brown fox", "", {"verifier": "contains_keyword", "keyword": "fox"}) is True

    def test_no_keyword(self):
        assert _score_programmatic("The quick brown fox", "", {"verifier": "no_keyword", "keyword": "cat"}) is True

    def test_json_valid(self):
        assert _score_programmatic('{"key": "value"}', "", {"verifier": "json_valid"}) is True

    def test_json_invalid(self):
        assert _score_programmatic("not json", "", {"verifier": "json_valid"}) is False

    def test_all_uppercase(self):
        assert _score_programmatic("HELLO WORLD", "", {"verifier": "all_uppercase"}) is True
        assert _score_programmatic("Hello World", "", {"verifier": "all_uppercase"}) is False

    def test_bullet_list(self):
        answer = "- item 1\n- item 2\n- item 3"
        assert _score_programmatic(answer, "", {"verifier": "bullet_list"}) is True

    def test_starts_with(self):
        assert _score_programmatic("Dear Sir,\nBody", "", {"verifier": "starts_with", "text": "Dear"}) is True

    def test_word_count_relation_at_most(self):
        assert _score_programmatic("one two three", "", {"verifier": "word_count", "count": 5, "relation": "at_most"}) is True
        assert _score_programmatic("one two three", "", {"verifier": "word_count", "count": 2, "relation": "at_most"}) is False

    def test_non_empty(self):
        assert _score_programmatic("something", "", {"verifier": "non_empty"}) is True

    def test_unknown_verifier_falls_back(self):
        # Unknown verifier checks if expected is substring
        assert _score_programmatic("The answer is Paris", "Paris", {"verifier": "unknown_check"}) is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_extract_answer_with_pattern(self):
        assert _extract_answer("#### 42", r"####\s*(\S+)") == "42"

    def test_extract_answer_no_match(self):
        assert _extract_answer("no pattern here", r"####\s*(\S+)") is None

    def test_extract_code_block_markdown(self):
        text = "Here is code:\n```python\ndef foo():\n    return 42\n```"
        code = _extract_code_block(text)
        assert "def foo():" in code

    def test_extract_code_block_raw(self):
        text = "def bar():\n    pass"
        code = _extract_code_block(text)
        assert "def bar():" in code

    def test_is_valid_json_object(self):
        assert _is_valid_json('{"a": 1}') is True

    def test_is_valid_json_array(self):
        assert _is_valid_json("[1, 2, 3]") is True

    def test_is_valid_json_embedded(self):
        assert _is_valid_json('Some text {"key": "val"} more text') is True

    def test_is_valid_json_invalid(self):
        assert _is_valid_json("not json at all") is False

    def test_normalize_text(self):
        result = _normalize_text("The Quick Brown Fox!")
        assert result == "quick brown fox"


# ---------------------------------------------------------------------------
# score_batch
# ---------------------------------------------------------------------------

class TestScoreBatch:
    def test_batch_scoring(self):
        questions = [
            {"id": "q1", "suite": "math", "expected": "42", "scoring_method": "exact_match"},
            {"id": "q2", "suite": "general", "expected": "B", "scoring_method": "multiple_choice"},
        ]
        answers = ["#### 42", "The answer is C"]
        results = score_batch(questions, answers)
        assert len(results) == 2
        assert results[0]["passed"] is True
        assert results[1]["passed"] is False

    def test_batch_empty(self):
        assert score_batch([], []) == []

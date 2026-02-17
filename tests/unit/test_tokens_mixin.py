#!/usr/bin/env python3
"""Unit tests for token estimation mixin."""

from src.llm_primitives.tokens import TokensMixin


class _TestableTokensMixin(TokensMixin):
    """Concrete class to test the mixin."""

    pass


class TestEstimatePromptTokens:
    """Test _estimate_prompt_tokens heuristic (len // 4)."""

    def setup_method(self):
        self.mixin = _TestableTokensMixin()

    def test_empty_string(self):
        assert self.mixin._estimate_prompt_tokens("") == 0

    def test_short_text(self):
        # "Hello" = 5 chars → 5 // 4 = 1
        assert self.mixin._estimate_prompt_tokens("Hello") == 1

    def test_exact_multiple(self):
        # 8 chars → 8 // 4 = 2
        assert self.mixin._estimate_prompt_tokens("12345678") == 2

    def test_long_text(self):
        text = "a" * 400
        assert self.mixin._estimate_prompt_tokens(text) == 100

    def test_unicode_text(self):
        # Unicode chars are multi-byte in UTF-8 but len() counts code points
        text = "\u00e9" * 12  # 12 code points
        assert self.mixin._estimate_prompt_tokens(text) == 3

    def test_single_char(self):
        assert self.mixin._estimate_prompt_tokens("x") == 0  # 1 // 4 = 0

    def test_three_chars(self):
        assert self.mixin._estimate_prompt_tokens("abc") == 0  # 3 // 4 = 0

    def test_four_chars(self):
        assert self.mixin._estimate_prompt_tokens("abcd") == 1  # 4 // 4 = 1


class TestEstimateCompletionTokens:
    """Test _estimate_completion_tokens heuristic (len // 4)."""

    def setup_method(self):
        self.mixin = _TestableTokensMixin()

    def test_empty_string(self):
        assert self.mixin._estimate_completion_tokens("") == 0

    def test_short_text(self):
        assert self.mixin._estimate_completion_tokens("Hello") == 1

    def test_long_text(self):
        text = "b" * 1000
        assert self.mixin._estimate_completion_tokens(text) == 250

    def test_unicode_text(self):
        text = "\u2192" * 20  # 20 arrow chars
        assert self.mixin._estimate_completion_tokens(text) == 5

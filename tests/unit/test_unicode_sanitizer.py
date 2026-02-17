#!/usr/bin/env python3
"""Unit tests for REPL unicode sanitizer (security-critical)."""

import pytest

from src.repl_environment.unicode_sanitizer import (
    _UNICODE_REPLACEMENTS,
    sanitize_code_unicode,
)


class TestFastPath:
    """Test fast-path returns for ASCII-only and empty input."""

    def test_empty_string_returns_empty(self):
        assert sanitize_code_unicode("") == ""

    def test_ascii_only_returns_unchanged(self):
        code = "x = 42 + y * 3\nprint(x)"
        assert sanitize_code_unicode(code) is code  # identity, not copy

    def test_pure_ascii_multiline(self):
        code = "def foo():\n    return 1\n"
        assert sanitize_code_unicode(code) is code


class TestMathOperators:
    """Test mathematical operator replacements."""

    def test_multiplication_sign(self):
        assert sanitize_code_unicode("a \u00d7 b") == "a * b"

    def test_division_sign(self):
        assert sanitize_code_unicode("a \u00f7 b") == "a / b"

    def test_minus_sign(self):
        assert sanitize_code_unicode("a \u2212 b") == "a - b"

    def test_en_dash(self):
        assert sanitize_code_unicode("a \u2013 b") == "a - b"

    def test_em_dash(self):
        assert sanitize_code_unicode("a \u2014 b") == "a - b"

    def test_plus_minus(self):
        assert sanitize_code_unicode("\u00b1 5") == "+- 5"

    def test_less_than_or_equal(self):
        assert sanitize_code_unicode("x \u2264 10") == "x <= 10"

    def test_greater_than_or_equal(self):
        assert sanitize_code_unicode("x \u2265 10") == "x >= 10"

    def test_not_equal(self):
        assert sanitize_code_unicode("x \u2260 y") == "x != y"


class TestDegreeSign:
    """Test degree sign stripping."""

    def test_degree_stripped(self):
        assert sanitize_code_unicode("angle = 47\u00b0") == "angle = 47"


class TestQuoteLookalikes:
    """Test curly/fancy quote normalization."""

    def test_left_single_curly(self):
        assert sanitize_code_unicode("\u2018hello\u2019") == "'hello'"

    def test_left_double_curly(self):
        assert sanitize_code_unicode("\u201chello\u201d") == '"hello"'

    def test_acute_accent(self):
        assert sanitize_code_unicode("\u00b4x\u00b4") == "'x'"

    def test_grave_accent_ascii_only_fast_path(self):
        # U+0060 (backtick) is ASCII, so the fast path returns unchanged
        assert sanitize_code_unicode("\u0060x\u0060") == "`x`"

    def test_grave_accent_with_non_ascii_triggers_replacement(self):
        # When non-ASCII chars force the regex path, backtick is also replaced
        assert sanitize_code_unicode("\u0060x\u00b0") == "'x"


class TestArrows:
    """Test arrow replacements."""

    def test_right_arrow(self):
        assert sanitize_code_unicode("def foo() \u2192 int:") == "def foo() -> int:"

    def test_left_arrow(self):
        assert sanitize_code_unicode("x \u2190 y") == "x <- y"


class TestWhitespaceLookalikes:
    """Test whitespace normalization."""

    def test_non_breaking_space(self):
        assert sanitize_code_unicode("a\u00a0=\u00a0b") == "a = b"

    def test_em_space(self):
        assert sanitize_code_unicode("a\u2003b") == "a b"

    def test_en_space(self):
        assert sanitize_code_unicode("a\u2002b") == "a b"

    def test_zero_width_space_stripped(self):
        assert sanitize_code_unicode("ab\u200bcd") == "abcd"

    def test_zero_width_non_joiner_stripped(self):
        assert sanitize_code_unicode("ab\u200ccd") == "abcd"

    def test_zero_width_joiner_stripped(self):
        assert sanitize_code_unicode("ab\u200dcd") == "abcd"

    def test_bom_stripped(self):
        assert sanitize_code_unicode("\ufeffx = 1") == "x = 1"


class TestSuperscripts:
    """Test superscript digit replacements."""

    def test_superscript_2(self):
        assert sanitize_code_unicode("x\u00b2") == "x**2"

    def test_superscript_3(self):
        assert sanitize_code_unicode("x\u00b3") == "x**3"

    def test_superscript_0(self):
        assert sanitize_code_unicode("x\u2070") == "x**0"

    def test_superscript_1(self):
        assert sanitize_code_unicode("x\u00b9") == "x**1"

    @pytest.mark.parametrize(
        "char, digit",
        [
            ("\u2074", "4"),
            ("\u2075", "5"),
            ("\u2076", "6"),
            ("\u2077", "7"),
            ("\u2078", "8"),
            ("\u2079", "9"),
        ],
    )
    def test_superscript_4_through_9(self, char, digit):
        assert sanitize_code_unicode(f"x{char}") == f"x**{digit}"


class TestSubscripts:
    """Test subscript digit replacements."""

    @pytest.mark.parametrize(
        "char, digit",
        [
            ("\u2080", "0"),
            ("\u2081", "1"),
            ("\u2082", "2"),
            ("\u2083", "3"),
            ("\u2084", "4"),
            ("\u2085", "5"),
            ("\u2086", "6"),
            ("\u2087", "7"),
            ("\u2088", "8"),
            ("\u2089", "9"),
        ],
    )
    def test_subscript_digits(self, char, digit):
        assert sanitize_code_unicode(f"x{char}") == f"x{digit}"


class TestAllMappingsCovered:
    """Verify every entry in _UNICODE_REPLACEMENTS is exercised."""

    def test_replacement_count(self):
        """Sanity check: we expect 45 replacement entries."""
        assert len(_UNICODE_REPLACEMENTS) == 45

    @pytest.mark.parametrize(
        "unicode_char, expected",
        [
            (ch, repl)
            for ch, repl in _UNICODE_REPLACEMENTS.items()
            if not ch.isascii()  # skip ASCII chars (fast-path bypasses regex)
        ],
    )
    def test_each_non_ascii_mapping(self, unicode_char, expected):
        """Every non-ASCII mapping individually produces the expected output."""
        result = sanitize_code_unicode(f"prefix{unicode_char}suffix")
        assert result == f"prefix{expected}suffix"


class TestMixedContent:
    """Test realistic code with mixed Unicode and ASCII."""

    def test_physics_formula(self):
        code = "E\u2082 = m \u00d7 c\u00b2"
        assert sanitize_code_unicode(code) == "E2 = m * c**2"

    def test_non_mapped_unicode_preserved(self):
        code = "name = '\u00e9l\u00e8ve'"  # French accented chars
        result = sanitize_code_unicode(code)
        assert "\u00e9" in result
        assert "\u00e8" in result

    def test_mixed_operators_in_expression(self):
        code = "result = (a \u00d7 b) \u00f7 (c \u2212 d)"
        assert sanitize_code_unicode(code) == "result = (a * b) / (c - d)"

"""Tests for script interception (zero-cost local resolution)."""

from __future__ import annotations

import re

import pytest

from src.api.routes.chat_pipeline.script_interceptor import (
    register_interceptor,
    try_intercept,
    _interceptors,
)


class TestTimestampInterceptor:
    """Tests for timestamp/time queries."""

    def test_what_time_is_it(self):
        result = try_intercept("what time is it")
        assert result.matched
        assert result.pattern_name == "timestamp"
        # Should contain time-like format
        assert re.search(r"\d{4}-\d{2}-\d{2}", result.result)

    def test_whats_the_current_time(self):
        result = try_intercept("what's the current time")
        assert result.matched
        assert result.pattern_name == "timestamp"

    def test_current_timestamp(self):
        result = try_intercept("current timestamp")
        assert result.matched

    def test_give_me_the_time(self):
        result = try_intercept("give me the time")
        assert result.matched

    def test_time_in_iso(self):
        result = try_intercept("what's the current time in iso")
        assert result.matched
        assert "T" in result.result  # ISO format contains T separator

    def test_time_in_unix(self):
        result = try_intercept("current timestamp in unix")
        assert result.matched
        assert result.result.isdigit()


class TestDateInterceptor:
    """Tests for date queries."""

    def test_whats_the_date(self):
        result = try_intercept("what's the date")
        assert result.matched
        assert result.pattern_name == "date"

    def test_what_date_is_today(self):
        result = try_intercept("what date is it today")
        assert result.matched

    def test_todays_date(self):
        result = try_intercept("today's date")
        assert result.matched
        # Should contain month name
        assert re.search(r"[A-Z][a-z]+", result.result)


class TestArithmeticInterceptor:
    """Tests for arithmetic evaluation."""

    def test_simple_addition(self):
        result = try_intercept("2 + 3")
        assert result.matched
        assert result.pattern_name == "arithmetic"
        assert result.result == "5"

    def test_whats_calculation(self):
        result = try_intercept("what's 15 * 4")
        assert result.matched
        assert result.result == "60"

    def test_complex_expression(self):
        result = try_intercept("calculate (100 + 50) / 3")
        assert result.matched
        assert result.result == "50"

    def test_modulo(self):
        result = try_intercept("17 % 5")
        assert result.matched
        assert result.result == "2"

    def test_float_result(self):
        result = try_intercept("10 / 3")
        assert result.matched
        assert "3.333" in result.result

    def test_rejects_non_numeric(self):
        """Expressions with letters should be rejected."""
        result = try_intercept("2 + x")
        # Should either not match or fall through
        assert not result.matched or result.result is None

    def test_rejects_import_attempt(self):
        """Must not eval dangerous strings."""
        result = try_intercept("__import__('os')")
        assert not result.matched


class TestUUIDInterceptor:
    """Tests for UUID generation."""

    def test_generate_uuid(self):
        result = try_intercept("generate a uuid")
        assert result.matched
        assert result.pattern_name == "uuid"
        # UUID v4 format
        assert re.match(
            r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            result.result,
        )

    def test_create_new_uuid(self):
        result = try_intercept("create a new uuid")
        assert result.matched

    def test_give_me_uuid(self):
        result = try_intercept("give me a uuid")
        assert result.matched

    def test_each_uuid_is_unique(self):
        r1 = try_intercept("generate uuid")
        r2 = try_intercept("generate uuid")
        assert r1.result != r2.result


class TestEdgeCases:
    """Tests for edge cases and safety constraints."""

    def test_empty_message(self):
        result = try_intercept("")
        assert not result.matched

    def test_whitespace_only(self):
        result = try_intercept("   ")
        assert not result.matched

    def test_long_message_not_intercepted(self):
        """Messages over 200 chars should never be intercepted."""
        long_msg = "what is the time " + "x" * 200
        result = try_intercept(long_msg)
        assert not result.matched

    def test_ambiguous_message_falls_through(self):
        """Complex questions should not be intercepted."""
        result = try_intercept("How do I implement a timestamp parser in Python?")
        assert not result.matched

    def test_no_match_returns_unmatched(self):
        result = try_intercept("Tell me about quantum computing")
        assert not result.matched
        assert result.pattern_name is None
        assert result.result is None

    def test_result_dataclass_is_frozen(self):
        result = try_intercept("generate uuid")
        with pytest.raises(AttributeError):
            result.matched = False  # type: ignore[misc]

    def test_elapsed_ms_populated_on_match(self):
        result = try_intercept("what time is it")
        assert result.matched
        assert result.elapsed_ms >= 0.0
        assert result.elapsed_ms < 100.0  # Should be sub-ms


class TestCustomInterceptor:
    """Test the register_interceptor() extension API."""

    def test_register_custom_interceptor(self):
        # Save original count
        original_count = len(_interceptors)

        try:
            register_interceptor(
                "test_ping",
                r"^ping$",
                lambda m: "pong",
            )

            result = try_intercept("ping")
            assert result.matched
            assert result.pattern_name == "test_ping"
            assert result.result == "pong"
        finally:
            # Clean up: remove the test interceptor
            while len(_interceptors) > original_count:
                _interceptors.pop()

    def test_handler_returning_none_falls_through(self):
        """If handler returns None, try next interceptor."""
        original_count = len(_interceptors)

        try:
            register_interceptor(
                "test_none",
                r"^fallthrough test$",
                lambda m: None,
            )

            result = try_intercept("fallthrough test")
            assert not result.matched
        finally:
            while len(_interceptors) > original_count:
                _interceptors.pop()

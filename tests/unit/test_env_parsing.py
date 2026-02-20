"""Unit tests for shared environment parsing helpers."""

from __future__ import annotations

import logging

from src.env_parsing import env_bool, env_float, env_int


def test_env_int_parses_valid(monkeypatch):
    monkeypatch.setenv("TEST_INT", "42")
    assert env_int("TEST_INT", 7) == 42


def test_env_int_uses_default_on_invalid(monkeypatch, caplog):
    monkeypatch.setenv("TEST_INT", "not-a-number")
    with caplog.at_level(logging.WARNING):
        value = env_int("TEST_INT", 9)
    assert value == 9
    assert "Invalid TEST_INT" in caplog.text


def test_env_bool_truthy_and_falsy(monkeypatch):
    monkeypatch.setenv("TEST_BOOL", "true")
    assert env_bool("TEST_BOOL", False) is True
    monkeypatch.setenv("TEST_BOOL", "off")
    assert env_bool("TEST_BOOL", True) is False


def test_env_bool_uses_default_for_unknown(monkeypatch):
    monkeypatch.setenv("TEST_BOOL", "maybe")
    assert env_bool("TEST_BOOL", True) is True
    assert env_bool("TEST_BOOL", False) is False


def test_env_float_parses_valid(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "3.14")
    assert env_float("TEST_FLOAT", 1.0) == 3.14


def test_env_float_uses_default_on_invalid(monkeypatch, caplog):
    monkeypatch.setenv("TEST_FLOAT", "abc")
    with caplog.at_level(logging.WARNING):
        value = env_float("TEST_FLOAT", 2.5)
    assert value == 2.5
    assert "Invalid TEST_FLOAT" in caplog.text

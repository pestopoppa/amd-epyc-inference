#!/usr/bin/env python3
"""Tests for the routing bindings module."""

from src.routing_bindings import BindingPriority, BindingRouter, RoutingBinding


class TestBindingPriority:
    """Tests for BindingPriority enum."""

    def test_ordering(self):
        assert BindingPriority.DEFAULT < BindingPriority.CLASSIFIER
        assert BindingPriority.CLASSIFIER < BindingPriority.Q_VALUE
        assert BindingPriority.Q_VALUE < BindingPriority.USER_PREF
        assert BindingPriority.USER_PREF < BindingPriority.SESSION


class TestBindingRouter:
    """Tests for BindingRouter class."""

    def test_empty_resolve_returns_none(self):
        router = BindingRouter()
        assert router.resolve("code") is None

    def test_single_binding(self):
        router = BindingRouter()
        router.add(RoutingBinding("code", "coder_primary", BindingPriority.DEFAULT))
        assert router.resolve("code") == "coder_primary"

    def test_higher_priority_wins(self):
        router = BindingRouter()
        router.add(RoutingBinding("code", "coder_primary", BindingPriority.DEFAULT))
        router.add(RoutingBinding("code", "architect_coding", BindingPriority.USER_PREF))
        assert router.resolve("code") == "architect_coding"

    def test_different_task_types_independent(self):
        router = BindingRouter()
        router.add(RoutingBinding("code", "coder_primary", BindingPriority.DEFAULT))
        router.add(RoutingBinding("ingest", "ingest_long_context", BindingPriority.DEFAULT))
        assert router.resolve("code") == "coder_primary"
        assert router.resolve("ingest") == "ingest_long_context"

    def test_inactive_binding_ignored(self):
        router = BindingRouter()
        router.add(RoutingBinding("code", "coder_primary", BindingPriority.DEFAULT))
        router.add(RoutingBinding("code", "architect_coding", BindingPriority.USER_PREF, active=False))
        assert router.resolve("code") == "coder_primary"

    def test_session_binding_override(self):
        router = BindingRouter()
        router.add(RoutingBinding("code", "coder_primary", BindingPriority.DEFAULT))
        router.set_session_binding("code", "architect_coding")
        assert router.resolve("code") == "architect_coding"

    def test_clear_session_bindings(self):
        router = BindingRouter()
        router.add(RoutingBinding("code", "coder_primary", BindingPriority.DEFAULT))
        router.set_session_binding("code", "architect_coding")
        router.clear_session_bindings()
        assert router.resolve("code") == "coder_primary"

    def test_resolve_with_info(self):
        router = BindingRouter()
        router.add(RoutingBinding(
            "code", "coder_primary", BindingPriority.CLASSIFIER, source="keyword_heuristic"
        ))
        role, priority, source = router.resolve_with_info("code")
        assert role == "coder_primary"
        assert priority == BindingPriority.CLASSIFIER
        assert source == "keyword_heuristic"

    def test_list_bindings(self):
        router = BindingRouter()
        router.add(RoutingBinding("code", "coder_primary", BindingPriority.DEFAULT))
        router.add(RoutingBinding("code", "architect_coding", BindingPriority.USER_PREF))

        bindings = router.list_bindings("code")
        assert len(bindings) == 2
        assert any(b["role"] == "coder_primary" for b in bindings)
        assert any(b["role"] == "architect_coding" for b in bindings)

    def test_list_all_bindings(self):
        router = BindingRouter()
        router.add(RoutingBinding("code", "coder_primary", BindingPriority.DEFAULT))
        router.add(RoutingBinding("ingest", "ingest_long_context", BindingPriority.DEFAULT))

        bindings = router.list_bindings()
        assert len(bindings) == 2

    def test_clear(self):
        router = BindingRouter()
        router.add(RoutingBinding("code", "coder_primary", BindingPriority.DEFAULT))
        router.clear()
        assert router.resolve("code") is None

    def test_session_binding_replaces_previous(self):
        router = BindingRouter()
        router.set_session_binding("code", "coder_primary")
        router.set_session_binding("code", "architect_coding")

        bindings = router.list_bindings("code")
        session_bindings = [b for b in bindings if b["priority"] == "SESSION"]
        assert len(session_bindings) == 1
        assert session_bindings[0]["role"] == "architect_coding"

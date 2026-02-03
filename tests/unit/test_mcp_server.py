#!/usr/bin/env python3
"""Tests for the read-only MCP server.

Tests tool functions directly (bypassing MCP transport).
All registry/file access is mocked.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.mcp_server import (
    list_roles,
    lookup_model,
    query_benchmarks,
    server_status,
)


# Mock role config objects for testing
@dataclass
class MockPerformance:
    optimized_tps: float | None = None
    baseline_tps: float | None = None
    speedup: str | None = None


@dataclass
class MockAcceleration:
    type: str = "none"


@dataclass
class MockModel:
    name: str = ""
    quant: str = ""
    size_gb: float = 0.0


@dataclass
class MockRoleConfig:
    name: str = ""
    tier: str = "A"
    description: str = ""
    model: MockModel = None
    acceleration: MockAcceleration = None
    performance: MockPerformance = None
    constraints: None = None
    notes: None = None

    def __post_init__(self):
        if self.model is None:
            self.model = MockModel()
        if self.acceleration is None:
            self.acceleration = MockAcceleration()
        if self.performance is None:
            self.performance = MockPerformance()


# ============ lookup_model ============


class TestLookupModel:
    @patch("src.registry_loader.RegistryLoader", autospec=False)
    def test_success(self, mock_loader_class):
        """Returns formatted role config."""
        mock_role = MockRoleConfig(
            name="coder_primary",
            tier="B",
            description="Primary code generation",
            model=MockModel(
                name="Qwen2.5-Coder-32B",
                quant="Q4_K_M",
                size_gb=20.0,
            ),
            acceleration=MockAcceleration(type="speculative_decoding"),
            performance=MockPerformance(
                optimized_tps=33.0,
                baseline_tps=3.0,
                speedup="11x",
            ),
        )

        mock_registry = MagicMock()
        mock_registry.get_role.return_value = mock_role
        mock_loader_class.return_value = mock_registry

        result = lookup_model("coder_primary")
        assert "coder_primary" in result
        assert "Qwen2.5-Coder-32B" in result
        assert "33.0 t/s" in result
        assert "speculative_decoding" in result

    @patch("src.registry_loader.RegistryLoader", autospec=False)
    def test_role_not_found(self, mock_loader_class):
        """Returns error for unknown role."""
        mock_registry = MagicMock()
        mock_registry.get_role.side_effect = KeyError("unknown_role")
        mock_loader_class.return_value = mock_registry

        result = lookup_model("unknown_role")
        assert "not found" in result.lower()


# ============ list_roles ============


class TestListRoles:
    @patch("src.registry_loader.RegistryLoader", autospec=False)
    def test_success(self, mock_loader_class):
        """Returns roles grouped by tier."""
        mock_role_a = MockRoleConfig(
            name="frontdoor",
            tier="A",
            model=MockModel(name="Qwen3-Coder-30B"),
            acceleration=MockAcceleration(type="moe_expert_reduction"),
            performance=MockPerformance(optimized_tps=18.0, baseline_tps=10.0),
        )

        mock_role_b = MockRoleConfig(
            name="coder_primary",
            tier="B",
            model=MockModel(name="Qwen2.5-Coder-32B"),
            acceleration=MockAcceleration(type="speculative_decoding"),
            performance=MockPerformance(optimized_tps=33.0, baseline_tps=3.0),
        )

        mock_registry = MagicMock()

        def get_roles(tier):
            if tier == "A":
                return [mock_role_a]
            if tier == "B":
                return [mock_role_b]
            return []

        mock_registry.get_roles_by_tier.side_effect = get_roles
        mock_loader_class.return_value = mock_registry

        result = list_roles()
        assert "Tier A" in result
        assert "frontdoor" in result
        assert "Tier B" in result
        assert "coder_primary" in result

    @patch("src.registry_loader.RegistryLoader", autospec=False)
    def test_empty_registry(self, mock_loader_class):
        """Handles empty registry."""
        mock_registry = MagicMock()
        mock_registry.get_roles_by_tier.return_value = []
        mock_loader_class.return_value = mock_registry

        result = list_roles()
        assert "No roles configured" in result


# ============ server_status ============


class TestServerStatus:
    def test_no_state_file(self, tmp_path, monkeypatch):
        """Returns message when state file doesn't exist."""
        monkeypatch.setattr("src.mcp_server.PROJECT_ROOT", tmp_path)
        result = server_status()
        assert "not be running" in result.lower() or "not found" in result.lower()

    def test_running_services(self, tmp_path, monkeypatch):
        """Returns formatted service info."""
        monkeypatch.setattr("src.mcp_server.PROJECT_ROOT", tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        state = {
            "frontdoor": {"pid": 1234, "port": 8080, "started_at": "2026-01-29T10:00:00"},
            "coder": {"pid": 5678, "port": 8081, "started_at": "2026-01-29T10:00:05"},
        }
        (logs_dir / "orchestrator_state.json").write_text(json.dumps(state))

        result = server_status()
        assert "frontdoor" in result
        assert "1234" in result
        assert "8080" in result
        assert "coder" in result

    def test_empty_state(self, tmp_path, monkeypatch):
        """Handles empty state file."""
        monkeypatch.setattr("src.mcp_server.PROJECT_ROOT", tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "orchestrator_state.json").write_text("{}")

        result = server_status()
        assert "No services running" in result


# ============ query_benchmarks ============


class TestQueryBenchmarks:
    def _write_csv(self, tmp_path, rows):
        """Helper to write a summary CSV."""
        results_dir = tmp_path / "benchmarks" / "results" / "reviews"
        results_dir.mkdir(parents=True)
        csv_path = results_dir / "summary.csv"

        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return csv_path

    def test_all_models(self, tmp_path, monkeypatch):
        """Returns all models when no filter."""
        monkeypatch.setattr("src.mcp_server.PROJECT_ROOT", tmp_path)
        self._write_csv(tmp_path, [
            {"model": "Qwen2.5-7B", "thinking": "8/10", "pct_str": "80%", "avg_tps": "15.0"},
            {"model": "Qwen3-235B", "thinking": "9/10", "pct_str": "90%", "avg_tps": "6.7"},
        ])

        result = query_benchmarks()
        assert "Qwen2.5-7B" in result
        assert "Qwen3-235B" in result

    def test_filter_by_model(self, tmp_path, monkeypatch):
        """Filters by model name substring."""
        monkeypatch.setattr("src.mcp_server.PROJECT_ROOT", tmp_path)
        self._write_csv(tmp_path, [
            {"model": "Qwen2.5-7B", "pct_str": "80%", "avg_tps": "15.0"},
            {"model": "Qwen3-235B", "pct_str": "90%", "avg_tps": "6.7"},
        ])

        result = query_benchmarks(model_name="235B")
        assert "Qwen3-235B" in result
        assert "Qwen2.5-7B" not in result

    def test_filter_by_suite(self, tmp_path, monkeypatch):
        """Shows suite-specific scores when suite filter applied."""
        monkeypatch.setattr("src.mcp_server.PROJECT_ROOT", tmp_path)
        self._write_csv(tmp_path, [
            {"model": "TestModel", "thinking": "9/10", "coder": "7/10", "pct_str": "85%", "avg_tps": "10.0"},
        ])

        result = query_benchmarks(suite="thinking")
        assert "thinking=9/10" in result

    def test_no_csv(self, tmp_path, monkeypatch):
        """Returns message when CSV doesn't exist."""
        monkeypatch.setattr("src.mcp_server.PROJECT_ROOT", tmp_path)
        result = query_benchmarks()
        assert "No benchmark summary" in result

    def test_no_match(self, tmp_path, monkeypatch):
        """Returns message when no models match filter."""
        monkeypatch.setattr("src.mcp_server.PROJECT_ROOT", tmp_path)
        self._write_csv(tmp_path, [
            {"model": "Qwen2.5-7B", "pct_str": "80%", "avg_tps": "15.0"},
        ])

        result = query_benchmarks(model_name="NonexistentModel")
        assert "No results matching" in result

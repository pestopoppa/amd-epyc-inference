#!/usr/bin/env python3
"""Unit tests for input formalizer (Phase 4).

Tests keyword detection heuristics, context injection formatting,
failure handling, and feature flag gating. All tests use mocks —
no subprocess calls or model loading.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.formalizer import (
    FormalizationResult,
    should_formalize_input,
    formalize_prompt,
    inject_formalization,
    _build_formalizer_prompt,
    _parse_formalizer_output,
)


# ---------------------------------------------------------------------------
# should_formalize_input tests
# ---------------------------------------------------------------------------

class TestShouldFormalizeInput:
    """Tests for keyword-based formalization detection."""

    def test_optimization_keywords(self):
        prompt = "Minimize the total cost subject to capacity constraints across 5 warehouses"
        should, hint = should_formalize_input(prompt)
        assert should is True
        assert hint == "optimization"

    def test_maximize_keyword(self):
        should, hint = should_formalize_input("Maximize throughput while keeping latency under 100ms")
        assert should is True
        assert hint == "optimization"

    def test_constraint_keyword(self):
        should, hint = should_formalize_input("Find a feasible solution given the constraint that x + y <= 10")
        assert should is True
        assert hint == "optimization"

    def test_proof_keywords(self):
        should, hint = should_formalize_input("Prove that the algorithm terminates for all inputs")
        assert should is True
        assert hint == "proof"

    def test_theorem_keyword(self):
        should, hint = should_formalize_input("Verify the following theorem about group homomorphisms")
        assert should is True
        assert hint == "proof"

    def test_invariant_keyword(self):
        should, hint = should_formalize_input("Show the loop invariant holds: sum equals partial sum of array")
        assert should is True
        assert hint == "proof"

    def test_algorithm_keywords(self):
        should, hint = should_formalize_input("Implement a lock-free concurrent queue with O(1) amortized enqueue")
        assert should is True
        assert hint == "algorithm"

    def test_complexity_keyword(self):
        should, hint = should_formalize_input("What is the complexity of this algorithm using dynamic programming?")
        assert should is True
        assert hint == "algorithm"

    def test_simple_question_no_formalize(self):
        should, hint = should_formalize_input("What is the capital of France?")
        assert should is False
        assert hint == ""

    def test_hello_world_no_formalize(self):
        should, hint = should_formalize_input("Write hello world in Python")
        assert should is False
        assert hint == ""

    def test_short_prompt_no_formalize(self):
        should, hint = should_formalize_input("Fix the bug")
        assert should is False
        assert hint == ""

    def test_empty_prompt_no_formalize(self):
        should, hint = should_formalize_input("")
        assert should is False
        assert hint == ""

    def test_vague_spec_long_prompt(self):
        # Long prompt starting with "Build a..." and no technical specifics
        prompt = (
            "Build a system that handles all the user requests and "
            "processes them in the right order making sure everything "
            "works together nicely and the users are happy with the "
            "results they get back from the system when they submit "
            "their requests through the interface that we need to "
            "design in a way that makes sense for everyone involved "
            "in the project"
        )
        should, hint = should_formalize_input(prompt)
        assert should is True
        assert hint == "ambiguous_spec"

    def test_vague_prefix_short_not_triggered(self):
        # Short "build a..." prompt shouldn't trigger vague detection
        should, hint = should_formalize_input("Build a REST API with Flask")
        assert should is False

    def test_priority_optimization_over_algorithm(self):
        # If both optimization and algorithm keywords present, optimization wins (checked first)
        prompt = "Design an algorithm to optimize the objective function subject to constraints"
        should, hint = should_formalize_input(prompt)
        assert should is True
        assert hint == "optimization"


# ---------------------------------------------------------------------------
# _parse_formalizer_output tests
# ---------------------------------------------------------------------------

class TestParseFormalizer:
    """Tests for JSON parsing of formalizer output."""

    def test_clean_json(self):
        data = {"problem_type": "optimization", "variables": [], "constraints": []}
        result = _parse_formalizer_output(json.dumps(data))
        assert result is not None
        assert result["problem_type"] == "optimization"

    def test_json_with_markdown_fences(self):
        data = {"problem_type": "proof", "variables": [], "constraints": []}
        raw = f"```json\n{json.dumps(data)}\n```"
        result = _parse_formalizer_output(raw)
        assert result is not None
        assert result["problem_type"] == "proof"

    def test_json_with_leading_text(self):
        data = {"problem_type": "algorithm", "variables": [], "constraints": []}
        raw = f"Here is the formalization:\n{json.dumps(data)}"
        result = _parse_formalizer_output(raw)
        assert result is not None
        assert result["problem_type"] == "algorithm"

    def test_empty_output_returns_none(self):
        assert _parse_formalizer_output("") is None

    def test_garbage_output_returns_none(self):
        assert _parse_formalizer_output("not json at all") is None

    def test_json_without_problem_type_returns_none(self):
        assert _parse_formalizer_output('{"foo": "bar"}') is None


# ---------------------------------------------------------------------------
# inject_formalization tests
# ---------------------------------------------------------------------------

class TestInjectFormalization:
    """Tests for context augmentation with formal spec."""

    def test_basic_injection(self):
        ir = {
            "problem_type": "optimization",
            "variables": [{"name": "x", "type": "float", "constraints": ["x >= 0"]}],
            "constraints": ["x + y <= 10"],
            "objective": "minimize cost",
            "edge_cases": [{"input": "x=0, y=0", "expected": "cost=0"}],
            "acceptance_criteria": ["Solution is feasible", "Cost is minimized"],
        }
        result = inject_formalization("test prompt", "", ir)

        assert "[FORMAL SPECIFICATION]" in result
        assert "[/FORMAL SPECIFICATION]" in result
        assert "Problem type: optimization" in result
        assert "x: float (x >= 0)" in result
        assert "x + y <= 10" in result
        assert "minimize cost" in result
        assert "x=0, y=0" in result
        assert "Solution is feasible" in result

    def test_injection_appends_to_existing_context(self):
        ir = {"problem_type": "proof", "variables": [], "constraints": []}
        result = inject_formalization("prompt", "existing context", ir)
        assert result.startswith("existing context\n\n[FORMAL SPECIFICATION]")

    def test_injection_empty_context(self):
        ir = {"problem_type": "algorithm", "variables": [], "constraints": []}
        result = inject_formalization("prompt", "", ir)
        assert result.startswith("[FORMAL SPECIFICATION]")

    def test_injection_missing_optional_fields(self):
        ir = {"problem_type": "validation"}
        result = inject_formalization("prompt", "", ir)
        assert "Problem type: validation" in result
        assert "[/FORMAL SPECIFICATION]" in result
        # Should not crash on missing fields
        assert "Variables" not in result
        assert "Constraints" not in result


# ---------------------------------------------------------------------------
# FormalizationResult tests
# ---------------------------------------------------------------------------

class TestFormalizationResult:
    """Tests for result dataclass behavior."""

    def test_success_result(self):
        r = FormalizationResult(
            success=True,
            ir_json={"problem_type": "optimization"},
            raw_output='{"problem_type": "optimization"}',
            elapsed_seconds=2.5,
            model_role="formalizer",
        )
        assert r.success is True
        assert r.ir_json["problem_type"] == "optimization"

    def test_failure_result(self):
        r = FormalizationResult(
            success=False,
            error="All formalizer attempts failed",
        )
        assert r.success is False
        assert r.ir_json is None
        assert "failed" in r.error

    def test_default_values(self):
        r = FormalizationResult(success=False)
        assert r.ir_json is None
        assert r.raw_output == ""
        assert r.elapsed_seconds == 0.0
        assert r.model_role == ""
        assert r.error == ""


# ---------------------------------------------------------------------------
# formalize_prompt tests (mocked subprocess)
# ---------------------------------------------------------------------------

class TestFormalizePrompt:
    """Tests for formalizer invocation with mocked subprocess."""

    def _make_mock_registry(self, has_role: bool = True):
        registry = MagicMock()
        if has_role:
            role = MagicMock()
            role.name = "formalizer"
            registry.get_role.return_value = role
            registry.generate_command.return_value = "echo test"
        else:
            registry.get_role.side_effect = KeyError("formalizer")
        return registry

    @patch("src.formalizer.subprocess.run")
    def test_successful_formalization(self, mock_run):
        ir = {"problem_type": "optimization", "variables": [], "constraints": []}
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(ir),
            stderr="",
        )
        registry = self._make_mock_registry()
        result = formalize_prompt("minimize x", "optimization", registry)

        assert result.success is True
        assert result.ir_json["problem_type"] == "optimization"
        assert result.model_role == "formalizer"

    @patch("src.formalizer.subprocess.run")
    def test_nonzero_exit_falls_through(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="error loading model",
        )
        registry = self._make_mock_registry()
        result = formalize_prompt("minimize x", "optimization", registry)

        # Falls through all roles and fails
        assert result.success is False

    @patch("src.formalizer.subprocess.run")
    def test_invalid_json_falls_through(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not valid json",
            stderr="",
        )
        registry = self._make_mock_registry()
        result = formalize_prompt("minimize x", "optimization", registry)

        assert result.success is False

    def test_no_formalizer_role_in_registry(self):
        registry = self._make_mock_registry(has_role=False)
        result = formalize_prompt("minimize x", "optimization", registry)
        assert result.success is False

    @patch("src.formalizer.subprocess.run")
    def test_timeout_handled(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=60)
        registry = self._make_mock_registry()
        result = formalize_prompt("minimize x", "optimization", registry, timeout=60)
        assert result.success is False


# ---------------------------------------------------------------------------
# Feature flag gating test
# ---------------------------------------------------------------------------

class TestFeatureFlagGating:
    """Verify formalization respects the input_formalizer feature flag."""

    def test_feature_flag_off_skips_formalization(self):
        from src.features import Features
        f = Features(input_formalizer=False)
        assert f.input_formalizer is False
        assert "input_formalizer" in f.summary()
        assert f.summary()["input_formalizer"] is False

    def test_feature_flag_on_enables_formalization(self):
        from src.features import Features
        f = Features(input_formalizer=True)
        assert f.input_formalizer is True
        assert f.summary()["input_formalizer"] is True


# ---------------------------------------------------------------------------
# _build_formalizer_prompt test
# ---------------------------------------------------------------------------

class TestBuildFormalizerPrompt:
    """Tests for prompt construction."""

    def test_includes_problem_hint(self):
        prompt = _build_formalizer_prompt("minimize x^2", "optimization")
        assert "optimization" in prompt
        assert "minimize x^2" in prompt
        assert "JSON" in prompt

    def test_no_hint(self):
        prompt = _build_formalizer_prompt("some task", "")
        assert "Problem type hint" not in prompt
        assert "some task" in prompt

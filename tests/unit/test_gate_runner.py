#!/usr/bin/env python3
"""Unit tests for gate runner."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

from src.gate_runner import (
    GateRunner,
    GateConfig,
    GateResult,
    GateRunnerError,
)


class TestGateConfig:
    """Test GateConfig dataclass."""

    def test_from_dict_minimal(self):
        """Test creating GateConfig from minimal dict."""
        data = {"name": "test", "command": "echo hello"}
        config = GateConfig.from_dict(data)

        assert config.name == "test"
        assert config.command == "echo hello"
        assert config.timeout == 60  # default
        assert config.required is True  # default

    def test_from_dict_full(self):
        """Test creating GateConfig from full dict."""
        data = {
            "name": "lint",
            "command": "make lint",
            "timeout": 120,
            "required": False,
            "retry_count": 2,
            "description": "Run linters",
        }
        config = GateConfig.from_dict(data)

        assert config.name == "lint"
        assert config.timeout == 120
        assert config.required is False
        assert config.retry_count == 2
        assert config.description == "Run linters"


class TestGateResult:
    """Test GateResult dataclass."""

    def test_summary_passed(self):
        """Test summary for passed gate."""
        result = GateResult(
            gate_name="test",
            passed=True,
            exit_code=0,
            output="OK",
            elapsed_seconds=1.5,
        )
        assert "[PASSED]" in result.summary
        assert "test" in result.summary

    def test_summary_failed(self):
        """Test summary for failed gate."""
        result = GateResult(
            gate_name="test",
            passed=False,
            exit_code=1,
            output="ERROR",
            elapsed_seconds=2.0,
        )
        assert "[FAILED]" in result.summary

    def test_to_dict(self):
        """Test to_dict method."""
        result = GateResult(
            gate_name="test",
            passed=True,
            exit_code=0,
            output="OK",
            elapsed_seconds=1.0,
            errors=["error1"],
            warnings=["warn1"],
        )
        d = result.to_dict()

        assert d["gate_name"] == "test"
        assert d["passed"] is True
        assert d["errors"] == ["error1"]
        assert d["warnings"] == ["warn1"]


class TestGateRunnerInit:
    """Test GateRunner initialization."""

    def test_default_gates_when_no_config(self, tmp_path):
        """Test that default gates are used when config doesn't exist."""
        runner = GateRunner(config_path=tmp_path / "nonexistent.yaml")

        assert len(runner.gates) > 0
        assert any(g.name == "format" for g in runner.gates)
        assert any(g.name == "unit" for g in runner.gates)

    def test_load_config_from_file(self, tmp_path):
        """Test loading gates from config file."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: custom
    command: echo custom
    timeout: 10
""")
        runner = GateRunner(config_path=config_file)

        assert len(runner.gates) == 1
        assert runner.gates[0].name == "custom"

    def test_get_gate_names(self, tmp_path):
        """Test get_gate_names method."""
        runner = GateRunner(config_path=tmp_path / "nonexistent.yaml")
        names = runner.get_gate_names()

        assert "format" in names
        assert "unit" in names


class TestRunGate:
    """Test running individual gates."""

    def test_run_gate_success(self, tmp_path):
        """Test running a gate that succeeds."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: echo
    command: echo hello
    timeout: 10
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        result = runner.run_gate(runner.gates[0])

        assert result.passed is True
        assert result.exit_code == 0
        assert "hello" in result.output

    def test_run_gate_failure(self, tmp_path):
        """Test running a gate that fails."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: fail
    command: exit 1
    timeout: 10
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        result = runner.run_gate(runner.gates[0])

        assert result.passed is False
        assert result.exit_code == 1

    def test_run_gate_timeout(self, tmp_path):
        """Test running a gate that times out."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: slow
    command: sleep 10
    timeout: 1
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        result = runner.run_gate(runner.gates[0])

        assert result.passed is False
        assert "timed out" in result.errors[0].lower()

    def test_run_gate_records_elapsed_time(self, tmp_path):
        """Test that elapsed time is recorded."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: echo
    command: echo hello
    timeout: 10
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        result = runner.run_gate(runner.gates[0])

        assert result.elapsed_seconds >= 0


class TestParseOutput:
    """Test output parsing."""

    def test_parse_errors(self, tmp_path):
        """Test that errors are extracted from output."""
        runner = GateRunner(config_path=tmp_path / "nonexistent.yaml")
        output = """
Some info line
error: something went wrong
Another line
fatal: another error
"""
        errors, warnings = runner._parse_output(output, "test")

        assert len(errors) >= 2
        assert any("something went wrong" in e for e in errors)
        assert any("another error" in e for e in errors)

    def test_parse_warnings(self, tmp_path):
        """Test that warnings are extracted from output."""
        runner = GateRunner(config_path=tmp_path / "nonexistent.yaml")
        output = """
warning: this is a warning
info: this is info
warn some other warning
"""
        errors, warnings = runner._parse_output(output, "test")

        assert len(warnings) >= 1
        assert any("warning" in w.lower() for w in warnings)

    def test_parse_skips_log_prefixes(self, tmp_path):
        """Test that common log prefixes are skipped."""
        runner = GateRunner(config_path=tmp_path / "nonexistent.yaml")
        output = """
build: error in build
info: error in info
main: error in main
actual error: real error
"""
        errors, warnings = runner._parse_output(output, "test")

        # Should only capture real error, not build/info/main lines
        assert len(errors) == 1
        assert "real error" in errors[0]


class TestRunAllGates:
    """Test running all gates."""

    def test_run_all_gates_all_pass(self, tmp_path):
        """Test running all gates when all pass."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: gate1
    command: echo gate1
    timeout: 10
  - name: gate2
    command: echo gate2
    timeout: 10
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = runner.run_all_gates()

        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_run_all_gates_stops_on_required_failure(self, tmp_path):
        """Test that execution stops on required gate failure."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: fail
    command: exit 1
    timeout: 10
    required: true
  - name: never_runs
    command: echo should_not_run
    timeout: 10
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = runner.run_all_gates(stop_on_first_failure=True)

        assert len(results) == 1
        assert not results[0].passed

    def test_run_all_gates_continues_on_optional_failure(self, tmp_path):
        """Test that execution continues on optional gate failure."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: optional_fail
    command: exit 1
    timeout: 10
    required: false
  - name: should_run
    command: echo runs
    timeout: 10
    required: true
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = runner.run_all_gates(stop_on_first_failure=True)

        assert len(results) == 2
        assert not results[0].passed
        assert results[1].passed

    def test_run_all_gates_required_only(self, tmp_path):
        """Test running only required gates."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: required
    command: echo required
    timeout: 10
    required: true
  - name: optional
    command: echo optional
    timeout: 10
    required: false
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = runner.run_all_gates(required_only=True)

        assert len(results) == 1
        assert results[0].gate_name == "required"


class TestRunGatesByName:
    """Test running gates by name."""

    def test_run_gates_by_name(self, tmp_path):
        """Test running specific gates by name."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: gate1
    command: echo gate1
    timeout: 10
  - name: gate2
    command: echo gate2
    timeout: 10
  - name: gate3
    command: echo gate3
    timeout: 10
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = runner.run_gates_by_name(["gate1", "gate3"])

        assert len(results) == 2
        assert results[0].gate_name == "gate1"
        assert results[1].gate_name == "gate3"

    def test_run_gates_unknown_name(self, tmp_path):
        """Test running gate with unknown name."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: gate1
    command: echo gate1
    timeout: 10
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = runner.run_gates_by_name(["nonexistent"])

        assert len(results) == 1
        assert not results[0].passed
        assert "Unknown gate" in results[0].errors[0]


class TestGetSummary:
    """Test summary generation."""

    def test_get_summary(self, tmp_path):
        """Test get_summary method."""
        runner = GateRunner(config_path=tmp_path / "nonexistent.yaml")
        results = [
            GateResult(
                gate_name="pass",
                passed=True,
                exit_code=0,
                output="",
                elapsed_seconds=1.0,
            ),
            GateResult(
                gate_name="fail",
                passed=False,
                exit_code=1,
                output="",
                elapsed_seconds=2.0,
                errors=["error1"],
            ),
        ]
        summary = runner.get_summary(results)

        assert "Gate Results" in summary
        assert "[PASSED] pass" in summary
        assert "[FAILED] fail" in summary
        assert "1/2 passed" in summary
        assert "1 failed" in summary


class TestGateConfigParallelizable:
    """Test parallelizable field on GateConfig."""

    def test_default_not_parallelizable(self):
        """GateConfig defaults to parallelizable=False."""
        config = GateConfig(name="test", command="echo hello")
        assert config.parallelizable is False

    def test_from_dict_parallelizable(self):
        """from_dict reads parallelizable flag."""
        data = {"name": "lint", "command": "make lint", "parallelizable": True}
        config = GateConfig.from_dict(data)
        assert config.parallelizable is True

    def test_from_dict_no_parallelizable(self):
        """from_dict defaults to False when not specified."""
        data = {"name": "unit", "command": "make test-unit"}
        config = GateConfig.from_dict(data)
        assert config.parallelizable is False


class TestRunAllGatesParallel:
    """Test async parallel gate execution."""

    @pytest.mark.asyncio
    async def test_parallel_gates_run_concurrently(self, tmp_path):
        """Two fast parallelizable gates complete faster than sequential."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: gate1
    command: "sleep 0.3 && echo gate1"
    timeout: 10
    parallelizable: true
  - name: gate2
    command: "sleep 0.3 && echo gate2"
    timeout: 10
    parallelizable: true
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        import time
        start = time.perf_counter()
        results = await runner.run_all_gates_parallel()
        elapsed = time.perf_counter() - start

        assert len(results) == 2
        assert all(r.passed for r in results)
        # Sequential would take ~0.6s, parallel should take ~0.3s (+overhead)
        assert elapsed < 0.55

    @pytest.mark.asyncio
    async def test_parallel_then_sequential(self, tmp_path):
        """Parallel gates run first, then sequential gates."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: fast_lint
    command: echo lint
    timeout: 10
    parallelizable: true
  - name: fast_format
    command: echo format
    timeout: 10
    parallelizable: true
  - name: heavy_unit
    command: echo unit
    timeout: 10
    parallelizable: false
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = await runner.run_all_gates_parallel()

        assert len(results) == 3
        assert all(r.passed for r in results)
        # Results in config order
        assert results[0].gate_name == "fast_lint"
        assert results[1].gate_name == "fast_format"
        assert results[2].gate_name == "heavy_unit"

    @pytest.mark.asyncio
    async def test_parallel_required_failure_stops(self, tmp_path):
        """Required parallel gate failure stops sequential phase."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: bad_lint
    command: exit 1
    timeout: 10
    required: true
    parallelizable: true
  - name: heavy_unit
    command: echo should_not_run
    timeout: 10
    required: true
    parallelizable: false
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = await runner.run_all_gates_parallel(stop_on_first_failure=True)

        # Only parallel gate result returned, sequential never ran
        assert len(results) == 1
        assert results[0].gate_name == "bad_lint"
        assert not results[0].passed

    @pytest.mark.asyncio
    async def test_parallel_optional_failure_continues(self, tmp_path):
        """Optional parallel gate failure doesn't stop execution."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: optional_lint
    command: exit 1
    timeout: 10
    required: false
    parallelizable: true
  - name: heavy_unit
    command: echo runs
    timeout: 10
    required: true
    parallelizable: false
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = await runner.run_all_gates_parallel(stop_on_first_failure=True)

        assert len(results) == 2
        assert not results[0].passed  # optional lint failed
        assert results[1].passed  # unit ran and passed

    @pytest.mark.asyncio
    async def test_parallel_preserves_order(self, tmp_path):
        """Results returned in config order, not completion order."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: slow
    command: "sleep 0.2 && echo slow"
    timeout: 10
    parallelizable: true
  - name: fast
    command: echo fast
    timeout: 10
    parallelizable: true
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = await runner.run_all_gates_parallel()

        # Even though "fast" finishes first, "slow" appears first in results
        assert results[0].gate_name == "slow"
        assert results[1].gate_name == "fast"

    @pytest.mark.asyncio
    async def test_parallel_with_retries(self, tmp_path):
        """Gate with retry_count gets retried after parallel batch."""
        # Create a file that the gate checks for (simulates flaky gate)
        marker = tmp_path / "attempt_counter"
        marker.write_text("0")
        config_file = tmp_path / "gates.yaml"
        # First attempt fails, retry succeeds
        config_file.write_text(f"""
gates:
  - name: flaky
    command: "count=$(cat {marker}); count=$((count+1)); echo $count > {marker}; test $count -ge 2"
    timeout: 10
    retry_count: 1
    parallelizable: true
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = await runner.run_all_gates_parallel()

        assert len(results) == 1
        assert results[0].passed  # Passed on retry

    @pytest.mark.asyncio
    async def test_all_sequential_fallback(self, tmp_path):
        """When no parallelizable gates, behaves like run_all_gates."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: gate1
    command: echo gate1
    timeout: 10
  - name: gate2
    command: echo gate2
    timeout: 10
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = await runner.run_all_gates_parallel()

        assert len(results) == 2
        assert all(r.passed for r in results)

    @pytest.mark.asyncio
    async def test_parallel_required_only(self, tmp_path):
        """required_only=True filters before parallelization."""
        config_file = tmp_path / "gates.yaml"
        config_file.write_text("""
gates:
  - name: optional_lint
    command: echo lint
    timeout: 10
    required: false
    parallelizable: true
  - name: required_unit
    command: echo unit
    timeout: 10
    required: true
    parallelizable: false
""")
        runner = GateRunner(config_path=config_file, working_dir=tmp_path)
        results = await runner.run_all_gates_parallel(required_only=True)

        assert len(results) == 1
        assert results[0].gate_name == "required_unit"

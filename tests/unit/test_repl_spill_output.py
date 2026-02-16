#!/usr/bin/env python3
"""Unit tests for REPL output spill-to-file functionality."""

import uuid
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from src.repl_environment import REPLConfig, REPLEnvironment


def _make_repl(output_cap=100, llm_primitives=None, spill_dir=None):
    """Create a REPL with a small output_cap for testing spill behavior."""
    if spill_dir is None:
        spill_dir = f"/mnt/raid0/llm/tmp/repl_spill_test_{uuid.uuid4().hex[:8]}"
    config = REPLConfig(output_cap=output_cap, spill_dir=spill_dir)
    return REPLEnvironment(
        context="test",
        config=config,
        llm_primitives=llm_primitives,
    )


class TestSpillOutputSmallOutput:
    """Output below cap should pass through unchanged."""

    def test_no_spill_for_short_output(self):
        repl = _make_repl(output_cap=1000)
        result = repl.execute('print("hello")')
        assert result.output.strip() == "hello"
        assert "peek(" not in result.output

    def test_no_spill_file_created(self):
        repl = _make_repl(output_cap=1000)
        repl.execute('print("short")')
        p = Path(repl.config.spill_dir) / repl._session_id
        assert not p.exists()


class TestSpillOutputLargeOutput:
    """Output above cap should spill to file and return summary."""

    def test_spill_file_created(self):
        repl = _make_repl(output_cap=100)
        code = 'print("\\n".join(f"line {i}" for i in range(50)))'
        result = repl.execute(code)

        assert "chars" in result.output
        assert "peek(" in result.output
        assert result.error is None

        # Cleanup
        shutil.rmtree(repl.config.spill_dir, ignore_errors=True)

    def test_spill_file_contains_full_output(self):
        repl = _make_repl(output_cap=100)
        code = 'print("\\n".join(f"line {i}: data" for i in range(50)))'
        repl.execute(code)

        spill_dir = Path(repl.config.spill_dir) / repl._session_id
        spill_files = list(spill_dir.glob("turn_*.txt"))
        assert len(spill_files) == 1

        content = spill_files[0].read_text()
        assert "line 0: data" in content
        assert "line 49: data" in content

        shutil.rmtree(repl.config.spill_dir, ignore_errors=True)

    def test_summary_contains_head_lines(self):
        repl = _make_repl(output_cap=100)
        code = 'print("\\n".join(f"line {i}" for i in range(50)))'
        result = repl.execute(code)

        assert "line 0" in result.output

        shutil.rmtree(repl.config.spill_dir, ignore_errors=True)

    def test_summary_contains_tail_lines(self):
        repl = _make_repl(output_cap=100)
        code = 'print("\\n".join(f"line {i}" for i in range(50)))'
        result = repl.execute(code)

        assert "line 49" in result.output

        shutil.rmtree(repl.config.spill_dir, ignore_errors=True)

    def test_multiple_executions_create_separate_files(self):
        repl = _make_repl(output_cap=100)
        code = 'print("x" * 200)'
        repl.execute(code)
        repl.execute(code)

        spill_dir = Path(repl.config.spill_dir) / repl._session_id
        spill_files = list(spill_dir.glob("turn_*.txt"))
        assert len(spill_files) == 2

        shutil.rmtree(repl.config.spill_dir, ignore_errors=True)


class TestSpillOutputWithWorkerSummary:
    """When llm_primitives is available, worker summary should be used."""

    def test_worker_summary_used(self):
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = "Summary: 50 lines of test data, all passed."

        repl = _make_repl(output_cap=100, llm_primitives=mock_llm)
        code = 'print("\\n".join(f"line {i}" for i in range(50)))'
        result = repl.execute(code)

        assert "Summary: 50 lines" in result.output
        mock_llm.llm_call.assert_called_once()
        # First call should not mention "Previous summary" in context
        call_kwargs = mock_llm.llm_call.call_args
        assert "Previous summary" not in call_kwargs.kwargs.get("context_slice", "")

        shutil.rmtree(repl.config.spill_dir, ignore_errors=True)

    def test_rolling_summary_passes_previous(self):
        """Second spill should pass previous summary to worker for update."""
        mock_llm = MagicMock()
        mock_llm.llm_call.side_effect = [
            "Turn 1: generated 50 lines of data.",
            "Turn 1: 50 lines. Turn 2: 100 lines, all values doubled.",
        ]

        repl = _make_repl(output_cap=100, llm_primitives=mock_llm)

        # First spill
        repl.execute('print("\\n".join(f"line {i}" for i in range(50)))')
        assert repl._last_spill_summary == "Turn 1: generated 50 lines of data."

        # Second spill — should pass previous summary
        repl.execute('print("\\n".join(f"line {i*2}" for i in range(100)))')
        second_call_kwargs = mock_llm.llm_call.call_args
        ctx = second_call_kwargs.kwargs.get("context_slice", "")
        assert "Previous summary:" in ctx
        assert "Turn 1: generated 50 lines" in ctx
        assert "Update" in second_call_kwargs.kwargs.get("prompt", "")

        shutil.rmtree(repl.config.spill_dir, ignore_errors=True)

    def test_rolling_summary_stored(self):
        """_last_spill_summary should be updated after each worker call."""
        mock_llm = MagicMock()
        mock_llm.llm_call.return_value = "All 50 tests passed."

        repl = _make_repl(output_cap=100, llm_primitives=mock_llm)
        assert repl._last_spill_summary is None

        repl.execute('print("\\n".join(f"line {i}" for i in range(50)))')
        assert repl._last_spill_summary == "All 50 tests passed."

        shutil.rmtree(repl.config.spill_dir, ignore_errors=True)

    def test_worker_failure_falls_back_to_static(self):
        mock_llm = MagicMock()
        mock_llm.llm_call.side_effect = RuntimeError("connection refused")

        repl = _make_repl(output_cap=100, llm_primitives=mock_llm)
        code = 'print("\\n".join(f"line {i}" for i in range(50)))'
        result = repl.execute(code)

        assert "line 0" in result.output
        assert "peek(" in result.output
        # Summary should not be stored on failure
        assert repl._last_spill_summary is None

        shutil.rmtree(repl.config.spill_dir, ignore_errors=True)


class TestSpillOutputStructuredMode:
    """Spill-to-file should work in structured mode too."""

    def test_structured_mode_spill(self):
        spill_dir = f"/mnt/raid0/llm/tmp/repl_spill_test_{uuid.uuid4().hex[:8]}"
        config = REPLConfig(
            output_cap=100,
            spill_dir=spill_dir,
            structured_mode=True,
        )
        repl = REPLEnvironment(context="test", config=config)
        code = 'print("\\n".join(f"line {i}" for i in range(50)))'
        result = repl.execute(code)

        assert "peek(" in result.output
        assert result.error is None

        shutil.rmtree(spill_dir, ignore_errors=True)

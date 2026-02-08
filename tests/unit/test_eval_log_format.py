"""Golden-output tests for eval_log_format.py.

These tests prevent accidental regressions to the eval script's
terminal logging format. The formatter is a pure module — functions
take data and return list[str] — so tests call them directly without
mocking the evaluation pipeline.

To update after an INTENTIONAL format change:
  1. Edit scripts/benchmark/eval_log_format.py
  2. Run these tests — they'll fail showing old vs new
  3. Update expected patterns below
  4. Document why the format changed
"""

import logging
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, "scripts/benchmark")

from eval_log_format import (
    compute_tps,
    format_self_direct,
    format_self_repl,
    format_architect_result,
    format_reward_skip,
    format_all_infra_skip,
    _dedup_consecutive,
    _format_delegation_events,
)


# ── Test data ─────────────────────────────────────────────────────────

_RESP_DIRECT = {
    "tokens_generated": 85,
    "predicted_tps": 23.5,
    "generation_ms": 3600.0,
    "tools_used": 0,
    "tools_called": [],
    "tool_timings": [],
    "delegation_events": [],
    "role_history": ["frontdoor"],
}

_RESP_REPL = {
    "tokens_generated": 240,
    "predicted_tps": 18.3,
    "generation_ms": 13100.0,
    "tools_used": 3,
    "tools_called": ["peek", "grep", "FINAL"],
    "tool_timings": [
        {"tool_name": "peek", "elapsed_ms": 120.0, "success": True},
        {"tool_name": "grep", "elapsed_ms": 85.0, "success": True},
        {"tool_name": "FINAL", "elapsed_ms": 10.0, "success": True},
    ],
    "delegation_events": [],
    "role_history": ["frontdoor"],
}

_RESP_ARCH_WITH_DELEGATION = {
    "tokens_generated": 724,
    "predicted_tps": 6.8,
    "generation_ms": 106500.0,
    "tools_used": 2,
    "tools_called": ["peek", "grep"],
    "tool_timings": [
        {"tool_name": "peek", "elapsed_ms": 200.0, "success": True},
        {"tool_name": "grep", "elapsed_ms": 150.0, "success": True},
    ],
    "delegation_events": [
        {
            "from_role": "architect_general",
            "to_role": "coder_escalation",
            "task_summary": "Implement solution",
            "success": True,
            "elapsed_ms": 42300.0,
            "tokens_generated": 774,
        },
        {
            "from_role": "architect_general",
            "to_role": "worker_explore",
            "task_summary": "Search codebase",
            "success": True,
            "elapsed_ms": 8200.0,
            "tokens_generated": 362,
        },
    ],
    "role_history": ["architect_general", "coder_escalation", "worker_explore"],
}

_RESP_ARCH_SINGLE_DELEGATE = {
    "tokens_generated": 850,
    "predicted_tps": 10.1,
    "generation_ms": 84100.0,
    "tools_used": 0,
    "tools_called": [],
    "tool_timings": [],
    "delegation_events": [
        {
            "from_role": "architect_coding",
            "to_role": "coder_escalation",
            "success": True,
            "elapsed_ms": 31500.0,
            "tokens_generated": 715,
        },
    ],
    "role_history": ["architect_coding", "coder_escalation"],
}


# ── Helpers ───────────────────────────────────────────────────────────


class TestComputeTps:
    def test_uses_predicted_tps_when_available(self):
        assert compute_tps({"predicted_tps": 23.5}) == 23.5

    def test_computes_from_generation_ms(self):
        tps = compute_tps({"predicted_tps": 0, "generation_ms": 1000.0, "tokens_generated": 50})
        assert tps == 50.0

    def test_returns_zero_when_no_data(self):
        assert compute_tps({}) == 0.0

    def test_returns_zero_when_generation_ms_zero(self):
        assert compute_tps({"predicted_tps": 0, "generation_ms": 0, "tokens_generated": 50}) == 0.0


class TestDedupConsecutive:
    def test_dedup_basic(self):
        assert _dedup_consecutive(["a", "a", "b", "b", "a"]) == ["a", "b", "a"]

    def test_empty(self):
        assert _dedup_consecutive([]) == []

    def test_no_dupes(self):
        assert _dedup_consecutive(["a", "b", "c"]) == ["a", "b", "c"]


# ── SELF:direct format ────────────────────────────────────────────────


class TestFormatSelfDirect:
    """SELF:direct → PASS (4.6s, 23.5 t/s, 85 tok)"""

    def test_pass_format(self):
        lines = format_self_direct("SELF:direct", True, None, 4.6, _RESP_DIRECT)
        assert len(lines) == 1
        assert "SELF:direct → PASS" in lines[0]
        assert "23.5 t/s" in lines[0]
        assert "85 tok" in lines[0]

    def test_fail_format(self):
        lines = format_self_direct("SELF:direct", False, None, 3.2, _RESP_DIRECT)
        assert "→ FAIL" in lines[0]

    def test_error_format(self):
        lines = format_self_direct("SELF:direct", False, "timeout", 120.0, _RESP_DIRECT)
        assert "→ ERROR" in lines[0]

    def test_infra_format(self):
        lines = format_self_direct("SELF:direct", False, "timeout", 120.0, _RESP_DIRECT, infra=True)
        assert "→ INFRA" in lines[0]

    def test_no_tps_omits_field(self):
        resp = {**_RESP_DIRECT, "predicted_tps": 0, "generation_ms": 0}
        lines = format_self_direct("SELF:direct", True, None, 1.0, resp)
        assert "t/s" not in lines[0]
        assert "tok" in lines[0]


# ── SELF:repl format ──────────────────────────────────────────────────


class TestFormatSelfRepl:
    """SELF:repl → PASS (16.2s, 18.3 t/s, 240 tok, 3 tools)
         tools: peek, grep, FINAL
         peek: 120ms (ok)
    """

    def test_status_line_includes_tool_count(self):
        lines = format_self_repl("SELF:repl", True, None, 16.2, _RESP_REPL)
        assert "3 tools" in lines[0]
        assert "18.3 t/s" in lines[0]

    def test_tool_list_present(self):
        lines = format_self_repl("SELF:repl", True, None, 16.2, _RESP_REPL)
        tool_lines = [ln for ln in lines if "tools:" in ln]
        assert len(tool_lines) == 1
        assert "peek, grep, FINAL" in tool_lines[0]

    def test_per_tool_timing_present(self):
        lines = format_self_repl("SELF:repl", True, None, 16.2, _RESP_REPL)
        timing_lines = [ln for ln in lines if "ms (" in ln and "tools:" not in ln]
        assert len(timing_lines) == 3
        assert "peek: 120ms (ok)" in timing_lines[0]
        assert "grep: 85ms (ok)" in timing_lines[1]
        assert "FINAL: 10ms (ok)" in timing_lines[2]

    def test_no_tools_omits_tool_lines(self):
        resp = {**_RESP_REPL, "tools_used": 0, "tools_called": [], "tool_timings": []}
        lines = format_self_repl("SELF:repl", True, None, 1.0, resp)
        assert len(lines) == 1  # Just the status line

    def test_tool_dedup(self):
        resp = {**_RESP_REPL, "tools_called": ["peek", "peek", "grep", "grep", "peek"]}
        lines = format_self_repl("SELF:repl", True, None, 1.0, resp)
        tool_lines = [ln for ln in lines if "tools:" in ln]
        assert "peek, grep, peek" in tool_lines[0]

    def test_failed_tool(self):
        resp = {
            **_RESP_REPL,
            "tool_timings": [{"tool_name": "fetch", "elapsed_ms": 500.0, "success": False}],
        }
        lines = format_self_repl("SELF:repl", True, None, 1.0, resp)
        timing_lines = [ln for ln in lines if "fetch:" in ln]
        assert "fail" in timing_lines[0]

    def test_infra_format(self):
        lines = format_self_repl("SELF:repl", False, "timeout", 120.0, _RESP_REPL, infra=True)
        assert "→ INFRA" in lines[0]


# ── ARCHITECT format ──────────────────────────────────────────────────


class TestFormatArchitectResult:
    """ARCHITECT → PASS (106.7s, 6.8 t/s, 724 tok)
         tools: peek, grep
         peek: 200ms (ok)
         delegates: 2 (coder_escalation, worker_explore)
         delegate: coder_escalation → ok (42300ms, 18.3 t/s, 774 tok)
         chain: architect_general → coder_escalation → worker_explore
    """

    def test_status_line(self):
        lines = format_architect_result("ARCHITECT", True, None, 106.7, _RESP_ARCH_WITH_DELEGATION)
        assert "ARCHITECT → PASS" in lines[0]
        assert "6.8 t/s" in lines[0]
        assert "724 tok" in lines[0]

    def test_infra_status(self):
        lines = format_architect_result("ARCHITECT", None, "timeout", 120.0, _RESP_ARCH_WITH_DELEGATION)
        assert "→ INFRA" in lines[0]

    def test_tool_list_present(self):
        lines = format_architect_result("ARCHITECT", True, None, 106.7, _RESP_ARCH_WITH_DELEGATION)
        tool_lines = [ln for ln in lines if "tools:" in ln]
        assert len(tool_lines) == 1
        assert "peek, grep" in tool_lines[0]

    def test_per_tool_timing(self):
        lines = format_architect_result("ARCHITECT", True, None, 106.7, _RESP_ARCH_WITH_DELEGATION)
        timing_lines = [ln for ln in lines if "ms (ok)" in ln and "delegate" not in ln]
        assert len(timing_lines) == 2

    def test_delegate_summary_for_multiple(self):
        lines = format_architect_result("ARCHITECT", True, None, 106.7, _RESP_ARCH_WITH_DELEGATION)
        summary = [ln for ln in lines if "delegates:" in ln]
        assert len(summary) == 1
        assert "2" in summary[0]
        assert "coder_escalation" in summary[0]
        assert "worker_explore" in summary[0]

    def test_no_delegate_summary_for_single(self):
        lines = format_architect_result("ARCHITECT", True, None, 84.2, _RESP_ARCH_SINGLE_DELEGATE)
        summary = [ln for ln in lines if "delegates:" in ln]
        assert len(summary) == 0  # Only 1 delegate, no summary

    def test_delegate_detail_lines(self):
        lines = format_architect_result("ARCHITECT", True, None, 106.7, _RESP_ARCH_WITH_DELEGATION)
        delegate_lines = [ln for ln in lines if "delegate:" in ln and "→" in ln]
        assert len(delegate_lines) == 2
        assert "coder_escalation → ok" in delegate_lines[0]
        assert "42300ms" in delegate_lines[0]
        assert "774 tok" in delegate_lines[0]
        assert "18.3 t/s" in delegate_lines[0]

    def test_delegate_with_tokens_shows_tps(self):
        lines = format_architect_result("ARCHITECT", True, None, 84.2, _RESP_ARCH_SINGLE_DELEGATE)
        delegate_lines = [ln for ln in lines if "delegate:" in ln]
        assert len(delegate_lines) == 1
        assert "t/s" in delegate_lines[0]
        assert "715 tok" in delegate_lines[0]

    def test_delegate_without_tokens_omits_tps(self):
        resp = {
            **_RESP_ARCH_SINGLE_DELEGATE,
            "delegation_events": [{
                "to_role": "coder", "success": True,
                "elapsed_ms": 5000.0, "tokens_generated": 0,
            }],
        }
        lines = format_architect_result("ARCHITECT", True, None, 10.0, resp)
        delegate_lines = [ln for ln in lines if "delegate:" in ln]
        assert "t/s" not in delegate_lines[0]
        assert "0 tok" in delegate_lines[0]

    def test_failed_delegate(self):
        resp = {
            **_RESP_ARCH_SINGLE_DELEGATE,
            "delegation_events": [{
                "to_role": "worker_explore", "success": False,
                "elapsed_ms": 3000.0, "tokens_generated": 0,
            }],
        }
        lines = format_architect_result("ARCHITECT", False, None, 10.0, resp)
        delegate_lines = [ln for ln in lines if "delegate:" in ln]
        assert "→ fail" in delegate_lines[0]

    def test_unknown_delegate_success(self):
        resp = {
            **_RESP_ARCH_SINGLE_DELEGATE,
            "delegation_events": [{
                "to_role": "worker", "success": None,
                "elapsed_ms": 1000.0, "tokens_generated": 0,
            }],
        }
        lines = format_architect_result("ARCHITECT", True, None, 5.0, resp)
        delegate_lines = [ln for ln in lines if "delegate:" in ln]
        assert "→ ?" in delegate_lines[0]

    def test_role_chain_present(self):
        lines = format_architect_result("ARCHITECT", True, None, 106.7, _RESP_ARCH_WITH_DELEGATION)
        chain_lines = [ln for ln in lines if "chain:" in ln]
        assert len(chain_lines) == 1
        assert "architect_general → coder_escalation → worker_explore" in chain_lines[0]

    def test_no_chain_for_single_role(self):
        resp = {**_RESP_ARCH_SINGLE_DELEGATE, "role_history": ["architect_coding"]}
        lines = format_architect_result("ARCHITECT", True, None, 10.0, resp)
        chain_lines = [ln for ln in lines if "chain:" in ln]
        assert len(chain_lines) == 0

    def test_no_tools_no_delegation(self):
        resp = {
            "tokens_generated": 100, "predicted_tps": 5.0,
            "tools_used": 0, "tools_called": [], "tool_timings": [],
            "delegation_events": [], "role_history": ["architect_general"],
        }
        lines = format_architect_result("ARCHITECT", True, None, 20.0, resp)
        assert len(lines) == 1  # Just the status line

    def test_parallel_workers_summary(self):
        """Multiple delegates to the same role show Nx counts."""
        resp = {
            "tokens_generated": 500, "predicted_tps": 5.0,
            "tools_used": 0, "tools_called": [], "tool_timings": [],
            "delegation_events": [
                {"to_role": "worker_explore", "success": True, "elapsed_ms": 8000, "tokens_generated": 300},
                {"to_role": "worker_explore", "success": True, "elapsed_ms": 7500, "tokens_generated": 280},
                {"to_role": "coder_escalation", "success": True, "elapsed_ms": 30000, "tokens_generated": 700},
            ],
            "role_history": ["architect_general"],
        }
        lines = format_architect_result("ARCHITECT", True, None, 50.0, resp)
        summary = [ln for ln in lines if "delegates:" in ln]
        assert len(summary) == 1
        assert "3" in summary[0]
        assert "2x worker_explore" in summary[0]


# ── Reward skip format ────────────────────────────────────────────────


class TestFormatRewardSkip:
    def test_infra_skip(self):
        lines = format_reward_skip("SELF:direct")
        assert len(lines) == 1
        assert "SELF:direct -> INFRA_SKIP" in lines[0]

    def test_all_infra_skip(self):
        lines = format_all_infra_skip("ARCHITECT")
        assert len(lines) == 1
        assert "ARCHITECT -> ALL INFRA_SKIP" in lines[0]


# ── Delegation events sub-formatter ───────────────────────────────────


class TestFormatDelegationEvents:
    def test_single_event_no_summary(self):
        events = [{"to_role": "coder", "success": True, "elapsed_ms": 5000, "tokens_generated": 100}]
        lines = _format_delegation_events(events)
        assert len(lines) == 1  # No summary line
        assert "delegate: coder" in lines[0]

    def test_multiple_events_has_summary(self):
        events = [
            {"to_role": "coder", "success": True, "elapsed_ms": 5000, "tokens_generated": 100},
            {"to_role": "worker", "success": True, "elapsed_ms": 3000, "tokens_generated": 50},
        ]
        lines = _format_delegation_events(events)
        assert len(lines) == 3  # 1 summary + 2 detail
        assert "delegates: 2" in lines[0]


# ── Integration: end-to-end eval with mocked backends ─────────────────

_MOCK_RESP_DIRECT = {**_RESP_DIRECT, "answer": "42", "routed_to": "frontdoor"}
_MOCK_RESP_REPL = {**_RESP_REPL, "answer": "42", "routed_to": "frontdoor"}
_MOCK_RESP_ARCH_GENERAL = {**_RESP_ARCH_WITH_DELEGATION, "answer": "42", "routed_to": "architect_general"}
_MOCK_RESP_ARCH_CODING = {**_RESP_ARCH_SINGLE_DELEGATE, "answer": "42", "routed_to": "architect_coding"}


def _mock_call_orchestrator(prompt, force_role, force_mode, **kwargs):
    """Return canned responses based on force_role."""
    if force_role == "frontdoor" and force_mode == "direct":
        return _MOCK_RESP_DIRECT
    elif force_role == "frontdoor" and force_mode == "repl":
        return _MOCK_RESP_REPL
    elif force_role == "architect_general":
        return _MOCK_RESP_ARCH_GENERAL
    elif force_role == "architect_coding":
        return _MOCK_RESP_ARCH_CODING
    return {"answer": "", "error": "unknown role"}


class TestEvalLogIntegration:
    """Integration test: runs evaluate_question_3way and verifies output ordering.

    This catches wiring bugs (formatter not called, wrong args, etc.)
    that unit tests on the formatter alone would miss.
    """

    @pytest.fixture
    def log_capture(self, caplog):
        with caplog.at_level(logging.INFO, logger="seed_specialist_routing"):
            yield caplog

    def _run_eval(self, log_capture):
        import httpx

        prompt_info = {
            "id": "test_q1",
            "suite": "thinking",
            "prompt": "What is the answer to life?",
            "expected": "42",
            "scoring_method": "exact_match",
        }
        with (
            patch("seed_specialist_routing.call_orchestrator_forced", side_effect=_mock_call_orchestrator),
            patch("seed_specialist_routing.score_answer_deterministic", return_value=True),
            patch("seed_specialist_routing._wait_for_heavy_models_idle"),
        ):
            from seed_specialist_routing import evaluate_question_3way

            client = httpx.Client(timeout=10)
            try:
                evaluate_question_3way(
                    prompt_info=prompt_info,
                    url="http://localhost:8000",
                    timeout=120,
                    client=client,
                    dry_run=True,
                )
            finally:
                client.close()

        return [rec.message for rec in log_capture.records]

    def test_output_ordering(self, log_capture):
        """SELF:direct → SELF:repl → ARCHITECT (general) → ARCHITECT (coding)."""
        lines = self._run_eval(log_capture)

        direct_idx = next(
            (i for i, ln in enumerate(lines) if "SELF:direct →" in ln and "tok" in ln), -1
        )
        repl_idx = next(
            (i for i, ln in enumerate(lines) if "SELF:repl →" in ln and "tok" in ln), -1
        )
        arch_idx = next(
            (i for i, ln in enumerate(lines) if "ARCHITECT →" in ln and "tok" in ln), -1
        )

        assert direct_idx >= 0, f"SELF:direct missing. Lines: {lines}"
        assert repl_idx >= 0, f"SELF:repl missing. Lines: {lines}"
        assert arch_idx >= 0, f"ARCHITECT missing. Lines: {lines}"
        assert direct_idx < repl_idx < arch_idx, (
            f"Wrong order: direct={direct_idx}, repl={repl_idx}, arch={arch_idx}"
        )

    def test_all_detail_sections_present(self, log_capture):
        """Verify tools, tool_timings, delegates, and chain all appear."""
        lines = self._run_eval(log_capture)
        joined = "\n".join(lines)

        assert "tools:" in joined, f"Missing tool list. Lines:\n{joined}"
        assert "ms (ok)" in joined, f"Missing per-tool timing. Lines:\n{joined}"
        assert "delegate:" in joined, f"Missing delegation details. Lines:\n{joined}"
        assert "delegates:" in joined, f"Missing delegation summary. Lines:\n{joined}"
        assert "chain:" in joined, f"Missing role chain. Lines:\n{joined}"

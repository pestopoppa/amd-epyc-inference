#!/usr/bin/env python3
"""Tests for architect delegation (investigate via specialist tools).

Tests the TOON/JSON parser, multi-loop delegation, feature flag gating,
and specialist document passthrough.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── Parser Tests ──────────────────────────────────────────────────────────


class TestParseArchitectDecision:
    """Tests for _parse_architect_decision()."""

    def test_toon_direct(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision("D|The answer is 42")
        assert result["mode"] == "direct"
        assert result["answer"] == "The answer is 42"
        assert result["brief"] == ""

    def test_toon_investigate(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision(
            "I|brief:Check src/api.py for error handling|to:coder_escalation"
        )
        assert result["mode"] == "investigate"
        assert "Check src/api.py" in result["brief"]
        assert result["delegate_to"] == "coder_escalation"
        assert result["delegate_mode"] == "react"

    def test_toon_investigate_repl_mode(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision(
            "I|brief:Draft implementation doc|to:coder_escalation|mode:repl"
        )
        assert result["mode"] == "investigate"
        assert result["delegate_to"] == "coder_escalation"
        assert result["delegate_mode"] == "repl"

    def test_toon_investigate_worker_coder_role(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision(
            "I|brief:split into parallel file tasks|to:worker_coder|mode:repl"
        )
        assert result["mode"] == "investigate"
        assert result["delegate_to"] == "worker_coder"
        assert result["delegate_mode"] == "repl"

    def test_json_direct(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        payload = json.dumps({"mode": "direct", "answer": "The answer is 42"})
        result = _parse_architect_decision(payload)
        assert result["mode"] == "direct"
        assert result["answer"] == "The answer is 42"

    def test_json_investigate(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        payload = json.dumps(
            {
                "mode": "investigate",
                "brief": "Search for X",
                "to": "worker_explore",
            }
        )
        result = _parse_architect_decision(payload)
        assert result["mode"] == "investigate"
        assert result["brief"] == "Search for X"
        assert result["delegate_to"] == "worker_explore"

    def test_markdown_wrapped_json(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        wrapped = '```json\n{"mode": "direct", "answer": "hello"}\n```'
        result = _parse_architect_decision(wrapped)
        assert result["mode"] == "direct"
        assert result["answer"] == "hello"

    def test_fallback_bare_text(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision("Just a plain text answer with no format")
        assert result["mode"] == "direct"
        assert result["answer"] == "Just a plain text answer with no format"

    def test_invalid_role_clamped(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision("I|brief:Check something|to:nonexistent_role")
        assert result["mode"] == "investigate"
        assert result["delegate_to"] == "coder_escalation"

    def test_invalid_mode_clamped(self):
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision("I|brief:Check something|to:coder_escalation|mode:invalid")
        assert result["delegate_mode"] == "react"


# ── Delegation Flow Tests ─────────────────────────────────────────────────


class TestArchitectDelegatedAnswer:
    """Tests for _architect_delegated_answer()."""

    def _mock_primitives(self, responses):
        """Create mock LLMPrimitives that returns responses in sequence."""
        primitives = MagicMock()
        primitives.llm_call = MagicMock(side_effect=responses)
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 100
        primitives.get_cache_stats.return_value = None
        return primitives

    def _mock_state(self):
        """Create mock AppState."""
        state = MagicMock()
        state.tool_registry = None
        return state

    def test_direct_answer_no_loops(self):
        """Architect answers directly — no delegation loops."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = self._mock_primitives(["D|The answer is 42"])
        state = self._mock_state()

        answer, stats = _architect_delegated_answer(
            question="What is 6*7?",
            context="",
            primitives=primitives,
            state=state,
        )

        assert answer == "The answer is 42"
        assert stats["loops"] == 0
        assert primitives.llm_call.call_count == 1

    def test_one_investigation_loop(self):
        """Architect investigates once, then synthesizes."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        responses = [
            # Loop 0 Phase A: architect decides to investigate
            "I|brief:Check the file src/api.py|to:coder_escalation",
            # Loop 0 Phase B: specialist ReAct (mocked via _react_mode_answer)
            # Loop 1 Phase A: architect synthesizes
            "D|Based on the investigation, the answer is X",
        ]

        primitives = self._mock_primitives(responses)
        state = self._mock_state()

        # Mock REPLEnvironment to avoid needing real REPL loop
        with patch("src.api.routes.chat_delegation.REPLEnvironment") as mock_repl_cls:
            mock_repl = MagicMock()
            mock_repl.get_prompt.return_value = "prompt"
            mock_repl.execute.return_value = {"final": "File contents: error handling at line 42"}
            mock_repl._tool_invocations = 2
            mock_repl.tool_registry = None
            mock_repl_cls.return_value = mock_repl

            # Need to override llm_call side effects to include REPL code call
            primitives.llm_call.side_effect = [
                responses[0],  # architect decides to investigate
                "code here",  # REPL code
                responses[1],  # architect synthesizes
            ]

            answer, stats = _architect_delegated_answer(
                question="Where is error handling?",
                context="",
                primitives=primitives,
                state=state,
            )

        assert "answer is X" in answer
        assert stats["loops"] == 1
        assert len(stats["phases"]) >= 2

    def test_multi_loop_investigation(self):
        """Architect requests two investigations before answering."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        responses = [
            # Loop 0 Phase A: first investigation request
            "I|brief:Check file A|to:coder_escalation",
            # Loop 1 Phase A: second investigation request
            "I|brief:Check file B|to:worker_explore",
            # Loop 2 Phase A: final answer
            "D|After investigating A and B, the answer is Z",
        ]

        primitives = self._mock_primitives(responses)
        state = self._mock_state()

        call_count = [0]
        reports = ["Report from A", "Report from B"]
        with patch("src.api.routes.chat_delegation.REPLEnvironment") as mock_repl_cls:
            def make_repl(*a, **kw):
                mock_repl = MagicMock()
                mock_repl.get_prompt.return_value = "prompt"
                idx = min(call_count[0], len(reports) - 1)
                mock_repl.execute.return_value = {"final": reports[idx]}
                mock_repl._tool_invocations = 1
                mock_repl.tool_registry = None
                call_count[0] += 1
                return mock_repl
            mock_repl_cls.side_effect = make_repl

            primitives.llm_call.side_effect = [
                responses[0],  # investigate A
                "code",  # REPL code for A
                responses[1],  # investigate B
                "code",  # REPL code for B
                responses[2],  # final answer
            ]

            answer, stats = _architect_delegated_answer(
                question="Compare A and B",
                context="",
                primitives=primitives,
                state=state,
                max_loops=3,
            )

        assert "answer is Z" in answer
        assert stats["loops"] == 2

    def test_max_loops_cap_forces_response(self):
        """When architect keeps investigating past max_loops, force synthesis."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        responses = [
            # Loop 0: investigate
            "I|brief:Check X|to:coder_escalation",
            # Loop 1: investigate again
            "I|brief:Check Y|to:coder_escalation",
            # Loop 2: investigate AGAIN (would exceed max_loops=2)
            "I|brief:Check Z|to:coder_escalation",
            # Forced synthesis after cap
            "Forced answer from all reports",
        ]

        primitives = self._mock_primitives(responses)
        state = self._mock_state()

        call_count = [0]
        reports = ["Report X", "Report Y", "Report Z"]
        with patch("src.api.routes.chat_delegation.REPLEnvironment") as mock_repl_cls:
            def make_repl(*a, **kw):
                mock_repl = MagicMock()
                mock_repl.get_prompt.return_value = "prompt"
                idx = min(call_count[0], len(reports) - 1)
                mock_repl.execute.return_value = {"final": reports[idx]}
                mock_repl._tool_invocations = 1
                mock_repl.tool_registry = None
                call_count[0] += 1
                return mock_repl
            mock_repl_cls.side_effect = make_repl

            primitives.llm_call.side_effect = [
                responses[0],  # investigate X
                "code",  # REPL code
                responses[1],  # investigate Y
                "code",  # REPL code
                responses[2],  # investigate Z (exceeds cap)
                "code",  # REPL code
                responses[3],  # forced synthesis
            ]

            answer, stats = _architect_delegated_answer(
                question="Complex question",
                context="",
                primitives=primitives,
                state=state,
                max_loops=2,
                force_response_on_cap=True,
            )

        assert answer  # Should have a forced answer
        assert stats["loops"] == 2

    def test_specialist_document_passthrough(self):
        """Specialist drafts document, architect says 'Approved' → document is answer."""
        from src.api.routes.chat_delegation import _architect_delegated_answer

        responses = [
            # Loop 0: architect requests REPL drafting
            "I|brief:Draft the implementation|to:coder_escalation|mode:repl",
            # Loop 1: architect approves
            "D|Approved",
        ]

        primitives = self._mock_primitives(responses)
        state = self._mock_state()

        specialist_doc = "def hello():\n    return 'world'"

        with patch(
            "src.api.routes.chat_delegation.REPLEnvironment",
        ) as mock_repl_cls:
            # Mock REPL to produce a document
            mock_repl = MagicMock()
            mock_repl.get_prompt.return_value = "prompt"
            mock_result = MagicMock()
            mock_result.is_final = True
            mock_result.final_answer = specialist_doc
            mock_result.output = ""
            mock_result.error = None
            mock_repl.execute.return_value = mock_result
            mock_repl.get_state.return_value = ""
            mock_repl._tool_invocations = 0
            mock_repl_cls.return_value = mock_repl

            # Need to add the REPL primitives call
            primitives.llm_call.side_effect = [
                responses[0],  # architect decides to delegate
                "code here",  # specialist REPL code
                responses[1],  # architect approves
            ]

            answer, stats = _architect_delegated_answer(
                question="Write a hello function",
                context="",
                primitives=primitives,
                state=state,
            )

        # The specialist document should be the final answer
        assert answer == specialist_doc

    def test_delegate_role_validation(self):
        """Invalid delegate role gets clamped to coder_escalation."""
        from src.api.routes.chat_delegation import _parse_architect_decision

        result = _parse_architect_decision("I|brief:Do something|to:invalid_role_xyz")
        assert result["delegate_to"] == "coder_escalation"


# ── Feature Flag Gating ──────────────────────────────────────────────────


class TestFeatureFlagGating:
    """Tests for architect_delegation feature flag."""

    def test_feature_flag_exists(self):
        """Feature flag is defined in Features dataclass."""
        from src.features import Features

        f = Features()
        assert hasattr(f, "architect_delegation")
        assert f.architect_delegation is False

    def test_feature_flag_in_summary(self):
        """Feature flag appears in summary dict."""
        from src.features import Features

        f = Features(architect_delegation=True)
        summary = f.summary()
        assert "architect_delegation" in summary
        assert summary["architect_delegation"] is True

    def test_env_var_enables_flag(self):
        """ORCHESTRATOR_ARCHITECT_DELEGATION=1 enables the flag."""
        import os
        from src.features import get_features

        os.environ["ORCHESTRATOR_ARCHITECT_DELEGATION"] = "1"
        try:
            f = get_features()
            assert f.architect_delegation is True
        finally:
            del os.environ["ORCHESTRATOR_ARCHITECT_DELEGATION"]


class TestDelegationTokenCaps:
    """Tests for delegation token budgeting guards."""

    def test_specialist_turn_has_default_token_cap(self):
        from src.api.routes.chat_delegation import _run_specialist_loop

        primitives = MagicMock()
        primitives.llm_call = MagicMock(return_value="FINAL('ok')")

        with patch("src.api.routes.chat_delegation.REPLEnvironment") as mock_repl_cls:
            mock_repl = MagicMock()
            mock_result = MagicMock()
            mock_result.is_final = True
            mock_result.final_answer = "ok"
            mock_result.output = ""
            mock_result.error = None
            mock_repl.execute.return_value = mock_result
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl_cls.return_value = mock_repl

            _run_specialist_loop(
                question="q",
                context="",
                brief="b",
                delegate_to="coder_escalation",
                delegate_mode="repl",
                primitives=primitives,
                tool_registry=None,
            )

        kwargs = primitives.llm_call.call_args.kwargs
        assert kwargs.get("n_tokens") == 224

    def test_forced_synthesis_has_token_cap(self):
        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        # Loop 0: investigate; cap reached immediately (max_loops=0) -> forced synthesis
        primitives.llm_call = MagicMock(side_effect=[
            "I|brief:check|to:coder_escalation",
            "forced synthesis answer",
        ])

        state = MagicMock()
        state.tool_registry = None

        answer, _stats = _architect_delegated_answer(
            question="q",
            context="",
            primitives=primitives,
            state=state,
            max_loops=0,
            force_response_on_cap=True,
        )
        assert answer
        assert primitives.llm_call.call_args_list[-1].kwargs.get("n_tokens") == 128

    def test_timeout_break_skips_forced_synthesis_call(self):
        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        primitives.llm_call = MagicMock(return_value="should_not_be_called")

        state = MagicMock()
        state.tool_registry = None

        with patch(
            "src.api.routes.chat_delegation._run_architect_decision",
            return_value=("I|brief:investigate|to:coder_escalation", 1, 0),
        ), patch(
            "src.api.routes.chat_delegation._run_specialist_loop",
            return_value=("[Delegation timeout after 2 turn(s), 45.0s]", 0, [], [], True, False, {}),
        ):
            answer, stats = _architect_delegated_answer(
                question="q",
                context="",
                primitives=primitives,
                state=state,
                max_loops=2,
                force_response_on_cap=True,
            )

        assert stats.get("break_reason") == "specialist_timeout"
        assert isinstance(answer, str) and answer
        # No additional architect synthesis call should be made after timeout.
        assert primitives.llm_call.call_count == 0

    def test_specialist_code_report_rescue_avoids_extra_turns(self):
        from src.api.routes.chat_delegation import _run_specialist_loop

        raw = """
class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last = 0.0

    def refill(self, now):
        elapsed = now - self.last
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last = now

    def consume(self, n=1):
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

class RateLimiter:
    def __init__(self):
        self.buckets = {}
""".strip()
        primitives = MagicMock()
        primitives.llm_call = MagicMock(return_value=raw)

        with patch("src.api.routes.chat_delegation.REPLEnvironment") as mock_repl_cls:
            mock_repl = MagicMock()
            mock_result = MagicMock()
            mock_result.is_final = False
            mock_result.final_answer = ""
            mock_result.output = ""
            mock_result.error = ""
            mock_repl.execute.return_value = mock_result
            mock_repl.get_state.return_value = ""
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl_cls.return_value = mock_repl

            with patch("src.prompt_builders.extract_code_from_response", return_value=raw), patch(
                "src.prompt_builders.auto_wrap_final", return_value=raw
            ):
                report, _tools, _called, _timings, timed_out, report_rescued, _infer = _run_specialist_loop(
                    question="q",
                    context="",
                    brief="b",
                    delegate_to="coder_escalation",
                    delegate_mode="react",
                    primitives=primitives,
                    tool_registry=None,
                )

        assert timed_out is False
        assert report_rescued is True
        assert "class TokenBucket" in report
        # Single specialist generation turn only.
        assert primitives.llm_call.call_count == 1

    def test_architect_returns_rescued_specialist_report_directly(self):
        from src.api.routes.chat_delegation import _architect_delegated_answer

        primitives = MagicMock()
        primitives._backends = {"test": True}
        primitives.total_tokens_generated = 0
        primitives.llm_call = MagicMock(return_value="should_not_be_called")
        state = MagicMock()
        state.tool_registry = None

        with patch(
            "src.api.routes.chat_delegation._run_architect_decision",
            return_value=("I|brief:investigate|to:coder_escalation", 1, 0),
        ), patch(
            "src.api.routes.chat_delegation._run_specialist_loop",
            return_value=("specialist report body", 0, [], [], False, True, {}),
        ):
            answer, stats = _architect_delegated_answer(
                question="q",
                context="",
                primitives=primitives,
                state=state,
                max_loops=3,
                force_response_on_cap=True,
            )

        assert answer == "specialist report body"
        assert stats.get("break_reason") == "specialist_report"
        assert primitives.llm_call.call_count == 0


# ── Prompt Builder Tests ─────────────────────────────────────────────────


class TestArchitectPromptBuilders:
    """Tests for architect prompt template functions."""

    def test_investigate_prompt_has_decision_instructions(self):
        from src.prompt_builders import build_architect_investigate_prompt

        prompt = build_architect_investigate_prompt("What is X?")
        assert "D|" in prompt
        assert "I|" in prompt
        assert "brief:" in prompt
        assert "Question:" in prompt

    def test_investigate_prompt_includes_context(self):
        from src.prompt_builders import build_architect_investigate_prompt

        prompt = build_architect_investigate_prompt("What is X?", context="some ctx")
        assert "some ctx" in prompt

    def test_synthesis_prompt_includes_report(self):
        from src.prompt_builders import build_architect_synthesis_prompt

        prompt = build_architect_synthesis_prompt(
            "What is X?",
            "Report: found Y",
            loop_num=1,
            max_loops=3,
        )
        assert "Report: found Y" in prompt
        assert "Question:" in prompt

    def test_synthesis_prompt_no_reinvestigate_on_good_report(self):
        """Non-failed specialist report suppresses I| re-delegation option."""
        from src.prompt_builders import build_architect_synthesis_prompt

        prompt = build_architect_synthesis_prompt(
            "Q",
            "R",
            loop_num=2,
            max_loops=3,
        )
        assert "Do NOT" in prompt
        assert "I|brief" not in prompt

    def test_synthesis_prompt_reinvestigate_on_failed_report(self):
        """Failed specialist report allows re-delegation with loop count."""
        from src.prompt_builders import build_architect_synthesis_prompt

        prompt = build_architect_synthesis_prompt(
            "Q",
            "[ERROR: specialist crashed]",
            loop_num=1,
            max_loops=3,
        )
        assert "I|brief" in prompt
        assert "1" in prompt and "3" in prompt


# ── Whitelist Tests ──────────────────────────────────────────────────────


class TestFileToolsWhitelist:
    """Tests that file tools are in REACT_TOOL_WHITELIST."""

    def test_read_file_in_whitelist(self):
        from src.prompt_builders import REACT_TOOL_WHITELIST

        assert "read_file" in REACT_TOOL_WHITELIST

    def test_list_directory_in_whitelist(self):
        from src.prompt_builders import REACT_TOOL_WHITELIST

        assert "list_directory" in REACT_TOOL_WHITELIST

    def test_original_tools_still_present(self):
        from src.prompt_builders import REACT_TOOL_WHITELIST

        assert "web_search" in REACT_TOOL_WHITELIST
        assert "calculate" in REACT_TOOL_WHITELIST
        assert "python_eval" in REACT_TOOL_WHITELIST


# ── Seeding Script Tests ─────────────────────────────────────────────────


class TestSeedingScriptArchitectModes:
    """Tests for updated seeding script with architect delegation."""

    def test_architect_roles_constant(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "benchmark"))
        from seed_specialist_routing import ARCHITECT_ROLES, ARCHITECT_MODES

        assert "architect_general" in ARCHITECT_ROLES
        assert "architect_coding" in ARCHITECT_ROLES
        assert "direct" in ARCHITECT_MODES
        assert "delegated" in ARCHITECT_MODES

    def test_build_combos_architect_gets_delegation(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "benchmark"))
        from seed_specialist_routing import _build_role_mode_combos

        combos = _build_role_mode_combos(
            roles=["frontdoor", "architect_general"],
            modes=["direct", "react", "repl"],
        )

        # Frontdoor gets all 3 modes
        frontdoor_modes = [m for r, m in combos if r == "frontdoor"]
        assert set(frontdoor_modes) == {"direct", "react", "repl"}

        # Architect gets direct + delegated (not react/repl)
        arch_modes = [m for r, m in combos if r == "architect_general"]
        assert set(arch_modes) == {"direct", "delegated"}

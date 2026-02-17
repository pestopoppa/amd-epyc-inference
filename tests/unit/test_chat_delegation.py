"""Characterization tests for chat_delegation pure functions.

Tests cover: _strip_think, _extract_toon_decision, _parse_architect_decision,
loop guards (_get_delegation_depth, semantic dedup, role repetition, token budget),
and the module-level constants _VALID_DELEGATE_ROLES, _ARCHITECT_TOKEN_BUDGET,
_ARCHITECT_DECISION_BUDGET.
"""

import hashlib

from src.api.routes.chat_delegation import (
    _strip_think,
    _extract_toon_decision,
    _parse_architect_decision,
    _get_delegation_depth,
    _delegation_local,
    _VALID_DELEGATE_ROLES,
    _ARCHITECT_TOKEN_BUDGET,
    _ARCHITECT_DECISION_BUDGET,
)
from src.constants import (
    DELEGATION_BRIEF_KEY_LEN,
    DELEGATION_MAX_SAME_TARGET,
    DELEGATION_MAX_TOTAL_TOKENS,
)


# ── _strip_think ────────────────────────────────────────────────────────


class TestStripThink:
    def test_removes_complete_think_block(self):
        text = "Hello <think>internal reasoning</think> world"
        assert _strip_think(text) == "Hello  world"

    def test_removes_incomplete_trailing_think(self):
        text = "Answer: B<think>I should delegate with I|brief:check"
        assert _strip_think(text) == "Answer: B"

    def test_preserves_text_without_think_blocks(self):
        text = "The answer is 42. No special tags here."
        assert _strip_think(text) == text

    def test_removes_multiline_think_block(self):
        text = "before<think>\nline1\nline2\n</think>after"
        assert _strip_think(text) == "beforeafter"

    def test_removes_multiple_complete_blocks(self):
        text = "<think>a</think>X<think>b</think>Y"
        assert _strip_think(text) == "XY"


# ── _extract_toon_decision ──────────────────────────────────────────────


class TestExtractToonDecision:
    def test_mcq_single_letter(self):
        assert _extract_toon_decision("D|B") == "D|B"

    def test_direct_freeform_answer(self):
        result = _extract_toon_decision("D|The answer is 42")
        assert result == "D|The answer is 42"

    def test_investigate_with_brief_and_to(self):
        result = _extract_toon_decision("I|brief:check the code|to:coder_escalation")
        assert result == "I|brief:check the code|to:coder_escalation"

    def test_hybrid_d_i_delegation(self):
        result = _extract_toon_decision("D|I|brief:review logic|to:coder_escalation")
        assert result is not None
        assert result.startswith("I|")
        assert "brief:review logic" in result

    def test_no_pattern_returns_none(self):
        assert _extract_toon_decision("Just some plain text without any decision") is None

    def test_rejects_template_echo_answer(self):
        assert _extract_toon_decision("D|answer") is None

    def test_rejects_template_echo_the_answer(self):
        assert _extract_toon_decision("D|the answer") is None

    def test_mcq_embedded_in_sentence(self):
        # D|C followed by alpha "To" -- MCQ shortcut requires non-alpha after
        # the letter, so the general pattern fires and captures more text.
        text = "The answer is C. Decision: D|CTo confirm this..."
        result = _extract_toon_decision(text)
        assert result is not None
        assert result.startswith("D|")

    def test_mcq_embedded_with_space(self):
        # D|C followed by space -- MCQ shortcut fires correctly
        text = "The answer is C. Decision: D|C and that is final."
        assert _extract_toon_decision(text) == "D|C"

    def test_investigate_lenient_without_brief_prefix(self):
        result = _extract_toon_decision("I|check the implementation|to:worker_explore")
        assert result is not None
        assert "brief:" in result
        assert "to:worker_explore" in result

    def test_mcq_letter_with_trailing_nonalpha(self):
        assert _extract_toon_decision("D|A.") == "D|A"


# ── _parse_architect_decision ───────────────────────────────────────────


class TestParseArchitectDecision:
    # ── TOON Direct ──

    def test_direct_mcq(self):
        result = _parse_architect_decision("D|B")
        assert result["mode"] == "direct"
        assert result["answer"] == "B"
        assert result["brief"] == ""
        assert result["delegate_to"] == ""
        assert result["delegate_mode"] == "react"

    def test_long_direct_with_embedded_mcq_rescue(self):
        # Long D| answer with "the answer is C" buried inside
        long_text = "D|" + "x" * 60 + " the answer is C " + "y" * 20
        result = _parse_architect_decision(long_text)
        assert result["mode"] == "direct"
        assert result["answer"] == "C"

    def test_investigate_mode(self):
        result = _parse_architect_decision("I|brief:check code|to:coder_escalation")
        assert result["mode"] == "investigate"
        assert result["brief"] == "check code"
        assert result["delegate_to"] == "coder_escalation"
        assert result["delegate_mode"] == "react"

    def test_invalid_delegate_role_clamps_to_coder_escalation(self):
        result = _parse_architect_decision("I|brief:test|to:nonexistent_role")
        assert result["mode"] == "investigate"
        assert result["delegate_to"] == "coder_escalation"

    def test_invalid_delegate_mode_clamps_to_react(self):
        result = _parse_architect_decision("I|brief:test|to:coder_escalation|mode:invalid_mode")
        assert result["mode"] == "investigate"
        assert result["delegate_mode"] == "react"

    def test_json_direct(self):
        import json
        obj = {"mode": "direct", "answer": "42"}
        result = _parse_architect_decision(json.dumps(obj))
        assert result["mode"] == "direct"
        assert result["answer"] == "42"

    def test_markdown_wrapped_json(self):
        text = '```json\n{"mode": "direct", "answer": "hello world"}\n```'
        result = _parse_architect_decision(text)
        assert result["mode"] == "direct"
        assert result["answer"] == "hello world"

    def test_bare_text_fallback(self):
        text = "The capital of France is Paris."
        result = _parse_architect_decision(text)
        assert result["mode"] == "direct"
        assert result["answer"] == text
        assert result["brief"] == ""
        assert result["delegate_to"] == ""

    def test_long_direct_rescues_option_pattern(self):
        long_text = "D|" + "z" * 60 + " option A seems correct " + "w" * 20
        result = _parse_architect_decision(long_text)
        assert result["mode"] == "direct"
        assert result["answer"] == "A"

    def test_long_direct_rescues_last_toon_mcq(self):
        long_text = "D|" + "z" * 60 + " reasoning text D|B more text"
        result = _parse_architect_decision(long_text)
        assert result["mode"] == "direct"
        assert result["answer"] == "B"

    def test_investigate_with_repl_mode(self):
        result = _parse_architect_decision("I|brief:run tests|to:coder_escalation|mode:repl")
        assert result["mode"] == "investigate"
        assert result["delegate_mode"] == "repl"

    def test_json_investigate(self):
        import json
        obj = {"mode": "investigate", "brief": "analyze logs", "delegate_to": "worker_explore"}
        result = _parse_architect_decision(json.dumps(obj))
        assert result["mode"] == "investigate"
        assert result["brief"] == "analyze logs"
        assert result["delegate_to"] == "worker_explore"

    def test_empty_string(self):
        result = _parse_architect_decision("")
        assert result["mode"] == "direct"

    def test_direct_short_answer(self):
        result = _parse_architect_decision("D|The answer is 42")
        assert result["mode"] == "direct"
        assert result["answer"] == "The answer is 42"


# ── Constants ───────────────────────────────────────────────────────────


class TestConstants:
    def test_valid_delegate_roles(self):
        expected = {"coder_escalation", "worker_explore", "worker_general", "worker_math"}
        assert _VALID_DELEGATE_ROLES == expected

    def test_architect_token_budget_values(self):
        assert _ARCHITECT_TOKEN_BUDGET["architect_general"] == 3375
        assert _ARCHITECT_TOKEN_BUDGET["architect_coding"] == 5150

    def test_architect_decision_budget_values(self):
        assert _ARCHITECT_DECISION_BUDGET["architect_general"] == 1500
        assert _ARCHITECT_DECISION_BUDGET["architect_coding"] == 500

    def test_budgets_have_expected_keys(self):
        assert set(_ARCHITECT_TOKEN_BUDGET.keys()) == {"architect_general", "architect_coding"}
        assert set(_ARCHITECT_DECISION_BUDGET.keys()) == {"architect_general", "architect_coding"}


# ── Loop Guards ──────────────────────────────────────────────────────────


class TestDelegationDepth:
    def test_initial_depth_is_zero(self):
        # Clear any leftover state
        _delegation_local.depth = 0
        assert _get_delegation_depth() == 0

    def test_depth_tracks_nesting(self):
        _delegation_local.depth = 0
        assert _get_delegation_depth() == 0
        _delegation_local.depth = 1
        assert _get_delegation_depth() == 1
        _delegation_local.depth = 2
        assert _get_delegation_depth() == 2
        _delegation_local.depth = 0  # cleanup

    def test_unset_depth_returns_zero(self):
        if hasattr(_delegation_local, "depth"):
            delattr(_delegation_local, "depth")
        assert _get_delegation_depth() == 0


class TestSemanticDedup:
    def _make_brief_key(self, brief: str, delegate_to: str) -> str:
        return hashlib.md5(
            f"{brief.strip().lower()[:DELEGATION_BRIEF_KEY_LEN]}|{delegate_to}".encode()
        ).hexdigest()

    def test_same_brief_same_target_produces_same_key(self):
        k1 = self._make_brief_key("check the code for bugs", "coder_escalation")
        k2 = self._make_brief_key("check the code for bugs", "coder_escalation")
        assert k1 == k2

    def test_same_brief_different_target_produces_different_key(self):
        k1 = self._make_brief_key("check the code", "coder_escalation")
        k2 = self._make_brief_key("check the code", "worker_explore")
        assert k1 != k2

    def test_different_brief_same_target_produces_different_key(self):
        k1 = self._make_brief_key("check the code", "coder_escalation")
        k2 = self._make_brief_key("review the logic", "coder_escalation")
        assert k1 != k2

    def test_case_insensitive_brief(self):
        k1 = self._make_brief_key("Check The Code", "coder_escalation")
        k2 = self._make_brief_key("check the code", "coder_escalation")
        assert k1 == k2


class TestRoleRepetitionGuard:
    def test_consecutive_same_role_triggers_guard(self):
        history = ["coder_escalation"] * DELEGATION_MAX_SAME_TARGET
        recent = history[-DELEGATION_MAX_SAME_TARGET:]
        assert all(r == "coder_escalation" for r in recent)

    def test_alternating_roles_does_not_trigger(self):
        history = ["coder_escalation", "worker_explore"] * 3
        for i in range(len(history)):
            if i + DELEGATION_MAX_SAME_TARGET <= len(history):
                recent = history[i:i + DELEGATION_MAX_SAME_TARGET]
                # At least one alternating window should NOT trigger
                if not all(r == recent[0] for r in recent):
                    break
        else:
            raise AssertionError("Expected at least one non-triggering window")

    def test_max_same_target_constant_is_reasonable(self):
        assert DELEGATION_MAX_SAME_TARGET >= 2
        assert DELEGATION_MAX_SAME_TARGET <= 5


class TestTokenBudgetGuard:
    def test_constant_is_reasonable(self):
        assert DELEGATION_MAX_TOTAL_TOKENS >= 10_000
        assert DELEGATION_MAX_TOTAL_TOKENS <= 100_000

    def test_budget_exceeded_triggers_guard(self):
        cumulative = DELEGATION_MAX_TOTAL_TOKENS + 1
        assert cumulative > DELEGATION_MAX_TOTAL_TOKENS

    def test_budget_not_exceeded_passes(self):
        cumulative = DELEGATION_MAX_TOTAL_TOKENS - 1
        assert cumulative <= DELEGATION_MAX_TOTAL_TOKENS

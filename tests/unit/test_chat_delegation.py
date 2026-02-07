"""Tests for src/api/routes/chat_delegation.py.

Covers: _parse_architect_decision TOON/JSON/bare text parsing.
"""



from src.api.routes.chat_delegation import (
    _parse_architect_decision,
    _VALID_DELEGATE_ROLES,
)


# ── _parse_architect_decision ────────────────────────────────────────────


class TestParseArchitectDecision:
    """Test TOON, JSON, and bare text parsing of architect responses."""

    # ── TOON Direct ──

    def test_toon_direct_answer(self):
        result = _parse_architect_decision("D|The answer is 42")
        assert result["mode"] == "direct"
        assert result["answer"] == "The answer is 42"

    def test_toon_direct_empty(self):
        result = _parse_architect_decision("D|")
        assert result["mode"] == "direct"
        assert result["answer"] == ""

    # ── TOON Investigate ──

    def test_toon_investigate_basic(self):
        result = _parse_architect_decision("I|brief:Check the logs|to:worker_explore")
        assert result["mode"] == "investigate"
        assert result["brief"] == "Check the logs"
        assert result["delegate_to"] == "worker_explore"
        assert result["delegate_mode"] == "react"  # default

    def test_toon_investigate_with_mode(self):
        result = _parse_architect_decision("I|brief:Draft code|to:coder_escalation|mode:repl")
        assert result["mode"] == "investigate"
        assert result["delegate_to"] == "coder_escalation"
        assert result["delegate_mode"] == "repl"

    def test_toon_investigate_invalid_role_clamps(self):
        result = _parse_architect_decision("I|brief:Check|to:nonexistent_role")
        assert result["delegate_to"] == "coder_primary"  # clamped

    def test_toon_investigate_invalid_mode_clamps(self):
        result = _parse_architect_decision("I|brief:Check|to:worker_explore|mode:invalid")
        assert result["delegate_mode"] == "react"  # clamped

    # ── JSON ──

    def test_json_direct_mode(self):
        result = _parse_architect_decision('{"mode": "direct", "answer": "42"}')
        assert result["mode"] == "direct"
        assert result["answer"] == "42"

    def test_json_investigate_mode(self):
        result = _parse_architect_decision(
            '{"mode": "investigate", "brief": "Check it", "to": "worker_explore"}'
        )
        assert result["mode"] == "investigate"
        assert result["brief"] == "Check it"
        assert result["delegate_to"] == "worker_explore"

    def test_json_investigate_invalid_role_clamps(self):
        result = _parse_architect_decision(
            '{"mode": "investigate", "brief": "X", "to": "bad_role"}'
        )
        assert result["delegate_to"] == "coder_primary"

    # ── Markdown-wrapped JSON ──

    def test_markdown_json(self):
        text = '```json\n{"mode": "direct", "answer": "hello"}\n```'
        result = _parse_architect_decision(text)
        assert result["mode"] == "direct"
        assert result["answer"] == "hello"

    # ── Bare text fallback ──

    def test_bare_text_becomes_direct(self):
        result = _parse_architect_decision("Just a plain text answer")
        assert result["mode"] == "direct"
        assert result["answer"] == "Just a plain text answer"

    def test_empty_string(self):
        result = _parse_architect_decision("")
        assert result["mode"] == "direct"

    # ── Valid delegate roles ──

    def test_valid_roles_frozen(self):
        assert "coder_primary" in _VALID_DELEGATE_ROLES
        assert "coder_escalation" in _VALID_DELEGATE_ROLES
        assert "worker_explore" in _VALID_DELEGATE_ROLES

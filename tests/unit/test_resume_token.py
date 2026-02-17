#!/usr/bin/env python3
"""Tests for the resume token module."""

import pytest

from src.graph.resume_token import ResumeToken
from src.graph.state import TaskState
from src.roles import Role


class TestResumeTokenEncodeDecode:
    """Tests for encode/decode round-trip."""

    def test_round_trip(self):
        token = ResumeToken(
            task_id="task-123",
            node_class="CoderNode",
            current_role="coder_escalation",
            turns=3,
            escalation_count=1,
            consecutive_failures=0,
            role_history=["frontdoor", "coder_escalation"],
            last_error="",
        )
        # Compute checksum
        import hashlib
        import json
        from dataclasses import asdict

        content = json.dumps(
            {k: v for k, v in asdict(token).items() if k != "checksum"},
            sort_keys=True,
        )
        token.checksum = hashlib.sha256(content.encode()).hexdigest()[:8]

        encoded = token.encode()
        decoded = ResumeToken.decode(encoded)

        assert decoded.task_id == "task-123"
        assert decoded.node_class == "CoderNode"
        assert decoded.current_role == "coder_escalation"
        assert decoded.turns == 3
        assert decoded.escalation_count == 1
        assert decoded.consecutive_failures == 0
        assert decoded.role_history == ["frontdoor", "coder_escalation"]

    def test_compact_encoding(self):
        """Encoded token should be reasonably compact (<500 bytes)."""
        token = ResumeToken(
            task_id="a" * 36,  # UUID length
            node_class="ArchitectCodingNode",
            current_role="architect_coding",
            turns=10,
            escalation_count=2,
            consecutive_failures=1,
            role_history=["frontdoor", "coder_escalation", "architect_general", "architect_coding"],
            last_error="Some error message that is truncated",
        )
        import hashlib
        import json
        from dataclasses import asdict

        content = json.dumps(
            {k: v for k, v in asdict(token).items() if k != "checksum"},
            sort_keys=True,
        )
        token.checksum = hashlib.sha256(content.encode()).hexdigest()[:8]

        encoded = token.encode()
        assert len(encoded) < 500

    def test_invalid_token_raises(self):
        with pytest.raises(ValueError, match="Invalid resume token"):
            ResumeToken.decode("not-valid-base64!!!")

    def test_checksum_mismatch_raises(self):
        token = ResumeToken(
            task_id="task-123",
            node_class="CoderNode",
            current_role="coder_escalation",
            checksum="badcheck",
        )
        encoded = token.encode()
        with pytest.raises(ValueError, match="checksum mismatch"):
            ResumeToken.decode(encoded)


class TestFromState:
    """Tests for ResumeToken.from_state()."""

    def test_from_state(self):
        state = TaskState(
            task_id="task-456",
            current_role=Role.CODER_ESCALATION,
            turns=5,
            escalation_count=1,
            consecutive_failures=2,
            role_history=["frontdoor", "coder_escalation"],
            last_error="LLM call timed out after 300s",
        )

        token = ResumeToken.from_state(state, "CoderNode")

        assert token.task_id == "task-456"
        assert token.node_class == "CoderNode"
        assert token.current_role == "coder_escalation"
        assert token.turns == 5
        assert token.escalation_count == 1
        assert token.consecutive_failures == 2
        assert token.role_history == ["frontdoor", "coder_escalation"]
        assert token.last_error == "LLM call timed out after 300s"
        assert len(token.checksum) == 8

    def test_from_state_truncates_error(self):
        state = TaskState(
            last_error="x" * 500,
        )
        token = ResumeToken.from_state(state, "FrontdoorNode")
        assert len(token.last_error) == 200

    def test_from_state_round_trip(self):
        state = TaskState(
            task_id="round-trip-test",
            current_role=Role.ARCHITECT_GENERAL,
            turns=8,
            escalation_count=2,
            consecutive_failures=0,
            role_history=["frontdoor", "coder_escalation", "architect_general"],
            last_error="",
        )

        token = ResumeToken.from_state(state, "ArchitectNode")
        encoded = token.encode()
        decoded = ResumeToken.decode(encoded)

        assert decoded.task_id == "round-trip-test"
        assert decoded.node_class == "ArchitectNode"
        assert decoded.turns == 8

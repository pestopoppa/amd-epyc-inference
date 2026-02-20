"""Unit tests for graph helper utilities."""

from src.escalation import ErrorCategory
from src.graph.helpers import (
    _classify_error,
    _detect_role_cycle,
    _select_and_broadcast_workspace_delta,
    _update_workspace_from_turn,
)
from src.graph.state import TaskState


class TestClassifyError:
    def test_classifies_timeout(self):
        assert _classify_error("Request timed out after 30s") == ErrorCategory.TIMEOUT

    def test_classifies_schema(self):
        assert _classify_error("JSON schema validation failed") == ErrorCategory.SCHEMA

    def test_classifies_code(self):
        assert _classify_error("TypeError: unsupported operand type") == ErrorCategory.CODE

    def test_classifies_unknown(self):
        assert _classify_error("completely novel failure") == ErrorCategory.UNKNOWN


class TestRoleCycleDetection:
    def test_detects_period_2_cycle(self):
        assert _detect_role_cycle(["frontdoor", "coder", "frontdoor", "coder"]) is True

    def test_detects_period_3_cycle(self):
        assert _detect_role_cycle(
            ["frontdoor", "coder", "architect", "frontdoor", "coder", "architect"]
        ) is True

    def test_no_cycle_for_short_history(self):
        assert _detect_role_cycle(["frontdoor", "coder", "architect"]) is False


class TestWorkspaceSelection:
    def test_select_and_broadcast_prioritizes_open_questions(self):
        ws = {
            "broadcast_version": 0,
            "proposals": [
                {"kind": "commitment", "owner": "coder", "text": "do X", "priority": "normal"},
                {"kind": "open_question", "owner": "coder", "text": "need API key", "priority": "high"},
                {"kind": "commitment", "owner": "architect", "text": "do Y", "priority": "high"},
            ],
            "open_questions": [],
            "commitments": [],
            "resolved_questions": [],
            "broadcast_log": [],
        }

        _select_and_broadcast_workspace_delta(ws)

        assert ws["broadcast_version"] == 1
        assert len(ws["broadcast_log"]) == 1
        assert any(item["text"] == "need API key" for item in ws["open_questions"])
        assert any(item["text"] == "do Y" for item in ws["commitments"])

    def test_update_workspace_from_turn_caps_proposals(self):
        state = TaskState(
            task_id="t1",
            prompt="Investigate issue",
            turns=1,
        )
        ws = state.workspace_state
        ws["proposals"] = [
            {"kind": "commitment", "owner": "worker", "text": f"old-{i}", "priority": "normal"}
            for i in range(12)
        ]

        _update_workspace_from_turn(
            state=state,
            role="coder",
            output="new proposal",
            error=None,
        )

        assert len(ws["proposals"]) == 12
        assert ws["proposals"][-1]["text"] == "new proposal"
        assert ws["updated_at"] != ""

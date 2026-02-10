"""Tests for extracted seeding modules.

Verifies the new module boundaries, dedup helpers, and re-export compatibility.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "benchmark"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── seeding_checkpoint ───────────────────────────────────────────────


class TestAtomicAppend:
    """Tests for _atomic_append (fcntl-locked write)."""

    def test_round_trip(self, tmp_path):
        from seeding_checkpoint import _atomic_append

        path = tmp_path / "test.jsonl"
        _atomic_append(path, '{"a": 1}')
        _atomic_append(path, '{"b": 2}')

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}

    def test_newline_terminated(self, tmp_path):
        from seeding_checkpoint import _atomic_append

        path = tmp_path / "test.jsonl"
        _atomic_append(path, "hello")
        content = path.read_text()
        assert content.endswith("\n")


class TestCheckpointResult:
    """Tests for unified checkpoint_result (dedup of old append_checkpoint + _checkpoint_3way)."""

    def test_checkpoint_writes_jsonl(self, tmp_path):
        from seeding_checkpoint import checkpoint_result
        from seeding_types import ComparativeResult

        with patch("seeding_checkpoint.EVAL_DIR", tmp_path):
            result = ComparativeResult(
                suite="thinking",
                question_id="t1_q1",
                prompt="test",
                expected="42",
            )
            checkpoint_result("test_session", result)

        path = tmp_path / "test_session.jsonl"
        assert path.exists()
        data = json.loads(path.read_text().strip())
        assert data["suite"] == "thinking"
        assert data["question_id"] == "t1_q1"

    def test_legacy_aliases_exist(self):
        from seeding_checkpoint import append_checkpoint, _checkpoint_3way, checkpoint_result

        assert append_checkpoint is checkpoint_result
        assert _checkpoint_3way is checkpoint_result


class TestPromptHash:
    def test_deterministic(self):
        from seeding_checkpoint import _prompt_hash

        h1 = _prompt_hash("test prompt")
        h2 = _prompt_hash("test prompt")
        assert h1 == h2
        assert len(h1) == 12

    def test_different_inputs(self):
        from seeding_checkpoint import _prompt_hash

        h1 = _prompt_hash("prompt A")
        h2 = _prompt_hash("prompt B")
        assert h1 != h2


# ── seeding_scoring ──────────────────────────────────────────────────


class TestClassifyError:
    def test_infra_patterns(self):
        from seeding_scoring import _classify_error

        assert _classify_error("ReadTimeout: timed out") == "infrastructure"
        assert _classify_error("HTTP 503 backend down") == "infrastructure"
        assert _classify_error("ConnectError refused") == "infrastructure"

    def test_task_failure(self):
        from seeding_scoring import _classify_error

        assert _classify_error("model produced wrong answer") == "task_failure"

    def test_none(self):
        from seeding_scoring import _classify_error

        assert _classify_error(None) == "none"


class TestIsCodingTask:
    def test_coding_indicators(self):
        from seeding_scoring import _is_coding_task

        assert _is_coding_task("Write a Python function to sort a list") is True
        assert _is_coding_task("Implement a binary search algorithm") is True

    def test_non_coding(self):
        from seeding_scoring import _is_coding_task

        assert _is_coding_task("What is the capital of France?") is False
        assert _is_coding_task("Summarize this document") is False

    def test_edge_cases(self):
        from seeding_scoring import _is_coding_task

        assert _is_coding_task("def hello():") is True
        assert _is_coding_task("import numpy") is True
        assert _is_coding_task("return value") is True


class TestTimeoutLogic:
    def test_adaptive_timeout_minimum(self):
        from seeding_scoring import _adaptive_timeout_s

        result = _adaptive_timeout_s(
            role="frontdoor", mode="direct", prompt="test",
            is_vl=False, hard_timeout_s=30,
        )
        assert result == 60  # min 60

    def test_adaptive_timeout_uses_hard(self):
        from seeding_scoring import _adaptive_timeout_s

        result = _adaptive_timeout_s(
            role="frontdoor", mode="direct", prompt="test",
            is_vl=False, hard_timeout_s=600,
        )
        assert result == 600

    def test_bump_no_observed(self):
        from seeding_scoring import _bump_timeout_from_observed

        result = _bump_timeout_from_observed(
            current_s=120, observed_s=0, factor=2.0,
            slack_s=30, hard_timeout_s=600, role_cap_s=300,
        )
        assert result == 120  # unchanged

    def test_bump_increases(self):
        from seeding_scoring import _bump_timeout_from_observed

        result = _bump_timeout_from_observed(
            current_s=60, observed_s=100, factor=2.0,
            slack_s=30, hard_timeout_s=600, role_cap_s=300,
        )
        assert result == 230  # 100*2 + 30 = 230


# ── seeding_orchestrator ─────────────────────────────────────────────


class TestNormalizeToolTelemetry:
    def test_basic_normalization(self):
        from seeding_orchestrator import _normalize_tool_telemetry

        data = {
            "tools_used": 2,
            "tools_called": ["peek", "grep"],
            "tool_timings": [],
        }
        _normalize_tool_telemetry(data)
        assert data["tools_used"] == 2
        assert len(data["tool_timings"]) == 2
        assert data["tool_timings"][0]["tool_name"] == "peek"

    def test_infers_from_timings(self):
        from seeding_orchestrator import _normalize_tool_telemetry

        data = {
            "tools_used": 0,
            "tools_called": [],
            "tool_timings": [
                {"tool_name": "fetch", "elapsed_ms": 100, "success": True},
            ],
        }
        _normalize_tool_telemetry(data)
        assert data["tools_used"] == 1
        assert data["tools_called"] == ["fetch"]

    def test_handles_non_dict(self):
        from seeding_orchestrator import _normalize_tool_telemetry

        _normalize_tool_telemetry("not a dict")  # Should not raise

    def test_empty_data(self):
        from seeding_orchestrator import _normalize_tool_telemetry

        data = {}
        _normalize_tool_telemetry(data)
        assert data["tools_used"] == 0
        assert data["tools_called"] == []
        assert data["tool_timings"] == []

    def test_slot_erase_capability_dict(self):
        from seeding_orchestrator import _SLOT_ERASE_CAPABILITY

        assert isinstance(_SLOT_ERASE_CAPABILITY, dict)


# ── seeding_eval ─────────────────────────────────────────────────────


class TestBuildRoleResult:
    def test_pass_case(self):
        from seeding_eval import _build_role_result

        with patch("seeding_eval.score_answer_deterministic", return_value=True):
            rr, et = _build_role_result(
                role="frontdoor", mode="direct",
                resp={"answer": "42", "tokens_generated": 50, "predicted_tps": 10.0},
                elapsed=2.5, expected="42",
                scoring_method="exact_match", scoring_config={},
            )
        assert rr.passed is True
        assert rr.role == "frontdoor"
        assert rr.mode == "direct"
        assert rr.elapsed_seconds == 2.5
        assert et == "none"

    def test_fail_case(self):
        from seeding_eval import _build_role_result

        with patch("seeding_eval.score_answer_deterministic", return_value=False):
            rr, et = _build_role_result(
                role="frontdoor", mode="repl",
                resp={"answer": "wrong", "tokens_generated": 30},
                elapsed=1.0, expected="42",
                scoring_method="exact_match", scoring_config={},
            )
        assert rr.passed is False
        assert et == "none"

    def test_infra_error(self):
        from seeding_eval import _build_role_result

        rr, et = _build_role_result(
            role="architect_general", mode="delegated",
            resp={"answer": "", "error": "ReadTimeout: timed out", "tokens_generated": 0},
            elapsed=600.0, expected="42",
            scoring_method="exact_match", scoring_config={},
        )
        assert rr.passed is False  # infra → passed=False (not None)
        assert et == "infrastructure"
        assert rr.error_type == "infrastructure"

    def test_task_error(self):
        from seeding_eval import _build_role_result

        rr, et = _build_role_result(
            role="frontdoor", mode="direct",
            resp={"answer": "", "error": "model crashed", "tokens_generated": 0},
            elapsed=5.0, expected="42",
            scoring_method="exact_match", scoring_config={},
        )
        assert rr.passed is False
        assert et == "task_failure"


class TestComputeMetadata:
    def test_basic_metadata(self):
        from seeding_eval import _compute_3way_metadata
        from seeding_types import RoleResult

        role_results = {
            "frontdoor:direct": RoleResult(
                role="frontdoor", mode="direct", answer="42",
                passed=True, elapsed_seconds=2.0, tokens_generated=50,
            ),
            "frontdoor:repl": RoleResult(
                role="frontdoor", mode="repl", answer="42",
                passed=True, elapsed_seconds=3.0, tokens_generated=80,
            ),
            "architect_general:delegated": RoleResult(
                role="architect_general", mode="delegated", answer="42",
                passed=True, elapsed_seconds=10.0, tokens_generated=200,
            ),
        }
        arch_results = {
            "architect_general": {
                "passed": True,
                "elapsed_seconds": 10.0,
                "generation_ms": 8000.0,
                "tokens_generated": 200,
                "predicted_tps": 5.0,
                "tools_used": 0,
                "tools_called": [],
                "role_history": [],
                "error": None,
                "error_type": "none",
            },
        }
        meta = _compute_3way_metadata(
            role_results, arch_results, "Write code for sorting",
            "coder", True, True,
            "frontdoor", "direct", "repl", "delegated",
        )
        assert meta["suite"] == "coder"
        assert meta["architect_role"] == "architect_general"
        assert "cost_metrics" in meta
        assert "SELF:direct" in meta["cost_metrics"]
        assert "SELF:repl" in meta["cost_metrics"]
        assert meta["all_infra"] is False


class TestThreeWayResult:
    def test_dataclass_creation(self):
        from seeding_eval import ThreeWayResult

        r = ThreeWayResult(
            suite="thinking",
            question_id="t1_q1",
            prompt="test",
            expected="42",
        )
        assert r.suite == "thinking"
        assert r.rewards_injected == 0
        assert r.metadata == {}


# ── Re-export compatibility ──────────────────────────────────────────


class TestReExportCompatibility:
    """Verify that symbols imported from the hub match what tests expect."""

    def test_classify_error_via_hub(self):
        import seed_specialist_routing_v2 as v2

        assert v2._classify_error("ReadTimeout: timed out") == "infrastructure"
        assert v2._classify_error(None) == "none"

    def test_build_role_mode_combos_via_hub(self):
        import seed_specialist_routing_v2 as v2

        combos = v2._build_role_mode_combos(
            roles=["frontdoor", "architect_general"],
            modes=["direct", "repl"],
        )
        frontdoor_modes = {m for r, m in combos if r == "frontdoor"}
        assert "direct" in frontdoor_modes
        assert "repl" in frontdoor_modes
        arch_modes = {m for r, m in combos if r == "architect_general"}
        assert arch_modes == {"direct", "delegated"}

    def test_call_orchestrator_forced_via_hub(self):
        import seed_specialist_routing_v2 as v2

        assert callable(v2.call_orchestrator_forced)

    def test_evaluate_question_3way_via_hub(self):
        import seed_specialist_routing_v2 as v2

        assert callable(v2.evaluate_question_3way)

    def test_checkpoint_functions_via_hub(self):
        import seed_specialist_routing_v2 as v2

        assert callable(v2.checkpoint_result)
        assert callable(v2.append_checkpoint)
        assert callable(v2._checkpoint_3way)
        assert v2.append_checkpoint is v2.checkpoint_result

    def test_three_way_result_via_hub(self):
        import seed_specialist_routing_v2 as v2

        r = v2.ThreeWayResult(suite="t", question_id="q", prompt="p", expected="e")
        assert r.suite == "t"

    def test_scoring_functions_via_hub(self):
        import seed_specialist_routing_v2 as v2

        assert callable(v2.score_answer_deterministic)
        assert callable(v2._is_coding_task)
        assert isinstance(v2.INFRA_PATTERNS, list)

    def test_orchestrator_functions_via_hub(self):
        import seed_specialist_routing_v2 as v2

        assert callable(v2._normalize_tool_telemetry)
        assert callable(v2._erase_slots)
        assert isinstance(v2._SLOT_ERASE_CAPABILITY, dict)

    def test_injection_functions_via_hub(self):
        import seed_specialist_routing_v2 as v2

        assert callable(v2._inject_3way_rewards_http)
        assert callable(v2._precompute_embedding)
        assert isinstance(v2.EMBEDDER_PORTS, list)

    def test_eval_functions_via_hub(self):
        import seed_specialist_routing_v2 as v2

        assert callable(v2._build_role_result)
        assert callable(v2._compute_3way_metadata)
        assert callable(v2._eval_single_config)

    def test_legacy_functions_via_hub(self):
        import seed_specialist_routing_v2 as v2

        assert callable(v2._deduplicate_roles)
        assert callable(v2._modes_for_role)
        assert callable(v2.evaluate_question)
        assert callable(v2.run_batch)
        assert callable(v2.print_batch_summary)
        assert callable(v2.print_stats)

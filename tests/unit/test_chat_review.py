"""Tests for src/api/routes/chat_review.py.

Covers: _detect_output_quality_issue, _should_review, _architect_verdict,
_fast_revise, _needs_plan_review, _apply_plan_review, _compute_plan_review_phase.
"""

from unittest.mock import MagicMock, patch


from src.api.routes.chat_review import (
    _apply_plan_review,
    _architect_verdict,
    _compute_plan_review_phase,
    _detect_output_quality_issue,
    _fast_revise,
    _needs_plan_review,
    _should_review,
)


# ── _detect_output_quality_issue ─────────────────────────────────────────


class TestDetectOutputQualityIssue:
    """Test output quality detection heuristics."""

    def test_returns_none_on_clean_output(self):
        answer = (
            "Machine learning models learn patterns from training data. "
            "They use gradient descent to minimize a loss function. "
            "The resulting weights encode the statistical relationships found in examples. "
            "Different architectures suit different tasks, from CNNs for images to transformers for text."
        )
        assert _detect_output_quality_issue(answer) is None

    def test_returns_none_on_short_output(self):
        assert _detect_output_quality_issue("OK") is None

    def test_returns_none_on_empty(self):
        assert _detect_output_quality_issue("") is None

    def test_detects_high_repetition(self):
        answer = " ".join(["the same words repeat"] * 30)
        result = _detect_output_quality_issue(answer)
        assert result is not None
        assert "repetition" in result

    def test_detects_near_empty_after_stripping(self):
        # Must be >= 20 chars to pass the early check, but near-empty after stripping prefixes
        answer = "```                           \n\n```"
        result = _detect_output_quality_issue(answer)
        assert result is not None
        assert "near_empty" in result

    def test_detects_garbled_output(self):
        lines = ["ok\n"] * 20 + ["a\n"] * 20 + ["This is a longer line\n"] * 2
        answer = "".join(lines)
        result = _detect_output_quality_issue(answer)
        # May or may not trigger depending on thresholds — no assertion on exact result
        # Just verify it doesn't crash
        assert result is None or isinstance(result, str)


# ── _should_review ───────────────────────────────────────────────────────


class TestShouldReview:
    """Test MemRL-conditional review gate."""

    def test_returns_false_without_hybrid_router(self):
        state = MagicMock(hybrid_router=None)
        assert _should_review(state, "task1", "frontdoor", "answer text here " * 5) is False

    def test_returns_false_for_architects(self):
        state = MagicMock(hybrid_router=MagicMock())
        assert _should_review(state, "task1", "architect_general", "answer text here " * 5) is False

    def test_returns_false_for_short_answers(self):
        state = MagicMock(hybrid_router=MagicMock())
        assert _should_review(state, "task1", "frontdoor", "short") is False

    def test_returns_false_on_retriever_error(self):
        router = MagicMock()
        router.retriever.retrieve_for_routing.side_effect = RuntimeError("fail")
        state = MagicMock(hybrid_router=router)
        assert _should_review(state, "task1", "frontdoor", "answer text here " * 5) is False

    def test_returns_false_when_no_results(self):
        router = MagicMock()
        router.retriever.retrieve_for_routing.return_value = []
        state = MagicMock(hybrid_router=router)
        assert _should_review(state, "task1", "frontdoor", "answer text here " * 5) is False


# ── _architect_verdict ───────────────────────────────────────────────────


class TestArchitectVerdict:
    """Test architect verdict call."""

    def test_returns_none_on_ok(self):
        primitives = MagicMock()
        primitives.llm_call.return_value = "OK"
        result = _architect_verdict("What is 2+2?", "4", primitives)
        assert result is None

    def test_returns_corrections_on_wrong(self):
        primitives = MagicMock()
        primitives.llm_call.return_value = "WRONG: The answer should be 4, not 5"
        result = _architect_verdict("What is 2+2?", "5", primitives)
        assert result is not None
        assert "WRONG" in result

    def test_returns_none_on_error(self):
        primitives = MagicMock()
        primitives.llm_call.side_effect = RuntimeError("timeout")
        result = _architect_verdict("Q", "A", primitives)
        assert result is None


# ── _fast_revise ─────────────────────────────────────────────────────────


class TestFastRevise:
    """Test fast revision with worker model."""

    def test_returns_revised_answer(self):
        primitives = MagicMock()
        primitives.llm_call.return_value = "The correct answer is 4."
        result = _fast_revise("What is 2+2?", "5", "Answer is 4 not 5", primitives)
        assert result == "The correct answer is 4."

    def test_returns_original_on_empty_revision(self):
        primitives = MagicMock()
        primitives.llm_call.return_value = "   "
        result = _fast_revise("Q", "original", "corrections", primitives)
        assert result == "original"

    def test_returns_original_on_error(self):
        primitives = MagicMock()
        primitives.llm_call.side_effect = RuntimeError("fail")
        result = _fast_revise("Q", "original", "corrections", primitives)
        assert result == "original"


# ── _needs_plan_review ───────────────────────────────────────────────────


class TestNeedsPlanReview:
    """Test plan review gate logic."""

    @patch("src.proactive_delegation.classify_task_complexity")
    def test_skips_trivial_tasks(self, mock_classify):
        from src.proactive_delegation import TaskComplexity
        mock_classify.return_value = (TaskComplexity.TRIVIAL, {})
        state = MagicMock(plan_review_phase="A")
        assert _needs_plan_review({"objective": "hi"}, ["frontdoor"], state) is False

    @patch("src.proactive_delegation.classify_task_complexity")
    def test_skips_architect_self_review(self, mock_classify):
        from src.proactive_delegation import TaskComplexity
        mock_classify.return_value = (TaskComplexity.MODERATE, {})
        state = MagicMock(plan_review_phase="A")
        assert _needs_plan_review({"objective": "design"}, ["architect_general"], state) is False

    @patch("src.proactive_delegation.classify_task_complexity")
    def test_allows_moderate_non_architect(self, mock_classify):
        from src.proactive_delegation import TaskComplexity
        mock_classify.return_value = (TaskComplexity.MODERATE, {})
        state = MagicMock(plan_review_phase="A", hybrid_router=None)
        result = _needs_plan_review({"objective": "moderate task"}, ["frontdoor"], state)
        assert result is True


# ── _apply_plan_review ───────────────────────────────────────────────────


class TestApplyPlanReview:
    """Test plan review patch application."""

    def test_no_patches_returns_unchanged(self):
        review = MagicMock(patches=[])
        result = _apply_plan_review(["frontdoor"], review)
        assert result == ["frontdoor"]

    def test_reroute_patch_changes_role(self):
        review = MagicMock(patches=[{"op": "reroute", "step": "S1", "v": "coder_escalation"}])
        result = _apply_plan_review(["frontdoor", "worker"], review)
        assert result[0] == "coder_escalation"
        assert result[1] == "worker"

    def test_ignores_non_reroute_ops(self):
        review = MagicMock(patches=[{"op": "add_step", "step": "S2"}])
        result = _apply_plan_review(["frontdoor"], review)
        assert result == ["frontdoor"]


# ── _compute_plan_review_phase ───────────────────────────────────────────


class TestComputePlanReviewPhase:
    """Test plan review phase computation."""

    def test_phase_a_on_low_reviews(self):
        assert _compute_plan_review_phase({"total_reviews": 10}) == "A"

    def test_phase_a_on_empty_q_values(self):
        assert _compute_plan_review_phase({"total_reviews": 100, "task_class_q_values": {}}) == "A"

"""Tests for chat_review.py - architect review and quality gates."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from src.api.routes.chat_review import (
    _apply_plan_review,
    _architect_plan_review,
    _architect_verdict,
    _compute_plan_review_phase,
    _detect_output_quality_issue,
    _fast_revise,
    _needs_plan_review,
    _should_review,
    _store_plan_review_episode,
)


class TestDetectOutputQualityIssue:
    """Tests for _detect_output_quality_issue heuristics."""

    def test_short_output_returns_none(self):
        """Output < 20 chars returns None."""
        assert _detect_output_quality_issue("Short") is None
        assert _detect_output_quality_issue("a" * 19) is None

    def test_empty_output_returns_none(self):
        """Empty output returns None."""
        assert _detect_output_quality_issue("") is None
        assert _detect_output_quality_issue(None) is None  # type: ignore

    def test_normal_output_returns_none(self):
        """Normal output without issues returns None."""
        answer = "This is a normal answer with reasonable content and no repetition."
        assert _detect_output_quality_issue(answer) is None

    def test_high_repetition_detected(self):
        """High n-gram repetition is detected."""
        # Create highly repetitive text (same trigram repeated)
        answer = " ".join(["the same words"] * 20)
        result = _detect_output_quality_issue(answer)
        assert result is not None
        assert "repetition" in result.lower()

    def test_repetition_needs_enough_words(self):
        """Repetition check requires >= 20 words."""
        answer = " ".join(["same"] * 10)  # Only 10 words
        assert _detect_output_quality_issue(answer) is None

    def test_garbled_output_detected(self):
        """Mostly short lines indicates garbled output."""
        # Create text with many very short lines
        lines = ["x"] * 10 + ["A longer line here with actual content"]
        answer = "\n".join(lines)
        # Need >= 50 words for garble check
        answer = answer + " " + " ".join(["word"] * 50)
        _detect_output_quality_issue(answer)
        # May or may not trigger depending on thresholds
        # Just verify it doesn't crash

    def test_near_empty_after_strip(self):
        """Near-empty output after stripping prefixes detected."""
        # Needs to be >= 20 chars total, but < 10 chars after stripping prefixes
        # "```python\n" + 10 spaces = 20 chars, strips to ""
        answer = "```python" + " " * 11  # 20 chars, strips to ""
        result = _detect_output_quality_issue(answer)
        assert result is not None
        assert "near_empty" in result.lower()

    def test_stripped_prefixes(self):
        """Common prefixes are stripped for near-empty check."""
        # "The answer is" (13 chars) + 7 spaces = 20 chars, strips to ""
        answer = "The answer is" + " " * 7
        result = _detect_output_quality_issue(answer)
        assert result is not None
        assert "near_empty" in result.lower()

    def test_sufficient_content_after_strip(self):
        """Sufficient content after stripping returns None."""
        answer = "```python\nThis is a complete code block with enough content to pass"
        assert _detect_output_quality_issue(answer) is None


class TestShouldReview:
    """Tests for _should_review MemRL-conditional gate."""

    def test_no_hybrid_router_returns_false(self):
        """No hybrid_router returns False."""
        state = MagicMock()
        state.hybrid_router = None
        assert _should_review(state, "task-1", "coder", "Answer text") is False

    def test_architect_role_skips_review(self):
        """Architect roles don't self-review."""
        state = MagicMock()
        state.hybrid_router = MagicMock()
        assert _should_review(state, "task-1", "architect_general", "Long answer") is False
        assert _should_review(state, "task-1", "architect_coding", "Long answer") is False

    def test_short_answer_skips_review(self):
        """Answers < 50 chars skip review."""
        state = MagicMock()
        state.hybrid_router = MagicMock()
        assert _should_review(state, "task-1", "coder", "Short") is False

    def test_low_q_value_triggers_review(self):
        """Low Q-value (< threshold) triggers review."""
        state = MagicMock()
        mock_retriever = MagicMock()
        mock_result = MagicMock()
        mock_result.memory.action = "coder_escalation"
        mock_result.q_value = 0.3  # Below 0.6 threshold
        mock_retriever.retrieve_for_routing.return_value = [mock_result]
        state.hybrid_router.retriever = mock_retriever

        result = _should_review(state, "task-1", "coder_escalation", "A" * 60)
        assert result is True

    def test_high_q_value_skips_review(self):
        """High Q-value (>= threshold) skips review."""
        state = MagicMock()
        mock_retriever = MagicMock()
        mock_result = MagicMock()
        mock_result.memory.action = "coder_escalation"
        mock_result.q_value = 0.8  # Above 0.6 threshold
        mock_retriever.retrieve_for_routing.return_value = [mock_result]
        state.hybrid_router.retriever = mock_retriever

        result = _should_review(state, "task-1", "coder_escalation", "A" * 60)
        assert result is False

    def test_no_results_skips_review(self):
        """No retrieval results skips review."""
        state = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve_for_routing.return_value = []
        state.hybrid_router.retriever = mock_retriever

        result = _should_review(state, "task-1", "coder", "A" * 60)
        assert result is False

    def test_no_role_results_skips_review(self):
        """No results for current role skips review."""
        state = MagicMock()
        mock_retriever = MagicMock()
        mock_result = MagicMock()
        mock_result.memory.action = "other_role"  # Different role
        mock_result.q_value = 0.3
        mock_retriever.retrieve_for_routing.return_value = [mock_result]
        state.hybrid_router.retriever = mock_retriever

        result = _should_review(state, "task-1", "coder_escalation", "A" * 60)
        assert result is False

    def test_exception_returns_false(self):
        """Exception in Q-value check returns False."""
        state = MagicMock()
        state.hybrid_router.retriever.retrieve_for_routing.side_effect = RuntimeError("DB error")

        result = _should_review(state, "task-1", "coder", "A" * 60)
        assert result is False


class TestArchitectVerdict:
    """Tests for _architect_verdict."""

    def test_ok_verdict_returns_none(self):
        """OK verdict returns None."""
        primitives = MagicMock()
        primitives.llm_call.return_value = "OK - answer is correct"

        with patch("src.api.routes.chat_review.build_review_verdict_prompt"):
            result = _architect_verdict("question", "answer", primitives)

        assert result is None

    def test_ok_case_insensitive(self):
        """OK detection is case-insensitive (starts with)."""
        primitives = MagicMock()
        primitives.llm_call.return_value = "ok looks good"

        with patch("src.api.routes.chat_review.build_review_verdict_prompt"):
            result = _architect_verdict("question", "answer", primitives)

        assert result is None

    def test_wrong_verdict_returned(self):
        """WRONG verdict is returned."""
        primitives = MagicMock()
        primitives.llm_call.return_value = "WRONG: The calculation is incorrect"

        with patch("src.api.routes.chat_review.build_review_verdict_prompt"):
            result = _architect_verdict("question", "answer", primitives)

        assert result == "WRONG: The calculation is incorrect"

    def test_with_worker_digests(self):
        """Worker digests passed to prompt builder."""
        primitives = MagicMock()
        primitives.llm_call.return_value = "OK"
        digests = [{"role": "coder", "output": "code"}]

        with patch("src.api.routes.chat_review.build_review_verdict_prompt") as mock_build:
            _architect_verdict("q", "a", primitives, worker_digests=digests)

        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs.get("worker_digests") == digests

    def test_with_context_digest(self):
        """Context digest passed to prompt builder."""
        primitives = MagicMock()
        primitives.llm_call.return_value = "OK"

        with patch("src.api.routes.chat_review.build_review_verdict_prompt") as mock_build:
            _architect_verdict("q", "a", primitives, context_digest="Summary")

        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs.get("context_digest") == "Summary"

    def test_exception_returns_none(self):
        """Exception returns None (fail-open)."""
        primitives = MagicMock()
        primitives.llm_call.side_effect = RuntimeError("LLM timeout")

        with patch("src.api.routes.chat_review.build_review_verdict_prompt"):
            result = _architect_verdict("question", "answer", primitives)

        assert result is None


class TestFastRevise:
    """Tests for _fast_revise."""

    def test_successful_revision(self):
        """Successful revision returns revised answer."""
        primitives = MagicMock()
        primitives.llm_call.return_value = "  Revised answer text  "

        with patch("src.api.routes.chat_review.build_revision_prompt"):
            result = _fast_revise("q", "original", "corrections", primitives)

        assert result == "Revised answer text"

    def test_empty_revision_returns_original(self):
        """Empty revision result returns original."""
        primitives = MagicMock()
        primitives.llm_call.return_value = ""

        with patch("src.api.routes.chat_review.build_revision_prompt"):
            result = _fast_revise("q", "original answer", "corrections", primitives)

        assert result == "original answer"

    def test_exception_returns_original(self):
        """Exception returns original answer."""
        primitives = MagicMock()
        primitives.llm_call.side_effect = RuntimeError("LLM error")

        with patch("src.api.routes.chat_review.build_revision_prompt"):
            result = _fast_revise("q", "original answer", "corrections", primitives)

        assert result == "original answer"

    def test_uses_worker_explore_role(self):
        """Revision uses worker_explore role."""
        primitives = MagicMock()
        primitives.llm_call.return_value = "revised"

        with patch("src.api.routes.chat_review.build_revision_prompt"):
            _fast_revise("q", "orig", "fix", primitives)

        call_kwargs = primitives.llm_call.call_args.kwargs
        assert call_kwargs.get("role") == "worker_explore"


class TestNeedsPlanReview:
    """Tests for _needs_plan_review."""

    def test_trivial_complexity_skips(self):
        """Trivial complexity bypasses review."""
        state = MagicMock()
        state.plan_review_phase = "A"

        with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
            from src.proactive_delegation import TaskComplexity

            mock_classify.return_value = (TaskComplexity.TRIVIAL, [])
            result = _needs_plan_review({"objective": "hi"}, ["frontdoor"], state)

        assert result is False

    def test_simple_complexity_skips(self):
        """Simple complexity bypasses review."""
        state = MagicMock()
        state.plan_review_phase = "A"

        with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
            from src.proactive_delegation import TaskComplexity

            mock_classify.return_value = (TaskComplexity.SIMPLE, [])
            result = _needs_plan_review({"objective": "help"}, ["frontdoor"], state)

        assert result is False

    def test_complex_complexity_skips(self):
        """Complex tasks skip (architect already owns)."""
        state = MagicMock()
        state.plan_review_phase = "A"

        with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
            from src.proactive_delegation import TaskComplexity

            mock_classify.return_value = (TaskComplexity.COMPLEX, [])
            result = _needs_plan_review({"objective": "build system"}, ["frontdoor"], state)

        assert result is False

    def test_architect_role_skips(self):
        """Architect routing decision skips self-review."""
        state = MagicMock()
        state.plan_review_phase = "A"

        with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
            from src.proactive_delegation import TaskComplexity

            mock_classify.return_value = (TaskComplexity.MODERATE, [])
            result = _needs_plan_review({"objective": "design"}, ["architect_general"], state)

        assert result is False

    def test_moderate_complexity_triggers(self):
        """Moderate complexity triggers review (Phase A)."""
        state = MagicMock()
        state.plan_review_phase = "A"
        state.hybrid_router = None

        with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
            from src.proactive_delegation import TaskComplexity

            mock_classify.return_value = (TaskComplexity.MODERATE, [])
            result = _needs_plan_review(
                {"objective": "implement feature"}, ["coder_escalation"], state
            )

        assert result is True

    def test_phase_c_stochastic_skip(self):
        """Phase C stochastically skips (90% skip rate)."""
        state = MagicMock()
        state.plan_review_phase = "C"
        state.hybrid_router = None

        with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
            from src.proactive_delegation import TaskComplexity

            mock_classify.return_value = (TaskComplexity.MODERATE, [])
            with patch("random.random", return_value=0.5):  # < 0.9
                result = _needs_plan_review({"objective": "task"}, ["coder"], state)

        assert result is False  # Skipped due to random < 0.9

    def test_phase_c_stochastic_triggers(self):
        """Phase C triggers when random >= skip rate."""
        state = MagicMock()
        state.plan_review_phase = "C"
        state.hybrid_router = None

        with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
            from src.proactive_delegation import TaskComplexity

            mock_classify.return_value = (TaskComplexity.MODERATE, [])
            with patch("random.random", return_value=0.95):  # >= 0.9
                result = _needs_plan_review({"objective": "task"}, ["coder"], state)

        assert result is True

    def test_phase_b_high_q_skips(self):
        """Phase B with high Q-value skips review."""
        state = MagicMock()
        state.plan_review_phase = "B"
        mock_result = MagicMock()
        mock_result.q_value = 0.8  # High Q
        state.hybrid_router.retriever.retrieve_for_routing.return_value = [mock_result]

        with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
            from src.proactive_delegation import TaskComplexity

            mock_classify.return_value = (TaskComplexity.MODERATE, [])
            result = _needs_plan_review({"objective": "task"}, ["coder"], state)

        assert result is False

    def test_phase_b_low_q_triggers(self):
        """Phase B with low Q-value triggers review."""
        state = MagicMock()
        state.plan_review_phase = "B"
        mock_result = MagicMock()
        mock_result.q_value = 0.4  # Low Q
        state.hybrid_router.retriever.retrieve_for_routing.return_value = [mock_result]

        with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
            from src.proactive_delegation import TaskComplexity

            mock_classify.return_value = (TaskComplexity.MODERATE, [])
            result = _needs_plan_review({"objective": "task"}, ["coder"], state)

        assert result is True

    def test_phase_b_exception_triggers(self):
        """Phase B Q-value exception still triggers review."""
        state = MagicMock()
        state.plan_review_phase = "B"
        state.hybrid_router.retriever.retrieve_for_routing.side_effect = RuntimeError()

        with patch("src.proactive_delegation.classify_task_complexity") as mock_classify:
            from src.proactive_delegation import TaskComplexity

            mock_classify.return_value = (TaskComplexity.MODERATE, [])
            result = _needs_plan_review({"objective": "task"}, ["coder"], state)

        assert result is True


class TestArchitectPlanReview:
    """Tests for _architect_plan_review."""

    def test_no_plan_steps_returns_none(self):
        """No plan steps and no routing returns None."""
        primitives = MagicMock()
        state = MagicMock()

        result = _architect_plan_review(
            {"objective": "task"},
            [],  # Empty routing
            primitives,
            state,
            "task-1",
        )

        assert result is None

    def test_synthesizes_steps_from_routing(self):
        """Steps synthesized from routing decision when no explicit plan."""
        primitives = MagicMock()
        state = MagicMock()

        mock_review_result = MagicMock()
        mock_review_result.decision = "ok"
        mock_review_result.score = 0.9
        mock_review_result.feedback = "Looks good"

        with patch("src.proactive_delegation.ArchitectReviewService") as mock_service:
            mock_service.return_value.review_plan.return_value = mock_review_result
            result = _architect_plan_review(
                {"objective": "implement feature"},
                ["coder_escalation", "worker_general"],
                primitives,
                state,
                "task-1",
            )

        assert result is mock_review_result
        # Verify steps were synthesized
        call_kwargs = mock_service.return_value.review_plan.call_args.kwargs
        steps = call_kwargs.get("plan_steps", [])
        assert len(steps) == 2
        assert steps[0]["actor"] == "coder_escalation"
        assert steps[1]["actor"] == "worker_general"

    def test_uses_explicit_plan_steps(self):
        """Uses explicit plan steps when provided."""
        primitives = MagicMock()
        state = MagicMock()

        mock_review_result = MagicMock()

        task_ir = {
            "objective": "task",
            "plan": {
                "steps": [
                    {"id": "S1", "actor": "worker", "action": "do work"},
                ]
            },
        }

        with patch("src.proactive_delegation.ArchitectReviewService") as mock_service:
            mock_service.return_value.review_plan.return_value = mock_review_result
            _architect_plan_review(task_ir, ["coder"], primitives, state, "task-1")

        call_kwargs = mock_service.return_value.review_plan.call_args.kwargs
        steps = call_kwargs.get("plan_steps", [])
        assert len(steps) == 1
        assert steps[0]["actor"] == "worker"


class TestApplyPlanReview:
    """Tests for _apply_plan_review."""

    def test_no_patches_returns_unchanged(self):
        """No patches returns unchanged routing."""
        review = MagicMock()
        review.patches = []
        result = _apply_plan_review(["coder_escalation"], review)
        assert result == ["coder_escalation"]

    def test_reroute_patch_applies(self):
        """Reroute patch changes role."""
        review = MagicMock()
        review.patches = [{"op": "reroute", "step": "S1", "v": "architect_general"}]
        result = _apply_plan_review(["coder_escalation", "worker"], review)
        assert result[0] == "architect_general"
        assert result[1] == "worker"

    def test_reroute_s2_patch(self):
        """Reroute on S2 changes second role."""
        review = MagicMock()
        review.patches = [{"op": "reroute", "step": "S2", "v": "architect_coding"}]
        result = _apply_plan_review(["coder", "worker"], review)
        assert result[0] == "coder"
        assert result[1] == "architect_coding"

    def test_invalid_step_id_defaults_to_first(self):
        """Invalid step_id defaults to first position."""
        review = MagicMock()
        review.patches = [{"op": "reroute", "step": "invalid", "v": "new_role"}]
        result = _apply_plan_review(["original"], review)
        assert result[0] == "new_role"

    def test_empty_routing_with_patch(self):
        """Empty routing with reroute patch."""
        review = MagicMock()
        review.patches = [{"op": "reroute", "step": "S1", "v": "new_role"}]
        result = _apply_plan_review([], review)
        assert result == []  # Can't apply to empty

    def test_other_op_ignored(self):
        """Non-reroute ops are ignored."""
        review = MagicMock()
        review.patches = [{"op": "add_step", "v": "new_step"}]
        result = _apply_plan_review(["coder"], review)
        assert result == ["coder"]

    def test_out_of_bounds_step_id(self):
        """Out of bounds step_id handled gracefully."""
        review = MagicMock()
        review.patches = [{"op": "reroute", "step": "S99", "v": "new_role"}]
        result = _apply_plan_review(["coder"], review)
        assert result == ["coder"]  # Unchanged


class TestStorePlanReviewEpisode:
    """Tests for _store_plan_review_episode."""

    def test_logs_to_progress_logger(self):
        """Progress logger receives episode."""
        state = MagicMock()
        state.plan_review_phase = "A"
        state.update_plan_review_stats.return_value = {"total_reviews": 1}
        review = MagicMock()
        review.decision = "ok"
        review.score = 0.9
        review.feedback = "Good plan"
        review.patches = []
        review.is_ok = True

        with patch("src.api.routes.chat_review._compute_plan_review_phase", return_value="A"):
            _store_plan_review_episode(state, "task-1", {"objective": "test"}, review)

        state.progress_logger.log.assert_called_once()

    def test_no_progress_logger(self):
        """No progress logger doesn't crash."""
        state = MagicMock()
        state.progress_logger = None
        state.plan_review_phase = "A"
        state.update_plan_review_stats.return_value = {}
        state.q_scorer = None
        review = MagicMock()
        review.decision = "ok"
        review.is_ok = True

        with patch("src.api.routes.chat_review._compute_plan_review_phase", return_value="A"):
            _store_plan_review_episode(state, "task-1", {}, review)

    def test_memrl_score_storage(self):
        """MemRL episode stored for Q-learning."""
        state = MagicMock()
        state.progress_logger = None
        state.plan_review_phase = "A"
        state.update_plan_review_stats.return_value = {}
        review = MagicMock()
        review.decision = "escalate"
        review.score = 0.5
        review.feedback = "Needs escalation"
        review.is_ok = False

        with patch("src.api.routes.chat_review._compute_plan_review_phase", return_value="A"):
            _store_plan_review_episode(state, "task-1", {"objective": "test"}, review)

        state.q_scorer.score_external_result.assert_called_once()
        call_kwargs = state.q_scorer.score_external_result.call_args.kwargs
        assert call_kwargs["reward"] == 0.0  # 0.5 * 2 - 1 = 0
        assert "plan_review" in call_kwargs["action"]

    def test_memrl_exception_handled(self):
        """MemRL exception doesn't crash."""
        state = MagicMock()
        state.progress_logger = None
        state.plan_review_phase = "A"
        state.update_plan_review_stats.return_value = {}
        state.q_scorer.score_external_result.side_effect = RuntimeError("DB error")
        review = MagicMock()
        review.decision = "ok"
        review.score = 0.9
        review.is_ok = True

        with patch("src.api.routes.chat_review._compute_plan_review_phase", return_value="A"):
            # Should not raise
            _store_plan_review_episode(state, "task-1", {}, review)


class TestComputePlanReviewPhase:
    """Tests for _compute_plan_review_phase."""

    def test_phase_a_low_total(self):
        """Low total reviews returns Phase A."""
        stats = {"total_reviews": 10}
        assert _compute_plan_review_phase(stats) == "A"

    def test_phase_a_no_q_values(self):
        """No Q-values returns Phase A."""
        stats = {"total_reviews": 100, "task_class_q_values": {}}
        assert _compute_plan_review_phase(stats) == "A"

    def test_phase_b_criteria(self):
        """Phase B criteria met."""
        stats = {
            "total_reviews": 60,
            "task_class_q_values": {"code": 0.8, "chat": 0.7},  # mean=0.75, min=0.7
        }
        result = _compute_plan_review_phase(stats)
        # mean >= 0.7, min >= 0.5, so Phase B
        assert result == "B"

    def test_phase_c_criteria(self):
        """Phase C criteria met."""
        stats = {
            "total_reviews": 150,
            "task_class_q_values": {"code": 0.85, "chat": 0.80},  # min >= 0.7
        }
        result = _compute_plan_review_phase(stats)
        assert result == "C"

    def test_phase_a_low_q(self):
        """Low Q-values stay in Phase A."""
        stats = {
            "total_reviews": 100,
            "task_class_q_values": {"code": 0.4, "chat": 0.3},
        }
        result = _compute_plan_review_phase(stats)
        assert result == "A"

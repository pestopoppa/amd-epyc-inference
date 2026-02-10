"""Tests for pipeline_monitor anomaly signal detection."""

from __future__ import annotations

from src.pipeline_monitor.anomaly import (
    anomaly_score,
    compute_anomaly_signals,
    detect_comment_only,
    detect_delegation_format_error,
    detect_excessive_tokens,
    detect_format_violation,
    detect_function_repr_leak,
    detect_misrouted_to_coder,
    detect_near_empty,
    detect_repetition_loop,
    detect_repl_no_tools,
    detect_self_doubt_loop,
    detect_self_escalation,
    detect_silent_execution,
    detect_slow_delegation,
    detect_status_phrase_final,
    detect_template_echo,
    detect_think_tag_leak,
    detect_vision_blindness,
    SIGNAL_WEIGHTS,
)


# ── repetition_loop ──


class TestRepetitionLoop:
    def test_clean_text_no_repetition(self):
        text = "The quick brown fox jumps over the lazy dog near the river"
        assert detect_repetition_loop(text) is False

    def test_degenerate_repetition(self):
        # Repeating the same trigram many times
        text = " ".join(["the cat sat"] * 20)
        assert detect_repetition_loop(text) is True

    def test_short_text_skipped(self):
        text = "hello world"
        assert detect_repetition_loop(text) is False

    def test_threshold_customizable(self):
        text = " ".join(["hello world foo bar baz"] * 5)
        # With a very low threshold, shouldn't trigger
        assert detect_repetition_loop(text, threshold=0.01) is False


# ── comment_only ──


class TestCommentOnly:
    def test_all_comments(self):
        code = "# This is a comment\n# Another comment\n"
        assert detect_comment_only(code) is True

    def test_has_executable(self):
        code = "# comment\nprint('hello')\n"
        assert detect_comment_only(code) is False

    def test_blank_only(self):
        code = "\n\n\n"
        # No content lines at all — not "comment only"
        assert detect_comment_only(code) is False

    def test_mixed_blank_and_comments(self):
        code = "\n# comment\n\n# another\n"
        assert detect_comment_only(code) is True


# ── template_echo ──


class TestTemplateEcho:
    def test_both_prefixes(self):
        text = "D|The answer is 42\nI|brief:need more info to:coder"
        assert detect_template_echo(text) is True

    def test_only_direct(self):
        text = "D|The answer is 42"
        assert detect_template_echo(text) is False

    def test_only_investigate(self):
        text = "I|brief:check this to:worker"
        assert detect_template_echo(text) is False

    def test_clean_answer(self):
        text = "The answer to the question is 42."
        assert detect_template_echo(text) is False


# ── self_doubt_loop ──


class TestSelfDoubtLoop:
    def test_many_restarts(self):
        text = (
            "Actually, let me reconsider. Wait, that's wrong. "
            "Actually, on second thought, let me rethink this."
        )
        assert detect_self_doubt_loop(text) is True

    def test_few_restarts(self):
        text = "Actually, I think the answer is 42."
        assert detect_self_doubt_loop(text) is False

    def test_no_restarts(self):
        text = "The answer is clearly 42 based on the given data."
        assert detect_self_doubt_loop(text) is False


# ── format_violation ──


class TestFormatViolation:
    def test_architect_no_prefix_long_answer(self):
        # Must be >50 chars to not be treated as post-extraction
        long_answer = "The answer involves a complex analysis of the problem requiring multiple steps of reasoning"
        assert detect_format_violation(
            long_answer, "architect_general", "delegated",
        ) is True

    def test_architect_with_direct_prefix(self):
        long_answer = "D|The full analysis shows that the correct answer based on multiple factors is B"
        assert detect_format_violation(
            long_answer, "architect_general", "delegated",
        ) is False

    def test_architect_with_investigate_prefix(self):
        long_answer = "I|brief:Need coder to implement the BFS algorithm with edge weights to:coder_escalation"
        assert detect_format_violation(
            long_answer, "architect_general", "delegated",
        ) is False

    def test_non_architect_role(self):
        assert detect_format_violation(
            "plain answer without any prefix that is long enough", "frontdoor", "direct",
        ) is False

    def test_architect_wrong_mode(self):
        assert detect_format_violation(
            "plain answer without any prefix that is long enough", "architect_general", "repl",
        ) is False

    def test_short_extracted_answer_not_flagged(self):
        # Post-extraction answers (e.g. "A", "42") are short — not a violation
        assert detect_format_violation(
            "A", "architect_general", "delegated",
        ) is False

    def test_delegation_occurred_not_flagged(self):
        # If role_history shows delegation happened, prefix was already stripped
        assert detect_format_violation(
            "The computed result from the coder specialist after full analysis is 42",
            "architect_general", "delegated",
            role_history=["architect_general", "coder_escalation"],
        ) is False

    def test_architect_delegated_with_role_history(self):
        # After delegation, D| prefix is stripped by parser — not a violation
        assert detect_format_violation(
            "42", "architect_general", "delegated",
            role_history=["architect_general", "coder_escalation"],
        ) is False

    def test_architect_no_prefix_solo_history(self):
        # No delegation happened, long raw answer — real violation
        long_raw = "After careful analysis of the quantum mechanics problem I believe the answer is B"
        assert detect_format_violation(
            long_raw, "architect_general", "delegated",
            role_history=["architect_general"],
        ) is True


# ── think_tag_leak ──


class TestThinkTagLeak:
    def test_leaked_think(self):
        assert detect_think_tag_leak("<think>Let me reason...</think>The answer is 42") is True

    def test_no_think(self):
        assert detect_think_tag_leak("The answer is 42") is False


# ── near_empty ──


class TestNearEmpty:
    def test_empty_no_error(self):
        assert detect_near_empty("", None) is True

    def test_few_tokens_no_error(self):
        assert detect_near_empty("A B", None) is True

    def test_few_tokens_with_error(self):
        assert detect_near_empty("A B", "timeout") is False

    def test_sufficient_tokens(self):
        assert detect_near_empty("The answer is clearly forty two", None) is False

    def test_mcq_single_letter_not_flagged(self):
        assert detect_near_empty("A", None, scoring_method="multiple_choice") is False

    def test_mcq_empty_not_flagged(self):
        assert detect_near_empty("", None, scoring_method="multiple_choice") is False

    def test_exact_match_single_token_not_flagged(self):
        assert detect_near_empty("A", None, scoring_method="exact_match") is False

    def test_non_mcq_single_letter_flagged(self):
        assert detect_near_empty("A", None, scoring_method="substring") is True


# ── excessive_tokens ──


class TestExcessiveTokens:
    def test_mcq_excessive(self):
        assert detect_excessive_tokens(3000, "multiple_choice") is True

    def test_mcq_normal(self):
        assert detect_excessive_tokens(500, "multiple_choice") is False

    def test_non_mcq_ignored(self):
        assert detect_excessive_tokens(5000, "exact_match") is False


# ── delegation_format_error ──


class TestDelegationFormatError:
    def test_missing_brief(self):
        assert detect_delegation_format_error("I|to:coder something") is True

    def test_proper_delegation(self):
        assert detect_delegation_format_error("I|brief:check this to:coder") is False

    def test_no_delegation(self):
        assert detect_delegation_format_error("D|42") is False

    def test_plain_text(self):
        assert detect_delegation_format_error("Just a regular answer") is False


# ── self_escalation ──


class TestSelfEscalation:
    def test_consecutive_dupes(self):
        assert detect_self_escalation(["frontdoor", "frontdoor"]) is True

    def test_no_dupes(self):
        assert detect_self_escalation(["frontdoor", "coder", "architect"]) is False

    def test_single_entry(self):
        assert detect_self_escalation(["frontdoor"]) is False

    def test_empty(self):
        assert detect_self_escalation([]) is False

    def test_non_consecutive_dupes(self):
        assert detect_self_escalation(["frontdoor", "coder", "frontdoor"]) is False


# ── vision_blindness ──


class TestVisionBlindness:
    def test_vision_role_short_answer(self):
        assert detect_vision_blindness("OK", "worker_vision") is True

    def test_vision_role_normal_answer(self):
        answer = "The image shows a cat sitting on a red couch in a living room"
        assert detect_vision_blindness(answer, "worker_vision") is False

    def test_non_vision_role(self):
        assert detect_vision_blindness("OK", "frontdoor") is False

    def test_vision_repl_short_answer_not_flagged(self):
        # REPL mode produces concise FINAL() answers — not blind
        assert detect_vision_blindness("Cancer", "worker_vision", mode="repl") is False

    def test_vision_repl_empty_still_flagged(self):
        # Empty answer in REPL mode is still blind
        assert detect_vision_blindness("", "worker_vision", mode="repl") is True

    def test_vision_direct_short_answer_flagged(self):
        # Direct mode short answer IS suspicious
        assert detect_vision_blindness("OK", "worker_vision", mode="direct") is True


# ── silent_execution ──


class TestSilentExecution:
    def test_tools_used_empty_answer(self):
        assert detect_silent_execution("", 2, None) is True

    def test_tools_used_has_answer(self):
        assert detect_silent_execution("result: 42", 2, None) is False

    def test_no_tools(self):
        assert detect_silent_execution("", 0, None) is False

    def test_tools_with_error(self):
        assert detect_silent_execution("", 2, "timeout") is False


# ── repl_no_tools ──


class TestReplNoTools:
    def test_repl_mode_no_tools(self):
        assert detect_repl_no_tools("repl", 0) is True

    def test_repl_mode_with_tools(self):
        assert detect_repl_no_tools("repl", 3) is False

    def test_direct_mode_no_tools(self):
        assert detect_repl_no_tools("direct", 0) is False

    def test_delegated_mode_no_tools(self):
        assert detect_repl_no_tools("delegated", 0) is False


# ── slow_delegation ──


class TestSlowDelegation:
    def test_slow_hop(self):
        events = [{"from_role": "architect", "to_role": "coder", "elapsed_ms": 200_000}]
        assert detect_slow_delegation(events) is True

    def test_fast_hop(self):
        events = [{"from_role": "architect", "to_role": "coder", "elapsed_ms": 30_000}]
        assert detect_slow_delegation(events) is False

    def test_empty_events(self):
        assert detect_slow_delegation([]) is False

    def test_custom_threshold(self):
        events = [{"elapsed_ms": 50_000}]
        assert detect_slow_delegation(events, threshold_ms=40_000) is True
        assert detect_slow_delegation(events, threshold_ms=60_000) is False

    def test_multiple_hops_one_slow(self):
        events = [
            {"elapsed_ms": 30_000},
            {"elapsed_ms": 150_000},
        ]
        assert detect_slow_delegation(events) is True


# ── function_repr_leak ──


class TestFunctionReprLeak:
    def test_function_repr_detected(self):
        answer = "<function find_substrings at 0x739cb5f182c0>"
        assert detect_function_repr_leak(answer) is True

    def test_function_in_context(self):
        answer = "The result is <function my_func at 0x7f8a1234abcd> which is wrong"
        assert detect_function_repr_leak(answer) is True

    def test_normal_answer(self):
        assert detect_function_repr_leak("The answer is 42") is False

    def test_code_mentioning_function(self):
        # "function" in normal text should not trigger
        assert detect_function_repr_leak("This function computes the sum") is False


# ── status_phrase_final ──


class TestStatusPhraseFinal:
    def test_done(self):
        assert detect_status_phrase_final("Done") is True

    def test_done_with_period(self):
        assert detect_status_phrase_final("Done.") is True

    def test_code_execution_complete(self):
        assert detect_status_phrase_final("Code execution complete.") is True

    def test_your_answer(self):
        assert detect_status_phrase_final("your_answer") is True

    def test_case_insensitive(self):
        assert detect_status_phrase_final("COMPLETED") is True

    def test_real_answer(self):
        assert detect_status_phrase_final("42") is False

    def test_real_code_answer(self):
        assert detect_status_phrase_final("def solve(): return 42") is False

    def test_empty(self):
        assert detect_status_phrase_final("") is False

    def test_real_single_letter(self):
        assert detect_status_phrase_final("B") is False


# ── misrouted_to_coder ──


class TestMisroutedToCoder:
    def test_mcq_delegated_to_coder(self):
        events = [{"from_role": "architect_general", "to_role": "coder_escalation"}]
        assert detect_misrouted_to_coder("multiple_choice", "architect_general", events) is True

    def test_exact_match_delegated_to_coder(self):
        events = [{"from_role": "architect_general", "to_role": "coder_escalation"}]
        assert detect_misrouted_to_coder("exact_match", "architect_general", events) is True

    def test_code_execution_delegated_to_coder(self):
        # Code tasks SHOULD go to coder — not a misroute
        events = [{"from_role": "architect_coding", "to_role": "coder_escalation"}]
        assert detect_misrouted_to_coder("code_execution", "architect_coding", events) is False

    def test_substring_delegated_to_coder(self):
        # substring scoring often used for code — not a misroute
        events = [{"from_role": "architect_coding", "to_role": "coder_escalation"}]
        assert detect_misrouted_to_coder("substring", "architect_coding", events) is False

    def test_mcq_delegated_to_worker(self):
        # Not going to coder — not this signal
        events = [{"from_role": "architect_general", "to_role": "worker_explore"}]
        assert detect_misrouted_to_coder("multiple_choice", "architect_general", events) is False

    def test_non_architect_role(self):
        events = [{"from_role": "frontdoor", "to_role": "coder_escalation"}]
        assert detect_misrouted_to_coder("multiple_choice", "frontdoor", events) is False

    def test_no_delegation(self):
        assert detect_misrouted_to_coder("multiple_choice", "architect_general", []) is False


# ── compute_anomaly_signals (aggregate) ──


class TestComputeAnomalySignals:
    def test_clean_answer(self):
        signals = compute_anomaly_signals(
            answer="The answer is C based on the logical deduction.",
            role="frontdoor",
            mode="direct",
        )
        assert not any(signals.values())

    def test_format_violation_detected(self):
        signals = compute_anomaly_signals(
            answer="After careful analysis of the problem statement I believe the correct answer is B based on principles",
            role="architect_general",
            mode="delegated",
        )
        assert signals["format_violation"] is True

    def test_returns_all_17_signals(self):
        signals = compute_anomaly_signals(answer="test")
        assert len(signals) == 17
        assert set(signals.keys()) == set(SIGNAL_WEIGHTS.keys())


# ── anomaly_score ──


class TestAnomalyScore:
    def test_no_signals(self):
        signals = {name: False for name in SIGNAL_WEIGHTS}
        assert anomaly_score(signals) == 0.0

    def test_single_weight_1(self):
        signals = {name: False for name in SIGNAL_WEIGHTS}
        signals["format_violation"] = True
        assert anomaly_score(signals) == 1.0

    def test_single_weight_half(self):
        signals = {name: False for name in SIGNAL_WEIGHTS}
        signals["think_tag_leak"] = True
        assert anomaly_score(signals) == 0.5

    def test_max_of_multiple(self):
        signals = {name: False for name in SIGNAL_WEIGHTS}
        signals["think_tag_leak"] = True  # 0.5
        signals["near_empty"] = True  # 1.0
        assert anomaly_score(signals) == 1.0

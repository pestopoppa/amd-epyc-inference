#!/usr/bin/env python3
"""Unit tests for failure router."""

from src.failure_router import (
    FailureRouter,
    FailureContext,
    RoutingDecision,
    EscalationChain,
    ErrorCategory,
)


class TestErrorCategory:
    """Test ErrorCategory enum."""

    def test_all_categories_exist(self):
        """Test that all expected categories exist."""
        assert ErrorCategory.CODE == "code"
        assert ErrorCategory.LOGIC == "logic"
        assert ErrorCategory.TIMEOUT == "timeout"
        assert ErrorCategory.SCHEMA == "schema"
        assert ErrorCategory.FORMAT == "format"
        assert ErrorCategory.INFRASTRUCTURE == "infrastructure"
        assert ErrorCategory.UNKNOWN == "unknown"


class TestFailureContext:
    """Test FailureContext dataclass."""

    def test_basic_creation(self):
        """Test creating a basic failure context."""
        context = FailureContext(
            role="worker",
            failure_count=1,
            error_category="code",
        )
        assert context.role == "worker"
        assert context.failure_count == 1
        assert context.error_category == ErrorCategory.CODE

    def test_string_to_enum_conversion(self):
        """Test that string error_category is converted to enum."""
        context = FailureContext(
            role="coder",
            failure_count=0,
            error_category="logic",
        )
        assert context.error_category == ErrorCategory.LOGIC

    def test_unknown_category(self):
        """Test that unknown category string becomes UNKNOWN enum."""
        context = FailureContext(
            role="worker",
            failure_count=0,
            error_category="invalid_category",
        )
        assert context.error_category == ErrorCategory.UNKNOWN

    def test_enum_category_preserved(self):
        """Test that ErrorCategory enum is preserved."""
        context = FailureContext(
            role="worker",
            failure_count=0,
            error_category=ErrorCategory.TIMEOUT,
        )
        assert context.error_category == ErrorCategory.TIMEOUT

    def test_full_context(self):
        """Test context with all fields."""
        context = FailureContext(
            role="coder",
            failure_count=2,
            error_category="code",
            gate_name="unit",
            error_message="Test failed: assertion error",
            task_id="task-123",
            escalation_count=1,
        )
        assert context.gate_name == "unit"
        assert context.error_message == "Test failed: assertion error"
        assert context.task_id == "task-123"
        assert context.escalation_count == 1

    def test_role_enum_repr_normalization(self):
        """Test that Role enum repr strings are normalized to values."""
        # Simulates the bug where Role.CODER_PRIMARY gets passed as string repr
        context = FailureContext(
            role="Role.CODER_PRIMARY",
            failure_count=1,
            error_category="code",
        )
        assert context.role == "coder_primary"

    def test_role_enum_object_normalization(self):
        """Test that Role enum objects are normalized to values."""
        from src.roles import Role

        context = FailureContext(
            role=Role.WORKER_GENERAL,
            failure_count=1,
            error_category="code",
        )
        assert context.role == "worker_general"

    def test_role_string_preserved(self):
        """Test that normal string roles are preserved."""
        context = FailureContext(
            role="frontdoor",
            failure_count=1,
            error_category="code",
        )
        assert context.role == "frontdoor"


class TestEscalationChain:
    """Test EscalationChain dataclass."""

    def test_basic_chain(self):
        """Test creating a basic chain."""
        chain = EscalationChain(
            role="worker",
            escalates_to="coder",
        )
        assert chain.role == "worker"
        assert chain.escalates_to == "coder"
        assert chain.max_retries == 2  # default
        assert chain.max_escalations == 2  # default

    def test_custom_chain(self):
        """Test chain with custom values."""
        chain = EscalationChain(
            role="specialist",
            escalates_to="architect",
            max_retries=5,
            max_escalations=1,
        )
        assert chain.max_retries == 5
        assert chain.max_escalations == 1

    def test_terminal_chain(self):
        """Test chain with no escalation."""
        chain = EscalationChain(
            role="architect",
            escalates_to=None,
        )
        assert chain.escalates_to is None


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_retry_decision(self):
        """Test a retry decision."""
        decision = RoutingDecision(
            action="retry",
            next_role="worker",
            reason="First failure, retrying",
            max_retries_remaining=1,
        )
        assert decision.action == "retry"
        assert decision.next_role == "worker"
        assert decision.should_include_context is True  # default

    def test_escalate_decision(self):
        """Test an escalate decision."""
        decision = RoutingDecision(
            action="escalate",
            next_role="coder",
            reason="Escalating after max retries",
        )
        assert decision.action == "escalate"
        assert decision.next_role == "coder"

    def test_fail_decision(self):
        """Test a fail decision."""
        decision = RoutingDecision(
            action="fail",
            next_role=None,
            reason="Max escalations reached",
        )
        assert decision.action == "fail"
        assert decision.next_role is None


class TestFailureRouterInit:
    """Test FailureRouter initialization."""

    def test_default_chains(self):
        """Test that default chains are created."""
        router = FailureRouter()

        assert "worker" in router.chains
        assert "coder" in router.chains
        assert "architect" in router.chains
        assert "ingest" in router.chains

    def test_custom_chains_merge(self):
        """Test that custom chains are merged."""
        custom = {
            "specialist": EscalationChain("specialist", "architect", max_retries=3),
        }
        router = FailureRouter(custom_chains=custom)

        assert "specialist" in router.chains
        assert router.chains["specialist"].max_retries == 3
        # Default chains still exist
        assert "worker" in router.chains

    def test_custom_optional_gates(self):
        """Test custom optional gates."""
        router = FailureRouter(optional_gates={"my_gate"})

        assert "my_gate" in router.optional_gates
        # Default gates still exist
        assert "typecheck" in router.optional_gates


class TestRouteFailureRetry:
    """Test retry logic in route_failure."""

    def test_first_failure_retries(self):
        """Test that first failure triggers retry."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=1,
            error_category="code",
        )
        decision = router.route_failure(context)

        assert decision.action == "retry"
        assert decision.next_role == "worker"
        assert decision.max_retries_remaining == 0  # 2 max, 1 used, 0 remaining

    def test_second_failure_escalates(self):
        """Test that second failure (at max) escalates."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=2,
            error_category="code",
        )
        decision = router.route_failure(context)

        assert decision.action == "escalate"
        assert decision.next_role == "coder_primary"  # Specific role name

    def test_zero_failures_retries(self):
        """Test that zero failures still allows retry."""
        router = FailureRouter()
        context = FailureContext(
            role="coder",
            failure_count=0,
            error_category="logic",
        )
        decision = router.route_failure(context)

        assert decision.action == "retry"
        assert decision.next_role == "coder"
        assert decision.max_retries_remaining == 1


class TestRouteFailureEscalation:
    """Test escalation logic in route_failure."""

    def test_worker_to_coder(self):
        """Test worker escalates to coder."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=2,
            error_category="code",
        )
        decision = router.route_failure(context)

        assert decision.action == "escalate"
        assert decision.next_role == "coder_primary"  # Specific role name

    def test_coder_to_architect(self):
        """Test coder escalates to architect."""
        router = FailureRouter()
        context = FailureContext(
            role="coder",
            failure_count=2,
            error_category="code",
        )
        decision = router.route_failure(context)

        assert decision.action == "escalate"
        assert decision.next_role == "architect_general"  # Specific role name

    def test_architect_cannot_escalate(self):
        """Test architect has no escalation path."""
        router = FailureRouter()
        context = FailureContext(
            role="architect",
            failure_count=3,  # At max retries
            error_category="code",
        )
        decision = router.route_failure(context)

        assert decision.action == "fail"
        assert decision.next_role is None

    def test_max_escalations_reached(self):
        """Test that max escalations prevents further escalation."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=2,
            error_category="code",
            escalation_count=2,  # Already escalated max times
        )
        decision = router.route_failure(context)

        assert decision.action == "fail"
        assert decision.next_role is None
        assert "Max escalations" in decision.reason


class TestRouteFailureSpecialCategories:
    """Test special category handling."""

    def test_format_error_never_escalates(self):
        """Test that format errors retry but never escalate."""
        router = FailureRouter()

        # First failure - retry
        context = FailureContext(
            role="worker",
            failure_count=1,
            error_category="format",
        )
        decision = router.route_failure(context)
        assert decision.action == "retry"

        # At max retries - fail, not escalate
        context.failure_count = 2
        decision = router.route_failure(context)
        assert decision.action == "fail"
        assert decision.next_role is None

    def test_schema_error_never_escalates(self):
        """Test that schema errors retry but never escalate."""
        router = FailureRouter()
        context = FailureContext(
            role="coder",
            failure_count=2,
            error_category="schema",
        )
        decision = router.route_failure(context)

        assert decision.action == "fail"
        assert decision.next_role is None

    def test_timeout_skips_optional_gate(self):
        """Test that timeout on optional gate results in skip."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=1,
            error_category="timeout",
            gate_name="typecheck",  # Optional gate
        )
        decision = router.route_failure(context)

        assert decision.action == "skip"
        assert decision.next_role == "worker"
        assert "Skipping" in decision.reason

    def test_timeout_on_required_gate(self):
        """Test that timeout on required gate follows normal retry logic."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=1,
            error_category="timeout",
            gate_name="unit",  # Required gate
        )
        decision = router.route_failure(context)

        # Should retry, not skip
        assert decision.action == "retry"


class TestRouteFailureUnknownRole:
    """Test handling of unknown roles."""

    def test_unknown_role_fails(self):
        """Test that unknown role results in fail."""
        router = FailureRouter()
        context = FailureContext(
            role="nonexistent",
            failure_count=0,
            error_category="code",
        )
        decision = router.route_failure(context)

        assert decision.action == "fail"
        assert decision.next_role is None
        assert "Unknown role" in decision.reason


class TestGetEscalationPath:
    """Test get_escalation_path method."""

    def test_worker_path(self):
        """Test escalation path from worker."""
        router = FailureRouter()
        path = router.get_escalation_path("worker")

        assert path == ["worker", "coder", "architect"]

    def test_coder_path(self):
        """Test escalation path from coder."""
        router = FailureRouter()
        path = router.get_escalation_path("coder")

        assert path == ["coder", "architect"]

    def test_architect_path(self):
        """Test escalation path from architect (terminal)."""
        router = FailureRouter()
        path = router.get_escalation_path("architect")

        assert path == ["architect"]

    def test_unknown_role_path(self):
        """Test path for unknown role."""
        router = FailureRouter()
        path = router.get_escalation_path("unknown")

        assert path == ["unknown"]


class TestEscalationHistory:
    """Test escalation history tracking."""

    def test_record_escalation(self):
        """Test recording an escalation."""
        router = FailureRouter()
        router.record_escalation("task-1", "worker", "coder")

        history = router.get_escalation_history("task-1")
        assert history == ["worker->coder"]

    def test_multiple_escalations(self):
        """Test recording multiple escalations."""
        router = FailureRouter()
        router.record_escalation("task-1", "worker", "coder")
        router.record_escalation("task-1", "coder", "architect")

        history = router.get_escalation_history("task-1")
        assert history == ["worker->coder", "coder->architect"]

    def test_separate_task_histories(self):
        """Test that task histories are separate."""
        router = FailureRouter()
        router.record_escalation("task-1", "worker", "coder")
        router.record_escalation("task-2", "coder", "architect")

        assert router.get_escalation_history("task-1") == ["worker->coder"]
        assert router.get_escalation_history("task-2") == ["coder->architect"]

    def test_nonexistent_task_history(self):
        """Test getting history for nonexistent task."""
        router = FailureRouter()
        history = router.get_escalation_history("nonexistent")

        assert history == []

    def test_clear_specific_history(self):
        """Test clearing specific task history."""
        router = FailureRouter()
        router.record_escalation("task-1", "worker", "coder")
        router.record_escalation("task-2", "worker", "coder")
        router.clear_history("task-1")

        assert router.get_escalation_history("task-1") == []
        assert router.get_escalation_history("task-2") == ["worker->coder"]

    def test_clear_all_history(self):
        """Test clearing all history."""
        router = FailureRouter()
        router.record_escalation("task-1", "worker", "coder")
        router.record_escalation("task-2", "worker", "coder")
        router.clear_history()

        assert router.get_escalation_history("task-1") == []
        assert router.get_escalation_history("task-2") == []


class TestShouldIncludeErrorContext:
    """Test should_include_error_context method."""

    def test_code_error_includes_context(self):
        """Test that code errors always include context."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=5,
            error_category="code",
        )
        assert router.should_include_error_context(context) is True

    def test_logic_error_includes_context(self):
        """Test that logic errors always include context."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=5,
            error_category="logic",
        )
        assert router.should_include_error_context(context) is True

    def test_first_failure_includes_context(self):
        """Test that first failure includes context."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=1,
            error_category="format",
        )
        assert router.should_include_error_context(context) is True

    def test_format_error_excludes_context_after_many_failures(self):
        """Test that format errors exclude context after many failures."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=5,
            error_category="format",
        )
        assert router.should_include_error_context(context) is False


class TestChainManagement:
    """Test chain management methods."""

    def test_get_chain(self):
        """Test getting a chain."""
        router = FailureRouter()
        chain = router.get_chain("worker")

        assert chain is not None
        assert chain.role == "worker"
        assert chain.escalates_to == "coder"

    def test_get_chain_nonexistent(self):
        """Test getting nonexistent chain."""
        router = FailureRouter()
        chain = router.get_chain("nonexistent")

        assert chain is None

    def test_add_chain(self):
        """Test adding a new chain."""
        router = FailureRouter()
        new_chain = EscalationChain("specialist", "architect", max_retries=5)
        router.add_chain(new_chain)

        assert router.get_chain("specialist") == new_chain

    def test_update_chain(self):
        """Test updating an existing chain."""
        router = FailureRouter()
        updated_chain = EscalationChain("worker", "architect", max_retries=10)
        router.add_chain(updated_chain)

        chain = router.get_chain("worker")
        assert chain.escalates_to == "architect"
        assert chain.max_retries == 10


class TestFormatFailureReport:
    """Test format_failure_report method."""

    def test_basic_report(self):
        """Test generating a basic failure report."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=1,
            error_category="code",
        )
        decision = RoutingDecision(
            action="retry",
            next_role="worker",
            reason="First failure, retrying",
        )
        report = router.format_failure_report(context, decision)

        assert "FAILURE REPORT" in report
        assert "Role: worker" in report
        assert "Failure Count: 1" in report
        assert "Error Category: code" in report
        assert "DECISION" in report
        assert "Action: RETRY" in report

    def test_report_with_gate(self):
        """Test report includes gate name."""
        router = FailureRouter()
        context = FailureContext(
            role="coder",
            failure_count=2,
            error_category="code",
            gate_name="unit",
        )
        decision = RoutingDecision(
            action="escalate",
            next_role="architect",
            reason="Escalating",
        )
        report = router.format_failure_report(context, decision)

        assert "Gate: unit" in report

    def test_report_with_error_message(self):
        """Test report includes error message."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=1,
            error_category="code",
            error_message="AssertionError: expected 5, got 3",
        )
        decision = RoutingDecision(
            action="retry",
            next_role="worker",
            reason="Retrying",
        )
        report = router.format_failure_report(context, decision)

        assert "Error Message:" in report
        assert "AssertionError" in report

    def test_report_truncates_long_error(self):
        """Test that long error messages are truncated."""
        router = FailureRouter()
        context = FailureContext(
            role="worker",
            failure_count=1,
            error_category="code",
            error_message="x" * 1000,
        )
        decision = RoutingDecision(
            action="retry",
            next_role="worker",
            reason="Retrying",
        )
        report = router.format_failure_report(context, decision)

        # Error should be truncated to 500 chars
        assert "x" * 500 in report
        assert "x" * 501 not in report


class TestIngestRole:
    """Test ingest role specific behavior."""

    def test_ingest_escalates_to_architect(self):
        """Test ingest role escalates directly to architect."""
        router = FailureRouter()
        context = FailureContext(
            role="ingest",
            failure_count=1,  # ingest has max_retries=1
            error_category="code",
        )
        decision = router.route_failure(context)

        assert decision.action == "escalate"
        assert decision.next_role == "architect_general"  # Specific role name

    def test_ingest_path(self):
        """Test ingest escalation path."""
        router = FailureRouter()
        path = router.get_escalation_path("ingest")

        assert path == ["ingest", "architect"]


class TestLearnedEscalationParsing:
    """Test LearnedEscalationPolicy action parsing."""

    def test_parse_arrow_format_action(self):
        """Test parsing 'escalate:from->to' format extracts target role."""
        from src.failure_router import LearnedEscalationPolicy
        from unittest.mock import MagicMock

        # Create mock retriever that returns action in arrow format
        mock_retriever = MagicMock()
        mock_memory = MagicMock()
        mock_memory.action = "escalate:worker->coder_primary"
        mock_memory.id = "test-id"

        mock_result = MagicMock()
        mock_result.memory = mock_memory
        mock_result.combined_score = 0.85

        mock_retriever.retrieve_for_escalation.return_value = [mock_result]
        mock_retriever.should_use_learned.return_value = True

        policy = LearnedEscalationPolicy(retriever=mock_retriever)
        result = policy.query(
            FailureContext(role="worker", failure_count=2, error_category="code")
        )

        assert result.should_use_learned is True
        assert result.suggested_action == "escalate"
        assert result.suggested_role == "coder_primary"  # Extracted from arrow format

    def test_parse_simple_format_action(self):
        """Test parsing 'escalate:role' format (no arrow)."""
        from src.failure_router import LearnedEscalationPolicy
        from unittest.mock import MagicMock

        mock_retriever = MagicMock()
        mock_memory = MagicMock()
        mock_memory.action = "escalate:architect_general"  # No arrow
        mock_memory.id = "test-id"

        mock_result = MagicMock()
        mock_result.memory = mock_memory
        mock_result.combined_score = 0.80

        mock_retriever.retrieve_for_escalation.return_value = [mock_result]
        mock_retriever.should_use_learned.return_value = True

        policy = LearnedEscalationPolicy(retriever=mock_retriever)
        result = policy.query(
            FailureContext(role="coder", failure_count=2, error_category="code")
        )

        assert result.suggested_role == "architect_general"

    def test_parse_retry_action(self):
        """Test parsing 'retry' action (no colon)."""
        from src.failure_router import LearnedEscalationPolicy
        from unittest.mock import MagicMock

        mock_retriever = MagicMock()
        mock_memory = MagicMock()
        mock_memory.action = "retry"
        mock_memory.id = "test-id"

        mock_result = MagicMock()
        mock_result.memory = mock_memory
        mock_result.combined_score = 0.75

        mock_retriever.retrieve_for_escalation.return_value = [mock_result]
        mock_retriever.should_use_learned.return_value = True

        policy = LearnedEscalationPolicy(retriever=mock_retriever)
        result = policy.query(
            FailureContext(role="worker", failure_count=1, error_category="code")
        )

        assert result.suggested_action == "retry"
        assert result.suggested_role is None

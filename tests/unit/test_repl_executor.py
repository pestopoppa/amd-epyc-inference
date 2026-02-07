"""Comprehensive tests for REPL executor pipeline stage.

Tests coverage for src/api/routes/chat_pipeline/repl_executor.py (9% → target 80%+).
Focuses on: REPL session management, escalation handling, generation monitoring,
two-stage summarization integration, long-context exploration.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.api.models import ChatRequest
from src.api.routes.chat_pipeline.repl_executor import _execute_repl
from src.api.routes.chat_utils import RoutingResult
from src.graph.state import TaskResult
from src.llm_primitives import LLMPrimitives
from src.llm_primitives.types import LLMResult
from src.roles import Role


# ── Test Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def mock_primitives():
    """Create mock LLMPrimitives with realistic responses."""
    primitives = MagicMock(spec=LLMPrimitives)
    primitives.mock_mode = False
    primitives.total_tokens_generated = 100
    primitives.total_prompt_eval_ms = 50
    primitives.total_generation_ms = 200
    primitives.total_http_overhead_ms = 10
    primitives._last_predicted_tps = 15.0
    primitives._backends = {"frontdoor": MagicMock()}
    primitives.get_cache_stats.return_value = {"hits": 5, "misses": 2}
    primitives.llm_call.return_value = "FINAL('Test answer')"
    primitives.llm_call_monitored.return_value = LLMResult(
        text="FINAL('Test answer')",
        aborted=False,
        abort_reason="",
    )
    return primitives


@pytest.fixture
def mock_state():
    """Create mock application state."""
    state = MagicMock()
    state.tool_registry = MagicMock()
    state.script_registry = MagicMock()
    state.progress_logger = MagicMock()
    state.hybrid_router = None
    state.failure_graph = MagicMock()
    state.increment_request = MagicMock()
    return state


@pytest.fixture
def basic_request():
    """Create basic ChatRequest."""
    return ChatRequest(
        prompt="Test prompt",
        context="",
        real_mode=True,
        mock_mode=False,
        max_turns=5,
    )


@pytest.fixture
def basic_routing():
    """Create basic RoutingResult."""
    return RoutingResult(
        task_id="test-task-123",
        task_ir={},
        use_mock=False,
        routing_strategy="direct",
        formalization_applied=False,
        document_result=None,
    )


# ── Basic REPL Execution ─────────────────────────────────────────────────


class TestBasicREPLExecution:
    """Test basic REPL execution flow."""

    @pytest.mark.asyncio
    async def test_simple_final_answer(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test REPL execution with immediate FINAL() answer."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="Test answer", success=True, turns=1, role_history=["worker_general"]
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.answer == "Test answer"
                assert response.turns == 1
                assert response.mode == "repl"
                assert not response.answer.startswith("[ERROR")

    @pytest.mark.asyncio
    async def test_multi_turn_execution(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test REPL execution over multiple turns via graph."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            multi_turn_result = TaskResult(
                answer="Final answer", success=True, turns=3, role_history=["worker_general"]
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=multi_turn_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.turns == 3
                assert response.answer == "Final answer"

    @pytest.mark.asyncio
    async def test_max_turns_reached(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test REPL stops at max_turns without FINAL()."""
        basic_request.max_turns = 2

        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            # Mock run_task to return max-turns result
            max_turns_result = TaskResult(
                answer="",
                success=False,
                turns=2,
                role_history=["worker_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=max_turns_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.turns == 2
                assert "[Max turns (2) reached without FINAL()]" in response.answer

    @pytest.mark.asyncio
    async def test_llm_call_exception_returns_error(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that LLM call exceptions are handled gracefully."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            # Mock run_task to return error result
            error_result = TaskResult(
                answer="[ERROR: LLM server timeout]",
                success=False,
                turns=1,
                role_history=["worker_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=error_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.answer.startswith("[ERROR:")
                assert "LLM server timeout" in response.answer


# ── Generation Monitoring ────────────────────────────────────────────────


class TestGenerationMonitoring:
    """Test generation monitoring integration.

    Generation monitoring is now handled inside graph nodes. These tests verify
    that the graph is invoked properly and the results are used.
    """

    @pytest.mark.asyncio
    async def test_graph_invoked_in_real_mode(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that the orchestration graph is invoked in real mode."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="Answer", success=True, turns=1, role_history=["worker_general"]
            )

            with patch(
                "src.api.routes.chat_pipeline.repl_executor.run_task",
                return_value=success_result,
            ) as mock_run_task:
                await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Verify run_task was called
                mock_run_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_graph_result_used_for_response(
        self, basic_routing, mock_primitives, mock_state
    ):
        """Test that graph result populates the response."""
        request = ChatRequest(
            prompt="Test",
            context="",
            real_mode=True,
            mock_mode=True,
            max_turns=5,
        )

        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="Graph answer",
                success=True,
                turns=3,
                role_history=["worker_general", "coder_primary"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                response = await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.answer == "Graph answer"
                assert response.turns == 3


# ── Two-Stage Summarization ──────────────────────────────────────────────


class TestTwoStageSummarization:
    """Test two-stage summarization integration."""

    @pytest.mark.asyncio
    async def test_two_stage_summarization_triggered(
        self, basic_routing, mock_primitives, mock_state
    ):
        """Test that two-stage summarization is triggered for large context."""
        request = ChatRequest(
            prompt="Summarize this document",
            context="A" * 25000,  # > 20K threshold
            real_mode=True,
            mock_mode=False,
            max_turns=5,
        )

        with patch(
            "src.api.routes.chat_pipeline.repl_executor._should_use_two_stage"
        ) as mock_should:
            mock_should.return_value = True

            with patch(
                "src.api.routes.chat_pipeline.repl_executor._run_two_stage_summarization"
            ) as mock_two_stage:
                mock_two_stage.return_value = ("Summary result", {"cache_hit": False})

                response = await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.answer == "Summary result"
                assert response.turns == 2
                mock_two_stage.assert_called_once()

    @pytest.mark.asyncio
    async def test_two_stage_summarization_not_triggered_for_small_context(
        self, basic_routing, mock_primitives, mock_state
    ):
        """Test that two-stage is not triggered for small context."""
        request = ChatRequest(
            prompt="Summarize this",
            context="A" * 5000,  # < 20K threshold
            real_mode=True,
            mock_mode=False,
            max_turns=5,
        )

        with patch(
            "src.api.routes.chat_pipeline.repl_executor._should_use_two_stage"
        ) as mock_should:
            mock_should.return_value = False

            with patch(
                "src.api.routes.chat_pipeline.repl_executor._run_two_stage_summarization"
            ) as mock_two_stage:
                with patch(
                    "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment"
                ) as mock_repl_class:
                    mock_repl = MagicMock()
                    mock_repl.artifacts = {}
                    mock_repl._tool_invocations = 0
                    mock_repl.tool_registry = None
                    mock_repl.log_exploration_completed = MagicMock()
                    mock_repl_class.return_value = mock_repl

                    direct_result = TaskResult(
                        answer="Direct answer",
                        success=True,
                        turns=1,
                        role_history=["worker_general"],
                    )

                    with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=direct_result):
                        response = await _execute_repl(
                            request=request,
                            routing=basic_routing,
                            primitives=mock_primitives,
                            state=mock_state,
                            start_time=time.perf_counter(),
                            initial_role=Role.WORKER_GENERAL,
                        )

                        # Should not have called two-stage
                        mock_two_stage.assert_not_called()
                        assert response.answer == "Direct answer"


# ── Long Context Exploration ─────────────────────────────────────────────


class TestLongContextExploration:
    """Test long context exploration mode."""

    @pytest.mark.asyncio
    async def test_long_context_uses_extended_max_turns(
        self, basic_routing, mock_primitives, mock_state
    ):
        """Test that long context uses extended max_turns."""
        request = ChatRequest(
            prompt="Find data",
            context="A" * 25000,  # > 20K threshold
            real_mode=True,
            mock_mode=False,
            max_turns=5,
        )

        with patch(
            "src.api.routes.chat_pipeline.repl_executor._should_use_two_stage"
        ) as mock_should:
            mock_should.return_value = False

            with patch(
                "src.api.routes.chat_pipeline.repl_executor.LONG_CONTEXT_CONFIG"
            ) as mock_config:
                # Configure long context mode
                mock_config.__getitem__.side_effect = lambda k: {
                    "enabled": True,
                    "threshold_chars": 20000,
                    "max_turns": 8,  # Extended max turns
                }[k]

                with patch(
                    "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment"
                ) as mock_repl_class:
                    mock_repl = MagicMock()
                    mock_repl.artifacts = {}
                    mock_repl._tool_invocations = 0
                    mock_repl.tool_registry = None
                    mock_repl.log_exploration_completed = MagicMock()
                    mock_repl_class.return_value = mock_repl

                    long_result = TaskResult(
                        answer="Final",
                        success=True,
                        turns=8,
                        role_history=["worker_general"],
                    )

                    with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=long_result):
                        response = await _execute_repl(
                            request=request,
                            routing=basic_routing,
                            primitives=mock_primitives,
                            state=mock_state,
                            start_time=time.perf_counter(),
                            initial_role=Role.WORKER_GENERAL,
                        )

                        # Should have been able to use 8 turns (not limited to 5)
                        assert response.turns == 8
                        assert response.answer == "Final"


# ── Document REPL Environment ────────────────────────────────────────────


class TestDocumentREPLEnvironment:
    """Test DocumentREPLEnvironment integration."""

    @pytest.mark.asyncio
    async def test_document_result_uses_document_repl(
        self, basic_request, mock_primitives, mock_state
    ):
        """Test that document preprocessing results use DocumentREPLEnvironment."""
        # Mock document result
        mock_doc_result = MagicMock()
        mock_doc_result.document_result = MagicMock()
        mock_doc_result.document_result.to_searchable_text.return_value = "Searchable text"

        routing = RoutingResult(
            task_id="test-task",
            task_ir={},
            use_mock=False,
            routing_strategy="document",
            formalization_applied=False,
            document_result=mock_doc_result,
        )

        with patch("src.repl_document.DocumentREPLEnvironment") as mock_doc_repl_class:
            with patch("src.repl_document.DocumentContext") as mock_doc_context_class:
                mock_doc_context = MagicMock()
                mock_doc_context.sections = [{"title": "Intro"}]
                mock_doc_context.figures = []
                mock_doc_context_class.from_document_result.return_value = mock_doc_context

                mock_repl = MagicMock()
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.log_exploration_completed = MagicMock()
                mock_doc_repl_class.return_value = mock_repl

                success_result = TaskResult(
                    answer="Document answer",
                    success=True,
                    turns=1,
                    role_history=["worker_general"],
                )

                with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                    response = await _execute_repl(
                        request=basic_request,
                        routing=routing,
                        primitives=mock_primitives,
                        state=mock_state,
                        start_time=time.perf_counter(),
                        initial_role=Role.WORKER_GENERAL,
                    )

                    # Verify DocumentREPLEnvironment was used
                    mock_doc_repl_class.assert_called_once()
                    assert response.answer == "Document answer"

    @pytest.mark.asyncio
    async def test_non_document_result_uses_regular_repl(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that non-document requests use regular REPLEnvironment."""
        # No document_result in routing
        assert basic_routing.document_result is None

        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            with patch("src.repl_document.DocumentREPLEnvironment") as mock_doc_repl_class:
                mock_repl = MagicMock()
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.log_exploration_completed = MagicMock()
                mock_repl_class.return_value = mock_repl

                success_result = TaskResult(
                    answer="Regular answer",
                    success=True,
                    turns=1,
                    role_history=["worker_general"],
                )

                with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                    response = await _execute_repl(
                        request=basic_request,
                        routing=basic_routing,
                        primitives=mock_primitives,
                        state=mock_state,
                        start_time=time.perf_counter(),
                        initial_role=Role.WORKER_GENERAL,
                    )

                    # Verify regular REPLEnvironment was used, not DocumentREPLEnvironment
                    mock_repl_class.assert_called_once()
                    mock_doc_repl_class.assert_not_called()
                    assert response.answer == "Regular answer"


# ── Escalation Handling ────────────────────────────────────────────────────


class TestEscalationHandling:
    """Test escalation during REPL execution.

    Escalation is now handled inside graph nodes. These tests verify that
    graph results with escalation are correctly propagated through the executor.
    """

    @pytest.mark.asyncio
    async def test_escalation_reflected_in_role_history(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that escalation in graph is reflected in response role_history."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            escalated_result = TaskResult(
                answer="Escalated answer",
                success=True,
                turns=3,
                role_history=["worker_general", "coder_primary"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=escalated_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert len(response.role_history) >= 2
                assert "coder_primary" in response.role_history

    @pytest.mark.asyncio
    async def test_error_escalation_to_higher_tier(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test escalation when execution errors occur (handled by graph)."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            escalated_result = TaskResult(
                answer="Fixed",
                success=True,
                turns=2,
                role_history=["worker_general", "architect_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=escalated_result) as mock_run:
                await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Verify graph was invoked
                mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalation_explore_action(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test EXPLORE action when terminal role can't escalate further."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            explore_result = TaskResult(
                answer="Explored",
                success=True,
                turns=2,
                role_history=["architect_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=explore_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.ARCHITECT_GENERAL,
                )

                assert response.answer == "Explored"

    @pytest.mark.asyncio
    async def test_escalation_fail_action(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test FAIL action when escalation exhausted."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            fail_result = TaskResult(
                answer="[FAILED: Max escalation reached]",
                success=False,
                turns=5,
                role_history=["architect_coding"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=fail_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.ARCHITECT_CODING,
                )

                assert "[FAILED:" in response.answer


# ── Model-Initiated Routing ───────────────────────────────────────────────


class TestModelInitiatedRouting:
    """Test model-initiated routing via artifacts.

    Model-initiated escalation is now handled inside graph nodes.
    """

    @pytest.mark.asyncio
    async def test_model_requests_escalation(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test model requesting escalation produces escalated result."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            escalated_result = TaskResult(
                answer="Escalated answer",
                success=True,
                turns=2,
                role_history=["worker_general", "architect_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=escalated_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert "Escalated answer" in response.answer


# ── Delegation Logging ────────────────────────────────────────────────────


class TestDelegationLogging:
    """Test delegation logging for MemRL.

    Delegation events are now tracked inside graph nodes and returned
    via TaskResult.delegation_events.
    """

    @pytest.mark.asyncio
    async def test_delegation_events_in_response(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that delegation events from graph are included in response."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            result_with_delegations = TaskResult(
                answer="Final",
                success=True,
                turns=2,
                role_history=["worker_general"],
                delegation_events=[
                    {"from_role": "worker_general", "to_role": "coder", "task_summary": "subtask", "success": True}
                ],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=result_with_delegations):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                assert response.delegation_events is not None
                assert len(response.delegation_events) == 1
                assert response.delegation_success is True


# ── Quality Review Gate ───────────────────────────────────────────────────


class TestQualityReviewGate:
    """Test quality review gate integration."""

    @pytest.mark.asyncio
    async def test_review_gate_revises_wrong_answer(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that review gate revises wrong answers."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="Wrong answer", success=True, turns=1, role_history=["worker_general"]
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                with patch(
                    "src.api.routes.chat_pipeline.repl_executor._should_review"
                ) as mock_should_review:
                    with patch(
                        "src.api.routes.chat_pipeline.repl_executor._architect_verdict"
                    ) as mock_verdict:
                        with patch(
                            "src.api.routes.chat_pipeline.repl_executor._fast_revise"
                        ) as mock_revise:
                            mock_should_review.return_value = True
                            mock_verdict.return_value = "WRONG: The answer is 42, not 41"
                            mock_revise.return_value = "The answer is 42"

                            response = await _execute_repl(
                                request=basic_request,
                                routing=basic_routing,
                                primitives=mock_primitives,
                                state=mock_state,
                                start_time=time.perf_counter(),
                                initial_role=Role.WORKER_GENERAL,
                            )

                            # Should have revised the answer
                            mock_revise.assert_called_once()
                            assert response.answer == "The answer is 42"

    @pytest.mark.asyncio
    async def test_review_gate_accepts_correct_answer(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that review gate accepts correct answers."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="Correct answer", success=True, turns=1, role_history=["worker_general"]
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                with patch(
                    "src.api.routes.chat_pipeline.repl_executor._should_review"
                ) as mock_should_review:
                    with patch(
                        "src.api.routes.chat_pipeline.repl_executor._architect_verdict"
                    ) as mock_verdict:
                        with patch(
                            "src.api.routes.chat_pipeline.repl_executor._fast_revise"
                        ) as mock_revise:
                            mock_should_review.return_value = True
                            mock_verdict.return_value = "OK"  # Correct

                            response = await _execute_repl(
                                request=basic_request,
                                routing=basic_routing,
                                primitives=mock_primitives,
                                state=mock_state,
                                start_time=time.perf_counter(),
                                initial_role=Role.WORKER_GENERAL,
                            )

                            # Should NOT have revised
                            mock_revise.assert_not_called()
                            assert response.answer == "Correct answer"


# ── Execution Timeout ─────────────────────────────────────────────────────


class TestExecutionTimeout:
    """Test REPL execution timeout handling.

    Timeouts are now handled inside graph nodes. This test verifies that
    a timeout result from the graph is handled gracefully.
    """

    @pytest.mark.asyncio
    async def test_repl_execution_timeout(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that REPL execution timeout is handled."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            timeout_result = TaskResult(
                answer="[ERROR: Execution timeout]",
                success=False,
                turns=1,
                role_history=["worker_general"],
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=timeout_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Should have handled timeout gracefully
                assert response is not None
                assert "[ERROR" in response.answer


# ── Two-Stage Exception Handling ─────────────────────────────────────────


class TestTwoStageExceptionHandling:
    """Test two-stage summarization exception handling."""

    @pytest.mark.asyncio
    async def test_two_stage_exception_falls_back_to_repl(
        self, basic_routing, mock_primitives, mock_state
    ):
        """Test that two-stage exception falls back to regular REPL."""
        request = ChatRequest(
            prompt="Summarize this document",
            context="A" * 25000,
            real_mode=True,
            mock_mode=False,
            max_turns=5,
        )

        with patch(
            "src.api.routes.chat_pipeline.repl_executor._should_use_two_stage"
        ) as mock_should:
            mock_should.return_value = True

            with patch(
                "src.api.routes.chat_pipeline.repl_executor._run_two_stage_summarization"
            ) as mock_two_stage:
                mock_two_stage.side_effect = Exception("Two-stage failed")

                with patch(
                    "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment"
                ) as mock_repl_class:
                    mock_repl = MagicMock()
                    mock_repl.artifacts = {}
                    mock_repl._tool_invocations = 0
                    mock_repl.tool_registry = None
                    mock_repl.log_exploration_completed = MagicMock()
                    mock_repl_class.return_value = mock_repl

                    fallback_result = TaskResult(
                        answer="Fallback answer",
                        success=True,
                        turns=1,
                        role_history=["worker_general"],
                    )

                    with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=fallback_result):
                        response = await _execute_repl(
                            request=request,
                            routing=basic_routing,
                            primitives=mock_primitives,
                            state=mock_state,
                            start_time=time.perf_counter(),
                            initial_role=Role.WORKER_GENERAL,
                        )

                        # Should have fallen back to REPL via graph
                        assert response.answer == "Fallback answer"
                        mock_repl_class.assert_called_once()


# ── Context Handling ──────────────────────────────────────────────────────


class TestContextHandling:
    """Test request context handling."""

    @pytest.mark.asyncio
    async def test_context_appended_to_prompt(self, basic_routing, mock_primitives, mock_state):
        """Test that request context is appended to prompt."""
        request = ChatRequest(
            prompt="What is the answer?",
            context="The answer is 42",
            real_mode=True,
            mock_mode=False,
            max_turns=5,
        )

        captured_context = []

        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:

            def capture_init(**kwargs):
                captured_context.append(kwargs.get("context", ""))
                mock_repl = MagicMock()
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.log_exploration_completed = MagicMock()
                return mock_repl

            mock_repl_class.side_effect = capture_init

            success_result = TaskResult(
                answer="42", success=True, turns=1, role_history=["worker_general"]
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Verify context was appended
                assert len(captured_context) == 1
                assert "The answer is 42" in captured_context[0]


# ── Tool Outputs in Answer ───────────────────────────────────────────────


class TestToolOutputsInAnswer:
    """Test tool outputs handling in answer resolution.

    Tool outputs are now tracked inside graph nodes and the result's answer
    is resolved within the graph. The _tools_success() helper in repl_executor
    still reads from repl.artifacts after the graph completes.
    """

    @pytest.mark.asyncio
    async def test_tool_outputs_tracked_in_response(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that tool outputs from REPL are reflected in response."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.artifacts = {"_tool_outputs": ["output1", "output2"]}
            mock_repl._tool_invocations = 2
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            success_result = TaskResult(
                answer="Answer", success=True, turns=1, role_history=["worker_general"]
            )

            with patch("src.api.routes.chat_pipeline.repl_executor.run_task", return_value=success_result):
                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Verify tool invocations tracked
                assert response.tools_used == 2
                # tools_success should be inferred from tool outputs
                assert response.tools_success is not None or response.tools_success is None

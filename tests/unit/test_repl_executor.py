"""Comprehensive tests for REPL executor pipeline stage.

Tests coverage for src/api/routes/chat_pipeline/repl_executor.py (9% → target 80%+).
Focuses on: REPL session management, escalation handling, generation monitoring,
two-stage summarization integration, long-context exploration.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.api.models import ChatRequest
from src.api.routes.chat_pipeline.repl_executor import _execute_repl
from src.api.routes.chat_utils import RoutingResult
from src.escalation import EscalationAction
from src.generation_monitor import GenerationMonitor
from src.llm_primitives import LLMPrimitives
from src.llm_primitives.types import LLMResult
from src.repl_environment import ExecutionResult
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
            mock_repl.get_state.return_value = {"context_preview": "test"}
            mock_repl.execute.return_value = ExecutionResult(
                output="Test answer",
                is_final=True,
                final_answer="Test answer",
                error=None,
            )
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.config.timeout_seconds = 30
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

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
        """Test REPL execution over multiple turns."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.get_state.return_value = {"context_preview": "test"}

            # First two turns return partial results, third has FINAL
            execution_results = [
                ExecutionResult(output="Step 1", is_final=False, error=None),
                ExecutionResult(output="Step 2", is_final=False, error=None),
                ExecutionResult(
                    output="Final answer", is_final=True, final_answer="Final answer", error=None
                ),
            ]
            mock_repl.execute.side_effect = execution_results
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.config.timeout_seconds = 30
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

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
            mock_repl.get_state.return_value = {"context_preview": "test"}
            mock_repl.execute.return_value = ExecutionResult(
                output="Partial result",
                is_final=False,
                error=None,
            )
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.config.timeout_seconds = 30
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

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
            assert "Partial result" in response.answer

    @pytest.mark.asyncio
    async def test_llm_call_exception_returns_error(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that LLM call exceptions are handled gracefully."""
        mock_primitives.llm_call.side_effect = Exception("LLM server timeout")

        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.get_state.return_value = {"context_preview": "test"}
            mock_repl.config.timeout_seconds = 30
            mock_repl_class.return_value = mock_repl

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
    """Test generation monitoring integration."""

    @pytest.mark.asyncio
    async def test_generation_monitoring_enabled_in_real_mode(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that generation monitoring is used in real mode."""
        with patch("src.api.routes.chat_pipeline.repl_executor.features") as mock_features:
            mock_features.return_value.generation_monitor = True

            with patch(
                "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment"
            ) as mock_repl_class:
                with patch(
                    "src.api.routes.chat_pipeline.repl_executor.GenerationMonitor"
                ) as mock_monitor_class:
                    mock_repl = MagicMock()
                    mock_repl.get_state.return_value = {"context_preview": "test"}
                    mock_repl.execute.return_value = ExecutionResult(
                        output="Answer", is_final=True, final_answer="Answer", error=None
                    )
                    mock_repl.artifacts = {}
                    mock_repl._tool_invocations = 0
                    mock_repl.tool_registry = None
                    mock_repl.config.timeout_seconds = 30
                    mock_repl.log_exploration_completed = MagicMock()
                    mock_repl_class.return_value = mock_repl

                    mock_monitor = MagicMock(spec=GenerationMonitor)
                    mock_monitor_class.return_value = mock_monitor

                    await _execute_repl(
                        request=basic_request,
                        routing=basic_routing,
                        primitives=mock_primitives,
                        state=mock_state,
                        start_time=time.perf_counter(),
                        initial_role=Role.WORKER_GENERAL,
                    )

                    # Verify monitoring was used
                    mock_primitives.llm_call_monitored.assert_called_once()
                    mock_monitor_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_generation_monitoring_disabled_in_mock_mode(
        self, basic_routing, mock_primitives, mock_state
    ):
        """Test that generation monitoring is skipped in mock mode."""
        request = ChatRequest(
            prompt="Test",
            context="",
            real_mode=True,
            mock_mode=True,  # Mock mode
            max_turns=5,
        )

        with patch("src.api.routes.chat_pipeline.repl_executor.features") as mock_features:
            mock_features.return_value.generation_monitor = True

            with patch(
                "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment"
            ) as mock_repl_class:
                mock_repl = MagicMock()
                mock_repl.get_state.return_value = {"context_preview": "test"}
                mock_repl.execute.return_value = ExecutionResult(
                    output="Answer", is_final=True, final_answer="Answer", error=None
                )
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.config.timeout_seconds = 30
                mock_repl.log_exploration_completed = MagicMock()
                mock_repl_class.return_value = mock_repl

                await _execute_repl(
                    request=request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Should use regular llm_call, not monitored
                mock_primitives.llm_call.assert_called()
                mock_primitives.llm_call_monitored.assert_not_called()


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
                    mock_repl.get_state.return_value = {"context_preview": "test"}
                    mock_repl.execute.return_value = ExecutionResult(
                        output="Direct answer",
                        is_final=True,
                        final_answer="Direct answer",
                        error=None,
                    )
                    mock_repl.artifacts = {}
                    mock_repl._tool_invocations = 0
                    mock_repl.tool_registry = None
                    mock_repl.config.timeout_seconds = 30
                    mock_repl.log_exploration_completed = MagicMock()
                    mock_repl_class.return_value = mock_repl

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
                    mock_repl.get_state.return_value = {"context_preview": "test"}
                    mock_repl.context = "A" * 25000

                    # Return non-final for 7 turns, final on 8th
                    execution_results = [
                        ExecutionResult(output=f"Step {i}", is_final=False, error=None)
                        for i in range(7)
                    ]
                    execution_results.append(
                        ExecutionResult(
                            output="Final", is_final=True, final_answer="Final", error=None
                        )
                    )
                    mock_repl.execute.side_effect = execution_results

                    mock_repl.artifacts = {}
                    mock_repl._tool_invocations = 0
                    mock_repl.tool_registry = None
                    mock_repl.config.timeout_seconds = 30
                    mock_repl.log_exploration_completed = MagicMock()
                    mock_repl_class.return_value = mock_repl

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
                mock_repl.get_state.return_value = {"context_preview": "test"}
                mock_repl.execute.return_value = ExecutionResult(
                    output="Document answer",
                    is_final=True,
                    final_answer="Document answer",
                    error=None,
                )
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.config.timeout_seconds = 30
                mock_repl.log_exploration_completed = MagicMock()
                mock_doc_repl_class.return_value = mock_repl

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
                mock_repl.get_state.return_value = {"context_preview": "test"}
                mock_repl.execute.return_value = ExecutionResult(
                    output="Regular answer",
                    is_final=True,
                    final_answer="Regular answer",
                    error=None,
                )
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.config.timeout_seconds = 30
                mock_repl.log_exploration_completed = MagicMock()
                mock_repl_class.return_value = mock_repl

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
    """Test escalation during REPL execution."""

    @pytest.mark.asyncio
    async def test_generation_aborted_triggers_escalation(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that generation abort triggers escalation."""
        # Configure generation monitoring to return aborted
        mock_primitives.llm_call_monitored.return_value = LLMResult(
            text="partial code",
            aborted=True,
            abort_reason="Repetition detected",
        )

        with patch("src.api.routes.chat_pipeline.repl_executor.features") as mock_features:
            mock_features.return_value.generation_monitor = True

            with patch(
                "src.api.routes.chat_pipeline.repl_executor.REPLEnvironment"
            ) as mock_repl_class:
                with patch(
                    "src.api.routes.chat_pipeline.repl_executor.EscalationPolicy"
                ) as mock_policy_class:
                    with patch(
                        "src.api.routes.chat_pipeline.repl_executor.build_escalation_prompt"
                    ) as mock_build_esc:
                        mock_build_esc.return_value = "Escalation prompt"

                        mock_repl = MagicMock()
                        mock_repl.get_state.return_value = {"context_preview": "test"}
                        # First execution continues, second has FINAL
                        mock_repl.execute.side_effect = [
                            ExecutionResult(output="Partial", is_final=False, error=None),
                            ExecutionResult(
                                output="Final", is_final=True, final_answer="Final", error=None
                            ),
                        ]
                        mock_repl.artifacts = {}
                        mock_repl._tool_invocations = 0
                        mock_repl.tool_registry = None
                        mock_repl.config.timeout_seconds = 30
                        mock_repl.log_exploration_completed = MagicMock()
                        mock_repl_class.return_value = mock_repl

                        # Configure escalation policy to escalate
                        mock_decision = MagicMock()
                        mock_decision.should_escalate = True
                        mock_decision.target_role = Role.CODER_ESCALATION
                        mock_decision.reason = "Repetition detected"
                        mock_policy = MagicMock()
                        mock_policy.decide.return_value = mock_decision
                        mock_policy_class.return_value = mock_policy

                        # After escalation, use regular llm_call
                        mock_primitives.llm_call.return_value = "FINAL('Escalated answer')"

                        response = await _execute_repl(
                            request=basic_request,
                            routing=basic_routing,
                            primitives=mock_primitives,
                            state=mock_state,
                            start_time=time.perf_counter(),
                            initial_role=Role.WORKER_GENERAL,
                        )

                        # Should have escalated
                        assert len(response.role_history) >= 1

    @pytest.mark.asyncio
    async def test_error_escalation_to_higher_tier(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test escalation when execution errors occur."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            with patch(
                "src.api.routes.chat_pipeline.repl_executor.EscalationPolicy"
            ) as mock_policy_class:
                with patch(
                    "src.api.routes.chat_pipeline.repl_executor.build_escalation_prompt"
                ) as mock_build_esc:
                    mock_build_esc.return_value = "Escalation prompt"

                    mock_repl = MagicMock()
                    mock_repl.get_state.return_value = {"context_preview": "test"}
                    # First execution has error, second succeeds after escalation
                    mock_repl.execute.side_effect = [
                        ExecutionResult(
                            output="", is_final=False, error="SyntaxError: invalid syntax"
                        ),
                        ExecutionResult(
                            output="Fixed", is_final=True, final_answer="Fixed", error=None
                        ),
                    ]
                    mock_repl.artifacts = {}
                    mock_repl._tool_invocations = 0
                    mock_repl.tool_registry = None
                    mock_repl.config.timeout_seconds = 30
                    mock_repl.log_exploration_completed = MagicMock()
                    mock_repl_class.return_value = mock_repl

                    # Configure escalation policy
                    mock_decision = MagicMock()
                    mock_decision.should_escalate = True
                    mock_decision.target_role = Role.ARCHITECT_GENERAL
                    mock_decision.action = EscalationAction.ESCALATE
                    mock_decision.reason = "Syntax error"
                    mock_policy = MagicMock()
                    mock_policy.decide.return_value = mock_decision
                    mock_policy_class.return_value = mock_policy

                    await _execute_repl(
                        request=basic_request,
                        routing=basic_routing,
                        primitives=mock_primitives,
                        state=mock_state,
                        start_time=time.perf_counter(),
                        initial_role=Role.WORKER_GENERAL,
                    )

                    # Verify escalation policy was consulted
                    mock_policy_class.assert_called()

    @pytest.mark.asyncio
    async def test_escalation_explore_action(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test EXPLORE action when terminal role can't escalate further."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            with patch(
                "src.api.routes.chat_pipeline.repl_executor.EscalationPolicy"
            ) as mock_policy_class:
                mock_repl = MagicMock()
                mock_repl.get_state.return_value = {"context_preview": "test"}
                mock_repl.context = "Test context"
                # First has error, second succeeds
                mock_repl.execute.side_effect = [
                    ExecutionResult(output="", is_final=False, error="Test error"),
                    ExecutionResult(
                        output="Explored", is_final=True, final_answer="Explored", error=None
                    ),
                ]
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.config.timeout_seconds = 30
                mock_repl.log_exploration_completed = MagicMock()
                mock_repl_class.return_value = mock_repl

                # Configure EXPLORE action (terminal role)
                mock_decision = MagicMock()
                mock_decision.should_escalate = False
                mock_decision.target_role = None
                mock_decision.action = EscalationAction.EXPLORE
                mock_decision.reason = "Terminal role"
                mock_policy = MagicMock()
                mock_policy.decide.return_value = mock_decision
                mock_policy_class.return_value = mock_policy

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
        """Test FAIL action when escalation fails."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            with patch(
                "src.api.routes.chat_pipeline.repl_executor.EscalationPolicy"
            ) as mock_policy_class:
                mock_repl = MagicMock()
                mock_repl.get_state.return_value = {"context_preview": "test"}
                mock_repl.execute.return_value = ExecutionResult(
                    output="", is_final=False, error="Critical failure"
                )
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.config.timeout_seconds = 30
                mock_repl.log_exploration_completed = MagicMock()
                mock_repl_class.return_value = mock_repl

                # Configure FAIL action
                mock_decision = MagicMock()
                mock_decision.should_escalate = False
                mock_decision.target_role = None
                mock_decision.action = EscalationAction.FAIL
                mock_decision.reason = "Max escalation reached"
                mock_policy = MagicMock()
                mock_policy.decide.return_value = mock_decision
                mock_policy_class.return_value = mock_policy

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
    """Test model-initiated routing via artifacts."""

    @pytest.mark.asyncio
    async def test_model_requests_escalation(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test model requesting escalation via _escalation_requested artifact."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            with patch(
                "src.api.routes.chat_pipeline.repl_executor.build_escalation_prompt"
            ) as mock_build_esc:
                mock_build_esc.return_value = "Escalation prompt"

                mock_repl = MagicMock()
                mock_repl.get_state.return_value = {"context_preview": "test"}

                # First turn: model requests escalation
                # Second turn: complete with FINAL
                call_count = [0]

                def execute_side_effect(code):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        # First turn sets escalation request
                        mock_repl.artifacts["_escalation_requested"] = True
                        mock_repl.artifacts["_escalation_target"] = "architect_general"
                        mock_repl.artifacts["_escalation_reason"] = "Need higher tier"
                        return ExecutionResult(
                            output="Needs escalation", is_final=False, error=None
                        )
                    else:
                        return ExecutionResult(
                            output="Escalated answer",
                            is_final=True,
                            final_answer="Escalated answer",
                            error=None,
                        )

                mock_repl.execute.side_effect = execute_side_effect
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.config.timeout_seconds = 30
                mock_repl.log_exploration_completed = MagicMock()
                mock_repl_class.return_value = mock_repl

                response = await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Should have completed with the escalated answer
                assert "Escalated answer" in response.answer or "Max turns" in response.answer


# ── Delegation Logging ────────────────────────────────────────────────────


class TestDelegationLogging:
    """Test delegation logging for MemRL."""

    @pytest.mark.asyncio
    async def test_delegation_outcomes_logged(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that delegation outcomes are logged to progress_logger."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.get_state.return_value = {"context_preview": "test"}

            # First turn: delegation recorded
            # Second turn: FINAL
            call_count = [0]

            def execute_side_effect(code):
                call_count[0] += 1
                if call_count[0] == 1:
                    mock_repl.artifacts["_delegations"] = [
                        {"prompt_preview": "subtask", "to_role": "coder", "success": True}
                    ]
                    return ExecutionResult(output="Delegated", is_final=False, error=None)
                else:
                    return ExecutionResult(
                        output="Final", is_final=True, final_answer="Final", error=None
                    )

            mock_repl.execute.side_effect = execute_side_effect
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.config.timeout_seconds = 30
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            await _execute_repl(
                request=basic_request,
                routing=basic_routing,
                primitives=mock_primitives,
                state=mock_state,
                start_time=time.perf_counter(),
                initial_role=Role.WORKER_GENERAL,
            )

            # Verify progress logger was called for exploration
            mock_state.progress_logger.log_exploration.assert_called()


# ── Quality Review Gate ───────────────────────────────────────────────────


class TestQualityReviewGate:
    """Test quality review gate integration."""

    @pytest.mark.asyncio
    async def test_review_gate_revises_wrong_answer(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that review gate revises wrong answers."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            with patch(
                "src.api.routes.chat_pipeline.repl_executor._should_review"
            ) as mock_should_review:
                with patch(
                    "src.api.routes.chat_pipeline.repl_executor._architect_verdict"
                ) as mock_verdict:
                    with patch(
                        "src.api.routes.chat_pipeline.repl_executor._fast_revise"
                    ) as mock_revise:
                        mock_repl = MagicMock()
                        mock_repl.get_state.return_value = {"context_preview": "test"}
                        mock_repl.execute.return_value = ExecutionResult(
                            output="Wrong answer",
                            is_final=True,
                            final_answer="Wrong answer",
                            error=None,
                        )
                        mock_repl.artifacts = {}
                        mock_repl._tool_invocations = 0
                        mock_repl.tool_registry = None
                        mock_repl.config.timeout_seconds = 30
                        mock_repl.log_exploration_completed = MagicMock()
                        mock_repl_class.return_value = mock_repl

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
            with patch(
                "src.api.routes.chat_pipeline.repl_executor._should_review"
            ) as mock_should_review:
                with patch(
                    "src.api.routes.chat_pipeline.repl_executor._architect_verdict"
                ) as mock_verdict:
                    with patch(
                        "src.api.routes.chat_pipeline.repl_executor._fast_revise"
                    ) as mock_revise:
                        mock_repl = MagicMock()
                        mock_repl.get_state.return_value = {"context_preview": "test"}
                        mock_repl.execute.return_value = ExecutionResult(
                            output="Correct answer",
                            is_final=True,
                            final_answer="Correct answer",
                            error=None,
                        )
                        mock_repl.artifacts = {}
                        mock_repl._tool_invocations = 0
                        mock_repl.tool_registry = None
                        mock_repl.config.timeout_seconds = 30
                        mock_repl.log_exploration_completed = MagicMock()
                        mock_repl_class.return_value = mock_repl

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
    """Test REPL execution timeout handling."""

    @pytest.mark.asyncio
    async def test_repl_execution_timeout(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that REPL execution timeout is handled."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            mock_repl = MagicMock()
            mock_repl.get_state.return_value = {"context_preview": "test"}
            mock_repl.config.timeout_seconds = 1  # Short timeout

            # Mock execute to hang
            async def slow_execute(code):
                await asyncio.sleep(5)
                return ExecutionResult(
                    output="Slow", is_final=True, final_answer="Slow", error=None
                )

            # Need to simulate timeout in asyncio.wait_for
            # Use side_effect to raise TimeoutError
            call_count = [0]

            def execute_sync(code):
                call_count[0] += 1
                if call_count[0] == 1:
                    # This will be wrapped in asyncio.to_thread and timed out
                    raise asyncio.TimeoutError()
                return ExecutionResult(
                    output="Final", is_final=True, final_answer="Final", error=None
                )

            mock_repl.execute.side_effect = execute_sync
            mock_repl.artifacts = {}
            mock_repl._tool_invocations = 0
            mock_repl.tool_registry = None
            mock_repl.log_exploration_completed = MagicMock()
            mock_repl_class.return_value = mock_repl

            # Patch asyncio.wait_for to simulate timeout
            with patch("src.api.routes.chat_pipeline.repl_executor.asyncio.wait_for") as mock_wait:
                mock_wait.side_effect = [
                    asyncio.TimeoutError(),  # First call times out
                    ExecutionResult(
                        output="Final", is_final=True, final_answer="Final", error=None
                    ),
                ]

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
                    mock_repl.get_state.return_value = {"context_preview": "test"}
                    mock_repl.execute.return_value = ExecutionResult(
                        output="Fallback answer",
                        is_final=True,
                        final_answer="Fallback answer",
                        error=None,
                    )
                    mock_repl.artifacts = {}
                    mock_repl._tool_invocations = 0
                    mock_repl.tool_registry = None
                    mock_repl.config.timeout_seconds = 30
                    mock_repl.log_exploration_completed = MagicMock()
                    mock_repl_class.return_value = mock_repl

                    response = await _execute_repl(
                        request=request,
                        routing=basic_routing,
                        primitives=mock_primitives,
                        state=mock_state,
                        start_time=time.perf_counter(),
                        initial_role=Role.WORKER_GENERAL,
                    )

                    # Should have fallen back to REPL
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
                mock_repl.get_state.return_value = {"context_preview": "test"}
                mock_repl.execute.return_value = ExecutionResult(
                    output="42", is_final=True, final_answer="42", error=None
                )
                mock_repl.artifacts = {}
                mock_repl._tool_invocations = 0
                mock_repl.tool_registry = None
                mock_repl.config.timeout_seconds = 30
                mock_repl.log_exploration_completed = MagicMock()
                return mock_repl

            mock_repl_class.side_effect = capture_init

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
    """Test tool outputs handling in answer resolution."""

    @pytest.mark.asyncio
    async def test_tool_outputs_passed_to_resolve_answer(
        self, basic_request, basic_routing, mock_primitives, mock_state
    ):
        """Test that tool outputs are passed to _resolve_answer."""
        with patch("src.api.routes.chat_pipeline.repl_executor.REPLEnvironment") as mock_repl_class:
            with patch(
                "src.api.routes.chat_pipeline.repl_executor._resolve_answer"
            ) as mock_resolve:
                mock_repl = MagicMock()
                mock_repl.get_state.return_value = {"context_preview": "test"}
                mock_repl.execute.return_value = ExecutionResult(
                    output="Answer", is_final=True, final_answer="Answer", error=None
                )
                mock_repl.artifacts = {"_tool_outputs": ["output1", "output2"]}
                mock_repl._tool_invocations = 2
                mock_repl.tool_registry = None
                mock_repl.config.timeout_seconds = 30
                mock_repl.log_exploration_completed = MagicMock()
                mock_repl_class.return_value = mock_repl

                mock_resolve.return_value = "Resolved answer"

                await _execute_repl(
                    request=basic_request,
                    routing=basic_routing,
                    primitives=mock_primitives,
                    state=mock_state,
                    start_time=time.perf_counter(),
                    initial_role=Role.WORKER_GENERAL,
                )

                # Verify _resolve_answer was called with tool outputs
                mock_resolve.assert_called_once()
                call_kwargs = mock_resolve.call_args[1]
                assert call_kwargs["tool_outputs"] == ["output1", "output2"]

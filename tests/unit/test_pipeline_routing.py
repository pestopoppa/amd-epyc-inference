"""Tests for chat_pipeline/routing.py - Stage 1-3 and 5 pipeline stages."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.models import ChatRequest
from src.api.routes.chat_pipeline.routing import (
    _init_primitives,
    _plan_review_gate,
    _preprocess,
    _route_request,
)
from src.api.routes.chat_utils import RoutingResult
from src.roles import Role


class TestRouteRequest:
    """Tests for _route_request (Stage 1)."""

    def test_mock_mode_uses_frontdoor(self):
        """Mock mode routes to frontdoor by default."""
        request = ChatRequest(prompt="test", mock_mode=True, real_mode=False)
        state = MagicMock()
        state.hybrid_router = None
        state.failure_graph = None
        state.progress_logger = None

        result = _route_request(request, state)

        assert result.use_mock is True
        assert result.routing_strategy == "mock"
        assert str(Role.FRONTDOOR) in [str(r) for r in result.routing_decision]

    def test_mock_mode_with_explicit_role(self):
        """Mock mode uses explicit role when provided."""
        request = ChatRequest(prompt="test", mock_mode=True, real_mode=False, role="coder_escalation")
        state = MagicMock()
        state.hybrid_router = None
        state.failure_graph = None
        state.progress_logger = None

        result = _route_request(request, state)

        assert result.routing_strategy == "mock"
        assert "coder_escalation" in [str(r) for r in result.routing_decision]

    def test_force_role_overrides_routing(self):
        """force_role bypasses all routing logic."""
        request = ChatRequest(prompt="test", real_mode=True, force_role="architect_general")
        state = MagicMock()
        state.hybrid_router = MagicMock()
        state.failure_graph = None
        state.progress_logger = None

        result = _route_request(request, state)

        assert result.routing_strategy == "forced"
        assert result.routing_decision == ["architect_general"]
        # hybrid_router should NOT be called
        state.hybrid_router.route.assert_not_called()

    def test_explicit_role_routes_directly(self):
        """Explicit role (non-frontdoor) routes directly."""
        request = ChatRequest(prompt="test", real_mode=True, role="coder_escalation")
        state = MagicMock()
        state.hybrid_router = None
        state.failure_graph = None
        state.progress_logger = None

        result = _route_request(request, state)

        assert result.routing_strategy == "explicit"
        assert result.routing_decision == ["coder_escalation"]

    def test_explicit_frontdoor_falls_through(self):
        """Explicit 'frontdoor' role falls through to classifier."""
        request = ChatRequest(prompt="test", real_mode=True, role="frontdoor")
        state = MagicMock()
        state.hybrid_router = None
        state.failure_graph = None
        state.progress_logger = None

        with patch("src.api.routes.chat_pipeline.routing._classify_and_route") as mock_classify:
            mock_classify.return_value = (str(Role.FRONTDOOR), "rules")
            result = _route_request(request, state)

        mock_classify.assert_called_once()
        assert result.routing_strategy == "rules"

    def test_hybrid_router_used_when_available(self):
        """HybridRouter is used for real_mode requests."""
        request = ChatRequest(prompt="test", real_mode=True)
        state = MagicMock()
        state.hybrid_router.route.return_value = (["coder_escalation"], "learned")
        state.failure_graph = None
        state.progress_logger = None

        result = _route_request(request, state)

        state.hybrid_router.route.assert_called_once()
        assert result.routing_strategy == "learned"
        assert result.routing_decision == ["coder_escalation"]

    def test_classifier_fallback_when_no_router(self):
        """Classifier used when no hybrid_router."""
        request = ChatRequest(prompt="test", real_mode=True)
        state = MagicMock()
        state.hybrid_router = None
        state.failure_graph = None
        state.progress_logger = None

        with patch("src.api.routes.chat_pipeline.routing._classify_and_route") as mock_classify:
            mock_classify.return_value = (str(Role.CODER_ESCALATION), "classified")
            result = _route_request(request, state)

        mock_classify.assert_called_once()
        assert result.routing_strategy == "classified"

    def test_failure_graph_veto_high_risk(self):
        """High-risk specialist reverted to frontdoor."""
        request = ChatRequest(prompt="test", real_mode=True)
        state = MagicMock()
        state.hybrid_router.route.return_value = (["coder_escalation"], "learned")
        state.failure_graph.get_failure_risk.return_value = 0.7  # > 0.5
        state.progress_logger = None

        result = _route_request(request, state)

        assert result.routing_strategy == "failure_vetoed"
        assert str(Role.FRONTDOOR) in result.routing_decision

    def test_failure_graph_allows_low_risk(self):
        """Low-risk specialist not reverted."""
        request = ChatRequest(prompt="test", real_mode=True)
        state = MagicMock()
        state.hybrid_router.route.return_value = (["coder_escalation"], "learned")
        state.failure_graph.get_failure_risk.return_value = 0.2  # < 0.5
        state.progress_logger = None

        result = _route_request(request, state)

        assert result.routing_strategy == "learned"
        assert result.routing_decision == ["coder_escalation"]

    def test_failure_graph_skips_mock_mode(self):
        """Failure veto skipped for mock mode."""
        request = ChatRequest(prompt="test", mock_mode=True, real_mode=False, role="coder_escalation")
        state = MagicMock()
        state.failure_graph.get_failure_risk.return_value = 0.9
        state.progress_logger = None

        _route_request(request, state)

        # Failure graph should NOT be checked for mock mode
        state.failure_graph.get_failure_risk.assert_not_called()

    def test_failure_graph_skips_forced_role(self):
        """Failure veto skipped for forced role."""
        request = ChatRequest(prompt="test", real_mode=True, force_role="coder_escalation")
        state = MagicMock()
        state.failure_graph.get_failure_risk.return_value = 0.9
        state.progress_logger = None

        _route_request(request, state)

        # Failure graph should NOT be checked for forced role
        state.failure_graph.get_failure_risk.assert_not_called()

    def test_failure_graph_exception_handled(self):
        """Failure graph exception doesn't break routing."""
        request = ChatRequest(prompt="test", real_mode=True)
        state = MagicMock()
        state.hybrid_router.route.return_value = (["coder_escalation"], "learned")
        state.failure_graph.get_failure_risk.side_effect = RuntimeError("DB error")
        state.progress_logger = None

        result = _route_request(request, state)

        # Should continue with original routing despite error
        assert result.routing_strategy == "learned"
        assert result.routing_decision == ["coder_escalation"]

    def test_progress_logger_called(self):
        """Progress logger logs task start."""
        request = ChatRequest(prompt="test", real_mode=True)
        state = MagicMock()
        state.hybrid_router = None
        state.failure_graph = None

        with patch("src.api.routes.chat_pipeline.routing._classify_and_route") as mock_classify:
            mock_classify.return_value = (str(Role.FRONTDOOR), "rules")
            result = _route_request(request, state)

        state.progress_logger.log_task_started.assert_called_once()
        call_kwargs = state.progress_logger.log_task_started.call_args.kwargs
        assert call_kwargs["task_id"] == result.task_id
        assert "routing_decision" in call_kwargs

    def test_task_id_generated(self):
        """Task ID is generated with chat prefix."""
        request = ChatRequest(prompt="test", mock_mode=True)
        state = MagicMock()
        state.progress_logger = None
        state.failure_graph = None

        result = _route_request(request, state)

        assert result.task_id.startswith("chat-")
        assert len(result.task_id) == 13  # "chat-" + 8 hex chars

    def test_role_specific_timeout(self):
        """Role-specific timeout is computed."""
        request = ChatRequest(prompt="test", real_mode=True)
        state = MagicMock()
        state.hybrid_router.route.return_value = (["architect_general"], "learned")
        state.failure_graph = None
        state.progress_logger = None

        result = _route_request(request, state)

        # architect_general has longer timeout (300s in ROLE_TIMEOUTS)
        assert result.timeout_s >= 120  # At least default

    def test_image_path_triggers_vision_classification(self):
        """Image path triggers vision classification."""
        request = ChatRequest(prompt="describe", real_mode=True, image_path="/path/to/img.png")
        state = MagicMock()
        state.hybrid_router = None
        state.failure_graph = None
        state.progress_logger = None

        with patch("src.api.routes.chat_pipeline.routing._classify_and_route") as mock_classify:
            mock_classify.return_value = ("worker_vision", "classified")
            _route_request(request, state)

        mock_classify.assert_called_once()
        args, kwargs = mock_classify.call_args
        assert kwargs.get("has_image") is True or args[-1] is True

    def test_memrl_initialized_for_real_mode(self):
        """MemRL initialized early for real_mode requests."""
        request = ChatRequest(prompt="test", real_mode=True)
        state = MagicMock()
        state.hybrid_router.route.return_value = ([str(Role.FRONTDOOR)], "learned")
        state.failure_graph = None
        state.progress_logger = None

        with patch("src.api.routes.chat_pipeline.routing.ensure_memrl_initialized") as mock_init:
            _route_request(request, state)

        mock_init.assert_called_once_with(state)


class TestPreprocess:
    """Tests for _preprocess (Stage 2)."""

    def test_formalization_disabled_by_feature(self):
        """Formalization skipped when feature disabled."""
        request = ChatRequest(prompt="test", real_mode=True)
        routing = RoutingResult(
            task_id="test-123",
            task_ir={},
            use_mock=False,
            routing_decision=[str(Role.FRONTDOOR)],
            routing_strategy="rules",
            timeout_s=120,
        )
        state = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.features") as mock_features:
            mock_features.return_value.input_formalizer = False
            _preprocess(request, state, routing)

        # No formalization applied (attribute should not be set)
        assert not getattr(routing, "formalization_applied", False)

    def test_formalization_skipped_for_mock_mode(self):
        """Formalization skipped for mock mode."""
        request = ChatRequest(prompt="test", real_mode=False)
        routing = RoutingResult(
            task_id="test-123",
            task_ir={},
            use_mock=True,
            routing_decision=[str(Role.FRONTDOOR)],
            routing_strategy="mock",
            timeout_s=120,
        )
        state = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.features") as mock_features:
            mock_features.return_value.input_formalizer = True
            _preprocess(request, state, routing)

        # Should not reach formalization check
        assert not getattr(routing, "formalization_applied", False)

    def test_formalization_applied_when_needed(self):
        """Formalization applied when should_formalize returns True."""
        request = ChatRequest(prompt="solve x^2 + 2x + 1 = 0", real_mode=True, context="")
        routing = RoutingResult(
            task_id="test-123",
            task_ir={},
            use_mock=False,
            routing_decision=[str(Role.FRONTDOOR)],
            routing_strategy="rules",
            timeout_s=120,
        )
        state = MagicMock()
        state.registry = MagicMock()

        mock_fml_result = MagicMock()
        mock_fml_result.success = True
        mock_fml_result.ir_json = '{"type": "polynomial"}'
        mock_fml_result.elapsed_seconds = 0.5
        mock_fml_result.model_role = "math_formalizer"

        with patch("src.api.routes.chat_pipeline.routing.features") as mock_features:
            mock_features.return_value.input_formalizer = True
            with patch("src.formalizer.should_formalize_input") as mock_should:
                mock_should.return_value = (True, "polynomial")
                with patch("src.formalizer.formalize_prompt") as mock_fml:
                    mock_fml.return_value = mock_fml_result
                    with patch("src.formalizer.inject_formalization") as mock_inject:
                        mock_inject.return_value = "formalized context"
                        _preprocess(request, state, routing)

        assert routing.formalization_applied is True
        mock_fml.assert_called_once()
        mock_inject.assert_called_once()

    def test_formalization_skipped_when_not_needed(self):
        """Formalization skipped when should_formalize returns False."""
        request = ChatRequest(prompt="hello", real_mode=True)
        routing = RoutingResult(
            task_id="test-123",
            task_ir={},
            use_mock=False,
            routing_decision=[str(Role.FRONTDOOR)],
            routing_strategy="rules",
            timeout_s=120,
        )
        state = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.features") as mock_features:
            mock_features.return_value.input_formalizer = True
            with patch("src.formalizer.should_formalize_input") as mock_should:
                mock_should.return_value = (False, None)
                _preprocess(request, state, routing)

        # formalization_applied should not be set
        assert not getattr(routing, "formalization_applied", False)


class TestInitPrimitives:
    """Tests for _init_primitives (Stage 3)."""

    def test_real_mode_creates_primitives(self):
        """Real mode creates LLMPrimitives."""
        request = ChatRequest(prompt="test", real_mode=True)
        state = MagicMock()
        state._real_primitives = None
        state.registry = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.LLMPrimitives") as mock_primitives_cls:
            mock_primitives = MagicMock()
            mock_primitives._backends = {"port8080": MagicMock()}
            mock_primitives_cls.return_value = mock_primitives

            result = _init_primitives(request, state)

        assert result is mock_primitives
        mock_primitives_cls.assert_called_once()

    def test_real_mode_reuses_cached_primitives(self):
        """Real mode reuses cached _real_primitives."""
        request = ChatRequest(prompt="test", real_mode=True)
        cached_primitives = MagicMock()
        cached_primitives._backends = {"port8080": MagicMock()}
        state = MagicMock()
        state._real_primitives = cached_primitives
        state.registry = MagicMock()

        result = _init_primitives(request, state)

        assert result is cached_primitives
        cached_primitives.reset_counters.assert_called_once()

    def test_real_mode_no_cache_with_custom_urls(self):
        """Custom server_urls bypass cache."""
        request = ChatRequest(
            prompt="test",
            real_mode=True,
            server_urls={"custom": "http://localhost:9000"},
        )
        cached_primitives = MagicMock()
        state = MagicMock()
        state._real_primitives = cached_primitives
        state.registry = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.LLMPrimitives") as mock_primitives_cls:
            mock_primitives = MagicMock()
            mock_primitives._backends = {"custom": MagicMock()}
            mock_primitives_cls.return_value = mock_primitives

            _init_primitives(request, state)

        # Should create NEW primitives, not reuse cache
        mock_primitives_cls.assert_called_once()
        # Cache should NOT be updated (custom URLs)
        assert state._real_primitives is cached_primitives

    def test_real_mode_raises_on_init_failure(self):
        """Real mode raises HTTPException on init failure."""
        request = ChatRequest(prompt="test", real_mode=True)
        state = MagicMock()
        state._real_primitives = None
        state.registry = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.LLMPrimitives") as mock_primitives_cls:
            mock_primitives_cls.side_effect = RuntimeError("Connection failed")

            with pytest.raises(HTTPException) as exc_info:
                _init_primitives(request, state)

        assert exc_info.value.status_code == 503
        assert "Failed to initialize" in exc_info.value.detail

    def test_real_mode_raises_on_no_backends(self):
        """Real mode raises HTTPException when no backends available."""
        request = ChatRequest(prompt="test", real_mode=True)
        state = MagicMock()
        state._real_primitives = None
        state.registry = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.LLMPrimitives") as mock_primitives_cls:
            mock_primitives = MagicMock()
            mock_primitives._backends = {}  # No backends
            mock_primitives_cls.return_value = mock_primitives

            with pytest.raises(HTTPException) as exc_info:
                _init_primitives(request, state)

        assert exc_info.value.status_code == 503
        assert "No backends available" in exc_info.value.detail

    def test_non_real_mode_creates_primitives(self):
        """Non-real mode creates primitives without server URLs."""
        request = ChatRequest(prompt="test", real_mode=False)
        state = MagicMock()
        state.registry = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.LLMPrimitives") as mock_primitives_cls:
            mock_primitives = MagicMock()
            mock_primitives.model_server = MagicMock()
            mock_primitives_cls.return_value = mock_primitives

            result = _init_primitives(request, state)

        assert result is mock_primitives

    def test_non_real_mode_raises_on_no_model_server(self):
        """Non-real mode raises HTTPException when no model server."""
        request = ChatRequest(prompt="test", real_mode=False)
        state = MagicMock()
        state.registry = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.LLMPrimitives") as mock_primitives_cls:
            mock_primitives = MagicMock()
            mock_primitives.model_server = None  # No model server
            mock_primitives_cls.return_value = mock_primitives

            with pytest.raises(HTTPException) as exc_info:
                _init_primitives(request, state)

        assert exc_info.value.status_code == 503
        assert "no model server" in exc_info.value.detail

    def test_cache_prompt_propagated(self):
        """cache_prompt flag propagated to primitives."""
        request = ChatRequest(prompt="test", real_mode=True, cache_prompt=True)
        state = MagicMock()
        state._real_primitives = None
        state.registry = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.LLMPrimitives") as mock_primitives_cls:
            mock_primitives = MagicMock()
            mock_primitives._backends = {"port8080": MagicMock()}
            mock_primitives_cls.return_value = mock_primitives

            _init_primitives(request, state)

        assert mock_primitives.cache_prompt is True


class TestPlanReviewGate:
    """Tests for _plan_review_gate (Stage 5)."""

    def test_plan_review_skipped_when_disabled(self):
        """Plan review skipped when feature disabled."""
        request = ChatRequest(prompt="test", real_mode=True)
        routing = MagicMock()
        primitives = MagicMock()
        state = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.features") as mock_features:
            mock_features.return_value.plan_review = False
            result = _plan_review_gate(request, routing, primitives, state)

        assert result is None

    def test_plan_review_skipped_for_non_real_mode(self):
        """Plan review skipped for non-real mode."""
        request = ChatRequest(prompt="test", real_mode=False)
        routing = MagicMock()
        primitives = MagicMock()
        state = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.features") as mock_features:
            mock_features.return_value.plan_review = True
            result = _plan_review_gate(request, routing, primitives, state)

        assert result is None

    def test_plan_review_skipped_when_not_needed(self):
        """Plan review skipped when _needs_plan_review returns False."""
        request = ChatRequest(prompt="test", real_mode=True)
        routing = MagicMock()
        primitives = MagicMock()
        state = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.features") as mock_features:
            mock_features.return_value.plan_review = True
            with patch("src.api.routes.chat_pipeline.routing._needs_plan_review") as mock_needs:
                mock_needs.return_value = False
                result = _plan_review_gate(request, routing, primitives, state)

        assert result is None

    def test_plan_review_runs_when_needed(self):
        """Plan review runs when all conditions met."""
        request = ChatRequest(prompt="test", real_mode=True)
        routing = MagicMock()
        routing.task_ir = {"task_type": "code"}
        routing.routing_decision = ["coder_escalation"]
        routing.task_id = "test-123"
        primitives = MagicMock()
        state = MagicMock()

        mock_review_result = MagicMock()
        mock_review_result.decision = "ok"

        with patch("src.api.routes.chat_pipeline.routing.features") as mock_features:
            mock_features.return_value.plan_review = True
            with patch("src.api.routes.chat_pipeline.routing._needs_plan_review") as mock_needs:
                mock_needs.return_value = True
                with patch(
                    "src.api.routes.chat_pipeline.routing._architect_plan_review"
                ) as mock_review:
                    mock_review.return_value = mock_review_result
                    with patch("src.api.routes.chat_pipeline.routing._store_plan_review_episode"):
                        result = _plan_review_gate(request, routing, primitives, state)

        assert result is mock_review_result
        mock_review.assert_called_once()

    def test_plan_review_applies_changes(self):
        """Plan review applies routing changes when decision != ok."""
        request = ChatRequest(prompt="test", real_mode=True)
        routing = MagicMock()
        routing.task_ir = {"task_type": "code"}
        routing.routing_decision = ["coder_escalation"]
        routing.task_id = "test-123"
        primitives = MagicMock()
        state = MagicMock()

        mock_review_result = MagicMock()
        mock_review_result.decision = "escalate"

        with patch("src.api.routes.chat_pipeline.routing.features") as mock_features:
            mock_features.return_value.plan_review = True
            with patch("src.api.routes.chat_pipeline.routing._needs_plan_review") as mock_needs:
                mock_needs.return_value = True
                with patch(
                    "src.api.routes.chat_pipeline.routing._architect_plan_review"
                ) as mock_review:
                    mock_review.return_value = mock_review_result
                    with patch(
                        "src.api.routes.chat_pipeline.routing._apply_plan_review"
                    ) as mock_apply:
                        mock_apply.return_value = ["architect_general"]
                        with patch(
                            "src.api.routes.chat_pipeline.routing._store_plan_review_episode"
                        ):
                            _plan_review_gate(request, routing, primitives, state)

        mock_apply.assert_called_once()
        assert routing.routing_decision == ["architect_general"]

    def test_plan_review_stores_episode(self):
        """Plan review stores episode when result exists."""
        request = ChatRequest(prompt="test", real_mode=True)
        routing = MagicMock()
        routing.task_ir = {"task_type": "code"}
        routing.routing_decision = ["coder_escalation"]
        routing.task_id = "test-123"
        primitives = MagicMock()
        state = MagicMock()

        mock_review_result = MagicMock()
        mock_review_result.decision = "ok"

        with patch("src.api.routes.chat_pipeline.routing.features") as mock_features:
            mock_features.return_value.plan_review = True
            with patch("src.api.routes.chat_pipeline.routing._needs_plan_review") as mock_needs:
                mock_needs.return_value = True
                with patch(
                    "src.api.routes.chat_pipeline.routing._architect_plan_review"
                ) as mock_review:
                    mock_review.return_value = mock_review_result
                    with patch(
                        "src.api.routes.chat_pipeline.routing._store_plan_review_episode"
                    ) as mock_store:
                        _plan_review_gate(request, routing, primitives, state)

        mock_store.assert_called_once_with(state, "test-123", routing.task_ir, mock_review_result)

    def test_plan_review_no_store_when_none(self):
        """Plan review doesn't store when result is None."""
        request = ChatRequest(prompt="test", real_mode=True)
        routing = MagicMock()
        routing.task_ir = {"task_type": "code"}
        routing.routing_decision = ["coder_escalation"]
        primitives = MagicMock()
        state = MagicMock()

        with patch("src.api.routes.chat_pipeline.routing.features") as mock_features:
            mock_features.return_value.plan_review = True
            with patch("src.api.routes.chat_pipeline.routing._needs_plan_review") as mock_needs:
                mock_needs.return_value = True
                with patch(
                    "src.api.routes.chat_pipeline.routing._architect_plan_review"
                ) as mock_review:
                    mock_review.return_value = None
                    with patch(
                        "src.api.routes.chat_pipeline.routing._store_plan_review_episode"
                    ) as mock_store:
                        _plan_review_gate(request, routing, primitives, state)

        mock_store.assert_not_called()

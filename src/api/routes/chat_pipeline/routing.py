"""Pipeline stages 1-3 + 5: routing, preprocessing, backend init, plan review."""

from __future__ import annotations

import logging
import uuid

from fastapi import HTTPException

from src.api.models import ChatRequest
from src.api.services.memrl import ensure_memrl_initialized
from src.config import get_config
from src.features import features
from src.llm_primitives import LLMPrimitives
from src.roles import Role

from src.api.routes.chat_review import (
    _apply_plan_review,
    _architect_plan_review,
    _needs_plan_review,
    _store_plan_review_episode,
)
from src.api.routes.chat_routing import _classify_and_route
from src.api.routes.chat_utils import (
    DEFAULT_TIMEOUT_S,
    ROLE_TIMEOUTS,
    RoutingResult,
)
from src.api.structured_logging import task_extra

log = logging.getLogger(__name__)


# ── Stage 1: Routing ────────────────────────────────────────────────────


def _route_request(request: ChatRequest, state) -> RoutingResult:
    """Determine routing decision, strategy, and task metadata.

    Produces a RoutingResult that captures all routing decisions made
    before execution begins. Includes failure graph veto and MemRL logging.
    """
    task_id = f"chat-{uuid.uuid4().hex[:8]}"
    task_ir = {
        "task_type": "chat",
        "objective": request.prompt[:200],
        "priority": "interactive",
    }

    use_mock = request.mock_mode and not request.real_mode

    # Initialize MemRL early for real_mode to enable HybridRouter
    if request.real_mode and not use_mock:
        ensure_memrl_initialized(state)

    # Determine routing using HybridRouter if available, otherwise rules
    if use_mock:
        routing_decision = [request.role or Role.FRONTDOOR]
        routing_strategy = "mock"
    elif request.force_role:
        routing_decision = [request.force_role]
        routing_strategy = "forced"
    elif request.role and request.role not in ("", "frontdoor"):
        routing_decision = [request.role]
        routing_strategy = "explicit"
    elif state.hybrid_router and request.real_mode:
        routing_decision, routing_strategy = state.hybrid_router.route(task_ir)
    else:
        has_image = bool(request.image_path or request.image_base64)
        classified_role, routing_strategy = _classify_and_route(
            request.prompt,
            request.context or "",
            has_image=has_image,
        )
        routing_decision = [classified_role]

    # Failure graph veto — revert high-risk specialists to frontdoor
    if (
        state.failure_graph
        and routing_decision
        and str(routing_decision[0]) != str(Role.FRONTDOOR)
        and routing_strategy not in ("mock", "forced")
    ):
        try:
            risk = state.failure_graph.get_failure_risk(str(routing_decision[0]))
            if risk > 0.5:
                log.warning(
                    "Failure veto: %s risk=%.2f > 0.5, reverting to frontdoor",
                    routing_decision[0],
                    risk,
                    extra=task_extra(
                        task_id=task_id,
                        role=str(routing_decision[0]),
                        stage="routing",
                        strategy="failure_vetoed",
                    ),
                )
                routing_decision = [str(Role.FRONTDOOR)]
                routing_strategy = "failure_vetoed"
        except Exception as exc:
            log.debug("Failure risk veto check failed: %s", exc)

    # Log task start (MemRL integration)
    if state.progress_logger:
        state.progress_logger.log_task_started(
            task_id=task_id,
            task_ir=task_ir,
            routing_decision=routing_decision,
            routing_strategy=routing_strategy,
        )

    # Compute role-specific timeout
    role_str = str(routing_decision[0]) if routing_decision else str(Role.FRONTDOOR)
    timeout_s = ROLE_TIMEOUTS.get(role_str, DEFAULT_TIMEOUT_S)

    return RoutingResult(
        task_id=task_id,
        task_ir=task_ir,
        use_mock=use_mock,
        routing_decision=routing_decision,
        routing_strategy=routing_strategy,
        timeout_s=timeout_s,
    )


# ── Stage 2: Preprocessing ──────────────────────────────────────────────


def _preprocess(request: ChatRequest, state, routing: RoutingResult) -> None:
    """Apply input formalization if enabled. Mutates request.context and routing."""
    if (
        features().input_formalizer
        and request.real_mode
        and not routing.use_mock
        and routing.routing_strategy not in ("mock",)
    ):
        from src.formalizer import should_formalize_input, formalize_prompt, inject_formalization

        should_fml, problem_hint = should_formalize_input(request.prompt)
        if should_fml:
            fml_result = formalize_prompt(request.prompt, problem_hint, state.registry)
            if fml_result.success:
                request.context = inject_formalization(
                    request.prompt, request.context or "", fml_result.ir_json
                )
                routing.formalization_applied = True
                log.info(
                    "Input formalization: %s (%.1fs, %s)",
                    problem_hint,
                    fml_result.elapsed_seconds,
                    fml_result.model_role,
                    extra=task_extra(
                        task_id=routing.task_id,
                        stage="preprocess",
                        latency_ms=fml_result.elapsed_seconds * 1000,
                    ),
                )


# ── Stage 3: Backend initialization ─────────────────────────────────────


def _init_primitives(request: ChatRequest, state) -> LLMPrimitives:
    """Initialize LLM backends for real inference.

    Reuses shared LLMPrimitives instance for connection pooling when possible.
    Raises HTTPException(503) if backends unavailable.
    """
    if request.real_mode:
        server_urls = request.server_urls or get_config().server_urls.as_dict()

        if (
            hasattr(state, "_real_primitives")
            and state._real_primitives is not None
            and not request.server_urls
        ):
            primitives = state._real_primitives
            primitives.reset_counters()
        else:
            try:
                primitives = LLMPrimitives(
                    mock_mode=False,
                    server_urls=server_urls,
                    registry=state.registry,
                    health_tracker=getattr(state, "health_tracker", None),
                    admission_controller=getattr(state, "admission", None),
                )
                if not request.server_urls:
                    state._real_primitives = primitives
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to initialize real mode backends: {e}",
                )

        primitives.cache_prompt = request.cache_prompt

        if not primitives._backends:
            raise HTTPException(
                status_code=503,
                detail="No backends available. Ensure llama-server is running on configured ports.",
            )
    else:
        primitives = LLMPrimitives(mock_mode=False, registry=state.registry)
        if primitives.model_server is None:
            raise HTTPException(
                status_code=503,
                detail="Real inference not available: no model server configured",
            )

    return primitives


# ── Stage 5: Plan review gate ───────────────────────────────────────────


def _plan_review_gate(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
) -> list | None:
    """Run architect plan review if applicable. Returns modified routing_decision or None."""
    plan_review_result = None
    if (
        request.real_mode
        and features().plan_review
        and _needs_plan_review(routing.task_ir, routing.routing_decision, state)
    ):
        plan_review_result = _architect_plan_review(
            routing.task_ir, routing.routing_decision, primitives, state, routing.task_id
        )
        if plan_review_result and plan_review_result.decision != "ok":
            routing.routing_decision = _apply_plan_review(
                routing.routing_decision, plan_review_result
            )
        if plan_review_result:
            _store_plan_review_episode(state, routing.task_id, routing.task_ir, plan_review_result)
    return plan_review_result

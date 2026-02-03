"""Delegate endpoint: accept TaskIR, execute via ProactiveDelegator.

POST /api/delegate
    Body: TaskIR JSON with plan.steps
    Returns: DelegateResponse with wave plan and results

Feature-gated: requires parallel_execution=True
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import dep_app_state
from src.api.state import AppState
from src.features import features
from src.parallel_step_executor import compute_waves

router = APIRouter()


class DelegateRequest(BaseModel):
    """Request body for the delegate endpoint."""

    task_ir: dict[str, Any] = Field(
        ...,
        description="TaskIR JSON with at least 'objective' and 'plan.steps'",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, return wave plan without executing steps",
    )


class DelegateResponse(BaseModel):
    """Response body for the delegate endpoint."""

    task_id: str
    objective: str
    waves: list[dict[str, Any]]
    subtask_results: list[dict[str, Any]]
    aggregated_output: str
    all_approved: bool
    elapsed_seconds: float


@router.post("/api/delegate", response_model=DelegateResponse)
async def delegate(
    request: DelegateRequest,
    state: AppState = Depends(dep_app_state),
) -> DelegateResponse:
    """Execute a TaskIR via wave-based delegation.

    When ``dry_run=True``, returns only the computed wave plan without
    executing any LLM calls.
    """
    if not features().parallel_execution:
        raise HTTPException(
            status_code=403,
            detail="parallel_execution feature is not enabled",
        )

    task_ir = request.task_ir
    plan = task_ir.get("plan", {})
    steps = plan.get("steps", [])

    if not steps:
        raise HTTPException(status_code=422, detail="TaskIR has no plan.steps")

    # Ensure task_id exists
    if "task_id" not in task_ir:
        task_ir["task_id"] = str(uuid.uuid4())

    # Compute waves
    try:
        waves = compute_waves(steps)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    wave_dicts = [{"index": w.index, "step_ids": w.step_ids} for w in waves]

    # Dry run: return wave plan only
    if request.dry_run:
        return DelegateResponse(
            task_id=task_ir["task_id"],
            objective=task_ir.get("objective", ""),
            waves=wave_dicts,
            subtask_results=[],
            aggregated_output="",
            all_approved=False,
            elapsed_seconds=0.0,
        )

    # Full execution: need LLM primitives
    if state.llm_primitives is None:
        raise HTTPException(
            status_code=503,
            detail="LLM primitives not initialized (server not ready)",
        )

    from src.proactive_delegation import ProactiveDelegator

    delegator = ProactiveDelegator(
        registry=state.registry,
        primitives=state.llm_primitives,
        progress_logger=state.progress_logger,
        hybrid_router=state.hybrid_router,
    )

    start = time.monotonic()
    result = await delegator.delegate(task_ir)
    elapsed = time.monotonic() - start

    return DelegateResponse(
        task_id=result.task_id,
        objective=result.objective,
        waves=wave_dicts,
        subtask_results=[
            {
                "subtask_id": sr.subtask_id,
                "role": sr.role,
                "output": sr.output[:2000],  # Truncate for response size
                "success": sr.success,
                "error": sr.error,
            }
            for sr in result.subtask_results
        ],
        aggregated_output=result.aggregated_output,
        all_approved=result.all_approved,
        elapsed_seconds=round(elapsed, 2),
    )

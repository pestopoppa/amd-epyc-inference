"""Gate execution endpoints."""

import time

from fastapi import APIRouter, HTTPException

from src.api.models import GateRequest, GateResultModel, GatesResponse
from src.api.state import get_state

router = APIRouter()


@router.post("/gates", response_model=GatesResponse)
async def run_gates(request: GateRequest) -> GatesResponse:
    """Run quality gates.

    Can run all gates, specific gates by name, or only required gates.
    """
    state = get_state()

    if state.gate_runner is None:
        raise HTTPException(status_code=503, detail="Gate runner not initialized")

    start_time = time.perf_counter()

    if request.gate_names:
        results = state.gate_runner.run_gates_by_name(request.gate_names)
    else:
        results = state.gate_runner.run_all_gates(
            stop_on_first_failure=request.stop_on_first_failure,
            required_only=request.required_only,
        )

    elapsed = time.perf_counter() - start_time

    return GatesResponse(
        results=[
            GateResultModel(
                gate_name=r.gate_name,
                passed=r.passed,
                exit_code=r.exit_code,
                elapsed_seconds=r.elapsed_seconds,
                errors=r.errors,
                warnings=r.warnings,
            )
            for r in results
        ],
        all_passed=all(r.passed for r in results),
        total_elapsed_seconds=elapsed,
    )


@router.get("/gates", response_model=list[str])
async def list_gates() -> list[str]:
    """List available gate names."""
    state = get_state()

    if state.gate_runner is None:
        raise HTTPException(status_code=503, detail="Gate runner not initialized")

    return state.gate_runner.get_gate_names()

"""Gate execution endpoints."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import dep_gate_runner
from src.api.models import GateRequest, GateResultModel, GatesResponse

if TYPE_CHECKING:
    from src.gate_runner import GateRunner

router = APIRouter()


@router.post("/gates", response_model=GatesResponse)
async def run_gates(
    request: GateRequest,
    gate_runner: "GateRunner" = Depends(dep_gate_runner),
) -> GatesResponse:
    """Run quality gates.

    Can run all gates, specific gates by name, or only required gates.
    dep_gate_runner raises 503 if gate runner is not initialized.
    """
    start_time = time.perf_counter()

    if request.gate_names:
        results = gate_runner.run_gates_by_name(request.gate_names)
    else:
        results = await gate_runner.run_all_gates_parallel(
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
async def list_gates(
    gate_runner: "GateRunner" = Depends(dep_gate_runner),
) -> list[str]:
    """List available gate names."""
    return gate_runner.get_gate_names()

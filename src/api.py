#!/usr/bin/env python3
"""FastAPI HTTP interface for the orchestrator.

This module provides an HTTP API for the orchestration system,
supporting both mock mode (for testing) and real inference mode.

Usage:
    # Development server
    uvicorn src.api:app --reload --port 8000

    # With mock mode (default)
    curl -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello", "mock_mode": true}'

    # Health check
    curl http://localhost:8000/health
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.repl_environment import REPLEnvironment, REPLConfig
from src.llm_primitives import LLMPrimitives, LLMPrimitivesConfig
from src.gate_runner import GateRunner
from src.failure_router import FailureRouter


# ============================================================================
# Request/Response Models
# ============================================================================


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    prompt: str = Field(..., description="The user prompt to process")
    context: str = Field(default="", description="Optional context to include")
    mock_mode: bool = Field(default=True, description="Use mock responses instead of real inference")
    max_turns: int = Field(default=10, ge=1, le=50, description="Maximum orchestration turns")
    role: str = Field(default="frontdoor", description="Initial role to use")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    answer: str = Field(..., description="The final answer")
    turns: int = Field(..., description="Number of orchestration turns used")
    tokens_used: int = Field(default=0, description="Approximate tokens used")
    elapsed_seconds: float = Field(..., description="Total processing time")
    mock_mode: bool = Field(..., description="Whether mock mode was used")


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str = Field(..., description="Health status")
    models_loaded: int = Field(default=0, description="Number of models loaded")
    mock_mode_available: bool = Field(default=True, description="Mock mode availability")
    version: str = Field(default="0.1.0", description="API version")


class GateRequest(BaseModel):
    """Request model for running gates."""

    gate_names: list[str] | None = Field(default=None, description="Specific gates to run (None = all)")
    stop_on_first_failure: bool = Field(default=True, description="Stop after first required gate fails")
    required_only: bool = Field(default=False, description="Only run required gates")


class GateResultModel(BaseModel):
    """Model for individual gate result."""

    gate_name: str
    passed: bool
    exit_code: int
    elapsed_seconds: float
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GatesResponse(BaseModel):
    """Response model for gates endpoint."""

    results: list[GateResultModel]
    all_passed: bool
    total_elapsed_seconds: float


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""

    total_requests: int
    total_turns: int
    average_turns_per_request: float
    mock_requests: int
    real_requests: int


# ============================================================================
# Application State
# ============================================================================


@dataclass
class AppState:
    """Application state container."""

    llm_primitives: LLMPrimitives | None = None
    gate_runner: GateRunner | None = None
    failure_router: FailureRouter | None = None

    # Stats tracking
    total_requests: int = 0
    total_turns: int = 0
    mock_requests: int = 0
    real_requests: int = 0

    def increment_request(self, mock_mode: bool, turns: int) -> None:
        """Track a completed request."""
        self.total_requests += 1
        self.total_turns += turns
        if mock_mode:
            self.mock_requests += 1
        else:
            self.real_requests += 1

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        return {
            "total_requests": self.total_requests,
            "total_turns": self.total_turns,
            "average_turns_per_request": (
                self.total_turns / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
            "mock_requests": self.mock_requests,
            "real_requests": self.real_requests,
        }


# Global state
_state = AppState()


# ============================================================================
# Lifespan Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    _state.llm_primitives = LLMPrimitives(mock_mode=True)
    _state.gate_runner = GateRunner()
    _state.failure_router = FailureRouter()

    yield

    # Shutdown
    _state.llm_primitives = None
    _state.gate_runner = None
    _state.failure_router = None


# ============================================================================
# FastAPI Application
# ============================================================================


app = FastAPI(
    title="Orchestrator API",
    description="HTTP interface for the hierarchical orchestration system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        models_loaded=0,  # In mock mode, no models loaded
        mock_mode_available=True,
        version="0.1.0",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat request through the orchestrator.

    In mock mode, returns a simulated response.
    In real mode (when implemented), runs the full orchestration loop.
    """
    start_time = time.perf_counter()

    if request.mock_mode:
        # Mock mode: simulate orchestration
        turns = 1
        answer = f"[MOCK] Processed prompt: {request.prompt[:100]}..."

        if request.context:
            answer += f" (with {len(request.context)} chars of context)"

        elapsed = time.perf_counter() - start_time
        _state.increment_request(mock_mode=True, turns=turns)

        return ChatResponse(
            answer=answer,
            turns=turns,
            tokens_used=0,
            elapsed_seconds=elapsed,
            mock_mode=True,
        )

    # Real mode: run orchestration loop
    # This would use the full REPL environment and LLM primitives
    primitives = LLMPrimitives(mock_mode=False)

    if primitives.model_server is None:
        raise HTTPException(
            status_code=503,
            detail="Real inference not available: no model server configured",
        )

    # Create REPL environment
    combined_context = request.prompt
    if request.context:
        combined_context += f"\n\nContext:\n{request.context}"

    repl = REPLEnvironment(
        context=combined_context,
        llm_primitives=primitives,
    )

    # Run orchestration loop
    turns = 0
    answer = ""

    for turn in range(request.max_turns):
        turns += 1
        # In real mode, we'd get code from the Root LM and execute it
        # For now, this is a placeholder
        answer = "[REAL MODE NOT IMPLEMENTED]"
        break

    elapsed = time.perf_counter() - start_time
    _state.increment_request(mock_mode=False, turns=turns)

    return ChatResponse(
        answer=answer,
        turns=turns,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
    )


@app.post("/gates", response_model=GatesResponse)
async def run_gates(request: GateRequest) -> GatesResponse:
    """Run quality gates.

    Can run all gates, specific gates by name, or only required gates.
    """
    if _state.gate_runner is None:
        raise HTTPException(status_code=503, detail="Gate runner not initialized")

    start_time = time.perf_counter()

    if request.gate_names:
        results = _state.gate_runner.run_gates_by_name(request.gate_names)
    else:
        results = _state.gate_runner.run_all_gates(
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


@app.get("/gates", response_model=list[str])
async def list_gates() -> list[str]:
    """List available gate names."""
    if _state.gate_runner is None:
        raise HTTPException(status_code=503, detail="Gate runner not initialized")

    return _state.gate_runner.get_gate_names()


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get API usage statistics."""
    stats = _state.get_stats()
    return StatsResponse(**stats)


@app.post("/stats/reset")
async def reset_stats() -> dict[str, str]:
    """Reset API usage statistics."""
    _state.total_requests = 0
    _state.total_turns = 0
    _state.mock_requests = 0
    _state.real_requests = 0
    return {"status": "reset"}


# ============================================================================
# Development Server
# ============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

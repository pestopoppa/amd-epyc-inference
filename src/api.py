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
    real_mode: bool = Field(default=False, description="Enable real inference with RadixAttention caching")
    max_turns: int = Field(default=10, ge=1, le=50, description="Maximum orchestration turns")
    role: str = Field(default="frontdoor", description="Initial role to use")
    server_urls: dict[str, str] | None = Field(
        default=None,
        description="Server URLs for real mode (e.g., {'frontdoor': 'http://localhost:8080'})"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    answer: str = Field(..., description="The final answer")
    turns: int = Field(..., description="Number of orchestration turns used")
    tokens_used: int = Field(default=0, description="Approximate tokens used")
    elapsed_seconds: float = Field(..., description="Total processing time")
    mock_mode: bool = Field(..., description="Whether mock mode was used")
    real_mode: bool = Field(default=False, description="Whether real inference was used")
    cache_stats: dict[str, Any] | None = Field(
        default=None,
        description="Cache performance statistics (real_mode only)"
    )


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
# Root LM Helper Functions
# ============================================================================


def _build_root_lm_prompt(
    state: str,
    original_prompt: str,
    last_output: str,
    last_error: str,
    turn: int,
) -> str:
    """Build the prompt for the Root LM (frontdoor).

    The Root LM generates Python code that executes in a sandboxed REPL.
    It has access to the context and can call sub-LMs for complex tasks.

    Args:
        state: Current REPL state from repl.get_state()
        original_prompt: The user's original prompt
        last_output: Output from the last code execution
        last_error: Error from the last code execution (if any)
        turn: Current turn number (0-indexed)

    Returns:
        Prompt string for the Root LM
    """
    prompt_parts = [
        "You are an orchestrator that generates Python code to solve tasks.",
        "",
        "## Available Tools",
        "- `context`: str - The full input context (large, do not send to LLM)",
        "- `artifacts`: dict - Store intermediate results",
        "- `peek(n)`: Return first n characters of context",
        "- `grep(pattern)`: Search context with regex, return matching lines",
        "- `llm_call(prompt, role='worker')`: Call a sub-LM for a task",
        "- `llm_batch(prompts, role='worker')`: Call sub-LM with multiple prompts in parallel",
        "- `FINAL(answer)`: Signal completion with the final answer",
        "",
        "## Rules",
        "1. NEVER send the full context to llm_call - use peek() or grep() to extract relevant parts",
        "2. Break complex tasks into smaller sub-tasks using llm_call/llm_batch",
        "3. Store intermediate results in artifacts dict",
        "4. Call FINAL(answer) when you have the complete answer",
        "5. Output only valid Python code - no explanations or markdown",
        "",
        f"## Current State (Turn {turn + 1})",
        state,
    ]

    if last_error:
        prompt_parts.extend([
            "",
            "## Last Error",
            f"```",
            last_error,
            "```",
            "Fix the error and try again.",
        ])
    elif last_output:
        prompt_parts.extend([
            "",
            "## Last Output",
            f"```",
            last_output[:500] + ("..." if len(last_output) > 500 else ""),
            "```",
        ])

    prompt_parts.extend([
        "",
        "## Task",
        original_prompt,
        "",
        "## Your Code",
        "Write Python code to complete the task. Output only the code:",
    ])

    return "\n".join(prompt_parts)


def _extract_code_from_response(response: str) -> str:
    """Extract Python code from an LLM response.

    Handles responses that may be wrapped in markdown code blocks
    or contain explanatory text.

    Args:
        response: Raw LLM response

    Returns:
        Extracted Python code
    """
    import re

    # Try to extract from markdown code block
    # Match ```python ... ``` or ``` ... ```
    code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    if matches:
        # Return the first code block
        return matches[0].strip()

    # If no code block, try to find code-like content
    # Look for lines that start with common Python patterns
    lines = response.split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        # Skip explanation-like lines
        if line.strip().startswith("#") and not in_code:
            # Could be a comment, include it
            pass
        elif any(line.strip().startswith(kw) for kw in [
            "import ", "from ", "def ", "class ", "if ", "for ", "while ",
            "try:", "except", "with ", "return ", "print(", "FINAL(",
            "artifacts[", "result =", "answer =", "output =",
        ]):
            in_code = True
        elif in_code and line.strip() == "":
            pass  # Keep empty lines in code
        elif not in_code and not any(c.isalnum() or c in "()[]{}=+-*/:,._'\"" for c in line):
            continue  # Skip non-code lines

        if in_code or line.strip().startswith("#") or "=" in line or "()" in line:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)

    # Fallback: return the whole response (let REPL handle errors)
    return response.strip()


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

    Modes:
    - mock_mode=True (default): Returns simulated response, no real inference
    - real_mode=True: Uses RadixAttention caching with live llama-server instances
    - Neither: Uses legacy model server (if configured)

    The real_mode flag enables:
    - CachingBackend with prefix routing
    - Cache statistics in response
    - Full orchestration loop with Root LM (Phase 8)
    """
    start_time = time.perf_counter()

    # Determine mode (real_mode takes precedence over mock_mode)
    use_mock = request.mock_mode and not request.real_mode

    if use_mock:
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
            real_mode=False,
            cache_stats=None,
        )

    # Real mode: use RadixAttention with CachingBackend
    if request.real_mode:
        # Get server URLs from request or use defaults
        server_urls = request.server_urls or LLMPrimitives.DEFAULT_SERVER_URLS

        try:
            primitives = LLMPrimitives(
                mock_mode=False,
                server_urls=server_urls,
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize real mode backends: {e}",
            )

        # Verify at least one backend is available
        if not primitives._backends:
            raise HTTPException(
                status_code=503,
                detail="No backends available. Ensure llama-server is running on configured ports.",
            )

    else:
        # Legacy mode: use ModelServer (if configured)
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

    # Run Root LM orchestration loop
    turns = 0
    answer = ""
    last_output = ""
    last_error = ""

    for turn in range(request.max_turns):
        turns += 1

        # 1. Get current REPL state
        state = repl.get_state()

        # 2. Build prompt for Root LM (frontdoor)
        root_prompt = _build_root_lm_prompt(
            state=state,
            original_prompt=request.prompt,
            last_output=last_output,
            last_error=last_error,
            turn=turn,
        )

        # 3. Call Root LM to generate Python code
        try:
            code = primitives.llm_call(
                root_prompt,
                role="frontdoor",
                n_tokens=1024,
            )
        except Exception as e:
            # If frontdoor call fails, return error
            answer = f"[ERROR: Root LM call failed: {e}]"
            break

        # Extract code from response (handle markdown code blocks)
        code = _extract_code_from_response(code)

        # 4. Execute code in REPL
        result = repl.execute(code)

        # 5. Check for FINAL() completion
        if result.is_final:
            answer = result.final_answer or ""
            break

        # 6. Handle errors - feed back to Root LM for recovery
        if result.error:
            last_error = result.error
            last_output = result.output
        else:
            last_error = ""
            last_output = result.output

    # If max turns reached without FINAL()
    if not answer:
        answer = f"[Max turns ({request.max_turns}) reached without FINAL()]"
        if last_output:
            answer += f"\n\nLast output:\n{last_output}"

    elapsed = time.perf_counter() - start_time
    _state.increment_request(mock_mode=False, turns=turns)

    # Get cache stats if using real_mode with RadixAttention
    cache_stats = None
    if request.real_mode and primitives._backends:
        cache_stats = primitives.get_cache_stats()

    return ChatResponse(
        answer=answer,
        turns=turns,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=request.real_mode,
        cache_stats=cache_stats,
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

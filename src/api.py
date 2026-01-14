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

import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Core imports (always available)
from src.repl_environment import REPLEnvironment, REPLConfig
from src.llm_primitives import LLMPrimitives, LLMPrimitivesConfig
from src.gate_runner import GateRunner
from src.failure_router import FailureRouter, FailureContext, ErrorCategory, RoutingDecision
from src.features import Features, get_features, features

# Optional imports - these are lazy-loaded based on feature flags
# MemRL components (Phase 4)
ProgressLogger: type | None = None
ProgressReader: type | None = None
EpisodicStore: type | None = None
TaskEmbedder: type | None = None
QScorer: type | None = None
ScoringConfig: type | None = None
TwoPhaseRetriever: type | None = None
HybridRouter: type | None = None
RuleBasedRouter: type | None = None
RetrievalConfig: type | None = None

# Tool/Script registries
ToolRegistry: type | None = None
ScriptRegistry: type | None = None


def _load_optional_imports() -> None:
    """Load optional module imports based on feature flags.

    This is called during app startup to load only the modules needed
    for the enabled features, improving startup time and reducing
    memory usage when features are disabled.
    """
    global ProgressLogger, ProgressReader, EpisodicStore, TaskEmbedder
    global QScorer, ScoringConfig, TwoPhaseRetriever, HybridRouter
    global RuleBasedRouter, RetrievalConfig, ToolRegistry, ScriptRegistry

    f = features()

    # MemRL components
    if f.memrl:
        from orchestration.repl_memory.progress_logger import (
            ProgressLogger as _PL,
            ProgressReader as _PR,
        )
        from orchestration.repl_memory.episodic_store import EpisodicStore as _ES
        from orchestration.repl_memory.embedder import TaskEmbedder as _TE
        from orchestration.repl_memory.q_scorer import QScorer as _QS, ScoringConfig as _SC
        from orchestration.repl_memory.retriever import (
            TwoPhaseRetriever as _TPR,
            HybridRouter as _HR,
            RuleBasedRouter as _RBR,
            RetrievalConfig as _RC,
        )
        ProgressLogger = _PL
        ProgressReader = _PR
        EpisodicStore = _ES
        TaskEmbedder = _TE
        QScorer = _QS
        ScoringConfig = _SC
        TwoPhaseRetriever = _TPR
        HybridRouter = _HR
        RuleBasedRouter = _RBR
        RetrievalConfig = _RC
    else:
        # Minimal progress logger that doesn't require MemRL
        from orchestration.repl_memory.progress_logger import (
            ProgressLogger as _PL,
            ProgressReader as _PR,
        )
        ProgressLogger = _PL
        ProgressReader = _PR

    # Tool registry
    if f.tools:
        from src.tool_registry import ToolRegistry as _TR
        ToolRegistry = _TR

    # Script registry (requires tools)
    if f.scripts:
        from src.script_registry import ScriptRegistry as _SR
        ScriptRegistry = _SR


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
    # Extended thinking support (Claude Code parity)
    thinking_budget: int = Field(
        default=0,
        ge=0,
        le=32000,
        description="Token budget for internal reasoning (0=disabled, max=32000)"
    )
    permission_mode: str = Field(
        default="normal",
        description="Permission mode: 'normal', 'auto-accept', or 'plan'"
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
# OpenAI-Compatible Models
# ============================================================================


class OpenAIMessage(BaseModel):
    """OpenAI message format."""

    role: str = Field(..., description="Role: system, user, assistant")
    content: str = Field(..., description="Message content")


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="orchestrator", description="Model/role to use")
    messages: list[OpenAIMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    stream: bool = Field(default=False, description="Enable streaming")
    # Extension fields
    x_orchestrator_role: str | None = Field(default=None, description="Force specific role")
    x_show_routing: bool = Field(default=False, description="Include routing metadata")


class OpenAIChoice(BaseModel):
    """OpenAI choice object."""

    index: int = 0
    message: OpenAIMessage | None = None
    delta: dict[str, str] | None = None
    finish_reason: str | None = None


class OpenAIUsage(BaseModel):
    """OpenAI usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "orchestrator"
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None
    # Extension fields
    x_orchestrator_metadata: dict[str, Any] | None = None


class OpenAIModelInfo(BaseModel):
    """OpenAI model info."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "orchestrator"


class OpenAIModelsResponse(BaseModel):
    """OpenAI models list response."""

    object: str = "list"
    data: list[OpenAIModelInfo]


# ============================================================================
# Application State
# ============================================================================


@dataclass
class AppState:
    """Application state container.

    Thread-safe statistics tracking using locks for concurrent request handling.
    """

    llm_primitives: LLMPrimitives | None = None
    gate_runner: GateRunner | None = None
    failure_router: FailureRouter | None = None
    progress_logger: ProgressLogger | None = None

    # Q-scorer components (for idle-time scoring)
    q_scorer: QScorer | None = None
    episodic_store: EpisodicStore | None = None

    # Hybrid routing (learned + rule-based)
    hybrid_router: HybridRouter | None = None

    # Tool and script registries for REPL
    tool_registry: ToolRegistry | None = None
    script_registry: ScriptRegistry | None = None

    # Registry loader (for role defaults)
    registry: Any = None  # RegistryLoader, typed as Any to avoid import cycle

    # Stats tracking (protected by _stats_lock)
    total_requests: int = 0
    total_turns: int = 0
    mock_requests: int = 0
    real_requests: int = 0

    # Idle scoring control (protected by _stats_lock)
    active_requests: int = 0
    q_scorer_enabled: bool = True
    _q_scorer_task: Any = None  # asyncio.Task

    # Lazy initialization flag for MemRL components
    # These are only loaded when real inference or Q-scoring is needed
    _memrl_initialized: bool = False

    # Thread safety lock for statistics
    _stats_lock: threading.Lock = field(default_factory=threading.Lock)

    def increment_request(self, mock_mode: bool, turns: int) -> None:
        """Track a completed request (thread-safe)."""
        with self._stats_lock:
            self.total_requests += 1
            self.total_turns += turns
            if mock_mode:
                self.mock_requests += 1
            else:
                self.real_requests += 1

    def increment_active(self) -> None:
        """Increment active request counter (thread-safe)."""
        with self._stats_lock:
            self.active_requests += 1

    def decrement_active(self) -> None:
        """Decrement active request counter (thread-safe)."""
        with self._stats_lock:
            self.active_requests -= 1

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics (thread-safe snapshot)."""
        with self._stats_lock:
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
                "active_requests": self.active_requests,
            }


# Global state
_state = AppState()


# ============================================================================
# Lifespan Management
# ============================================================================


def _score_completed_task(task_id: str) -> None:
    """Score a completed task immediately (no inference required).

    Called after log_task_completed() to update Q-values in real-time.
    This is lightweight - just DB reads/writes, no LLM inference.
    """
    if not _state.q_scorer or not _state.q_scorer_enabled:
        return

    try:
        # Score just this task
        _state.q_scorer._score_task(task_id)
        # Flush logger to persist Q-value updates
        if _state.progress_logger:
            _state.progress_logger.flush()
    except Exception as e:
        # Log but don't fail the user request - scoring is non-critical
        logger.warning(f"Q-scoring failed for task {task_id}: {e}", exc_info=True)


async def _background_cleanup():
    """Background task for opportunistic cleanup when idle.

    Processes backlog of unscored tasks when server has no active requests.
    This catches any tasks that weren't scored immediately (e.g., from restarts).
    """
    import asyncio

    while True:
        try:
            # Check every 10 seconds
            await asyncio.sleep(10)

            # Only run when idle and Q-scorer is available
            # Note: Don't call _ensure_memrl_initialized() here - only init on real use
            if _state.active_requests == 0 and _state.q_scorer and _state.q_scorer_enabled:
                # Score a small batch of pending tasks
                results = _state.q_scorer.score_pending_tasks()

                if results and not results.get("skipped"):
                    tasks_processed = results.get("tasks_processed", 0)
                    if tasks_processed > 0 and _state.progress_logger:
                        _state.progress_logger.flush()

        except asyncio.CancelledError:
            break
        except Exception as e:
            # Log background task errors but continue running
            logger.warning(f"Background cleanup error: {e}", exc_info=True)


def _ensure_memrl_initialized() -> bool:
    """Lazy-load MemRL components on first real use.

    This function respects the memrl feature flag - if disabled, returns False
    immediately without attempting to load any MemRL components.

    MemRL components are only needed for:
    - Real inference with Q-scoring/HybridRouter
    - Background Q-value updates

    Returns:
        True if MemRL is initialized and available, False otherwise.
    """
    # Check feature flag first
    if not features().memrl:
        return False

    if _state._memrl_initialized:
        return _state.q_scorer is not None

    # Mark as initialized (even if it fails) to prevent repeated attempts
    _state._memrl_initialized = True

    # Ensure optional imports are loaded
    if EpisodicStore is None or TaskEmbedder is None or QScorer is None:
        logger.warning("MemRL feature enabled but imports not available")
        return False

    try:
        _state.episodic_store = EpisodicStore()
        embedder = TaskEmbedder()
        reader = ProgressReader()
        config = ScoringConfig(
            use_claude_judge=False,  # Basic mode only - no LLM required
            min_score_interval_seconds=30,  # Background cleanup interval
            batch_size=10,  # Process 10 tasks per cleanup round
        )
        _state.q_scorer = QScorer(
            store=_state.episodic_store,
            embedder=embedder,
            logger=_state.progress_logger,
            reader=reader,
            config=config,
        )

        # Initialize HybridRouter for learned routing
        retrieval_config = RetrievalConfig(
            semantic_k=20,
            min_similarity=0.3,
            q_weight=0.7,
            confidence_threshold=0.6,
        )
        retriever = TwoPhaseRetriever(
            store=_state.episodic_store,
            embedder=embedder,
            config=retrieval_config,
        )
        # Load routing hints from registry for rule-based fallback
        # Reuse existing registry if loaded at startup, otherwise load it
        if _state.registry is None:
            from src.registry_loader import RegistryLoader
            _state.registry = RegistryLoader(validate_paths=False)
        rule_router = RuleBasedRouter(routing_hints=_state.registry.routing_hints)
        _state.hybrid_router = HybridRouter(retriever=retriever, rule_based_router=rule_router)

        return True
    except Exception as e:
        # MemRL initialization failed - log and continue without it
        logger.warning(f"MemRL initialization failed, continuing without it: {e}", exc_info=True)
        _state.q_scorer = None
        _state.episodic_store = None
        _state.hybrid_router = None
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan.

    Initialization order:
    1. Load feature flags from environment
    2. Load optional module imports based on features
    3. Load registry (YAML parsing only)
    4. Initialize core components (LLM primitives, gate runner, failure router)
    5. Initialize optional components based on features
    6. Start background tasks if MemRL enabled
    """
    import asyncio

    # Load feature flags and optional imports
    f = features()
    _load_optional_imports()

    logger.info(f"Starting orchestrator with features: {f.enabled_features()}")

    # Load registry for role-based generation defaults (YAML parsing only, no models)
    try:
        from src.registry_loader import RegistryLoader
        _state.registry = RegistryLoader(validate_paths=False)
    except Exception as e:
        logger.info(f"Registry file not found or invalid, using defaults: {e}")
        _state.registry = None

    # Core components (always initialized)
    _state.llm_primitives = LLMPrimitives(mock_mode=f.mock_mode, registry=_state.registry)
    _state.progress_logger = ProgressLogger() if ProgressLogger else None
    _state.gate_runner = GateRunner(progress_logger=_state.progress_logger)
    _state.failure_router = FailureRouter()

    # Tool registry (feature-gated)
    if f.tools and ToolRegistry:
        try:
            _state.tool_registry = ToolRegistry()
            _state.tool_registry.load_permissions_from_registry(
                "/mnt/raid0/llm/claude/orchestration/model_registry.yaml"
            )
        except Exception as e:
            logger.info(f"Tool registry not available: {e}")
            _state.tool_registry = None
    else:
        _state.tool_registry = None

    # Script registry (feature-gated, requires tools)
    if f.scripts and ScriptRegistry and _state.tool_registry:
        try:
            _state.script_registry = ScriptRegistry()
            _state.script_registry.load_from_directory(
                "/mnt/raid0/llm/claude/orchestration/script_registry"
            )
        except Exception as e:
            logger.info(f"Script registry not available: {e}")
            _state.script_registry = None
    else:
        _state.script_registry = None

    # Background Q-scoring task (only if MemRL feature enabled)
    if f.memrl:
        _state._q_scorer_task = asyncio.create_task(_background_cleanup())
    else:
        _state._q_scorer_task = None

    yield

    # Shutdown - cancel background task first
    if _state._q_scorer_task:
        _state._q_scorer_task.cancel()
        try:
            await _state._q_scorer_task
        except asyncio.CancelledError:
            pass

    # Flush progress logger before cleanup
    if _state.progress_logger:
        _state.progress_logger.flush()

    _state.llm_primitives = None
    _state.gate_runner = None
    _state.failure_router = None
    _state.progress_logger = None
    _state.q_scorer = None
    _state.episodic_store = None
    _state.hybrid_router = None
    _state.tool_registry = None
    _state.script_registry = None
    _state.registry = None


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
        "4. ALWAYS call FINAL(answer) to return your answer - this is REQUIRED",
        "5. For code generation: FINAL('''def example(): pass''')",
        "6. For questions: FINAL('answer text')",
        "7. Output only valid Python code - no markdown or explanations",
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
        "Write Python code that ends with FINAL(...) to return the answer:",
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


def _auto_wrap_final(code: str) -> str:
    """Auto-wrap code in FINAL() if it looks like a final answer.

    The model often outputs clean code without FINAL(). This detects when
    the code is a complete answer (no exploration functions) and wraps it.

    Args:
        code: Extracted Python code

    Returns:
        Code wrapped in FINAL() if appropriate, otherwise unchanged
    """
    # Already has FINAL() - return as-is
    if "FINAL(" in code:
        return code

    # Check for exploration function calls - don't wrap if exploring
    exploration_patterns = ["peek(", "grep(", "llm_call(", "llm_batch(", "artifacts["]
    for pattern in exploration_patterns:
        if pattern in code:
            return code

    # Check if it's primarily a function/class definition (code generation task)
    lines = [l.strip() for l in code.split("\n") if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return code

    # If starts with def/class and looks like complete code, wrap it
    if lines[0].startswith(("def ", "class ")):
        # Escape triple quotes in the code for FINAL()
        escaped_code = code.replace("'''", r"\'\'\'")
        return f"FINAL('''{escaped_code}''')"

    # If it's a simple expression/value, wrap it
    if len(lines) == 1 and not any(kw in lines[0] for kw in ["import ", "from ", "for ", "while ", "if "]):
        return f"FINAL({code.strip()})"

    return code


def _classify_error(error_message: str, gate_name: str = "") -> ErrorCategory:
    """Classify an error message into an ErrorCategory.

    Args:
        error_message: The error message to classify.
        gate_name: Optional gate name if error came from a gate.

    Returns:
        ErrorCategory for the error.
    """
    error_lower = error_message.lower()

    # Schema/format errors (from gates or parsing)
    if gate_name in ("schema", "format", "lint", "mdformat", "shfmt"):
        return ErrorCategory.FORMAT
    if "schema" in error_lower or "validation" in error_lower:
        return ErrorCategory.SCHEMA
    if "format" in error_lower or "style" in error_lower:
        return ErrorCategory.FORMAT

    # Code errors (syntax, type, import)
    if any(kw in error_lower for kw in [
        "syntaxerror", "indentationerror", "typeerror", "nameerror",
        "importerror", "modulenotfound", "attributeerror"
    ]):
        return ErrorCategory.CODE

    # Logic errors (test failures, assertions)
    if any(kw in error_lower for kw in [
        "assertionerror", "test failed", "expected", "actual"
    ]):
        return ErrorCategory.LOGIC

    # Timeout errors
    if "timeout" in error_lower or "timed out" in error_lower:
        return ErrorCategory.TIMEOUT

    # Early abort (from generation monitor)
    if "early abort" in error_lower or "high entropy" in error_lower:
        return ErrorCategory.EARLY_ABORT

    return ErrorCategory.UNKNOWN


def _build_escalation_prompt(
    original_prompt: str,
    state: str,
    failure_context: FailureContext,
    decision: RoutingDecision,
) -> str:
    """Build a prompt for an escalated role.

    Includes failure context to help the higher-tier model understand
    what went wrong and what was tried.

    Args:
        original_prompt: The user's original prompt.
        state: Current REPL state.
        failure_context: Context about the failure.
        decision: The routing decision with escalation reason.

    Returns:
        Prompt for the escalated role.
    """
    prompt_parts = [
        f"# Escalation from {failure_context.role}",
        "",
        f"The {failure_context.role} failed after {failure_context.failure_count} attempts.",
        f"Reason: {decision.reason}",
        "",
        "## Error Details",
        f"Category: {failure_context.error_category}",
    ]

    if failure_context.gate_name:
        prompt_parts.append(f"Gate: {failure_context.gate_name}")

    if failure_context.error_message:
        prompt_parts.extend([
            "",
            "Error message:",
            "```",
            failure_context.error_message[:500],
            "```",
        ])

    prompt_parts.extend([
        "",
        "## Current State",
        state,
        "",
        "## Original Task",
        original_prompt,
        "",
        "## Instructions",
        "Fix the issue and complete the task. You have more capability than the previous role.",
        "Output Python code that will execute in the REPL environment.",
    ])

    return "\n".join(prompt_parts)


# Escalation role mapping (from lower to higher tier)
ESCALATION_ROLES = {
    "worker": "coder",
    "coder": "architect_general",
    "frontdoor": "coder",
    "ingest": "architect_general",
}


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
    # Track active requests for idle-time Q-scoring (thread-safe)
    _state.increment_active()
    try:
        return await _handle_chat(request)
    finally:
        _state.decrement_active()


async def _handle_chat(request: ChatRequest) -> ChatResponse:
    """Internal handler for chat requests."""
    start_time = time.perf_counter()

    # Generate task ID for MemRL tracking
    task_id = f"chat-{uuid.uuid4().hex[:8]}"

    # Construct task_ir for logging
    task_ir = {
        "task_type": "chat",
        "objective": request.prompt[:200],
        "priority": "interactive",
    }

    # Determine mode (real_mode takes precedence over mock_mode)
    use_mock = request.mock_mode and not request.real_mode

    # Determine routing for logging
    routing_decision = [request.role]
    routing_strategy = "mock" if use_mock else "rules"

    # Log task start (MemRL integration)
    if _state.progress_logger:
        _state.progress_logger.log_task_started(
            task_id=task_id,
            task_ir=task_ir,
            routing_decision=routing_decision,
            routing_strategy=routing_strategy,
        )

    if use_mock:
        # Mock mode: simulate orchestration
        turns = 1
        answer = f"[MOCK] Processed prompt: {request.prompt[:100]}..."

        if request.context:
            answer += f" (with {len(request.context)} chars of context)"

        elapsed = time.perf_counter() - start_time
        _state.increment_request(mock_mode=True, turns=turns)

        # Log task completion (MemRL integration)
        if _state.progress_logger:
            _state.progress_logger.log_task_completed(
                task_id=task_id,
                success=True,
                details=f"Mock response in {elapsed:.3f}s",
            )
            _score_completed_task(task_id)

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
        # Initialize MemRL components on first real use (lazy loading)
        # This loads TaskEmbedder (0.5B model) for Q-scoring and HybridRouter
        _ensure_memrl_initialized()

        # Get server URLs from request or use defaults
        server_urls = request.server_urls or LLMPrimitives.DEFAULT_SERVER_URLS

        try:
            primitives = LLMPrimitives(
                mock_mode=False,
                server_urls=server_urls,
                registry=_state.registry,
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
        primitives = LLMPrimitives(mock_mode=False, registry=_state.registry)

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
        tool_registry=_state.tool_registry,
        script_registry=_state.script_registry,
        role=request.role or "frontdoor",
    )

    # Run Root LM orchestration loop with escalation support
    turns = 0
    answer = ""
    last_output = ""
    last_error = ""

    # Escalation tracking
    current_role = request.role or "frontdoor"
    consecutive_failures = 0
    role_history = [current_role]
    escalation_prompt = ""  # Set when escalating

    for turn in range(request.max_turns):
        turns += 1

        # 1. Get current REPL state
        state = repl.get_state()

        # 2. Build prompt - use escalation prompt if we just escalated
        if escalation_prompt:
            root_prompt = escalation_prompt
            escalation_prompt = ""  # Clear after use
        else:
            root_prompt = _build_root_lm_prompt(
                state=state,
                original_prompt=request.prompt,
                last_output=last_output,
                last_error=last_error,
                turn=turn,
            )

        # 3. Call Root LM (current role) to generate Python code
        try:
            code = primitives.llm_call(
                root_prompt,
                role=current_role,
                n_tokens=1024,
            )
        except Exception as e:
            # If LLM call fails, return error
            answer = f"[ERROR: {current_role} LM call failed: {e}]"
            break

        # Extract code from response (handle markdown code blocks)
        code = _extract_code_from_response(code)

        # Auto-wrap in FINAL() if it looks like a complete answer
        code = _auto_wrap_final(code)

        # 4. Execute code in REPL
        result = repl.execute(code)

        # 5. Check for FINAL() completion
        if result.is_final:
            answer = result.final_answer or ""
            consecutive_failures = 0  # Success resets failure count
            break

        # 6. Handle errors with FailureRouter for escalation decisions
        if result.error:
            consecutive_failures += 1
            last_error = result.error
            last_output = result.output

            # Consult FailureRouter for escalation decision
            if _state.failure_router:
                error_category = _classify_error(result.error)
                failure_ctx = FailureContext(
                    role=current_role,
                    error_message=result.error,
                    error_category=error_category,
                    failure_count=consecutive_failures,
                    task_type="code",
                    code_generated=code[:500] if code else None,
                )
                decision = _state.failure_router.route_failure(failure_ctx)

                # Log escalation decision (for escalate actions)
                if decision.action == "escalate" and _state.progress_logger:
                    _state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=current_role,
                        to_tier=decision.next_role or current_role,
                        reason=f"{decision.reason} (failures: {consecutive_failures})",
                    )

                # Act on routing decision
                if decision.action == "escalate" and decision.next_role:
                    # Switch to higher-tier role
                    current_role = decision.next_role
                    role_history.append(current_role)
                    consecutive_failures = 0  # Reset for new role
                    # Build escalation prompt with failure context
                    escalation_prompt = _build_escalation_prompt(
                        original_prompt=request.prompt,
                        state=state,
                        failure_context=failure_ctx,
                        decision=decision,
                    )
                elif decision.action == "fail":
                    # Max retries/escalations reached - stop with error
                    answer = f"[FAILED: {decision.reason}]"
                    break
        else:
            # Success - reset failure count but keep role
            consecutive_failures = 0
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

    # Log task completion (MemRL integration)
    success = not answer.startswith("[ERROR") and not answer.startswith("[Max turns")
    if _state.progress_logger:
        # Include role history if escalation occurred
        role_info = f", roles: {' -> '.join(role_history)}" if len(role_history) > 1 else ""
        _state.progress_logger.log_task_completed(
            task_id=task_id,
            success=success,
            details=f"Real inference: {turns} turns, {elapsed:.3f}s{role_info}",
        )
        _score_completed_task(task_id)

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
# SSE Streaming Endpoint
# ============================================================================


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE streaming endpoint with routing metadata.

    Streams events in NDJSON format:
    - turn_start: {type: "turn_start", turn: N, role: "..."}
    - thinking: {type: "thinking", content: "..."} (when thinking_budget > 0)
    - token: {type: "token", content: "..."}
    - tool: {type: "tool", name: "...", args: {...}, result: ...}
    - permission_request: {type: "permission_request", id: "...", tool: "...", args: {...}}
    - file: {type: "file", path: "...", content: "...", action: "create"|"modify"}
    - turn_end: {type: "turn_end", tokens: N, elapsed_ms: N}
    - error: {type: "error", message: "..."}
    - [DONE] when complete

    Parameters:
    - thinking_budget: Token budget for internal reasoning (0=disabled)
    - permission_mode: "normal", "auto-accept", or "plan"
    """
    # Generate task ID for MemRL tracking (outside generator for closure)
    task_id = f"stream-{uuid.uuid4().hex[:8]}"

    async def generate() -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()
        use_mock = request.mock_mode and not request.real_mode

        # Construct task_ir and log start (MemRL integration)
        task_ir = {
            "task_type": "chat_stream",
            "objective": request.prompt[:200],
            "priority": "interactive",
        }
        if _state.progress_logger:
            _state.progress_logger.log_task_started(
                task_id=task_id,
                task_ir=task_ir,
                routing_decision=[request.role],
                routing_strategy="mock" if use_mock else "rules",
            )

        # Mock mode
        if use_mock:
            # Emit turn start
            yield f"data: {json.dumps({'type': 'turn_start', 'turn': 1, 'role': 'frontdoor'})}\n\n"

            # Emit thinking events if thinking_budget > 0 (Claude Code parity)
            if request.thinking_budget > 0:
                thinking_steps = [
                    "Analyzing the user's request...",
                    f"Request type: {request.prompt[:30].split()[0] if request.prompt else 'unknown'}",
                    "Determining appropriate response strategy...",
                    "Preparing response...",
                ]
                for step in thinking_steps:
                    yield f"data: {json.dumps({'type': 'thinking', 'content': step})}\n\n"

            # Check permission mode - in plan mode, only emit analysis
            if request.permission_mode == "plan":
                analysis = f"[PLAN MODE] Would process: {request.prompt[:100]}..."
                yield f"data: {json.dumps({'type': 'token', 'content': analysis})}\n\n"
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                yield f"data: {json.dumps({'type': 'turn_end', 'tokens': len(analysis), 'elapsed_ms': elapsed_ms})}\n\n"
                # Log completion (MemRL)
                if _state.progress_logger:
                    _state.progress_logger.log_task_completed(task_id, success=True, details="Plan mode")
                    _score_completed_task(task_id)
                yield "data: [DONE]\n\n"
                return

            # Simulate streaming tokens
            mock_response = f"[MOCK] Processed: {request.prompt[:50]}..."
            for char in mock_response:
                yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"

            # Emit turn end
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            yield f"data: {json.dumps({'type': 'turn_end', 'tokens': len(mock_response), 'elapsed_ms': elapsed_ms})}\n\n"

            # Log completion (MemRL)
            if _state.progress_logger:
                _state.progress_logger.log_task_completed(task_id, success=True, details="Mock stream")
                _score_completed_task(task_id)
            yield "data: [DONE]\n\n"
            return

        # Real mode - initialize MemRL components on first real use (lazy loading)
        _ensure_memrl_initialized()

        server_urls = request.server_urls or LLMPrimitives.DEFAULT_SERVER_URLS
        try:
            primitives = LLMPrimitives(mock_mode=False, server_urls=server_urls, registry=_state.registry)
        except Exception as e:
            # Log failure (MemRL)
            if _state.progress_logger:
                _state.progress_logger.log_task_completed(task_id, success=False, details=str(e))
                _score_completed_task(task_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Create REPL
        combined_context = request.prompt
        if request.context:
            combined_context += f"\n\nContext:\n{request.context}"

        repl = REPLEnvironment(
            context=combined_context,
            llm_primitives=primitives,
            tool_registry=_state.tool_registry,
            script_registry=_state.script_registry,
            role=request.role or "frontdoor",
        )

        # Root LM loop with streaming
        last_output = ""
        last_error = ""

        for turn in range(request.max_turns):
            turn_start_time = time.perf_counter()

            # Emit turn start
            yield f"data: {json.dumps({'type': 'turn_start', 'turn': turn + 1, 'role': 'frontdoor'})}\n\n"

            # Get state and build prompt
            state = repl.get_state()
            root_prompt = _build_root_lm_prompt(
                state=state,
                original_prompt=request.prompt,
                last_output=last_output,
                last_error=last_error,
                turn=turn,
            )

            # Call Root LM
            try:
                code = primitives.llm_call(root_prompt, role="frontdoor", n_tokens=1024)
            except Exception as e:
                # Log failure (MemRL)
                if _state.progress_logger:
                    _state.progress_logger.log_task_completed(task_id, success=False, details=f"Root LM failed: {e}")
                    _score_completed_task(task_id)
                yield f"data: {json.dumps({'type': 'error', 'message': f'Root LM call failed: {e}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Stream the generated code tokens
            code = _extract_code_from_response(code)
            code = _auto_wrap_final(code)
            for line in code.split("\n"):
                yield f"data: {json.dumps({'type': 'token', 'content': line + chr(10)})}\n\n"

            # Execute in REPL
            result = repl.execute(code)

            # Emit tool calls if any (from REPL execution)
            # TODO: Hook into REPL to capture tool calls

            # Emit turn end
            turn_elapsed_ms = int((time.perf_counter() - turn_start_time) * 1000)
            yield f"data: {json.dumps({'type': 'turn_end', 'tokens': len(code), 'elapsed_ms': turn_elapsed_ms})}\n\n"

            # Check for completion
            if result.is_final:
                yield f"data: {json.dumps({'type': 'final', 'answer': result.final_answer or ''})}\n\n"
                break

            # Update state for next turn
            if result.error:
                last_error = result.error
                last_output = result.output
            else:
                last_error = ""
                last_output = result.output

        # Log completion (MemRL) - success if we got a final answer
        if _state.progress_logger:
            # Check if result exists and is final (may not exist if loop didn't run)
            try:
                success = result.is_final
            except NameError:
                success = False
            _state.progress_logger.log_task_completed(task_id, success=success, details="Stream complete")
            _score_completed_task(task_id)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ============================================================================
# OpenAI-Compatible Endpoints
# ============================================================================


# Available roles/models
AVAILABLE_ROLES = [
    "orchestrator",  # Auto-routing via frontdoor
    "frontdoor",     # Tier A - Root LM
    "coder",         # Tier B - Coder specialist
    "architect",     # Tier B - Architecture specialist
    "worker",        # Tier C - General worker
]


@app.get("/v1/models", response_model=OpenAIModelsResponse)
async def list_models() -> OpenAIModelsResponse:
    """List available models (roles) in OpenAI format."""
    return OpenAIModelsResponse(
        data=[
            OpenAIModelInfo(id=role)
            for role in AVAILABLE_ROLES
        ]
    )


@app.post("/v1/chat/completions", response_model=None)
async def openai_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming modes.
    The 'model' field maps to orchestrator roles:
    - orchestrator: Auto-routing via frontdoor
    - frontdoor: Direct to frontdoor
    - coder: Direct to coder specialist
    - etc.
    """
    # Extract the last user message as the prompt
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    prompt = user_messages[-1].content

    # Map model to role
    role = request.x_orchestrator_role or (
        "frontdoor" if request.model in ("orchestrator", "gpt-4", "gpt-3.5-turbo", "claude-3")
        else request.model
    )

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if request.stream:
        # Streaming mode
        async def generate_stream() -> AsyncGenerator[str, None]:
            start_time = time.perf_counter()

            # For now, use mock mode for simplicity
            # TODO: Wire to real inference with streaming
            mock_response = f"[MOCK] Processed via {role}: {prompt[:100]}..."

            # Stream chunks
            for i, char in enumerate(mock_response):
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": char} if i > 0 else {"role": "assistant", "content": char},
                        "finish_reason": None,
                    }],
                }
                if request.x_show_routing:
                    chunk["x_role"] = role
                    chunk["x_turn"] = 1

                yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk
            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-streaming mode
        start_time = time.perf_counter()

        # Mock response for now
        # TODO: Wire to real inference
        mock_response = f"[MOCK] Processed via {role}: {prompt[:100]}..."

        elapsed = time.perf_counter() - start_time

        return OpenAIChatResponse(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content=mock_response),
                    finish_reason="stop",
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=len(prompt) // 4,  # Rough estimate
                completion_tokens=len(mock_response) // 4,
                total_tokens=(len(prompt) + len(mock_response)) // 4,
            ),
            x_orchestrator_metadata={
                "role": role,
                "elapsed_seconds": elapsed,
            } if request.x_show_routing else None,
        )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> OpenAIModelInfo:
    """Get info for a specific model."""
    if model_id not in AVAILABLE_ROLES:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return OpenAIModelInfo(id=model_id)


# ============================================================================
# Session Management (Claude Code parity)
# ============================================================================


class SessionInfo(BaseModel):
    """Session information."""
    id: str
    name: str | None = None
    created_at: str
    last_active: str
    message_count: int
    working_directory: str | None = None


class SessionListResponse(BaseModel):
    """Response for session list."""
    sessions: list[SessionInfo]


class PermissionResponse(BaseModel):
    """Response for permission requests."""
    request_id: str
    approved: bool
    tool: str


# In-memory session store (would be persisted in production)
_sessions: dict[str, dict] = {}
_pending_permissions: dict[str, dict] = {}


@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """List available sessions."""
    sessions = [
        SessionInfo(
            id=sid,
            name=data.get("name"),
            created_at=data.get("created_at", ""),
            last_active=data.get("last_active", ""),
            message_count=data.get("message_count", 0),
            working_directory=data.get("working_directory"),
        )
        for sid, data in _sessions.items()
    ]
    return SessionListResponse(sessions=sessions)


@app.post("/sessions/{session_id}/resume")
async def resume_session(session_id: str) -> dict[str, Any]:
    """Resume a previous session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session = _sessions[session_id]
    session["last_active"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return {
        "status": "resumed",
        "session_id": session_id,
        "message_count": session.get("message_count", 0),
    }


@app.post("/sessions/current/rename")
async def rename_session(name: str, session_id: str | None = None) -> dict[str, str]:
    """Rename current or specified session."""
    sid = session_id or "current"
    if sid not in _sessions:
        # Create new session with this name
        _sessions[sid] = {
            "name": name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_active": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message_count": 0,
        }
    else:
        _sessions[sid]["name"] = name

    return {"status": "renamed", "session_id": sid, "name": name}


@app.post("/permission/{request_id}")
async def respond_to_permission(request_id: str, approved: bool) -> PermissionResponse:
    """Approve or reject a pending tool execution.

    This is used for interactive permission flows in Normal mode.
    """
    if request_id not in _pending_permissions:
        raise HTTPException(status_code=404, detail=f"Permission request '{request_id}' not found")

    perm = _pending_permissions.pop(request_id)

    return PermissionResponse(
        request_id=request_id,
        approved=approved,
        tool=perm.get("tool", "unknown"),
    )


@app.get("/permission/pending")
async def list_pending_permissions() -> list[dict]:
    """List pending permission requests."""
    return [
        {"id": pid, **data}
        for pid, data in _pending_permissions.items()
    ]


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

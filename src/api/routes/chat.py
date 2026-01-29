"""Chat endpoints for the orchestrator API."""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.models import ChatRequest, ChatResponse
from src.api.state import get_state
from src.api.services.orchestrator import (
    extract_code_from_response,
    auto_wrap_final,
    classify_error,
    build_escalation_prompt,
)
from src.prompt_builders import (
    build_root_lm_prompt,
    build_stage2_review_prompt,
    build_long_context_exploration_prompt,
    build_routing_context,
    build_review_verdict_prompt,
    build_revision_prompt,
)
from src.services.draft_cache import get_draft_cache, CachedDraft
from src.services.prompt_compressor import PromptCompressor
from src.api.services.memrl import (
    ensure_memrl_initialized,
    score_completed_task,
)
from src.features import features
from src.llm_primitives import LLMPrimitives
from src.repl_environment import REPLEnvironment
from src.escalation import (
    EscalationPolicy,
    EscalationContext,
    EscalationAction,
    ErrorCategory as EscalationErrorCategory,
)
from src.generation_monitor import GenerationMonitor, MonitorConfig
from src.roles import Role
from src.sse_utils import (
    create_sse_response,
    token_event,
    thinking_event,
    turn_start_event,
    turn_end_event,
    error_event,
    final_event,
    done_event,
)

# Three-stage summarization configuration (Stage 0: compression, Stage 1: draft, Stage 2: review)
THREE_STAGE_CONFIG = {
    "enabled": True,
    "threshold_tokens": 5000,  # ~20K chars triggers Stage 1+2
    "multi_doc_discount": 0.7,  # Lower threshold for multiple documents
    "stage1_role": Role.FRONTDOOR,
    "stage2_role": Role.INGEST_LONG_CONTEXT,
    # Stage 0: Compression settings (LLMLingua-2)
    # DISABLED: Extractive compression causes quality regression (hallucinations, typos)
    # See handoffs/active/cmprsr_prompt_compression.md for details
    # Re-enable when Cmprsr (abstractive) weights become available
    "compression": {
        "enabled": False,  # Disabled due to quality issues with LLMLingua-2
        "min_chars": 30000,
        "target_ratio": 0.5,
        "stage1_context_limit": 20000,
    },
}

# Backwards compatibility alias
TWO_STAGE_CONFIG = THREE_STAGE_CONFIG

# Long context exploration configuration
# When context exceeds this threshold, use REPL-based chunked exploration
# instead of dumping the full context into a single model's window
LONG_CONTEXT_CONFIG = {
    "enabled": True,
    "threshold_chars": 20000,  # ~5K tokens triggers exploration mode
    "max_turns": 8,  # Allow more turns for multi-step exploration
}


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough: 4 chars per token)."""
    return len(text) // 4


def _is_summarization_task(prompt: str) -> bool:
    """Detect if the prompt is a summarization task.

    Args:
        prompt: The user's prompt.

    Returns:
        True if this looks like a summarization request.
    """
    summarization_keywords = [
        "summarize", "summary", "summarise", "summarisation",
        "executive summary", "overview", "key points",
        "main ideas", "tl;dr", "tldr", "synopsis",
    ]
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in summarization_keywords)


def _should_use_two_stage(
    prompt: str,
    context: str | None,
    doc_count: int = 1,
) -> bool:
    """Determine if two-stage context processing should be used.

    Triggers for ANY large context (not just summarization). The REPL
    exploration approach scored 0/9 on long context benchmarks because
    models generate standalone code instead of calling peek()/grep().
    Two-stage worker digest → frontdoor synthesis is more reliable.

    Args:
        prompt: The user's prompt.
        context: The context (document content).
        doc_count: Number of documents being processed.

    Returns:
        True if two-stage pipeline should be used.
    """
    if not TWO_STAGE_CONFIG["enabled"]:
        return False

    if not context:
        return False

    # Trigger for any context above threshold — not just summarization
    context_chars = len(context)
    threshold_chars = LONG_CONTEXT_CONFIG["threshold_chars"]  # 20K chars

    # Apply multi-doc discount
    if doc_count > 1:
        threshold_chars = int(threshold_chars * TWO_STAGE_CONFIG["multi_doc_discount"])

    return context_chars > threshold_chars


async def _run_two_stage_summarization(
    prompt: str,
    context: str,
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
) -> tuple[str, dict]:
    """Run two-stage context processing pipeline.

    Generalized for ALL large-context tasks (not just summarization):
    Stage 1: Workers digest chunks in parallel (44 t/s each)
    Stage 2: Frontdoor synthesizes answer from digests (18 t/s)

    For summarization tasks, falls through to the original
    Stage 1 (draft) + Stage 2 (large model review) pipeline.

    Args:
        prompt: The user's prompt.
        context: The full document context.
        primitives: LLMPrimitives instance for LLM calls.
        state: Application state.
        task_id: Task ID for logging.

    Returns:
        Tuple of (final_answer, stats_dict).
    """
    import time

    is_summarization = _is_summarization_task(prompt)

    stats = {
        "pipeline": "two_stage_context",
        "stage1_time_ms": 0,
        "stage2_time_ms": 0,
        "context_tokens": _estimate_tokens(context),
        "chunks": 0,
        "cache_hit": False,
    }

    # Determine chunking
    n_chunks = max(2, min(8, len(context) // 16000))  # ~4K tokens per chunk
    stats["chunks"] = n_chunks

    # Stage 1: Worker parallel digest
    stage1_start = time.perf_counter()

    chunk_size = len(context) // n_chunks
    overlap = 200
    chunks = []
    for i in range(n_chunks):
        start_idx = max(0, i * chunk_size - (overlap if i > 0 else 0))
        end_idx = min(len(context), (i + 1) * chunk_size + (overlap if i < n_chunks - 1 else 0))
        chunks.append({"index": i, "text": context[start_idx:end_idx]})

    # Build worker prompts — task-specific instructions
    worker_prompts = []
    for chunk in chunks:
        worker_prompt = (
            f"Analyze this section ({chunk['index']+1}/{n_chunks}) of a larger document.\n"
            f"Task context: {prompt[:200]}\n\n"
            f"## Section Content\n{chunk['text'][:15000]}\n\n"
            f"## Instructions\n"
            f"Extract: key facts, relevant quotes, any findings related to the task.\n"
            f"If the task asks to FIND something specific, look for it and report exact matches.\n"
            f"Be concise. Output structured findings only."
        )
        worker_prompts.append(worker_prompt)

    # Dispatch to workers in parallel via llm_batch
    try:
        digests = primitives.llm_batch(worker_prompts, role="worker_explore", n_tokens=500)
    except Exception:
        # Fallback: sequential calls
        digests = []
        for wp in worker_prompts:
            try:
                d = primitives.llm_call(wp, role="worker_explore", n_tokens=500)
                digests.append(d)
            except Exception:
                digests.append("[Worker failed to process this section]")

    stage1_time = time.perf_counter() - stage1_start
    stats["stage1_time_ms"] = int(stage1_time * 1000)

    # Stage 2: Frontdoor synthesis from digests
    stage2_start = time.perf_counter()

    digest_text = "\n\n".join(
        f"[Section {i+1}/{len(digests)}]\n{d}"
        for i, d in enumerate(digests)
    )

    if is_summarization:
        synthesis_instruction = (
            "Synthesize a comprehensive summary from the section findings above.\n"
            "Cover: main thesis, key innovations, how it works, benefits and audience.\n"
            "Be thorough and well-structured."
        )
    else:
        synthesis_instruction = (
            "Synthesize a complete answer from the section findings above.\n"
            "If searching for specific items, report exact values found.\n"
            "If analyzing the document, provide a thorough answer.\n"
            "Be precise and include specific details from the findings."
        )

    synthesis_prompt = (
        f"You analyzed a large document in {n_chunks} sections. Here are the worker findings:\n\n"
        f"{digest_text}\n\n"
        f"## Original Question\n{prompt}\n\n"
        f"## Instructions\n{synthesis_instruction}"
    )

    # Use frontdoor for synthesis (18 t/s) — much faster than architect
    try:
        answer = primitives.llm_call(
            synthesis_prompt,
            role=TWO_STAGE_CONFIG["stage1_role"],  # frontdoor
            n_tokens=2000,
        )
    except Exception as e:
        # Use digest text directly as fallback
        answer = f"Worker findings:\n{digest_text}"

    stage2_time = time.perf_counter() - stage2_start
    stats["stage2_time_ms"] = int(stage2_time * 1000)

    # Log to progress logger if available
    if state.progress_logger:
        state.progress_logger.log_exploration(
            task_id=task_id,
            query=prompt[:100],
            strategy_used="two_stage_context",
            tokens_spent=_estimate_tokens(synthesis_prompt),
            success=True,
        )

    # Store digests for potential review gate use (Step 6)
    stats["worker_digests"] = [
        {"section": i + 1, "summary": d[:500]}
        for i, d in enumerate(digests)
    ]

    return answer.strip(), stats


async def _handle_vision_request(
    request: "ChatRequest",
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
) -> str:
    """Route a vision request to VL workers (ports 8086/8087).

    Calls the vision API internally using the analyze endpoint,
    then returns the VL model's description as the answer.

    Args:
        request: Chat request with image_path or image_base64.
        primitives: LLMPrimitives (unused, for interface compat).
        state: Application state.
        task_id: Task ID for logging.

    Returns:
        Answer string from the vision model.
    """
    import httpx

    # Build vision analyze request
    vision_payload: dict = {
        "vl_prompt": request.prompt,
        "analyzers": ["vl"],  # Use VL analyzer for question answering
        "store_results": False,
    }
    if request.image_path:
        vision_payload["image_path"] = request.image_path
    elif request.image_base64:
        vision_payload["image_base64"] = request.image_base64

    # Call vision endpoint on the same API server
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "http://localhost:8000/v1/vision/analyze",
            json=vision_payload,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Vision API returned {resp.status_code}: {resp.text[:200]}")

    data = resp.json()

    # Extract VL description from the response
    vl_results = data.get("vl", {})
    description = vl_results.get("description", "")

    if not description:
        # Fall back to general description
        description = data.get("description", "[Vision model returned no description]")

    return description


_STUB_PATTERNS = {
    "complete", "see above", "analysis complete", "estimation complete",
    "done", "finished", "see results above", "see output above",
    "see structured output above", "see integrated results above",
    "see the structured output above",
}


def _is_stub_final(text: str) -> bool:
    """Detect when FINAL() arg is a stub pointing to printed output.

    Models often print their analysis via print(), then call
    FINAL("Analysis complete. See above.") — the real content
    is in result.output, not result.final_answer.
    """
    normalized = text.strip().rstrip(".").lower()
    return any(p in normalized for p in _STUB_PATTERNS)


def _resolve_answer(result: "ExecutionResult") -> str:
    """Extract the best answer from an ExecutionResult.

    Handles cases where the model prints content then uses a stub FINAL().
    """
    captured = result.output.strip() if result.output else ""
    final = result.final_answer or ""

    if captured and _is_stub_final(final):
        return captured
    elif captured and final and captured != final:
        # Prepend captured output if FINAL() doesn't already contain it
        if final not in captured:
            return f"{captured}\n\n{final}"
        return final
    else:
        return final


def _should_review(state: "AppState", task_id: str, role: str, answer: str) -> bool:
    """MemRL-conditional: review only when confidence < threshold.

    Checks Q-values for the current role+task combination. If average
    Q-value is below 0.6, the role historically struggles with this
    task type and a brief architect review is warranted.

    Args:
        state: Application state with hybrid_router.
        task_id: Current task ID.
        role: The role that generated the answer.
        answer: The answer to potentially review.

    Returns:
        True if architect review should be triggered.
    """
    if not state.hybrid_router:
        return False
    if "architect" in str(role):
        return False  # Architects ARE the reviewer — don't self-review
    if len(answer) < 50:
        return False  # Trivial answers don't need review
    try:
        # Get Q-values for this role from MemRL
        retriever = state.hybrid_router.retriever
        task_ir = {"task_type": "chat", "objective": answer[:100]}
        results = retriever.retrieve_for_routing(task_ir)
        if not results:
            return False
        # Filter for current role
        role_results = [r for r in results if r.memory.action == str(role)]
        if not role_results:
            return False
        avg_q = sum(r.q_value for r in role_results) / len(role_results)
        return avg_q < 0.6
    except Exception:
        return False


def _architect_verdict(
    question: str,
    answer: str,
    primitives: "LLMPrimitives",
    worker_digests: list[dict] | None = None,
    context_digest: str = "",
) -> str | None:
    """Get architect's hyper-concise verdict on an answer.

    The architect emits ONLY a short verdict (~20-50 tokens at 6.75 t/s → ~6s).
    Returns None if OK, or "WRONG: <corrections>" if incorrect.

    Args:
        question: Original user question.
        answer: The answer to review.
        primitives: LLM primitives for inference.
        worker_digests: Optional TOON-encodable worker digests.
        context_digest: Optional compact context summary.

    Returns:
        None if answer is OK, or "WRONG: ..." string if corrections needed.
    """
    prompt = build_review_verdict_prompt(
        question, answer,
        context_digest=context_digest,
        worker_digests=worker_digests,
    )
    try:
        result = primitives.llm_call(
            prompt,
            role="architect_general",
            n_tokens=80,  # Hard cap — verdict only
        )
        text = result.strip()
        if text.upper().startswith("OK"):
            return None
        return text  # "WRONG: <corrections>"
    except Exception:
        return None  # On error, don't block — return original answer


def _fast_revise(
    question: str,
    original_answer: str,
    corrections: str,
    primitives: "LLMPrimitives",
) -> str:
    """Fast worker expands architect's corrections into full answer.

    Uses worker_explore (port 8082, 44 t/s) — the fastest model in the stack.
    7B is sufficient since the architect already specified exactly what to fix.

    Args:
        question: Original user question.
        original_answer: The answer to revise.
        corrections: Architect's correction notes.
        primitives: LLM primitives for inference.

    Returns:
        Revised answer, or original if revision fails.
    """
    prompt = build_revision_prompt(question, original_answer, corrections)
    try:
        result = primitives.llm_call(
            prompt,
            role="worker_explore",
            n_tokens=2000,
        )
        return result.strip() or original_answer
    except Exception:
        return original_answer  # Fallback to original on error


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
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
    state = get_state()

    # Track active requests for idle-time Q-scoring (thread-safe)
    state.increment_active()
    try:
        return await _handle_chat(request)
    finally:
        state.decrement_active()


async def _handle_chat(request: ChatRequest) -> ChatResponse:
    """Internal handler for chat requests."""
    state = get_state()
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

    # Initialize MemRL early for real_mode to enable HybridRouter
    if request.real_mode and not use_mock:
        ensure_memrl_initialized(state)

    # Determine routing using HybridRouter if available, otherwise rules
    if use_mock:
        routing_decision = [request.role or Role.FRONTDOOR]
        routing_strategy = "mock"
    elif state.hybrid_router and request.real_mode:
        # Use learned routing for real-mode requests
        routing_decision, routing_strategy = state.hybrid_router.route(task_ir)
    else:
        # Fall back to request role or default
        routing_decision = [request.role or Role.FRONTDOOR]
        routing_strategy = "rules"

    # Log task start (MemRL integration)
    if state.progress_logger:
        state.progress_logger.log_task_started(
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
        state.increment_request(mock_mode=True, turns=turns)

        # Log task completion (MemRL integration)
        if state.progress_logger:
            state.progress_logger.log_task_completed(
                task_id=task_id,
                success=True,
                details=f"Mock response in {elapsed:.3f}s",
            )
            score_completed_task(state, task_id)

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
        # MemRL already initialized earlier for HybridRouter routing

        # Get server URLs from request or use defaults
        server_urls = request.server_urls or LLMPrimitives.DEFAULT_SERVER_URLS

        try:
            primitives = LLMPrimitives(
                mock_mode=False,
                server_urls=server_urls,
                registry=state.registry,
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
        primitives = LLMPrimitives(mock_mode=False, registry=state.registry)

        if primitives.model_server is None:
            raise HTTPException(
                status_code=503,
                detail="Real inference not available: no model server configured",
            )

    # Vision routing: when image data is present, route through vision pipeline
    # instead of standard text-only orchestration
    if request.real_mode and (request.image_path or request.image_base64):
        try:
            answer = await _handle_vision_request(request, primitives, state, task_id)
            elapsed = time.perf_counter() - start_time
            state.increment_request(mock_mode=False, turns=1)

            if state.progress_logger:
                state.progress_logger.log_task_completed(
                    task_id=task_id,
                    success=True,
                    details=f"Vision pipeline, {elapsed:.3f}s",
                )
                score_completed_task(state, task_id)

            return ChatResponse(
                answer=answer,
                turns=1,
                tokens_used=0,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=True,
                cache_stats=None,
                routed_to="worker_vision",
                role_history=["worker_vision"],
                routing_strategy=routing_strategy,
            )
        except Exception as e:
            import logging
            logging.warning(f"Vision pipeline failed: {type(e).__name__}: {e}")
            # Make image info available to REPL as fallback context
            if request.image_path:
                request.context = (request.context or "") + (
                    f"\n\n[IMAGE: {request.image_path} — Vision pipeline unavailable. "
                    f"Use image analysis tools if available, otherwise note the limitation.]"
                )
            # Fall through to standard orchestration

    # Create REPL environment
    combined_context = request.prompt
    if request.context:
        combined_context += f"\n\nContext:\n{request.context}"

    # Use first role from routing decision (may be learned or rule-based)
    initial_role = routing_decision[0] if routing_decision else Role.FRONTDOOR

    repl = REPLEnvironment(
        context=combined_context,
        llm_primitives=primitives,
        tool_registry=state.tool_registry,
        script_registry=state.script_registry,
        role=initial_role,
        progress_logger=state.progress_logger,
        task_id=task_id,
        # MemRL components for model self-routing
        retriever=state.hybrid_router.retriever if state.hybrid_router else None,
        hybrid_router=state.hybrid_router,
    )

    # Check for two-stage summarization opportunity
    if request.real_mode and _should_use_two_stage(
        prompt=request.prompt,
        context=request.context,
    ):
        try:
            answer, two_stage_stats = await _run_two_stage_summarization(
                prompt=request.prompt,
                context=request.context or "",
                primitives=primitives,
                state=state,
                task_id=task_id,
            )

            elapsed = time.perf_counter() - start_time
            state.increment_request(mock_mode=False, turns=2)  # Count as 2 turns

            # Log task completion
            if state.progress_logger:
                cache_info = "cache hit" if two_stage_stats.get("cache_hit") else "cache miss"
                state.progress_logger.log_task_completed(
                    task_id=task_id,
                    success=True,
                    details=f"Two-stage summarization ({cache_info}), {elapsed:.3f}s",
                )
                score_completed_task(state, task_id)

            cache_stats = primitives.get_cache_stats() if primitives._backends else None

            return ChatResponse(
                answer=answer,
                turns=2,
                tokens_used=primitives.total_tokens_generated,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=True,
                cache_stats=cache_stats,
                routed_to=str(initial_role),
                role_history=[str(initial_role), str(TWO_STAGE_CONFIG["stage2_role"])],
                routing_strategy=routing_strategy,
                tokens_generated=primitives.total_tokens_generated,
            )
        except Exception as e:
            # Fall back to standard orchestration on two-stage failure
            import logging
            logging.warning(f"Two-stage summarization failed: {type(e).__name__}: {e}")
            # Continue to normal loop below

    # Detect long context → use REPL-based exploration strategy
    # Instead of dumping full context into one model, the frontdoor uses
    # peek/grep/summarize_chunks to explore and synthesize.
    context_chars = len(combined_context)
    use_long_context_exploration = (
        LONG_CONTEXT_CONFIG["enabled"]
        and request.real_mode
        and context_chars > LONG_CONTEXT_CONFIG["threshold_chars"]
    )

    if use_long_context_exploration:
        import logging
        logging.info(
            f"Long context detected ({context_chars:,} chars). "
            f"Using REPL exploration strategy."
        )

    # Run Root LM orchestration loop with escalation support
    turns = 0
    answer = ""
    last_output = ""
    last_error = ""

    # Escalation tracking
    current_role = initial_role
    consecutive_failures = 0
    role_history = [current_role]
    escalation_prompt = ""  # Set when escalating

    # Long context exploration allows more turns
    max_turns = (
        LONG_CONTEXT_CONFIG["max_turns"]
        if use_long_context_exploration
        else request.max_turns
    )

    for turn in range(max_turns):
        turns += 1

        # 1. Get current REPL state
        repl_state = repl.get_state()

        # 2. Build prompt - use escalation prompt if we just escalated
        if escalation_prompt:
            root_prompt = escalation_prompt
            escalation_prompt = ""  # Clear after use
        elif use_long_context_exploration and turn == 0:
            # First turn with long context: use exploration-aware prompt
            root_prompt = build_long_context_exploration_prompt(
                original_prompt=request.prompt,
                context_chars=context_chars,
                state=repl_state,
            )
        else:
            # Inject routing context on turn 0 (MemRL intelligence)
            routing_ctx = ""
            if turn == 0 and state.hybrid_router:
                routing_ctx = build_routing_context(
                    role=current_role,
                    hybrid_router=state.hybrid_router,
                    task_description=request.prompt,
                )
            root_prompt = build_root_lm_prompt(
                state=repl_state,
                original_prompt=request.prompt,
                last_output=last_output,
                last_error=last_error,
                turn=turn,
                routing_context=routing_ctx,
            )

        # 3. Call Root LM (current role) to generate Python code
        # Use generation monitoring for early failure detection if feature enabled
        f = features()
        generation_aborted = False
        abort_reason = ""

        try:
            if f.generation_monitor and not request.mock_mode:
                # Create monitor with tier-appropriate config
                monitor_config = MonitorConfig.for_tier(current_role)
                monitor = GenerationMonitor(
                    config=monitor_config,
                    mock_mode=request.mock_mode,
                )
                llm_result = primitives.llm_call_monitored(
                    root_prompt,
                    role=current_role,
                    monitor=monitor,
                )
                code = llm_result.text
                generation_aborted = llm_result.aborted
                abort_reason = llm_result.abort_reason
            else:
                # Standard call without monitoring
                code = primitives.llm_call(
                    root_prompt,
                    role=current_role,
                    n_tokens=1024,
                )
        except Exception as e:
            # If LLM call fails, return error
            answer = f"[ERROR: {current_role} LM call failed: {e}]"
            break

        # Handle early abort from generation monitoring
        if generation_aborted:
            # Treat as early failure detection - escalate immediately
            escalation_ctx = EscalationContext(
                current_role=current_role,
                error_message=f"Generation aborted: {abort_reason}",
                error_category="early_abort",
                failure_count=1,  # Treat as first failure to trigger escalation
                task_id=task_id,
            )
            policy = EscalationPolicy()
            decision = policy.decide(escalation_ctx)

            if decision.should_escalate and decision.target_role:
                current_role = str(decision.target_role)
                role_history.append(current_role)
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=escalation_ctx,
                    decision=decision,
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=role_history[-2],
                        to_tier=current_role,
                        reason=f"Early abort: {abort_reason}",
                    )
                continue  # Skip to next turn with escalated role
            else:
                # Can't escalate further - try to use partial output
                pass  # Continue with partial code

        # Extract code from response (handle markdown code blocks)
        code = extract_code_from_response(code)
        # Auto-wrap in FINAL() if code looks like a complete answer
        code = auto_wrap_final(code)

        # 4. Execute code in REPL
        result = repl.execute(code)

        # 4a. Check model-initiated routing (escalation/delegation artifacts)
        if repl.artifacts.get("_escalation_requested"):
            target = repl.artifacts.pop("_escalation_target", None)
            reason = repl.artifacts.pop("_escalation_reason", "Model requested")
            repl.artifacts.pop("_escalation_requested", None)

            new_role = None
            if target:
                resolved = Role.from_string(target)
                if resolved:
                    new_role = str(resolved)
            else:
                # No specific target — use standard escalation chain
                esc_ctx = EscalationContext(
                    current_role=current_role,
                    error_category="early_abort",
                    error_message=reason,
                    failure_count=1,
                    task_id=task_id,
                )
                esc_decision = EscalationPolicy().decide(esc_ctx)
                if esc_decision.should_escalate and esc_decision.target_role:
                    new_role = str(esc_decision.target_role)

            if new_role and new_role != current_role:
                current_role = new_role
                role_history.append(current_role)
                consecutive_failures = 0
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=EscalationContext(
                        current_role=role_history[-2],
                        error_message=reason,
                        error_category="early_abort",
                        task_id=task_id,
                    ),
                    decision=EscalationPolicy().decide(EscalationContext(
                        current_role=role_history[-2],
                        error_category="early_abort",
                        error_message=reason,
                        task_id=task_id,
                    )),
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=role_history[-2],
                        to_tier=current_role,
                        reason=f"Model-initiated: {reason}",
                    )
                continue  # Next turn with new role

        # 4b. Log delegation outcomes (MemRL learning)
        if repl.artifacts.get("_delegations"):
            for deleg in repl.artifacts["_delegations"]:
                if state.progress_logger:
                    state.progress_logger.log_exploration(
                        task_id=task_id,
                        query=deleg.get("prompt_preview", ""),
                        strategy_used=f"delegate:{deleg.get('to_role', 'unknown')}",
                        success=deleg.get("success", False),
                    )
            repl.artifacts["_delegations"] = []  # Clear after logging

        # 5. Check for FINAL() completion
        if result.is_final:
            answer = _resolve_answer(result)
            consecutive_failures = 0  # Success resets failure count

            # MemRL-informed quality review gate (blocking)
            if request.real_mode and _should_review(state, task_id, current_role, answer):
                import logging
                logging.info(f"Review gate triggered for {current_role} (task {task_id})")
                verdict = _architect_verdict(
                    question=request.prompt,
                    answer=answer,
                    primitives=primitives,
                )
                if verdict and verdict.upper().startswith("WRONG"):
                    corrections = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
                    logging.info(f"Review verdict: WRONG — revising ({corrections[:80]})")
                    answer = _fast_revise(
                        question=request.prompt,
                        original_answer=answer,
                        corrections=corrections,
                        primitives=primitives,
                    )
                else:
                    logging.info("Review verdict: OK")

            break

        # 6. Handle errors with EscalationPolicy for escalation decisions
        if result.error:
            consecutive_failures += 1
            last_error = result.error
            last_output = result.output

            # Consult EscalationPolicy for escalation decision (unified module)
            error_category = classify_error(result.error)
            escalation_ctx = EscalationContext(
                current_role=current_role,
                error_message=result.error,
                error_category=error_category.value,  # Pass as string for compatibility
                failure_count=consecutive_failures,
                task_id=task_id,
            )
            policy = EscalationPolicy()
            decision = policy.decide(escalation_ctx)

            # Log escalation decision (for escalate actions)
            if decision.should_escalate and state.progress_logger:
                state.progress_logger.log_escalation(
                    task_id=task_id,
                    from_tier=current_role,
                    to_tier=str(decision.target_role) if decision.target_role else current_role,
                    reason=f"{decision.reason} (failures: {consecutive_failures})",
                )

            # Act on routing decision
            if decision.should_escalate and decision.target_role:
                # Switch to higher-tier role
                current_role = str(decision.target_role)
                role_history.append(current_role)
                consecutive_failures = 0  # Reset for new role
                # Build escalation prompt with failure context
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=escalation_ctx,
                    decision=decision,
                )
            elif decision.action == EscalationAction.EXPLORE:
                # Terminal role — fall back to REPL exploration
                # Switch to exploration prompt so the model uses
                # peek/grep/summarize_chunks instead of raw LLM calls
                consecutive_failures = 0
                escalation_prompt = build_long_context_exploration_prompt(
                    original_prompt=request.prompt,
                    context_chars=len(repl.context),
                    state=repl_state,
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=current_role,
                        to_tier=f"{current_role}+explore",
                        reason="Terminal role: switching to REPL exploration",
                    )
            elif decision.action == EscalationAction.FAIL:
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
        answer = f"[Max turns ({max_turns}) reached without FINAL()]"
        if last_output:
            answer += f"\n\nLast output:\n{last_output}"

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=False, turns=turns)

    # Get cache stats if using real_mode with RadixAttention
    cache_stats = None
    if request.real_mode and primitives._backends:
        cache_stats = primitives.get_cache_stats()

    # Log exploration telemetry (tool usage, function counts)
    success = not answer.startswith("[ERROR") and not answer.startswith("[Max turns")
    repl.log_exploration_completed(success=success, result=answer)

    # Log task completion (MemRL integration)
    if state.progress_logger:
        # Include role history if escalation occurred
        role_info = f", roles: {' -> '.join(role_history)}" if len(role_history) > 1 else ""
        state.progress_logger.log_task_completed(
            task_id=task_id,
            success=success,
            details=f"Real inference: {turns} turns, {elapsed:.3f}s{role_info}",
        )
        score_completed_task(state, task_id)

    return ChatResponse(
        answer=answer,
        turns=turns,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=request.real_mode,
        cache_stats=cache_stats,
        routed_to=str(current_role),
        role_history=[str(r) for r in role_history],
        routing_strategy=routing_strategy,
        tokens_generated=primitives.total_tokens_generated,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE streaming endpoint with routing metadata.

    Streams events using standardized SSE format (via sse_utils):
    - turn_start: {type: "turn_start", turn: N, role: "..."}
    - thinking: {type: "thinking", content: "..."} (when thinking_budget > 0)
    - token: {type: "token", content: "..."}
    - tool: {type: "tool", name: "...", args: {...}, result: ...}
    - permission_request: {type: "permission_request", id: "...", tool: "...", args: {...}}
    - file: {type: "file", path: "...", content: "...", action: "create"|"modify"}
    - turn_end: {type: "turn_end", tokens: N, elapsed_ms: N}
    - error: {type: "error", message: "..."}
    - done: [DONE] when complete

    Parameters:
    - thinking_budget: Token budget for internal reasoning (0=disabled)
    - permission_mode: "normal", "auto-accept", or "plan"

    Note: Uses sse-starlette when available (via feature flag), otherwise
    falls back to manual SSE formatting for backward compatibility.
    """
    state = get_state()

    # Generate task ID for MemRL tracking (outside generator for closure)
    task_id = f"stream-{uuid.uuid4().hex[:8]}"

    async def generate() -> AsyncGenerator[dict, None]:
        start_time = time.perf_counter()
        use_mock = request.mock_mode and not request.real_mode

        # Construct task_ir and log start (MemRL integration)
        task_ir = {
            "task_type": "chat_stream",
            "objective": request.prompt[:200],
            "priority": "interactive",
        }
        if state.progress_logger:
            state.progress_logger.log_task_started(
                task_id=task_id,
                task_ir=task_ir,
                routing_decision=[request.role],
                routing_strategy="mock" if use_mock else "rules",
            )

        # Mock mode
        if use_mock:
            # Emit turn start
            yield turn_start_event(turn=1, role=str(Role.FRONTDOOR))

            # Emit thinking events if thinking_budget > 0 (Claude Code parity)
            if request.thinking_budget > 0:
                thinking_steps = [
                    "Analyzing the user's request...",
                    f"Request type: {request.prompt[:30].split()[0] if request.prompt else 'unknown'}",
                    "Determining appropriate response strategy...",
                    "Preparing response...",
                ]
                for step in thinking_steps:
                    yield thinking_event(step)

            # Check permission mode - in plan mode, only emit analysis
            if request.permission_mode == "plan":
                analysis = f"[PLAN MODE] Would process: {request.prompt[:100]}..."
                yield token_event(analysis)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                yield turn_end_event(tokens=len(analysis), elapsed_ms=elapsed_ms)
                # Log completion (MemRL)
                if state.progress_logger:
                    state.progress_logger.log_task_completed(task_id, success=True, details="Plan mode")
                    score_completed_task(state, task_id)
                yield done_event()
                return

            # Simulate streaming tokens
            mock_response = f"[MOCK] Processed: {request.prompt[:50]}..."
            for char in mock_response:
                yield token_event(char)

            # Emit turn end
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            yield turn_end_event(tokens=len(mock_response), elapsed_ms=elapsed_ms)

            # Log completion (MemRL)
            if state.progress_logger:
                state.progress_logger.log_task_completed(task_id, success=True, details="Mock stream")
                score_completed_task(state, task_id)
            yield done_event()
            return

        # Real mode - initialize MemRL components on first real use (lazy loading)
        ensure_memrl_initialized(state)

        server_urls = request.server_urls or LLMPrimitives.DEFAULT_SERVER_URLS
        try:
            primitives = LLMPrimitives(mock_mode=False, server_urls=server_urls, registry=state.registry)
        except Exception as e:
            # Log failure (MemRL)
            if state.progress_logger:
                state.progress_logger.log_task_completed(task_id, success=False, details=str(e))
                score_completed_task(state, task_id)
            yield error_event(str(e))
            yield done_event()
            return

        # Create REPL
        combined_context = request.prompt
        if request.context:
            combined_context += f"\n\nContext:\n{request.context}"

        repl = REPLEnvironment(
            context=combined_context,
            llm_primitives=primitives,
            tool_registry=state.tool_registry,
            script_registry=state.script_registry,
            role=request.role or Role.FRONTDOOR,
            # MemRL components for model self-routing
            retriever=state.hybrid_router.retriever if state.hybrid_router else None,
            hybrid_router=state.hybrid_router,
        )

        # Root LM loop with streaming + escalation support
        last_output = ""
        last_error = ""
        result = None

        # Escalation tracking (parity with main endpoint)
        current_role = request.role or Role.FRONTDOOR
        consecutive_failures = 0
        role_history = [current_role]
        escalation_prompt = ""

        for turn in range(request.max_turns):
            turn_start_time_inner = time.perf_counter()

            # Emit turn start with current role
            yield turn_start_event(turn=turn + 1, role=str(current_role))

            # Get state and build prompt
            repl_state = repl.get_state()
            if escalation_prompt:
                root_prompt = escalation_prompt
                escalation_prompt = ""
            else:
                # Inject routing context on turn 0
                routing_ctx = ""
                if turn == 0 and state.hybrid_router:
                    routing_ctx = build_routing_context(
                        role=current_role,
                        hybrid_router=state.hybrid_router,
                        task_description=request.prompt,
                    )
                root_prompt = build_root_lm_prompt(
                    state=repl_state,
                    original_prompt=request.prompt,
                    last_output=last_output,
                    last_error=last_error,
                    turn=turn,
                    routing_context=routing_ctx,
                )

            # Call Root LM with current role
            try:
                code = primitives.llm_call(root_prompt, role=current_role, n_tokens=1024)
            except Exception as e:
                # Log failure (MemRL)
                if state.progress_logger:
                    state.progress_logger.log_task_completed(task_id, success=False, details=f"Root LM failed: {e}")
                    score_completed_task(state, task_id)
                yield error_event(f"Root LM call failed: {e}")
                yield done_event()
                return

            # Stream the generated code tokens
            code = extract_code_from_response(code)
            # Auto-wrap in FINAL() if code looks like a complete answer
            code = auto_wrap_final(code)
            for line in code.split("\n"):
                yield token_event(line + "\n")

            # Execute in REPL
            result = repl.execute(code)

            # Check model-initiated routing artifacts
            if repl.artifacts.get("_escalation_requested"):
                target = repl.artifacts.pop("_escalation_target", None)
                reason = repl.artifacts.pop("_escalation_reason", "Model requested")
                repl.artifacts.pop("_escalation_requested", None)

                new_role = None
                if target:
                    resolved = Role.from_string(target)
                    if resolved:
                        new_role = str(resolved)
                else:
                    esc_ctx = EscalationContext(
                        current_role=current_role,
                        error_category="early_abort",
                        error_message=reason,
                        failure_count=1,
                        task_id=task_id,
                    )
                    esc_decision = EscalationPolicy().decide(esc_ctx)
                    if esc_decision.should_escalate and esc_decision.target_role:
                        new_role = str(esc_decision.target_role)

                if new_role and new_role != current_role:
                    # Emit turn end before role switch
                    turn_elapsed_ms = int((time.perf_counter() - turn_start_time_inner) * 1000)
                    yield turn_end_event(tokens=len(code), elapsed_ms=turn_elapsed_ms)

                    current_role = new_role
                    role_history.append(current_role)
                    consecutive_failures = 0
                    escalation_prompt = build_escalation_prompt(
                        original_prompt=request.prompt,
                        state=repl_state,
                        failure_context=EscalationContext(
                            current_role=role_history[-2],
                            error_message=reason,
                            error_category="early_abort",
                            task_id=task_id,
                        ),
                        decision=EscalationPolicy().decide(EscalationContext(
                            current_role=role_history[-2],
                            error_category="early_abort",
                            error_message=reason,
                            task_id=task_id,
                        )),
                    )
                    if state.progress_logger:
                        state.progress_logger.log_escalation(
                            task_id=task_id,
                            from_tier=role_history[-2],
                            to_tier=current_role,
                            reason=f"Model-initiated: {reason}",
                        )
                    continue  # Next turn with new role

            # Log delegation outcomes
            if repl.artifacts.get("_delegations"):
                for deleg in repl.artifacts["_delegations"]:
                    if state.progress_logger:
                        state.progress_logger.log_exploration(
                            task_id=task_id,
                            query=deleg.get("prompt_preview", ""),
                            strategy_used=f"delegate:{deleg.get('to_role', 'unknown')}",
                            success=deleg.get("success", False),
                        )
                repl.artifacts["_delegations"] = []

            # Emit turn end
            turn_elapsed_ms = int((time.perf_counter() - turn_start_time_inner) * 1000)
            yield turn_end_event(tokens=len(code), elapsed_ms=turn_elapsed_ms)

            # Check for completion
            if result.is_final:
                stream_answer = _resolve_answer(result)

                # MemRL-informed quality review gate (blocking, streaming parity)
                if _should_review(state, task_id, current_role, stream_answer):
                    verdict = _architect_verdict(
                        question=request.prompt,
                        answer=stream_answer,
                        primitives=primitives,
                    )
                    if verdict and verdict.upper().startswith("WRONG"):
                        corrections = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
                        stream_answer = _fast_revise(
                            question=request.prompt,
                            original_answer=stream_answer,
                            corrections=corrections,
                            primitives=primitives,
                        )

                yield final_event(stream_answer)
                break

            # Handle errors with escalation policy
            if result.error:
                consecutive_failures += 1
                last_error = result.error
                last_output = result.output

                error_category = classify_error(result.error)
                esc_ctx = EscalationContext(
                    current_role=current_role,
                    error_message=result.error,
                    error_category=error_category.value,
                    failure_count=consecutive_failures,
                    task_id=task_id,
                )
                decision = EscalationPolicy().decide(esc_ctx)

                if decision.should_escalate and decision.target_role:
                    current_role = str(decision.target_role)
                    role_history.append(current_role)
                    consecutive_failures = 0
                    escalation_prompt = build_escalation_prompt(
                        original_prompt=request.prompt,
                        state=repl_state,
                        failure_context=esc_ctx,
                        decision=decision,
                    )
                    if state.progress_logger:
                        state.progress_logger.log_escalation(
                            task_id=task_id,
                            from_tier=role_history[-2],
                            to_tier=current_role,
                            reason=f"{decision.reason} (failures: {consecutive_failures})",
                        )
                elif decision.action == EscalationAction.FAIL:
                    yield error_event(f"[FAILED: {decision.reason}]")
                    break
            else:
                consecutive_failures = 0
                last_error = ""
                last_output = result.output

        # Log completion (MemRL) - success if we got a final answer
        if state.progress_logger:
            success = result is not None and result.is_final
            role_info = f", roles: {' -> '.join(str(r) for r in role_history)}" if len(role_history) > 1 else ""
            state.progress_logger.log_task_completed(
                task_id, success=success,
                details=f"Stream complete{role_info}",
            )
            score_completed_task(state, task_id)
        yield done_event()

    # Use SSE utilities for response (handles sse-starlette vs manual fallback)
    return create_sse_response(generate())

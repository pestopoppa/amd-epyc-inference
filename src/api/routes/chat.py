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
    build_root_lm_prompt,
    extract_code_from_response,
    auto_wrap_final,
    classify_error,
    build_escalation_prompt,
)
from src.prompt_builders import build_stage2_review_prompt
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
    """Determine if two-stage summarization should be used.

    Args:
        prompt: The user's prompt.
        context: The context (document content).
        doc_count: Number of documents being processed.

    Returns:
        True if two-stage pipeline should be used.
    """
    if not TWO_STAGE_CONFIG["enabled"]:
        return False

    if not _is_summarization_task(prompt):
        return False

    if not context:
        return False

    # Estimate total tokens
    total_text = prompt + (context or "")
    token_estimate = _estimate_tokens(total_text)

    # Apply multi-doc discount
    threshold = TWO_STAGE_CONFIG["threshold_tokens"]
    if doc_count > 1:
        threshold = int(threshold * TWO_STAGE_CONFIG["multi_doc_discount"])

    return token_estimate > threshold


async def _run_two_stage_summarization(
    prompt: str,
    context: str,
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
) -> tuple[str, dict]:
    """Run two-stage summarization pipeline.

    Stage 1: Fast draft from frontdoor + grep hits capture
    Stage 2: Quality review from large model with reduced context

    Args:
        prompt: The user's summarization prompt.
        context: The full document context.
        primitives: LLMPrimitives instance for LLM calls.
        state: Application state.
        task_id: Task ID for logging.

    Returns:
        Tuple of (final_summary, stats_dict).
    """
    import time

    stats = {
        "pipeline": "three_stage",
        "stage0_time_ms": 0,  # Compression time
        "stage1_time_ms": 0,
        "stage2_time_ms": 0,
        "context_tokens": _estimate_tokens(context),
        "compressed_tokens": 0,
        "compression_ratio": 1.0,
        "cache_hit": False,
    }

    # Check cache first
    cache = get_draft_cache()
    cache_key = cache.make_key(context)

    cached = cache.get(cache_key)
    if cached:
        stats["cache_hit"] = True
        draft_summary = cached.draft_summary
        grep_hits = cached.grep_hits
        figures = cached.figures
    else:
        # Stage 0: Compress context if above threshold (LLMLingua-2)
        compression_config = THREE_STAGE_CONFIG["compression"]
        context_for_stage1 = context

        if compression_config["enabled"] and len(context) > compression_config["min_chars"]:
            import logging
            logging.info(f"Stage 0: Compressing {len(context)} chars with LLMLingua-2")

            stage0_start = time.perf_counter()
            try:
                compressor = PromptCompressor.get_instance()
                compression_result = compressor.compress(
                    context,
                    target_ratio=compression_config["target_ratio"],
                )
                context_for_stage1 = compression_result.compressed_text
                stats["stage0_time_ms"] = int(compression_result.latency_ms)
                stats["compressed_tokens"] = compression_result.compressed_tokens
                stats["compression_ratio"] = compression_result.actual_ratio

                logging.info(
                    f"Stage 0 complete: {compression_result.original_chars} → "
                    f"{compression_result.compressed_chars} chars "
                    f"({compression_result.actual_ratio:.1%}) in {compression_result.latency_ms:.1f}ms"
                )
            except Exception as e:
                # Fall back to truncation if compression fails
                logging.warning(f"Stage 0 compression failed: {e}, falling back to truncation")
                stats["stage0_time_ms"] = int((time.perf_counter() - stage0_start) * 1000)

        # Stage 1: Run frontdoor to generate draft summary
        stage1_start = time.perf_counter()

        # Apply context limit for Stage 1 (in case compression didn't reduce enough)
        context_limit = compression_config["stage1_context_limit"]
        if len(context_for_stage1) > context_limit:
            context_preview = context_for_stage1[:context_limit]
            context_preview += f"\n\n[... {len(context_for_stage1) - context_limit} more chars available in full document ...]"
        else:
            context_preview = context_for_stage1

        # Build Stage 1 prompt - direct summarization request
        stage1_prompt = f"""You are summarizing a technical document. Write a draft executive summary.

## Task
{prompt}

## Document Content (preview)
{context_preview}

## Instructions
Write a draft summary covering:
1. Main thesis and purpose
2. Key innovations or contributions
3. How it works (high-level mechanics)
4. Benefits and target audience

Be concise but comprehensive. This draft will be reviewed and refined.

Draft Summary:"""

        # Get Stage 1 response - direct LLM call, no REPL
        draft_summary = primitives.llm_call(
            stage1_prompt,
            role=TWO_STAGE_CONFIG["stage1_role"],
            n_tokens=800,
        )

        # Extract key excerpts by pattern matching for Stage 2
        # These serve as "grep hits" for verification
        grep_hits = []
        key_patterns = ["abstract", "introduction", "conclusion", "key", "innovation", "benefit"]
        for pattern in key_patterns:
            import re
            matches = re.findall(
                f"(.{{0,100}}{pattern}.{{0,200}})",
                context,
                re.IGNORECASE
            )
            if matches:
                grep_hits.append({
                    "pattern": pattern,
                    "source": "context",
                    "match_count": len(matches),
                    "hits": [{"context": m[:500]} for m in matches[:3]],
                })

        figures = []  # No figures in simple mode

        stage1_time = time.perf_counter() - stage1_start
        stats["stage1_time_ms"] = int(stage1_time * 1000)

        # Cache the Stage 1 results
        cache.set(cache_key, CachedDraft(
            draft_summary=draft_summary,
            grep_hits=grep_hits,
            figures=figures,
            timestamp=time.time(),
            context_tokens=stats["context_tokens"],
            processing_time_ms=stats["stage1_time_ms"],
        ))

    # Stage 2: Review with large model using reduced context
    stage2_start = time.perf_counter()

    stage2_prompt = build_stage2_review_prompt(
        draft_summary=draft_summary,
        grep_hits=grep_hits,
        figures=figures,
        original_task=prompt,
    )

    # Call Stage 2 model with extended timeout (large model is slower)
    # Temporarily increase timeout for Stage 2 inference
    original_timeout = primitives.config.call_timeout
    primitives.config.call_timeout = 300  # 5 minute timeout for Stage 2
    try:
        final_summary = primitives.llm_call(
            stage2_prompt,
            role=TWO_STAGE_CONFIG["stage2_role"],
            n_tokens=1024,
        )
    finally:
        primitives.config.call_timeout = original_timeout

    stage2_time = time.perf_counter() - stage2_start
    stats["stage2_time_ms"] = int(stage2_time * 1000)
    stats["stage2_context_tokens"] = _estimate_tokens(stage2_prompt)

    # Log to progress logger if available
    if state.progress_logger:
        state.progress_logger.log_exploration(
            task_id=task_id,
            query=prompt[:100],
            strategy_used="two_stage_summarization",
            tokens_spent=stats["stage2_context_tokens"],
            success=True,
        )

    return final_summary, stats


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
            )
        except Exception as e:
            # Fall back to standard orchestration on two-stage failure
            import logging
            logging.warning(f"Two-stage summarization failed: {type(e).__name__}: {e}")
            # Continue to normal loop below

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

    for turn in range(request.max_turns):
        turns += 1

        # 1. Get current REPL state
        repl_state = repl.get_state()

        # 2. Build prompt - use escalation prompt if we just escalated
        if escalation_prompt:
            root_prompt = escalation_prompt
            escalation_prompt = ""  # Clear after use
        else:
            root_prompt = build_root_lm_prompt(
                state=repl_state,
                original_prompt=request.prompt,
                last_output=last_output,
                last_error=last_error,
                turn=turn,
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

        # 5. Check for FINAL() completion
        if result.is_final:
            answer = result.final_answer or ""
            consecutive_failures = 0  # Success resets failure count
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
        answer = f"[Max turns ({request.max_turns}) reached without FINAL()]"
        if last_output:
            answer += f"\n\nLast output:\n{last_output}"

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=False, turns=turns)

    # Get cache stats if using real_mode with RadixAttention
    cache_stats = None
    if request.real_mode and primitives._backends:
        cache_stats = primitives.get_cache_stats()

    # Log task completion (MemRL integration)
    success = not answer.startswith("[ERROR") and not answer.startswith("[Max turns")
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
        )

        # Root LM loop with streaming
        last_output = ""
        last_error = ""
        result = None

        for turn in range(request.max_turns):
            turn_start_time_inner = time.perf_counter()

            # Emit turn start
            yield turn_start_event(turn=turn + 1, role=str(Role.FRONTDOOR))

            # Get state and build prompt
            repl_state = repl.get_state()
            root_prompt = build_root_lm_prompt(
                state=repl_state,
                original_prompt=request.prompt,
                last_output=last_output,
                last_error=last_error,
                turn=turn,
            )

            # Call Root LM
            try:
                code = primitives.llm_call(root_prompt, role=Role.FRONTDOOR, n_tokens=1024)
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

            # Emit turn end
            turn_elapsed_ms = int((time.perf_counter() - turn_start_time_inner) * 1000)
            yield turn_end_event(tokens=len(code), elapsed_ms=turn_elapsed_ms)

            # Check for completion
            if result.is_final:
                yield final_event(result.final_answer or "")
                break

            # Update state for next turn
            if result.error:
                last_error = result.error
                last_output = result.output
            else:
                last_error = ""
                last_output = result.output

        # Log completion (MemRL) - success if we got a final answer
        if state.progress_logger:
            success = result is not None and result.is_final
            state.progress_logger.log_task_completed(task_id, success=success, details="Stream complete")
            score_completed_task(state, task_id)
        yield done_event()

    # Use SSE utilities for response (handles sse-starlette vs manual fallback)
    return create_sse_response(generate())

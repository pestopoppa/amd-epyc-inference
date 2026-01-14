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
        routing_decision = [request.role or "frontdoor"]
        routing_strategy = "mock"
    elif state.hybrid_router and request.real_mode:
        # Use learned routing for real-mode requests
        routing_decision, routing_strategy = state.hybrid_router.route(task_ir)
    else:
        # Fall back to request role or default
        routing_decision = [request.role or "frontdoor"]
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
    initial_role = routing_decision[0] if routing_decision else "frontdoor"

    repl = REPLEnvironment(
        context=combined_context,
        llm_primitives=primitives,
        tool_registry=state.tool_registry,
        script_registry=state.script_registry,
        role=initial_role,
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
            yield turn_start_event(turn=1, role="frontdoor")

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
            role=request.role or "frontdoor",
        )

        # Root LM loop with streaming
        last_output = ""
        last_error = ""
        result = None

        for turn in range(request.max_turns):
            turn_start_time_inner = time.perf_counter()

            # Emit turn start
            yield turn_start_event(turn=turn + 1, role="frontdoor")

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
                code = primitives.llm_call(root_prompt, role="frontdoor", n_tokens=1024)
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

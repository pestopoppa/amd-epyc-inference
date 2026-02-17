"""OpenAI-compatible endpoints for the orchestrator API.

These endpoints allow tools like Aider, LM Studio, and other OpenAI-compatible
clients to use our orchestrator backend for inference.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.api.dependencies import dep_app_state
from src.api.models import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIModelInfo,
    OpenAIModelsResponse,
    OpenAIUsage,
)
from src.api.state import AppState
from src.prompt_builders import (
    build_root_lm_prompt,
    extract_code_from_response,
    auto_wrap_final,
)
from src.repl_environment import REPLEnvironment
from src.roles import Role

logger = logging.getLogger(__name__)

router = APIRouter()

# Available roles/models
AVAILABLE_ROLES = [
    "orchestrator",  # Auto-routing via frontdoor
    "frontdoor",  # Tier A - Root LM
    "coder_escalation",  # Tier B - Coder escalation
    "architect",  # Tier B - Architecture specialist
    "architect_general",  # Tier B - General architect
    "architect_coding",  # Tier B - Coding architect
    "worker",  # Tier C - General worker
    "worker_general",  # Tier C - General worker
    "worker_math",  # Tier C - Math worker
    "worker_vision",  # Tier C - Vision worker
    "ingest_long_context",  # Tier B - Long context ingestion
]


@router.get("/models", response_model=OpenAIModelsResponse)
async def list_models() -> OpenAIModelsResponse:
    """List available models (roles) in OpenAI format."""
    return OpenAIModelsResponse(data=[OpenAIModelInfo(id=role) for role in AVAILABLE_ROLES])


@router.post("/chat/completions", response_model=None)
async def openai_chat_completions(
    request: OpenAIChatRequest,
    state: AppState = Depends(dep_app_state),
):
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming modes.
    The 'model' field maps to orchestrator roles:
    - orchestrator: Auto-routing via frontdoor
    - frontdoor: Direct to frontdoor
    - coder: Direct to coder specialist
    - etc.

    For Aider integration:
    - Configure ~/.aider.conf.yml with openai-api-base: http://localhost:8000/v1
    - Aider will use this endpoint for all LLM calls
    """

    # Extract the last user message as the prompt
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided")

    prompt = user_messages[-1].content

    # Build conversation context from message history
    context_parts = []
    for msg in request.messages[:-1]:  # All but last message
        if msg.role == "system":
            context_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            context_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            context_parts.append(f"Assistant: {msg.content}")
    context = "\n\n".join(context_parts) if context_parts else None

    # Map model to role
    role = request.x_orchestrator_role or (
        Role.FRONTDOOR
        if request.model in ("orchestrator", "gpt-4", "gpt-3.5-turbo", "claude-3")
        else request.model
    )

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Determine if we should use real inference
    # Real mode requires: registry loaded AND at least one backend available
    # For testing without live servers, we always fall back to mock mode
    from src.features import features

    f = features()
    use_real_mode = (
        state.registry is not None and not f.mock_mode  # Respect mock_mode feature flag
    )

    if request.stream:
        # Streaming mode with real orchestration
        async def generate_stream() -> AsyncGenerator[str, None]:
            start_time = time.perf_counter()
            total_tokens = 0
            response_text = ""

            if not use_real_mode:
                # Mock mode fallback
                mock_response = f"[MOCK] Processed via {role}: {prompt[:100]}..."
                for i, char in enumerate(mock_response):
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": char}
                                if i > 0
                                else {"role": "assistant", "content": char},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                response_text = mock_response
            else:
                # Real orchestration with streaming - use singleton LLMPrimitives
                if state.llm_primitives is None:
                    error_msg = "LLM primitives not initialized"
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": error_msg},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    response_text = error_msg
                    primitives = None
                else:
                    primitives = state.llm_primitives

                if primitives:
                    # Build combined context
                    combined_context = prompt
                    if context:
                        combined_context = f"{context}\n\nUser: {prompt}"

                    # Create REPL environment
                    repl = REPLEnvironment(
                        context=combined_context,
                        llm_primitives=primitives,
                        tool_registry=state.tool_registry,
                        script_registry=state.script_registry,
                        role=role,
                    )

                    # Run orchestration loop (simplified for streaming)
                    max_turns = request.max_tokens // 500 if request.max_tokens else 3
                    max_turns = min(max(max_turns, 1), 5)

                    for turn in range(max_turns):
                        repl_state = repl.get_state()
                        root_prompt = build_root_lm_prompt(
                            state=repl_state,
                            original_prompt=prompt,
                            last_output="",
                            last_error="",
                            turn=turn,
                        )

                        try:
                            code = primitives.llm_call(root_prompt, role=role, n_tokens=1024)
                            code = extract_code_from_response(code)
                            code = auto_wrap_final(code)
                        except Exception as e:
                            code = f'FINAL("Error during generation: {e}")'

                        # Execute in REPL
                        result = repl.execute(code)

                        if result.is_final:
                            response_text = result.final_answer or ""
                            break
                        elif result.output:
                            response_text = result.output
                    else:
                        # Max turns reached
                        response_text = response_text or f"[Completed {max_turns} turns]"

                    total_tokens = primitives.total_tokens_generated

                    # Stream the response character by character (OpenAI format)
                    first_chunk = True
                    for char in response_text:
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": char}
                                    if first_chunk
                                    else {"content": char},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        first_chunk = False
                        if request.x_show_routing:
                            chunk["x_role"] = role
                        yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk with finish_reason
            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            if request.x_show_routing:
                final_chunk["x_orchestrator_metadata"] = {
                    "role": role,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "tokens": total_tokens,
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
        # Non-streaming mode with real orchestration
        start_time = time.perf_counter()
        total_tokens = 0

        if not use_real_mode:
            # Mock mode fallback
            response_text = f"[MOCK] Processed via {role}: {prompt[:100]}..."
        else:
            # Real orchestration - use singleton LLMPrimitives
            try:
                if state.llm_primitives is None:
                    raise HTTPException(
                        status_code=503,
                        detail="LLM primitives not initialized",
                    )
                primitives = state.llm_primitives

                combined_context = prompt
                if context:
                    combined_context = f"{context}\n\nUser: {prompt}"

                repl = REPLEnvironment(
                    context=combined_context,
                    llm_primitives=primitives,
                    tool_registry=state.tool_registry,
                    script_registry=state.script_registry,
                    role=role,
                )

                max_turns = request.max_tokens // 500 if request.max_tokens else 3
                max_turns = min(max(max_turns, 1), 5)

                response_text = ""
                for turn in range(max_turns):
                    repl_state = repl.get_state()
                    root_prompt = build_root_lm_prompt(
                        state=repl_state,
                        original_prompt=prompt,
                        last_output="",
                        last_error="",
                        turn=turn,
                    )

                    code = primitives.llm_call(root_prompt, role=role, n_tokens=1024)
                    code = extract_code_from_response(code)
                    code = auto_wrap_final(code)

                    result = repl.execute(code)

                    if result.is_final:
                        response_text = result.final_answer or ""
                        break
                    elif result.output:
                        response_text = result.output

                total_tokens = primitives.total_tokens_generated

            except Exception as e:
                response_text = f"[ERROR] Backend failed: {e}"

        elapsed = time.perf_counter() - start_time

        return OpenAIChatResponse(
            id=chat_id,
            created=created,
            model=request.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=len(prompt) // 4,
                completion_tokens=total_tokens or len(response_text) // 4,
                total_tokens=(len(prompt) // 4) + (total_tokens or len(response_text) // 4),
            ),
            x_orchestrator_metadata={
                "role": role,
                "elapsed_seconds": elapsed,
            }
            if request.x_show_routing
            else None,
        )


@router.get("/models/{model_id}")
async def get_model(model_id: str) -> OpenAIModelInfo:
    """Get info for a specific model."""
    if model_id not in AVAILABLE_ROLES:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return OpenAIModelInfo(id=model_id)

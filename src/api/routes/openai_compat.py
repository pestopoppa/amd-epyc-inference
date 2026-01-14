"""OpenAI-compatible endpoints for the orchestrator API."""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.models import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIModelInfo,
    OpenAIModelsResponse,
    OpenAIUsage,
)

router = APIRouter()

# Available roles/models
AVAILABLE_ROLES = [
    "orchestrator",  # Auto-routing via frontdoor
    "frontdoor",     # Tier A - Root LM
    "coder",         # Tier B - Coder specialist
    "architect",     # Tier B - Architecture specialist
    "worker",        # Tier C - General worker
]


@router.get("/models", response_model=OpenAIModelsResponse)
async def list_models() -> OpenAIModelsResponse:
    """List available models (roles) in OpenAI format."""
    return OpenAIModelsResponse(
        data=[
            OpenAIModelInfo(id=role)
            for role in AVAILABLE_ROLES
        ]
    )


@router.post("/chat/completions", response_model=None)
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


@router.get("/models/{model_id}")
async def get_model(model_id: str) -> OpenAIModelInfo:
    """Get info for a specific model."""
    if model_id not in AVAILABLE_ROLES:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return OpenAIModelInfo(id=model_id)

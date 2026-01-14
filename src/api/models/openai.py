"""OpenAI-compatible models for the orchestrator API."""

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


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

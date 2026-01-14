"""Pydantic models for the orchestrator API."""

from src.api.models.requests import (
    ChatRequest,
    GateRequest,
)
from src.api.models.responses import (
    ChatResponse,
    HealthResponse,
    GateResultModel,
    GatesResponse,
    StatsResponse,
)
from src.api.models.openai import (
    OpenAIMessage,
    OpenAIChatRequest,
    OpenAIChoice,
    OpenAIUsage,
    OpenAIChatResponse,
    OpenAIModelInfo,
    OpenAIModelsResponse,
)
from src.api.models.sessions import (
    SessionInfo,
    SessionListResponse,
    PermissionResponse,
)

__all__ = [
    # Requests
    "ChatRequest",
    "GateRequest",
    # Responses
    "ChatResponse",
    "HealthResponse",
    "GateResultModel",
    "GatesResponse",
    "StatsResponse",
    # OpenAI
    "OpenAIMessage",
    "OpenAIChatRequest",
    "OpenAIChoice",
    "OpenAIUsage",
    "OpenAIChatResponse",
    "OpenAIModelInfo",
    "OpenAIModelsResponse",
    # Sessions
    "SessionInfo",
    "SessionListResponse",
    "PermissionResponse",
]

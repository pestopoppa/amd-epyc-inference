"""Pydantic models for the orchestrator API."""

from __future__ import annotations

from src.api.models.requests import (
    ChatRequest,
    GateRequest,
    RewardRequest,
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
    CheckpointInfo,
    FindingCreateRequest,
    FindingInfo,
    PermissionResponse,
    SessionCreateRequest,
    SessionInfo,
    SessionListResponse,
    SessionResumeResponse,
)

__all__ = [
    # Requests
    "ChatRequest",
    "GateRequest",
    "RewardRequest",
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
    "CheckpointInfo",
    "FindingCreateRequest",
    "FindingInfo",
    "PermissionResponse",
    "SessionCreateRequest",
    "SessionInfo",
    "SessionListResponse",
    "SessionResumeResponse",
]

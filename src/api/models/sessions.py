"""Session management models for the orchestrator API."""

from pydantic import BaseModel


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

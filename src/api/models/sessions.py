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
    # New fields for session persistence
    project: str | None = None
    status: str = "active"
    tags: list[str] = []
    resume_count: int = 0
    summary: str | None = None
    last_topic: str | None = None


class SessionListResponse(BaseModel):
    """Response for session list."""

    sessions: list[SessionInfo]


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""

    name: str | None = None
    project: str | None = None
    working_directory: str | None = None


class SessionResumeResponse(BaseModel):
    """Response for session resume with context."""

    status: str
    session_id: str
    message_count: int
    # Context injection for LLM
    context_summary: str | None = None
    key_findings: list[str] = []
    warnings: list[str] = []
    document_changes: list[dict] = []


class FindingInfo(BaseModel):
    """Key finding information."""

    id: str
    session_id: str
    content: str
    source: str
    created_at: str
    confidence: float = 1.0
    confirmed: bool = False
    tags: list[str] = []


class FindingCreateRequest(BaseModel):
    """Request to create a finding."""

    content: str
    tags: list[str] = []


class CheckpointInfo(BaseModel):
    """Checkpoint information."""

    id: str
    session_id: str
    created_at: str
    message_count: int
    trigger: str


class PermissionResponse(BaseModel):
    """Response for permission requests."""

    request_id: str
    approved: bool
    tool: str

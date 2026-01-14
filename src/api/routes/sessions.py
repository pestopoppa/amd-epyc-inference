"""Session management endpoints (Claude Code parity)."""

import time
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.models import SessionInfo, SessionListResponse, PermissionResponse

router = APIRouter()

# In-memory session store (would be persisted in production)
_sessions: dict[str, dict] = {}
_pending_permissions: dict[str, dict] = {}


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """List available sessions."""
    sessions = [
        SessionInfo(
            id=sid,
            name=data.get("name"),
            created_at=data.get("created_at", ""),
            last_active=data.get("last_active", ""),
            message_count=data.get("message_count", 0),
            working_directory=data.get("working_directory"),
        )
        for sid, data in _sessions.items()
    ]
    return SessionListResponse(sessions=sessions)


@router.post("/sessions/{session_id}/resume")
async def resume_session(session_id: str) -> dict[str, Any]:
    """Resume a previous session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    session = _sessions[session_id]
    session["last_active"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return {
        "status": "resumed",
        "session_id": session_id,
        "message_count": session.get("message_count", 0),
    }


@router.post("/sessions/current/rename")
async def rename_session(name: str, session_id: str | None = None) -> dict[str, str]:
    """Rename current or specified session."""
    sid = session_id or "current"
    if sid not in _sessions:
        # Create new session with this name
        _sessions[sid] = {
            "name": name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "last_active": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message_count": 0,
        }
    else:
        _sessions[sid]["name"] = name

    return {"status": "renamed", "session_id": sid, "name": name}


@router.post("/permission/{request_id}")
async def respond_to_permission(request_id: str, approved: bool) -> PermissionResponse:
    """Approve or reject a pending tool execution.

    This is used for interactive permission flows in Normal mode.
    """
    if request_id not in _pending_permissions:
        raise HTTPException(status_code=404, detail=f"Permission request '{request_id}' not found")

    perm = _pending_permissions.pop(request_id)

    return PermissionResponse(
        request_id=request_id,
        approved=approved,
        tool=perm.get("tool", "unknown"),
    )


@router.get("/permission/pending")
async def list_pending_permissions() -> list[dict]:
    """List pending permission requests."""
    return [
        {"id": pid, **data}
        for pid, data in _pending_permissions.items()
    ]

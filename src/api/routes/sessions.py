"""Session management endpoints with persistent storage.

Replaces in-memory session store with SQLiteSessionStore for:
- Conversation continuity across restarts
- Document caching and change detection
- Key findings persistence
- Crash recovery via checkpoints
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Depends

from src.api.models import (
    CheckpointInfo,
    FindingCreateRequest,
    FindingInfo,
    PermissionResponse,
    SessionCreateRequest,
    SessionInfo,
    SessionListResponse,
    SessionResumeResponse,
)
from src.session import SQLiteSessionStore, Session, Finding, FindingSource
from src.api.dependencies import dep_session_store

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory pending permissions (not persisted - short-lived)
_pending_permissions: dict[str, dict] = {}


def _session_to_info(session: Session) -> SessionInfo:
    """Convert Session model to SessionInfo response."""
    return SessionInfo(
        id=session.id,
        name=session.name,
        created_at=session.created_at.isoformat(),
        last_active=session.last_active.isoformat(),
        message_count=session.message_count,
        working_directory=session.working_directory,
        project=session.project,
        status=session.status.value,
        tags=session.tags,
        resume_count=session.resume_count,
        summary=session.summary,
        last_topic=session.last_topic,
    )


# =========================================================================
# Session CRUD
# =========================================================================


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    status: str | None = Query(None, description="Filter by status"),
    project: str | None = Query(None, description="Filter by project"),
    limit: int = Query(50, description="Maximum results"),
    offset: int = Query(0, description="Skip first N results"),
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> SessionListResponse:
    """List available sessions with optional filtering."""

    # Build filter
    where = {}
    if status:
        where["status"] = status
    if project:
        where["project"] = project

    sessions = store.list_sessions(
        where=where if where else None,
        limit=limit,
        offset=offset,
    )

    return SessionListResponse(sessions=[_session_to_info(s) for s in sessions])


@router.post("/sessions", response_model=SessionInfo)
async def create_session(
    request: SessionCreateRequest,
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> SessionInfo:
    """Create a new session."""

    session = Session.create(
        name=request.name,
        project=request.project,
        working_directory=request.working_directory,
    )

    store.create_session(session)
    logger.info(f"Created session {session.id[:8]}")

    return _session_to_info(session)


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str,
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> SessionInfo:
    """Get session details."""
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return _session_to_info(session)


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> dict[str, str]:
    """Delete a session and all associated data."""
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return {"status": "deleted", "session_id": session_id}


# =========================================================================
# Session resume (with context injection)
# =========================================================================


@router.post("/sessions/{session_id}/resume", response_model=SessionResumeResponse)
async def resume_session(
    session_id: str,
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> SessionResumeResponse:
    """Resume a previous session with full context.

    Returns context injection payload for the LLM including:
    - Key findings from previous session
    - Document change warnings
    - Session summary
    """
    # Build full resume context
    context = store.build_resume_context(session_id)
    if not context:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    # Update session activity and fork task_id for MemRL
    session = context.session
    session.update_activity()
    new_task_id = session.fork_task_id()
    store.update_session(session)

    logger.info(
        f"Resumed session {session_id[:8]}, "
        f"resume_count={session.resume_count}, task_id={new_task_id[:8]}"
    )

    return SessionResumeResponse(
        status="resumed",
        session_id=session_id,
        message_count=session.message_count,
        context_summary=context.context_summary,
        key_findings=[f.content for f in context.findings],
        warnings=context.warnings,
        document_changes=[
            {
                "file_path": c.file_path,
                "changed": c.new_hash is not None,
                "exists": c.exists,
            }
            for c in context.document_changes
        ],
    )


# =========================================================================
# Session rename (backward compatibility)
# =========================================================================


@router.post("/sessions/current/rename")
async def rename_session(
    name: str,
    session_id: str | None = None,
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> dict[str, str]:
    """Rename current or specified session."""
    if session_id:
        session = store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        session.name = name
        session.update_activity()
        store.update_session(session)
    else:
        # Create new session with this name (backward compatibility)
        session = Session.create(name=name)
        store.create_session(session)
        session_id = session.id

    return {"status": "renamed", "session_id": session_id, "name": name}


# =========================================================================
# Findings
# =========================================================================


@router.get("/sessions/{session_id}/findings", response_model=list[FindingInfo])
async def get_findings(
    session_id: str,
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> list[FindingInfo]:
    """Get key findings for a session."""
    # Verify session exists
    if not store.get_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    findings = store.get_findings(session_id)

    return [
        FindingInfo(
            id=f.id,
            session_id=f.session_id,
            content=f.content,
            source=f.source.value,
            created_at=f.created_at.isoformat(),
            confidence=f.confidence,
            confirmed=f.confirmed,
            tags=f.tags,
        )
        for f in findings
    ]


@router.post("/sessions/{session_id}/findings", response_model=FindingInfo)
async def add_finding(
    session_id: str,
    request: FindingCreateRequest,
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> FindingInfo:
    """Add a key finding to a session."""
    # Verify session exists
    if not store.get_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    finding = Finding(
        id=str(uuid.uuid4()),
        session_id=session_id,
        content=request.content,
        source=FindingSource.USER_MARKED,
        created_at=datetime.utcnow(),
        confidence=1.0,
        confirmed=True,
        tags=request.tags,
    )

    store.add_finding(finding)
    logger.info(f"Added finding to session {session_id[:8]}: {request.content[:50]}")

    return FindingInfo(
        id=finding.id,
        session_id=finding.session_id,
        content=finding.content,
        source=finding.source.value,
        created_at=finding.created_at.isoformat(),
        confidence=finding.confidence,
        confirmed=finding.confirmed,
        tags=finding.tags,
    )


@router.delete("/sessions/{session_id}/findings/{finding_id}")
async def delete_finding(
    session_id: str,
    finding_id: str,
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> dict[str, str]:
    """Delete a finding."""
    if not store.delete_finding(finding_id):
        raise HTTPException(status_code=404, detail=f"Finding '{finding_id}' not found")

    return {"status": "deleted", "finding_id": finding_id}


# =========================================================================
# Search
# =========================================================================


@router.get("/sessions/search")
async def search_sessions(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum results"),
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> list[SessionInfo]:
    """Search sessions by name, summary, or last topic."""
    sessions = store.search_sessions(q, limit=limit)
    return [_session_to_info(s) for s in sessions]


@router.get("/sessions/{session_id}/findings/search")
async def search_findings(
    session_id: str,
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum results"),
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> list[FindingInfo]:
    """Search findings in a session."""
    findings = store.search_findings(q, session_id=session_id, limit=limit)

    return [
        FindingInfo(
            id=f.id,
            session_id=f.session_id,
            content=f.content,
            source=f.source.value,
            created_at=f.created_at.isoformat(),
            confidence=f.confidence,
            confirmed=f.confirmed,
            tags=f.tags,
        )
        for f in findings
    ]


# =========================================================================
# Tags
# =========================================================================


@router.post("/sessions/{session_id}/tags/{tag}")
async def add_tag(
    session_id: str,
    tag: str,
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> dict[str, Any]:
    """Add a tag to a session."""
    if not store.get_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    added = store.add_tag(session_id, tag)
    return {"status": "added" if added else "exists", "session_id": session_id, "tag": tag}


@router.delete("/sessions/{session_id}/tags/{tag}")
async def remove_tag(
    session_id: str,
    tag: str,
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> dict[str, Any]:
    """Remove a tag from a session."""
    if not store.remove_tag(session_id, tag):
        raise HTTPException(status_code=404, detail=f"Tag '{tag}' not found")

    return {"status": "removed", "session_id": session_id, "tag": tag}


# =========================================================================
# Checkpoints
# =========================================================================


@router.get("/sessions/{session_id}/checkpoints", response_model=list[CheckpointInfo])
async def get_checkpoints(
    session_id: str,
    limit: int = Query(10, description="Maximum results"),
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> list[CheckpointInfo]:
    """Get recent checkpoints for a session."""
    if not store.get_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    checkpoints = store.get_checkpoints(session_id, limit=limit)

    return [
        CheckpointInfo(
            id=c.id,
            session_id=c.session_id,
            created_at=c.created_at.isoformat(),
            message_count=c.message_count,
            trigger=c.trigger,
        )
        for c in checkpoints
    ]


# =========================================================================
# Archive
# =========================================================================


@router.post("/sessions/{session_id}/archive")
async def archive_session(
    session_id: str,
    store: SQLiteSessionStore = Depends(dep_session_store),
) -> dict[str, str]:
    """Archive a session to cold storage."""
    if not store.archive_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return {"status": "archived", "session_id": session_id}


# =========================================================================
# Permissions (kept in-memory, not persisted)
# =========================================================================


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
    return [{"id": pid, **data} for pid, data in _pending_permissions.items()]

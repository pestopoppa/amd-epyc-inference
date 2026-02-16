#!/usr/bin/env python3
"""Unit tests for src/session/models.py."""

from datetime import datetime

import src.session.models as session_models

from src.session.models import (
    Checkpoint,
    Finding,
    FindingSource,
    Session,
    SessionDocument,
    SessionStatus,
)


class TestSessionToFromDict:
    """Test Session serialization."""

    def test_session_to_dict(self):
        """Test Session.to_dict() serialization."""
        session = Session.create(name="test session", project="test-project")
        session_dict = session.to_dict()

        assert session_dict["id"] == session.id
        assert session_dict["name"] == "test session"
        assert session_dict["project"] == "test-project"
        assert "created_at" in session_dict
        assert "task_id" in session_dict

    def test_session_from_dict(self):
        """Test Session.from_dict() deserialization."""
        session = Session.create(name="test", project="proj")
        session_dict = session.to_dict()

        restored = Session.from_dict(session_dict)

        assert restored.id == session.id
        assert restored.name == session.name
        assert restored.project == session.project
        assert restored.task_id == session.task_id

    def test_session_roundtrip(self):
        """Test Session.to_dict() and from_dict() roundtrip."""
        original = Session.create(name="roundtrip", working_directory="/test")
        original.message_count = 42
        original.summary = "Test summary"

        # Roundtrip
        restored = Session.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.message_count == 42
        assert restored.summary == "Test summary"


class TestFindingModel:
    """Test Finding dataclass."""

    def test_finding_creation_all_fields(self):
        """Test Finding creation with all fields."""
        finding = Finding(
            id="f1",
            session_id="s1",
            content="Important insight",
            source=FindingSource.USER_MARKED,
            created_at=datetime.utcnow(),
            confidence=1.0,
            confirmed=True,
            tags=["important"],
            source_file="test.pdf",
            source_page=42,
        )

        assert finding.id == "f1"
        assert finding.content == "Important insight"
        assert finding.source == FindingSource.USER_MARKED
        assert finding.confidence == 1.0
        assert finding.confirmed is True
        assert finding.tags == ["important"]
        assert finding.source_page == 42

    def test_finding_to_dict(self):
        """Test Finding.to_dict() includes all fields."""
        finding = Finding(
            id="f1",
            session_id="s1",
            content="Test",
            source=FindingSource.LLM_EXTRACTED,
            created_at=datetime.utcnow(),
            confidence=0.8,
        )
        finding_dict = finding.to_dict()

        assert finding_dict["id"] == "f1"
        assert finding_dict["content"] == "Test"
        assert finding_dict["source"] == "llm_extracted"
        assert finding_dict["confidence"] == 0.8

    def test_finding_from_dict(self):
        """Test Finding.from_dict() deserialization."""
        data = {
            "id": "f1",
            "session_id": "s1",
            "content": "Test",
            "source": "user_marked",
            "created_at": datetime.utcnow().isoformat(),
            "confidence": 0.9,
            "confirmed": False,
            "tags": ["tag1"],
            "source_file": None,
            "source_section_id": None,
            "source_page": None,
            "source_turn": None,
        }

        finding = Finding.from_dict(data)

        assert finding.id == "f1"
        assert finding.content == "Test"
        assert finding.source == FindingSource.USER_MARKED
        assert finding.confidence == 0.9


class TestCheckpointSerialization:
    """Test Checkpoint serialization."""

    def test_checkpoint_to_dict(self):
        """Test Checkpoint.to_dict() serialization."""
        checkpoint = Checkpoint(
            id="cp1",
            session_id="s1",
            created_at=datetime.utcnow(),
            context_hash="sha256:abc123",
            artifacts={"key": "value"},
            execution_count=10,
            exploration_calls=5,
            message_count=20,
            trigger="turns",
        )

        cp_dict = checkpoint.to_dict()

        assert cp_dict["id"] == "cp1"
        assert cp_dict["context_hash"] == "sha256:abc123"
        assert cp_dict["execution_count"] == 10
        assert cp_dict["trigger"] == "turns"

    def test_checkpoint_from_dict(self):
        """Test Checkpoint.from_dict() deserialization."""
        data = {
            "id": "cp1",
            "session_id": "s1",
            "created_at": datetime.utcnow().isoformat(),
            "context_hash": "hash",
            "artifacts": {},
            "execution_count": 5,
            "exploration_calls": 2,
            "message_count": 10,
            "trigger": "explicit",
        }

        checkpoint = Checkpoint.from_dict(data)

        assert checkpoint.id == "cp1"
        assert checkpoint.execution_count == 5
        assert checkpoint.trigger == "explicit"


class TestSessionDocument:
    """Test SessionDocument model."""

    def test_document_change_detection(self, tmp_path):
        """Test compute_file_hash for change detection."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")

        # Compute hash
        hash1 = SessionDocument.compute_file_hash(test_file)
        assert hash1.startswith("sha256:")

        # Same content should have same hash
        hash2 = SessionDocument.compute_file_hash(test_file)
        assert hash1 == hash2

        # Different content should have different hash
        test_file.write_text("modified content")
        hash3 = SessionDocument.compute_file_hash(test_file)
        assert hash3 != hash1


class TestSessionStatus:
    """Test SessionStatus enum."""

    def test_session_status_values(self):
        """Test SessionStatus enum values."""
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.IDLE == "idle"
        assert SessionStatus.STALE == "stale"
        assert SessionStatus.ARCHIVED == "archived"

    def test_session_status_update(self):
        """Test session status updates based on activity."""
        session = Session.create()

        # New session should be active
        assert session.status == SessionStatus.ACTIVE

        # Manually set to archived
        session.status = SessionStatus.ARCHIVED
        session.update_activity()

        # Archived status should not auto-change
        assert session.status == SessionStatus.ARCHIVED

    def test_session_status_uses_configurable_thresholds(self, monkeypatch):
        """Status transitions should respect centralized lifecycle thresholds."""
        monkeypatch.setattr(session_models, "_ACTIVE_TO_IDLE_HOURS", 0.5)
        monkeypatch.setattr(session_models, "_IDLE_TO_STALE_HOURS", 2.0)
        session = Session.create()

        session.last_active = datetime.utcnow()
        session._update_status()
        assert session.status == SessionStatus.ACTIVE

        session.last_active = datetime.fromtimestamp(datetime.utcnow().timestamp() - 3600)
        session._update_status()
        assert session.status == SessionStatus.IDLE

        session.last_active = datetime.fromtimestamp(datetime.utcnow().timestamp() - 3 * 3600)
        session._update_status()
        assert session.status == SessionStatus.STALE

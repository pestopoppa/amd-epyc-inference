"""Extended unit tests for SQLiteSessionStore.

Tests the where filter operators, checkpoint operations, embeddings,
and other uncovered functionality in src/session/sqlite_store.py.
"""

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from src.session import Session, Finding, FindingSource, Checkpoint
from src.session.sqlite_store import SQLiteSessionStore


@pytest.fixture
def temp_store(tmp_path):
    """Create a temporary SQLite store for testing."""
    db_path = tmp_path / "test.db"
    embeddings_path = tmp_path / "embeddings.npy"
    store = SQLiteSessionStore(
        db_path=db_path,
        embeddings_path=embeddings_path,
    )
    yield store
    store.close()


@pytest.fixture
def sample_session():
    """Create a sample session for testing."""
    return Session.create(
        name="Test Session",
        project="test-project",
        working_directory="/tmp",
    )


class TestWhereFilters:
    """Test _apply_where_filter with various operators."""

    def test_in_operator(self, temp_store, sample_session):
        """Test $in operator for filtering."""
        # Create multiple sessions with different statuses
        session1 = Session.create(name="S1", project="p1")
        session2 = Session.create(name="S2", project="p2")
        session3 = Session.create(name="S3", project="p1")

        temp_store.create_session(session1)
        temp_store.create_session(session2)
        temp_store.create_session(session3)

        # Filter with $in operator
        results = temp_store.list_sessions(where={"project": {"$in": ["p1", "p2"]}})
        assert len(results) >= 2

    def test_gte_operator(self, temp_store):
        """Test $gte (greater than or equal) operator."""
        # Create sessions with different message counts
        s1 = Session.create(name="S1")
        s1.message_count = 10
        s2 = Session.create(name="S2")
        s2.message_count = 20
        s3 = Session.create(name="S3")
        s3.message_count = 5

        temp_store.create_session(s1)
        temp_store.create_session(s2)
        temp_store.create_session(s3)

        # Filter for sessions with >= 10 messages
        results = temp_store.list_sessions(where={"message_count": {"$gte": 10}})
        assert len(results) >= 2
        for session in results:
            assert session.message_count >= 10

    def test_lte_operator(self, temp_store):
        """Test $lte (less than or equal) operator."""
        s1 = Session.create(name="S1")
        s1.message_count = 10
        s2 = Session.create(name="S2")
        s2.message_count = 5

        temp_store.create_session(s1)
        temp_store.create_session(s2)

        # Filter for sessions with <= 10 messages
        results = temp_store.list_sessions(where={"message_count": {"$lte": 10}})
        assert len(results) >= 2

    def test_ne_operator(self, temp_store):
        """Test $ne (not equal) operator."""
        s1 = Session.create(name="S1", project="exclude-me")
        s2 = Session.create(name="S2", project="keep-me")

        temp_store.create_session(s1)
        temp_store.create_session(s2)

        # Filter for projects != "exclude-me"
        results = temp_store.list_sessions(where={"project": {"$ne": "exclude-me"}})
        for session in results:
            assert session.project != "exclude-me"

    def test_gt_operator(self, temp_store):
        """Test $gt (greater than) operator."""
        s1 = Session.create(name="S1")
        s1.message_count = 15
        s2 = Session.create(name="S2")
        s2.message_count = 5

        temp_store.create_session(s1)
        temp_store.create_session(s2)

        # Filter for sessions with > 10 messages
        results = temp_store.list_sessions(where={"message_count": {"$gt": 10}})
        for session in results:
            assert session.message_count > 10

    def test_lt_operator(self, temp_store):
        """Test $lt (less than) operator."""
        s1 = Session.create(name="S1")
        s1.message_count = 3
        s2 = Session.create(name="S2")
        s2.message_count = 15

        temp_store.create_session(s1)
        temp_store.create_session(s2)

        # Filter for sessions with < 10 messages
        results = temp_store.list_sessions(where={"message_count": {"$lt": 10}})
        for session in results:
            assert session.message_count < 10


class TestCheckpoints:
    """Test checkpoint save and restore cycle."""

    def test_checkpoint_save_and_retrieve(self, temp_store, sample_session):
        """Test saving and retrieving a checkpoint."""
        temp_store.create_session(sample_session)

        # Create a checkpoint
        checkpoint = Checkpoint(
            id=str(uuid.uuid4()),
            session_id=sample_session.id,
            created_at=datetime.now(timezone.utc),
            context_hash="abc123",
            artifacts={"key": "value", "data": [1, 2, 3]},
            execution_count=5,
            exploration_calls=2,
            message_count=10,
            trigger="manual",
            user_globals={"total": 42},
            variable_lineage={"total": {"role": "frontdoor"}},
            skipped_user_globals=["tmp_fn"],
        )

        # Save it
        temp_store.save_checkpoint(checkpoint)

        # Retrieve it
        retrieved = temp_store.get_latest_checkpoint(sample_session.id)
        assert retrieved is not None
        assert retrieved.id == checkpoint.id
        assert retrieved.context_hash == "abc123"
        assert retrieved.artifacts == {"key": "value", "data": [1, 2, 3]}
        assert retrieved.execution_count == 5
        assert retrieved.trigger == "manual"
        assert retrieved.user_globals == {"total": 42}
        assert retrieved.variable_lineage["total"]["role"] == "frontdoor"
        assert retrieved.skipped_user_globals == ["tmp_fn"]

    def test_get_checkpoints_with_limit(self, temp_store, sample_session):
        """Test retrieving multiple checkpoints with limit."""
        temp_store.create_session(sample_session)

        # Create multiple checkpoints
        for i in range(5):
            checkpoint = Checkpoint(
                id=str(uuid.uuid4()),
                session_id=sample_session.id,
                created_at=datetime.now(timezone.utc),
                context_hash=f"hash-{i}",
                artifacts={},
                execution_count=i,
                exploration_calls=0,
                message_count=i * 2,
                trigger="auto",
            )
            temp_store.save_checkpoint(checkpoint)

        # Get with limit
        checkpoints = temp_store.get_checkpoints(sample_session.id, limit=3)
        assert len(checkpoints) == 3


class TestFindingCRUD:
    """Test finding create, read, update, delete operations."""

    def test_finding_lifecycle(self, temp_store, sample_session):
        """Test complete CRUD cycle for findings."""
        temp_store.create_session(sample_session)

        # Create
        finding = Finding(
            id=str(uuid.uuid4()),
            session_id=sample_session.id,
            content="Important discovery",
            source=FindingSource.USER_MARKED,
            created_at=datetime.now(timezone.utc),
            confidence=1.0,
            confirmed=True,
            tags=["critical", "verified"],
        )
        temp_store.add_finding(finding)

        # Read
        findings = temp_store.get_findings(sample_session.id)
        assert len(findings) == 1
        assert findings[0].content == "Important discovery"
        assert findings[0].tags == ["critical", "verified"]

        # Update
        finding.content = "Updated discovery"
        finding.confidence = 0.9
        temp_store.update_finding(finding)

        updated = temp_store.get_findings(sample_session.id)[0]
        assert updated.content == "Updated discovery"
        assert updated.confidence == 0.9

        # Delete
        result = temp_store.delete_finding(finding.id)
        assert result is True

        findings_after = temp_store.get_findings(sample_session.id)
        assert len(findings_after) == 0


class TestTagManagement:
    """Test tag operations."""

    def test_add_tag_to_session(self, temp_store, sample_session):
        """Test adding tags to a session."""
        temp_store.create_session(sample_session)

        # Add tag
        result = temp_store.add_tag(sample_session.id, "urgent")
        assert result is True

        # Verify tag was added
        session = temp_store.get_session(sample_session.id)
        assert "urgent" in session.tags

    def test_remove_tag_from_session(self, temp_store, sample_session):
        """Test removing tags from a session."""
        temp_store.create_session(sample_session)
        temp_store.add_tag(sample_session.id, "temp-tag")

        # Remove tag
        result = temp_store.remove_tag(sample_session.id, "temp-tag")
        assert result is True

        # Verify tag was removed
        session = temp_store.get_session(sample_session.id)
        assert "temp-tag" not in session.tags

    def test_get_sessions_by_tag(self, temp_store):
        """Test retrieving sessions by tag."""
        s1 = Session.create(name="S1")
        s2 = Session.create(name="S2")

        temp_store.create_session(s1)
        temp_store.create_session(s2)

        temp_store.add_tag(s1.id, "feature-x")
        temp_store.add_tag(s2.id, "feature-x")

        # Get sessions by tag
        sessions = temp_store.get_sessions_by_tag("feature-x")
        assert len(sessions) == 2


class TestSearchSessions:
    """Test full-text search functionality."""

    def test_search_sessions_by_name(self, temp_store):
        """Test searching sessions by name."""
        s1 = Session.create(name="Database Optimization")
        s1.summary = "Working on query performance"
        s2 = Session.create(name="Frontend Refactor")
        s2.summary = "UI component cleanup"

        temp_store.create_session(s1)
        temp_store.create_session(s2)

        # Search by name
        results = temp_store.search_sessions("Database")
        assert len(results) >= 1
        assert any(s.name == "Database Optimization" for s in results)

    def test_search_findings(self, temp_store, sample_session):
        """Test searching findings by content."""
        temp_store.create_session(sample_session)

        finding = Finding(
            id=str(uuid.uuid4()),
            session_id=sample_session.id,
            content="The cache miss ratio is 42%",
            source=FindingSource.USER_MARKED,
            created_at=datetime.now(timezone.utc),
        )
        temp_store.add_finding(finding)

        # Search for findings
        results = temp_store.search_findings("cache miss", session_id=sample_session.id)
        assert len(results) >= 1
        assert results[0].content == "The cache miss ratio is 42%"


class TestEmbeddings:
    """Test embedding storage and similarity search."""

    def test_grow_embeddings_array(self, temp_store):
        """Test that embeddings array grows when full."""
        # Get initial size
        initial_size = len(temp_store._embeddings)

        # Fill the array beyond capacity
        session_id = str(uuid.uuid4())
        for i in range(initial_size + 10):
            embedding = np.random.randn(temp_store.embedding_dim).astype(np.float32)
            temp_store.store_embedding(session_id, embedding)

        # Verify array grew
        new_size = len(temp_store._embeddings)
        assert new_size > initial_size

    def test_store_and_search_embeddings(self, temp_store):
        """Test storing embeddings and cosine similarity search."""
        # Create some sessions
        s1 = Session.create(name="S1")
        s2 = Session.create(name="S2")
        temp_store.create_session(s1)
        temp_store.create_session(s2)

        # Store embeddings
        emb1 = np.random.randn(temp_store.embedding_dim).astype(np.float32)
        emb2 = np.random.randn(temp_store.embedding_dim).astype(np.float32)

        idx1 = temp_store.store_embedding(s1.id, emb1, content_type="session")
        idx2 = temp_store.store_embedding(s2.id, emb2, content_type="session")

        assert idx1 >= 0
        assert idx2 >= 0

        # Search by embedding (should find similar ones)
        query_emb = emb1 + 0.01 * np.random.randn(temp_store.embedding_dim).astype(np.float32)
        results = temp_store.search_by_embedding(query_emb, content_type="session", limit=5)

        assert isinstance(results, list)
        # Results are tuples of (session_id, similarity)
        if results:
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 2

    def test_load_embeddings_error_recovery(self, tmp_path):
        """Test that corrupt embeddings file is recreated."""
        db_path = tmp_path / "test.db"
        embeddings_path = tmp_path / "embeddings.npy"

        # Create a corrupt embeddings file
        embeddings_path.write_bytes(b"corrupt data")

        # Should recreate the file
        store = SQLiteSessionStore(db_path=db_path, embeddings_path=embeddings_path)
        assert store._embeddings is not None
        assert len(store._embeddings) > 0
        store.close()

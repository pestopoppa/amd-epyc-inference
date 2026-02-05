"""Unit tests for DocumentCache.

Tests hash-based cache hit/miss logic and statistics in src/session/document_cache.py.
"""

import uuid

import pytest

from src.session.document_cache import DocumentCache
from src.models.document import DocumentPreprocessResult, Section


@pytest.fixture
def temp_session_id():
    """Generate a temporary session ID."""
    return str(uuid.uuid4())


@pytest.fixture
def temp_cache(temp_session_id, tmp_path, monkeypatch):
    """Create a temporary document cache."""
    # Monkeypatch SESSION_STATE_DIR to use tmp_path
    import src.session.document_cache as dc_module

    monkeypatch.setattr(dc_module, "SESSION_STATE_DIR", tmp_path / "sessions" / "state")

    cache = DocumentCache(temp_session_id, session_store=None)
    yield cache


@pytest.fixture
def sample_preprocess_result():
    """Create a sample preprocessing result."""
    return DocumentPreprocessResult(
        original_path="/tmp/test.pdf",
        total_pages=5,
        sections=[
            Section(
                id="sec-1",
                title="Introduction",
                level=1,
                page_start=1,
                page_end=2,
                content="This is the introduction section.",
                figure_ids=[],
            ),
        ],
        figures=[],
        failed_pages=[],
        processing_time=1.5,
    )


class TestDocumentCache:
    """Test DocumentCache functionality."""

    def test_cache_initialization(self, temp_cache):
        """Test that cache database is initialized properly."""
        assert temp_cache.cache_db_path.exists()
        assert temp_cache.cache_db_path.suffix == ".db"

    def test_cache_miss(self, temp_cache, tmp_path):
        """Test cache miss when document not previously cached."""
        # Create a test file
        test_file = tmp_path / "test_doc.txt"
        test_file.write_text("Test content")

        # Attempt to get from cache (should be None)
        result = temp_cache.get_cached(test_file)
        assert result is None

    def test_cache_hit(self, temp_cache, tmp_path, sample_preprocess_result):
        """Test cache hit when document previously cached and unchanged."""
        # Create a test file
        test_file = tmp_path / "test_doc.txt"
        test_file.write_text("Test content for caching")

        # Cache the result
        file_hash = temp_cache.cache_result(
            test_file,
            sample_preprocess_result,
            track_in_session=False,
        )
        assert file_hash.startswith("sha256:")

        # Now retrieve from cache (should hit)
        cached_result = temp_cache.get_cached(test_file)
        assert cached_result is not None
        assert cached_result.total_pages == 5
        assert len(cached_result.sections) == 1
        assert cached_result.sections[0].title == "Introduction"

    def test_cache_invalidation_on_file_change(
        self, temp_cache, tmp_path, sample_preprocess_result
    ):
        """Test that cache is invalidated when file changes."""
        # Create and cache a file
        test_file = tmp_path / "mutable_doc.txt"
        test_file.write_text("Original content")

        temp_cache.cache_result(test_file, sample_preprocess_result, track_in_session=False)

        # Verify cache hit
        assert temp_cache.get_cached(test_file) is not None

        # Modify the file
        test_file.write_text("Modified content with different hash")

        # Cache should miss now (hash differs)
        cached_result = temp_cache.get_cached(test_file)
        assert cached_result is None

    def test_cache_statistics(self, temp_cache, tmp_path, sample_preprocess_result):
        """Test cache statistics tracking."""
        # Initially empty
        stats = temp_cache.get_stats()
        initial_files = stats["total_files"]
        initial_pages = stats["total_pages"]

        # Cache a document
        test_file = tmp_path / "stats_test.txt"
        test_file.write_text("Content for stats")
        temp_cache.cache_result(test_file, sample_preprocess_result, track_in_session=False)

        # Check updated stats
        stats = temp_cache.get_stats()
        assert stats["total_files"] == initial_files + 1
        assert stats["total_pages"] == initial_pages + 5  # sample has 5 pages
        assert stats["cache_size_bytes"] > 0

    def test_list_cached_files(self, temp_cache, tmp_path, sample_preprocess_result):
        """Test listing all cached files."""
        # Cache multiple files
        for i in range(3):
            test_file = tmp_path / f"doc_{i}.txt"
            test_file.write_text(f"Content {i}")
            temp_cache.cache_result(test_file, sample_preprocess_result, track_in_session=False)

        # List cached files
        cached_files = temp_cache.list_cached_files()
        assert len(cached_files) >= 3

        # Verify structure
        for entry in cached_files:
            assert "file_path" in entry
            assert "file_hash" in entry
            assert "cached_at" in entry
            assert "total_pages" in entry

    def test_invalidate_specific_file(self, temp_cache, tmp_path, sample_preprocess_result):
        """Test invalidating cache for a specific file."""
        test_file = tmp_path / "invalidate_test.txt"
        test_file.write_text("Content to invalidate")

        # Cache it
        temp_cache.cache_result(test_file, sample_preprocess_result, track_in_session=False)
        assert temp_cache.get_cached(test_file) is not None

        # Invalidate
        result = temp_cache.invalidate(test_file)
        assert result is True

        # Should be cache miss now
        assert temp_cache.get_cached(test_file) is None

    def test_invalidate_all(self, temp_cache, tmp_path, sample_preprocess_result):
        """Test clearing all cached results."""
        # Cache multiple files
        for i in range(3):
            test_file = tmp_path / f"clear_{i}.txt"
            test_file.write_text(f"Content {i}")
            temp_cache.cache_result(test_file, sample_preprocess_result, track_in_session=False)

        # Clear all
        count = temp_cache.invalidate_all()
        assert count >= 3

        # Stats should show empty cache
        stats = temp_cache.get_stats()
        assert stats["total_files"] == 0

    def test_check_changes(self, temp_cache, tmp_path, sample_preprocess_result):
        """Test checking for document changes."""
        # Cache a file
        test_file = tmp_path / "change_check.txt"
        test_file.write_text("Original")
        temp_cache.cache_result(test_file, sample_preprocess_result, track_in_session=False)

        # Check changes (should be unchanged)
        changes = temp_cache.check_changes()
        assert isinstance(changes, list)
        for change in changes:
            if change["file_path"] == str(test_file):
                assert change["changed"] is False
                assert change["exists"] is True

        # Modify the file
        test_file.write_text("Modified")

        # Check again (should detect change)
        changes = temp_cache.check_changes()
        for change in changes:
            if change["file_path"] == str(test_file):
                assert change["changed"] is True
                assert change["exists"] is True

    def test_missing_file_detected(self, temp_cache, tmp_path, sample_preprocess_result):
        """Test that missing source files are detected."""
        # Cache a file
        test_file = tmp_path / "will_delete.txt"
        test_file.write_text("Content")
        temp_cache.cache_result(test_file, sample_preprocess_result, track_in_session=False)

        # Delete the file
        test_file.unlink()

        # Check changes (should detect missing file)
        changes = temp_cache.check_changes()
        for change in changes:
            if "will_delete.txt" in change["file_path"]:
                assert change["exists"] is False
                assert change["changed"] is True
                assert change["current_hash"] is None

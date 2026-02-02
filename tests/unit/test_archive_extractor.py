#!/usr/bin/env python3
"""Unit tests for archive extraction service.

Tests:
- Archive type detection
- Manifest generation (zip, tar variants)
- Validation (size limits, zip bomb detection)
- File extraction
- Pattern extraction
- Security (path traversal prevention)
"""

from __future__ import annotations

import io
import tarfile
import tempfile
import zipfile
from pathlib import Path

import pytest

from src.services.archive_extractor import (
    ArchiveExtractor,
    ArchiveManifest,
    ArchiveType,
    ExtractionResult,
    ExtractionStrategy,
    FileEntry,
    ValidationResult,
    ValidationStatus,
)


@pytest.fixture
def extractor():
    """Create an ArchiveExtractor instance."""
    return ArchiveExtractor()


@pytest.fixture
def sample_zip(tmp_path):
    """Create a sample ZIP archive with test files."""
    archive_path = tmp_path / "test_archive.zip"

    with zipfile.ZipFile(archive_path, 'w') as zf:
        zf.writestr("readme.md", "# Test Archive\n\nThis is a test.")
        zf.writestr("src/main.py", "print('hello world')")
        zf.writestr("src/utils.py", "def helper(): pass")
        zf.writestr("docs/guide.pdf", b"%PDF-1.4 fake pdf content")
        zf.writestr("data/config.json", '{"key": "value"}')

    return archive_path


@pytest.fixture
def sample_tar_gz(tmp_path):
    """Create a sample tar.gz archive."""
    archive_path = tmp_path / "test_archive.tar.gz"

    with tarfile.open(archive_path, 'w:gz') as tf:
        # Add string content
        for name, content in [
            ("readme.txt", b"Test readme content"),
            ("src/app.py", b"# Python app\nprint('test')"),
            ("data/items.csv", b"id,name\n1,foo\n2,bar"),
        ]:
            data = io.BytesIO(content)
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, data)

    return archive_path


@pytest.fixture
def nested_archive(tmp_path):
    """Create an archive containing another archive."""
    # Create inner archive
    inner_path = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner_path, 'w') as zf:
        zf.writestr("inner_file.txt", "content from inner archive")

    # Create outer archive containing inner
    outer_path = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer_path, 'w') as zf:
        zf.writestr("outer_file.txt", "content from outer archive")
        zf.write(inner_path, "nested/inner.zip")

    return outer_path


class TestArchiveTypeDetection:
    """Test archive type detection from file extensions."""

    def test_zip_extension(self, extractor):
        """Test .zip detection."""
        assert extractor.get_archive_type(Path("test.zip")) == ArchiveType.ZIP

    def test_tar_extension(self, extractor):
        """Test .tar detection."""
        assert extractor.get_archive_type(Path("test.tar")) == ArchiveType.TAR

    def test_tar_gz_extension(self, extractor):
        """Test .tar.gz detection."""
        assert extractor.get_archive_type(Path("test.tar.gz")) == ArchiveType.TAR_GZ

    def test_tgz_extension(self, extractor):
        """Test .tgz detection."""
        assert extractor.get_archive_type(Path("test.tgz")) == ArchiveType.TAR_GZ

    def test_tar_bz2_extension(self, extractor):
        """Test .tar.bz2 detection."""
        assert extractor.get_archive_type(Path("test.tar.bz2")) == ArchiveType.TAR_BZ2

    def test_tar_xz_extension(self, extractor):
        """Test .tar.xz detection."""
        assert extractor.get_archive_type(Path("test.tar.xz")) == ArchiveType.TAR_XZ

    def test_7z_extension(self, extractor):
        """Test .7z detection."""
        assert extractor.get_archive_type(Path("test.7z")) == ArchiveType.SEVEN_Z

    def test_unsupported_extension(self, extractor):
        """Test unsupported extension returns None."""
        assert extractor.get_archive_type(Path("test.rar")) is None
        assert extractor.get_archive_type(Path("test.txt")) is None

    def test_case_insensitive(self, extractor):
        """Test case insensitive extension matching."""
        assert extractor.get_archive_type(Path("TEST.ZIP")) == ArchiveType.ZIP
        assert extractor.get_archive_type(Path("Archive.TAR.GZ")) == ArchiveType.TAR_GZ


class TestManifestGeneration:
    """Test archive manifest generation."""

    def test_zip_manifest(self, extractor, sample_zip):
        """Test manifest generation for ZIP archive."""
        manifest = extractor.list_contents(sample_zip)

        assert isinstance(manifest, ArchiveManifest)
        assert manifest.archive_type == ArchiveType.ZIP
        assert manifest.total_files == 5
        assert manifest.total_size_bytes > 0
        assert ".md" in manifest.type_summary
        assert ".py" in manifest.type_summary
        assert ".pdf" in manifest.type_summary
        assert ".json" in manifest.type_summary

    def test_tar_gz_manifest(self, extractor, sample_tar_gz):
        """Test manifest generation for tar.gz archive."""
        manifest = extractor.list_contents(sample_tar_gz)

        assert isinstance(manifest, ArchiveManifest)
        assert manifest.archive_type == ArchiveType.TAR_GZ
        assert manifest.total_files == 3
        assert ".txt" in manifest.type_summary
        assert ".py" in manifest.type_summary
        assert ".csv" in manifest.type_summary

    def test_manifest_file_entries(self, extractor, sample_zip):
        """Test file entries in manifest."""
        manifest = extractor.list_contents(sample_zip)

        # Find specific file
        readme = next((f for f in manifest.file_tree if f.name == "readme.md"), None)
        assert readme is not None
        assert readme.extension == ".md"
        assert not readme.is_dir
        assert readme.size > 0

    def test_manifest_nested_detection(self, extractor, nested_archive):
        """Test detection of nested archives."""
        manifest = extractor.list_contents(nested_archive)

        assert len(manifest.nested_archives) == 1
        assert "inner.zip" in manifest.nested_archives[0]

    def test_manifest_strategy_recommendation(self, extractor, sample_zip):
        """Test extraction strategy recommendation."""
        manifest = extractor.list_contents(sample_zip)

        # Small archive should recommend auto_all
        strategy = manifest.recommend_strategy()
        assert strategy == ExtractionStrategy.AUTO_ALL

    def test_manifest_hash(self, extractor, sample_zip):
        """Test manifest hash for change detection."""
        manifest1 = extractor.list_contents(sample_zip)
        manifest2 = extractor.list_contents(sample_zip)

        # Same archive should have same hash
        assert manifest1.manifest_hash == manifest2.manifest_hash

    def test_files_matching_pattern(self, extractor, sample_zip):
        """Test file pattern matching."""
        manifest = extractor.list_contents(sample_zip)

        py_files = manifest.files_matching("*.py")
        assert len(py_files) == 2

        all_files = manifest.files_matching("*")
        assert len(all_files) == 5


class TestValidation:
    """Test archive validation."""

    def test_valid_archive(self, extractor, sample_zip):
        """Test validation of valid archive."""
        result = extractor.validate(sample_zip)

        assert result.status == ValidationStatus.VALID
        assert result.is_safe
        assert result.archive_type == ArchiveType.ZIP

    def test_missing_file(self, extractor, tmp_path):
        """Test validation of non-existent file."""
        result = extractor.validate(tmp_path / "nonexistent.zip")

        assert result.status == ValidationStatus.INVALID
        assert not result.is_safe
        assert "not found" in result.issues[0].lower()

    def test_unsupported_type(self, extractor, tmp_path):
        """Test validation of unsupported archive type."""
        fake_archive = tmp_path / "test.rar"
        fake_archive.write_text("not an archive")

        result = extractor.validate(fake_archive)

        assert result.status == ValidationStatus.INVALID
        assert not result.is_safe

    def test_too_large_archive(self, extractor, tmp_path):
        """Test validation rejects oversized archives."""
        # Create extractor with very low limit
        small_extractor = ArchiveExtractor(max_archive_size=100)

        archive_path = tmp_path / "large.zip"
        with zipfile.ZipFile(archive_path, 'w') as zf:
            zf.writestr("big_file.txt", "x" * 1000)

        result = small_extractor.validate(archive_path)
        assert result.status == ValidationStatus.TOO_LARGE

    def test_too_many_files(self, extractor, tmp_path):
        """Test validation rejects archives with too many files."""
        # Create extractor with very low limit
        small_extractor = ArchiveExtractor(max_files=3)

        archive_path = tmp_path / "many_files.zip"
        with zipfile.ZipFile(archive_path, 'w') as zf:
            for i in range(10):
                zf.writestr(f"file_{i}.txt", f"content {i}")

        result = small_extractor.validate(archive_path)
        assert result.status == ValidationStatus.TOO_MANY_FILES


class TestExtraction:
    """Test file extraction."""

    def test_extract_specific_files(self, extractor, sample_zip, tmp_path):
        """Test extracting specific files."""
        dest = tmp_path / "extracted"
        result = extractor.extract_files(
            sample_zip,
            files=["readme.md", "src/main.py"],
            dest=dest,
        )

        assert result.success
        assert len(result.extracted_files) == 2
        assert "readme.md" in result.extracted_files
        assert "src/main.py" in result.extracted_files

        # Check content
        readme_path = result.extracted_files["readme.md"]
        assert readme_path.exists()
        assert "Test Archive" in readme_path.read_text()

    def test_extract_with_pattern(self, extractor, sample_zip, tmp_path):
        """Test extracting files matching a pattern."""
        dest = tmp_path / "extracted"
        result = extractor.extract_pattern(sample_zip, "*.py", dest)

        assert result.success
        assert len(result.extracted_files) == 2
        assert all(name.endswith(".py") for name in result.extracted_files.keys())

    def test_extract_all(self, extractor, sample_zip, tmp_path):
        """Test extracting all files."""
        dest = tmp_path / "extracted"
        result = extractor.extract_all(sample_zip, dest)

        assert result.success
        assert len(result.extracted_files) == 5

    def test_extract_nonexistent_file(self, extractor, sample_zip, tmp_path):
        """Test extracting non-existent file."""
        dest = tmp_path / "extracted"
        result = extractor.extract_files(
            sample_zip,
            files=["nonexistent.txt"],
            dest=dest,
        )

        assert not result.success
        assert "nonexistent.txt" in result.skipped_files
        assert any("not found" in e.lower() for e in result.errors)

    def test_extract_creates_directories(self, extractor, sample_zip, tmp_path):
        """Test extraction creates nested directories."""
        dest = tmp_path / "extracted"
        result = extractor.extract_files(
            sample_zip,
            files=["src/main.py"],
            dest=dest,
        )

        assert result.success
        assert (dest / "src" / "main.py").exists()

    def test_extract_tar_gz(self, extractor, sample_tar_gz, tmp_path):
        """Test extracting from tar.gz archive."""
        dest = tmp_path / "extracted"
        result = extractor.extract_all(sample_tar_gz, dest)

        assert result.success
        assert len(result.extracted_files) == 3


class TestSecurity:
    """Test security measures."""

    def test_path_traversal_prevention(self, extractor, tmp_path):
        """Test that path traversal is prevented."""
        # Create archive with path traversal attempt
        archive_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive_path, 'w') as zf:
            zf.writestr("../../../etc/passwd", "malicious content")

        dest = tmp_path / "extracted"
        result = extractor.extract_all(archive_path, dest)

        # Should skip the malicious file
        assert len(result.extracted_files) == 0
        assert len(result.errors) > 0
        assert any("unsafe" in e.lower() for e in result.errors)

    def test_large_file_rejection(self, extractor, tmp_path):
        """Test that oversized files are rejected."""
        # Create extractor with low per-file limit
        small_extractor = ArchiveExtractor()
        small_extractor.MAX_SINGLE_FILE = 100  # 100 bytes

        archive_path = tmp_path / "big_file.zip"
        with zipfile.ZipFile(archive_path, 'w') as zf:
            zf.writestr("small.txt", "ok")
            zf.writestr("big.txt", "x" * 1000)

        dest = tmp_path / "extracted"
        result = small_extractor.extract_all(archive_path, dest)

        # Small file should extract, big file should be skipped
        assert "small.txt" in result.extracted_files
        assert "big.txt" in result.skipped_files


class TestCleanup:
    """Test cleanup functionality."""

    def test_cleanup_expired(self, tmp_path):
        """Test cleanup of expired archives."""
        import time

        # Create mock archive directory structure
        base_dir = tmp_path / "archives"
        base_dir.mkdir()

        old_dir = base_dir / "old_session"
        old_dir.mkdir()
        (old_dir / "file.txt").write_text("old content")

        new_dir = base_dir / "new_session"
        new_dir.mkdir()
        (new_dir / "file.txt").write_text("new content")

        # Make old_dir appear old by modifying mtime
        import os
        old_time = time.time() - (48 * 3600)  # 48 hours ago
        os.utime(old_dir, (old_time, old_time))

        # Mock config to point at our temp base_dir
        from unittest.mock import MagicMock, patch
        from src import config as config_module

        mock_cfg = MagicMock()
        mock_cfg.services.archive_extract_dir = base_dir

        with patch.object(config_module, "get_config", return_value=mock_cfg):
            cleaned = ArchiveExtractor.cleanup_expired(max_age_hours=24)

            # Old dir should be cleaned
            assert cleaned == 1
            assert not old_dir.exists()
            assert new_dir.exists()


class TestDataclasses:
    """Test dataclass functionality."""

    def test_file_entry_compression_ratio(self):
        """Test FileEntry compression ratio calculation."""
        entry = FileEntry(
            name="test.txt",
            size=1000,
            compressed_size=100,
            is_dir=False,
            extension=".txt",
        )
        assert entry.compression_ratio == 10.0

    def test_file_entry_no_compression(self):
        """Test FileEntry with no compression data."""
        entry = FileEntry(
            name="test.txt",
            size=1000,
            compressed_size=None,
            is_dir=False,
            extension=".txt",
        )
        assert entry.compression_ratio is None

    def test_manifest_summary_dict(self, extractor, sample_zip):
        """Test manifest to_summary_dict method."""
        manifest = extractor.list_contents(sample_zip)
        summary = manifest.to_summary_dict()

        assert "total_files" in summary
        assert "types" in summary
        assert "recommendation" in summary

    def test_extraction_result_summary(self, extractor, sample_zip, tmp_path):
        """Test ExtractionResult to_summary_dict method."""
        dest = tmp_path / "extracted"
        result = extractor.extract_all(sample_zip, dest)
        summary = result.to_summary_dict()

        assert "success" in summary
        assert "extracted" in summary
        assert "total_size" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

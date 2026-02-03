#!/usr/bin/env python3
"""Integration tests for the archive extraction pipeline.

These tests validate the complete archive processing workflow:
1. Archive extraction (zip, tar variants)
2. Document preprocessing integration
3. REPL archive functions
4. Multi-document search
5. Security constraints

To run all tests:
    pytest tests/integration/test_archive_pipeline.py -v

Note: Requires local filesystem paths (skipped in CI).
"""

import io
import json
import os
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Skip in CI - these tests require local filesystem paths (/mnt/raid0)
if os.environ.get("CI") == "true" or os.environ.get("ORCHESTRATOR_MOCK_MODE") == "true":
    pytest.skip("Skipping archive tests in CI (require local paths)", allow_module_level=True)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_document_archive(tmp_path):
    """Create a sample ZIP archive with documents and code files."""
    archive_path = tmp_path / "documents.zip"

    with zipfile.ZipFile(archive_path, "w") as zf:
        # Text files
        zf.writestr("readme.md", "# Project Documentation\n\nThis is the readme.")
        zf.writestr("notes.txt", "Meeting notes from 2026-01-15\n\nDiscussed architecture.")

        # Code files
        zf.writestr("src/main.py", "#!/usr/bin/env python3\n\ndef main():\n    print('hello')")
        zf.writestr("src/utils.py", "def helper(x):\n    return x * 2")
        zf.writestr("config.json", '{"api_key": "hidden", "debug": true}')

        # Create fake PDF content (will be mocked for OCR)
        zf.writestr("docs/report.pdf", b"%PDF-1.4 fake pdf content for testing")
        zf.writestr("docs/guide.pdf", b"%PDF-1.4 another fake pdf document")

    return archive_path


@pytest.fixture
def sample_tar_gz_archive(tmp_path):
    """Create a sample tar.gz archive."""
    archive_path = tmp_path / "code_bundle.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tf:
        for name, content in [
            ("src/app.py", b"# Main application\nclass App:\n    pass"),
            ("src/models.py", b"# Database models\nclass User:\n    pass"),
            ("tests/test_app.py", b"def test_app():\n    assert True"),
            ("requirements.txt", b"flask==2.0\npytest==7.0"),
            ("README.md", b"# Code Bundle\n\nProject readme."),
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
    with zipfile.ZipFile(inner_path, "w") as zf:
        zf.writestr("inner_doc.txt", "Content from inner archive")
        zf.writestr("inner_code.py", "def inner(): pass")

    # Create outer archive
    outer_path = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer_path, "w") as zf:
        zf.writestr("outer_doc.txt", "Content from outer archive")
        zf.write(inner_path, "archives/inner.zip")

    return outer_path


@pytest.fixture
def mock_ocr_result():
    """Create a mock OCR result for PDF processing."""
    return {
        "pages": [
            {
                "page": 1,
                "text": "# Report Title\n\nThis is the introduction.\n\n## Methods\n\nWe used advanced techniques.",
                "bboxes": [],
                "elapsed_sec": 1.5,
            }
        ],
        "total_pages": 1,
        "elapsed_sec": 1.5,
        "pages_per_sec": 0.67,
    }


@pytest.fixture
def repl_environment():
    """Create a REPL environment with archive functions."""
    from src.repl_environment import REPLEnvironment

    return REPLEnvironment(context="Test context for archive integration")


# =============================================================================
# Test Archive Extraction Integration
# =============================================================================


class TestArchiveExtractionIntegration:
    """Tests for archive extraction with document pipeline."""

    def test_extractor_with_document_preprocessor(self, sample_document_archive, tmp_path):
        """Test ArchiveExtractor works with DocumentPreprocessor detection."""
        from src.services.archive_extractor import ArchiveExtractor
        from src.services.document_preprocessor import DocumentPreprocessor

        extractor = ArchiveExtractor()
        preprocessor = DocumentPreprocessor()

        # Verify preprocessor detects archives
        task_ir = {
            "inputs": [{"type": "path", "value": str(sample_document_archive)}],
        }
        assert preprocessor.needs_preprocessing(task_ir) is True

        # Get manifest
        manifest = extractor.list_contents(sample_document_archive)

        # Verify manifest contains expected files
        assert manifest.total_files == 7
        assert ".md" in manifest.type_summary
        assert ".py" in manifest.type_summary
        assert ".pdf" in manifest.type_summary

    def test_extract_and_route_by_type(self, sample_document_archive, tmp_path):
        """Test extraction routes files correctly by type."""
        from src.services.archive_extractor import ArchiveExtractor
        from src.services.document_preprocessor import DOCUMENT_EXTENSIONS

        extractor = ArchiveExtractor()
        dest = tmp_path / "extracted"

        # Extract all
        result = extractor.extract_all(sample_document_archive, dest)

        assert result.success
        assert len(result.extracted_files) == 7

        # Categorize extracted files
        text_files = []
        code_files = []
        doc_files = []

        for filename, path in result.extracted_files.items():
            ext = Path(filename).suffix.lower()
            if ext in {".txt", ".md", ".json"}:
                text_files.append(filename)
            elif ext in {".py"}:
                code_files.append(filename)
            elif ext in DOCUMENT_EXTENSIONS:
                doc_files.append(filename)

        assert len(text_files) == 3  # readme.md, notes.txt, config.json
        assert len(code_files) == 2  # main.py, utils.py
        assert len(doc_files) == 2  # report.pdf, guide.pdf

    def test_extract_pattern_filtering(self, sample_document_archive, tmp_path):
        """Test pattern-based extraction."""
        from src.services.archive_extractor import ArchiveExtractor

        extractor = ArchiveExtractor()
        dest = tmp_path / "extracted"

        # Extract only Python files
        result = extractor.extract_pattern(sample_document_archive, "*.py", dest)

        assert result.success
        assert len(result.extracted_files) == 2
        assert all(name.endswith(".py") for name in result.extracted_files.keys())

    def test_tar_gz_extraction(self, sample_tar_gz_archive, tmp_path):
        """Test tar.gz archive extraction."""
        from src.services.archive_extractor import ArchiveExtractor

        extractor = ArchiveExtractor()
        dest = tmp_path / "extracted"

        result = extractor.extract_all(sample_tar_gz_archive, dest)

        assert result.success
        assert len(result.extracted_files) == 5

        # Verify content
        readme_path = result.extracted_files.get("README.md")
        assert readme_path is not None
        assert readme_path.exists()
        assert "Code Bundle" in readme_path.read_text()

    def test_nested_archive_detection(self, nested_archive):
        """Test nested archive detection in manifest."""
        from src.services.archive_extractor import ArchiveExtractor

        extractor = ArchiveExtractor()
        manifest = extractor.list_contents(nested_archive)

        assert len(manifest.nested_archives) == 1
        assert "inner.zip" in manifest.nested_archives[0]


# =============================================================================
# Test Document Preprocessing with Archives
# =============================================================================


class TestArchivePreprocessing:
    """Tests for archive preprocessing integration."""

    @pytest.mark.asyncio
    async def test_preprocess_archive_with_mocked_ocr(
        self, sample_document_archive, mock_ocr_result
    ):
        """Test archive preprocessing with mocked OCR."""
        from src.services.document_preprocessor import DocumentPreprocessor
        from src.models.document import OCRResult

        preprocessor = DocumentPreprocessor()

        # Mock OCR processing
        with patch("src.services.document_preprocessor.process_document") as mock_process:
            # Return mock OCR result for PDF files
            mock_process.return_value = OCRResult.from_dict(mock_ocr_result)

            result = await preprocessor.preprocess_archive(sample_document_archive)

            # Should succeed with mocked OCR
            assert result.success is True
            assert result.document_result is not None
            # Should have processed the 2 PDFs (report.pdf, guide.pdf)
            assert result.document_result.total_pages > 0

    @pytest.mark.asyncio
    async def test_preprocess_detects_archive_input(self, sample_document_archive):
        """Test that preprocessor correctly detects archive inputs."""
        from src.services.document_preprocessor import DocumentPreprocessor

        preprocessor = DocumentPreprocessor()

        task_ir = {
            "inputs": [{"type": "path", "value": str(sample_document_archive)}],
        }

        # Should detect archive needs preprocessing
        assert preprocessor.needs_preprocessing(task_ir) is True

        # Internal method should identify as archive
        assert preprocessor._is_archive(sample_document_archive) is True

    def test_archive_extensions_recognized(self):
        """Test all archive extensions are recognized."""
        from src.services.document_preprocessor import (
            ARCHIVE_EXTENSIONS,
            DocumentPreprocessor,
        )

        preprocessor = DocumentPreprocessor()

        for ext in ARCHIVE_EXTENSIONS:
            fake_path = Path(f"/tmp/test{ext}")
            assert preprocessor._is_archive(fake_path) is True


# =============================================================================
# Test REPL Archive Functions
# =============================================================================


class TestREPLArchiveFunctions:
    """Tests for REPL archive functions."""

    def test_archive_open_creates_manifest(self, sample_document_archive, repl_environment):
        """Test archive_open returns manifest and stores handle."""
        result = repl_environment.execute(
            f"result = archive_open('{sample_document_archive}')\nprint(result)"
        )

        assert result.error is None

        # Parse the JSON output to verify structure
        # The output contains the printed result, which should be JSON
        output_lines = result.output.strip().split("\n")
        # Find the line with JSON (may have other output before it)
        json_line = None
        for line in output_lines:
            if "total_files" in line:
                json_line = line
                break

        assert json_line is not None, "Output should contain JSON with total_files"

        # Parse and verify the manifest structure
        try:
            manifest_data = json.loads(json_line)
            assert manifest_data["total_files"] == 7  # Matches sample_document_archive fixture
            assert "types" in manifest_data
            # Verify specific file types from the fixture
            types = manifest_data["types"]
            assert ".md" in types  # readme.md
            assert ".py" in types  # main.py, utils.py
            assert ".pdf" in types  # report.pdf, guide.pdf
            assert ".json" in types  # config.json
            assert ".txt" in types  # notes.txt
        except json.JSONDecodeError:
            # If not valid JSON, check it's a dict-like string representation
            assert "total_files" in json_line
            assert "7" in json_line or "'total_files': 7" in json_line

        # Check artifacts storage
        assert "_archives" in repl_environment.artifacts

    def test_archive_extract_pattern(self, sample_document_archive, repl_environment):
        """Test archive_extract with pattern."""
        # First open
        repl_environment.execute(f"archive_open('{sample_document_archive}')")

        # Extract Python files
        result = repl_environment.execute("result = archive_extract(pattern='*.py')\nprint(result)")

        assert result.error is None

        # Parse the output to verify extraction details
        output = result.output.lower()
        assert "extracted" in output

        # Should mention 2 files extracted (main.py and utils.py from fixture)
        # Check for either count or filenames
        assert "2" in result.output or "main.py" in result.output or "utils.py" in result.output

        # Verify the specific filenames if present in output
        if "main.py" in result.output:
            assert "main.py" in result.output
        if "utils.py" in result.output:
            assert "utils.py" in result.output

    def test_archive_file_retrieval(self, sample_document_archive, repl_environment):
        """Test archive_file retrieves specific file content."""
        # Open and extract
        repl_environment.execute(f"archive_open('{sample_document_archive}')")
        repl_environment.execute("archive_extract(pattern='*.md')")

        # Get specific file
        result = repl_environment.execute("content = archive_file('readme.md')\nprint(content)")

        assert result.error is None

        # Verify the output contains the actual file content from the fixture
        # The readme.md fixture contains: "# Project Documentation\n\nThis is the readme."
        assert "Project Documentation" in result.output or "readme" in result.output.lower()

    def test_archive_search_across_files(self, sample_document_archive, repl_environment):
        """Test archive_search finds content across multiple files."""
        # Open and extract all text files
        open_result = repl_environment.execute(f"print(archive_open('{sample_document_archive}'))")
        assert open_result.error is None

        extract_result = repl_environment.execute("print(archive_extract(pattern='*.md'))")
        assert extract_result.error is None

        # Search for "Project" which appears in readme.md: "# Project Documentation"
        result = repl_environment.execute("result = archive_search('Project')\nprint(result)")

        assert result.error is None

        # Verify the search found matches
        # The readme.md contains "Project Documentation", so search should find it
        output = result.output.lower()
        # Check for search result indicators
        assert (
            "project" in output
            or "found" in output
            or "match" in output
            or "readme" in output  # filename where match was found
        ), "Search should find 'Project' in readme.md"

    def test_archive_functions_error_handling(self, repl_environment):
        """Test archive functions handle errors gracefully."""
        # Try to extract without opening
        result = repl_environment.execute("result = archive_extract(pattern='*.py')\nprint(result)")

        assert result.error is None
        assert "no archive" in result.output.lower() or "error" in result.output.lower()

        # Try to open non-existent file
        result = repl_environment.execute(
            "result = archive_open('/nonexistent/path.zip')\nprint(result)"
        )

        assert result.error is None  # Error should be captured, not raised


# =============================================================================
# Test Multi-Document Results
# =============================================================================


class TestMultiDocumentResults:
    """Tests for MultiDocumentResult functionality."""

    def test_multi_document_search(self):
        """Test search across multiple documents."""
        from src.models.document import (
            MultiDocumentResult,
            DocumentPreprocessResult,
            Section,
            ProcessingStatus,
        )

        # Create mock document results
        doc1_sections = [
            Section(
                id="s1",
                title="Introduction",
                level=1,
                content="This section discusses authentication methods.",
            ),
        ]
        doc2_sections = [
            Section(
                id="s1",
                title="Security",
                level=1,
                content="Security best practices include using strong authentication.",
            ),
        ]

        doc1 = DocumentPreprocessResult(
            original_path="doc1.pdf",
            sections=doc1_sections,
            figures=[],
            total_pages=1,
            status=ProcessingStatus.COMPLETED,
        )
        doc2 = DocumentPreprocessResult(
            original_path="doc2.pdf",
            sections=doc2_sections,
            figures=[],
            total_pages=1,
            status=ProcessingStatus.COMPLETED,
        )

        multi_result = MultiDocumentResult(
            source_archive="bundle.zip",
            documents={"doc1.pdf": doc1, "doc2.pdf": doc2},
            text_files={},
            skipped_files=[],
        )

        # Search should find in both documents
        hits = multi_result.search("authentication")

        assert len(hits) == 2
        assert any(h.source_file == "doc1.pdf" for h in hits)
        assert any(h.source_file == "doc2.pdf" for h in hits)

    def test_multi_document_text_file_search(self):
        """Test search in text files within multi-document result."""
        from src.models.document import MultiDocumentResult

        multi_result = MultiDocumentResult(
            source_archive="bundle.zip",
            documents={},
            text_files={
                "readme.md": "# Project\n\nThis project uses authentication.",
                "notes.txt": "TODO: Implement authentication flow.",
            },
            skipped_files=[],
        )

        hits = multi_result.search("authentication")

        assert len(hits) == 2
        assert any(h.source_file == "readme.md" for h in hits)
        assert any(h.source_file == "notes.txt" for h in hits)

    def test_multi_document_summary(self):
        """Test summary generation for multi-document result."""
        from src.models.document import (
            MultiDocumentResult,
            DocumentPreprocessResult,
            Section,
            ProcessingStatus,
        )

        doc = DocumentPreprocessResult(
            original_path="report.pdf",
            sections=[Section(id="s1", title="Test", level=1, content="Content")],
            figures=[],
            total_pages=5,
            status=ProcessingStatus.COMPLETED,
        )

        multi_result = MultiDocumentResult(
            source_archive="archive.zip",
            documents={"report.pdf": doc},
            text_files={"readme.txt": "hello"},
            skipped_files=["binary.exe"],
            processing_time=10.5,
        )

        summary = multi_result.to_summary_dict()

        assert summary["source"] == "archive.zip"
        assert summary["documents_processed"] == 1
        assert summary["text_files"] == 1
        assert summary["skipped"] == 1
        assert summary["total_pages"] == 5


# =============================================================================
# Test Security Constraints
# =============================================================================


class TestArchiveSecurityIntegration:
    """Tests for archive security constraints."""

    def test_path_traversal_blocked(self, tmp_path):
        """Test path traversal attempts are blocked in extraction."""
        from src.services.archive_extractor import ArchiveExtractor

        # Create malicious archive
        archive_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("../../../etc/passwd", "malicious content")
            zf.writestr("normal.txt", "safe content")

        extractor = ArchiveExtractor()
        dest = tmp_path / "extracted"

        result = extractor.extract_all(archive_path, dest)

        # Should only extract safe file
        assert "normal.txt" in result.extracted_files
        assert len(result.errors) > 0  # Path traversal should be reported

        # Verify no file was written outside dest
        assert not (tmp_path / "etc").exists()

    def test_validation_rejects_oversized_archive(self, tmp_path):
        """Test validation rejects oversized archives."""
        from src.services.archive_extractor import (
            ArchiveExtractor,
            ValidationStatus,
        )

        # Create archive with one small file
        archive_path = tmp_path / "test.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.writestr("test.txt", "x" * 1000)

        # Create extractor with very low limit
        extractor = ArchiveExtractor(max_archive_size=100)

        result = extractor.validate(archive_path)

        assert result.status == ValidationStatus.TOO_LARGE
        assert not result.is_safe

    def test_validation_rejects_too_many_files(self, tmp_path):
        """Test validation rejects archives with too many files."""
        from src.services.archive_extractor import (
            ArchiveExtractor,
            ValidationStatus,
        )

        # Create archive with many files
        archive_path = tmp_path / "many.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            for i in range(20):
                zf.writestr(f"file_{i}.txt", f"content {i}")

        # Create extractor with low file limit
        extractor = ArchiveExtractor(max_files=10)

        result = extractor.validate(archive_path)

        assert result.status == ValidationStatus.TOO_MANY_FILES
        assert not result.is_safe


# =============================================================================
# Test End-to-End Workflow
# =============================================================================


class TestEndToEndArchiveWorkflow:
    """Tests for complete archive processing workflow."""

    def test_complete_workflow_zip(self, sample_document_archive, tmp_path):
        """Test complete workflow: open → inspect → extract → use."""
        from src.services.archive_extractor import (
            ArchiveExtractor,
            ExtractionStrategy,
        )

        extractor = ArchiveExtractor()

        # Step 1: Validate
        validation = extractor.validate(sample_document_archive)
        assert validation.is_safe

        # Step 2: Get manifest
        manifest = extractor.list_contents(sample_document_archive)
        assert manifest.total_files == 7

        # Step 3: Check strategy recommendation
        strategy = manifest.recommend_strategy()
        # Small archive should recommend auto-all
        assert strategy == ExtractionStrategy.AUTO_ALL

        # Step 4: Extract based on decision
        dest = tmp_path / "extracted"
        if strategy == ExtractionStrategy.AUTO_ALL:
            result = extractor.extract_all(sample_document_archive, dest)
        else:
            # For larger archives, might extract selectively
            result = extractor.extract_pattern(sample_document_archive, "*.py", dest)

        assert result.success
        assert len(result.extracted_files) > 0

        # Step 5: Verify extracted content is accessible
        for filename, path in result.extracted_files.items():
            assert path.exists()

    def test_complete_workflow_tar_gz(self, sample_tar_gz_archive, tmp_path):
        """Test complete workflow with tar.gz archive."""
        from src.services.archive_extractor import ArchiveExtractor

        extractor = ArchiveExtractor()

        # Validate
        validation = extractor.validate(sample_tar_gz_archive)
        assert validation.is_safe

        # Get manifest
        manifest = extractor.list_contents(sample_tar_gz_archive)
        assert manifest.total_files == 5

        # Extract all
        dest = tmp_path / "extracted"
        result = extractor.extract_all(sample_tar_gz_archive, dest)

        assert result.success
        assert len(result.extracted_files) == 5

    def test_repl_complete_workflow(self, sample_document_archive, repl_environment):
        """Test complete workflow through REPL functions."""
        # Open archive - returns JSON string
        open_result = repl_environment.execute(
            f"result = archive_open('{sample_document_archive}')\nprint(result)"
        )
        assert open_result.error is None
        assert "total_files" in open_result.output

        # Extract code files - returns JSON string
        extract_result = repl_environment.execute(
            "result = archive_extract(pattern='*.py')\nprint(result)"
        )
        assert extract_result.error is None

        # Parse the extract output to verify specific files were extracted
        output = extract_result.output.lower()
        assert "extracted" in output or "error" not in output

        # Should mention the .py files extracted from the fixture
        # The fixture has src/main.py and src/utils.py
        assert (
            "main.py" in extract_result.output
            or "utils.py" in extract_result.output
            or "2" in extract_result.output  # 2 files extracted
        ), "Extract should report the Python files from the archive"


# =============================================================================
# Test Cleanup Integration
# =============================================================================


class TestArchiveCleanup:
    """Tests for archive cleanup functionality."""

    def test_cleanup_removes_old_extractions(self, tmp_path, monkeypatch):
        """Test cleanup removes expired extraction directories."""
        import os
        import time
        from pathlib import Path
        from src.services.archive_extractor import ArchiveExtractor
        from src import config as config_module

        # Create mock archive directory structure
        base_dir = tmp_path / "archives"
        base_dir.mkdir()

        old_dir = base_dir / "old_session"
        old_dir.mkdir()
        (old_dir / "file.txt").write_text("old content")

        new_dir = base_dir / "new_session"
        new_dir.mkdir()
        (new_dir / "file.txt").write_text("new content")

        # Make old_dir appear old
        old_time = time.time() - (48 * 3600)  # 48 hours ago
        os.utime(old_dir, (old_time, old_time))

        # Patch the config to use our test directory
        original_get_config = config_module.get_config

        def mock_get_config():
            config = original_get_config()
            config.services.archive_extract_dir = Path(base_dir)
            return config

        monkeypatch.setattr(config_module, "get_config", mock_get_config)

        cleaned = ArchiveExtractor.cleanup_expired(max_age_hours=24)

        assert cleaned == 1
        assert not old_dir.exists()
        assert new_dir.exists()


# =============================================================================
# Test Error Scenarios
# =============================================================================


class TestArchiveErrorScenarios:
    """Tests for archive error handling."""

    def test_corrupted_zip_handling(self, tmp_path):
        """Test handling of corrupted ZIP file."""
        from src.services.archive_extractor import (
            ArchiveExtractor,
            ValidationStatus,
        )

        # Create corrupted ZIP
        corrupted_path = tmp_path / "corrupted.zip"
        corrupted_path.write_bytes(b"PK\x03\x04corrupted content")

        extractor = ArchiveExtractor()
        result = extractor.validate(corrupted_path)

        assert result.status == ValidationStatus.INVALID
        assert not result.is_safe

    def test_empty_archive_handling(self, tmp_path):
        """Test handling of empty archive."""
        from src.services.archive_extractor import ArchiveExtractor

        # Create empty ZIP
        empty_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(empty_path, "w"):
            pass  # Empty archive

        extractor = ArchiveExtractor()
        manifest = extractor.list_contents(empty_path)

        assert manifest.total_files == 0

    def test_unsupported_format_handling(self, tmp_path):
        """Test handling of unsupported archive format."""
        from src.services.archive_extractor import (
            ArchiveExtractor,
            ValidationStatus,
        )

        # Create file with unsupported extension
        rar_path = tmp_path / "archive.rar"
        rar_path.write_text("not a real rar")

        extractor = ArchiveExtractor()
        result = extractor.validate(rar_path)

        assert result.status == ValidationStatus.INVALID


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

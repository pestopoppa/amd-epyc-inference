#!/usr/bin/env python3
"""Unit tests for the REPL archive tools (_ArchiveToolsMixin)."""

from unittest.mock import Mock, patch
from pathlib import Path

from src.repl_environment import REPLEnvironment


class TestArchiveOpen:
    """Test _archive_open() / archive_open() function."""

    def test_archive_open_path_validation(self):
        """Test archive_open() validates file paths."""
        repl = REPLEnvironment(context="test")
        # Try to open a file outside allowed paths
        result = repl.execute("print(archive_open('/etc/passwd'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "not in allowed" in result.output or "Path" in result.output

    @patch("src.services.archive_extractor.ArchiveExtractor")
    def test_archive_open_success(self, mock_extractor_class):
        """Test archive_open() successfully opens an archive."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor

        # Mock validation result
        mock_validation = Mock()
        mock_validation.is_safe = True
        mock_validation.issues = []
        mock_extractor.validate.return_value = mock_validation

        # Mock manifest
        mock_manifest = Mock()
        mock_manifest.to_summary_dict.return_value = {
            "total_files": 5,
            "total_size": 1024,
            "file_types": {"txt": 3, "pdf": 2},
        }
        mock_extractor.list_contents.return_value = mock_manifest

        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = archive_open('/mnt/raid0/llm/test.zip')
data = json.loads(output)
print(data['total_files'])
""")

        assert result.error is None
        assert "5" in result.output
        # Should store in artifacts
        assert "_archives" in repl.artifacts

    @patch("src.services.archive_extractor.ArchiveExtractor")
    def test_archive_open_unsafe_archive(self, mock_extractor_class):
        """Test archive_open() rejects unsafe archives."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor

        # Mock unsafe validation
        mock_validation = Mock()
        mock_validation.is_safe = False
        mock_validation.status = Mock(value="path_traversal")
        mock_validation.issues = ["Contains path traversal"]
        mock_extractor.validate.return_value = mock_validation

        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = archive_open('/mnt/raid0/llm/malicious.zip')
print('error' in output.lower())
""")

        assert result.error is None
        assert "True" in result.output

    def test_archive_open_increments_exploration(self):
        """Test archive_open() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        # Will fail validation but still increment
        repl.execute("archive_open('/tmp/test.zip')")

        assert repl._exploration_calls > initial_calls


class TestArchiveExtract:
    """Test _archive_extract() / archive_extract() function."""

    def test_archive_extract_no_archive_opened(self):
        """Test archive_extract() fails when no archive opened."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(archive_extract())")

        assert result.error is None
        assert "ERROR" in result.output
        assert "No archive opened" in result.output

    @patch("src.services.archive_extractor.ArchiveExtractor")
    def test_archive_extract_all(self, mock_extractor_class):
        """Test archive_extract() extracts all files."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor

        # Setup mock extraction result
        mock_result = Mock()
        mock_result.success = True
        mock_result.extracted_files = {
            "file1.txt": Path("/tmp/extracted/file1.txt"),
            "file2.txt": Path("/tmp/extracted/file2.txt"),
        }
        mock_result.skipped_files = []
        mock_result.errors = []
        mock_extractor.extract_all.return_value = mock_result

        # Setup archive in artifacts
        repl = REPLEnvironment(context="test")
        mock_manifest = Mock()
        repl.artifacts["_archives"] = {
            "test.zip": {
                "manifest": mock_manifest,
                "path": "/mnt/raid0/llm/test.zip",
                "extracted_to": None,
                "processed_files": {},
            }
        }

        result = repl.execute("""
output = archive_extract('test.zip', process_documents=False)
data = json.loads(output)
print(data.get('extracted', 0))
""")

        assert result.error is None
        assert "2" in result.output

    @patch("src.services.archive_extractor.ArchiveExtractor")
    def test_archive_extract_pattern(self, mock_extractor_class):
        """Test archive_extract() with pattern matching."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor

        mock_result = Mock()
        mock_result.success = True
        mock_result.extracted_files = {
            "doc.pdf": Path("/tmp/extracted/doc.pdf"),
        }
        mock_result.skipped_files = []
        mock_result.errors = []
        mock_extractor.extract_pattern.return_value = mock_result

        repl = REPLEnvironment(context="test")
        mock_manifest = Mock()
        repl.artifacts["_archives"] = {
            "test.zip": {
                "manifest": mock_manifest,
                "path": "/mnt/raid0/llm/test.zip",
                "extracted_to": None,
                "processed_files": {},
            }
        }

        result = repl.execute("""
output = archive_extract('test.zip', pattern='*.pdf', process_documents=False)
data = json.loads(output)
print(data.get('extracted', 0))
""")

        assert result.error is None
        assert "1" in result.output

    def test_archive_extract_increments_exploration(self):
        """Test archive_extract() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        initial_calls = repl._exploration_calls
        repl.execute("archive_extract()")

        assert repl._exploration_calls > initial_calls


class TestArchiveFile:
    """Test _archive_file() / archive_file() function."""

    def test_archive_file_no_archives(self):
        """Test archive_file() fails when no archives opened."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("print(archive_file('file.txt'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "No archives opened" in result.output

    def test_archive_file_text_content(self):
        """Test archive_file() returns text file content."""
        repl = REPLEnvironment(context="test")
        repl.artifacts["_archives"] = {
            "test.zip": {
                "processed_files": {
                    "readme.txt": {
                        "type": "text",
                        "content": "This is the readme",
                        "lines": 1,
                    }
                }
            }
        }

        result = repl.execute("print(archive_file('readme.txt', 'test.zip'))")

        assert result.error is None
        assert "This is the readme" in result.output

    def test_archive_file_document_metadata(self):
        """Test archive_file() returns document metadata."""
        repl = REPLEnvironment(context="test")
        repl.artifacts["_archives"] = {
            "test.zip": {
                "processed_files": {
                    "doc.pdf": {
                        "type": "document",
                        "sections": 5,
                        "figures": 2,
                    }
                }
            }
        }

        result = repl.execute("""
output = archive_file('doc.pdf', 'test.zip')
data = json.loads(output)
print(data.get('sections'))
""")

        assert result.error is None
        assert "5" in result.output

    def test_archive_file_not_found(self):
        """Test archive_file() handles missing files."""
        repl = REPLEnvironment(context="test")
        repl.artifacts["_archives"] = {"test.zip": {"processed_files": {}}}

        result = repl.execute("print(archive_file('missing.txt', 'test.zip'))")

        assert result.error is None
        assert "ERROR" in result.output
        assert "not found" in result.output

    def test_archive_file_increments_exploration(self):
        """Test archive_file() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        repl.artifacts["_archives"] = {"test.zip": {"processed_files": {}}}
        initial_calls = repl._exploration_calls
        repl.execute("archive_file('file.txt')")

        assert repl._exploration_calls > initial_calls


class TestArchiveSearch:
    """Test _archive_search() / archive_search() function."""

    def test_archive_search_no_archives(self):
        """Test archive_search() returns empty when no archives."""
        repl = REPLEnvironment(context="test")
        result = repl.execute("""
output = archive_search('query')
results = json.loads(output)
print(len(results))
""")

        assert result.error is None
        assert "0" in result.output

    def test_archive_search_text_files(self):
        """Test archive_search() searches text file content."""
        repl = REPLEnvironment(context="test")
        repl.artifacts["_archives"] = {
            "test.zip": {
                "processed_files": {
                    "file1.txt": {
                        "type": "text",
                        "content": "This is a test file\nIt contains important data\nMore text here",
                    },
                    "file2.txt": {
                        "type": "text",
                        "content": "Another file\nNo match here",
                    },
                }
            }
        }

        result = repl.execute("""
output = archive_search('important')
results = json.loads(output)
print(len(results))
print(results[0]['file'])
""")

        assert result.error is None
        assert "1" in result.output
        assert "file1.txt" in result.output

    def test_archive_search_document_files(self):
        """Test archive_search() searches document previews."""
        repl = REPLEnvironment(context="test")
        repl.artifacts["_archives"] = {
            "docs.zip": {
                "processed_files": {
                    "report.pdf": {
                        "type": "document",
                        "text_preview": "This report contains analysis of the data",
                    }
                }
            }
        }

        result = repl.execute("""
output = archive_search('analysis')
results = json.loads(output)
print(len(results))
""")

        assert result.error is None
        assert "1" in result.output

    def test_archive_search_case_insensitive(self):
        """Test archive_search() is case insensitive."""
        repl = REPLEnvironment(context="test")
        repl.artifacts["_archives"] = {
            "test.zip": {
                "processed_files": {
                    "file.txt": {
                        "type": "text",
                        "content": "UPPERCASE text here",
                    }
                }
            }
        }

        result = repl.execute("""
output = archive_search('uppercase')
results = json.loads(output)
print(len(results))
""")

        assert result.error is None
        assert "1" in result.output

    def test_archive_search_limits_results(self):
        """Test archive_search() limits results to 50."""
        repl = REPLEnvironment(context="test")
        # Create many matching lines
        content = "\n".join([f"match line {i}" for i in range(100)])
        repl.artifacts["_archives"] = {
            "test.zip": {
                "processed_files": {
                    "large.txt": {
                        "type": "text",
                        "content": content,
                    }
                }
            }
        }

        result = repl.execute("""
output = archive_search('match')
results = json.loads(output)
print(len(results))
""")

        assert result.error is None
        # Should be capped at 50
        assert "50" in result.output or int(result.output.strip()) <= 50

    def test_archive_search_increments_exploration(self):
        """Test archive_search() increments exploration calls."""
        repl = REPLEnvironment(context="test")
        repl.artifacts["_archives"] = {}
        initial_calls = repl._exploration_calls
        repl.execute("archive_search('query')")

        assert repl._exploration_calls > initial_calls

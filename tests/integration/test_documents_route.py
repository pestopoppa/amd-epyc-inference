"""Integration tests for document processing endpoints.

Tests the FastAPI routes in src/api/routes/documents.py.

Note: The documents router is mounted at /v1/documents prefix.
Note: Tests require local environment configuration (skipped in CI).
"""

import base64
import os

import pytest
from fastapi.testclient import TestClient

# Skip in CI - these tests require local path validation configuration
if os.environ.get("CI") == "true" or os.environ.get("ORCHESTRATOR_MOCK_MODE") == "true":
    pytest.skip("Skipping document route tests in CI", allow_module_level=True)

from src.api import create_app


# Create test client
app = create_app()
client = TestClient(app)


class TestDocumentEndpoints:
    """Test document processing endpoints."""

    def test_ocr_health_check(self):
        """Test OCR server health check endpoint."""
        response = client.get("/v1/documents/health")
        assert response.status_code == 200

        data = response.json()
        assert "healthy" in data
        assert "server_url" in data
        # May be true or false depending on OCR server availability

    def test_process_document_missing_input(self):
        """Test that missing both file_path and file_base64 returns 400."""
        request_data = {
            "output_format": "text",
        }
        response = client.post("/v1/documents/process", json=request_data)
        assert response.status_code == 400
        assert "file_path or file_base64" in response.json()["detail"].lower()

    def test_process_document_nonexistent_file(self):
        """Test processing a file that doesn't exist returns 404."""
        request_data = {
            "file_path": "/nonexistent/path/to/file.pdf",
            "output_format": "text",
        }
        response = client.post("/v1/documents/process", json=request_data)
        assert response.status_code in [404, 403]  # 403 if path validation rejects it

    def test_process_document_with_base64(self):
        """Test processing a document via base64 encoding."""
        # Create a small test file
        test_content = b"Test PDF content (not a real PDF)"
        encoded = base64.b64encode(test_content).decode("utf-8")

        request_data = {
            "file_base64": encoded,
            "output_format": "text",
            "max_pages": 10,
            "extract_figures": False,
        }
        response = client.post("/v1/documents/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "success" in data
        # May succeed or fail depending on OCR availability
        # Just verify structure

    def test_process_document_with_file_path(self, tmp_path):
        """Test processing a document via file path."""
        # Create a temporary text file
        test_file = tmp_path / "test_doc.txt"
        test_file.write_text("This is a test document for processing.")

        request_data = {
            "file_path": str(test_file),
            "output_format": "text",
            "max_pages": 10,
        }
        response = client.post("/v1/documents/process", json=request_data)

        # May fail with 403 (path validation) or 200
        # Just verify it doesn't crash
        assert response.status_code in [200, 403]

    def test_process_document_base64_too_large(self):
        """Test that excessively large base64 payload is rejected."""
        # Create a payload larger than 100MB limit
        large_data = b"x" * (101 * 1024 * 1024)
        encoded = base64.b64encode(large_data).decode("utf-8")

        request_data = {
            "file_base64": encoded,
            "output_format": "text",
        }
        response = client.post("/v1/documents/process", json=request_data)
        assert response.status_code == 413  # Payload too large

    def test_process_uploaded_document(self, tmp_path):
        """Test processing an uploaded document via multipart form."""
        # Create a test file
        test_file = tmp_path / "upload_test.txt"
        test_file.write_text("Uploaded document content")

        # Upload via multipart form
        with open(test_file, "rb") as f:
            files = {"file": ("upload_test.txt", f, "text/plain")}
            data = {
                "output_format": "text",
                "max_pages": "10",
                "extract_figures": "true",
                "describe_figures": "false",
            }
            response = client.post("/v1/documents/process/upload", files=files, data=data)

        # Should return 200 (structure check only)
        assert response.status_code == 200
        result = response.json()
        assert "success" in result

    def test_preprocess_taskir_no_documents(self):
        """Test preprocessing TaskIR with no document inputs."""
        task_ir = {
            "task_id": "test-123",
            "task_type": "code",
            "inputs": [],  # No documents
        }

        response = client.post("/v1/documents/preprocess-taskir", json={"task_ir": task_ir})
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["needs_preprocessing"] is False

    def test_preprocess_taskir_with_document(self, tmp_path):
        """Test preprocessing TaskIR with a document input."""
        # Create a test document
        test_doc = tmp_path / "taskir_test.txt"
        test_doc.write_text("Document content for TaskIR")

        task_ir = {
            "task_id": "test-456",
            "task_type": "ingest",
            "inputs": [
                {"type": "path", "value": str(test_doc)},
            ],
        }

        response = client.post("/v1/documents/preprocess-taskir", json={"task_ir": task_ir})
        assert response.status_code == 200

        data = response.json()
        assert "success" in data
        assert "needs_preprocessing" in data

    def test_get_section_content_not_implemented(self):
        """Test that section retrieval returns 501 (not implemented)."""
        response = client.get("/v1/documents/section/sec-123?task_id=task-456")
        assert response.status_code == 501
        assert "session state" in response.json()["detail"].lower()

    def test_process_response_structure(self):
        """Test that process response has expected structure."""
        # Use base64 with minimal data
        encoded = base64.b64encode(b"test").decode("utf-8")
        request_data = {
            "file_base64": encoded,
            "output_format": "bbox",
        }

        response = client.post("/v1/documents/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # Verify response model structure
        assert "success" in data
        assert "total_pages" in data
        assert "sections" in data
        assert "figures" in data
        assert "failed_pages" in data
        assert isinstance(data["sections"], list)
        assert isinstance(data["figures"], list)
        assert isinstance(data["failed_pages"], list)

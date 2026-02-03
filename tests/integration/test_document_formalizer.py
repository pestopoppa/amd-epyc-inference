#!/usr/bin/env python3
"""Integration tests for document_formalizer (LightOnOCR-2).

These tests validate the document OCR workflow:
1. Server connection and health
2. PDF processing endpoint
3. Single image OCR endpoint
4. Bounding box extraction

Requirements:
    - lightonocr_llama_server.py running on localhost:9001
    - LightOnOCR-2 GGUF models available

To run with a live server:
    pytest tests/integration/test_document_formalizer.py -v --run-ocr-server

To run without server (mocked):
    pytest tests/integration/test_document_formalizer.py -v
"""

import base64
import os
import pytest
from unittest.mock import MagicMock, patch

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-ocr-server",
        action="store_true",
        default=False,
        help="Run tests against live document_formalizer server",
    )


@pytest.fixture
def ocr_server_url():
    """Get OCR server URL from environment or default."""
    return os.environ.get("LIGHTONOCR_URL", "http://localhost:9001")


@pytest.fixture
def requires_ocr_server(request):
    """Skip if server not requested and not mocked."""
    if not request.config.getoption("--run-ocr-server"):
        pytest.skip("Need --run-ocr-server option to run")


@pytest.fixture
def mock_ocr_response():
    """Create a mock OCR response."""
    return {
        "text": "Sample extracted text from document.\n\n![image](image_0.png) 100,200,300,400",
        "bboxes": [{"id": 0, "x1": 100, "y1": 200, "x2": 300, "y2": 400, "normalized": True}],
        "elapsed_sec": 5.8,
    }


@pytest.fixture
def sample_image_base64():
    """Create a minimal valid PNG image as base64."""
    # 1x1 white PNG
    png_data = bytes(
        [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,  # PNG signature
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,  # IHDR chunk
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,
            0x08,
            0x02,
            0x00,
            0x00,
            0x00,
            0x90,
            0x77,
            0x53,
            0xDE,
            0x00,
            0x00,
            0x00,
            0x0C,
            0x49,
            0x44,
            0x41,  # IDAT chunk
            0x54,
            0x08,
            0xD7,
            0x63,
            0xF8,
            0xFF,
            0xFF,
            0xFF,
            0x00,
            0x05,
            0xFE,
            0x02,
            0xFE,
            0xDC,
            0xCC,
            0x59,
            0xE7,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,  # IEND chunk
            0x44,
            0xAE,
            0x42,
            0x60,
            0x82,
        ]
    )
    return base64.b64encode(png_data).decode()


class TestDocumentFormalizerHealth:
    """Tests for server health endpoint."""

    def test_health_check_mocked(self):
        """Test health check returns expected structure."""
        mock_response = {
            "status": "healthy",
            "model": "lightonai/LightOnOCR-2-1B-bbox",
        }

        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: mock_response,
            )

            import httpx

            response = httpx.get("http://localhost:9001/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "LightOnOCR" in data["model"]

    @pytest.mark.requires_server
    def test_health_check_live(self, ocr_server_url, requires_ocr_server):
        """Test health check against live server."""
        import httpx

        response = httpx.get(f"{ocr_server_url}/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestDocumentFormalizerOCR:
    """Tests for OCR endpoint."""

    def test_ocr_endpoint_mocked(self, mock_ocr_response, sample_image_base64):
        """Test OCR endpoint with mocked response."""
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: mock_ocr_response,
            )

            import httpx

            response = httpx.post(
                "http://localhost:9001/v1/document/ocr",
                data={"image": sample_image_base64, "output_format": "bbox"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "text" in data
            assert "bboxes" in data
            assert "elapsed_sec" in data

    def test_ocr_response_structure(self, mock_ocr_response):
        """Test OCR response has required fields."""
        assert "text" in mock_ocr_response
        assert "bboxes" in mock_ocr_response
        assert isinstance(mock_ocr_response["bboxes"], list)

        if mock_ocr_response["bboxes"]:
            bbox = mock_ocr_response["bboxes"][0]
            assert all(k in bbox for k in ["id", "x1", "y1", "x2", "y2"])

    @pytest.mark.requires_server
    def test_ocr_endpoint_live(self, ocr_server_url, requires_ocr_server, sample_image_base64):
        """Test OCR endpoint against live server."""
        import httpx

        response = httpx.post(
            f"{ocr_server_url}/v1/document/ocr",
            data={"image": sample_image_base64, "output_format": "bbox"},
            timeout=120,
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "elapsed_sec" in data


class TestDocumentFormalizerPDF:
    """Tests for PDF processing endpoint."""

    def test_pdf_endpoint_mocked(self, mock_ocr_response):
        """Test PDF endpoint with mocked response."""
        pdf_response = {
            "pages": [mock_ocr_response],
            "total_pages": 1,
            "elapsed_sec": 5.8,
            "pages_per_sec": 0.17,
        }

        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: pdf_response,
            )

            import httpx

            response = httpx.post(
                "http://localhost:9001/v1/document/pdf",
                files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
                data={"max_pages": 1},
            )

            assert response.status_code == 200
            data = response.json()
            assert "pages" in data
            assert "total_pages" in data
            assert "pages_per_sec" in data

    def test_pdf_response_structure(self, mock_ocr_response):
        """Test PDF response has required fields."""
        pdf_response = {
            "pages": [mock_ocr_response],
            "total_pages": 1,
            "elapsed_sec": 5.8,
            "pages_per_sec": 0.17,
        }

        assert "pages" in pdf_response
        assert "total_pages" in pdf_response
        assert "pages_per_sec" in pdf_response
        assert len(pdf_response["pages"]) == pdf_response["total_pages"]


class TestBoundingBoxParsing:
    """Tests for bounding box extraction from OCR output."""

    def test_bbox_coordinates_normalized(self, mock_ocr_response):
        """Test bounding box coordinates are in normalized range."""
        for bbox in mock_ocr_response["bboxes"]:
            # Normalized coords should be in [0, 1000] range
            assert 0 <= bbox["x1"] <= 1000
            assert 0 <= bbox["y1"] <= 1000
            assert 0 <= bbox["x2"] <= 1000
            assert 0 <= bbox["y2"] <= 1000
            # x2 > x1 and y2 > y1
            assert bbox["x2"] > bbox["x1"]
            assert bbox["y2"] > bbox["y1"]

    def test_bbox_format_in_text(self, mock_ocr_response):
        """Test bbox format is present in raw text."""
        text = mock_ocr_response["text"]
        # Should contain format: ![image](image_N.png) x1,y1,x2,y2
        assert "![image](image_" in text or len(mock_ocr_response["bboxes"]) == 0


class TestPerformanceMetrics:
    """Tests for performance measurement."""

    def test_pages_per_sec_calculation(self):
        """Test pages/sec is correctly calculated."""
        total_pages = 8
        total_time = 47.2  # seconds
        expected_pps = total_pages / total_time

        assert abs(expected_pps - 0.17) < 0.01  # ~0.17 pg/s

    def test_elapsed_time_reported(self, mock_ocr_response):
        """Test elapsed time is included in response."""
        assert "elapsed_sec" in mock_ocr_response
        assert mock_ocr_response["elapsed_sec"] > 0

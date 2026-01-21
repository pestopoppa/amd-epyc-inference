#!/usr/bin/env python3
"""Integration tests for the LightOnOCR document pipeline.

These tests validate the complete document processing workflow:
1. Document models and data classes
2. Document client functionality
3. Document chunking by markdown headers
4. Document preprocessing service
5. REPL document functions

To run all tests:
    pytest tests/integration/test_document_pipeline.py -v

To run with live OCR server:
    pytest tests/integration/test_document_pipeline.py -v --run-ocr-server
"""

import asyncio
import base64
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_ocr_page_result():
    """Create a mock OCR page result."""
    return {
        "page": 1,
        "text": """# Introduction

This is the introduction section with some text.

## Methods

We used the following methods:
- Method A
- Method B

![image](image_0.png) 100,200,300,400

## Results

The results show significant improvement.

### Subsection 1

More detailed results here.
""",
        "bboxes": [
            {"id": 0, "x1": 100, "y1": 200, "x2": 300, "y2": 400, "normalized": True}
        ],
        "elapsed_sec": 5.8,
    }


@pytest.fixture
def mock_ocr_result(mock_ocr_page_result):
    """Create a mock complete OCR result."""
    return {
        "pages": [mock_ocr_page_result],
        "total_pages": 1,
        "elapsed_sec": 5.8,
        "pages_per_sec": 0.17,
    }


@pytest.fixture
def sample_image_base64():
    """Create a minimal valid PNG image as base64."""
    # 1x1 white PNG
    png_data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0xFF,
        0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
        0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
        0x44, 0xAE, 0x42, 0x60, 0x82
    ])
    return base64.b64encode(png_data).decode()


# =============================================================================
# Test Document Models
# =============================================================================


class TestDocumentModels:
    """Tests for document data models."""

    def test_bounding_box_creation(self):
        """Test BoundingBox creation and properties."""
        from src.models.document import BoundingBox

        bbox = BoundingBox(id=0, x1=100, y1=200, x2=300, y2=400)

        assert bbox.width == 200
        assert bbox.height == 200
        assert bbox.area == 40000

    def test_bounding_box_pixel_conversion(self):
        """Test normalized to pixel coordinate conversion."""
        from src.models.document import BoundingBox

        bbox = BoundingBox(id=0, x1=100, y1=200, x2=300, y2=400, normalized=True)

        # Convert to 2000x1000 image
        x1, y1, x2, y2 = bbox.to_pixel_coords(2000, 1000)

        assert x1 == 200  # 100 * 2000 / 1000
        assert y1 == 200  # 200 * 1000 / 1000
        assert x2 == 600  # 300 * 2000 / 1000
        assert y2 == 400  # 400 * 1000 / 1000

    def test_page_ocr_result_from_dict(self, mock_ocr_page_result):
        """Test PageOCRResult creation from dictionary."""
        from src.models.document import PageOCRResult

        result = PageOCRResult.from_dict(mock_ocr_page_result)

        assert result.page == 1
        assert "Introduction" in result.text
        assert len(result.bboxes) == 1
        assert result.bboxes[0].id == 0

    def test_ocr_result_from_dict(self, mock_ocr_result):
        """Test OCRResult creation from dictionary."""
        from src.models.document import OCRResult

        result = OCRResult.from_dict(mock_ocr_result)

        assert result.total_pages == 1
        assert len(result.pages) == 1
        assert result.pages_per_sec == 0.17

    def test_ocr_result_full_text(self, mock_ocr_result):
        """Test full text concatenation."""
        from src.models.document import OCRResult

        result = OCRResult.from_dict(mock_ocr_result)
        full_text = result.full_text

        assert "Introduction" in full_text
        assert "Methods" in full_text
        assert "Results" in full_text

    def test_section_creation(self):
        """Test Section data class."""
        from src.models.document import Section

        section = Section(
            id="s1",
            title="Introduction",
            level=1,
            content="Some content here.",
            page_start=1,
            page_end=2,
            figure_ids=["fig_0"],
        )

        assert section.id == "s1"
        assert section.page_range == (1, 2)

    def test_document_preprocess_result(self):
        """Test DocumentPreprocessResult creation."""
        from src.models.document import (
            DocumentPreprocessResult,
            Section,
            FigureRef,
            BoundingBox,
            ProcessingStatus,
        )

        section = Section(id="s1", title="Intro", level=1, content="Text")
        figure = FigureRef(
            id="p1_fig0",
            page=1,
            bbox=BoundingBox(id=0, x1=100, y1=200, x2=300, y2=400),
        )

        result = DocumentPreprocessResult(
            original_path="/path/to/doc.pdf",
            sections=[section],
            figures=[figure],
            total_pages=5,
            failed_pages=[3],
            processing_time=10.5,
        )

        assert result.success_rate == 0.8  # 4/5 pages
        assert len(result.sections) == 1
        assert len(result.figures) == 1

    def test_document_preprocess_result_to_searchable_text(self):
        """Test searchable text generation."""
        from src.models.document import DocumentPreprocessResult, Section

        sections = [
            Section(id="s1", title="Intro", level=1, content="Introduction text."),
            Section(id="s2", title="Methods", level=2, content="Method details."),
        ]

        result = DocumentPreprocessResult(
            original_path="/path/to/doc.pdf",
            sections=sections,
            figures=[],
            total_pages=1,
        )

        text = result.to_searchable_text()
        assert "# Intro" in text
        assert "## Methods" in text
        assert "Introduction text" in text


# =============================================================================
# Test Document Chunker
# =============================================================================


class TestDocumentChunker:
    """Tests for document chunking by markdown headers."""

    def test_chunk_by_headers_basic(self):
        """Test basic header chunking."""
        from src.services.document_chunker import DocumentChunker

        text = """# Introduction

This is intro content.

## Methods

Method details here.

### Subsection

More details.
"""
        chunker = DocumentChunker()
        sections = chunker.chunk_by_headers(text)

        assert len(sections) >= 2
        assert any(s.title == "Introduction" for s in sections)
        assert any(s.title == "Methods" for s in sections)

    def test_chunk_preserves_content(self):
        """Test that chunking preserves content."""
        from src.services.document_chunker import DocumentChunker

        text = """# Title

Paragraph 1.

Paragraph 2.
"""
        chunker = DocumentChunker()
        sections = chunker.chunk_by_headers(text)

        # Content should be preserved
        total_content = "".join(s.content for s in sections)
        assert "Paragraph 1" in total_content
        assert "Paragraph 2" in total_content

    def test_chunk_assigns_levels(self):
        """Test header level assignment."""
        from src.services.document_chunker import DocumentChunker

        text = """# Level 1

## Level 2

### Level 3
"""
        chunker = DocumentChunker()
        sections = chunker.chunk_by_headers(text)

        levels = {s.title: s.level for s in sections}
        assert levels.get("Level 1") == 1
        assert levels.get("Level 2") == 2
        assert levels.get("Level 3") == 3

    def test_extract_figures(self, mock_ocr_result):
        """Test figure extraction from OCR result."""
        from src.services.document_chunker import DocumentChunker
        from src.models.document import OCRResult

        ocr_result = OCRResult.from_dict(mock_ocr_result)
        chunker = DocumentChunker()
        figures = chunker.extract_figures(ocr_result)

        assert len(figures) == 1
        assert figures[0].page == 1

    def test_process_complete(self, mock_ocr_result):
        """Test complete document processing."""
        from src.services.document_chunker import DocumentChunker, chunk_document
        from src.models.document import OCRResult

        ocr_result = OCRResult.from_dict(mock_ocr_result)
        result = chunk_document(ocr_result, "/path/to/doc.pdf")

        assert result.original_path == "/path/to/doc.pdf"
        assert len(result.sections) > 0
        assert result.total_pages == 1


# =============================================================================
# Test Document Client
# =============================================================================


class TestDocumentClient:
    """Tests for the OCR client."""

    @pytest.mark.asyncio
    async def test_client_health_check_mocked(self):
        """Test health check with mocked response."""
        from src.services.document_client import DocumentFormalizerClient

        client = DocumentFormalizerClient()

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            result = await client.health_check()
            assert result is True

        await client.close()

    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test client as async context manager."""
        from src.services.document_client import DocumentFormalizerClient

        async with DocumentFormalizerClient() as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_ocr_image_mocked(self, mock_ocr_page_result, sample_image_base64):
        """Test image OCR with mocked response."""
        from src.services.document_client import DocumentFormalizerClient

        client = DocumentFormalizerClient()

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_page_result
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_http

            result = await client.ocr_image(sample_image_base64)

            assert result.text is not None
            assert "Introduction" in result.text

        await client.close()


# =============================================================================
# Test Document Preprocessor
# =============================================================================


class TestDocumentPreprocessor:
    """Tests for the document preprocessing service."""

    def test_needs_preprocessing_pdf(self):
        """Test PDF detection."""
        from src.services.document_preprocessor import DocumentPreprocessor

        preprocessor = DocumentPreprocessor()

        task_ir = {
            "inputs": [{"type": "path", "value": "/path/to/doc.pdf"}],
        }

        assert preprocessor.needs_preprocessing(task_ir) is True

    def test_needs_preprocessing_image(self):
        """Test image detection."""
        from src.services.document_preprocessor import DocumentPreprocessor

        preprocessor = DocumentPreprocessor()

        task_ir = {
            "inputs": [{"type": "path", "value": "/path/to/image.png"}],
        }

        assert preprocessor.needs_preprocessing(task_ir) is True

    def test_needs_preprocessing_explicit(self):
        """Test explicit preprocessing config."""
        from src.services.document_preprocessor import DocumentPreprocessor

        preprocessor = DocumentPreprocessor()

        task_ir = {
            "inputs": [{"type": "text", "value": "some text"}],
            "preprocessing": {"ocr": {"enabled": True}},
        }

        assert preprocessor.needs_preprocessing(task_ir) is True

    def test_needs_preprocessing_task_type(self):
        """Test task_type detection."""
        from src.services.document_preprocessor import DocumentPreprocessor

        preprocessor = DocumentPreprocessor()

        task_ir = {
            "task_type": "doc",
            "inputs": [],
        }

        assert preprocessor.needs_preprocessing(task_ir) is True

    def test_needs_preprocessing_negative(self):
        """Test non-document task."""
        from src.services.document_preprocessor import DocumentPreprocessor

        preprocessor = DocumentPreprocessor()

        task_ir = {
            "task_type": "code",
            "inputs": [{"type": "text", "value": "Write a function"}],
        }

        assert preprocessor.needs_preprocessing(task_ir) is False

    def test_detect_ocr_intent(self):
        """Test OCR intent detection from prompts."""
        from src.services.document_preprocessor import DocumentPreprocessor

        preprocessor = DocumentPreprocessor()

        assert preprocessor.detect_ocr_intent("Please OCR this document") is True
        assert preprocessor.detect_ocr_intent("Extract text from PDF") is True
        assert preprocessor.detect_ocr_intent("Scan this page") is True
        assert preprocessor.detect_ocr_intent("Write some code") is False

    def test_enrich_task_ir(self):
        """Test TaskIR enrichment."""
        from src.services.document_preprocessor import DocumentPreprocessor
        from src.models.document import DocumentPreprocessResult, Section

        preprocessor = DocumentPreprocessor()

        task_ir = {"inputs": [{"type": "path", "value": "/doc.pdf"}]}

        doc_result = DocumentPreprocessResult(
            original_path="/doc.pdf",
            sections=[Section(id="s1", title="Intro", level=1, content="Text")],
            figures=[],
            total_pages=1,
        )

        enriched = preprocessor.enrich_task_ir(task_ir, doc_result)

        assert "ocr_result" in enriched
        assert "sections" in enriched["ocr_result"]
        assert len(enriched["ocr_result"]["sections"]) == 1


# =============================================================================
# Test Document REPL
# =============================================================================


class TestDocumentREPL:
    """Tests for document REPL extension."""

    @pytest.fixture
    def document_repl(self):
        """Create a document REPL with sample data."""
        from src.repl_document import DocumentREPLEnvironment, DocumentContext
        from src.models.document import Section, FigureRef, BoundingBox

        sections = [
            Section(id="s0", title="Introduction", level=1, content="Intro content.", page_start=1, page_end=1),
            Section(id="s1", title="Methods", level=2, content="Method content.", page_start=2, page_end=3),
            Section(id="s2", title="Results", level=2, content="Results here.", page_start=4, page_end=5),
        ]

        figures = [
            FigureRef(
                id="p1_fig0",
                page=1,
                bbox=BoundingBox(id=0, x1=100, y1=200, x2=300, y2=400),
                section_id="s0",
            ),
        ]

        context = DocumentContext(
            sections=sections,
            figures=figures,
            total_pages=5,
            failed_pages=[],
            original_path="/path/to/doc.pdf",
        )

        # Generate context text
        context_text = "\n".join(f"{'#' * s.level} {s.title}\n{s.content}" for s in sections)

        return DocumentREPLEnvironment(
            context=context_text,
            document_context=context,
        )

    def test_sections_function(self, document_repl):
        """Test sections() function."""
        result = document_repl.execute("print(sections())")

        assert result.error is None
        assert "Introduction" in result.output
        assert "Methods" in result.output
        assert "Results" in result.output

    def test_section_function(self, document_repl):
        """Test section(n) function."""
        result = document_repl.execute("print(section(1))")

        assert result.error is None
        assert "Introduction" in result.output
        assert "Intro content" in result.output

    def test_section_out_of_range(self, document_repl):
        """Test section(n) with invalid index."""
        result = document_repl.execute("print(section(100))")

        assert result.error is None
        assert "not found" in result.output

    def test_figures_function(self, document_repl):
        """Test figures() function."""
        result = document_repl.execute("print(figures())")

        assert result.error is None
        assert "p1_fig0" in result.output

    def test_figures_by_section(self, document_repl):
        """Test figures(section=n) filtering."""
        result = document_repl.execute("print(figures(section=1))")

        assert result.error is None
        assert "p1_fig0" in result.output

    def test_search_sections(self, document_repl):
        """Test search_sections() function."""
        result = document_repl.execute("print(search_sections('Method'))")

        assert result.error is None
        assert "Methods" in result.output

    def test_document_info(self, document_repl):
        """Test document_info() function."""
        result = document_repl.execute("print(document_info())")

        assert result.error is None
        assert "total_pages" in result.output
        assert "5" in result.output

    def test_get_state_includes_document_functions(self, document_repl):
        """Test get_state() includes document functions."""
        state = document_repl.get_state()

        assert "sections()" in state
        assert "section(n)" in state
        assert "figures()" in state

    def test_from_document_result_factory(self):
        """Test factory method from DocumentPreprocessResult."""
        from src.repl_document import DocumentREPLEnvironment
        from src.models.document import DocumentPreprocessResult, Section

        doc_result = DocumentPreprocessResult(
            original_path="/doc.pdf",
            sections=[Section(id="s1", title="Test", level=1, content="Content")],
            figures=[],
            total_pages=1,
        )

        repl = DocumentREPLEnvironment.from_document_result(doc_result)

        assert repl is not None
        result = repl.execute("print(sections())")
        assert "Test" in result.output


# =============================================================================
# Test Dispatcher Integration
# =============================================================================


class TestDispatcherDocumentRoles:
    """Tests for document roles in dispatcher."""

    def test_role_mapping_includes_document(self):
        """Test dispatcher includes document roles."""
        from src.dispatcher import Dispatcher

        assert "doc" in Dispatcher.ROLE_MAPPING
        assert "document" in Dispatcher.ROLE_MAPPING
        assert "document_formalizer" in Dispatcher.ROLE_MAPPING
        assert "ocr" in Dispatcher.ROLE_MAPPING

    def test_role_mapping_targets(self):
        """Test document roles map to document_formalizer."""
        from src.dispatcher import Dispatcher

        assert Dispatcher.ROLE_MAPPING["doc"] == "document_formalizer"
        assert Dispatcher.ROLE_MAPPING["document"] == "document_formalizer"
        assert Dispatcher.ROLE_MAPPING["ocr"] == "document_formalizer"


# =============================================================================
# Test API Endpoints (Mocked)
# =============================================================================


class TestDocumentAPIEndpoints:
    """Tests for document API endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test document health endpoint."""
        from fastapi.testclient import TestClient
        from unittest.mock import AsyncMock, patch

        # Mock the document client
        with patch("src.api.routes.documents.get_document_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.health_check = AsyncMock(return_value=True)
            mock_client.base_url = "http://localhost:9001"
            mock_get_client.return_value = mock_client

            from src.api.routes.documents import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router)

            client = TestClient(app)
            response = client.get("/documents/health")

            assert response.status_code == 200
            data = response.json()
            assert data["healthy"] is True

    @pytest.mark.asyncio
    async def test_process_endpoint_missing_input(self):
        """Test process endpoint with missing input."""
        from fastapi.testclient import TestClient
        from src.api.routes.documents import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        response = client.post("/documents/process", json={})

        assert response.status_code == 400

"""Unit tests for PDF Router."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.services.pdf_router import (
    PDFRouter,
    PDFExtractionResult,
    BoundingBox,
    ExtractedFigure,
    extract_pdf,
)


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_bounding_box_fields(self):
        bbox = BoundingBox(x0=0.1, y0=0.2, x1=0.9, y1=0.8, page=1, width_px=100, height_px=200)
        assert bbox.x0 == 0.1
        assert bbox.y0 == 0.2
        assert bbox.x1 == 0.9
        assert bbox.y1 == 0.8
        assert bbox.page == 1
        assert bbox.width_px == 100
        assert bbox.height_px == 200


class TestExtractedFigure:
    """Tests for ExtractedFigure dataclass."""

    def test_extracted_figure_fields(self):
        bbox = BoundingBox(x0=0.1, y0=0.2, x1=0.9, y1=0.8, page=1)
        figure = ExtractedFigure(index=1, bbox=bbox, image_path="/tmp/fig1.png", format="png")
        assert figure.index == 1
        assert figure.bbox == bbox
        assert figure.image_path == "/tmp/fig1.png"
        assert figure.format == "png"


class TestPDFExtractionResult:
    """Tests for PDFExtractionResult dataclass."""

    def test_extraction_result_defaults(self):
        result = PDFExtractionResult(text="Hello world")
        assert result.text == "Hello world"
        assert result.figures == []
        assert result.page_count == 0
        assert result.method == "unknown"
        assert result.quality_score == 0.0
        assert result.latency_ms == 0.0
        assert result.ocr_required is False


class TestPDFRouterQuality:
    """Tests for text quality assessment."""

    def test_calculate_entropy_english(self):
        router = PDFRouter()
        # English text has high entropy
        text = "The quick brown fox jumps over the lazy dog."
        entropy = router._calculate_entropy(text)
        assert entropy > 3.5  # Typical English entropy

    def test_calculate_entropy_repetitive(self):
        router = PDFRouter()
        # Repetitive text has low entropy
        text = "aaaaaaaaaa"
        entropy = router._calculate_entropy(text)
        assert entropy < 1.0

    def test_calculate_entropy_empty(self):
        router = PDFRouter()
        assert router._calculate_entropy("") == 0.0

    def test_calculate_garbage_ratio_clean(self):
        router = PDFRouter()
        text = "This is clean text with no garbage."
        ratio = router._calculate_garbage_ratio(text)
        assert ratio < 0.05  # Very low garbage

    def test_calculate_garbage_ratio_binary(self):
        router = PDFRouter()
        text = "\x00\x01\x02\x03\x04\x05"
        ratio = router._calculate_garbage_ratio(text)
        assert ratio > 0.9  # High garbage

    def test_avg_word_length(self):
        router = PDFRouter()
        text = "The cat sat"  # 3, 3, 3
        avg = router._calculate_avg_word_length(text)
        assert avg == 3.0

    def test_assess_quality_good_text(self):
        router = PDFRouter()
        text = """
        This is a well-formatted document with proper English text.
        It contains multiple sentences and paragraphs of content.
        The text should have high entropy and low garbage ratio.
        We expect this to pass the quality check with flying colors.
        """
        score, needs_ocr = router._assess_text_quality(text)
        assert score > 0.7
        assert needs_ocr is False

    def test_assess_quality_garbage_text(self):
        router = PDFRouter()
        text = "x x x x x x x x x x x x x x x x x x x x x x x x x x x"
        score, needs_ocr = router._assess_text_quality(text)
        assert score < 0.5
        assert needs_ocr is True

    def test_assess_quality_short_text(self):
        router = PDFRouter()
        text = "Hi"
        score, needs_ocr = router._assess_text_quality(text)
        assert score == 0.0
        assert needs_ocr is True


class TestPDFRouterExtraction:
    """Tests for PDF extraction methods."""

    @patch("subprocess.run")
    def test_pdftotext_extraction(self, mock_run):
        router = PDFRouter()

        # Mock successful pdftotext output
        mock_run.return_value = MagicMock(
            returncode=0, stdout="This is extracted PDF text.", stderr=""
        )

        text, latency = router._extract_with_pdftotext(Path("/fake/test.pdf"))

        assert text == "This is extracted PDF text."
        assert latency > 0
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_pdftotext_failure(self, mock_run):
        router = PDFRouter()

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")

        text, latency = router._extract_with_pdftotext(Path("/fake/test.pdf"))

        assert text == ""

    def test_extract_figures_no_pymupdf(self):
        router = PDFRouter()
        router._has_pymupdf = False

        figures = router._extract_figures_pymupdf(Path("/fake/test.pdf"))
        assert figures == []


class TestPDFRouterIntegration:
    """Integration tests with real PDFs."""

    @pytest.mark.integration
    def test_extract_real_pdf(self):
        """Test extraction with a real PDF file."""
        pdf_path = Path("/mnt/raid0/llm/claude/tmp/Twyne_V1_Whitepaper.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not available")

        router = PDFRouter()
        result = router.extract_sync(pdf_path)

        assert result.text
        assert len(result.text) > 10000
        assert result.method == "pdftotext"
        assert result.quality_score > 0.8
        assert result.page_count > 0
        assert result.ocr_required is False

    @pytest.mark.integration
    def test_extract_with_figures(self):
        """Test figure extraction with a real PDF."""
        pdf_path = Path("/mnt/raid0/llm/claude/tmp/Twyne_V1_Whitepaper.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not available")

        router = PDFRouter()
        result = router.extract_sync(pdf_path, extract_figures=True)

        assert len(result.figures) > 0
        for fig in result.figures:
            assert fig.index > 0
            assert 0 <= fig.bbox.x0 <= 1
            assert 0 <= fig.bbox.y0 <= 1
            assert fig.bbox.page >= 1


class TestExtractPdfFunction:
    """Tests for convenience function."""

    @patch.object(PDFRouter, "extract_sync")
    def test_extract_pdf_calls_router(self, mock_extract):
        mock_extract.return_value = PDFExtractionResult(text="test")

        result = extract_pdf("/fake/path.pdf")

        mock_extract.assert_called_once()
        assert result.text == "test"

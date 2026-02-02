"""Comprehensive tests for figure analyzer service.

Tests coverage for src/services/figure_analyzer.py (currently under-tested).
"""

import asyncio
import base64
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from PIL import Image

from src.models.document import BBox, FigureRef
from src.services.figure_analyzer import (
    FigureAnalyzer,
    analyze_figures,
    get_figure_analyzer,
)


class TestFigureAnalyzerInit:
    """Test FigureAnalyzer initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        analyzer = FigureAnalyzer()
        assert analyzer.vision_api_url is not None
        assert analyzer.timeout > 0
        assert analyzer.max_concurrent > 0
        assert analyzer.vl_prompt is not None
        assert analyzer._client is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        analyzer = FigureAnalyzer(
            vision_api_url="http://custom:8000/vision",
            timeout=60.0,
            max_concurrent=5,
            vl_prompt="Custom prompt",
        )
        assert analyzer.vision_api_url == "http://custom:8000/vision"
        assert analyzer.timeout == 60.0
        assert analyzer.max_concurrent == 5
        assert analyzer.vl_prompt == "Custom prompt"


class TestFigureAnalyzerClientManagement:
    """Test HTTP client lifecycle management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_new_client(self):
        """Test that _get_client creates a new client if none exists."""
        analyzer = FigureAnalyzer()
        assert analyzer._client is None

        client = await analyzer._get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert analyzer._client is client

        # Cleanup
        await analyzer.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing_client(self):
        """Test that _get_client reuses existing client."""
        analyzer = FigureAnalyzer()
        client1 = await analyzer._get_client()
        client2 = await analyzer._get_client()
        assert client1 is client2

        await analyzer.close()

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the HTTP client."""
        analyzer = FigureAnalyzer()
        await analyzer._get_client()
        assert analyzer._client is not None

        await analyzer.close()
        assert analyzer._client is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test using FigureAnalyzer as async context manager."""
        async with FigureAnalyzer() as analyzer:
            assert analyzer is not None
            client = await analyzer._get_client()
            assert client is not None

        # Should be closed after exiting context
        assert analyzer._client is None


class TestImageProcessing:
    """Test image processing utilities."""

    def test_image_to_base64(self):
        """Test converting PIL Image to base64."""
        analyzer = FigureAnalyzer()

        # Create a test image
        img = Image.new("RGB", (100, 100), color="red")

        # Convert to base64
        b64_str = analyzer._image_to_base64(img)

        # Verify it's valid base64
        assert isinstance(b64_str, str)
        assert len(b64_str) > 0

        # Verify we can decode it back
        decoded = base64.b64decode(b64_str)
        assert len(decoded) > 0

    def test_crop_figure_normalized_coords(self):
        """Test cropping with normalized coordinates (0-1000)."""
        analyzer = FigureAnalyzer()

        # Create a 1000x1000 test image
        img = Image.new("RGB", (1000, 1000), color="blue")

        # Crop top-left quarter (normalized coords)
        bbox = (0, 0, 500, 500)
        cropped = analyzer._crop_figure(img, bbox, normalized=True)

        assert cropped.size == (500, 500)

    def test_crop_figure_pixel_coords(self):
        """Test cropping with pixel coordinates."""
        analyzer = FigureAnalyzer()

        # Create a 1000x1000 test image
        img = Image.new("RGB", (1000, 1000), color="green")

        # Crop with pixel coords
        bbox = (100, 100, 300, 300)
        cropped = analyzer._crop_figure(img, bbox, normalized=False)

        assert cropped.size == (200, 200)

    def test_crop_figure_bounds_clamping(self):
        """Test that crop coords are clamped to image bounds."""
        analyzer = FigureAnalyzer()

        img = Image.new("RGB", (100, 100), color="yellow")

        # Try to crop outside image bounds
        bbox = (-50, -50, 150, 150)
        cropped = analyzer._crop_figure(img, bbox, normalized=False)

        # Should clamp to (0, 0, 100, 100)
        assert cropped.size == (100, 100)


class TestAnalyzeSingleFigure:
    """Test single figure analysis."""

    @pytest.mark.asyncio
    async def test_analyze_single_figure_success(self):
        """Test successful figure analysis."""
        analyzer = FigureAnalyzer(vision_api_url="http://test:8000/vision")

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "description": "A bar chart showing quarterly revenue"
        }

        with patch.object(analyzer, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            description = await analyzer._analyze_single_figure(
                image_base64="fake_base64_data",
                figure_id="fig1",
            )

            assert description == "A bar chart showing quarterly revenue"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_single_figure_http_error(self):
        """Test figure analysis with HTTP error."""
        analyzer = FigureAnalyzer()

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch.object(analyzer, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            description = await analyzer._analyze_single_figure("fake_data", "fig1")

            assert description.startswith("[Analysis failed: HTTP 500]")

    @pytest.mark.asyncio
    async def test_analyze_single_figure_timeout(self):
        """Test figure analysis with timeout."""
        analyzer = FigureAnalyzer()

        with patch.object(analyzer, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timeout")
            mock_get_client.return_value = mock_client

            description = await analyzer._analyze_single_figure("fake_data", "fig1")

            assert description == "[Analysis timeout]"

    @pytest.mark.asyncio
    async def test_analyze_single_figure_exception(self):
        """Test figure analysis with general exception."""
        analyzer = FigureAnalyzer()

        with patch.object(analyzer, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("Network error")
            mock_get_client.return_value = mock_client

            description = await analyzer._analyze_single_figure("fake_data", "fig1")

            assert description.startswith("[Analysis error:")
            assert "Network error" in description

    @pytest.mark.asyncio
    async def test_analyze_single_figure_empty_description(self):
        """Test figure analysis returning empty description."""
        analyzer = FigureAnalyzer()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"description": ""}

        with patch.object(analyzer, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            description = await analyzer._analyze_single_figure("fake_data", "fig1")

            assert description == "[No description generated]"


class TestAnalyzeFigures:
    """Test batch figure analysis."""

    @pytest.mark.asyncio
    async def test_analyze_figures_empty_list(self):
        """Test analyzing empty figure list."""
        analyzer = FigureAnalyzer()
        result = await analyzer.analyze_figures("/fake/path.pdf", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_figures_pdf_not_found(self):
        """Test analyzing figures with non-existent PDF."""
        analyzer = FigureAnalyzer()
        bbox = BBox(x1=0, y1=0, x2=100, y2=100, normalized=True)
        figures = [FigureRef(id="fig1", page=1, bbox=bbox)]

        result = await analyzer.analyze_figures("/nonexistent/path.pdf", figures)

        # Should return figures unchanged (logged error)
        assert len(result) == 1
        assert result[0].description is None

    @pytest.mark.asyncio
    async def test_analyze_figures_success(self):
        """Test successful batch figure analysis."""
        analyzer = FigureAnalyzer()

        # Create mock figures
        bbox1 = BBox(x1=0, y1=0, x2=500, y2=500, normalized=True)
        bbox2 = BBox(x1=500, y1=0, x2=1000, y2=500, normalized=True)
        figures = [
            FigureRef(id="fig1", page=1, bbox=bbox1),
            FigureRef(id="fig2", page=1, bbox=bbox2),
        ]

        # Mock PDF rendering
        mock_image = Image.new("RGB", (1000, 1000), color="white")

        # Mock vision API responses
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"description": "A test figure"}

        with patch.object(analyzer, "_render_pdf_page", return_value=mock_image):
            with patch.object(analyzer, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_get_client.return_value = mock_client

                with patch("pathlib.Path.exists", return_value=True):
                    result = await analyzer.analyze_figures("/fake/path.pdf", figures)

                    assert len(result) == 2
                    assert result[0].description == "A test figure"
                    assert result[1].description == "A test figure"
                    assert result[0].image_base64 is not None
                    assert result[1].image_base64 is not None

    @pytest.mark.asyncio
    async def test_analyze_figures_page_render_error(self):
        """Test figure analysis when page rendering fails."""
        analyzer = FigureAnalyzer()

        bbox = BBox(x1=0, y1=0, x2=100, y2=100, normalized=True)
        figures = [FigureRef(id="fig1", page=1, bbox=bbox)]

        with patch.object(analyzer, "_render_pdf_page", side_effect=Exception("Render failed")):
            with patch("pathlib.Path.exists", return_value=True):
                result = await analyzer.analyze_figures("/fake/path.pdf", figures)

                # Should handle error gracefully
                assert len(result) == 1
                assert result[0].description is None


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_figure_analyzer_singleton(self):
        """Test singleton pattern for get_figure_analyzer."""
        analyzer1 = get_figure_analyzer()
        analyzer2 = get_figure_analyzer()
        assert analyzer1 is analyzer2

    @pytest.mark.asyncio
    async def test_analyze_figures_function_default_prompt(self):
        """Test analyze_figures convenience function with default prompt."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=100, normalized=True)
        figures = [FigureRef(id="fig1", page=1, bbox=bbox)]

        mock_image = Image.new("RGB", (1000, 1000), color="white")
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"description": "Test description"}

        with patch("src.services.figure_analyzer.FigureAnalyzer._render_pdf_page", return_value=mock_image):
            with patch("src.services.figure_analyzer.FigureAnalyzer._get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_get_client.return_value = mock_client

                with patch("pathlib.Path.exists", return_value=True):
                    result = await analyze_figures("/fake/path.pdf", figures)

                    assert len(result) == 1
                    assert result[0].description == "Test description"

    @pytest.mark.asyncio
    async def test_analyze_figures_function_custom_prompt(self):
        """Test analyze_figures convenience function with custom prompt."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=100, normalized=True)
        figures = [FigureRef(id="fig1", page=1, bbox=bbox)]

        mock_image = Image.new("RGB", (1000, 1000), color="white")
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"description": "Custom description"}

        with patch("src.services.figure_analyzer.FigureAnalyzer._render_pdf_page", return_value=mock_image):
            with patch("src.services.figure_analyzer.FigureAnalyzer._get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_get_client.return_value = mock_client

                with patch("pathlib.Path.exists", return_value=True):
                    # Get singleton and check that prompt is restored after
                    analyzer = get_figure_analyzer()
                    original_prompt = analyzer.vl_prompt

                    result = await analyze_figures(
                        "/fake/path.pdf",
                        figures,
                        vl_prompt="Custom prompt for analysis",
                    )

                    assert len(result) == 1
                    assert result[0].description == "Custom description"
                    # Verify prompt was restored
                    assert analyzer.vl_prompt == original_prompt

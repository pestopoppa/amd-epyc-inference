"""Figure analysis using vision model.

This module provides functionality to analyze figures detected in documents
by cropping the figure regions and sending them to the vision API.

The figure analyzer:
1. Takes a document path and list of FigureRef objects (with bboxes)
2. Renders relevant pages to images using pypdfium2
3. Crops figure regions based on bounding box coordinates
4. Calls the vision API for each figure (with concurrency control)
5. Returns FigureRef objects with populated descriptions
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from PIL import Image

from src.models.document import FigureRef

logger = logging.getLogger(__name__)

# Vision API configuration — sourced from centralized config
from src.config import get_config as _get_config

VISION_API_URL = _get_config().server_urls.vision_api
VISION_TIMEOUT = _get_config().timeouts.vision_figure
MAX_CONCURRENT_ANALYSIS = _get_config().delegation.max_concurrent_analysis

# Default prompt for figure description
DEFAULT_FIGURE_PROMPT = (
    "Describe this figure from a technical document. "
    "Focus on: what type of visualization it is (chart, diagram, graph, etc.), "
    "the data or concepts it represents, and any key takeaways. "
    "Be concise but informative."
)


class FigureAnalyzer:
    """Analyze figures using the vision model.

    Usage:
        analyzer = FigureAnalyzer()
        figures_with_descriptions = await analyzer.analyze_figures(
            pdf_path="/path/to/doc.pdf",
            figures=figure_refs,
        )
    """

    def __init__(
        self,
        vision_api_url: str = VISION_API_URL,
        timeout: float = VISION_TIMEOUT,
        max_concurrent: int = MAX_CONCURRENT_ANALYSIS,
        vl_prompt: str | None = None,
    ):
        """Initialize the figure analyzer.

        Args:
            vision_api_url: URL of the vision API endpoint.
            timeout: Timeout per figure analysis in seconds.
            max_concurrent: Maximum concurrent vision API calls.
            vl_prompt: Custom prompt for figure description.
        """
        self.vision_api_url = vision_api_url
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.vl_prompt = vl_prompt or DEFAULT_FIGURE_PROMPT
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0)
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "FigureAnalyzer":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def _render_pdf_page(self, pdf_path: Path, page_num: int, dpi: int = 200) -> "Image.Image":
        """Render a single PDF page to an image.

        Args:
            pdf_path: Path to the PDF file.
            page_num: Page number (1-indexed).
            dpi: Resolution for rendering.

        Returns:
            PIL Image of the rendered page.
        """
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[page_num - 1]  # pypdfium uses 0-indexed pages
        scale = dpi / 72
        bitmap = page.render(scale=scale)
        return bitmap.to_pil()

    def _crop_figure(
        self,
        page_image: "Image.Image",
        bbox: tuple[int, int, int, int],
        normalized: bool = True,
    ) -> "Image.Image":
        """Crop a figure region from a page image.

        Args:
            page_image: Full page image.
            bbox: Bounding box (x1, y1, x2, y2).
            normalized: If True, coords are 0-1000 normalized.

        Returns:
            Cropped image of the figure.
        """
        x1, y1, x2, y2 = bbox

        if normalized:
            # Convert normalized coords (0-1000) to pixel coords
            w, h = page_image.size
            x1 = x1 * w // 1000
            y1 = y1 * h // 1000
            x2 = x2 * w // 1000
            y2 = y2 * h // 1000

        # Ensure valid crop bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(page_image.width, x2)
        y2 = min(page_image.height, y2)

        return page_image.crop((x1, y1, x2, y2))

    def _image_to_base64(self, image: "Image.Image", format: str = "PNG") -> str:
        """Convert PIL Image to base64 string.

        Args:
            image: PIL Image to convert.
            format: Output format (PNG, JPEG, etc.).

        Returns:
            Base64-encoded image string.
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def _analyze_single_figure(
        self,
        image_base64: str,
        figure_id: str,
    ) -> str:
        """Send a single figure to the vision API for analysis.

        Args:
            image_base64: Base64-encoded figure image.
            figure_id: Figure ID for logging.

        Returns:
            Description string from the vision model.
        """
        client = await self._get_client()

        try:
            response = await client.post(
                self.vision_api_url,
                json={
                    "image_base64": image_base64,
                    "analyzers": ["vl_describe"],
                    "vl_prompt": self.vl_prompt,
                    "store_results": False,  # Don't store in vision DB
                },
            )

            if response.status_code != 200:
                logger.warning(
                    f"Vision API error for {figure_id}: {response.status_code} - {response.text}"
                )
                return f"[Analysis failed: HTTP {response.status_code}]"

            data = response.json()
            description = data.get("description", "")

            if not description:
                return "[No description generated]"

            return description

        except httpx.TimeoutException:
            logger.warning(f"Vision API timeout for {figure_id}")
            return "[Analysis timeout]"
        except Exception as e:
            logger.exception(f"Vision API error for {figure_id}: {e}")
            return f"[Analysis error: {e}]"

    async def analyze_figures(
        self,
        pdf_path: str | Path,
        figures: list[FigureRef],
        dpi: int = 200,
    ) -> list[FigureRef]:
        """Analyze all figures in a document.

        Args:
            pdf_path: Path to the PDF document.
            figures: List of FigureRef objects with bboxes.
            dpi: Resolution for page rendering.

        Returns:
            List of FigureRef objects with descriptions populated.
        """
        if not figures:
            return figures

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return figures

        # Group figures by page to minimize page renders
        figures_by_page: dict[int, list[tuple[int, FigureRef]]] = defaultdict(list)
        for idx, fig in enumerate(figures):
            figures_by_page[fig.page].append((idx, fig))

        logger.info(
            f"Analyzing {len(figures)} figures across {len(figures_by_page)} pages"
        )

        # Prepare analysis tasks
        async def analyze_with_semaphore(
            semaphore: asyncio.Semaphore,
            page_image: "Image.Image",
            fig: FigureRef,
            idx: int,
        ) -> tuple[int, str]:
            """Analyze a figure with semaphore-controlled concurrency."""
            async with semaphore:
                # Crop the figure region
                bbox = (fig.bbox.x1, fig.bbox.y1, fig.bbox.x2, fig.bbox.y2)
                cropped = self._crop_figure(page_image, bbox, normalized=fig.bbox.normalized)

                # Convert to base64
                image_b64 = self._image_to_base64(cropped)

                # Optionally store the base64 in the figure
                fig.image_base64 = image_b64

                # Get description from vision model
                description = await self._analyze_single_figure(image_b64, fig.id)

                return (idx, description)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Process page by page
        tasks = []
        page_images: dict[int, "Image.Image"] = {}

        for page_num, page_figures in figures_by_page.items():
            # Render page once for all figures on it
            try:
                page_image = self._render_pdf_page(pdf_path, page_num, dpi)
                page_images[page_num] = page_image
            except Exception as e:
                logger.error(f"Failed to render page {page_num}: {e}")
                continue

            # Create tasks for all figures on this page
            for idx, fig in page_figures:
                task = asyncio.create_task(
                    analyze_with_semaphore(semaphore, page_image, fig, idx)
                )
                tasks.append(task)

        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update figure descriptions
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Figure analysis failed: {result}")
                continue

            idx, description = result
            figures[idx].description = description

        logger.info(f"Completed analysis of {len(figures)} figures")

        return figures


# Singleton instance
_analyzer: FigureAnalyzer | None = None


def get_figure_analyzer() -> FigureAnalyzer:
    """Get the singleton figure analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = FigureAnalyzer()
    return _analyzer


async def analyze_figures(
    pdf_path: str | Path,
    figures: list[FigureRef],
    vl_prompt: str | None = None,
) -> list[FigureRef]:
    """Convenience function to analyze figures in a document.

    Args:
        pdf_path: Path to the PDF document.
        figures: List of FigureRef objects with bboxes.
        vl_prompt: Optional custom prompt for figure description.

    Returns:
        List of FigureRef objects with descriptions populated.
    """
    analyzer = get_figure_analyzer()

    if vl_prompt:
        original_prompt = analyzer.vl_prompt
        analyzer.vl_prompt = vl_prompt
        try:
            return await analyzer.analyze_figures(pdf_path, figures)
        finally:
            analyzer.vl_prompt = original_prompt
    else:
        return await analyzer.analyze_figures(pdf_path, figures)

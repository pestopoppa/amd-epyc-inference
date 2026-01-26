"""HTTP client for LightOnOCR document processing server.

This module provides an async client for the LightOnOCR-2 server running
on port 9001. It handles:
- Single image OCR
- Multi-page PDF processing
- Partial success handling (continue on page failures)
- Async job submission for large documents
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from src.models.document import (
    BoundingBox,
    DocumentJobStatus,
    DocumentProcessRequest,
    OCRResult,
    PageOCRResult,
    ProcessingStatus,
)

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Default server URL
DEFAULT_OCR_URL = "http://localhost:9001"

# Timeouts
SINGLE_PAGE_TIMEOUT = 120.0  # 2 minutes for single page
PDF_TIMEOUT = 600.0  # 10 minutes for full PDF
HEALTH_CHECK_TIMEOUT = 5.0


class OCRServerError(Exception):
    """Error from the OCR server."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class OCRServerUnavailable(OCRServerError):
    """OCR server is not available."""

    pass


class DocumentFormalizerClient:
    """Async HTTP client for LightOnOCR-2 document processing.

    Usage:
        client = DocumentFormalizerClient()

        # Check server health
        if await client.health_check():
            # Process a PDF
            result = await client.ocr_pdf("/path/to/doc.pdf")
            print(result.full_text)
    """

    def __init__(self, base_url: str = DEFAULT_OCR_URL):
        """Initialize the client.

        Args:
            base_url: Base URL of the LightOnOCR server (default: http://localhost:9001)
        """
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(PDF_TIMEOUT, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> DocumentFormalizerClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def health_check(self) -> bool:
        """Check if the OCR server is healthy.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=HEALTH_CHECK_TIMEOUT)
            return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    async def ocr_image(
        self,
        image: Image.Image | bytes | str,
        output_format: str = "bbox",
    ) -> PageOCRResult:
        """OCR a single image.

        Args:
            image: PIL Image, bytes, or base64 string.
            output_format: Output format ("text", "bbox", or "json").

        Returns:
            PageOCRResult with text and bounding boxes.

        Raises:
            OCRServerError: If the server returns an error.
            OCRServerUnavailable: If the server is not available.
        """
        # Convert image to base64 if needed
        if isinstance(image, str):
            # Already base64
            image_b64 = image
        elif isinstance(image, bytes):
            image_b64 = base64.b64encode(image).decode("utf-8")
        else:
            # PIL Image
            import io

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        try:
            client = await self._get_client()
            response = await client.post(
                "/v1/document/ocr",
                data={"image": image_b64, "output_format": output_format},
                timeout=SINGLE_PAGE_TIMEOUT,
            )

            if response.status_code != 200:
                raise OCRServerError(
                    f"OCR failed: {response.text}",
                    status_code=response.status_code,
                )

            data = response.json()
            return PageOCRResult.from_dict(data)

        except httpx.ConnectError as e:
            raise OCRServerUnavailable(f"OCR server not available: {e}") from e
        except httpx.TimeoutException as e:
            raise OCRServerError(f"OCR request timed out: {e}") from e

    async def ocr_pdf(
        self,
        path: str | Path,
        output_format: str = "bbox",
        max_pages: int = 100,
    ) -> OCRResult:
        """OCR an entire PDF document.

        Args:
            path: Path to the PDF file.
            output_format: Output format ("text", "bbox", or "json").
            max_pages: Maximum pages to process.

        Returns:
            OCRResult with all pages.

        Raises:
            OCRServerError: If the server returns an error.
            OCRServerUnavailable: If the server is not available.
            FileNotFoundError: If the PDF file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            client = await self._get_client()

            with open(path, "rb") as f:
                response = await client.post(
                    "/v1/document/pdf",
                    files={"file": (path.name, f, "application/pdf")},
                    data={"output_format": output_format, "max_pages": str(max_pages)},
                    timeout=PDF_TIMEOUT,
                )

            if response.status_code != 200:
                raise OCRServerError(
                    f"PDF OCR failed: {response.text}",
                    status_code=response.status_code,
                )

            data = response.json()
            return OCRResult.from_dict(data)

        except httpx.ConnectError as e:
            raise OCRServerUnavailable(f"OCR server not available: {e}") from e
        except httpx.TimeoutException as e:
            raise OCRServerError(f"PDF OCR timed out: {e}") from e

    async def ocr_pdf_with_partial_success(
        self,
        path: str | Path,
        output_format: str = "bbox",
        max_pages: int = 100,
        page_timeout: float = 120.0,
        max_concurrent: int = 8,
    ) -> OCRResult:
        """OCR a PDF with partial success handling.

        Unlike ocr_pdf(), this method processes pages individually and
        continues even if some pages fail. Failed pages are recorded
        in the result.

        Pages are processed in parallel (up to max_concurrent) for speed.

        Args:
            path: Path to the PDF file.
            output_format: Output format ("text", "bbox", or "json").
            max_pages: Maximum pages to process.
            page_timeout: Timeout per page in seconds.
            max_concurrent: Maximum concurrent page requests (default: 8).

        Returns:
            OCRResult with successful pages and failed_pages list.
        """
        import time

        import pypdfium2 as pdfium

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        # Get page count
        pdf = pdfium.PdfDocument(path)
        num_pages = min(len(pdf), max_pages)

        total_start = time.time()

        # Render all pages to images first (CPU-bound, fast)
        page_images: list[tuple[int, "Image.Image"]] = []
        for page_num in range(num_pages):
            page = pdf[page_num]
            scale = 200 / 72  # 200 DPI
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            page_images.append((page_num + 1, image))

        # Process pages in parallel with semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[tuple[int, PageOCRResult | Exception]] = []

        async def process_page(page_num: int, image: "Image.Image") -> tuple[int, PageOCRResult | Exception]:
            async with semaphore:
                try:
                    result = await self.ocr_image(image, output_format)
                    result.page = page_num
                    return (page_num, result)
                except Exception as e:
                    logger.warning(f"Failed to process page {page_num}: {e}")
                    return (page_num, e)

        # Launch all tasks in parallel
        tasks = [process_page(page_num, image) for page_num, image in page_images]
        results = await asyncio.gather(*tasks)

        # Separate successes and failures, maintaining page order
        pages: list[PageOCRResult] = []
        failed_pages: list[dict] = []

        for page_num, result in sorted(results, key=lambda x: x[0]):
            if isinstance(result, Exception):
                failed_pages.append(
                    {
                        "page": page_num,
                        "error": str(result),
                        "timestamp": time.time(),
                    }
                )
            else:
                pages.append(result)

        total_elapsed = time.time() - total_start
        pages_per_sec = len(pages) / total_elapsed if total_elapsed > 0 else 0

        status = ProcessingStatus.COMPLETED
        if failed_pages:
            if pages:
                status = ProcessingStatus.PARTIAL
            else:
                status = ProcessingStatus.FAILED

        return OCRResult(
            pages=pages,
            total_pages=num_pages,
            elapsed_sec=total_elapsed,
            pages_per_sec=pages_per_sec,
            failed_pages=failed_pages,
            status=status,
        )


# Singleton instance
_client: DocumentFormalizerClient | None = None


def get_document_client() -> DocumentFormalizerClient:
    """Get the singleton document client instance."""
    global _client
    if _client is None:
        _client = DocumentFormalizerClient()
    return _client


async def process_document(request: DocumentProcessRequest) -> OCRResult:
    """Process a document with the given request parameters.

    This is a convenience function that handles:
    - File path vs base64 input
    - Sync vs async processing based on page count
    - Partial success handling

    Args:
        request: Document processing request.

    Returns:
        OCRResult with processed pages.
    """
    client = get_document_client()

    if request.file_path is None and request.file_base64 is None:
        raise ValueError("Either file_path or file_base64 must be provided")

    if request.file_path:
        path = Path(request.file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            # Use server-side PDF processing (faster, handles parallelism internally)
            return await client.ocr_pdf(
                path,
                output_format=request.output_format,
                max_pages=request.max_pages,
            )
        elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
            # Single image
            from PIL import Image

            image = Image.open(path)
            page_result = await client.ocr_image(image, request.output_format)
            return OCRResult(
                pages=[page_result],
                total_pages=1,
                elapsed_sec=page_result.elapsed_sec,
                pages_per_sec=1.0 / page_result.elapsed_sec if page_result.elapsed_sec > 0 else 0,
            )
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    else:
        # Base64 encoded image
        image_bytes = base64.b64decode(request.file_base64)
        page_result = await client.ocr_image(image_bytes, request.output_format)
        return OCRResult(
            pages=[page_result],
            total_pages=1,
            elapsed_sec=page_result.elapsed_sec,
            pages_per_sec=1.0 / page_result.elapsed_sec if page_result.elapsed_sec > 0 else 0,
        )

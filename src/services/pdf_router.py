"""PDF Router - Intelligent routing between pdftotext and LightOnOCR.

Routes PDF processing based on content type:
- Born-digital PDFs: pdftotext (fast) + PyMuPDF (figure extraction)
- Scanned/image PDFs: LightOnOCR (OCR with bounding boxes)

Architecture:
    PDF Input
        ↓
    [pdftotext probe] → Quick text extraction (~100ms)
        ↓
    [Quality check] → Is text readable? (entropy, char ratio)
        │
        ├─ YES (born-digital):
        │   ├─ Text: pdftotext output
        │   └─ Figures: PyMuPDF extracts with bboxes
        │
        └─ NO (scanned/image):
            └─ LightOnOCR (text + bboxes)
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box for an extracted figure."""

    x0: float  # Left edge (0-1 normalized)
    y0: float  # Top edge (0-1 normalized)
    x1: float  # Right edge
    y1: float  # Bottom edge
    page: int  # Page number (1-indexed)
    width_px: int = 0  # Original pixel width
    height_px: int = 0  # Original pixel height


@dataclass
class ExtractedFigure:
    """An extracted figure/image from the PDF."""

    index: int  # Figure number (1, 2, 3...)
    bbox: BoundingBox
    image_path: Optional[str] = None  # Path to extracted image file
    image_bytes: Optional[bytes] = None  # Raw image bytes
    format: str = "png"  # Image format


@dataclass
class PDFExtractionResult:
    """Result of PDF text and figure extraction."""

    text: str
    figures: list[ExtractedFigure] = field(default_factory=list)
    page_count: int = 0
    method: str = "unknown"  # "pdftotext", "lightonocr", "hybrid"
    quality_score: float = 0.0  # 0-1, higher = better text quality
    latency_ms: float = 0.0
    ocr_required: bool = False


class PDFRouter:
    """Routes PDF processing to optimal extraction method."""

    # Quality thresholds for text extraction
    MIN_ENTROPY = 3.5  # Minimum Shannon entropy for readable text
    MAX_GARBAGE_RATIO = 0.15  # Max ratio of non-printable chars
    MIN_WORD_LENGTH_AVG = 2.5  # Average word length threshold
    MIN_TEXT_LENGTH = 100  # Minimum chars for quality assessment

    def __init__(
        self,
        lightonocr_url: str = "http://localhost:9001",
        temp_dir: str = "/mnt/raid0/llm/tmp/pdf_router",
        pdftotext_path: str = "pdftotext",
    ):
        """Initialize PDF router.

        Args:
            lightonocr_url: URL of LightOnOCR server for OCR fallback
            temp_dir: Directory for temporary files
            pdftotext_path: Path to pdftotext binary
        """
        self.lightonocr_url = lightonocr_url
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.pdftotext_path = pdftotext_path

        # Check for PyMuPDF
        try:
            import fitz

            self._has_pymupdf = True
        except ImportError:
            self._has_pymupdf = False
            logger.warning("PyMuPDF not available - figure extraction disabled")

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0

        # Count character frequencies
        freq = Counter(text)
        total = len(text)

        # Calculate entropy
        entropy = 0.0
        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def _calculate_garbage_ratio(self, text: str) -> float:
        """Calculate ratio of non-printable/garbage characters."""
        if not text:
            return 1.0

        # Count printable ASCII + common unicode
        printable = sum(
            1
            for c in text
            if c.isprintable() or c in "\n\t\r"
        )

        return 1.0 - (printable / len(text))

    def _calculate_avg_word_length(self, text: str) -> float:
        """Calculate average word length."""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    def _assess_text_quality(self, text: str) -> tuple[float, bool]:
        """Assess quality of extracted text.

        Returns:
            (quality_score, needs_ocr): Score 0-1 and whether OCR is needed
        """
        if len(text) < self.MIN_TEXT_LENGTH:
            return 0.0, True

        entropy = self._calculate_entropy(text)
        garbage_ratio = self._calculate_garbage_ratio(text)
        avg_word_len = self._calculate_avg_word_length(text)

        # Score components (0-1 each)
        entropy_score = min(1.0, entropy / 5.0)  # 5.0 is typical for English
        garbage_score = 1.0 - min(1.0, garbage_ratio / 0.3)
        word_len_score = min(1.0, avg_word_len / 5.0)

        # Weighted average
        quality_score = (
            entropy_score * 0.4 + garbage_score * 0.4 + word_len_score * 0.2
        )

        # Determine if OCR needed
        needs_ocr = (
            entropy < self.MIN_ENTROPY
            or garbage_ratio > self.MAX_GARBAGE_RATIO
            or avg_word_len < self.MIN_WORD_LENGTH_AVG
        )

        logger.debug(
            f"Text quality: entropy={entropy:.2f}, garbage={garbage_ratio:.2%}, "
            f"word_len={avg_word_len:.1f}, score={quality_score:.2f}, ocr={needs_ocr}"
        )

        return quality_score, needs_ocr

    def _extract_with_pdftotext(self, pdf_path: Path) -> tuple[str, float]:
        """Extract text using pdftotext (fast path).

        Returns:
            (text, latency_ms)
        """
        start = time.perf_counter()

        try:
            result = subprocess.run(
                [self.pdftotext_path, "-layout", str(pdf_path), "-"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            if result.returncode != 0:
                logger.warning(f"pdftotext failed: {result.stderr}")
                return "", latency_ms

            return result.stdout, latency_ms

        except subprocess.TimeoutExpired:
            return "", (time.perf_counter() - start) * 1000
        except FileNotFoundError:
            logger.error(f"pdftotext not found at {self.pdftotext_path}")
            return "", 0.0

    def _extract_figures_pymupdf(
        self, pdf_path: Path, output_dir: Optional[Path] = None
    ) -> list[ExtractedFigure]:
        """Extract figures using PyMuPDF with bounding boxes.

        Args:
            pdf_path: Path to PDF file
            output_dir: Optional directory to save extracted images

        Returns:
            List of ExtractedFigure objects
        """
        if not self._has_pymupdf:
            return []

        import fitz

        figures = []
        figure_index = 0

        try:
            doc = fitz.open(str(pdf_path))

            for page_num, page in enumerate(doc, start=1):
                page_rect = page.rect
                page_width = page_rect.width
                page_height = page_rect.height

                # Get images on this page
                image_list = page.get_images(full=True)

                for img_info in image_list:
                    xref = img_info[0]

                    try:
                        # Get image bounding box
                        img_rects = page.get_image_rects(xref)
                        if not img_rects:
                            continue

                        rect = img_rects[0]  # Use first occurrence

                        # Normalize bbox to 0-1
                        bbox = BoundingBox(
                            x0=rect.x0 / page_width,
                            y0=rect.y0 / page_height,
                            x1=rect.x1 / page_width,
                            y1=rect.y1 / page_height,
                            page=page_num,
                            width_px=int(rect.width),
                            height_px=int(rect.height),
                        )

                        figure_index += 1
                        figure = ExtractedFigure(
                            index=figure_index,
                            bbox=bbox,
                        )

                        # Extract image bytes if output_dir provided
                        if output_dir:
                            try:
                                base_image = doc.extract_image(xref)
                                if base_image:
                                    ext = base_image.get("ext", "png")
                                    img_path = output_dir / f"figure_{figure_index}.{ext}"
                                    img_path.write_bytes(base_image["image"])
                                    figure.image_path = str(img_path)
                                    figure.format = ext
                                    figure.image_bytes = base_image["image"]
                            except Exception as e:
                                logger.debug(f"Could not extract image {xref}: {e}")

                        figures.append(figure)

                    except Exception as e:
                        logger.debug(f"Error processing image {xref}: {e}")
                        continue

            doc.close()

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")

        return figures

    async def _extract_with_lightonocr(
        self, pdf_path: Path
    ) -> tuple[str, list[ExtractedFigure], float]:
        """Extract text and figures using LightOnOCR.

        Returns:
            (text, figures, latency_ms)
        """
        import httpx

        start = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                with open(pdf_path, "rb") as f:
                    files = {"file": (pdf_path.name, f, "application/pdf")}
                    response = await client.post(
                        f"{self.lightonocr_url}/ocr/pdf",
                        files=files,
                    )

                latency_ms = (time.perf_counter() - start) * 1000

                if response.status_code != 200:
                    logger.error(f"LightOnOCR failed: {response.text}")
                    return "", [], latency_ms

                data = response.json()

                # Extract text from pages
                text_parts = []
                figures = []
                figure_index = 0

                for page in data.get("pages", []):
                    text_parts.append(page.get("text", ""))

                    # Extract bounding boxes
                    for bbox_data in page.get("bboxes", []):
                        figure_index += 1
                        figures.append(
                            ExtractedFigure(
                                index=figure_index,
                                bbox=BoundingBox(
                                    x0=bbox_data.get("x1", 0) / 1000,
                                    y0=bbox_data.get("y1", 0) / 1000,
                                    x1=bbox_data.get("x2", 0) / 1000,
                                    y1=bbox_data.get("y2", 0) / 1000,
                                    page=page.get("page", 1),
                                ),
                            )
                        )

                return "\n\n".join(text_parts), figures, latency_ms

        except Exception as e:
            logger.error(f"LightOnOCR request failed: {e}")
            return "", [], (time.perf_counter() - start) * 1000

    async def extract(
        self,
        pdf_path: str | Path,
        force_ocr: bool = False,
        extract_figures: bool = True,
        save_figures: bool = False,
    ) -> PDFExtractionResult:
        """Extract text and figures from PDF.

        Args:
            pdf_path: Path to PDF file
            force_ocr: Force LightOnOCR even for born-digital PDFs
            extract_figures: Whether to extract figures
            save_figures: Whether to save extracted figures to disk

        Returns:
            PDFExtractionResult with text, figures, and metadata
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        total_start = time.perf_counter()

        # Get page count using PyMuPDF if available
        page_count = 0
        if self._has_pymupdf:
            import fitz

            try:
                doc = fitz.open(str(pdf_path))
                page_count = len(doc)
                doc.close()
            except Exception:
                pass

        # Step 1: Try pdftotext (fast path)
        if not force_ocr:
            text, pdftotext_latency = self._extract_with_pdftotext(pdf_path)
            quality_score, needs_ocr = self._assess_text_quality(text)

            if not needs_ocr and text:
                # Good quality text - use fast path
                figures = []
                if extract_figures:
                    output_dir = self.temp_dir / pdf_path.stem if save_figures else None
                    if output_dir:
                        output_dir.mkdir(parents=True, exist_ok=True)
                    figures = self._extract_figures_pymupdf(pdf_path, output_dir)

                total_latency = (time.perf_counter() - total_start) * 1000

                logger.info(
                    f"PDF extracted via pdftotext: {len(text)} chars, "
                    f"{len(figures)} figures in {total_latency:.0f}ms"
                )

                return PDFExtractionResult(
                    text=text,
                    figures=figures,
                    page_count=page_count,
                    method="pdftotext",
                    quality_score=quality_score,
                    latency_ms=total_latency,
                    ocr_required=False,
                )

        # Step 2: Fall back to LightOnOCR
        logger.info(f"Using LightOnOCR for {pdf_path.name} (OCR required)")

        text, figures, ocr_latency = await self._extract_with_lightonocr(pdf_path)

        total_latency = (time.perf_counter() - total_start) * 1000

        # Re-assess quality of OCR output
        quality_score, _ = self._assess_text_quality(text)

        logger.info(
            f"PDF extracted via LightOnOCR: {len(text)} chars, "
            f"{len(figures)} figures in {total_latency:.0f}ms"
        )

        return PDFExtractionResult(
            text=text,
            figures=figures,
            page_count=page_count,
            method="lightonocr",
            quality_score=quality_score,
            latency_ms=total_latency,
            ocr_required=True,
        )

    def extract_sync(
        self,
        pdf_path: str | Path,
        force_ocr: bool = False,
        extract_figures: bool = True,
        save_figures: bool = False,
    ) -> PDFExtractionResult:
        """Synchronous wrapper for extract().

        For use in non-async contexts.
        """
        import asyncio

        return asyncio.run(
            self.extract(pdf_path, force_ocr, extract_figures, save_figures)
        )


# Convenience function
def extract_pdf(
    pdf_path: str | Path,
    force_ocr: bool = False,
    extract_figures: bool = True,
) -> PDFExtractionResult:
    """Extract text and figures from PDF using optimal method.

    Args:
        pdf_path: Path to PDF file
        force_ocr: Force LightOnOCR even for born-digital PDFs
        extract_figures: Whether to extract figures

    Returns:
        PDFExtractionResult with text, figures, and metadata
    """
    router = PDFRouter()
    return router.extract_sync(pdf_path, force_ocr, extract_figures)

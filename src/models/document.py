"""Document processing models for LightOnOCR pipeline.

This module defines data classes for:
- OCR results (pages, bboxes, figures)
- Document preprocessing results (sections, figures)
- Processing status and error handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ProcessingStatus(Enum):
    """Status of document processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some pages processed, some failed


@dataclass
class BoundingBox:
    """Bounding box for a figure or region in a document page.

    Coordinates are normalized to [0, 1000] range by LightOnOCR.
    """

    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    normalized: bool = True
    page: int = 1

    @property
    def width(self) -> int:
        """Width of bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area of bounding box."""
        return self.width * self.height

    def to_pixel_coords(self, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """Convert normalized coords to pixel coordinates.

        Args:
            img_width: Image width in pixels.
            img_height: Image height in pixels.

        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates.
        """
        if not self.normalized:
            return (self.x1, self.y1, self.x2, self.y2)

        return (
            self.x1 * img_width // 1000,
            self.y1 * img_height // 1000,
            self.x2 * img_width // 1000,
            self.y2 * img_height // 1000,
        )


@dataclass
class PageOCRResult:
    """OCR result for a single page."""

    page: int
    text: str
    bboxes: list[BoundingBox] = field(default_factory=list)
    elapsed_sec: float = 0.0
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageOCRResult:
        """Create from API response dict."""
        bboxes = [
            BoundingBox(
                id=b["id"],
                x1=b["x1"],
                y1=b["y1"],
                x2=b["x2"],
                y2=b["y2"],
                normalized=b.get("normalized", True),
                page=data.get("page", 1),
            )
            for b in data.get("bboxes", [])
        ]
        return cls(
            page=data.get("page", 1),
            text=data.get("text", ""),
            bboxes=bboxes,
            elapsed_sec=data.get("elapsed_sec", 0.0),
            error=data.get("error"),
        )


@dataclass
class OCRResult:
    """Complete OCR result for a document."""

    pages: list[PageOCRResult]
    total_pages: int
    elapsed_sec: float
    pages_per_sec: float
    failed_pages: list[dict[str, Any]] = field(default_factory=list)
    job_id: str | None = None
    status: ProcessingStatus = ProcessingStatus.COMPLETED

    @property
    def success_rate(self) -> float:
        """Fraction of pages processed successfully."""
        successful = len(self.pages)
        total = successful + len(self.failed_pages)
        return successful / total if total > 0 else 0.0

    @property
    def full_text(self) -> str:
        """Concatenate all page texts."""
        return "\n\n".join(p.text for p in self.pages)

    @property
    def all_bboxes(self) -> list[BoundingBox]:
        """Get all bounding boxes across all pages."""
        result = []
        for page in self.pages:
            result.extend(page.bboxes)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCRResult:
        """Create from API response dict."""
        pages = [PageOCRResult.from_dict(p) for p in data.get("pages", [])]

        # Determine status
        if data.get("status") == "queued":
            status = ProcessingStatus.PENDING
        elif data.get("failed_pages"):
            status = ProcessingStatus.PARTIAL
        elif data.get("error"):
            status = ProcessingStatus.FAILED
        else:
            status = ProcessingStatus.COMPLETED

        return cls(
            pages=pages,
            total_pages=data.get("total_pages", len(pages)),
            elapsed_sec=data.get("elapsed_sec", 0.0),
            pages_per_sec=data.get("pages_per_sec", 0.0),
            failed_pages=data.get("failed_pages", []),
            job_id=data.get("job_id"),
            status=status,
        )


@dataclass
class Section:
    """A semantic section of a document (by markdown header)."""

    id: str  # e.g., "s1", "s2"
    title: str  # Header text
    level: int  # 1, 2, or 3 (# = 1, ## = 2, ### = 3)
    content: str  # Section body text
    page_start: int = 1
    page_end: int = 1
    figure_ids: list[str] = field(default_factory=list)

    @property
    def page_range(self) -> tuple[int, int]:
        """Get (start_page, end_page) tuple."""
        return (self.page_start, self.page_end)


@dataclass
class FigureRef:
    """Reference to an extracted figure."""

    id: str  # e.g., "p1_fig0"
    page: int
    bbox: BoundingBox
    description: str = ""
    image_path: str | None = None  # Path to cropped image file
    image_base64: str | None = None  # Base64 encoded image
    section_id: str | None = None  # Which section contains this figure


@dataclass
class DocumentPreprocessResult:
    """Complete preprocessing result for a document.

    This is the enriched context that flows back to the orchestrator.
    """

    original_path: str
    sections: list[Section]
    figures: list[FigureRef]
    total_pages: int
    failed_pages: list[int] = field(default_factory=list)
    processing_time: float = 0.0
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error: str | None = None

    # Raw OCR result for reference
    ocr_result: OCRResult | None = None

    @property
    def success_rate(self) -> float:
        """Fraction of pages processed successfully."""
        successful = self.total_pages - len(self.failed_pages)
        return successful / self.total_pages if self.total_pages > 0 else 0.0

    def to_searchable_text(self) -> str:
        """Convert to searchable text for REPL context."""
        lines = []

        for section in self.sections:
            prefix = "#" * section.level
            lines.append(f"{prefix} {section.title}")
            lines.append(section.content)
            lines.append("")

        return "\n".join(lines)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert to metadata dict for REPL."""
        return {
            "document_path": self.original_path,
            "total_pages": self.total_pages,
            "sections": len(self.sections),
            "figures": [f.id for f in self.figures],
            "failed_pages": self.failed_pages,
            "processing_time": self.processing_time,
            "success_rate": self.success_rate,
        }


@dataclass
class DocumentProcessRequest:
    """Request to process a document."""

    file_path: str | None = None
    file_base64: str | None = None
    output_format: str = "bbox"  # "text", "bbox", or "json"
    max_pages: int = 100
    extract_figures: bool = True
    describe_figures: bool = False  # Whether to route figures to VL model
    async_mode: bool = False  # True for background processing
    dpi: int = 200  # Resolution for PDF rendering


@dataclass
class DocumentJobStatus:
    """Status of an async document processing job."""

    job_id: str
    status: ProcessingStatus
    pages_completed: int = 0
    total_pages: int = 0
    elapsed_sec: float = 0.0
    error: str | None = None
    result: DocumentPreprocessResult | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def progress(self) -> float:
        """Get progress as fraction 0-1."""
        if self.total_pages == 0:
            return 0.0
        return self.pages_completed / self.total_pages

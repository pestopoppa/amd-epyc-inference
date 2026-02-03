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

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize for session cache."""
        return {
            "id": self.id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "normalized": self.normalized,
            "page": self.page,
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "BoundingBox":
        """Deserialize from session cache."""
        return cls(
            id=data["id"],
            x1=data["x1"],
            y1=data["y1"],
            x2=data["x2"],
            y2=data["y2"],
            normalized=data.get("normalized", True),
            page=data.get("page", 1),
        )

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

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize for session cache."""
        return {
            "page": self.page,
            "text": self.text,
            "bboxes": [b.to_cache_dict() for b in self.bboxes],
            "elapsed_sec": self.elapsed_sec,
            "error": self.error,
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "PageOCRResult":
        """Deserialize from session cache."""
        return cls(
            page=data["page"],
            text=data["text"],
            bboxes=[BoundingBox.from_cache_dict(b) for b in data.get("bboxes", [])],
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

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize for session cache."""
        return {
            "pages": [p.to_cache_dict() for p in self.pages],
            "total_pages": self.total_pages,
            "elapsed_sec": self.elapsed_sec,
            "pages_per_sec": self.pages_per_sec,
            "failed_pages": self.failed_pages,
            "job_id": self.job_id,
            "status": self.status.value,
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "OCRResult":
        """Deserialize from session cache."""
        return cls(
            pages=[PageOCRResult.from_cache_dict(p) for p in data.get("pages", [])],
            total_pages=data["total_pages"],
            elapsed_sec=data.get("elapsed_sec", 0.0),
            pages_per_sec=data.get("pages_per_sec", 0.0),
            failed_pages=data.get("failed_pages", []),
            job_id=data.get("job_id"),
            status=ProcessingStatus(data.get("status", "completed")),
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

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize for session cache."""
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "figure_ids": self.figure_ids,
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "Section":
        """Deserialize from session cache."""
        return cls(
            id=data["id"],
            title=data["title"],
            level=data["level"],
            content=data["content"],
            page_start=data.get("page_start", 1),
            page_end=data.get("page_end", 1),
            figure_ids=data.get("figure_ids", []),
        )


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

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize for session cache.

        Note: image_base64 is NOT cached (too large, can be re-extracted).
        """
        return {
            "id": self.id,
            "page": self.page,
            "bbox": self.bbox.to_cache_dict(),
            "description": self.description,
            "image_path": self.image_path,
            # Intentionally omit image_base64 - it's too large
            "section_id": self.section_id,
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "FigureRef":
        """Deserialize from session cache."""
        return cls(
            id=data["id"],
            page=data["page"],
            bbox=BoundingBox.from_cache_dict(data["bbox"]),
            description=data.get("description", ""),
            image_path=data.get("image_path"),
            image_base64=None,  # Not cached - re-extract if needed
            section_id=data.get("section_id"),
        )


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

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize for session cache.

        This creates a JSON-serializable dict that can be stored in SQLite
        or written to a JSON file. The OCR result is included if available
        (it's expensive to recompute).

        Note: Figure images (image_base64) are NOT cached to save space.
        They can be re-extracted from the source document if needed.
        """
        return {
            "original_path": self.original_path,
            "sections": [s.to_cache_dict() for s in self.sections],
            "figures": [f.to_cache_dict() for f in self.figures],
            "total_pages": self.total_pages,
            "failed_pages": self.failed_pages,
            "processing_time": self.processing_time,
            "status": self.status.value,
            "error": self.error,
            "ocr_result": self.ocr_result.to_cache_dict() if self.ocr_result else None,
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "DocumentPreprocessResult":
        """Deserialize from session cache.

        This reconstructs the DocumentPreprocessResult from cached data.
        Figure images will be None (not cached) - re-extract if needed.
        """
        return cls(
            original_path=data["original_path"],
            sections=[Section.from_cache_dict(s) for s in data.get("sections", [])],
            figures=[FigureRef.from_cache_dict(f) for f in data.get("figures", [])],
            total_pages=data["total_pages"],
            failed_pages=data.get("failed_pages", []),
            processing_time=data.get("processing_time", 0.0),
            status=ProcessingStatus(data.get("status", "completed")),
            error=data.get("error"),
            ocr_result=(
                OCRResult.from_cache_dict(data["ocr_result"]) if data.get("ocr_result") else None
            ),
        )


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


@dataclass
class SearchHit:
    """A search result from multi-document search."""

    source_file: str  # Original filename in archive
    section_id: str | None  # Section ID if in a document
    line: int | None  # Line number if in text file
    match: str  # Matched text (with context)
    score: float = 1.0  # Relevance score


@dataclass
class MultiDocumentResult:
    """Aggregated result from processing multiple documents (e.g., from an archive).

    This provides unified access to content across multiple documents while
    maintaining source tracking for each piece of content.
    """

    source_archive: str  # Path to source archive
    documents: dict[str, DocumentPreprocessResult]  # filename -> result
    text_files: dict[str, str]  # filename -> raw text content (for non-document files)
    skipped_files: list[str]  # Files that couldn't be processed
    processing_time: float = 0.0

    @property
    def total_sections(self) -> int:
        """Total sections across all documents."""
        return sum(len(d.sections) for d in self.documents.values())

    @property
    def total_figures(self) -> int:
        """Total figures across all documents."""
        return sum(len(d.figures) for d in self.documents.values())

    @property
    def total_pages(self) -> int:
        """Total pages across all documents."""
        return sum(d.total_pages for d in self.documents.values())

    @property
    def merged_sections(self) -> list[tuple[str, Section]]:
        """Get all sections with source file tracking.

        Returns:
            List of (source_filename, section) tuples.
        """
        result = []
        for filename, doc in self.documents.items():
            for section in doc.sections:
                result.append((filename, section))
        return result

    @property
    def source_map(self) -> dict[str, str]:
        """Map section IDs to source filenames.

        Returns:
            Dict mapping "{filename}:{section_id}" to filename.
        """
        result = {}
        for filename, doc in self.documents.items():
            for section in doc.sections:
                key = f"{filename}:{section.id}"
                result[key] = filename
        return result

    def section(self, filename: str, section_id: str) -> Section | None:
        """Get a specific section by filename and section ID.

        Args:
            filename: Source filename.
            section_id: Section ID (e.g., "s0", "s1").

        Returns:
            Section or None if not found.
        """
        doc = self.documents.get(filename)
        if doc is None:
            return None

        for section in doc.sections:
            if section.id == section_id:
                return section
        return None

    def sections_from_file(self, filename: str) -> list[Section]:
        """Get all sections from a specific file.

        Args:
            filename: Source filename.

        Returns:
            List of sections.
        """
        doc = self.documents.get(filename)
        if doc is None:
            return []
        return doc.sections

    def search(self, query: str, case_sensitive: bool = False) -> list[SearchHit]:
        """Search across all documents and text files.

        Args:
            query: Search query string.
            case_sensitive: Whether to match case.

        Returns:
            List of SearchHit results.
        """
        import re

        results = []
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags)

        # Search in document sections
        for filename, doc in self.documents.items():
            for section in doc.sections:
                matches = list(pattern.finditer(section.content))
                for match in matches:
                    # Extract context around match
                    start = max(0, match.start() - 50)
                    end = min(len(section.content), match.end() + 50)
                    context = section.content[start:end]
                    if start > 0:
                        context = "..." + context
                    if end < len(section.content):
                        context = context + "..."

                    results.append(
                        SearchHit(
                            source_file=filename,
                            section_id=section.id,
                            line=None,
                            match=context,
                        )
                    )

        # Search in text files
        for filename, content in self.text_files.items():
            for i, line in enumerate(content.splitlines(), 1):
                if pattern.search(line):
                    results.append(
                        SearchHit(
                            source_file=filename,
                            section_id=None,
                            line=i,
                            match=line[:200],  # Truncate long lines
                        )
                    )

        return results

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert to summary dict for LLM context.

        Returns:
            Token-efficient summary.
        """
        return {
            "source": self.source_archive,
            "documents_processed": len(self.documents),
            "text_files": len(self.text_files),
            "skipped": len(self.skipped_files),
            "total_sections": self.total_sections,
            "total_figures": self.total_figures,
            "total_pages": self.total_pages,
            "processing_time": round(self.processing_time, 2),
            "files": {
                **{
                    f: {"type": "document", "sections": len(d.sections), "pages": d.total_pages}
                    for f, d in self.documents.items()
                },
                **{
                    f: {"type": "text", "lines": content.count("\n") + 1}
                    for f, content in self.text_files.items()
                },
            },
        }

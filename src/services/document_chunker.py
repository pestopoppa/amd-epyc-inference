"""Semantic document chunking by markdown headers.

LightOnOCR outputs structured markdown text. This module provides
chunking by header boundaries (# through ###) for efficient
context management and section-level access.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field

from src.models.document import (
    DocumentPreprocessResult,
    FigureRef,
    OCRResult,
    Section,
)

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    # Header levels to split on (1 = #, 2 = ##, 3 = ###)
    max_header_level: int = 3

    # Minimum section content length to keep
    min_section_length: int = 10

    # Maximum section length before forced splitting
    max_section_length: int = 10000

    # Figure extraction settings
    min_figure_area: int = 5000  # Minimum bbox area to consider a figure
    figure_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"image", "figure", "chart", "graph", "table", "diagram"})
    )


class DocumentChunker:
    """Chunk documents by markdown headers and extract figures.

    Usage:
        chunker = DocumentChunker()
        result = chunker.process(ocr_result, "/path/to/doc.pdf")

        for section in result.sections:
            print(f"{section.title}: {len(section.content)} chars")

        for fig in result.figures:
            print(f"Figure {fig.id} on page {fig.page}")
    """

    # Pattern to match markdown headers (# through ###)
    HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    # Pattern to match figure references in LightOnOCR output
    FIGURE_REF_PATTERN = re.compile(r"!\[image\]\(image_(\d+)\.png\)")

    def __init__(self, config: ChunkingConfig | None = None):
        """Initialize the chunker.

        Args:
            config: Optional chunking configuration.
        """
        self.config = config or ChunkingConfig()

    def chunk_by_headers(self, text: str) -> list[Section]:
        """Split text into sections based on markdown headers.

        Args:
            text: OCR text with markdown structure.

        Returns:
            List of Section objects.
        """
        sections: list[Section] = []
        current_section = Section(
            id="s0",
            title="Introduction",
            level=0,
            content="",
        )

        lines = text.split("\n")
        line_idx = 0

        for line in lines:
            match = self.HEADER_PATTERN.match(line)
            if match:
                # Save previous section if non-empty
                if (
                    current_section.content.strip()
                    and len(current_section.content.strip()) >= self.config.min_section_length
                ):
                    sections.append(current_section)

                # Start new section
                level = len(match.group(1))
                if level <= self.config.max_header_level:
                    section_id = f"s{len(sections)}"
                    current_section = Section(
                        id=section_id,
                        title=match.group(2).strip(),
                        level=level,
                        content="",
                    )
                else:
                    # Keep lower-level headers as content
                    current_section.content += line + "\n"
            else:
                current_section.content += line + "\n"

                # Force split if section too long
                if len(current_section.content) > self.config.max_section_length:
                    sections.append(current_section)
                    section_id = f"s{len(sections)}"
                    current_section = Section(
                        id=section_id,
                        title=f"{current_section.title} (continued)",
                        level=current_section.level,
                        content="",
                    )

            line_idx += 1

        # Don't forget last section
        if (
            current_section.content.strip()
            and len(current_section.content.strip()) >= self.config.min_section_length
        ):
            sections.append(current_section)

        # Assign figure IDs to sections
        self._assign_figures_to_sections(sections)

        return sections

    def _assign_figures_to_sections(self, sections: list[Section]) -> None:
        """Find figure references in section content and assign figure IDs."""
        for section in sections:
            # Find all figure references in this section's content
            matches = self.FIGURE_REF_PATTERN.findall(section.content)
            section.figure_ids = [f"fig_{m}" for m in matches]

    def extract_figures(
        self,
        ocr_result: OCRResult,
        sections: list[Section] | None = None,
    ) -> list[FigureRef]:
        """Extract figure references from OCR bounding boxes.

        Args:
            ocr_result: OCR result with bounding boxes.
            sections: Optional sections for assigning figures.

        Returns:
            List of FigureRef objects.
        """
        figures: list[FigureRef] = []

        for page in ocr_result.pages:
            for bbox in page.bboxes:
                # Filter by minimum area
                if bbox.area < self.config.min_figure_area:
                    continue

                fig_id = f"p{page.page}_fig{bbox.id}"
                figure = FigureRef(
                    id=fig_id,
                    page=page.page,
                    bbox=bbox,
                )

                # Try to find which section contains this figure
                if sections:
                    for section in sections:
                        if f"fig_{bbox.id}" in section.figure_ids:
                            figure.section_id = section.id
                            break

                figures.append(figure)

        return figures

    def assign_page_ranges(
        self,
        sections: list[Section],
        ocr_result: OCRResult,
    ) -> None:
        """Estimate which pages each section spans.

        This is approximate - based on proportional text distribution.

        Args:
            sections: List of sections to update.
            ocr_result: OCR result with page texts.
        """
        if not sections or not ocr_result.pages:
            return

        # Build cumulative character count per page
        page_char_counts: list[int] = []
        cumulative = 0
        for page in ocr_result.pages:
            cumulative += len(page.text)
            page_char_counts.append(cumulative)

        total_chars = cumulative
        if total_chars == 0:
            return

        # For each section, estimate page range based on character position
        section_start = 0
        for section in sections:
            section_end = section_start + len(section.content)

            # Find start page
            start_page = 1
            for i, cum_count in enumerate(page_char_counts):
                if cum_count > section_start:
                    start_page = i + 1
                    break

            # Find end page
            end_page = len(ocr_result.pages)
            for i, cum_count in enumerate(page_char_counts):
                if cum_count >= section_end:
                    end_page = i + 1
                    break

            section.page_start = start_page
            section.page_end = end_page
            section_start = section_end

    def process(
        self,
        ocr_result: OCRResult,
        original_path: str,
    ) -> DocumentPreprocessResult:
        """Process an OCR result into chunked sections and figures.

        Args:
            ocr_result: OCR result to process.
            original_path: Original document path.

        Returns:
            DocumentPreprocessResult with sections and figures.
        """
        import time

        start_time = time.time()

        # Get full text
        full_text = ocr_result.full_text

        # Chunk by headers
        sections = self.chunk_by_headers(full_text)

        # Assign page ranges
        self.assign_page_ranges(sections, ocr_result)

        # Extract figures
        figures = self.extract_figures(ocr_result, sections)

        processing_time = time.time() - start_time

        # Determine status
        status = ocr_result.status
        failed_pages = [fp["page"] for fp in ocr_result.failed_pages]

        return DocumentPreprocessResult(
            original_path=original_path,
            sections=sections,
            figures=figures,
            total_pages=ocr_result.total_pages,
            failed_pages=failed_pages,
            processing_time=ocr_result.elapsed_sec + processing_time,
            status=status,
            ocr_result=ocr_result,
        )


# Singleton instance
_chunker: DocumentChunker | None = None
_chunker_lock = threading.Lock()


def get_document_chunker() -> DocumentChunker:
    """Get the singleton document chunker instance (thread-safe)."""
    global _chunker
    if _chunker is None:
        with _chunker_lock:
            if _chunker is None:
                _chunker = DocumentChunker()
    return _chunker


def chunk_document(
    ocr_result: OCRResult,
    original_path: str,
    config: ChunkingConfig | None = None,
) -> DocumentPreprocessResult:
    """Convenience function to chunk a document.

    Args:
        ocr_result: OCR result to chunk.
        original_path: Original document path.
        config: Optional chunking configuration.

    Returns:
        DocumentPreprocessResult with sections and figures.
    """
    if config:
        chunker = DocumentChunker(config)
    else:
        chunker = get_document_chunker()

    return chunker.process(ocr_result, original_path)

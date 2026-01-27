"""Document preprocessing service for the orchestration pipeline.

This module provides the main entry point for document preprocessing,
combining OCR, chunking, and optional figure analysis into a single
service that can be called from the frontdoor or dispatcher.

The preprocessor:
1. Detects document inputs (PDF, images)
2. Runs OCR via LightOnOCR server
3. Chunks text by markdown headers
4. Extracts figure references
5. Optionally routes figures to VL model for description
6. Returns enriched context for the orchestrator
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.document import (
    DocumentPreprocessResult,
    DocumentProcessRequest,
    FigureRef,
    OCRResult,
    ProcessingStatus,
    Section,
)
from src.services.document_client import (
    DocumentFormalizerClient,
    OCRServerError,
    OCRServerUnavailable,
    get_document_client,
    process_document,
)
from src.services.document_chunker import (
    ChunkingConfig,
    DocumentChunker,
    chunk_document,
    get_document_chunker,
)
from src.services.figure_analyzer import (
    FigureAnalyzer,
    analyze_figures as analyze_figures_async,
    get_figure_analyzer,
)

logger = logging.getLogger(__name__)

# Document extensions that trigger OCR
DOCUMENT_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".docx"})

# Archive extensions that contain documents
ARCHIVE_EXTENSIONS = frozenset({".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".7z"})

# MIME types that trigger OCR
DOCUMENT_MIME_TYPES = frozenset({
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
})

# Phrases that indicate OCR intent
OCR_TRIGGER_PHRASES = frozenset({
    "ocr",
    "scan",
    "extract text",
    "read pdf",
    "transcribe document",
    "document text",
    "pdf text",
})

# Maximum characters for document summary context passed to VL model
MAX_SUMMARY_CONTEXT_CHARS = 8000

# Figure analysis prompt template with document context
FIGURE_PROMPT_WITH_CONTEXT = """You are analyzing a figure from a technical document. Below is a summary of the document for context.

=== DOCUMENT SUMMARY ===
{summary}
=== END SUMMARY ===

Now analyze this figure:
1. What type of visualization is this? (chart, diagram, graph, plot, table, etc.)
2. What parameters, data, or concepts are shown?
3. What is the main insight or finding illustrated?

Be concise but informative. Focus on the specific content visible in the figure."""


@dataclass
class PreprocessingConfig:
    """Configuration for document preprocessing."""

    # OCR settings
    output_format: str = "bbox"
    max_pages: int = 100
    dpi: int = 200

    # Chunking settings
    max_header_level: int = 3
    min_section_length: int = 10
    max_section_length: int = 10000

    # Figure settings
    extract_figures: bool = True
    describe_figures: bool = False  # Route to VL model
    min_figure_area: int = 5000

    # Processing mode
    async_threshold_pages: int = 6  # Pages above this use async mode


@dataclass
class PreprocessingResult:
    """Result of document preprocessing with status tracking."""

    success: bool
    document_result: DocumentPreprocessResult | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    job_id: str | None = None  # For async processing


class DocumentPreprocessor:
    """Main service for document preprocessing.

    Usage:
        preprocessor = DocumentPreprocessor()

        # Check if a TaskIR needs document preprocessing
        if preprocessor.needs_preprocessing(task_ir):
            result = await preprocessor.preprocess(task_ir)

            if result.success:
                # Use result.document_result for enriched context
                sections = result.document_result.sections
                figures = result.document_result.figures
    """

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        client: DocumentFormalizerClient | None = None,
        chunker: DocumentChunker | None = None,
        figure_analyzer: FigureAnalyzer | None = None,
    ):
        """Initialize the preprocessor.

        Args:
            config: Preprocessing configuration.
            client: OCR client. Uses singleton if None.
            chunker: Document chunker. Uses singleton if None.
            figure_analyzer: Optional FigureAnalyzer instance. Uses singleton if None.
        """
        self.config = config or PreprocessingConfig()
        self.client = client or get_document_client()
        self.chunker = chunker or get_document_chunker()
        self.figure_analyzer = figure_analyzer or get_figure_analyzer()

    def _extract_summary_context(
        self,
        document_result: DocumentPreprocessResult,
        max_chars: int = MAX_SUMMARY_CONTEXT_CHARS,
    ) -> str:
        """Extract a summary from document sections for figure analysis context.

        Prioritizes:
        1. Abstract/Summary/Introduction sections
        2. First few sections if no priority sections found
        3. Truncates to max_chars to keep VL model fast

        Args:
            document_result: The processed document with sections.
            max_chars: Maximum characters for the summary.

        Returns:
            Summary text for figure analysis context.
        """
        if not document_result.sections:
            return ""

        # Priority section titles (case-insensitive partial match)
        priority_keywords = [
            "abstract", "summary", "executive", "introduction",
            "overview", "background", "main thesis", "key"
        ]

        priority_sections: list[Section] = []
        other_sections: list[Section] = []

        for section in document_result.sections:
            title_lower = section.title.lower()
            if any(kw in title_lower for kw in priority_keywords):
                priority_sections.append(section)
            else:
                other_sections.append(section)

        # Build summary from priority sections first, then others
        summary_parts: list[str] = []
        total_chars = 0

        for section in priority_sections + other_sections:
            section_text = f"## {section.title}\n{section.content}\n"
            if total_chars + len(section_text) > max_chars:
                # Add truncated remainder
                remaining = max_chars - total_chars
                if remaining > 100:  # Only add if meaningful space left
                    summary_parts.append(section_text[:remaining] + "...")
                break
            summary_parts.append(section_text)
            total_chars += len(section_text)

        return "\n".join(summary_parts)

    def _build_figure_prompt(self, summary: str) -> str | None:
        """Build a figure analysis prompt with document context.

        Args:
            summary: Document summary text.

        Returns:
            Prompt string for the VL model, or None to use default.
        """
        if not summary:
            # Fall back to default prompt if no summary
            return None
        return FIGURE_PROMPT_WITH_CONTEXT.format(summary=summary)

    def needs_preprocessing(self, task_ir: dict[str, Any]) -> bool:
        """Check if a TaskIR needs document preprocessing.

        Args:
            task_ir: TaskIR dictionary.

        Returns:
            True if document preprocessing is needed.
        """
        # Check task_type
        task_type = task_ir.get("task_type", "")
        if task_type in ("doc", "ingest", "document"):
            return True

        # Check explicit preprocessing config
        preprocessing = task_ir.get("preprocessing", {})
        if preprocessing.get("ocr", {}).get("enabled"):
            return True

        # Check inputs for document files
        for inp in task_ir.get("inputs", []):
            inp_type = inp.get("type", "")
            value = inp.get("value", "")

            # Check file path
            if inp_type == "path":
                ext = Path(value).suffix.lower()
                if ext in DOCUMENT_EXTENSIONS:
                    return True
                # Check for archives (compound extensions)
                name = Path(value).name.lower()
                for arch_ext in ARCHIVE_EXTENSIONS:
                    if name.endswith(arch_ext):
                        return True

            # Check MIME type
            content_type = inp.get("content_type", "")
            if content_type in DOCUMENT_MIME_TYPES:
                return True

        # Check objective for OCR trigger phrases
        objective = task_ir.get("objective", "").lower()
        for phrase in OCR_TRIGGER_PHRASES:
            if phrase in objective:
                return True

        return False

    def detect_ocr_intent(self, prompt: str) -> bool:
        """Check if a prompt indicates OCR intent.

        Args:
            prompt: User prompt text.

        Returns:
            True if OCR intent is detected.
        """
        prompt_lower = prompt.lower()
        for phrase in OCR_TRIGGER_PHRASES:
            if phrase in prompt_lower:
                return True
        return False

    async def preprocess(
        self,
        task_ir: dict[str, Any],
    ) -> PreprocessingResult:
        """Preprocess documents in a TaskIR.

        Args:
            task_ir: TaskIR with document inputs.

        Returns:
            PreprocessingResult with document sections and figures.
        """
        warnings: list[str] = []

        # Find document inputs
        doc_paths = self._extract_document_paths(task_ir)

        if not doc_paths:
            return PreprocessingResult(
                success=True,
                document_result=None,
                warnings=["No document inputs found"],
            )

        # Process first document (TODO: support multiple documents)
        doc_path = doc_paths[0]
        if len(doc_paths) > 1:
            warnings.append(f"Multiple documents found, processing only: {doc_path}")

        # Check if it's an archive
        if self._is_archive(doc_path):
            result = await self.preprocess_archive(doc_path)
            if result.warnings:
                warnings.extend(result.warnings)
            return PreprocessingResult(
                success=result.success,
                document_result=result.document_result,
                error=result.error,
                warnings=warnings,
            )

        try:
            # Build processing request
            request = DocumentProcessRequest(
                file_path=str(doc_path),
                output_format=self.config.output_format,
                max_pages=self.config.max_pages,
                extract_figures=self.config.extract_figures,
                describe_figures=self.config.describe_figures,
                dpi=self.config.dpi,
            )

            # Run OCR
            ocr_result = await process_document(request)

            # Check for partial success
            if ocr_result.status == ProcessingStatus.PARTIAL:
                warnings.append(
                    f"Partial OCR success: {len(ocr_result.failed_pages)} pages failed"
                )
            elif ocr_result.status == ProcessingStatus.FAILED:
                return PreprocessingResult(
                    success=False,
                    error="OCR processing failed for all pages",
                    warnings=warnings,
                )

            # Chunk the document
            document_result = chunk_document(
                ocr_result,
                str(doc_path),
                ChunkingConfig(
                    max_header_level=self.config.max_header_level,
                    min_section_length=self.config.min_section_length,
                    max_section_length=self.config.max_section_length,
                    min_figure_area=self.config.min_figure_area,
                ),
            )

            # Optionally analyze figures with vision model
            if self.config.describe_figures and document_result.figures:
                try:
                    # Extract summary context for better figure understanding
                    summary = self._extract_summary_context(document_result)
                    vl_prompt = self._build_figure_prompt(summary)

                    if summary:
                        logger.info(
                            f"Using {len(summary)} char summary context for figure analysis"
                        )

                    document_result.figures = await analyze_figures_async(
                        pdf_path=str(doc_path),
                        figures=document_result.figures,
                        vl_prompt=vl_prompt,
                    )
                    logger.info(
                        f"Analyzed {len(document_result.figures)} figures with vision model"
                    )
                except Exception as e:
                    logger.warning(f"Figure analysis failed (continuing without): {e}")
                    warnings.append(f"Figure analysis failed: {e}")

            return PreprocessingResult(
                success=True,
                document_result=document_result,
                warnings=warnings,
            )

        except OCRServerUnavailable as e:
            return PreprocessingResult(
                success=False,
                error=f"OCR server unavailable: {e}",
                warnings=warnings,
            )
        except OCRServerError as e:
            return PreprocessingResult(
                success=False,
                error=f"OCR error: {e}",
                warnings=warnings,
            )
        except FileNotFoundError as e:
            return PreprocessingResult(
                success=False,
                error=str(e),
                warnings=warnings,
            )
        except Exception as e:
            logger.exception(f"Document preprocessing failed: {e}")
            return PreprocessingResult(
                success=False,
                error=f"Preprocessing failed: {e}",
                warnings=warnings,
            )

    def _extract_document_paths(self, task_ir: dict[str, Any]) -> list[Path]:
        """Extract document file paths from TaskIR inputs."""
        paths = []

        for inp in task_ir.get("inputs", []):
            inp_type = inp.get("type", "")
            value = inp.get("value", "")

            if inp_type == "path":
                path = Path(value)
                ext = path.suffix.lower()
                if ext in DOCUMENT_EXTENSIONS:
                    paths.append(path)
                # Also check for archives
                elif self._is_archive(path):
                    paths.append(path)

        return paths

    def _is_archive(self, path: Path) -> bool:
        """Check if a path is an archive file."""
        name = path.name.lower()
        for ext in ARCHIVE_EXTENSIONS:
            if name.endswith(ext):
                return True
        return False

    async def preprocess_archive(
        self,
        archive_path: Path,
    ) -> PreprocessingResult:
        """Preprocess documents from an archive file.

        Extracts the archive, processes all document files found,
        and returns a merged result.

        Args:
            archive_path: Path to the archive file.

        Returns:
            PreprocessingResult with merged document results.
        """
        from src.models.document import MultiDocumentResult
        from src.services.archive_extractor import ArchiveExtractor

        warnings: list[str] = []

        try:
            extractor = ArchiveExtractor()

            # Validate archive
            validation = extractor.validate(archive_path)
            if not validation.is_safe:
                return PreprocessingResult(
                    success=False,
                    error=f"Archive validation failed: {'; '.join(validation.issues)}",
                )

            # Get manifest
            manifest = extractor.list_contents(archive_path)

            # Find document files
            doc_files = [
                f.name for f in manifest.file_tree
                if not f.is_dir and f.extension in DOCUMENT_EXTENSIONS
            ]

            if not doc_files:
                return PreprocessingResult(
                    success=True,
                    document_result=None,
                    warnings=["No document files found in archive"],
                )

            # Extract document files
            result = extractor.extract_files(archive_path, doc_files)

            if not result.extracted_files:
                return PreprocessingResult(
                    success=False,
                    error="Failed to extract document files from archive",
                    warnings=result.errors,
                )

            # Process each extracted document
            documents: dict[str, DocumentPreprocessResult] = {}
            text_files: dict[str, str] = {}
            skipped: list[str] = []

            for filename, file_path in result.extracted_files.items():
                try:
                    # Process the document
                    doc_result = await self.preprocess_file(file_path)

                    if doc_result.success and doc_result.document_result:
                        documents[filename] = doc_result.document_result
                    else:
                        warnings.append(f"Failed to process {filename}: {doc_result.error}")
                        skipped.append(filename)

                except Exception as e:
                    warnings.append(f"Error processing {filename}: {e}")
                    skipped.append(filename)

            if not documents:
                return PreprocessingResult(
                    success=False,
                    error="No documents were successfully processed",
                    warnings=warnings,
                )

            # Create merged result
            multi_result = MultiDocumentResult(
                source_archive=str(archive_path),
                documents=documents,
                text_files=text_files,
                skipped_files=skipped,
                processing_time=result.extraction_time,
            )

            # Convert to DocumentPreprocessResult for compatibility
            # (merge all sections and figures)
            merged_sections: list[Section] = []
            merged_figures: list[FigureRef] = []
            total_pages = 0

            for filename, doc in documents.items():
                # Prefix section and figure IDs with filename for uniqueness
                for section in doc.sections:
                    new_section = Section(
                        id=f"{filename}:{section.id}",
                        title=f"[{filename}] {section.title}",
                        level=section.level,
                        content=section.content,
                        page_start=total_pages + section.page_start,
                        page_end=total_pages + section.page_end,
                        figure_ids=[f"{filename}:{fid}" for fid in section.figure_ids],
                    )
                    merged_sections.append(new_section)

                for figure in doc.figures:
                    new_figure = FigureRef(
                        id=f"{filename}:{figure.id}",
                        page=total_pages + figure.page,
                        bbox=figure.bbox,
                        description=figure.description,
                        image_path=figure.image_path,
                        section_id=f"{filename}:{figure.section_id}" if figure.section_id else None,
                    )
                    merged_figures.append(new_figure)

                total_pages += doc.total_pages

            merged_result = DocumentPreprocessResult(
                original_path=str(archive_path),
                sections=merged_sections,
                figures=merged_figures,
                total_pages=total_pages,
                processing_time=multi_result.processing_time,
                status=ProcessingStatus.COMPLETED,
            )

            return PreprocessingResult(
                success=True,
                document_result=merged_result,
                warnings=warnings,
            )

        except ImportError:
            return PreprocessingResult(
                success=False,
                error="Archive extractor not available",
            )
        except Exception as e:
            logger.exception(f"Archive preprocessing failed: {e}")
            return PreprocessingResult(
                success=False,
                error=f"Archive preprocessing failed: {e}",
                warnings=warnings,
            )

    async def preprocess_file(
        self,
        file_path: str | Path,
    ) -> PreprocessingResult:
        """Preprocess a single file directly.

        Args:
            file_path: Path to the document.

        Returns:
            PreprocessingResult with document sections and figures.
        """
        task_ir = {
            "inputs": [{"type": "path", "value": str(file_path)}],
        }
        return await self.preprocess(task_ir)

    def enrich_task_ir(
        self,
        task_ir: dict[str, Any],
        document_result: DocumentPreprocessResult,
    ) -> dict[str, Any]:
        """Enrich a TaskIR with document preprocessing results.

        This adds the OCR results and sections to the TaskIR while
        preserving the original inputs.

        Args:
            task_ir: Original TaskIR.
            document_result: Preprocessing result.

        Returns:
            Enriched TaskIR with ocr_result field.
        """
        enriched = task_ir.copy()

        # Add OCR result
        enriched["ocr_result"] = {
            "text": document_result.to_searchable_text(),
            "sections": [
                {
                    "id": s.id,
                    "title": s.title,
                    "level": s.level,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                    "figure_ids": s.figure_ids,
                }
                for s in document_result.sections
            ],
            "figures": [
                {
                    "id": f.id,
                    "page": f.page,
                    "description": f.description,
                    "section_id": f.section_id,
                }
                for f in document_result.figures
            ],
            "total_pages": document_result.total_pages,
            "failed_pages": document_result.failed_pages,
            "processing_time_sec": document_result.processing_time,
        }

        return enriched


# Singleton instance
_preprocessor: DocumentPreprocessor | None = None


def get_document_preprocessor() -> DocumentPreprocessor:
    """Get the singleton document preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = DocumentPreprocessor()
    return _preprocessor


async def preprocess_documents(task_ir: dict[str, Any]) -> PreprocessingResult:
    """Convenience function to preprocess documents in a TaskIR.

    Args:
        task_ir: TaskIR dictionary.

    Returns:
        PreprocessingResult with document sections and figures.
    """
    preprocessor = get_document_preprocessor()

    if not preprocessor.needs_preprocessing(task_ir):
        return PreprocessingResult(
            success=True,
            document_result=None,
            warnings=["No document preprocessing needed"],
        )

    return await preprocessor.preprocess(task_ir)

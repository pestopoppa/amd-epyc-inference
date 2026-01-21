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
from typing import Any, Callable

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

logger = logging.getLogger(__name__)

# Document extensions that trigger OCR
DOCUMENT_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".docx"})

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
        figure_analyzer: Callable[[list[FigureRef]], list[FigureRef]] | None = None,
    ):
        """Initialize the preprocessor.

        Args:
            config: Preprocessing configuration.
            client: OCR client. Uses singleton if None.
            chunker: Document chunker. Uses singleton if None.
            figure_analyzer: Optional callable for analyzing figures.
                            Takes list of FigureRef, returns list with descriptions.
        """
        self.config = config or PreprocessingConfig()
        self.client = client or get_document_client()
        self.chunker = chunker or get_document_chunker()
        self.figure_analyzer = figure_analyzer

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

            # Optionally analyze figures
            if self.config.describe_figures and self.figure_analyzer:
                document_result.figures = self.figure_analyzer(document_result.figures)

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

        return paths

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

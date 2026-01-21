"""Document processing API endpoints for LightOnOCR pipeline."""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from src.models.document import (
    DocumentPreprocessResult,
    ProcessingStatus,
    Section,
    FigureRef,
)
from src.services.document_preprocessor import (
    DocumentPreprocessor,
    PreprocessingConfig,
    PreprocessingResult,
    get_document_preprocessor,
    preprocess_documents,
)
from src.services.document_client import (
    DocumentFormalizerClient,
    OCRServerError,
    OCRServerUnavailable,
    get_document_client,
)
from src.services.document_chunker import get_document_chunker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


# Request/Response models
class DocumentProcessRequest(BaseModel):
    """Request to process a document."""

    file_path: str | None = Field(None, description="Path to local file")
    file_base64: str | None = Field(None, description="Base64-encoded document")
    output_format: str = Field("bbox", description="Output format: text, bbox, or json")
    max_pages: int = Field(100, ge=1, le=500, description="Maximum pages to process")
    extract_figures: bool = Field(True, description="Extract figure references")
    describe_figures: bool = Field(False, description="Route figures to VL model")


class SectionResponse(BaseModel):
    """Section information."""

    id: str
    title: str
    level: int
    page_start: int
    page_end: int
    content_preview: str
    figure_ids: list[str]


class FigureResponse(BaseModel):
    """Figure information."""

    id: str
    page: int
    section_id: str | None
    description: str
    has_image: bool


class DocumentProcessResponse(BaseModel):
    """Response from document processing."""

    success: bool
    original_path: str | None = None
    total_pages: int = 0
    sections: list[SectionResponse] = []
    figures: list[FigureResponse] = []
    failed_pages: list[int] = []
    processing_time_sec: float = 0.0
    full_text: str | None = None
    error: str | None = None
    warnings: list[str] = []


class OCRHealthResponse(BaseModel):
    """OCR server health status."""

    healthy: bool
    server_url: str
    error: str | None = None


class TaskIRPreprocessRequest(BaseModel):
    """Request to preprocess documents in a TaskIR."""

    task_ir: dict[str, Any]


class TaskIRPreprocessResponse(BaseModel):
    """Response from TaskIR preprocessing."""

    success: bool
    needs_preprocessing: bool
    enriched_task_ir: dict[str, Any] | None = None
    error: str | None = None
    warnings: list[str] = []


@router.get("/health", response_model=OCRHealthResponse)
async def check_ocr_health() -> OCRHealthResponse:
    """Check if the OCR server is healthy."""
    client = get_document_client()

    try:
        healthy = await client.health_check()
        return OCRHealthResponse(
            healthy=healthy,
            server_url=client.base_url,
        )
    except Exception as e:
        return OCRHealthResponse(
            healthy=False,
            server_url=client.base_url,
            error=str(e),
        )


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document_endpoint(request: DocumentProcessRequest) -> DocumentProcessResponse:
    """Process a document and return structured sections and figures.

    Provide ONE of:
    - file_path: Path to local file
    - file_base64: Base64-encoded document data
    """
    if not request.file_path and not request.file_base64:
        raise HTTPException(
            status_code=400,
            detail="Provide file_path or file_base64",
        )

    # Prepare file path
    temp_file = None
    file_path = None

    if request.file_path:
        file_path = Path(request.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    elif request.file_base64:
        # Decode base64 to temp file
        try:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            )
            doc_data = base64.b64decode(request.file_base64)
            temp_file.write(doc_data)
            temp_file.close()
            file_path = Path(temp_file.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    try:
        # Build TaskIR for preprocessing
        task_ir = {
            "inputs": [{"type": "path", "value": str(file_path)}],
        }

        # Configure preprocessor
        config = PreprocessingConfig(
            output_format=request.output_format,
            max_pages=request.max_pages,
            extract_figures=request.extract_figures,
            describe_figures=request.describe_figures,
        )
        preprocessor = DocumentPreprocessor(config=config)

        # Run preprocessing
        result = await preprocessor.preprocess(task_ir)

        if not result.success:
            return DocumentProcessResponse(
                success=False,
                error=result.error,
                warnings=result.warnings,
            )

        doc_result = result.document_result
        if doc_result is None:
            return DocumentProcessResponse(
                success=True,
                warnings=["No document content extracted"],
            )

        # Build response
        sections = [
            SectionResponse(
                id=s.id,
                title=s.title,
                level=s.level,
                page_start=s.page_start,
                page_end=s.page_end,
                content_preview=s.content[:200] + "..." if len(s.content) > 200 else s.content,
                figure_ids=s.figure_ids,
            )
            for s in doc_result.sections
        ]

        figures = [
            FigureResponse(
                id=f.id,
                page=f.page,
                section_id=f.section_id,
                description=f.description or "",
                has_image=bool(f.image_base64),
            )
            for f in doc_result.figures
        ]

        return DocumentProcessResponse(
            success=True,
            original_path=doc_result.original_path,
            total_pages=doc_result.total_pages,
            sections=sections,
            figures=figures,
            failed_pages=doc_result.failed_pages,
            processing_time_sec=doc_result.processing_time,
            full_text=doc_result.to_searchable_text(),
            warnings=result.warnings,
        )

    except OCRServerUnavailable as e:
        return DocumentProcessResponse(
            success=False,
            error=f"OCR server unavailable: {e}",
        )
    except OCRServerError as e:
        return DocumentProcessResponse(
            success=False,
            error=f"OCR error: {e}",
        )
    except Exception as e:
        logger.exception(f"Document processing failed: {e}")
        return DocumentProcessResponse(
            success=False,
            error=f"Processing failed: {e}",
        )
    finally:
        # Clean up temp file
        if temp_file:
            Path(temp_file.name).unlink(missing_ok=True)


@router.post("/process/upload", response_model=DocumentProcessResponse)
async def process_uploaded_document(
    file: UploadFile = File(...),
    output_format: str = Form("bbox"),
    max_pages: int = Form(100),
    extract_figures: bool = Form(True),
    describe_figures: bool = Form(False),
) -> DocumentProcessResponse:
    """Process an uploaded document file.

    Accepts multipart/form-data with:
    - file: The document file (PDF, PNG, JPG, etc.)
    - output_format: "text", "bbox", or "json"
    - max_pages: Maximum pages to process
    - extract_figures: Whether to extract figures
    - describe_figures: Whether to describe figures with VL model
    """
    # Save uploaded file to temp location
    suffix = Path(file.filename or "document").suffix or ".pdf"
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)

    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        # Process using the file path
        request = DocumentProcessRequest(
            file_path=temp_file.name,
            output_format=output_format,
            max_pages=max_pages,
            extract_figures=extract_figures,
            describe_figures=describe_figures,
        )

        return await process_document_endpoint(request)

    finally:
        # Clean up temp file
        Path(temp_file.name).unlink(missing_ok=True)


@router.post("/preprocess-taskir", response_model=TaskIRPreprocessResponse)
async def preprocess_task_ir(request: TaskIRPreprocessRequest) -> TaskIRPreprocessResponse:
    """Preprocess documents in a TaskIR and return enriched version.

    This endpoint:
    1. Checks if the TaskIR contains document inputs
    2. Runs OCR and chunking if needed
    3. Returns the TaskIR enriched with ocr_result field
    """
    preprocessor = get_document_preprocessor()

    # Check if preprocessing is needed
    if not preprocessor.needs_preprocessing(request.task_ir):
        return TaskIRPreprocessResponse(
            success=True,
            needs_preprocessing=False,
            enriched_task_ir=request.task_ir,
        )

    # Run preprocessing
    result = await preprocessor.preprocess(request.task_ir)

    if not result.success:
        return TaskIRPreprocessResponse(
            success=False,
            needs_preprocessing=True,
            error=result.error,
            warnings=result.warnings,
        )

    if result.document_result is None:
        return TaskIRPreprocessResponse(
            success=True,
            needs_preprocessing=True,
            enriched_task_ir=request.task_ir,
            warnings=result.warnings,
        )

    # Enrich the TaskIR
    enriched = preprocessor.enrich_task_ir(request.task_ir, result.document_result)

    return TaskIRPreprocessResponse(
        success=True,
        needs_preprocessing=True,
        enriched_task_ir=enriched,
        warnings=result.warnings,
    )


@router.get("/section/{section_id}")
async def get_section_content(section_id: str, task_id: str) -> dict[str, Any]:
    """Get full content of a specific section.

    Note: This requires the document to have been processed in a previous request.
    The task_id parameter refers to the processing task that created the sections.

    For stateless access, use the /process endpoint and include the section content
    in the initial response.
    """
    # This is a placeholder - in production, sections would be cached or stored
    raise HTTPException(
        status_code=501,
        detail="Section retrieval requires session state. Use /process endpoint instead.",
    )

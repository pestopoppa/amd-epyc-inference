"""Pipeline stage 6: Vision preprocessing.

Runs DocumentPreprocessor to extract text, sections, and figures from
images/documents. Stores results on routing.document_result so that
_execute_repl() can use DocumentREPLEnvironment.
"""

from __future__ import annotations

import logging

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat_utils import RoutingResult
from src.api.structured_logging import task_extra
from src.llm_primitives import LLMPrimitives

log = logging.getLogger(__name__)


async def _execute_vision(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
) -> ChatResponse | None:
    """Preprocess vision inputs through the document pipeline.

    Instead of answering directly, this stage runs DocumentPreprocessor to
    extract text, sections, and figures from the image/document. The result
    is stored on routing.document_result so that _execute_repl() can use
    DocumentREPLEnvironment with structured document access tools.

    Returns None to fall through to REPL mode (the normal path).
    Only returns ChatResponse on unrecoverable errors.
    """
    has_vision_input = request.image_path or request.image_base64 or request.files
    if not (request.real_mode and has_vision_input):
        return None

    from src.services.document_preprocessor import (
        DocumentPreprocessor,
        PreprocessingConfig,
    )

    config = PreprocessingConfig(
        extract_figures=True,
        describe_figures=True,
    )
    preprocessor = DocumentPreprocessor(config=config)

    try:
        if request.image_path:
            log.info(
                "Vision preprocessing file: %s",
                request.image_path,
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
            )
            result = await preprocessor.preprocess_file(request.image_path)
        elif request.image_base64:
            log.info(
                "Vision preprocessing base64 image",
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
            )
            task_ir = {"inputs": [{"type": "base64", "value": request.image_base64}]}
            result = await preprocessor.preprocess(task_ir)
        elif request.files:
            log.info(
                "Vision preprocessing %d files",
                len(request.files),
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
            )
            task_ir = {"inputs": [{"type": "path", "value": f} for f in request.files]}
            result = await preprocessor.preprocess(task_ir)
        else:
            result = None

        if result and result.success and result.document_result:
            routing.document_result = result
            log.info(
                "Document preprocessing succeeded: %d sections, %d figures",
                len(result.document_result.sections),
                len(result.document_result.figures),
                extra=task_extra(
                    task_id=routing.task_id, stage="execute", mode="vision_preprocess"
                ),
            )
            return None  # Fall through to REPL with document context

        # Preprocessing returned but without usable document result
        warn_msg = result.error if result else "unknown"
        log.warning(
            "Document preprocessing failed: %s",
            warn_msg,
            extra=task_extra(
                task_id=routing.task_id,
                stage="execute",
                mode="vision_preprocess",
                error_type="preprocess_failed",
            ),
        )

    except Exception as e:
        log.warning(
            "Vision preprocessing exception: %s: %s",
            type(e).__name__,
            e,
            extra=task_extra(
                task_id=routing.task_id,
                stage="execute",
                mode="vision_preprocess",
                error_type=type(e).__name__,
            ),
        )

    # Preprocessing failed — inject context note and fall through to text modes
    image_ref = request.image_path or "(base64 image)"
    request.context = (request.context or "") + (
        f"\n\n[IMAGE: {image_ref} — Document pipeline failed. Answering without OCR context.]"
    )
    return None  # Fall through to standard orchestration

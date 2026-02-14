"""Pipeline stage 6: Vision preprocessing + multimodal VL routing.

Runs DocumentPreprocessor to extract text, sections, and figures from
images/documents. Stores results on routing.document_result so that
_execute_repl() can use DocumentREPLEnvironment.

Stage 7.5 (_execute_vision_multimodal): Routes vision-role requests through
the actual multimodal VL handler instead of text-only paths. Without this,
_execute_direct/_execute_repl discard image data and VL models answer blind.
"""

from __future__ import annotations

import base64
import logging
import time

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat_utils import RoutingResult
from src.api.services.memrl import score_completed_task
from src.api.structured_logging import task_extra
from src.llm_primitives import LLMPrimitives

_VISION_ROLES = frozenset({"worker_vision", "vision_escalation"})
_VL_PORT_MAP = {"worker_vision": 8086, "vision_escalation": 8087}

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


async def _execute_vision_multimodal(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
    initial_role,
    execution_mode: str,
) -> ChatResponse | None:
    """Route vision-role requests through multimodal VL handlers.

    Text-only paths (_execute_direct, _execute_repl) discard image data.
    When a vision role has image data, this routes through:
    - _handle_vision_request (direct mode): OCR + multimodal VL completion
    - _vision_react_mode_answer (repl mode): multimodal ReAct tool loop

    Returns None if not a vision request or if the handler fails
    (caller falls through to text-only mode as last resort).
    """
    if str(initial_role) not in _VISION_ROLES:
        return None
    if not (request.image_path or request.image_base64):
        return None

    from src.api.routes.chat_vision import (
        _handle_vision_request,
        _vision_react_mode_answer,
    )

    task_id = routing.task_id
    tools_used = 0
    tools_called: list[str] = []

    try:
        if execution_mode == "repl":
            # Vision ReAct: multimodal tool loop with image
            image_b64 = request.image_base64
            if not image_b64 and request.image_path:
                from src.api.routes.path_validation import validate_api_path

                img_path = validate_api_path(request.image_path)
                if not img_path.exists():
                    log.warning("Vision image not found: %s", request.image_path)
                    return None
                image_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

            if not image_b64:
                return None

            # Detect MIME type from header bytes
            mime_type = "image/jpeg"
            try:
                raw = base64.b64decode(image_b64[:32])
                if raw[:4] == b"\x89PNG":
                    mime_type = "image/png"
                elif raw[:4] == b"RIFF":
                    mime_type = "image/webp"
            except Exception:
                pass

            vl_port = _VL_PORT_MAP.get(str(initial_role), 8086)
            answer, tools_used, tools_called = await _vision_react_mode_answer(
                prompt=request.prompt,
                image_b64=image_b64,
                mime_type=mime_type,
                context=request.context or "",
                vl_port=vl_port,
            )
        else:
            # Direct VL: multimodal completion (OCR + image + text → VL model)
            answer = await _handle_vision_request(
                request, primitives, state, task_id,
                force_server=str(initial_role),
            )

    except Exception as e:
        log.warning(
            "Vision multimodal failed: %s: %s — falling through to text mode",
            type(e).__name__,
            e,
            extra=task_extra(
                task_id=task_id,
                role=str(initial_role),
                stage="execute",
                mode="vision_multimodal",
                error_type=type(e).__name__,
            ),
        )
        return None  # Fall through to text-only mode

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=False, turns=1)

    if state.progress_logger:
        state.progress_logger.log_task_completed(
            task_id=task_id,
            success=True,
            details=f"Vision multimodal ({initial_role}:{execution_mode}), {elapsed:.3f}s",
        )
        score_completed_task(
            state,
            task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )

    # Estimate tokens from answer length — vision backends don't expose
    # completion token counts, but word count is a reasonable proxy.
    tokens_est = len(answer.split()) if answer else 0

    return ChatResponse(
        answer=answer,
        turns=1,
        tokens_used=tokens_est,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=request.real_mode,
        routed_to=str(initial_role),
        role_history=[str(initial_role)],
        routing_strategy=routing.routing_strategy,
        mode=execution_mode,
        tokens_generated=tokens_est,
        formalization_applied=routing.formalization_applied,
        tools_used=tools_used,
        tools_called=tools_called,
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
    )

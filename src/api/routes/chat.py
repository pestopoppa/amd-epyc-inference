"""Chat endpoints for the orchestrator API."""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.models import ChatRequest, ChatResponse
from src.api.state import get_state
from src.api.services.orchestrator import (
    extract_code_from_response,
    auto_wrap_final,
    classify_error,
    build_escalation_prompt,
)
from src.prompt_builders import (
    build_root_lm_prompt,
    build_stage2_review_prompt,
    build_long_context_exploration_prompt,
    build_routing_context,
    build_review_verdict_prompt,
    build_revision_prompt,
    build_formalizer_prompt,
    detect_format_constraints,
)
from src.services.draft_cache import get_draft_cache, CachedDraft
from src.services.prompt_compressor import PromptCompressor
from src.api.services.memrl import (
    ensure_memrl_initialized,
    score_completed_task,
)
from src.features import features
from src.llm_primitives import LLMPrimitives
from src.repl_environment import REPLEnvironment
from src.escalation import (
    EscalationPolicy,
    EscalationContext,
    EscalationAction,
    ErrorCategory as EscalationErrorCategory,
)
from src.generation_monitor import GenerationMonitor, MonitorConfig
from src.roles import Role
from src.sse_utils import (
    create_sse_response,
    token_event,
    thinking_event,
    turn_start_event,
    turn_end_event,
    error_event,
    final_event,
    done_event,
)

# Three-stage summarization configuration (Stage 0: compression, Stage 1: draft, Stage 2: review)
THREE_STAGE_CONFIG = {
    "enabled": True,
    "threshold_tokens": 5000,  # ~20K chars triggers Stage 1+2
    "multi_doc_discount": 0.7,  # Lower threshold for multiple documents
    "stage1_role": Role.FRONTDOOR,
    "stage2_role": Role.INGEST_LONG_CONTEXT,
    # Stage 0: Compression settings (LLMLingua-2)
    # DISABLED: Extractive compression causes quality regression (hallucinations, typos)
    # See handoffs/active/cmprsr_prompt_compression.md for details
    # Re-enable when Cmprsr (abstractive) weights become available
    "compression": {
        "enabled": False,  # Disabled due to quality issues with LLMLingua-2
        "min_chars": 30000,
        "target_ratio": 0.5,
        "stage1_context_limit": 20000,
    },
}

# Backwards compatibility alias
TWO_STAGE_CONFIG = THREE_STAGE_CONFIG

# Long context exploration configuration
# When context exceeds this threshold, use REPL-based chunked exploration
# instead of dumping the full context into a single model's window
LONG_CONTEXT_CONFIG = {
    "enabled": True,
    "threshold_chars": 20000,  # ~5K tokens triggers exploration mode
    "max_turns": 8,  # Allow more turns for multi-step exploration
}


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (rough: 4 chars per token)."""
    return len(text) // 4


def _is_summarization_task(prompt: str) -> bool:
    """Detect if the prompt is a summarization task.

    Args:
        prompt: The user's prompt.

    Returns:
        True if this looks like a summarization request.
    """
    summarization_keywords = [
        "summarize", "summary", "summarise", "summarisation",
        "executive summary", "overview", "key points",
        "main ideas", "tl;dr", "tldr", "synopsis",
    ]
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in summarization_keywords)


def _should_use_two_stage(
    prompt: str,
    context: str | None,
    doc_count: int = 1,
) -> bool:
    """Determine if two-stage context processing should be used.

    Triggers for ANY large context (not just summarization). The REPL
    exploration approach scored 0/9 on long context benchmarks because
    models generate standalone code instead of calling peek()/grep().
    Two-stage worker digest → frontdoor synthesis is more reliable.

    Args:
        prompt: The user's prompt.
        context: The context (document content).
        doc_count: Number of documents being processed.

    Returns:
        True if two-stage pipeline should be used.
    """
    if not TWO_STAGE_CONFIG["enabled"]:
        return False

    if not context:
        return False

    # Trigger for any context above threshold — not just summarization
    context_chars = len(context)
    threshold_chars = LONG_CONTEXT_CONFIG["threshold_chars"]  # 20K chars

    # Apply multi-doc discount
    if doc_count > 1:
        threshold_chars = int(threshold_chars * TWO_STAGE_CONFIG["multi_doc_discount"])

    return context_chars > threshold_chars


async def _run_two_stage_summarization(
    prompt: str,
    context: str,
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
) -> tuple[str, dict]:
    """Run two-stage context processing pipeline.

    Generalized for ALL large-context tasks (not just summarization):
    Stage 1: Workers digest chunks in parallel (44 t/s each)
    Stage 2: Frontdoor synthesizes answer from digests (18 t/s)

    For summarization tasks, falls through to the original
    Stage 1 (draft) + Stage 2 (large model review) pipeline.

    Args:
        prompt: The user's prompt.
        context: The full document context.
        primitives: LLMPrimitives instance for LLM calls.
        state: Application state.
        task_id: Task ID for logging.

    Returns:
        Tuple of (final_answer, stats_dict).
    """
    import time

    is_summarization = _is_summarization_task(prompt)

    stats = {
        "pipeline": "two_stage_context",
        "stage1_time_ms": 0,
        "stage2_time_ms": 0,
        "context_tokens": _estimate_tokens(context),
        "chunks": 0,
        "cache_hit": False,
    }

    # Determine chunking — sized for 1.5B fast workers (4K context window)
    # Use ~2K tokens per chunk (~8K chars) to leave room for prompt overhead
    n_chunks = max(2, min(8, len(context) // 8000))  # ~2K tokens per chunk
    stats["chunks"] = n_chunks

    # Stage 1: Worker parallel digest
    stage1_start = time.perf_counter()

    chunk_size = len(context) // n_chunks
    overlap = 200
    chunks = []
    for i in range(n_chunks):
        start_idx = max(0, i * chunk_size - (overlap if i > 0 else 0))
        end_idx = min(len(context), (i + 1) * chunk_size + (overlap if i < n_chunks - 1 else 0))
        chunks.append({"index": i, "text": context[start_idx:end_idx]})

    # Build worker prompts — task-specific instructions
    worker_prompts = []
    for chunk in chunks:
        worker_prompt = (
            f"Analyze this section ({chunk['index']+1}/{n_chunks}) of a larger document.\n"
            f"Task context: {prompt[:200]}\n\n"
            f"## Section Content\n{chunk['text'][:6000]}\n\n"
            f"## Instructions\n"
            f"Extract: key facts, relevant quotes, any findings related to the task.\n"
            f"If the task asks to FIND something specific, look for it and report exact matches.\n"
            f"Be concise. Output structured findings only."
        )
        worker_prompts.append(worker_prompt)

    # Dispatch to workers in parallel via llm_batch
    # Prefer worker_fast (1.5B, ports 8102/8112) for speed, but these are
    # WARM tier and may not be running. Fall back to worker_explore (7B, 8082).
    import httpx

    worker_role = "worker_fast"
    try:
        resp = httpx.get("http://localhost:8102/health", timeout=2)
        if resp.status_code != 200:
            worker_role = "worker_explore"
    except Exception:
        worker_role = "worker_explore"

    try:
        digests = primitives.llm_batch(worker_prompts, role=worker_role, n_tokens=500)
    except Exception:
        # Fallback: sequential calls with worker_explore (always HOT)
        digests = []
        for wp in worker_prompts:
            try:
                d = primitives.llm_call(wp, role="worker_explore", n_tokens=500)
                digests.append(d)
            except Exception:
                digests.append("[Worker failed to process this section]")

    stage1_time = time.perf_counter() - stage1_start
    stats["stage1_time_ms"] = int(stage1_time * 1000)

    # Stage 2: Frontdoor synthesis from digests
    stage2_start = time.perf_counter()

    digest_text = "\n\n".join(
        f"[Section {i+1}/{len(digests)}]\n{d}"
        for i, d in enumerate(digests)
    )

    if is_summarization:
        synthesis_instruction = (
            "Synthesize a comprehensive summary from the section findings above.\n"
            "Cover: main thesis, key innovations, how it works, benefits and audience.\n"
            "Be thorough and well-structured."
        )
    else:
        synthesis_instruction = (
            "Synthesize a complete answer from the section findings above.\n"
            "If searching for specific items, report exact values found.\n"
            "If analyzing the document, provide a thorough answer.\n"
            "Be precise and include specific details from the findings."
        )

    synthesis_prompt = (
        f"You analyzed a large document in {n_chunks} sections. Here are the worker findings:\n\n"
        f"{digest_text}\n\n"
        f"## Original Question\n{prompt}\n\n"
        f"## Instructions\n{synthesis_instruction}"
    )

    # Use frontdoor for synthesis (18 t/s) — much faster than architect
    try:
        answer = primitives.llm_call(
            synthesis_prompt,
            role=TWO_STAGE_CONFIG["stage1_role"],  # frontdoor
            n_tokens=4096,
        )
    except Exception as e:
        # Use digest text directly as fallback
        answer = f"Worker findings:\n{digest_text}"

    stage2_time = time.perf_counter() - stage2_start
    stats["stage2_time_ms"] = int(stage2_time * 1000)

    # Log to progress logger if available
    if state.progress_logger:
        state.progress_logger.log_exploration(
            task_id=task_id,
            query=prompt[:100],
            strategy_used="two_stage_context",
            tokens_spent=_estimate_tokens(synthesis_prompt),
            success=True,
        )

    # Store digests for potential review gate use (Step 6)
    stats["worker_digests"] = [
        {"section": i + 1, "summary": d[:500]}
        for i, d in enumerate(digests)
    ]

    return answer.strip(), stats


def _is_ocr_heavy_prompt(prompt: str) -> bool:
    """Detect if a vision prompt would benefit from deterministic OCR pre-processing.

    ALWAYS returns True — OCR is cheap (~1s via LightOnOCR-2-1B on port 9001)
    and provides text context that helps the VL model for ANY task:
    - Text extraction: OCR is primary answer source
    - Chart analysis: OCR extracts axis labels, values, legends
    - Document analysis: OCR extracts full text for Twyne/whitepaper/protocol analysis
    - Scene understanding: OCR captures signage, labels, text overlays

    Args:
        prompt: The user's vision prompt.

    Returns:
        Always True. OCR context is universally beneficial.
    """
    return True


def _needs_structured_analysis(prompt: str) -> bool:
    """Detect if a vision prompt needs full structured analysis beyond OCR.

    Structured analysis triggers VisionPipeline with VL_STRUCTURED analyzer
    for document forensics, diagram parsing, protocol architecture, etc.

    Args:
        prompt: The user's vision prompt.

    Returns:
        True if structured analysis should complement OCR.
    """
    prompt_lower = prompt.lower()
    structured_keywords = [
        "analyze", "architecture", "diagram", "protocol",
        "economic model", "security audit", "security analysis",
        "whitepaper", "smart contract", "incentive",
        "trust assumption", "attack vector", "forensic",
        "entity extraction", "business relationship",
        "flow chart", "flowchart", "sequence diagram",
        "system design", "data flow", "state machine",
    ]
    return any(kw in prompt_lower for kw in structured_keywords)


async def _handle_vision_request(
    request: "ChatRequest",
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
) -> str:
    """Route a vision request through OCR + VL pipeline.

    For OCR-heavy prompts (text extraction, sign reading):
    1. LightOnOCR-2-1B (port 9001) extracts text deterministically
    2. Extracted text + image + prompt sent to VL model for understanding

    For non-OCR prompts (chart analysis, scene understanding):
    1. Image + prompt sent directly to VL model

    Fallback chain for VL:
    1. VL worker (port 8086, Qwen2.5-VL-7B, ~15 t/s)
    2. VL escalation (port 8087, Qwen3-VL-30B-A3B, ~10 t/s)
    3. Vision pipeline endpoint (legacy subprocess path)

    Args:
        request: Chat request with image_path or image_base64.
        primitives: LLMPrimitives (unused, for interface compat).
        state: Application state.
        task_id: Task ID for logging.

    Returns:
        Answer string from the vision model.
    """
    import httpx
    import base64
    import logging

    logger = logging.getLogger(__name__)

    # Get image as base64
    image_b64 = request.image_base64
    if not image_b64 and request.image_path:
        from pathlib import Path
        img_path = Path(request.image_path)
        if not img_path.exists():
            raise RuntimeError(f"Image not found: {request.image_path}")
        image_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

    if not image_b64:
        raise RuntimeError("No image data provided (need image_path or image_base64)")

    # ── Always-on OCR pre-processing (~1s, deterministic, port 9001) ─────
    # OCR is cheap and universally beneficial: extracts text for docs,
    # axis labels for charts, signage for scenes, protocol details for Twyne.
    ocr_text = ""
    try:
        from src.services.document_client import get_document_client
        client = get_document_client()
        ocr_result = await client.ocr_image(
            image=image_b64,
            output_format="text",
        )
        ocr_text = ocr_result.text.strip() if ocr_result.text else ""
        if ocr_text:
            logger.info(f"OCR pre-processing extracted {len(ocr_text)} chars for VL")
    except Exception as e:
        logger.warning(f"OCR pre-processing failed (continuing without): {e}")
        ocr_text = ""

    # ── Structured analysis for document/diagram prompts ───────────────
    structured_context = ""
    if _needs_structured_analysis(request.prompt):
        try:
            from src.vision.pipeline import VisionPipeline
            import io
            from PIL import Image

            # Decode base64 to PIL Image for VisionPipeline
            img_bytes = base64.b64decode(image_b64)
            pil_image = Image.open(io.BytesIO(img_bytes))

            pipeline = VisionPipeline()
            result = pipeline.analyze(pil_image, analyzers=["vl_structured", "vl_ocr"])

            # Format structured findings
            parts = []
            if hasattr(result, "structured") and result.structured:
                parts.append(f"Structured elements:\n{result.structured}")
            if hasattr(result, "description") and result.description:
                parts.append(f"Visual description:\n{result.description}")
            structured_context = "\n\n".join(parts)
            if structured_context:
                logger.info(f"Structured analysis added {len(structured_context)} chars context")
        except ImportError:
            logger.debug("VisionPipeline not available, skipping structured analysis")
        except Exception as e:
            logger.warning(f"Structured analysis failed (continuing without): {e}")

    # Build VL prompt — inject OCR text and structured context
    vl_prompt = request.prompt
    context_parts = []
    if ocr_text:
        context_parts.append(f"OCR-extracted text from image:\n---\n{ocr_text}\n---")
    if structured_context:
        context_parts.append(f"Structured analysis:\n---\n{structured_context}\n---")
    if context_parts:
        context_block = "\n\n".join(context_parts)
        vl_prompt = (
            f"{context_block}\n\n"
            f"Using the above extracted information and the image, answer:\n{request.prompt}"
        )

    # Detect image MIME type from header bytes
    mime_type = "image/jpeg"  # default
    try:
        raw = base64.b64decode(image_b64[:32])
        if raw[:4] == b'\x89PNG':
            mime_type = "image/png"
        elif raw[:4] == b'RIFF':
            mime_type = "image/webp"
    except Exception:
        pass

    # Build multimodal chat completion payload for llama-server
    vl_payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": vl_prompt,
                    },
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    # Try VL servers in order: worker (8086) → escalation (8087)
    vl_servers = [
        ("worker_vision", "http://localhost:8086"),
        ("vision_escalation", "http://localhost:8087"),
    ]

    last_error = None
    for server_name, server_url in vl_servers:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{server_url}/v1/chat/completions",
                    json=vl_payload,
                )

            if resp.status_code == 200:
                data = resp.json()
                # Extract answer from chat completions format
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    if content:
                        logger.info(f"VL answer from {server_name} ({len(content)} chars)")
                        return content.strip()

                # Try legacy completion format
                content = data.get("content", "")
                if content:
                    return content.strip()

                last_error = f"{server_name}: empty response"
                logger.warning(f"VL {server_name}: got 200 but empty content")
            else:
                last_error = f"{server_name}: HTTP {resp.status_code}"
                logger.warning(f"VL {server_name}: HTTP {resp.status_code}: {resp.text[:200]}")

        except httpx.ConnectError:
            last_error = f"{server_name}: connection refused (server not running?)"
            logger.warning(f"VL {server_name}: connection refused at {server_url}")
        except Exception as e:
            last_error = f"{server_name}: {type(e).__name__}: {e}"
            logger.warning(f"VL {server_name} failed: {type(e).__name__}: {e}")

    # All VL servers failed — try legacy vision pipeline as last resort
    logger.warning(f"All VL servers failed ({last_error}), trying vision pipeline")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            legacy_payload = {
                "image_path": request.image_path,
                "image_base64": request.image_base64,
                "analyzers": ["vl_describe"],
                "vl_prompt": request.prompt,
                "store_results": False,
            }
            resp = await client.post(
                "http://localhost:8000/vision/analyze",
                json=legacy_payload,
            )
        if resp.status_code == 200:
            data = resp.json()
            # AnalyzeResult has .description at top level
            description = data.get("description", "")
            if description:
                return description
    except Exception as e:
        logger.warning(f"Legacy vision pipeline also failed: {e}")

    raise RuntimeError(f"All vision paths failed. Last error: {last_error}")


async def _handle_multi_file_vision(
    request: "ChatRequest",
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
) -> str:
    """Handle vision requests with multiple files or archives.

    For archives (zip, tar, etc.): Extract → process each file → aggregate.
    For multiple files: Process each → aggregate via MultiDocumentResult.

    Uses:
    - ArchiveExtractor for ZIP/TAR/7Z extraction with security constraints
    - DocumentFormalizerClient for multi-page PDF OCR
    - BatchProcessor for parallel image processing
    - Two-stage summarization for synthesis

    Args:
        request: Chat request with files field.
        primitives: LLMPrimitives for synthesis.
        state: Application state.
        task_id: Task ID for logging.

    Returns:
        Synthesized answer from all files.
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)
    files = request.files or []

    if not files:
        raise RuntimeError("No files provided")

    # Check for archives
    archive_extensions = {".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".7z"}
    extracted_files = []

    for fpath in files:
        p = Path(fpath)
        suffix = "".join(p.suffixes).lower()
        if suffix in archive_extensions or p.suffix.lower() in archive_extensions:
            try:
                from src.services.archive_extractor import ArchiveExtractor
                extractor = ArchiveExtractor()
                manifest = extractor.list_contents(fpath)
                logger.info(
                    f"Archive {p.name}: {manifest.total_files} files, "
                    f"{manifest.total_size_bytes / 1024:.1f}KB"
                )
                # Extract all (respects security limits: 500MB max, 1000 files max)
                result = extractor.extract_all(fpath)
                extracted_files.extend(str(ef) for ef in result.extracted_paths)
            except Exception as e:
                logger.warning(f"Archive extraction failed for {fpath}: {e}")
                extracted_files.append(fpath)  # Try to process as-is
        else:
            extracted_files.append(fpath)

    if not extracted_files:
        raise RuntimeError("No files to process after extraction")

    # Process each file — OCR for documents, VisionPipeline for images
    file_results = []
    for fpath in extracted_files:
        p = Path(fpath)
        try:
            if p.suffix.lower() == ".pdf":
                # Multi-page PDF via DocumentFormalizerClient
                from src.services.document_client import get_document_client
                client = get_document_client()
                ocr_result = await client.ocr_pdf_with_partial_success(fpath)
                file_results.append(
                    f"[File: {p.name}]\n{ocr_result.text or '[OCR failed]'}"
                )
            elif p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}:
                # Single image — OCR + VL
                import base64
                img_b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                from src.services.document_client import get_document_client
                client = get_document_client()
                ocr_result = await client.ocr_image(image=img_b64, output_format="text")
                file_results.append(
                    f"[File: {p.name}]\n{ocr_result.text or '[No text extracted]'}"
                )
            else:
                # Text files
                content = p.read_text(errors="replace")[:10000]
                file_results.append(f"[File: {p.name}]\n{content}")
        except Exception as e:
            file_results.append(f"[File: {p.name}] Error: {e}")

    # Synthesize: two-stage if many files, direct if few
    all_content = "\n\n".join(file_results)
    if len(all_content) > 20000:
        answer, _stats = await _run_two_stage_summarization(
            request.prompt, all_content, primitives, state, task_id,
        )
    else:
        synthesis_prompt = (
            f"{all_content}\n\n"
            f"Based on the files above, answer:\n{request.prompt}"
        )
        answer = primitives.llm_call(
            synthesis_prompt,
            role="frontdoor",
            n_tokens=2048,
            skip_suffix=True,
        )

    return answer.strip()


_STUB_PATTERNS = {
    "complete", "see above", "analysis complete", "estimation complete",
    "done", "finished", "see results above", "see output above",
    "see structured output above", "see integrated results above",
    "see the structured output above",
}


def _is_stub_final(text: str) -> bool:
    """Detect when FINAL() arg is a stub pointing to printed output.

    Models often print their analysis via print(), then call
    FINAL("Analysis complete. See above.") — the real content
    is in result.output, not result.final_answer.
    """
    normalized = text.strip().rstrip(".").lower()
    return any(p in normalized for p in _STUB_PATTERNS)


def _strip_tool_outputs(text: str, tool_outputs: list[str]) -> str:
    """Strip known tool outputs from captured REPL output.

    Routing tools (my_role, route_advice, list_dir, recall) return JSON/TOON
    strings that get captured in stdout. When the model prints these, they
    contaminate the final answer. Strip them.

    Uses structured delimiters (<<<TOOL_OUTPUT>>>...<<<END_TOOL_OUTPUT>>>)
    for reliable regex-based stripping, with legacy exact-string matching
    as fallback for outputs without delimiters.

    Args:
        text: The captured stdout text.
        tool_outputs: List of exact tool output strings to strip.

    Returns:
        Text with tool outputs removed, cleaned up.
    """
    import re

    if not text:
        return text

    result = text

    # Primary: strip structured delimiters (reliable, regex-based)
    result = re.sub(
        r"<<<TOOL_OUTPUT>>>.*?<<<END_TOOL_OUTPUT>>>",
        "",
        result,
        flags=re.DOTALL,
    )

    # Fallback: legacy exact-string matching for pre-delimiter tool outputs
    if tool_outputs:
        for output in tool_outputs:
            if output in result:
                result = result.replace(output, "")

    # Also strip common prefixes the model adds around tool outputs
    for prefix in [
        r"Current [Rr]ole:\s*",
        r"Available files:\s*",
        r"Routing advice:\s*",
        r"Could not get routing advice:\s*",
        r"Creating a ticket for further investigation\s*",
    ]:
        result = re.sub(prefix, "", result)

    # Clean up: collapse multiple blank lines, strip
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


def _resolve_answer(result: "ExecutionResult", tool_outputs: list[str] | None = None) -> str:
    """Extract the best answer from an ExecutionResult.

    Handles cases where the model prints content then uses a stub FINAL().
    Strips tool outputs (my_role, route_advice, list_dir) from captured output.
    """
    captured = result.output.strip() if result.output else ""
    final = result.final_answer or ""

    # Strip tool outputs from captured stdout
    if tool_outputs:
        captured = _strip_tool_outputs(captured, tool_outputs)

    if captured and _is_stub_final(final):
        return captured
    elif captured and final and captured != final:
        # Prepend captured output if FINAL() doesn't already contain it
        if final not in captured:
            return f"{captured}\n\n{final}"
        return final
    else:
        return final


def _detect_output_quality_issue(answer: str) -> str | None:
    """Detect quality issues in model output using text-based heuristics.

    Inspired by GenerationMonitor's entropy/repetition signals but operating
    on complete output text (no streaming logits needed). Returns a description
    of the issue if detected, None if output looks fine.

    This is SAFE routing: only triggers on detected failure patterns,
    never on input keywords (which caused Wave 2 regressions).

    Args:
        answer: The model's complete output.

    Returns:
        Issue description if quality problem detected, None otherwise.
    """
    if not answer or len(answer) < 20:
        return None  # Too short to analyze

    words = answer.split()
    n_words = len(words)

    # 1. High n-gram repetition (degeneration loops)
    if n_words >= 20:
        trigrams = [" ".join(words[i:i+3]) for i in range(n_words - 2)]
        if trigrams:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            if unique_ratio < 0.5:  # More than 50% repeated trigrams
                return f"high_repetition (unique_ratio={unique_ratio:.2f})"

    # 2. Self-contradictory trace (model says X then says not-X)
    # Common in confused binary search analysis: "arr[mid] < target" when arr[mid] > target
    lines = answer.strip().split("\n")
    if n_words >= 50:
        # Check for confused analysis: very short non-empty lines mixed with long ones
        # indicates garbled/confused trace
        short_lines = sum(1 for l in lines if 0 < len(l.strip()) < 10)
        total_lines = sum(1 for l in lines if l.strip())
        if total_lines > 5 and short_lines / total_lines > 0.6:
            return "garbled_output (mostly very short lines)"

    # 3. Empty or near-empty after stripping common prefixes
    stripped = answer.strip()
    for prefix in ["```", "```python", "```json", "Here is", "The answer is"]:
        stripped = stripped.removeprefix(prefix).strip()
    if len(stripped) < 10:
        return "near_empty_output"

    return None


def _truncate_looped_answer(answer: str, prompt: str) -> str:
    """Defense-in-depth: truncate answer if prompt text reappears in it.

    Some models loop back to echoing the prompt after completing their answer.
    Detect this and truncate before the repeated prompt content.

    Args:
        answer: The model's raw output.
        prompt: The original prompt sent to the model.

    Returns:
        Truncated answer if loop detected, original answer otherwise.
    """
    if not answer or not prompt or len(prompt) < 40:
        return answer

    # Use a suffix of the prompt (last 80 chars) as the probe — if this
    # appears in the answer, the model is looping back to the prompt.
    probe = prompt[-80:].strip()
    if not probe:
        return answer

    idx = answer.find(probe)
    if idx > 0:
        # Truncate everything from the repeated prompt onwards
        truncated = answer[:idx].rstrip()
        if len(truncated) > 20:  # Only truncate if we keep a meaningful answer
            return truncated

    return answer


def _should_review(state: "AppState", task_id: str, role: str, answer: str) -> bool:
    """MemRL-conditional: review only when confidence < threshold.

    Checks Q-values for the current role+task combination. If average
    Q-value is below 0.6, the role historically struggles with this
    task type and a brief architect review is warranted.

    Args:
        state: Application state with hybrid_router.
        task_id: Current task ID.
        role: The role that generated the answer.
        answer: The answer to potentially review.

    Returns:
        True if architect review should be triggered.
    """
    if not state.hybrid_router:
        return False
    if "architect" in str(role):
        return False  # Architects ARE the reviewer — don't self-review
    if len(answer) < 50:
        return False  # Trivial answers don't need review
    try:
        # Get Q-values for this role from MemRL
        retriever = state.hybrid_router.retriever
        task_ir = {"task_type": "chat", "objective": answer[:100]}
        results = retriever.retrieve_for_routing(task_ir)
        if not results:
            return False
        # Filter for current role
        role_results = [r for r in results if r.memory.action == str(role)]
        if not role_results:
            return False
        avg_q = sum(r.q_value for r in role_results) / len(role_results)
        return avg_q < 0.6
    except Exception:
        return False


def _architect_verdict(
    question: str,
    answer: str,
    primitives: "LLMPrimitives",
    worker_digests: list[dict] | None = None,
    context_digest: str = "",
) -> str | None:
    """Get architect's hyper-concise verdict on an answer.

    The architect emits ONLY a short verdict (~20-50 tokens at 6.75 t/s → ~6s).
    Returns None if OK, or "WRONG: <corrections>" if incorrect.

    Args:
        question: Original user question.
        answer: The answer to review.
        primitives: LLM primitives for inference.
        worker_digests: Optional TOON-encodable worker digests.
        context_digest: Optional compact context summary.

    Returns:
        None if answer is OK, or "WRONG: ..." string if corrections needed.
    """
    prompt = build_review_verdict_prompt(
        question, answer,
        context_digest=context_digest,
        worker_digests=worker_digests,
    )
    try:
        result = primitives.llm_call(
            prompt,
            role="architect_general",
            n_tokens=80,  # Hard cap — verdict only
        )
        text = result.strip()
        if text.upper().startswith("OK"):
            return None
        return text  # "WRONG: <corrections>"
    except Exception:
        return None  # On error, don't block — return original answer


def _fast_revise(
    question: str,
    original_answer: str,
    corrections: str,
    primitives: "LLMPrimitives",
) -> str:
    """Fast worker expands architect's corrections into full answer.

    Uses worker_explore (port 8082, 44 t/s) — the fastest model in the stack.
    7B is sufficient since the architect already specified exactly what to fix.

    Args:
        question: Original user question.
        original_answer: The answer to revise.
        corrections: Architect's correction notes.
        primitives: LLM primitives for inference.

    Returns:
        Revised answer, or original if revision fails.
    """
    prompt = build_revision_prompt(question, original_answer, corrections)
    try:
        result = primitives.llm_call(
            prompt,
            role="worker_explore",
            n_tokens=2000,
        )
        return result.strip() or original_answer
    except Exception:
        return original_answer  # Fallback to original on error


# ── Architect Plan Review Gate ─────────────────────────────────────────────


def _needs_plan_review(
    task_ir: dict,
    routing_decision: list,
    state: "AppState",
) -> bool:
    """Determine whether the plan needs architect review before execution.

    Bypass conditions (skip review when any is true):
    1. TaskComplexity is TRIVIAL or SIMPLE
    2. TaskComplexity is COMPLEX (architect already owns plan)
    3. Single-step plan (no multi-step coordination to review)
    4. Architect is already the actor (no self-review)
    5. Phase B: Q-value >= 0.6 for task class
    6. Phase C: 90% skip (stochastic)
    7. Feature flag disabled (checked by caller)

    Args:
        task_ir: TaskIR dict with objective, task_type.
        routing_decision: List of roles selected for routing.
        state: Application state.

    Returns:
        True if architect plan review should run.
    """
    import random
    from src.proactive_delegation import classify_task_complexity, TaskComplexity

    objective = task_ir.get("objective", "")
    complexity, _signals = classify_task_complexity(objective)

    # Bypass 1+2: Only review MODERATE complexity
    if complexity != TaskComplexity.MODERATE:
        return False

    # Bypass 3: Single-step plans don't need coordination review
    plan = task_ir.get("plan", {})
    steps = plan.get("steps", [])
    if len(steps) <= 1:
        # No explicit plan steps yet — check routing for multi-role indication
        # For chat requests, routing_decision is typically 1 role, but plan
        # review is still useful if complexity is MODERATE
        pass  # Allow review for MODERATE tasks even without explicit steps

    # Bypass 4: Don't self-review architect
    if routing_decision and "architect" in str(routing_decision[0]):
        return False

    # Phase-dependent gating
    phase = state.plan_review_phase

    # Bypass 6: Phase C — 90% stochastic skip
    if phase == "C":
        if random.random() > 0.10:
            return False

    # Bypass 5: Phase B — Q-value gating
    if phase == "B" and state.hybrid_router:
        try:
            retriever = state.hybrid_router.retriever
            results = retriever.retrieve_for_routing(task_ir)
            if results:
                avg_q = sum(r.q_value for r in results) / len(results)
                if avg_q >= 0.6:
                    return False
        except Exception:
            pass  # On error, allow review

    return True


def _architect_plan_review(
    task_ir: dict,
    routing_decision: list,
    primitives: "LLMPrimitives",
    state: "AppState",
    task_id: str,
) -> "PlanReviewResult | None":
    """Execute architect plan review and return result.

    Non-blocking: returns None on timeout or error.

    Args:
        task_ir: TaskIR dict.
        routing_decision: Current routing decision.
        primitives: LLM primitives for calling architect.
        state: Application state.
        task_id: Current task ID.

    Returns:
        PlanReviewResult or None on failure.
    """
    import logging
    from src.proactive_delegation import ArchitectReviewService

    log = logging.getLogger(__name__)

    objective = task_ir.get("objective", "")
    task_type = task_ir.get("task_type", "chat")

    # Construct plan steps from routing decision (minimal for chat requests)
    plan = task_ir.get("plan", {})
    plan_steps = plan.get("steps", [])

    # If no explicit plan, synthesize from routing_decision
    if not plan_steps and routing_decision:
        plan_steps = [
            {
                "id": f"S{i+1}",
                "actor": str(role),
                "action": objective[:50],
                "outputs": [],
            }
            for i, role in enumerate(routing_decision)
        ]

    if not plan_steps:
        return None

    review_service = ArchitectReviewService(primitives)
    result = review_service.review_plan(
        objective=objective,
        task_type=task_type,
        plan_steps=plan_steps,
    )

    if result:
        log.info(
            "Plan review: decision=%s score=%.2f feedback=%s",
            result.decision,
            result.score,
            result.feedback[:60],
        )

    return result


def _apply_plan_review(
    routing_decision: list,
    review: "PlanReviewResult",
) -> list:
    """Apply architect's plan review corrections to routing decision.

    Handles 'reroute' patches that change which specialist handles a step.

    Args:
        routing_decision: Current routing decision list.
        review: Architect's plan review result.

    Returns:
        Updated routing decision (may be unchanged if no reroute patches).
    """
    if not review.patches:
        return routing_decision

    updated = list(routing_decision)

    for patch in review.patches:
        op = patch.get("op", "")
        if op == "reroute":
            new_role = patch.get("v", "")
            step_id = patch.get("step", "")
            if new_role:
                # Map step index to routing decision index
                # S1 → index 0, S2 → index 1, etc.
                try:
                    idx = int(step_id.replace("S", "")) - 1
                    if 0 <= idx < len(updated):
                        updated[idx] = new_role
                    elif idx == 0 and len(updated) >= 1:
                        updated[0] = new_role
                except (ValueError, IndexError):
                    # If step_id isn't parseable, reroute the first step
                    if updated:
                        updated[0] = new_role

    return updated


def _store_plan_review_episode(
    state: "AppState",
    task_id: str,
    task_ir: dict,
    review: "PlanReviewResult",
) -> None:
    """Store plan review result as MemRL episode and progress log entry.

    Architect corrections become high-quality training signals:
    - review.score mapped to reward: score * 2 - 1 (0-1 → -1..+1)
    - Action: plan:{role1},{role2} (the routing decision)

    Args:
        state: Application state.
        task_id: Current task ID.
        task_ir: TaskIR dict.
        review: Architect's plan review result.
    """
    # Log to progress JSONL
    if state.progress_logger:
        from orchestration.repl_memory.progress_logger import ProgressEntry, EventType
        state.progress_logger.log(
            ProgressEntry(
                event_type=EventType.PLAN_REVIEWED,
                task_id=task_id,
                agent_role="architect_general",
                data={
                    "decision": review.decision,
                    "score": review.score,
                    "feedback": review.feedback[:100],
                    "patches": review.patches[:5],
                },
                outcome="success" if review.is_ok else "corrected",
            )
        )

    # Update plan review stats (thread-safe via GIL for dict mutations)
    stats = state._plan_review_stats
    stats["total_reviews"] = stats.get("total_reviews", 0) + 1
    if review.is_ok:
        stats["approved"] = stats.get("approved", 0) + 1
    else:
        stats["corrected"] = stats.get("corrected", 0) + 1

    # Recompute phase
    state.plan_review_phase = _compute_plan_review_phase(stats)

    # Store as MemRL episode for Q-learning (expert demonstration)
    if state.q_scorer and state.hybrid_router:
        try:
            reward = review.score * 2 - 1  # Map 0-1 to -1..+1
            state.q_scorer.score_external_result(
                task_description=task_ir.get("objective", "")[:200],
                action=f"plan_review:{review.decision}",
                reward=reward,
                context={
                    "task_type": task_ir.get("task_type", "chat"),
                    "review_decision": review.decision,
                    "review_feedback": review.feedback[:100],
                    "source": "plan_review",
                },
            )
        except Exception:
            pass  # MemRL storage is non-critical


def _compute_plan_review_phase(stats: dict) -> str:
    """Compute current plan review phase from statistics.

    Phase A (bootstrap): < 50 reviews or low Q-values
    Phase B (supervised fade): mean Q >= 0.7, min Q >= 0.5
    Phase C (spot-check): min Q >= 0.7 and >= 100 reviews

    Args:
        stats: Plan review statistics dict.

    Returns:
        Phase string: "A", "B", or "C".
    """
    total = stats.get("total_reviews", 0)
    if total < 50:
        return "A"

    q_vals = stats.get("task_class_q_values", {})
    if not q_vals:
        return "A"

    values = list(q_vals.values())
    mean_q = sum(values) / len(values)
    min_q = min(values)

    if min_q >= 0.7 and total >= 100:
        return "C"
    if mean_q >= 0.7 and min_q >= 0.5:
        return "B"
    return "A"


# ── Output Formalizer ──────────────────────────────────────────────────────

def _should_formalize(prompt: str) -> tuple[bool, str]:
    """Detect if the prompt has format constraints that need enforcement.

    Args:
        prompt: The user's prompt.

    Returns:
        Tuple of (should_formalize, format_spec_description).
    """
    if not features().output_formalizer:
        return False, ""

    constraints = detect_format_constraints(prompt)
    if constraints:
        return True, "; ".join(constraints)
    return False, ""


def _formalize_output(
    answer: str,
    prompt: str,
    format_spec: str,
    primitives: "LLMPrimitives",
) -> str:
    """Reformat an answer to satisfy detected format constraints.

    Uses worker_explore (7B, 44 t/s) for fast reformatting.
    The answer content is correct — only format needs fixing.

    Args:
        answer: The correct-content answer to reformat.
        prompt: The original user prompt.
        format_spec: Description of format constraints to satisfy.
        primitives: LLM primitives for inference.

    Returns:
        Reformatted answer, or original if formalization fails.
    """
    import logging
    log = logging.getLogger(__name__)

    formalizer_prompt = build_formalizer_prompt(answer, prompt, format_spec)
    try:
        result = primitives.llm_call(
            formalizer_prompt,
            role="worker_explore",
            n_tokens=2000,
            skip_suffix=True,
        )
        reformatted = result.strip()
        if reformatted and len(reformatted) > 5:
            log.info(f"Formalized output for constraint: {format_spec}")
            return reformatted
        return answer
    except Exception as e:
        log.warning(f"Output formalization failed: {e}")
        return answer


# ── ReAct Tool Loop ─────────────────────────────────────────────────────────

def _parse_react_args(args_str: str) -> dict[str, Any]:
    """Parse ReAct action arguments safely using ast.literal_eval.

    Parses key="value", key=3 format into a dict.
    No eval, no imports — uses ast.literal_eval on individual values.

    Args:
        args_str: The argument string from a ReAct Action line.
            e.g. 'query="quantum computing", max_results=5'

    Returns:
        Dictionary of parsed arguments.
    """
    import ast
    result = {}
    if not args_str or not args_str.strip():
        return result

    # Split on commas, but respect quoted strings
    # Simple state machine: track whether we're inside quotes
    parts = []
    current = []
    in_quotes = False
    quote_char = None
    for ch in args_str:
        if ch in ('"', "'") and not in_quotes:
            in_quotes = True
            quote_char = ch
            current.append(ch)
        elif ch == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
            current.append(ch)
        elif ch == ',' and not in_quotes:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())

    for part in parts:
        if '=' not in part:
            continue
        key, _, val = part.partition('=')
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        try:
            result[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            # If literal_eval fails, treat as string
            result[key] = val
    return result


def _should_use_react_mode(prompt: str, context: str = "") -> bool:
    """Decide if the prompt should use ReAct tool loop instead of direct answer.

    Triggers on tool-needing keywords that don't require REPL
    (search, calculate, date, lookup). Returns False for large context,
    file ops, format-only prompts.

    Args:
        prompt: The user's prompt.
        context: Optional context text.

    Returns:
        True if ReAct mode should be used.
    """
    # Feature flag check
    if not features().react_mode:
        return False

    # REPL handles large context better
    if context and len(context) > 5000:
        return False

    prompt_lower = prompt.lower()

    # Detect tool-needing keywords
    react_indicators = [
        "search for", "look up", "find information",
        "what is the current", "today's date", "what time",
        "calculate", "compute", "evaluate",
        "search arxiv", "search papers", "search wikipedia",
        "look up on wikipedia", "web search",
        "what year", "when did", "how many",
    ]

    return any(ind in prompt_lower for ind in react_indicators)


def _react_mode_answer(
    prompt: str,
    context: str,
    primitives: "LLMPrimitives",
    role: str,
    tool_registry: "Any | None" = None,
    max_turns: int = 5,
) -> str:
    """Execute a ReAct-style tool loop for direct-mode prompts needing tools.

    Builds a ReAct prompt, then loops: LLM generates Thought/Action,
    we execute the Action tool and append Observation, until Final Answer
    is found or max_turns reached.

    Args:
        prompt: The user's question.
        context: Optional context text.
        primitives: LLM primitives for inference.
        role: The LLM role to use.
        tool_registry: Optional tool registry for tool execution.
        max_turns: Maximum Thought/Action/Observation cycles.

    Returns:
        The final answer string.
    """
    import logging
    from src.prompt_builders import build_react_prompt, REACT_TOOL_WHITELIST

    log = logging.getLogger(__name__)

    react_prompt = build_react_prompt(
        prompt=prompt,
        context=context,
        tool_registry=tool_registry,
        max_turns=max_turns,
    )

    conversation = react_prompt

    for turn in range(max_turns):
        # Generate next Thought/Action or Final Answer
        response = primitives.llm_call(
            conversation,
            role=role,
            n_tokens=2048,
            skip_suffix=True,
            stop_sequences=["Observation:", "\n\n\n"],
        )
        response = response.strip()

        if not response:
            log.warning(f"ReAct turn {turn}: empty response")
            break

        conversation += "\n" + response

        # Check for Final Answer
        if "Final Answer:" in response:
            # Extract everything after "Final Answer:"
            idx = response.index("Final Answer:")
            answer = response[idx + len("Final Answer:"):].strip()
            log.info(f"ReAct completed in {turn + 1} turns")
            return answer

        # Parse Action line
        action_match = None
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("Action:"):
                action_match = line[len("Action:"):].strip()
                break

        if not action_match:
            # No action and no final answer — treat entire response as answer
            log.info(f"ReAct turn {turn}: no Action found, treating as answer")
            # Try to extract useful text (skip Thought: prefix)
            lines = response.split("\n")
            answer_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("Thought:"):
                    stripped = stripped[len("Thought:"):].strip()
                answer_lines.append(stripped)
            return "\n".join(answer_lines).strip()

        # Parse tool name and args: "tool_name(arg1="val1", arg2=val2)"
        import re as _re
        tool_match = _re.match(r"(\w+)\((.*)\)$", action_match, _re.DOTALL)
        if not tool_match:
            observation = f"[ERROR: Could not parse action: {action_match}]"
        else:
            tool_name = tool_match.group(1)
            args_str = tool_match.group(2)

            # Safety: only whitelisted tools
            if tool_name not in REACT_TOOL_WHITELIST:
                observation = f"[ERROR: Tool '{tool_name}' is not available in ReAct mode]"
            elif tool_registry is None:
                observation = f"[ERROR: No tool registry available to execute {tool_name}]"
            else:
                try:
                    args = _parse_react_args(args_str)
                    result = tool_registry.invoke(tool_name, "frontdoor", **args)
                    # Truncate large results
                    result_str = str(result)
                    if len(result_str) > 2000:
                        result_str = result_str[:2000] + "... [truncated]"
                    observation = result_str
                except Exception as e:
                    observation = f"[ERROR: {tool_name} failed: {e}]"

        conversation += f"\nObservation: {observation}\n"

    # Max turns reached — extract best answer from conversation
    log.warning(f"ReAct reached max turns ({max_turns})")
    # Look for last Thought that contains useful info
    last_thought = ""
    for line in conversation.split("\n"):
        if line.strip().startswith("Thought:"):
            last_thought = line.strip()[len("Thought:"):].strip()

    if last_thought:
        return f"[ReAct max turns reached]\n{last_thought}"
    return f"[ReAct: Could not determine answer after {max_turns} turns]"


def _should_use_direct_mode(prompt: str, context: str = "") -> bool:
    """Decide if the prompt should bypass the REPL and get a direct LLM answer.

    The REPL wrapper forces the model to generate Python code and call FINAL(),
    which destroys quality on instruction-precision, formatting, and constraint-
    satisfaction tasks. When the task doesn't need tools (peek, grep, list_dir),
    direct mode produces much higher quality output.

    Bypass REPL when:
    - No context or short context (no files to explore)
    - Prompt doesn't reference file operations
    - Prompt doesn't ask for code execution

    Keep REPL when:
    - Large context (needs chunked exploration via peek/grep)
    - Prompt explicitly asks to read/write files
    - Prompt asks to execute or run code

    Args:
        prompt: The user's prompt.
        context: Optional context text.

    Returns:
        True if direct mode should be used.
    """
    prompt_lower = prompt.lower()

    # Keep REPL for large contexts (needs peek/grep/summarize_chunks)
    if context and len(context) > 20000:
        return False

    # Keep REPL when prompt explicitly needs file/tool operations
    repl_indicators = [
        "read the file", "list files", "list the files", "look at the file",
        "open the file", "read from", "write to", "save to",
        "execute", "run the", "run this",
        "search the codebase", "find in the", "grep for",
        "explore the", "scan the",
    ]
    if any(ind in prompt_lower for ind in repl_indicators):
        return False

    # Direct mode for everything else — reasoning, formatting, QA,
    # math proofs, instruction following, tool-call JSON generation, etc.
    return True


def _select_mode(
    prompt: str,
    context: str,
    state: "Any",
) -> str:
    """Select execution mode: direct, react, or repl.

    Uses MemRL route_with_mode() if available, falls back to heuristic chain:
    _should_use_direct_mode() → _should_use_react_mode() → repl.

    Args:
        prompt: The user's prompt.
        context: Optional context text.
        state: Application state (may have hybrid_router).

    Returns:
        One of "direct", "react", or "repl".
    """
    # Try MemRL-based mode selection if available
    if hasattr(state, 'hybrid_router') and state.hybrid_router is not None:
        try:
            task_ir = {
                "task_type": "chat",
                "objective": prompt[:200],
                "priority": "interactive",
                "context_length": len(context) if context else 0,
            }
            _routing, _strategy, mode = state.hybrid_router.route_with_mode(task_ir)
            if mode in ("direct", "react", "repl"):
                return mode
        except Exception:
            pass  # Fall through to heuristic

    # Heuristic fallback: direct → react → repl
    if _should_use_direct_mode(prompt, context):
        if _should_use_react_mode(prompt, context):
            return "react"
        return "direct"
    return "repl"


def _classify_and_route(prompt: str, context: str = "", has_image: bool = False) -> tuple[str, str]:
    """Classify prompt intent and proactively route to the best specialist.

    Zero-latency keyword heuristic. Returns (role, strategy).
    Falls back to frontdoor if no strong signal.

    Args:
        prompt: The user's prompt.
        context: Optional context text.
        has_image: Whether the request includes an image.

    Returns:
        Tuple of (role_name, routing_strategy).
    """
    # Vision: has image data — route to VL server (different model type)
    if has_image:
        return "worker_vision", "classified"

    # Specialist routing: when enabled, use keyword heuristics to route
    # code/architecture tasks to stronger specialists (32B, 235B, 480B).
    # Gated behind feature flag — only activate when Q-values demonstrate
    # clear benefit via comparative seeding (Phase 3).
    if features().specialist_routing:
        prompt_lower = prompt.lower()

        # Code generation / debugging → 32B coder (39 t/s with spec decode)
        code_keywords = [
            "implement", "write code", "function", "class ", "debug",
            "refactor", "fix the bug", "code review", "unit test",
            "algorithm", "data structure", "regex", "parse",
        ]
        if any(kw in prompt_lower for kw in code_keywords):
            return str(Role.CODER_PRIMARY), "classified"

        # Complex code requiring escalation → 32B coder escalation
        complex_code_keywords = [
            "concurrent", "lock-free", "distributed", "optimize performance",
            "memory leak", "race condition", "deadlock",
        ]
        if any(kw in prompt_lower for kw in complex_code_keywords):
            return str(Role.CODER_ESCALATION), "classified"

        # Architecture / system design → 235B architect (6.75 t/s)
        arch_keywords = [
            "architecture", "system design", "design pattern",
            "scalab", "microservice", "trade-off", "tradeoff",
            "invariant", "constraint", "cap theorem",
        ]
        if any(kw in prompt_lower for kw in arch_keywords):
            return str(Role.ARCHITECT_GENERAL), "classified"

    # Default: frontdoor (30B MoE) handles all text prompts
    return str(Role.FRONTDOOR), "rules"


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat request through the orchestrator.

    Modes:
    - mock_mode=True (default): Returns simulated response, no real inference
    - real_mode=True: Uses RadixAttention caching with live llama-server instances
    - Neither: Uses legacy model server (if configured)

    The real_mode flag enables:
    - CachingBackend with prefix routing
    - Cache statistics in response
    - Full orchestration loop with Root LM (Phase 8)
    """
    state = get_state()

    # Track active requests for idle-time Q-scoring (thread-safe)
    state.increment_active()
    try:
        return await _handle_chat(request)
    finally:
        state.decrement_active()


async def _handle_chat(request: ChatRequest) -> ChatResponse:
    """Internal handler for chat requests."""
    state = get_state()
    start_time = time.perf_counter()

    # Generate task ID for MemRL tracking
    task_id = f"chat-{uuid.uuid4().hex[:8]}"

    # Construct task_ir for logging
    task_ir = {
        "task_type": "chat",
        "objective": request.prompt[:200],
        "priority": "interactive",
    }

    # Determine mode (real_mode takes precedence over mock_mode)
    use_mock = request.mock_mode and not request.real_mode

    # Initialize MemRL early for real_mode to enable HybridRouter
    if request.real_mode and not use_mock:
        ensure_memrl_initialized(state)

    # Determine routing using HybridRouter if available, otherwise rules
    if use_mock:
        routing_decision = [request.role or Role.FRONTDOOR]
        routing_strategy = "mock"
    elif request.force_role:
        # Forced role bypasses ALL routing (for comparative seeding/testing)
        routing_decision = [request.force_role]
        routing_strategy = "forced"
    elif request.role and request.role not in ("", "frontdoor"):
        # Explicit specialist role requested — honor it
        routing_decision = [request.role]
        routing_strategy = "explicit"
    elif state.hybrid_router and request.real_mode:
        # Use learned routing for real-mode requests
        routing_decision, routing_strategy = state.hybrid_router.route(task_ir)
    else:
        # Proactive intent classification — route to specialist based on prompt
        has_image = bool(request.image_path or request.image_base64)
        classified_role, routing_strategy = _classify_and_route(
            request.prompt, request.context or "", has_image=has_image,
        )
        routing_decision = [classified_role]

    # Phase 3: Failure graph veto — if a specialist has high failure risk,
    # revert to frontdoor as safety layer. This catches sudden specialist
    # degradation that gradual Q-value updates might miss.
    if (
        state.failure_graph
        and routing_decision
        and str(routing_decision[0]) != str(Role.FRONTDOOR)
        and routing_strategy not in ("mock", "forced")
    ):
        try:
            risk = state.failure_graph.get_failure_risk(str(routing_decision[0]))
            if risk > 0.5:
                import logging as _veto_log
                _veto_log.getLogger(__name__).warning(
                    f"Failure veto: {routing_decision[0]} risk={risk:.2f} > 0.5, "
                    f"reverting to frontdoor"
                )
                routing_decision = [str(Role.FRONTDOOR)]
                routing_strategy = "failure_vetoed"
        except Exception:
            pass  # Veto check is non-critical

    # Log task start (MemRL integration)
    if state.progress_logger:
        state.progress_logger.log_task_started(
            task_id=task_id,
            task_ir=task_ir,
            routing_decision=routing_decision,
            routing_strategy=routing_strategy,
        )

    # Phase 4: Input formalization preprocessing
    # If enabled and prompt qualifies, extract formal specification
    # before specialist handles the request.
    formalization_applied = False
    if (
        features().input_formalizer
        and request.real_mode
        and not use_mock
        and routing_strategy not in ("mock",)
    ):
        from src.formalizer import should_formalize_input, formalize_prompt, inject_formalization
        should_fml, problem_hint = should_formalize_input(request.prompt)
        if should_fml:
            fml_result = formalize_prompt(
                request.prompt, problem_hint, state.registry
            )
            if fml_result.success:
                request.context = inject_formalization(
                    request.prompt, request.context or "", fml_result.ir_json
                )
                formalization_applied = True
                import logging as _fml_log
                _fml_log.getLogger(__name__).info(
                    "Input formalization: %s (%.1fs, %s)",
                    problem_hint,
                    fml_result.elapsed_seconds,
                    fml_result.model_role,
                )

    if use_mock:
        # Mock mode: simulate orchestration
        turns = 1
        answer = f"[MOCK] Processed prompt: {request.prompt[:100]}..."

        if request.context:
            answer += f" (with {len(request.context)} chars of context)"

        elapsed = time.perf_counter() - start_time
        state.increment_request(mock_mode=True, turns=turns)

        # Log task completion (MemRL integration)
        if state.progress_logger:
            state.progress_logger.log_task_completed(
                task_id=task_id,
                success=True,
                details=f"Mock response in {elapsed:.3f}s",
            )
            score_completed_task(state, task_id)

        return ChatResponse(
            answer=answer,
            turns=turns,
            tokens_used=0,
            elapsed_seconds=elapsed,
            mock_mode=True,
            real_mode=False,
            cache_stats=None,
        )

    # Real mode: use RadixAttention with CachingBackend
    if request.real_mode:
        # MemRL already initialized earlier for HybridRouter routing

        # Get server URLs from request or use defaults
        server_urls = request.server_urls or LLMPrimitives.DEFAULT_SERVER_URLS

        try:
            primitives = LLMPrimitives(
                mock_mode=False,
                server_urls=server_urls,
                registry=state.registry,
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize real mode backends: {e}",
            )

        # Verify at least one backend is available
        if not primitives._backends:
            raise HTTPException(
                status_code=503,
                detail="No backends available. Ensure llama-server is running on configured ports.",
            )

    else:
        # Legacy mode: use ModelServer (if configured)
        primitives = LLMPrimitives(mock_mode=False, registry=state.registry)

        if primitives.model_server is None:
            raise HTTPException(
                status_code=503,
                detail="Real inference not available: no model server configured",
            )

    # ── Architect plan review gate ───────────────────────────────────────
    # Synchronous architect review of the frontdoor's tentative plan.
    # Non-blocking: returns None on timeout/error, proceeds without review.
    plan_review_result = None
    if (
        request.real_mode
        and features().plan_review
        and _needs_plan_review(task_ir, routing_decision, state)
    ):
        plan_review_result = _architect_plan_review(
            task_ir, routing_decision, primitives, state, task_id
        )
        if plan_review_result and plan_review_result.decision != "ok":
            routing_decision = _apply_plan_review(routing_decision, plan_review_result)
        if plan_review_result:
            _store_plan_review_episode(state, task_id, task_ir, plan_review_result)

    # Vision routing: when image data or files are present, route through vision pipeline
    # instead of standard text-only orchestration
    has_vision_input = (request.image_path or request.image_base64 or request.files)
    if request.real_mode and has_vision_input:
        try:
            # Multi-file/archive path vs single-image path
            if request.files and len(request.files) > 0:
                answer = await _handle_multi_file_vision(request, primitives, state, task_id)
            else:
                answer = await _handle_vision_request(request, primitives, state, task_id)
            elapsed = time.perf_counter() - start_time
            state.increment_request(mock_mode=False, turns=1)

            if state.progress_logger:
                state.progress_logger.log_task_completed(
                    task_id=task_id,
                    success=True,
                    details=f"Vision pipeline, {elapsed:.3f}s",
                )
                score_completed_task(state, task_id)

            return ChatResponse(
                answer=answer,
                turns=1,
                tokens_used=0,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=True,
                cache_stats=None,
                routed_to="worker_vision",
                role_history=["worker_vision"],
                routing_strategy=routing_strategy,
            )
        except Exception as e:
            import logging
            logging.warning(f"Vision pipeline failed: {type(e).__name__}: {e}")
            # Make image info available to REPL as fallback context
            if request.image_path:
                request.context = (request.context or "") + (
                    f"\n\n[IMAGE: {request.image_path} — Vision pipeline unavailable. "
                    f"Use image analysis tools if available, otherwise note the limitation.]"
                )
            # Fall through to standard orchestration

    # ── Direct-answer mode ──────────────────────────────────────────────
    # For tasks that don't need file access or tool exploration, call the
    # model directly without the REPL Python-code wrapper.  This preserves
    # the model's native instruction-following quality (e.g. 11/11 on
    # instruction_precision vs 2/11 through REPL).
    initial_role = routing_decision[0] if routing_decision else Role.FRONTDOOR

    # Three-way mode selection: direct → react → repl
    # Uses MemRL route_with_mode() when available, heuristic fallback otherwise.
    execution_mode = _select_mode(request.prompt, request.context or "", state)

    # ── ReAct tool loop mode ───────────────────────────────────────────
    # For direct-mode prompts that need tool access (search, calculate, date).
    # Uses ReAct Thought/Action/Observation loop with whitelisted read-only tools.
    if execution_mode == "react" and request.real_mode:
        import logging as _react_log
        _react_log.getLogger(__name__).info(
            f"ReAct mode for {initial_role} (prompt: {len(request.prompt)} chars)"
        )

        try:
            answer = _react_mode_answer(
                prompt=request.prompt,
                context=request.context or "",
                primitives=primitives,
                role=str(initial_role),
                tool_registry=state.tool_registry if hasattr(state, 'tool_registry') else None,
                max_turns=5,
            )
            answer = answer.strip()
        except Exception as e:
            import logging as _rl
            _rl.getLogger(__name__).warning(f"ReAct mode failed ({e}), falling back to direct")
            answer = None

        if answer:
            # Apply post-processing: truncation, quality check, review gate
            answer = _truncate_looped_answer(answer, request.prompt)

            if answer and not answer.startswith("[ERROR") and features().generation_monitor:
                quality_issue = _detect_output_quality_issue(answer)
                if quality_issue:
                    try:
                        escalated = primitives.llm_call(
                            request.prompt,
                            role="coder_escalation",
                            n_tokens=2048,
                            skip_suffix=True,
                        )
                        if escalated.strip():
                            answer = escalated.strip()
                            initial_role = Role.CODER_ESCALATION
                    except Exception:
                        pass

            elapsed = time.perf_counter() - start_time
            state.increment_request(mock_mode=False, turns=1)
            if state.progress_logger:
                state.progress_logger.log_task_completed(
                    task_id=task_id,
                    success=True,
                    details=f"ReAct mode ({initial_role}), {elapsed:.3f}s",
                )
                score_completed_task(state, task_id)

            cache_stats = primitives.get_cache_stats() if primitives._backends else None
            return ChatResponse(
                answer=answer,
                turns=1,
                tokens_used=primitives.total_tokens_generated,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=True,
                cache_stats=cache_stats,
                routed_to=str(initial_role),
                role_history=[str(initial_role)],
                routing_strategy="react",
                tokens_generated=primitives.total_tokens_generated,
                formalization_applied=formalization_applied,
            )

    if execution_mode == "direct" and request.real_mode:
        import logging as _log
        _log.getLogger(__name__).info(
            f"Direct-answer mode for {initial_role} (prompt: {len(request.prompt)} chars)"
        )

        # Build a clean prompt — just the user's question + context, no REPL wrapper
        # NO preamble — any system instruction (even soft) degrades model quality.
        # Wave 1 scored 100% thinking with zero preamble.
        # NOTE: No date injection here. Factual context (date, time, etc.) should
        # come from tools (get_current_date in tool_registry.yaml), not prompt
        # hacking. Direct-answer mode needs a ReAct tool loop (Wave 4) to access
        # tools like web_search, calculate, get_current_date.
        direct_prompt = request.prompt
        if request.context:
            direct_prompt = f"{request.context}\n\n{request.prompt}"

        try:
            answer = primitives.llm_call(
                direct_prompt,
                role=str(initial_role),
                n_tokens=2048,
                skip_suffix=True,  # No registry suffix — it forces elaboration
                stop_sequences=["\n\n\n"],  # Triple-newline = end of response (anti-loop)
            )
            answer = answer.strip()
        except Exception as e:
            # Retry once on transient LLM backend failures
            import logging as _retry_log
            _retry_log.getLogger(__name__).warning(
                f"Direct LLM call failed ({e}), retrying once..."
            )
            try:
                answer = primitives.llm_call(
                    direct_prompt,
                    role=str(initial_role),
                    n_tokens=4096,
                    skip_suffix=True,
                    stop_sequences=["\n\n\n"],
                )
                answer = answer.strip()
            except Exception as e2:
                answer = f"[ERROR: Direct LLM call failed after retry: {e2}]"

        # ── Defense-in-depth: truncate if model loops back to prompt ──
        if answer and not answer.startswith("[ERROR"):
            answer = _truncate_looped_answer(answer, direct_prompt)

        # ── Output formalizer: enforce format constraints ─────────────
        # If the prompt specifies format constraints (word count, JSON, list),
        # reformat the answer to comply. Runs BEFORE quality check so the
        # check sees the finalized output.
        if answer and not answer.startswith("[ERROR"):
            should_fmt, fmt_spec = _should_formalize(request.prompt)
            if should_fmt:
                answer = _formalize_output(answer, request.prompt, fmt_spec, primitives)

        # ── Entropy-inspired output quality check ─────────────────────
        # Post-hoc detection of confused/repetitive output. If the
        # frontdoor model produced garbled output, escalate to a
        # stronger model (coder_primary 32B or architect_general 235B).
        # This is SAFE routing: only triggers on detected failure
        # (unlike keyword routing which caused Wave 2 regressions).
        if answer and not answer.startswith("[ERROR") and features().generation_monitor:
            quality_issue = _detect_output_quality_issue(answer)
            if quality_issue:
                _log.getLogger(__name__).info(
                    f"Output quality issue detected ({quality_issue}), "
                    f"escalating from {initial_role} to coder_escalation"
                )
                try:
                    escalated_answer = primitives.llm_call(
                        direct_prompt,
                        role="coder_escalation",
                        n_tokens=2048,
                        skip_suffix=True,
                    )
                    if escalated_answer.strip():
                        answer = escalated_answer.strip()
                        initial_role = Role.CODER_ESCALATION
                except Exception:
                    pass  # Keep original answer if escalation fails

        # MemRL-informed quality review gate
        if answer and not answer.startswith("[ERROR") and _should_review(
            state, task_id, initial_role, answer
        ):
            verdict = _architect_verdict(
                question=request.prompt,
                answer=answer,
                primitives=primitives,
            )
            if verdict and verdict.upper().startswith("WRONG"):
                corrections = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
                answer = _fast_revise(
                    question=request.prompt,
                    original_answer=answer,
                    corrections=corrections,
                    primitives=primitives,
                )

        elapsed = time.perf_counter() - start_time
        state.increment_request(mock_mode=False, turns=1)
        success = not answer.startswith("[ERROR")

        if state.progress_logger:
            state.progress_logger.log_task_completed(
                task_id=task_id,
                success=success,
                details=f"Direct answer mode ({initial_role}), {elapsed:.3f}s",
            )
            score_completed_task(state, task_id)

        cache_stats = primitives.get_cache_stats() if primitives._backends else None

        return ChatResponse(
            answer=answer,
            turns=1,
            tokens_used=primitives.total_tokens_generated,
            elapsed_seconds=elapsed,
            mock_mode=False,
            real_mode=True,
            cache_stats=cache_stats,
            routed_to=str(initial_role),
            role_history=[str(initial_role)],
            routing_strategy=routing_strategy,
            tokens_generated=primitives.total_tokens_generated,
            formalization_applied=formalization_applied,
        )

    # ── REPL orchestration mode ───────────────────────────────────────
    # For tasks needing file exploration, tool access, or large context.
    # Create REPL environment
    combined_context = request.prompt
    if request.context:
        combined_context += f"\n\nContext:\n{request.context}"

    repl = REPLEnvironment(
        context=combined_context,
        llm_primitives=primitives,
        tool_registry=state.tool_registry,
        script_registry=state.script_registry,
        role=initial_role,
        progress_logger=state.progress_logger,
        task_id=task_id,
        # MemRL components for model self-routing
        retriever=state.hybrid_router.retriever if state.hybrid_router else None,
        hybrid_router=state.hybrid_router,
    )

    # Check for two-stage summarization opportunity
    if request.real_mode and _should_use_two_stage(
        prompt=request.prompt,
        context=request.context,
    ):
        try:
            answer, two_stage_stats = await _run_two_stage_summarization(
                prompt=request.prompt,
                context=request.context or "",
                primitives=primitives,
                state=state,
                task_id=task_id,
            )

            elapsed = time.perf_counter() - start_time
            state.increment_request(mock_mode=False, turns=2)  # Count as 2 turns

            # Log task completion
            if state.progress_logger:
                cache_info = "cache hit" if two_stage_stats.get("cache_hit") else "cache miss"
                state.progress_logger.log_task_completed(
                    task_id=task_id,
                    success=True,
                    details=f"Two-stage summarization ({cache_info}), {elapsed:.3f}s",
                )
                score_completed_task(state, task_id)

            cache_stats = primitives.get_cache_stats() if primitives._backends else None

            return ChatResponse(
                answer=answer,
                turns=2,
                tokens_used=primitives.total_tokens_generated,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=True,
                cache_stats=cache_stats,
                routed_to=str(initial_role),
                role_history=[str(initial_role), str(TWO_STAGE_CONFIG["stage2_role"])],
                routing_strategy=routing_strategy,
                tokens_generated=primitives.total_tokens_generated,
                formalization_applied=formalization_applied,
            )
        except Exception as e:
            # Fall back to standard orchestration on two-stage failure
            import logging
            logging.warning(f"Two-stage summarization failed: {type(e).__name__}: {e}")
            # Continue to normal loop below

    # Detect long context → use REPL-based exploration strategy
    # Instead of dumping full context into one model, the frontdoor uses
    # peek/grep/summarize_chunks to explore and synthesize.
    context_chars = len(combined_context)
    use_long_context_exploration = (
        LONG_CONTEXT_CONFIG["enabled"]
        and request.real_mode
        and context_chars > LONG_CONTEXT_CONFIG["threshold_chars"]
    )

    if use_long_context_exploration:
        import logging
        logging.info(
            f"Long context detected ({context_chars:,} chars). "
            f"Using REPL exploration strategy."
        )

    # Run Root LM orchestration loop with escalation support
    turns = 0
    answer = ""
    last_output = ""
    last_error = ""

    # Escalation tracking
    current_role = initial_role
    consecutive_failures = 0
    role_history = [current_role]
    escalation_prompt = ""  # Set when escalating

    # Long context exploration allows more turns
    max_turns = (
        LONG_CONTEXT_CONFIG["max_turns"]
        if use_long_context_exploration
        else request.max_turns
    )

    for turn in range(max_turns):
        turns += 1

        # 1. Get current REPL state
        repl_state = repl.get_state()

        # 2. Build prompt - use escalation prompt if we just escalated
        if escalation_prompt:
            root_prompt = escalation_prompt
            escalation_prompt = ""  # Clear after use
        elif use_long_context_exploration and turn == 0:
            # First turn with long context: use exploration-aware prompt
            root_prompt = build_long_context_exploration_prompt(
                original_prompt=request.prompt,
                context_chars=context_chars,
                state=repl_state,
            )
        else:
            # Inject routing context on turn 0 (MemRL intelligence)
            routing_ctx = ""
            if turn == 0 and state.hybrid_router:
                routing_ctx = build_routing_context(
                    role=current_role,
                    hybrid_router=state.hybrid_router,
                    task_description=request.prompt,
                )
            root_prompt = build_root_lm_prompt(
                state=repl_state,
                original_prompt=request.prompt,
                last_output=last_output,
                last_error=last_error,
                turn=turn,
                routing_context=routing_ctx,
            )

        # 3. Call Root LM (current role) to generate Python code
        # Use generation monitoring for early failure detection if feature enabled
        f = features()
        generation_aborted = False
        abort_reason = ""

        try:
            if f.generation_monitor and not request.mock_mode:
                # Create monitor with tier-appropriate config
                monitor_config = MonitorConfig.for_tier(current_role)
                monitor = GenerationMonitor(
                    config=monitor_config,
                    mock_mode=request.mock_mode,
                )
                llm_result = primitives.llm_call_monitored(
                    root_prompt,
                    role=current_role,
                    monitor=monitor,
                )
                code = llm_result.text
                generation_aborted = llm_result.aborted
                abort_reason = llm_result.abort_reason
            else:
                # Standard call without monitoring
                code = primitives.llm_call(
                    root_prompt,
                    role=current_role,
                    n_tokens=1024,
                )
        except Exception as e:
            # If LLM call fails, return error
            answer = f"[ERROR: {current_role} LM call failed: {e}]"
            break

        # Handle early abort from generation monitoring
        if generation_aborted:
            # Treat as early failure detection - escalate immediately
            escalation_ctx = EscalationContext(
                current_role=current_role,
                error_message=f"Generation aborted: {abort_reason}",
                error_category="early_abort",
                failure_count=1,  # Treat as first failure to trigger escalation
                task_id=task_id,
            )
            policy = EscalationPolicy()
            decision = policy.decide(escalation_ctx)

            if decision.should_escalate and decision.target_role:
                # Record failure in graph for the role that failed
                if state.failure_graph:
                    try:
                        state.failure_graph.record_failure(
                            memory_id=task_id,
                            symptoms=["early_abort", abort_reason[:100]],
                            description=f"{role_history[-1]} failed: {abort_reason[:200]}",
                            severity=3,
                        )
                    except Exception:
                        pass  # Failure recording is non-critical
                current_role = str(decision.target_role)
                role_history.append(current_role)
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=escalation_ctx,
                    decision=decision,
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=role_history[-2],
                        to_tier=current_role,
                        reason=f"Early abort: {abort_reason}",
                    )
                continue  # Skip to next turn with escalated role
            else:
                # Can't escalate further - try to use partial output
                pass  # Continue with partial code

        # Extract code from response (handle markdown code blocks)
        code = extract_code_from_response(code)
        # Auto-wrap in FINAL() if code looks like a complete answer
        code = auto_wrap_final(code)

        # 4. Execute code in REPL
        result = repl.execute(code)

        # 4a. Check model-initiated routing (escalation/delegation artifacts)
        if repl.artifacts.get("_escalation_requested"):
            target = repl.artifacts.pop("_escalation_target", None)
            reason = repl.artifacts.pop("_escalation_reason", "Model requested")
            repl.artifacts.pop("_escalation_requested", None)

            new_role = None
            if target:
                resolved = Role.from_string(target)
                if resolved:
                    new_role = str(resolved)
            else:
                # No specific target — use standard escalation chain
                esc_ctx = EscalationContext(
                    current_role=current_role,
                    error_category="early_abort",
                    error_message=reason,
                    failure_count=1,
                    task_id=task_id,
                )
                esc_decision = EscalationPolicy().decide(esc_ctx)
                if esc_decision.should_escalate and esc_decision.target_role:
                    new_role = str(esc_decision.target_role)

            if new_role and new_role != current_role:
                current_role = new_role
                role_history.append(current_role)
                consecutive_failures = 0
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=EscalationContext(
                        current_role=role_history[-2],
                        error_message=reason,
                        error_category="early_abort",
                        task_id=task_id,
                    ),
                    decision=EscalationPolicy().decide(EscalationContext(
                        current_role=role_history[-2],
                        error_category="early_abort",
                        error_message=reason,
                        task_id=task_id,
                    )),
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=role_history[-2],
                        to_tier=current_role,
                        reason=f"Model-initiated: {reason}",
                    )
                continue  # Next turn with new role

        # 4b. Log delegation outcomes (MemRL learning)
        if repl.artifacts.get("_delegations"):
            for deleg in repl.artifacts["_delegations"]:
                if state.progress_logger:
                    state.progress_logger.log_exploration(
                        task_id=task_id,
                        query=deleg.get("prompt_preview", ""),
                        strategy_used=f"delegate:{deleg.get('to_role', 'unknown')}",
                        success=deleg.get("success", False),
                    )
            repl.artifacts["_delegations"] = []  # Clear after logging

        # 5. Check for FINAL() completion
        if result.is_final:
            tool_outputs = repl.artifacts.get("_tool_outputs", [])
            answer = _resolve_answer(result, tool_outputs=tool_outputs)
            consecutive_failures = 0  # Success resets failure count

            # MemRL-informed quality review gate (blocking)
            if request.real_mode and _should_review(state, task_id, current_role, answer):
                import logging
                logging.info(f"Review gate triggered for {current_role} (task {task_id})")
                verdict = _architect_verdict(
                    question=request.prompt,
                    answer=answer,
                    primitives=primitives,
                )
                if verdict and verdict.upper().startswith("WRONG"):
                    corrections = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
                    logging.info(f"Review verdict: WRONG — revising ({corrections[:80]})")
                    answer = _fast_revise(
                        question=request.prompt,
                        original_answer=answer,
                        corrections=corrections,
                        primitives=primitives,
                    )
                else:
                    logging.info("Review verdict: OK")

            break

        # 6. Handle errors with EscalationPolicy for escalation decisions
        if result.error:
            consecutive_failures += 1
            last_error = result.error
            last_output = result.output

            # Consult EscalationPolicy for escalation decision (unified module)
            error_category = classify_error(result.error)
            escalation_ctx = EscalationContext(
                current_role=current_role,
                error_message=result.error,
                error_category=error_category.value,  # Pass as string for compatibility
                failure_count=consecutive_failures,
                task_id=task_id,
            )
            policy = EscalationPolicy()
            decision = policy.decide(escalation_ctx)

            # Log escalation decision (for escalate actions)
            if decision.should_escalate and state.progress_logger:
                state.progress_logger.log_escalation(
                    task_id=task_id,
                    from_tier=current_role,
                    to_tier=str(decision.target_role) if decision.target_role else current_role,
                    reason=f"{decision.reason} (failures: {consecutive_failures})",
                )

            # Act on routing decision
            if decision.should_escalate and decision.target_role:
                # Record failure in graph for the role that triggered escalation
                if state.failure_graph:
                    try:
                        state.failure_graph.record_failure(
                            memory_id=task_id,
                            symptoms=[error_category.value, last_error[:100]],
                            description=f"{current_role} failed: {last_error[:200]}",
                            severity=min(consecutive_failures + 2, 5),
                        )
                    except Exception:
                        pass  # Failure recording is non-critical
                # Switch to higher-tier role
                current_role = str(decision.target_role)
                role_history.append(current_role)
                consecutive_failures = 0  # Reset for new role
                # Build escalation prompt with failure context
                escalation_prompt = build_escalation_prompt(
                    original_prompt=request.prompt,
                    state=repl_state,
                    failure_context=escalation_ctx,
                    decision=decision,
                )
            elif decision.action == EscalationAction.EXPLORE:
                # Terminal role — fall back to REPL exploration
                # Switch to exploration prompt so the model uses
                # peek/grep/summarize_chunks instead of raw LLM calls
                consecutive_failures = 0
                escalation_prompt = build_long_context_exploration_prompt(
                    original_prompt=request.prompt,
                    context_chars=len(repl.context),
                    state=repl_state,
                )
                if state.progress_logger:
                    state.progress_logger.log_escalation(
                        task_id=task_id,
                        from_tier=current_role,
                        to_tier=f"{current_role}+explore",
                        reason="Terminal role: switching to REPL exploration",
                    )
            elif decision.action == EscalationAction.FAIL:
                # Max retries/escalations reached - stop with error
                answer = f"[FAILED: {decision.reason}]"
                break
        else:
            # Success - reset failure count but keep role
            consecutive_failures = 0
            last_error = ""
            last_output = result.output

    # If max turns reached without FINAL()
    if not answer:
        # Try to extract substantive content from last_output, stripping tool outputs
        tool_outputs = repl.artifacts.get("_tool_outputs", [])
        cleaned_output = _strip_tool_outputs(last_output, tool_outputs) if last_output else ""

        if cleaned_output and len(cleaned_output) > 20:
            answer = f"[Max turns ({max_turns}) reached without FINAL()]\n\n{cleaned_output}"
        elif last_output:
            answer = f"[Max turns ({max_turns}) reached without FINAL()]\n\nLast output:\n{last_output}"
        else:
            answer = f"[Max turns ({max_turns}) reached without FINAL()]"

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=False, turns=turns)

    # Get cache stats if using real_mode with RadixAttention
    cache_stats = None
    if request.real_mode and primitives._backends:
        cache_stats = primitives.get_cache_stats()

    # Log exploration telemetry (tool usage, function counts)
    success = not answer.startswith("[ERROR") and not answer.startswith("[Max turns")
    repl.log_exploration_completed(success=success, result=answer)

    # Log task completion (MemRL integration)
    if state.progress_logger:
        # Include role history if escalation occurred
        role_info = f", roles: {' -> '.join(role_history)}" if len(role_history) > 1 else ""
        state.progress_logger.log_task_completed(
            task_id=task_id,
            success=success,
            details=f"Real inference: {turns} turns, {elapsed:.3f}s{role_info}",
        )
        score_completed_task(state, task_id)

    return ChatResponse(
        answer=answer,
        turns=turns,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=request.real_mode,
        cache_stats=cache_stats,
        routed_to=str(current_role),
        role_history=[str(r) for r in role_history],
        routing_strategy=routing_strategy,
        tokens_generated=primitives.total_tokens_generated,
        formalization_applied=formalization_applied,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE streaming endpoint with routing metadata.

    Streams events using standardized SSE format (via sse_utils):
    - turn_start: {type: "turn_start", turn: N, role: "..."}
    - thinking: {type: "thinking", content: "..."} (when thinking_budget > 0)
    - token: {type: "token", content: "..."}
    - tool: {type: "tool", name: "...", args: {...}, result: ...}
    - permission_request: {type: "permission_request", id: "...", tool: "...", args: {...}}
    - file: {type: "file", path: "...", content: "...", action: "create"|"modify"}
    - turn_end: {type: "turn_end", tokens: N, elapsed_ms: N}
    - error: {type: "error", message: "..."}
    - done: [DONE] when complete

    Parameters:
    - thinking_budget: Token budget for internal reasoning (0=disabled)
    - permission_mode: "normal", "auto-accept", or "plan"

    Note: Uses sse-starlette when available (via feature flag), otherwise
    falls back to manual SSE formatting for backward compatibility.
    """
    state = get_state()

    # Generate task ID for MemRL tracking (outside generator for closure)
    task_id = f"stream-{uuid.uuid4().hex[:8]}"

    async def generate() -> AsyncGenerator[dict, None]:
        start_time = time.perf_counter()
        use_mock = request.mock_mode and not request.real_mode

        # Construct task_ir and log start (MemRL integration)
        task_ir = {
            "task_type": "chat_stream",
            "objective": request.prompt[:200],
            "priority": "interactive",
        }
        if state.progress_logger:
            state.progress_logger.log_task_started(
                task_id=task_id,
                task_ir=task_ir,
                routing_decision=[request.role],
                routing_strategy="mock" if use_mock else "rules",
            )

        # Mock mode
        if use_mock:
            # Emit turn start
            yield turn_start_event(turn=1, role=str(Role.FRONTDOOR))

            # Emit thinking events if thinking_budget > 0 (Claude Code parity)
            if request.thinking_budget > 0:
                thinking_steps = [
                    "Analyzing the user's request...",
                    f"Request type: {request.prompt[:30].split()[0] if request.prompt else 'unknown'}",
                    "Determining appropriate response strategy...",
                    "Preparing response...",
                ]
                for step in thinking_steps:
                    yield thinking_event(step)

            # Check permission mode - in plan mode, only emit analysis
            if request.permission_mode == "plan":
                analysis = f"[PLAN MODE] Would process: {request.prompt[:100]}..."
                yield token_event(analysis)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                yield turn_end_event(tokens=len(analysis), elapsed_ms=elapsed_ms)
                # Log completion (MemRL)
                if state.progress_logger:
                    state.progress_logger.log_task_completed(task_id, success=True, details="Plan mode")
                    score_completed_task(state, task_id)
                yield done_event()
                return

            # Simulate streaming tokens
            mock_response = f"[MOCK] Processed: {request.prompt[:50]}..."
            for char in mock_response:
                yield token_event(char)

            # Emit turn end
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            yield turn_end_event(tokens=len(mock_response), elapsed_ms=elapsed_ms)

            # Log completion (MemRL)
            if state.progress_logger:
                state.progress_logger.log_task_completed(task_id, success=True, details="Mock stream")
                score_completed_task(state, task_id)
            yield done_event()
            return

        # Real mode - initialize MemRL components on first real use (lazy loading)
        ensure_memrl_initialized(state)

        server_urls = request.server_urls or LLMPrimitives.DEFAULT_SERVER_URLS
        try:
            primitives = LLMPrimitives(mock_mode=False, server_urls=server_urls, registry=state.registry)
        except Exception as e:
            # Log failure (MemRL)
            if state.progress_logger:
                state.progress_logger.log_task_completed(task_id, success=False, details=str(e))
                score_completed_task(state, task_id)
            yield error_event(str(e))
            yield done_event()
            return

        # Create REPL
        combined_context = request.prompt
        if request.context:
            combined_context += f"\n\nContext:\n{request.context}"

        repl = REPLEnvironment(
            context=combined_context,
            llm_primitives=primitives,
            tool_registry=state.tool_registry,
            script_registry=state.script_registry,
            role=request.role or Role.FRONTDOOR,
            # MemRL components for model self-routing
            retriever=state.hybrid_router.retriever if state.hybrid_router else None,
            hybrid_router=state.hybrid_router,
        )

        # Root LM loop with streaming + escalation support
        last_output = ""
        last_error = ""
        result = None

        # Escalation tracking (parity with main endpoint)
        current_role = request.role or Role.FRONTDOOR
        consecutive_failures = 0
        role_history = [current_role]
        escalation_prompt = ""

        for turn in range(request.max_turns):
            turn_start_time_inner = time.perf_counter()

            # Emit turn start with current role
            yield turn_start_event(turn=turn + 1, role=str(current_role))

            # Get state and build prompt
            repl_state = repl.get_state()
            if escalation_prompt:
                root_prompt = escalation_prompt
                escalation_prompt = ""
            else:
                # Inject routing context on turn 0
                routing_ctx = ""
                if turn == 0 and state.hybrid_router:
                    routing_ctx = build_routing_context(
                        role=current_role,
                        hybrid_router=state.hybrid_router,
                        task_description=request.prompt,
                    )
                root_prompt = build_root_lm_prompt(
                    state=repl_state,
                    original_prompt=request.prompt,
                    last_output=last_output,
                    last_error=last_error,
                    turn=turn,
                    routing_context=routing_ctx,
                )

            # Call Root LM with current role
            try:
                code = primitives.llm_call(root_prompt, role=current_role, n_tokens=1024)
            except Exception as e:
                # Log failure (MemRL)
                if state.progress_logger:
                    state.progress_logger.log_task_completed(task_id, success=False, details=f"Root LM failed: {e}")
                    score_completed_task(state, task_id)
                yield error_event(f"Root LM call failed: {e}")
                yield done_event()
                return

            # Stream the generated code tokens
            code = extract_code_from_response(code)
            # Auto-wrap in FINAL() if code looks like a complete answer
            code = auto_wrap_final(code)
            for line in code.split("\n"):
                yield token_event(line + "\n")

            # Execute in REPL
            result = repl.execute(code)

            # Check model-initiated routing artifacts
            if repl.artifacts.get("_escalation_requested"):
                target = repl.artifacts.pop("_escalation_target", None)
                reason = repl.artifacts.pop("_escalation_reason", "Model requested")
                repl.artifacts.pop("_escalation_requested", None)

                new_role = None
                if target:
                    resolved = Role.from_string(target)
                    if resolved:
                        new_role = str(resolved)
                else:
                    esc_ctx = EscalationContext(
                        current_role=current_role,
                        error_category="early_abort",
                        error_message=reason,
                        failure_count=1,
                        task_id=task_id,
                    )
                    esc_decision = EscalationPolicy().decide(esc_ctx)
                    if esc_decision.should_escalate and esc_decision.target_role:
                        new_role = str(esc_decision.target_role)

                if new_role and new_role != current_role:
                    # Emit turn end before role switch
                    turn_elapsed_ms = int((time.perf_counter() - turn_start_time_inner) * 1000)
                    yield turn_end_event(tokens=len(code), elapsed_ms=turn_elapsed_ms)

                    current_role = new_role
                    role_history.append(current_role)
                    consecutive_failures = 0
                    escalation_prompt = build_escalation_prompt(
                        original_prompt=request.prompt,
                        state=repl_state,
                        failure_context=EscalationContext(
                            current_role=role_history[-2],
                            error_message=reason,
                            error_category="early_abort",
                            task_id=task_id,
                        ),
                        decision=EscalationPolicy().decide(EscalationContext(
                            current_role=role_history[-2],
                            error_category="early_abort",
                            error_message=reason,
                            task_id=task_id,
                        )),
                    )
                    if state.progress_logger:
                        state.progress_logger.log_escalation(
                            task_id=task_id,
                            from_tier=role_history[-2],
                            to_tier=current_role,
                            reason=f"Model-initiated: {reason}",
                        )
                    continue  # Next turn with new role

            # Log delegation outcomes
            if repl.artifacts.get("_delegations"):
                for deleg in repl.artifacts["_delegations"]:
                    if state.progress_logger:
                        state.progress_logger.log_exploration(
                            task_id=task_id,
                            query=deleg.get("prompt_preview", ""),
                            strategy_used=f"delegate:{deleg.get('to_role', 'unknown')}",
                            success=deleg.get("success", False),
                        )
                repl.artifacts["_delegations"] = []

            # Emit turn end
            turn_elapsed_ms = int((time.perf_counter() - turn_start_time_inner) * 1000)
            yield turn_end_event(tokens=len(code), elapsed_ms=turn_elapsed_ms)

            # Check for completion
            if result.is_final:
                tool_outputs = repl.artifacts.get("_tool_outputs", [])
                stream_answer = _resolve_answer(result, tool_outputs=tool_outputs)

                # MemRL-informed quality review gate (blocking, streaming parity)
                if _should_review(state, task_id, current_role, stream_answer):
                    verdict = _architect_verdict(
                        question=request.prompt,
                        answer=stream_answer,
                        primitives=primitives,
                    )
                    if verdict and verdict.upper().startswith("WRONG"):
                        corrections = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
                        stream_answer = _fast_revise(
                            question=request.prompt,
                            original_answer=stream_answer,
                            corrections=corrections,
                            primitives=primitives,
                        )

                yield final_event(stream_answer)
                break

            # Handle errors with escalation policy
            if result.error:
                consecutive_failures += 1
                last_error = result.error
                last_output = result.output

                error_category = classify_error(result.error)
                esc_ctx = EscalationContext(
                    current_role=current_role,
                    error_message=result.error,
                    error_category=error_category.value,
                    failure_count=consecutive_failures,
                    task_id=task_id,
                )
                decision = EscalationPolicy().decide(esc_ctx)

                if decision.should_escalate and decision.target_role:
                    current_role = str(decision.target_role)
                    role_history.append(current_role)
                    consecutive_failures = 0
                    escalation_prompt = build_escalation_prompt(
                        original_prompt=request.prompt,
                        state=repl_state,
                        failure_context=esc_ctx,
                        decision=decision,
                    )
                    if state.progress_logger:
                        state.progress_logger.log_escalation(
                            task_id=task_id,
                            from_tier=role_history[-2],
                            to_tier=current_role,
                            reason=f"{decision.reason} (failures: {consecutive_failures})",
                        )
                elif decision.action == EscalationAction.FAIL:
                    yield error_event(f"[FAILED: {decision.reason}]")
                    break
            else:
                consecutive_failures = 0
                last_error = ""
                last_output = result.output

        # Log completion (MemRL) - success if we got a final answer
        if state.progress_logger:
            success = result is not None and result.is_final
            role_info = f", roles: {' -> '.join(str(r) for r in role_history)}" if len(role_history) > 1 else ""
            state.progress_logger.log_task_completed(
                task_id, success=success,
                details=f"Stream complete{role_info}",
            )
            score_completed_task(state, task_id)
        yield done_event()

    # Use SSE utilities for response (handles sse-starlette vs manual fallback)
    return create_sse_response(generate())

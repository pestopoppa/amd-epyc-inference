"""Summarization pipeline for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: summarization task detection, two-stage/three-stage
context processing pipeline with worker digest → frontdoor synthesis.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.api.routes.chat_utils import (
    _estimate_tokens,
    TWO_STAGE_CONFIG,
    LONG_CONTEXT_CONFIG,
)

if TYPE_CHECKING:
    from src.api.state import AppState
    from src.llm_primitives import LLMPrimitives


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

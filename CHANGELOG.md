# Changelog

## 2026-02-09

- **Early-stop streaming**: `_early_stop_check` on LLMPrimitives + StopIteration from `on_chunk` aborts generation the moment FINAL() or D| detected. Saves 100-3000 tokens of post-answer rambling.
- **Early-stop regex fix**: `D\|.{2,}` → `D\|.+` — old regex missed single-char answers like `D|7` (`.{2,}` needs 2+ chars, `.` doesn't match `\n`). Now catches any `D|X` with 1+ characters.
- **Architect delegation prompt rewrite**: Old prompt showed `D|<answer>` and `I|brief:<spec>` as side-by-side template examples. Qwen3-235B echoed both (template fill-in-the-blank). Restructured as bullet-list alternatives with "EXACTLY ONE line" + "Do NOT output both". Architect now correctly delegates code tasks to `coder_escalation` instead of answering directly.
- **FrontdoorNode escalation**: Now escalates to `CoderEscalationNode` (port 8081, Qwen2.5-Coder-32B) instead of `CoderNode` (port 8080, same model as frontdoor).
- **REPL defensive mechanisms**: Comment-only guard (all 4 loops), FINAL() rescue (extracts answer from failed code), early-stop (all 3 REPL loops). See Chapter 18.
- **Test parallelism**: `pytest -n 8` is default via pyproject.toml. 4x speedup (67s → 17s). Safe on this machine.
- **Vision pipeline fix**: `_handle_vision_request()` was orphaned — VL models received text-only prompts without images. Added `_execute_vision_multimodal()` Stage 7.5 in `chat.py` that intercepts vision-role requests and routes to the multimodal handler (OCR + base64 image → VL backend).
- **Early-stop timing fix**: `infer_stream_text()` returned `generation_ms=0` when early-stop broke the SSE stream before the `stop:true` event (which carries timing). Now computes timing from wall clock on early-stop.
- **REPL code-artifact clarification**: Instruction updated to tell model "FINAL must contain the code itself, not a status message" for code tasks.
- **Review gate skip**: `force_role` seeding/eval calls skip MemRL quality review gate in `direct_stage.py` and `repl_executor.py`. Prevents expensive architect reviews during seeding.
- **Silent execution guard**: REPL nudges model to call FINAL() when code runs but produces no output/error. Prevents infinite regeneration loops on class/function definitions.
- **REPL safe imports**: `_safe_import()` wrapper allows ~35 whitelisted modules (math, collections, itertools, numpy, re, heapq, etc.) while blocking dangerous ones (os, sys, subprocess). Previously ALL imports failed.
- **`run_python_code(code, stdin, timeout)`**: New REPL tool — runs code as subprocess with stdin support. Alternative to blocked exec() for USACO-style problems.
- **REPL tap separation**: Code execution writes to `/mnt/raid0/llm/tmp/repl_tap.log` (separate from inference tap). TUI shows REPL panel with styled output.
- **TUI 4-panel layout**: 1-line header, 3:5 column ratio (inference maximized), 7:3 right split (stream 70%, REPL 30%).
- **Prompt hot-swap**: All prompts (system prompts, architect, review, formalizer) now resolve via `resolve_prompt()` in `src/prompt_builders/resolver.py`. Reads from `orchestration/prompts/{name}.md` (uncached ~1ms) → fallback constant. Edit .md file → next request uses updated prompt, no API restart. A/B variants via `PROMPT_VARIANT__{name}=v2` env var → reads `{name}.v2.md`.
- **REPL Unicode sanitizer**: `sanitize_code_unicode()` in `src/repl_environment/unicode_sanitizer.py` replaces Unicode chars models copy from questions (°, ×, ÷, −, curly quotes, superscripts, zero-width spaces) with ASCII equivalents before exec(). Fixes `SyntaxError: invalid character '°'` on chemistry/physics problems.

## 2026-02-07

- **Embedder switch**: TaskEmbedder now uses **BGE-large** (1024-dim) instead of Qwen 0.5B.
- **FAISS rebuild required** after the switch; existing 896-d FAISS indexes are incompatible.
- **Reset/backfill flow** now recreates FAISS at 1024-d and updates SQLite `embedding_idx`.

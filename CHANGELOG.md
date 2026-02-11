# Changelog

## 2026-02-11

- **KV cache pressure / cascading timeouts fix** (resolves handoff `bug-kv-cache-pressure-cascading-timeouts.md`):
  - Differentiated timeouts: workers 30-60s, frontdoor/coder 90-120s, architects 600s (was uniform 600s). Circuit breaker opens 10x faster for stalled workers.
  - Explicit HTTP error codes: chat endpoint returns 502/503/504/429 instead of silent 200 OK. `Retry-After` header on 503.
  - Uvicorn workers 2→6 with `--limit-concurrency 4` — reduces head-of-line blocking surface.
  - KV cache budgets: architect_general ctx 32K→16K, architect_coding 32K→8K, plus `--cache-type-k q8_0` (halves KV memory).
  - Per-backend admission control: `AdmissionController` (threading.Semaphore) limits architects to 1 concurrent request, workers to 2-4. Rejects with 429 when queue full.
  - NUMA-aware placement: architects pinned to preferred NUMA nodes (`numactl --preferred=N`) to reduce page migration during concurrent generation.
- **MCQ extraction fix**: `_extract_toon_decision()` MCQ regex `D\|([A-D])(?=[^a-zA-Z]|$)` truncated 42 free-form answers starting with A-D (e.g. "D|A full analysis..." → "D|A"). New regex `D\|([A-D])[.)\],;:]*\s*(?:$|\n)` only matches when the letter is sole content.
- **Early-stop MCQ shortcut removed**: Streaming early-stop regex dropped the MCQ shortcut `D\|[A-D](?=[^a-zA-Z]|$)` — `$` matches end-of-current-text mid-stream, firing before the model finishes. All D| answers now wait for `\n`. Cost: 1 extra token for true MCQ.
- **General D| period truncation fix**: `D\|(.+?)(?:\.\s|\n|D\||$)` stopped at first `. `, truncating "D|B. The reason is..." to "D|B". Removed `\.\s` from termination set.
- **Function repr leak defense**: `FINAL(str(func))` bypassed the `callable()` check. New `_FUNC_REPR_RE` regex in `context.py` catches `<function|class|method X at 0x...>` strings in `_final()`. Safety nets at both `FinalSignal` catch sites in `environment.py` return error results instead of leaking reprs.
- **Debugger prompt bias fix**: System prompt rewritten — code fixes listed FIRST with signal→fix-type taxonomy, edit budget (3 edits/file/session), "When NOT to edit" section. `_edit_counts` dict tracks edits per file; `_build_prompt()` shows history with "BUDGET EXCEEDED" tags.
- **Architect prompt clarification**: `architect_investigate.md` D| format instruction now says "on its own line" to align with own-line extraction regex.

## 2026-02-10

- **Retry race fix**: `pop_retries()` in `ClaudeDebugger` called non-blocking `_collect_background()` — Claude subprocess (40-130s) almost always still running → empty retries. Switched to blocking `_wait_background()`. Retries now fire correctly.
- **Retry queue persistence**: 94 session restarts during overnight run wiped in-memory retry state. New JSONL persistence (`logs/retry_queue.jsonl`) survives script crashes. `_persist_retries()` on queue, `_load_persisted_retries()` on init, `_clear_persisted_retries()` on consume.
- **5 new anomaly detectors** (12 → 17 signals): `repl_no_tools` (REPL mode, 0 tools), `slow_delegation` (hop >120s), `function_repr_leak` (`<function foo at 0x...>` in answer), `status_phrase_final` ("Done"/"Complete" as answer), `misrouted_to_coder` (factual/MCQ sent to coder_escalation).
- **Auto-discovery mechanism**: Debugger now parses `NEW_SIGNAL:` structured proposals from Claude's analysis output. Proposed detectors persisted to `logs/proposed_signals.jsonl` with batch/session context for human review.
- **Architect routing optimization**: `architect_investigate.md` rewritten with explicit rules — factual/MCQ/reading-comprehension → `D|answer` immediately, NEVER delegate to coder_escalation. Competitive programming/debugging → ALWAYS delegate. Added valid roles list.
- **Seeding script refactor parity confirmed**: Monolithic `seed_specialist_routing_v2.py` (1134 LOC) vs 11 refactored modules (4885 LOC) — all 28 CLI flags, all evaluation modes, debugger integration, checkpoint/resume, TUI verified equivalent. Ready to transition.

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

# Context Window Management & Compaction Infrastructure

**Created**: 2026-02-19
**Updated**: 2026-02-19
**Status**: ✅ IMPLEMENTED (all 4 tasks: C2→C3→C1→C4)
**Priority**: HIGH (conversation compactor), MEDIUM (token counter, tool clearing), LOW (cache telemetry)
**Primary Goal**: Prevent context overflow in long orchestrator sessions by adding conversation compaction, accurate token counting, proactive tool-result clearing, and cache-hit telemetry

## Implementation Summary (2026-02-19)

All 4 tasks implemented and tested, plus Delethink research integration (P1-P3). 119 targeted tests pass (21 compactor + 12 tokenizer + others), 0 regressions in full suite (1568 pass).

| Task | What Was Built | Key Files |
|------|---------------|-----------|
| **C2** | `LlamaTokenizer` — HTTP client calling `/tokenize` with MD5-keyed LRU cache, 500ms timeout, `len//4` fallback | `src/llm_primitives/tokenizer.py` (NEW), `tokens.py`, `primitives.py`, `features.py` |
| **C3** | `_clear_stale_tool_outputs()` — regex-based clearing of `<<<TOOL_OUTPUT>>>` blocks, keeps last N, fires at 40% context | `src/graph/helpers.py`, `features.py`, `responses.py` |
| **C1** | Virtual memory compaction — dumps full context to file (zero info loss), generates navigable index with line coordinates via `worker_explore`, keeps recent 20% verbatim | `src/graph/helpers.py` (rewrite), `orchestration/prompts/compaction_index.md` (NEW), `state.py`, `repl_executor.py` |
| **C4** | Enriched `CachingBackend.get_stats()` with `slot_stats` and `token_savings_pct` | `src/prefix_cache.py` |

**Design decisions**:
- C1 uses "virtual memory" pattern (context externalization) instead of lossy summarization — model can `read_file(path, offset=N, limit=M)` to page in details on demand
- Two-stage pressure management: C3 fires at 40% (cheap regex), C1 fires at 60% (LLM call)
- Index prompt includes line coordinates for one-shot retrieval
- `worker_explore` role avoids SERIAL_ROLES serialization constraint
- All features gated behind feature flags (default off): `accurate_token_counting`, `tool_result_clearing`, `session_compaction`

**Test files**: `tests/unit/test_tokenizer.py` (12 tests), `tests/unit/test_context_compactor.py` (21 tests — 15 original + 6 Delethink P2/P3)
**Related**:
- `handoffs/active/rlm-orchestrator-roadmap.md` (R1 budget propagation)
- `handoffs/active/skillbank-distillation.md` (trajectory compression — different scope)
- `handoffs/active/perf-parallel-tools-concurrent-sweep-prefix-cache.md` (worker_summarize SERIAL_ROLES inconsistency documented there)
- Anthropic API docs: [context-windows](https://platform.claude.com/docs/en/build-with-claude/context-windows), [compaction](https://platform.claude.com/docs/en/build-with-claude/compaction), [context-editing](https://platform.claude.com/docs/en/build-with-claude/context-editing), [prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [token-counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)

**Known blocker for C1**: `worker_summarize` is in `SERIAL_ROLES` (orchestrator_stack.py:157), so compaction calls would be serialized. Use `worker_explore` or `worker_general` role instead (same 7B server, not serial-gated). See perf handoff for full inconsistency details.

### Post-Implementation Verification Update (2026-02-19)

During follow-up verification, C1 tests exposed an environment-sensitive bug: context externalization wrote to hardcoded `/mnt/raid0/llm/tmp/...`, which is not always writable in sandboxed runs. This could silently skip compaction.

Fix applied:
- `src/graph/helpers.py`: replaced hardcoded path with writable tmp-path resolution + write-probe fallback (`ORCHESTRATOR_PATHS_TMP_DIR` / configured tmp / `/mnt/raid0/llm/claude/tmp` / system tmp).
- Added task-id sanitization for generated context filenames.

C4 verification gap closed:
- Added integration assertion that `/chat` response includes `cache_stats` from primitives:
  - `tests/integration/test_chat_pipeline.py::test_chat_response_includes_cache_stats_from_primitives`

Verification rerun:
- `pytest -q tests/unit/test_tokenizer.py tests/unit/test_context_compactor.py` → **33 passed**
- `pytest -q tests/integration/test_chat_pipeline.py -k cache_stats_from_primitives` → **1 passed**
- `pytest -q -n 0 tests/integration/test_cache_integration.py::TestLLMPrimitivesCachingIntegration::test_get_cache_stats_aggregates_all_backends tests/integration/test_cache_integration.py::TestLLMPrimitivesCachingIntegration::test_get_stats_includes_cache_info_with_correct_structure` → **2 passed**

---

## 11) Live Deploy-Gate Validation (2026-02-19)

Goal: decide whether C1/C3 are production-rollout ready using runtime evidence.

### What was run

- Script: `scripts/benchmark/validate_compaction_live.py` (updated to force `force_mode=repl`)
- Output artifact: `benchmarks/results/runs/compaction_validation/results_20260219_134334.json`
- Extra focused probe: forced REPL + 26K context + `max_turns=20` (frontdoor)

### Observed runtime behavior

- `baseline_small`: repl, turns=2, coherent answer, C1=false, C3=0
- `c1_medium_context`: repl, turns=4, coherent answer, C1=false, C3=0
- `c1_large_context`: HTTP 504 timeout before compaction, C1=false, C3=0
- Focused probe (`max_turns=20`, 26K context): repl, turns=2, `error_code=504`, C1=false, C3=0

### Interpretation

- Functional correctness is good (unit/integration coverage passes), but **live C1/C3 efficacy is not yet demonstrated** under current orchestration/runtime behavior.
- Requests often terminate (timeout/finalization) before `turns > 5`, so `_maybe_compact_context()` never executes the compaction branch.
- This blocks the handoff’s intended deployment proof: we do not yet have evidence that compaction/clearing materially protect long-running sessions in production conditions.

### Deployment recommendation

- **Do not promote C1/C3 as production-proven optimizations yet.**
- Keep feature flags available, but gate production default-on for these specific claims until live compaction trigger evidence is collected.
- Required follow-up for promotion:
  1. Add a deterministic integration harness that drives >5 REPL turns and asserts `compaction_triggered=true`.
  2. Re-run live validation with relaxed/controlled timeout budget to allow compaction stage to execute.
  3. Run quality comparison (baseline vs post-compaction) only after (1)-(2) consistently trigger C1.

### Live Validation Update (2026-02-19, follow-up)

Implemented deterministic trigger improvements and re-ran live validation:

- Added config knob `session_compaction_min_turns` (default `5`; env `ORCHESTRATOR_CHAT_SESSION_COMPACTION_MIN_TURNS`) so validation can lower the guard to `1`.
- Hardened compaction path: if `worker_explore` index generation fails/timeouts, compaction now falls back to a deterministic index instead of aborting.
- Updated benchmark harness to force REPL mode and surface turn/role telemetry.

Evidence (`benchmarks/results/runs/compaction_validation/results_20260219_135956.json`):
- `baseline_small`: C1=no (expected), answer OK
- `c1_medium_context`: **C1=YES**, `compaction_tokens_saved=2968`, answer OK
- `c1_large_context`: bounded 504 before completion (still no silent hang)
- Context externalization files were written under `tmp/session_*_ctx_*.md` during run.

Revised rollout recommendation:
- **C1 is now live-trigger validated** (with deterministic fallback + configurable turn guard).
- **C3 remains unproven** in this benchmark path (`tool_results_cleared=0` in observed runs), so keep C3 rollout gated pending a tool-heavy live scenario.

Operator rollout decision (2026-02-19):
- Production defaults set to enable both C1 and C3:
  - `session_compaction=True` (already enabled)
  - `tool_result_clearing=True` (newly enabled)
- Runtime API reloaded with both flags enabled.
- Follow-up remains: collect tool-heavy live evidence for C3 effectiveness and tune `keep_recent`/thresholds if needed.

## 0) Research Origin

These features are inspired by Anthropic's Claude API context management suite (compaction, context editing, prompt caching, token counting). The Claude API features themselves are **not directly usable** since our orchestrator runs local llama.cpp models — but the **architectural patterns** address real gaps in our stack. This handoff adapts those patterns for local inference.

---

## 1) Current State Snapshot

### What Exists

| Component | File | What It Does | Gap |
|-----------|------|-------------|-----|
| `ContextManager` | `src/context_manager.py` (482 lines) | Key-value store with 10KB/entry, 100KB total LRU. `build_prompt_context()` truncates to max_chars | Manages step-to-step context, **not** conversation history |
| `TokensMixin` | `src/llm_primitives/tokens.py` (31 lines) | `len(text) // 4` heuristic | Inaccurate; not used for pre-flight validation |
| `_estimate_tokens()` | `src/api/routes/chat_utils.py:101` | Same `len(text) // 4` heuristic | No pre-flight gate |
| `PrefixRouter` + `RadixCache` | `src/prefix_cache.py` (618 lines), `src/radix_cache.py` | RadixAttention-style KV reuse with canonicalization | Cache stats not surfaced in API response |
| `TokenizedRadixCache` | `src/radix_cache.py:391+` | Token-level radix tree with `set_tokenizer()` hook | Tokenizer hook exists but unused for counting |
| `_strip_tool_outputs()` | `src/api/routes/chat_utils.py:118` | Post-hoc removal of tool output strings from captured stdout | Not proactive; doesn't manage conversation context tokens |
| `Checkpoint` system | `src/session/models.py` | Snapshots every 5 turns / 30min idle / explicit save | No auto-trigger on "context getting full" |
| `ResumeContext.last_exchanges` | `src/session/models.py:381` | Optional conversation history for session resume | Not managed for size; no summarization |
| `ChatResponse.cache_stats` | `src/api/models/responses.py:40` | Field exists for cache performance stats | Only populated for RadixAttention backends; no per-role breakdown |
| `PromptCompressor` | `src/services/prompt_compressor.py` (196 lines) | LLMLingua-2 BERT extractive compression (0.3-0.7 ratio) | Compresses documents, not conversation turns |
| `chat_summarization` | `src/api/routes/chat_summarization.py` | Two-stage summarization for large documents (>20K chars) | Document-level, not conversation-level |
| Model context windows | `orchestration/model_registry.yaml` | Per-model `max_context` (65K-196K) | Not enforced before inference |

### What's Missing

1. **Conversation history compaction** — no mechanism to auto-summarize old turns when approaching context limits
2. **Accurate token counting** — no pre-flight validation using actual tokenizer
3. **Proactive tool result clearing** — no configurable keep-last-N policy for tool results in conversation
4. **Cache-hit telemetry in API response** — `cache_stats` field underpopulated

---

## 2) Implementation Tasks

### C1: Conversation History Compactor — HIGH priority

**Goal**: When conversation context exceeds a configurable threshold (default: 75% of model's `max_context`), auto-summarize older turns into a compact summary block, keeping recent turns verbatim.

**Design** (adapted from Anthropic's compaction pattern):

```
Conversation: [sys_prompt, user1, asst1, user2, asst2, ..., userN, asstN, userN+1]
                |<-- summarize these older turns -->|  |<-- keep verbatim -->|
```

**Key decisions**:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Summarizer model | Qwen2.5-7B (port 8082, 43.9 t/s measured w/ spec+lookup; up to 103.7 t/s single-request no contention) | Cheap, fast, always loaded (HOT tier). Don't burn architect tokens on summarization. ~700 token summary = 7-16s. **Use `worker_explore` role** (not `worker_summarize` — it's in SERIAL_ROLES). |
| Trigger | `input_tokens > 0.75 * model_max_context` | Leave 25% headroom for response + tool results |
| Keep verbatim | Last 3 user+assistant turns + all tool results from last 2 turns | Recent context is highest-value; tool results may be referenced |
| Summary format | `<compaction_summary>...</compaction_summary>` block replacing older turns | Clear delimiter; model treats as background context |
| Summary prompt | Domain-specific: preserve variable names, file paths, decisions made, errors encountered, next steps | Matches our REPL/coding workflow; not generic chat |
| Pause-after-compact | Optional: `compact_pause=True` lets caller inspect/modify summary before continuing | Useful for debugging; off by default |
| Re-compaction | Allow cascading: if summary itself grows (many compactions), re-summarize the summary | Prevents unbounded growth in ultra-long sessions |

**Implementation plan**:

1. New module: `src/context_compactor.py`
   ```python
   @dataclass
   class CompactionConfig:
       trigger_ratio: float = 0.75          # fraction of max_context
       keep_recent_turns: int = 3           # user+assistant pairs to keep verbatim
       keep_recent_tool_results: int = 2    # tool result turns to keep
       summarizer_role: str = "worker_explore"  # maps to Qwen2.5-7B via model registry (NOT worker_summarize — it's SERIAL_ROLES gated)
       summary_max_tokens: int = 2048       # cap summary length
       enabled: bool = True
       pause_after_compact: bool = False

   class ConversationCompactor:
       def __init__(self, config: CompactionConfig, primitives: LLMPrimitives):
           ...

       def should_compact(self, messages: list[dict], model_max_context: int) -> bool:
           """Check if conversation needs compaction."""
           estimated_tokens = self._count_tokens(messages)
           return estimated_tokens > config.trigger_ratio * model_max_context

       def compact(self, messages: list[dict]) -> CompactionResult:
           """Summarize older turns, return compacted message list."""
           older, recent = self._split_messages(messages)
           summary = self._generate_summary(older)
           return CompactionResult(
               messages=[summary_block] + recent,
               compacted_turns=len(older),
               original_tokens=...,
               compacted_tokens=...,
           )
   ```

2. Integration point: `src/api/routes/chat_pipeline/stages.py` — insert compaction check before inference stage
3. Integration point: `src/session/models.py` — store compaction metadata in `Checkpoint` for session resume
4. Summary prompt template: `orchestration/prompts/compaction_summary.md` (hot-swappable)

**Summary prompt draft**:
```
You are summarizing a conversation between a user and an AI coding assistant.
The summary replaces the older turns and must preserve:
- All variable names, file paths, and code identifiers mentioned
- Decisions made and their rationale
- Errors encountered and resolutions
- Current state of any in-progress work
- Tool invocation results that produced lasting state changes
- Any user preferences or constraints expressed

Write a concise summary (max 500 words) in structured format:
## State: [what was built/modified so far]
## Decisions: [key choices and why]
## Errors: [what failed and how it was fixed]
## Next: [what was about to happen when summary was triggered]
```

**Tests**:
- Unit: compaction triggers at correct threshold
- Unit: recent turns preserved correctly (edge cases: odd number of turns, tool-use mid-turn)
- Unit: summary prompt includes required context
- Integration: `/chat` endpoint survives compaction mid-session (answer quality doesn't degrade)

**Telemetry**: Add to `ChatResponse`:
```python
compaction_triggered: bool = Field(default=False)
compaction_turns_summarized: int = Field(default=0)
compaction_tokens_saved: int = Field(default=0)
```

---

### C2: Pre-flight Token Counter — MEDIUM priority

**Goal**: Replace `len(text) // 4` heuristic with accurate tokenizer-based counting. Add pre-flight validation gate.

**Approach**: llama-server exposes `/tokenize` endpoint. We already have `TokenizedRadixCache.set_tokenizer()` hook in `src/radix_cache.py`. Unify around a shared tokenizer.

**Implementation plan**:

1. New utility: `src/llm_primitives/tokenizer.py`
   ```python
   class LlamaTokenizer:
       """Accurate token counting via llama-server /tokenize endpoint."""

       def __init__(self, server_url: str):
           self.url = f"{server_url}/tokenize"
           self._cache: dict[int, int] = {}  # hash(text) -> token_count (LRU)

       async def count_tokens(self, text: str) -> int:
           """Count tokens using the model's actual tokenizer."""
           ...

       def count_tokens_sync(self, text: str) -> int:
           """Sync wrapper for non-async contexts."""
           ...

       def estimate_tokens(self, text: str) -> int:
           """Fast heuristic fallback when server unavailable."""
           return len(text) // 4  # preserve existing behavior as fallback
   ```

2. Replace `TokensMixin` in `src/llm_primitives/tokens.py` to use `LlamaTokenizer` with heuristic fallback
3. Add pre-flight gate in `src/api/routes/chat_pipeline/stages.py`:
   ```python
   async def preflight_token_check(messages, model_config):
       token_count = await tokenizer.count_tokens(full_prompt)
       if token_count > model_config.max_context * 0.95:
           # trigger compaction (C1) or return 413
           ...
   ```

4. Wire `LlamaTokenizer` into `TokenizedRadixCache.set_tokenizer()` for unified counting

**Fallback behavior**: If `/tokenize` endpoint is unreachable (model server down), fall back to `len(text) // 4` heuristic — never block on tokenizer unavailability.

**Performance**: Cache tokenizer results (prompt hash → token count) to avoid repeated `/tokenize` calls for identical prompts. LRU cache with 1000-entry cap.

**Tie-in with R1**: The RLM roadmap R1 task (budget propagation) can consume accurate token counts instead of heuristics. Update `budget_diagnostics` to include `prompt_tokens_actual` alongside `prompt_tokens_estimated`.

---

### C3: Proactive Tool Result Clearing — MEDIUM priority

**Goal**: Automatically clear old tool results from conversation context when approaching token limits, keeping recent results verbatim. Inspired by Anthropic's `clear_tool_uses_20250919` strategy.

**Design**:

```python
@dataclass
class ToolClearingConfig:
    trigger_ratio: float = 0.60       # trigger before compaction (C1 triggers at 0.75)
    keep_recent: int = 3              # keep last N tool use/result pairs
    exclude_tools: list[str] = field(default_factory=lambda: [
        "recall",        # memory lookups — always valuable
        "my_role",       # role context — cheap, useful
    ])
    clear_inputs: bool = False        # also clear tool call params (not just results)
    placeholder: str = "[Tool result cleared to save context space]"
```

**Integration**: Runs as a pre-compaction step. Order of operations:
1. **C3 (tool clearing)** at 60% context → cheap, preserves conversation structure
2. **C1 (compaction)** at 75% context → expensive but thorough, when clearing isn't enough
3. **Pre-flight gate (C2)** at 95% context → hard stop / reject

**What gets cleared**:
- Old `tool_result` content blocks in conversation history → replaced with placeholder
- Optionally: `tool_use` input params (the request side) when `clear_inputs=True`
- Never cleared: results from `exclude_tools`, results from the last `keep_recent` tool uses

**Implementation**:

1. Add `ToolResultClearer` class to `src/context_compactor.py` (co-located with compactor):
   ```python
   class ToolResultClearer:
       def clear(self, messages: list[dict], config: ToolClearingConfig) -> ClearingResult:
           """Replace old tool results with placeholders."""
           tool_uses = self._find_tool_results(messages)
           to_clear = tool_uses[:-config.keep_recent]
           to_clear = [t for t in to_clear if t.tool_name not in config.exclude_tools]
           for t in to_clear:
               t.content = config.placeholder
           return ClearingResult(
               cleared_count=len(to_clear),
               tokens_freed=...,
           )
   ```

2. Hook into the same pipeline stage as C1, but with lower trigger threshold

**Telemetry**: Add to `ChatResponse`:
```python
tool_results_cleared: int = Field(default=0)
tool_clearing_tokens_freed: int = Field(default=0)
```

---

### C4: Cache-Hit Telemetry — LOW priority

**Goal**: Surface RadixCache hit/miss statistics in `ChatResponse.cache_stats` and add per-role breakdown.

**Current state**: `ChatResponse.cache_stats` (line 40-42 of `src/api/models/responses.py`) exists but is only populated for RadixAttention backends, and the population is incomplete.

**Implementation**:

1. Expand `cache_stats` schema:
   ```python
   cache_stats: dict[str, Any] | None = Field(
       default=None,
       description="Cache performance: {hit: bool, prefix_tokens_cached: int, "
                   "hit_rate_pct: float, slot_id: int, role: str}"
   )
   ```

2. In `src/prefix_cache.py`, `PrefixRouter.get_slot_for_prompt()` already tracks per-slot stats (`hit_count`, `miss_count`, `hit_rate`). Surface these through `LLMPrimitives` into the response path.

3. Add per-role aggregation:
   ```python
   # In PrefixRouter
   def get_role_stats(self) -> dict[str, dict]:
       """Aggregate cache stats by role."""
       return {
           role: {"hits": n_hits, "misses": n_misses, "hit_rate": rate}
           for role, (n_hits, n_misses, rate) in self._role_stats.items()
       }
   ```

4. Wire into `ChatResponse` construction in `src/api/routes/chat_pipeline/stages.py`

**Optional**: Add `/stats/cache` endpoint to `src/api/routes/sessions.py` for dashboarding.

---

## 3) Dependency Map

```
C2 (Token Counter) ─── needed by ──→ C3 (Tool Clearing) ─── runs before ──→ C1 (Compaction)
                                                                                    │
                                                                                    ▼
                                                                        C4 (Cache Telemetry)
                                                                        [independent, any order]
```

**Recommended implementation order**: C2 → C3 → C1 → C4

Rationale: C2 (token counter) is the foundation — both C1 and C3 need accurate token counts to know when to trigger. C3 is simpler than C1 (no summarization model call) and handles the easy case. C1 is the biggest piece. C4 is independent.

---

## 4) Files to Create / Modify

### New Files

| File | Purpose |
|------|---------|
| `src/context_compactor.py` | `ConversationCompactor`, `ToolResultClearer`, `CompactionConfig`, `ToolClearingConfig` |
| `src/llm_primitives/tokenizer.py` | `LlamaTokenizer` — accurate counting via `/tokenize` endpoint |
| `orchestration/prompts/compaction_summary.md` | Hot-swappable summary prompt for C1 |
| `tests/unit/test_context_compactor.py` | Unit tests for compaction logic, tool clearing, edge cases |
| `tests/unit/test_tokenizer.py` | Unit tests for tokenizer with mock server |

### Modified Files

| File | Change |
|------|--------|
| `src/llm_primitives/tokens.py` | Replace `len(text) // 4` with `LlamaTokenizer` + heuristic fallback |
| `src/api/routes/chat_pipeline/stages.py` | Add pre-flight token check, tool clearing stage, compaction stage |
| `src/api/models/responses.py` | Add `compaction_triggered`, `compaction_turns_summarized`, `compaction_tokens_saved`, `tool_results_cleared`, `tool_clearing_tokens_freed` to `ChatResponse` |
| `src/session/models.py` | Store compaction metadata in `Checkpoint` (summary text, turn count, timestamp) |
| `src/prefix_cache.py` | Add `get_role_stats()` method to `PrefixRouter` |
| `src/api/routes/chat_utils.py` | Remove standalone `_estimate_tokens()`, import from tokenizer module |
| `src/features.py` | Add feature flags: `context_compaction`, `tool_result_clearing`, `accurate_token_counting` |
| `src/context_manager.py` | Add `token_count` field to `ContextEntry`, use accurate counting |

---

## 5) Configuration Surface

All features gated by feature flags (disabled by default, enable incrementally):

```python
# src/features.py additions
context_compaction: bool = False           # C1
context_compaction_trigger: float = 0.75   # fraction of max_context
context_compaction_keep_turns: int = 3
tool_result_clearing: bool = False         # C3
tool_result_clearing_trigger: float = 0.60
tool_result_clearing_keep: int = 3
accurate_token_counting: bool = False      # C2
cache_telemetry_verbose: bool = False      # C4
```

Environment variable overrides:
```
ORCHESTRATOR_COMPACTION_ENABLED=1
ORCHESTRATOR_COMPACTION_TRIGGER=0.75
ORCHESTRATOR_TOOL_CLEARING_ENABLED=1
ORCHESTRATOR_TOOL_CLEARING_TRIGGER=0.60
ORCHESTRATOR_ACCURATE_TOKENS=1
```

---

## 6) Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Summarizer loses critical context | Answer quality degrades after compaction | Domain-specific prompt; keep recent turns verbatim; add integration test comparing pre/post-compaction answer quality |
| `/tokenize` endpoint latency adds overhead | Slower request processing | Cache results (LRU 1000 entries); async call; heuristic fallback on timeout (50ms) |
| Tool clearing removes still-needed results | Model references a cleared result | `exclude_tools` allowlist; keep last N results; placeholder text tells model result was cleared |
| Compaction + clearing both fire | Double processing overhead | Ordered triggers (clearing at 60% → compaction at 75%); clearing usually prevents compaction from triggering |
| Summary prompt drifts from coding domain | Summaries miss code context | Hot-swappable prompt in `orchestration/prompts/`; can A/B test via feature flags |

---

## 7) Verification

### Per-task gates

- **C1**: `pytest tests/unit/test_context_compactor.py -k compaction` — threshold logic, turn splitting, summary assembly
- **C2**: `pytest tests/unit/test_tokenizer.py` — accuracy vs heuristic, fallback on server down, caching
- **C3**: `pytest tests/unit/test_context_compactor.py -k tool_clearing` — keep-N logic, exclude_tools, placeholder insertion
- **C4**: Verify `cache_stats` populated in `ChatResponse` via integration test

### End-to-end validation

1. Start orchestrator with all features enabled
2. Run a 20-turn conversation that exceeds 75% of context window
3. Verify: tool results cleared at ~60%, compaction at ~75%, session continues normally
4. Verify: `ChatResponse` includes compaction/clearing telemetry
5. Verify: session resume after compaction preserves critical context

### Quality gate

Compare answer quality on 10 representative prompts:
- Baseline: no compaction (fresh context each time)
- With compaction: after 15+ turns with compaction
- Acceptance: <5% degradation in Claude-as-Judge scores (reuse `scripts/benchmark/corpus_quality_gate.py` pattern)

---

## 8) Resume Commands

```bash
# Start implementation (C2 first)
cd /mnt/raid0/llm/claude

# Verify model registry has context lengths
grep -c 'max_context' orchestration/model_registry.yaml

# Check llama-server /tokenize endpoint availability
curl -s http://localhost:8082/tokenize -d '{"content": "hello world"}' | python3 -m json.tool

# Run existing tests before modifying
pytest tests/unit/test_primitives_extended.py tests/unit/test_session_models.py -x -q

# After implementation
make gates
pytest tests/unit/test_context_compactor.py tests/unit/test_tokenizer.py -v
```

---

## 10) Research References

### Delethink / Markovian Thinker (arXiv:2510.06557, McGill NLP)

**Core claim**: Reasoning in LLMs exhibits a Markovian property — future steps depend primarily on recent state, not full history. Delethink exploits this via fixed-chunk context resets with 100-token positional carryover. With RL training, DeepSeek-R1-Distill-Qwen-1.5B matches 24K LongCoT using only 8K active context.

**How it informed our design**:

1. **State carryover > positional carryover**: Delethink keeps the last 100 tokens (positional). Our compaction index prompt now generates a semantic "Current Execution State" block as the first section — capturing what the system is working on, key values, and next action. This is the semantic equivalent of Delethink's carryover but generated by the 7B indexer rather than positionally sliced. (Insight I1)

2. **Lossless externalization compensates for no RL training**: Delethink's hard context reset works because the model was RL-trained for Markovian behavior. Our models aren't. The `read_file()` escape hatch (paging in old context on demand) compensates — the model can recover any information lost during compaction without needing special training.

3. **Configurable retention ratio**: Delethink shows 100 tokens (<2%) can be sufficient with training. We default to 20% (`keep_recent_ratio=0.20`) as a conservative choice for untrained models, but the ratio is now configurable via `ORCHESTRATOR_CHAT_SESSION_COMPACTION_KEEP_RECENT_RATIO` for future tuning. (Insight I2)

4. **Periodic recompaction**: Delethink uses fixed chunk boundaries. We added optional turn-based recompaction (`session_compaction_recompaction_interval`) to prevent context regrowth after initial compaction. Default off (0); suggested starting value for experimentation: 10 turns. (Insight I3)

5. **Zero-shot Markovian property**: The paper claims off-the-shelf models produce Markovian traces naturally. Worth testing whether our 7B indexer can produce good "execution state" summaries without a more expensive model. (Insight I4 — validation task, no code change needed)

**Not adopted**: Hard context reset (too aggressive for multi-topic sessions), RL training (out of scope), fixed chunk boundaries (threshold-based is more appropriate for heterogeneous conversations).

---

## 9) Non-Goals (Explicitly Out of Scope)

- **Anthropic API integration**: We don't call `api.anthropic.com`; these are local-model adaptations
- **Extended thinking / thinking block clearing**: Not applicable to local llama.cpp models
- **1M context window**: Our models are 64K-196K; no 1M support
- **Claude API `cache_control` field**: We use RadixAttention at the llama.cpp level instead
- **SkillBank trajectory compression**: Different scope — compresses *skills*, not conversation history
- **Document-level summarization**: Already handled by `chat_summarization.py` and `PromptCompressor`

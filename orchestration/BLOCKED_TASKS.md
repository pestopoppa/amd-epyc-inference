# Blocked Tasks

**Last Updated**: 2026-02-19 (Delethink research integration into context-window-management)
**Active blockers**: PR #15225 (MTP), PR #18747 (Paged Attention review), Cmprsr weights, Moshi arch in llama.cpp

---

## Active Tasks

| Task | Blocked On | Priority | Handoff | Status |
|------|------------|----------|---------|--------|
| **Security Audit: GGUF Vulns** | CVE verification on prod | **CRITICAL** | `handoffs/active/security_audit_orchestration_stack.md` | ✅ PARTIAL (P0 fixed: localhost binding) |
| **MTP Refactoring** | PR #15225 merge | **HIGH** | `research/mtp_investigation.md` | ✅ PLAN READY |
| **RLM Orchestrator Roadmap** | — | **HIGH** | `handoffs/active/rlm-orchestrator-roadmap.md` | 🔥 ACTIVE refresh complete. Closed: R1, R2, R3, R5, R6, Phase 6 load-validation. Open: Phase 7 hyperparameter tuning (D5/D6 deferred). |
| **Cmprsr Prompt Compression** | Cmprsr weights release | **HIGH** | `handoffs/active/cmprsr_prompt_compression.md` | 📋 NEW (weights unavailable) |
| **Draft model benchmarks** | — | **HIGH** | `handoffs/active/draft-benchmark.md` | 📋 PARTIAL (Gemma-3 + Qwen3-1.7B→32B done, 5 combos remaining) |
| **Formalizer eval** | — | **HIGH** | `handoffs/active/formalizer-evaluation.md` | 📋 READY (not yet executed) |
| **Paged Attention CoW** | PR #18747 reviewer response | **MEDIUM** | `handoffs/active/paged-attention.md` (Section 9) | 🔄 BLOCKED |
| **SkillBank Experience Distillation** | — | **HIGH** | `handoffs/active/skillbank-distillation.md` | 🔥 ACTIVE Phase 1: SkillBank core (schema, CRUD, FAISS, retriever). 8 phases total. Paper: SkillRL (arXiv:2602.08234). |
| **Delegation/Escalation Factual-Risk Routing Track** | — | **HIGH** | `handoffs/active/delegation-escalation-factual-risk-routing-track.md` | 🔥 ACTIVE Research handoff complete. Next: Phase 0 telemetry integrity (logger contract fixes), then shadow-mode factual-risk routing with seeding+ClaudeDebugger tuning loop. |
| **Orchestration Architecture Optimization (Review Follow-up)** | — | **HIGH** | `handoffs/active/orchestration-architecture-optimization-handoff.md` | 📋 READY. Full phased implementation handoff with code-level guidance, telemetry contract, calibration/CRC plan, workspace architecture, diagrams, and chapter update recommendations. |
| **Hybrid Lookup + Corpus-Augmented Spec Decode** | — | **HIGH** | `handoffs/active/hybrid-lookup-spec-decode.md` | 🔥 ACTIVE Phases 0+0.5+1 COMPLETE. 480B: MoE3+spec=12.74 t/s (2.16x). 30B: MoE6+spec+lookup=47.11 t/s (2.58x). Phase 2: SoftMatcha v2 corpus augmentation pending. |
| **PersonaPlex Voice Interface** | Moshi arch in llama.cpp | **MEDIUM** | `handoffs/active/personaplex_voice_interface.md` | 🔄 BLOCKED |
| **LEANN Vector DB** | — | **MEDIUM** | `handoffs/active/leann_vector_db.md` | 📋 READY (proactive for MemRL scaling, trigger: retrieval >50ms) |
| **MemRL Fading Memory** | — | **MEDIUM** | `handoffs/active/memrl_fading_memory.md` | 📋 NEW (Q-value decay for memory management) |
| **Orchestrator Quality Regression** | — | **HIGH** | `progress/2026-01/2026-01-29.md` | ✅ FIXES APPLIED (direct-answer mode, VL pipeline rewrite, port mapping fix). Re-run benchmark pending. |
| **Orchestrator Quality Roadmap** | — | **HIGH** | `handoffs/active/orchestrator-quality-roadmap.md` | ✅ PHASES 1-3 + DEV TASKS + ALL OPTIMIZATIONS COMPLETE. 1517 tests pass. `POST /config` endpoint added. MemRL DB cleaned. **Live validation pending** (seeding + learning loop + regression gate). |
| **Architecture Review (Final: A / 93/100)** | — | **HIGH** | `handoffs/active/orchestrator-architecture-review.md` | ✅ COMPLETE. All review WIs done + 314 coverage tests (2026-02-02). 2015 tests, 67.48% coverage. Ready for archival. |
| **Orchestration Architecture Roadmap** | — | **MEDIUM** | `handoffs/active/orchestration-architecture-roadmap.md` | ✅ ALL 7 ITEMS COMPLETE (A-G). chat_pipeline.py→package decomposition done. 1517 tests. Ready for archival. |
| **MCP Knowledge Tools** | — | **MEDIUM** | `handoffs/active/mcp-knowledge-tools.md` | ✅ PHASES 1-2 COMPLETE (5 knowledge tools + MCP server, 35 tests passing). Phase 3 (MCP client) deferred. |
| **Document Pipeline Tests** | — | **LOW** | `handoffs/active/document_test_failures.md` | 📋 READY (pytest-asyncio now installed) |
| **Orchestrator real mode** | — | **LOW** | `handoffs/active/orchestrator.md` | 📋 READY (stack infrastructure complete, live verification pending) |
| **JSON Canvas + Plugin Architecture** | — | **MEDIUM** | `handoffs/active/json-canvas-plugin-architecture.md` | 📋 NEW (visual reasoning + MCP tool plugins) |
| **Replay Evaluation Harness** | — | **HIGH** | `handoffs/active/replay-evaluation-harness.md` | ✅ IMPLEMENTATION COMPLETE (8/8 phases done, 75 unit tests passing, 3386 full suite. Live smoke tests + baseline replay run pending.) |
| **Orchestrator Intelligence Improvements** | — | **HIGH** | `handoffs/active/orchestrator-intelligence-improvements.md` | ✅ COMPLETE (7/7 improvements implemented, 3746 tests pass). Live validation pending (seeding with new tunables). |
| **Perf: Parallel Tools + Concurrent Sweep + Prefix Cache** | Live servers for sweep | **HIGH** | `handoffs/active/perf-parallel-tools-concurrent-sweep-prefix-cache.md` | 🔥 ACTIVE. WS1+WS3A/B/C implemented (commit 882aaa0). Remaining: run concurrent sweep (WS2), validate escalation compression quality, verify pre-warmer hit rate. |
| **Context Window Management & Compaction** | — | **HIGH** | `handoffs/active/context-window-management.md` | ✅ IMPLEMENTED (C2→C3→C1→C4 all done + Delethink research integration: P1 state carryover, P2 configurable ratio, P3 recompaction interval). 1568 tests pass. Live validation pending. |
| **Nightshift Automated Maintenance** | Permission model fix | **HIGH** | `nightshift.yaml` + `scripts/nightshift/` | ⚠️ FIRST RUN FAILED. All 7 tasks hit 3-iteration limit with "permission denied" (analysis-only mode). Feb 16 instant exits (status 2) — likely hook failures from dirty git state. Needs: permission config for write-capable staging branch, or task redefinition to stdout/log output. |

---

## MemRL Episodic Memory Integration

**Master Handoff**: `handoffs/active/memrl-episodic-memory.md`
**Benchmark**: `benchmarks/prompts/v1/orchestrator_planning.yaml`
**Paper**: arXiv:2601.03192 (MemRL)

| Phase | Description | Status | Dependencies |
|-------|-------------|--------|--------------|
| 1 | Core Implementation | ✅ COMPLETE | None |
| 2 | Wire Logging | ✅ COMPLETE | Phase 1 |
| 3 | Enable Hybrid Routing | ✅ COMPLETE | Phase 2 |
| 4 | Escalation Learning | ✅ COMPLETE | Phase 3 |
| 4b | Memory Seeding (~5K) | ✅ COMPLETE | Phase 4 |
| 4c | REPL Tool Seeding (48) | ✅ COMPLETE | Phase 4 |
| 5 | REPL Exploration Learning | ✅ COMPLETE | Phase 3 |
| 6 | Claude-as-Judge | OPTIONAL | Phase 3 |
| 7 | FAISS Migration | ✅ COMPLETE | Phase 1 |
| 8 | Model Self-Routing | ✅ COMPLETE | Phase 3 |

### Model Self-Routing Complete (2026-01-29)

Models now have agency in routing decisions via 5 REPL functions:
- `my_role()`, `route_advice()`, `delegate()`, `escalate()`, `recall()` (fixed)
- chat.py checks routing artifacts after every `execute()`
- MemRL Q-values injected on turn 0 as routing context
- Streaming endpoint has full routing parity
- Tier C workers blocked from delegation (deterministic tools only)

Files modified: `repl_environment.py`, `escalation.py`, `prompt_builders/` (now a package), `chat.py`

### Memory Seeding Complete (2026-01-14)

~5,000 memories seeded with 67%/33% success/failure ratio:
- Hierarchical decomposition patterns (70)
- Coding/diverse/template failures (~1,340)
- Probabilistic strategies (~450)
- Tool registry created (608 tools mined)

Seeding scripts: `scripts/seed_*.py`

### REPL Tool Seeding Complete (2026-01-24)

48 canonical REPL tool examples seeded with Q=0.90:
- filesystem (8): `list_dir`, `file_info`, `peek`
- document (6): `ocr_document`, `extract_figure`
- complex (7): Multi-step with `llm_call`
- shell (5): git, ls, find
- search/vision/web/artifacts/memory/escalation/parallel (20)

Seeding scripts: `orchestration/repl_memory/seed_loader.py`

### FAISS Migration Complete (2026-01-27)

Replaced O(n) NumPy mmap with O(log n) FAISS for embedding search:
- **Performance**: ~35x speedup at 500K entries (70ms → 2ms)
- **Backend selection**: `EpisodicStore(use_faiss=True)` (default)
- **Fallback**: `use_faiss=False` for legacy NumPy backend
- **Migration**: `python scripts/migrate_to_faiss.py --db-path PATH`
- **Tests**: 24 unit tests passing

Files created:
- `orchestration/repl_memory/faiss_store.py` - FAISS + NumPy backends
- `scripts/migrate_to_faiss.py` - Migration script
- `tests/unit/test_faiss_store.py` - Unit tests

---

## TOON Format Integration (Token Optimization) — COMPLETE

**Evaluation**: `research/TOON_EVALUATION.md`
**Handoff**: Archived 2026-01-30 (findings in `research/TOON_EVALUATION.md`, code in `src/services/toon_encoder.py`)

### Status: COMPLETE (2026-01-28)

| Phase | Description | Status |
|-------|-------------|--------|
| 1.1 | Install toon library | ✅ COMPLETE |
| 1.2 | Round-trip tests | ✅ COMPLETE (100% pass) |
| 1.3 | LLM generation test | SKIPPED (input-only focus) |
| 2.1 | Token counting | ✅ COMPLETE (55% reduction) |
| 2.2 | TTFT benchmark | 📋 PENDING (needs production) |
| 3.1 | Tool output prototype | ✅ COMPLETE |
| 3.2 | A/B testing | 📋 PENDING |

### Results

| Use Case | JSON tokens | TOON tokens | Reduction |
|----------|-------------|-------------|-----------|
| File listings | 302 | 107 | **64.6%** |
| OCR sections | 521 | 233 | **55.3%** |
| Escalation context | 196 | 113 | **42.3%** |
| Grep hits | — | — | **REJECTED** (Markdown better) |

### Key Files

- `src/services/toon_encoder.py` - TOON encoding utilities (7 functions)
- `tests/unit/test_toon_encoder.py` - 17 unit tests
- `src/repl_environment.py` - `use_toon_encoding` config flag
- `pyproject.toml` - `[toon]` optional dependency

### Integrated REPL Tools

| Tool | TOON Benefit | Status |
|------|--------------|--------|
| `_list_dir()` | **64.6%** | ✅ Integrated |
| `_list_procedures()` | ~55% | ✅ Integrated |
| `_recall()` | ~55% | ✅ Integrated |
| `_file_info()` | Minimal (single object) | Not integrated |
| `_grep()` | **-18%** (worse) | Rejected |

### Usage

```python
config = REPLConfig(use_toon_encoding=True)  # Opt-in
repl = REPLEnvironment(context="...", config=config)
```

---

### Phase 1: Core Implementation (COMPLETE)
- [x] `episodic_store.py` - SQLite + numpy memory storage
- [x] `embedder.py` - Task embedding via BGE-large (1024-dim)
- [x] `retriever.py` - Two-phase retrieval + hybrid router
- [x] `progress_logger.py` - Structured JSONL logging
- [x] `q_scorer.py` - Async Q-value update agent
- [x] `model_registry.yaml` - repl_memory configuration
- [x] `orchestrator_planning.yaml` - Claude-as-Judge benchmark

### Phase 2: Wire Logging (COMPLETE - 2026-01-13)
- [x] Add `ProgressLogger` to dispatcher (`src/dispatcher.py`)
- [x] Log routing decisions in Front Door (`src/api.py`)
- [x] Log gate results in GateRunner (`src/gate_runner.py`)
- [ ] Log escalations in FailureRouter (`src/failure_router.py`) - Deferred to Phase 4

### Phase 3: Enable Hybrid Routing (COMPLETE - 2026-01-13)
- [x] Replace hard-coded routing with `HybridRouter` (`src/dispatcher.py`)
- [x] Add confidence logging for monitoring
- [x] Q-scorer integrated (real-time + idle cleanup in API)

### Phase 4: Escalation Learning (COMPLETE - 2026-01-14)
- [x] Store failure contexts with escalation decisions
- [x] Implement `LearnedEscalationPolicy` in FailureRouter
- [x] Connect to episodic memory

**Implementation:**
- Added `LearnedEscalationPolicy` class that queries episodic memory
- Added `LearnedEscalationResult` dataclass for query results
- Updated `FailureRouter` with optional `retriever` and `progress_logger` parameters
- Hybrid routing: queries learned policy first, falls back to rules
- Escalation decisions logged via `progress_logger.log_escalation()`
- Strategy counts tracked for monitoring ("learned" vs "rules")

### Phase 5: REPL Exploration Learning
- [x] File access in peek()/grep() — `file_path` parameter added (2026-01-24)
- [x] Log exploration strategies in REPLEnvironment — delegation outcomes logged (2026-01-29)
- [x] Implement `EpisodicREPL.suggest_exploration()` — `route_advice()` provides MemRL recommendations (2026-01-29)
- [ ] Track token efficiency metrics

### Architect Delegation (2026-01-30) — COMPLETE
- [x] `read_file` + `list_directory` in REACT_TOOL_WHITELIST
- [x] `build_architect_investigate_prompt()` + `build_architect_synthesis_prompt()` with TOON support
- [x] `_parse_architect_decision()` — TOON/JSON/bare-text parser
- [x] `_architect_delegated_answer()` — multi-loop delegation (max_loops=3)
- [x] `architect_delegation` feature flag + env var `ORCHESTRATOR_ARCHITECT_DELEGATION`
- [x] `"delegated"` in `force_mode` valid set
- [x] Wired into `_handle_chat()` direct-mode block
- [x] Seeding script: `ARCHITECT_MODES = {"direct", "delegated"}`
- [x] 27 unit tests, 884 total tests pass
- [x] Validation script env var prefix bug fixed (bare `SPECIALIST_ROUTING` → `ORCHESTRATOR_SPECIALIST_ROUTING`)
- [x] `ORCHESTRATOR_ARCHITECT_DELEGATION=1` wired into validation steps 2-5b
- [ ] Live validation: `bash scripts/benchmark/run_phase3_validation.sh`
- [ ] Comparative seeding results: `delegated` vs `direct` for architect roles

### Phase 6: Claude-as-Judge (Optional)
- [ ] Run orchestrator_planning.yaml benchmark
- [ ] Evaluate baseline scores
- [ ] Enable graded rewards if beneficial

### MemRL Resume Commands

```bash
# Verify module imports
python3 -c "from orchestration.repl_memory import EpisodicStore, TaskEmbedder; print('OK')"

# Check memory stats
python3 -c "from orchestration.repl_memory import EpisodicStore; print(EpisodicStore().get_stats())"

# Run Q-scorer manually
python3 -c "
from orchestration.repl_memory import EpisodicStore, TaskEmbedder, ProgressLogger, ProgressReader, QScorer
scorer = QScorer(EpisodicStore(), TaskEmbedder(), ProgressLogger(), ProgressReader())
print(scorer.score_pending_tasks())
"
```

---

## RLM-Enhanced Orchestrator Development Phases

**Master Handoff**: `handoffs/active/rlm-orchestrator-roadmap.md`
**Research**: `research/rlm_analysis.md`

| Phase | Description | Status | Dependencies |
|-------|-------------|--------|--------------|
| 1 | Backend Completion | ✅ COMPLETE | None |
| 2 | RLM Enhancements | ✅ COMPLETE | Phase 1 |
| 3 | Escalation Integration | ✅ COMPLETE | Phase 1 |
| 4 | Formalizer Integration | ✅ COMPLETE | Phase 3 |
| 5 | Tool/Script Completion | ✅ COMPLETE (44 tools wired, MCP server + client done, invoke() + find_scripts() done) | None |
| 6 | Early Failure Detection | ✅ IMPLEMENTED (needs fresh contention evidence) | Phase 3 |
| 7 | Hyperparameter Tuning | 🟡 PARTIAL (framework exists, closure evidence pending) | Benchmarks |
| 8 | Trajectory Visualization | 🟡 PARTIAL (debugger diagnostics upgraded; full wave-level UX pending) | Phase 2 |

### Phase 1: Backend Completion (COMPLETE - 2026-01-14)
- [x] Complete LlamaServerBackend HTTP (`src/backends/llama_server.py`)
- [x] Wire CachingBackend init (`src/llm_primitives.py`)
- [x] Connect role→backend routing (`src/llm_primitives.py`)
- [x] Fix real mode initialization (`src/api.py`)

**Note**: All infrastructure is complete. Real inference requires starting llama-server instances.
To test: `llama-server -m MODEL.gguf --host 0.0.0.0 --port 8080` then call API with `real_mode=True`.

### Phase 2: RLM Enhancements (COMPLETE - 2026-01-14)
- [x] Forced exploration validation (`src/repl_environment.py` - REPLConfig.require_exploration_before_final)
- [x] Async `llm_batch_async()` (`src/llm_primitives.py`)
- [x] Configurable recursion depth (`src/llm_primitives.py` - LLMPrimitivesConfig.max_recursion_depth)
- [x] Per-query cost tracking (`src/llm_primitives.py` - QueryCost, start_query/end_query)

**Implementation:**
- Forced exploration: tracks peek/grep/llm_call before FINAL(); opt-in via config
- Async batch: `llm_batch_async()` using asyncio.gather for parallel execution
- Recursion depth: max 5 levels by default, RecursionError on exceed
- Cost tracking: QueryCost dataclass, token estimation, per-query cost in dollars

### Phase 3: Escalation Integration (COMPLETE - 2026-01-14)
- [x] Error classification (`src/api.py` - `_classify_error()`)
- [x] Wire FailureRouter into Root LM loop (`src/api.py`)
- [x] Role switching on escalation (`src/api.py`)
- [x] Gate execution integration (`src/api.py` - FailureContext supports gate_name)

**Implementation:**
- Added `_classify_error()` helper in api.py to map errors to ErrorCategory
- Added `_build_escalation_prompt()` for escalated role context
- Root LM loop now tracks current_role, consecutive_failures, and role_history
- FailureRouter consulted on errors, returns RoutingDecision (retry/escalate/fail)
- Role switching on "escalate" action with escalation prompt
- Escalations logged via `progress_logger.log_escalation()`
- 26 API tests + 51 failure_router tests pass

### Phase 4: Formalizer Integration (COMPLETE - 2026-01-30)
- [x] Feature flag `input_formalizer` (`src/features.py`)
- [x] Formalizer module with detection + invocation + injection (`src/formalizer.py`)
- [x] Formalizer routing (`src/dispatcher.py` — ROLE_MAPPING entries)
- [x] IR → context injection in chat.py (after routing, before execution)
- [x] `formalization_applied` metadata on ChatResponse (`src/api/models/responses.py`)
- [x] Unit tests: 37 tests passing (`tests/unit/test_formalizer.py`)

### Phase 5: Tool/Script Completion (COMPLETE - 2026-01-30)
- [x] Tool registry wired (41 deterministic tools: math, symbolic, numerical, format, etc.)
- [x] Tool result capture (`src/repl_environment.py`) — integrated via REPL tools
- [x] MCP client implementation (`src/mcp_client.py` — shared module, wired into both registries)
- [x] Script `invoke()` method (`src/script_registry.py:313-366` — code, MCP, and command backends)
- [x] Script `find_scripts()` method (`src/script_registry.py:225-300` — fuzzy search with category/tag filters)

### Phase 6: Early Failure Detection
- [x] Generation monitor path exists (`src/generation_monitor.py`, `llm_call_monitored`)
- [x] Feature gating and pipeline wiring (`src/features.py`, `src/api/routes/chat_pipeline/stages.py`)
- [ ] Closure evidence under live contention/delegation load

### Phase 7: Hyperparameter Tuning
- [x] Benchmark infrastructure and dataset adapters exist
- [ ] Consolidated roadmap-owned sweep output artifact (temperature/top_p/expert-count by role)
- [ ] Promote selected tunables into documented runtime defaults with before/after evidence

### Phase 8: Trajectory Visualization
- [x] Debugger includes chain diagnostics (`tool_chains`) and execution metadata
- [ ] Standardize wave-level schema + timeline rendering in debugger
- [ ] Optional UI surfacing (Gradio/API visualization) after schema is stable

---

## Orchestrator Self-Management Infrastructure

**Handoff**: `handoffs/active/orchestrator_self_management.md`
**Plan**: `/home/daniele/.claude/plans/mighty-prancing-pillow.md`
**Status**: ✅ PHASES 1-8 COMPLETE (Phase 9 optional/deferred)

**Goal**: Enable deterministic self-management with ~350 tokens/operation (vs 3000-5000 manual).

| Phase | Description | Status | Dependencies |
|-------|-------------|--------|--------------|
| 1 | Procedure Registry Core | ✅ COMPLETE | — |
| 2 | REPL Integration (9 tools) | ✅ COMPLETE | Phase 1 |
| 3 | Checkpointing & Hot-Swap | ✅ COMPLETE | Phase 2 |
| 4 | Pausable Procedures | ✅ COMPLETE | Phase 2 |
| 5 | Rollback & Approval | ✅ COMPLETE | Phase 2 |
| 6 | Core Procedures (6 YAML) | ✅ COMPLETE | Phase 1 |
| 7 | Memory Integration | ✅ COMPLETE | Phase 6 |
| 8 | Advanced Procedures (5 YAML) | ✅ COMPLETE | Phase 6 |
| 9 | Self-Optimization (Optuna) | ⏸️ DEFERRED | Phase 7 |

### Implementation Stats (2026-01-24)

| Component | Files | Lines |
|-----------|-------|-------|
| Procedure Registry | `procedure_registry.py` | ~980 |
| Procedure Scheduler | `procedure_scheduler.py` | 522 |
| JSON Schema | `procedure.schema.json` | 319 |
| REPL Tools | `repl_environment.py` | +180 |
| Procedure YAMLs | 11 procedures | ~1100 |
| Seed Examples | `seed_examples.json` | 56 examples |
| Unit Tests | `test_procedure_registry.py` | 486 (25 tests) |

### Key Decisions
- **No K8s/Terraform**: Single-machine EPYC 9655 doesn't benefit; NUMA/mmap friction
- **Token efficiency**: Procedures do ALL parsing, file updates, validation
- **Approval workflow**: All changes via patches for owner review

### Test Commands
```bash
# Run procedure registry tests
python -m pytest tests/unit/test_procedure_registry.py -v

# List all procedures
python3 -c "from orchestration.procedure_registry import ProcedureRegistry; r=ProcedureRegistry(); print(r.list_procedures())"
```

---

## Session Persistence Layer

**Handoff**: `handoffs/completed/session_persistence.md`
**Status**: ✅ ALL 7 PHASES COMPLETE (2026-01-26)

**Goal**: Enable session checkpoint/resume, document caching, key findings tracking, CLI tools.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core Persistence (SQLiteSessionStore) | ✅ COMPLETE |
| 2 | Document Caching (hash-based change detection) | ✅ COMPLETE |
| 3 | Checkpoint & Resume (REPL state serialization) | ✅ COMPLETE |
| 4 | Idle Monitoring & Auto-Summary | ✅ COMPLETE |
| 5 | Key Findings (mark_finding, heuristic extraction) | ✅ COMPLETE |
| 6 | MemRL Integration (ProgressLogger events) | ✅ COMPLETE |
| 7 | CLI & UX (orch sessions commands) | ✅ COMPLETE |

### CLI Commands
```bash
orch sessions list [--status STATUS] [--project PROJECT]
orch sessions search QUERY
orch sessions show SESSION_ID [--findings] [--checkpoints]
orch sessions resume SESSION_ID [--output json|text]
orch sessions archive SESSION_ID
orch sessions findings SESSION_ID
orch sessions delete SESSION_ID [--force]
orch status
```

---

## Resume Commands

### When PR #15225 Merges (MTP Support)

```bash
# 1. Check PR status
# https://github.com/ggml-org/llama.cpp/pull/15225

# 2. Update llama.cpp and rebuild
cd /mnt/raid0/llm/llama.cpp
git fetch origin master
git merge origin/master
cmake --build build --config Release -j 96

# 3. Test MTP on GLM-4.6
numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/lmstudio/models/unsloth/GLM-4.6-GGUF/GLM-4.6-Q4_K_S-00001-of-00005.gguf \
  --mtp 2 --override-kv glm4moe.expert_used_count=int:4 \
  -t 96 -n 100 -p "Write a Python quicksort:" --no-display-prompt

# 4. Read refactoring plan
cat /mnt/raid0/llm/claude/research/mtp_investigation.md | grep -A 100 "MTP Refactoring Plan"
```

---

## MTP Refactoring Plan (Ready for Implementation)

**Problem:** PR #15225 uses sequential token-by-token processing (defeats MTP benefit).

**Solution:** Batched drafting + parallel verification (like vLLM/SGLang).

**Expected Speedup:** 30-50% over PR #15225 baseline.

**Full details:** `research/mtp_investigation.md`

---

## DEPRECATED: AVX-512 VNNI Q8_0 Optimization

**Status:** NOT SUBMITTING - 8% speedup on small models, 0% on larger models. Bottleneck is elsewhere.

**Benchmark Results (2026-01-08):**
- Qwen2.5-Coder-0.5B Q8_0: 155 t/s vs 144 t/s = 8% speedup
- Qwen3-1.7B Q8_0: ~50 t/s both = 0% speedup
- DeepSeek-R1-8B Q8_0: ~13 t/s both = 0% speedup

**Code still in tree** (tests pass) but not worth PR overhead for 8% gain.

### Files Changed

**PR1 - VNNI optimization:**
- `ggml/src/ggml-cpu/arch/x86/quants.c` (added ~40 lines in `ggml_vec_dot_q8_0_q8_0`)

**PR2 - Shared helper header:**
- `ggml/src/ggml-cpu/arch/x86/avx512-helpers.h` (NEW - 42 lines)
- `ggml/src/ggml-cpu/arch/x86/repack.cpp` (removed ~25 lines, added include)

**PR3 - Use shared helper:**
- `ggml/src/ggml-cpu/arch/x86/quants.c` (added include, simplified VNNI code)

### Commit Message: PR2

```
ggml : add shared AVX-512 int8 dot product helpers

Move AVX-512F helper functions from repack.cpp to a new shared header
(arch/x86/avx512-helpers.h) to enable reuse across x86 quantization code.

Moved helpers:
- sum_i16_pairs_acc_int32x16: int16 pairwise sum with accumulator
- mul_sum_us8_pairs_acc_int32x16: unsigned×signed int8 dot product
- mul_sum_i8_pairs_acc_int32x16: signed×signed int8 dot product

The signed×signed helper uses the abs(x) * sign-adjusted(y) pattern to
convert for VNNI's dpbusd instruction, which expects unsigned×signed input.

This refactoring was motivated by PR #XXXXX (AVX-512 VNNI Q8_0 optimization)
which needed the same helper logic. Having these in a shared header avoids
duplication and ensures consistent implementation across quants.c and
repack.cpp.

No functional changes - existing code paths unchanged.
```

### Commit Message: PR3

```
ggml : use shared helper in AVX-512 VNNI Q8_0 vec_dot

Refactor the AVX-512 VNNI path in ggml_vec_dot_q8_0_q8_0 to use the
shared mul_sum_i8_pairs_acc_int32x16 helper from avx512-helpers.h.

This replaces 10 lines of inline signed×signed conversion logic with
a single helper call, improving readability while maintaining identical
generated code.

Before:
  const __m512i ax = _mm512_abs_epi8(qx);
  const __mmask64 blt0 = _mm512_movepi8_mask(qx);
  const __m512i sy = _mm512_mask_sub_epi8(qy, blt0, zero, qy);
  const __m512i sums = _mm512_dpbusd_epi32(zero, ax, sy);

After:
  const __m512i sums = mul_sum_i8_pairs_acc_int32x16(zero, qx, qy);

Depends on PR #YYYY (shared AVX-512 helpers).
```

### PR1 Commit Message Template (needs speed results)

```
ggml : add AVX-512 VNNI optimization for Q8_0 vec_dot

Add AVX-512 VNNI path to ggml_vec_dot_q8_0_q8_0, providing [X]x speedup
over AVX2 on VNNI-capable CPUs (Ice Lake, Zen 4, Zen 5, Sapphire Rapids).

Key features:
- Process 2 Q8_0 blocks per iteration using 512-bit registers
- Use _mm512_dpbusd_epi32 for efficient int8 dot product
- Handle signed×signed via abs(x) * sign-adjusted(y) pattern
- Use broadcast instructions for efficient scale vector creation

Tested on AMD EPYC 9655 (Zen 5):
- Qwen3-0.6B Q8_0: [BASELINE] t/s → [OPTIMIZED] t/s ([X]x speedup)
- Qwen3-1.7B Q8_0: [BASELINE] t/s → [OPTIMIZED] t/s ([X]x speedup)

test-quantize-fns: all 32 types pass
AddressSanitizer: clean
UndefinedBehaviorSanitizer: clean
```

### Submission Order

1. **PR1** first (standalone value)
2. **PR2** after PR1 merged/in-flight (references PR1 number)
3. **PR3** after PR2 merged (depends on PR2)

### Revert Commands (if needed)

```bash
# Revert all changes
cd /mnt/raid0/llm/llama.cpp
git checkout ggml/src/ggml-cpu/arch/x86/quants.c
git checkout ggml/src/ggml-cpu/arch/x86/repack.cpp
rm ggml/src/ggml-cpu/arch/x86/avx512-helpers.h
```

---

```bash
# 1. Formalizer evaluation (3 models)
./scripts/benchmark/bench_formalizers.sh \
  --model /mnt/raid0/llm/models/xLAM-2-1B-fc-r-Q4_K_M.gguf \
  --prompts benchmarks/prompts/v1/formalizer/

./scripts/benchmark/bench_formalizers.sh \
  --model /mnt/raid0/llm/models/xLAM-1b-fc-r.Q4_K_M.gguf \
  --prompts benchmarks/prompts/v1/formalizer/

./scripts/benchmark/bench_formalizers.sh \
  --model /mnt/raid0/llm/models/nexusraven-v2-13b.Q4_K_M.gguf \
  --prompts benchmarks/prompts/v1/formalizer/

# 2. Tree speculation benchmark
./scripts/benchmark/bench_tree_speculation.sh
```

### When YOLO Agent Available

```bash
# Set up path
export PATH="/mnt/raid0/llm/npm-global/bin:/mnt/raid0/llm/tools/devc/bin:$PATH"

# Launch devcontainer
devc /mnt/raid0/llm/claude

# ✅ DONE: Claude-as-Judge BLIND Re-Scoring (77 models, 2026-01-16)
# Results: benchmarks/results/reviews/BLIND_RESCORE_2026-01-16.md
# Handoff deprecated: docs/deprecated/claude_as_judge_consistency_review_2026-01-16.md

# Orchestrator Integration (CODE COMPLETE - LIVE VERIFICATION PENDING):
claude --dangerously-skip-permissions -p \
  "Read research/orchestration_integration_handoff.md. All code is written. \
   Your job is to: 1) Start llama-server instances, 2) Run tests, \
   3) Fix any failures, 4) Run benchmarks until >50% cache hit rate."
```

### When Model Servers Running

```bash
# Start test server (after benchmark completes)
/mnt/raid0/llm/llama.cpp/build/bin/llama-server \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 -c 4096 -np 4 -t 16

# Enable real inference mode in orchestrator
# See: research/orchestrator_handoff.md
```

### When llama.cpp PR #15225 Merges (MTP Support)

```bash
# 1. Check if PR is merged
# https://github.com/ggml-org/llama.cpp/pull/15225

# 2. Update and rebuild llama.cpp
cd /mnt/raid0/llm/llama.cpp
git pull origin master
cmake --build build --config Release -j 96

# 3. Test MTP on GLM-4.6
# See: research/mtp_investigation.md for full details
numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/glm-4-9b-0414-GGUF/glm-4-9b-0414-Q4_K_M.gguf \
  --mtp 2 -t 96 -n 100 \
  -p "Write a Python function to sort a list:"
```

---

## Completion Tracking

### Draft Model Benchmarks (Speed Tests)
- [x] Gemma-3-1B → Gemma-3-12B-IT (K=8,16,24) — WORKS (upstream b7684+)
- [x] Gemma-3-1B → Gemma-3-27B-IT-QAT (K=8,16,24) — WORKS (42-81% acceptance, PR #18720)
- [x] Qwen3-1.7B → Qwen3-32B (K=8,16,24) — 31% accept, 2.4x speedup (benchmarked 2026-01-28)
- [ ] Qwen3-0.6B → Qwen3-32B (K=8,16,24)
- [ ] Qwen3-1.7B → Qwen3-235B-A22B + MoE4 (K=8,16)
- [ ] jukofyork-0.75B → Qwen3-Coder-30B + MoE6 (K=8,16,24)
- [ ] jukofyork-0.75B → Qwen3-Coder-480B + MoE3 (if 30B works)
- [ ] Documentation updated (registry, RESULTS_SUMMARY, etc.)

### Formalizer Evaluation
- [ ] MathSmith-Qwen3-8B evaluated (problem formalization)
- [ ] xLAM-2-1B-fc-r evaluated (tool sequences)
- [ ] xLAM-1B-fc-r evaluated (tool sequences)
- [ ] NexusRaven-V2-13B evaluated (complex functions)
- [ ] Results compared (parsability, completeness, speed)
- [ ] Best model added to `model_registry.yaml`
- [ ] `research/formalizer_evaluation.md` written

### Tree Speculation — ✅ COMPLETE (K=24 optimal)
- [x] Benchmark complete — K=24 identified as optimal draft depth
- [x] Optimal parameters identified — `--draft-max 24` used in production (orchestrator_stack.py)
- [x] Results in RESULTS.md (33 t/s coder, 39 t/s with lookup)
- [x] `model_registry.yaml` updated with K=24 for spec decode roles

### RadixAttention (YOLO Agent) — ✅ COMPLETE (2026-01-07)
- [x] Phase A: Persistent server mode (`src/backends/llama_server.py`)
- [x] Phase B: Sticky slot routing (`src/prefix_cache.py` - PrefixRouter)
- [x] Phase C: Prompt canonicalization (`src/prefix_cache.py` - canonicalize_prompt)
- [x] Phase D: Radix tree cache (`src/radix_cache.py`)
- [x] Phase E: Slot persistence (`src/prefix_cache.py` - save/restore_hot_prefixes)
- [x] Unit tests: 46/46 passing (`tests/unit/test_prefix_cache.py`)
- [ ] Integration benchmark (requires running llama-server)

### Orchestrator Integration (CODE COMPLETE) — 9 Phases
- [x] Phase 1: Server infrastructure (manual startup commands in handoff)
- [x] Phase 2: LLM Primitives integration (`src/llm_primitives.py`)
- [x] Phase 3: Model server factory (`src/model_server.py`)
- [x] Phase 4: Registry update (`orchestration/model_registry.yaml`)
- [x] Phase 5: Integration tests (`tests/integration/test_cache_integration.py`)
- [x] Phase 6: Benchmark script (`scripts/benchmark/bench_cache_performance.py`)
- [x] Phase 7: API integration (`src/api.py` - real_mode param)
- [x] **Phase 8: Root LM Loop** (`src/api.py` - recursive pattern implemented)
- [x] Phase 9: E2E validation (`scripts/test_recursive_orchestration.py`)
- [ ] Cache hit rate >50% on RLM workloads (YOLO agent to verify)
- [ ] Root LM completes multi-turn tasks (YOLO agent to verify)

### MathSmith Re-conversion — ✅ COMPLETE (2026-01-08)
- [x] Downloaded Q4_K_M from mradermacher (no re-conversion needed)
- [x] Path: `/mnt/raid0/llm/models/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M.gguf` (4.7GB)
- [ ] Verify speed (~40-60 t/s expected) — blocked on benchmark
- [ ] Run formalizer benchmark — blocked on benchmark
- [ ] Update model registry

### Orchestrator Real Mode
- [x] Model servers defined (ports 8080-8090, orchestrator_stack.py)
- [x] `llm_call()` implementation complete (src/llm_primitives.py)
- [x] `llm_batch()` implementation complete (src/llm_primitives.py)
- [ ] End-to-end live inference verification (requires all servers running)

### GLM-4.6 MTP Testing (Blocked on PR #15225)
- [ ] llama.cpp PR #15225 merged
- [ ] Rebuild llama.cpp with MTP support
- [ ] Test MTP on GLM-4.6 (n_mtp=2, n_mtp=3)
- [ ] Benchmark acceptance rates and throughput
- [ ] Compare results with vLLM baseline
- [ ] Update model_registry.yaml with MTP performance
- [ ] Correct GLM-4.6 entry (remove incorrect MoE optimization reference)

---

## Paged Attention Copy-on-Write (CoW)

**Blocked on**: Current paged attention PR must be submitted and reviewed first.

**Purpose**: Enable prefix sharing between sequences (e.g., shared system prompts in llama-server).

**Expected benefit**: Additional 10-30% memory savings in multi-sequence scenarios.

**Implementation plan**: See `handoffs/active/paged-attention.md` Section 9 for:
- Infrastructure already in place (ref_count, is_shared(), add_ref())
- What needs to be implemented (seq_cp, cpy_k, cpy_v modifications)
- Test cases to add
- Benchmark plan

**Resume command**:
```bash
cd /mnt/raid0/llm/llama.cpp-experimental
git checkout feature/paged-attention
cat handoffs/active/paged-attention.md  # Section 9 has full plan
```

---

## Notes

- **Benchmark ETA**: Check with `./run_benchmark.py --status` or `pgrep -af llama`
- **Formalizer models**: Already downloaded to `/mnt/raid0/llm/models/`
- **Tree speculation**: Script at `scripts/benchmark/bench_tree_speculation.sh`
- **RadixAttention**: Full implementation plan in `research/radix_attention_handoff.md`
- **MTP for GLM-4.6**: Self-speculative decoding using built-in heads. See `research/mtp_investigation.md`

---

## Archived / Completed

Items moved from Active table on 2026-01-29. Kept for historical reference.

| Task | Handoff | Status |
|------|---------|--------|
| Orchestrator Document Pipeline | `handoffs/completed/orchestrator_document_pipeline.md` | ✅ RESOLVED |
| Orchestrator API Dependencies | `handoffs/active/orchestrator_deps.md` | ✅ RESOLVED |
| REPL File Access | (archived 2026-01-30, handoff deleted) | ✅ RESOLVED |
| LightOnOCR 3.4x Slowdown | `handoffs/completed/lightonocr_slowdown.md` | ✅ RESOLVED (server-side PDF + 300s timeout) |
| Figure Analysis Missing | `handoffs/completed/figure_analysis_missing.md` | ✅ RESOLVED |
| Claude-as-Judge BLIND Re-Scoring | `docs/deprecated/claude_as_judge_consistency_review_2026-01-16.md` | ✅ COMPLETE (77 models scored, deprecated 2026-01-30) |
| Model Registry: Paged Attention Flag | `orchestration/model_registry.yaml` | ✅ COMPLETE (flag in 13 model entries) |
| KV Cache Pressure / Cascading Timeouts | `handoffs/active/bug-kv-cache-pressure-cascading-timeouts.md` | ✅ RESOLVED (6 mitigations: diff timeouts, HTTP codes, admission ctrl, KV budgets, NUMA, workers) |
| MTP ISWA Fix | `handoffs/active/gemma3-swa-spec-decode-fix.md` | ✅ FIXED (3 commits on mtp-branch) |
| Gemma-3 SWA Spec Decode | `handoffs/active/gemma3-swa-spec-decode-fix.md` | ✅ PR #18720 SUBMITTED (94% mem reduction) |
| Prompt Lookup/Lookahead Bugs | `handoffs/completed/swa_prompt_lookup.md` | ✅ PRs #18729 + #18730 SUBMITTED |
| Qwen3-A3B MoE Instability | — | ✅ RESOLVED (stale build issue) |
| Hybrid Lookup+Spec Decode (original) | `handoffs/active/hybrid-lookup-spec-decode.md` | 🔥 PHASES 0-1 COMPLETE. 480B: 12.74 t/s (2.16x). 30B: 47.11 t/s (2.58x). Phase 2 (SoftMatcha corpus augmentation) pending. |
| AVX-512 VNNI Q8_0 | See section below | ❌ NOT SUBMITTING (8% speedup) |
| Tree speculation | `handoffs/active/cpu-optimization.md` | ✅ COMPLETE (K=24 optimal, in production) |
| RadixAttention | `handoffs/active/radix-attention.md` | ✅ VERIFIED (80% hit rate) |
| Orchestrator integration | `handoffs/active/orchestration-integration.md` | ✅ VERIFIED (12/12 tests) |
| MathSmith re-conversion | `handoffs/active/mathsmith-reconversion.md` | ✅ COMPLETE |
| Kernel development | See AVX-512 section | ✅ COMPLETE (no PR — gains too small) |
| Frontend Architecture | `handoffs/active/orchestrator.md` | ✅ COMPLETE |
| CLI Parity Features | `handoffs/active/orchestrator.md` | ✅ COMPLETE |
| AMD PACE Testing | — | ✅ COMPLETE (not adopting) |
| MemRL Episodic Memory | `handoffs/active/memrl-episodic-memory.md` | ✅ PHASES 1-8 COMPLETE (model self-routing) |
| Tool/Script Registry Wiring | `progress/2026-01/2026-01-15.md` | ✅ COMPLETE (41 tools wired) |
| Native Computational Tools | `handoffs/active/native-computational-tools.md` | ✅ PHASES 1-5 COMPLETE |
| Role Mapping Bug | `progress/2026-01/2026-01-15.md` | ✅ FIXED (str(Role.X) returns value) |
| Orchestrator Multi-Model Live Test | `progress/2026-01/2026-01-15.md` | ✅ VERIFIED (5 models, 459GB) |
| Model REPL Tool Compliance | `handoffs/completed/model_repl_tool_compliance.md` | ✅ COMPLETE (34 tests) |
| Orchestrator Self-Management | `handoffs/active/orchestrator_self_management.md` | ✅ PHASES 1-8 COMPLETE |
| Session Persistence Layer | `handoffs/completed/session_persistence.md` | ✅ ALL 7 PHASES COMPLETE |
| TOON Format Integration | `research/TOON_EVALUATION.md` (handoff deleted 2026-01-30) | ✅ COMPLETE (55.6% token reduction) |
| VL Suite Assignment Fix | `progress/2026-01/2026-01-27.md` | ✅ FIXED |
| Orchestrator Benchmark Fixes | `progress/2026-01/2026-01-29.md` | ✅ COMPLETE (7 fixes + direct-answer mode + VL rewrite + port fix + CLI metrics + 11 safety tests) |
| Graphiti MemRL Enhancement | `handoffs/completed/graphiti_memrl_enhancement.md` | ✅ COMPLETE (52 tests) |
| SWA Prompt Lookup Fix | `handoffs/completed/swa_prompt_lookup.md` | ✅ RESOLVED (PRs #18729 + #18730) |

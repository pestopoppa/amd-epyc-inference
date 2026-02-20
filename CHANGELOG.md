# Changelog

## 2026-02-20

- **GraphRouter MemRL augmentation (ICLR 2025, arXiv:2410.03834)**:
  - GNN-based parallel routing signal for cold-start optimization. New models get routing predictions in minutes instead of hours of organic data accumulation.
  - **BipartiteRoutingGraph** (`routing_graph.py`): Kuzu bipartite graph with TaskType/QueryCluster/LLMRole nodes and PERFORMANCE_ON edges. MiniBatchKMeans clustering from EpisodicStore.
  - **LightweightGAT** (`lightweight_gat.py`): Pure numpy 2-layer heterogeneous GAT (1024→32×4→32). Multi-head attention with scatter aggregation, save/load `.npz`.
  - **GraphRouterPredictor** (`graph_router_predictor.py`): TTL-cached inference wrapper, <0.5ms warm cache. Predicts per-role routing scores.
  - **HybridRouter integration** (`retriever.py`): Blend `posterior = (1-w)×retriever + w×graph`. Weight anneals 0.1→0.3 by store size (500→2000 memories). Injected in `route()` and `route_with_mode()`. Telemetry: `graph_router_ready`, `graph_router_weight` in decision metadata.
  - **Feature flag**: `graph_router` (requires `specialist_routing`). Env: `ORCHESTRATOR_GRAPH_ROUTER=1`. Defaults to False in prod (needs GAT training first).
  - **Scripts**: `scripts/graph_router/train_graph_router.py` (offline GAT training with edge masking, BCE loss, cosine LR, early stopping), `scripts/graph_router/onboard_model.py` (inductive new model onboarding).
  - 49 tests across 5 test files. No changes needed to `seed_specialist_routing.py` — data flows are decoupled.
  - Files: `routing_graph.py`, `lightweight_gat.py`, `graph_router_predictor.py` (NEW), `retriever.py`, `src/features.py`, `src/api/services/memrl.py` (MODIFIED), `train_graph_router.py`, `onboard_model.py` (NEW scripts).

- **Fix: OpenAI-format tool_call translation for REPL executor**:
  - Qwen3-Coder-30B emits OpenAI JSON tool_calls (`[{"function":{"name":"web_search",...}}]`) instead of REPL `CALL()` syntax due to instruct training artifact. REPL couldn't execute them, causing degenerate 20+ identical tool_call repetition loops.
  - Added `translate_openai_tool_calls()`: balanced-bracket JSON extraction, deduplication (collapses identical calls), translation to `CALL()` + `print()` syntax. Integrated as preprocessing step in `extract_code_from_response()`.
  - Added `CALL(` to `code_starters` list for direct detection of properly-formatted CALL invocations.
  - 129 existing tests pass, no false positives on normal JSON content.
  - Files: `src/prompt_builders/code_utils.py` (MODIFIED), `src/prompt_builders/__init__.py` (MODIFIED).

- **REPL tool syntax hints on error**:
  - When REPL code execution fails and the code looks like a failed tool call attempt (OpenAI JSON tool_call format, direct tool name mentions), the error message now includes a hint showing correct `CALL('tool_name', arg=value)` syntax and lists available tools.
  - Detects 4 patterns: JSON `"function"/"name"` keys, direct tool names like `web_search(...)`, `tool_call` mentions, `"type": "function"`.
  - Injected in both structured mode (`_execute_structured`) and normal mode (`execute`) error handlers.
  - Addresses Qwen3-Coder degenerate behavior: after a SyntaxError from malformed tool calls, the model previously abandoned tool use entirely due to no guidance on correct syntax.
  - Files: `src/repl_environment/environment.py` (MODIFIED).

- **Slot-erase-on-timeout + delegation timeout fix**:
  - Backend resource leak: when inference lock times out, the holder's llama-server keeps generating tokens nobody reads. Next request can't proceed until the stale generation finishes.
  - Added `_erase_port_slots(port)` and `_lock_holder_ports()` to `inference_lock.py`. On lock timeout, erases holder's processing slots. On error inside lock, erases own slots. Caches working erase strategy per port.
  - Propagated `port` parameter through `inference_lock()` → `_real_call_single()` / `_call_caching_backend()` / `_real_call_monitored()` via `_extract_port(url)` helper.
  - Delegation timeout fix: specialists now get their full role timeout via nested `request_context(deadline_s=...)` instead of being squeezed by parent's remaining deadline. Specialist loop already enforces wall-clock limits via elapsed checks.
  - 202 tests pass (62 lock + 140 delegation).
  - Files: `src/inference_lock.py`, `src/llm_primitives/inference.py`, `src/api/routes/chat_delegation.py` (MODIFIED).
  - **llama.cpp server fix** (production-consolidated): `POST /slots/:id?action=erase` was gated behind `--slot-save-path` (erase doesn't need disk). Also changed erase to force-release processing slots instead of deferring — critical for cancelling in-flight inference. File: `tools/server/server-context.cpp`.

- **Slot/admission alignment: eliminate 50% KV cache waste**:
  - Every backend had 2x more llama-server slots than admission controller allowed. KV cache partitioned across all slots — 50% wasted on idle slots.
  - Aligned based on `concurrent_sweep_20260219` results: frontdoor 4→2 slots, coder_escalation 4→1 (p95 1.98x at concurrency=2), worker 8→1 (all concurrent levels rejected), architects 2→1.
  - Admission limits aligned: coder_escalation 2→1, worker 4→1.
  - Frontdoor timeout: 90→180s (was never applied to running process).
  - Created investigation handoff: `handoffs/active/backend-saturation-504-429.md` — 6 hypotheses, 6-step playbook, timeline of Feb 11-20 hardening work.
  - Files: `orchestration/model_registry.yaml`, `src/api/admission.py` (MODIFIED), `scripts/benchmark/feature_comparison.py` (NEW).

- **Feature Validation Battery: 15 features enabled in production**:
  - Live A/B testing (hot-reload via `POST /config`) across tiers 1-3 with raw response persistence.
  - **Tier 1 (MemRL chain)**: All 4 PASS — specialist_routing (-25.0s), plan_review (-24.8s), architect_delegation (-24.9s), parallel_execution (-25.5s).
  - **Tier 2 (independent)**: 5 PASS — react_mode (-36.8s), output_formalizer (-21.3s), input_formalizer (-16.2s), unified_streaming (-7.9s), model_fallback (-1.5s). 1 BORDERLINE enabled: escalation_compression (+4.8s). 2 FAIL: binding_routing (+6.5s), personas (+20.6s).
  - **Tier 3 (safety)**: 5/6 PASS — approval_gates (-20.6s), cascading_tool_policy (-15.3s), side_effect_tracking (-28.3s), structured_tool_output (-8.1s), resume_tokens (-1.1s). 1 FAIL: credential_redaction (+15.1s, already enabled as safety feature).
  - `src/features.py` production defaults updated (commits `9b7f345` + `123c272`). README updated with active feature table and chapter links.
  - Borderline features rerun with raw response persistence (per-prompt status, tokens, routing, answers).
  - Files: `src/features.py` (MODIFIED), `README.md` (MODIFIED), `feature-validation-battery.md` (UPDATED), results in `benchmarks/results/runs/feature_validation/live/`.

- **ColBERT-Zero research integration (arXiv:2602.16609)**:
  - Query/document prompt prefixes confirmed **unnecessary** for current models (LateOn-Code requires no prefixes per model card). No code changes.
  - PLAID PQ compression confirmed **already active** (`nbits=4`, IVF+PQ hybrid) in `index_codebase.py:79`. Code index 336MB, docs index 31MB.
  - **GTE-ModernColBERT-v1** identified as docs model upgrade candidate: BEIR avg 54.67, LongEmbed 88.39, 128-dim (matches code index), ModernBERT backbone. Replaces answerai-colbert-small-v1 (33M, 96-dim, mismatched). Needs ONNX conversion (no official export).
  - **MemRL distillation architecture** designed: 3-stage pipeline mirroring ColBERT-Zero (unsupervised task embeddings → supervised Q-weighted training → HybridRouter distillation). Key insight: supervised stage before distillation bridges performance gap.
  - Literature review: `research/colbert_zero_review.md`. Handoff: `handoffs/active/colbert-zero-research-integration.md`.
  - Files: `research/colbert_zero_review.md` (NEW), `handoffs/active/colbert-zero-research-integration.md` (NEW), `docs/reference/agent-config/MEMRL_DISTILLATION_DESIGN.md` (NEW), `CHANGELOG.md`, `orchestration/BLOCKED_TASKS.md`, `docs/reference/models/QUIRKS.md`, `docs/reference/benchmarks/RESULTS.md`.

## 2026-02-19

- **Feature Validation Battery created**:
  - `scripts/benchmark/feature_validation.py`: Automated validation runner with FeatureProfile registry, OfflineValidator (unit + replay), LiveValidator (hot-reload A/B testing via POST /config), and ReportGenerator (markdown + CSV).
  - 23 features organized into 5 tiers (T0 trivial → T4 deferred). MemRL chain (T1) tested incrementally B0→B4 for clean attribution.
  - 9 prompt manifests in `benchmarks/prompts/v1/feature_validation/` (45 prompts total).
  - Handoff: `handoffs/active/feature-validation-battery.md`.
  - Files: `feature_validation.py` (NEW), `feature-validation-battery.md` (NEW), `benchmarks/prompts/v1/feature_validation/*.json` (9 NEW), `BLOCKED_TASKS.md` (UPDATED).

- **Phase 1 classifier refactoring COMPLETE**:
  - Extracted final 3 inline heuristics into `src/classifiers/`: `detect_output_quality_issue` → `quality_detector.py`, `strip_tool_outputs` + `truncate_looped_answer` → `output_parser.py`.
  - All 9 heuristics from the routing-intelligence handoff now delegate to the classifiers module. Original functions in `chat_review.py` and `chat_utils.py` are thin wrappers (zero import breakage).
  - Added `output_parsing` section to `orchestration/classifier_config.yaml` documenting tool-stripping patterns and loop detection constants.
  - 61 classifier tests passing; 1592 full suite tests pass (no regressions).
  - Phases 2-6 (factual-risk routing, MemRL input classification) deferred as separate future work.
  - Files: `src/classifiers/output_parser.py` (NEW), `src/classifiers/quality_detector.py` (NEW), `src/classifiers/__init__.py`, `src/api/routes/chat_utils.py`, `src/api/routes/chat_review.py`, `orchestration/classifier_config.yaml`, `tests/unit/test_classifiers.py`.

- **Q3 CLOSED: First-20-token re-query vs keyword-only retrieval**:
  - Ablation across 6 quality gate prompts (32B outputs) measuring V3 4-gram hits: keyword NL = 21 hits, first-20-token re-query = 53 hits (+152%), full output = 981 hits (+4571%).
  - V3 index confirmed working for code n-grams (`"for i in range"` → 24,987 matches). NL keywords hit sparsely via code comments.
  - Re-query latency: 185ms (cached retriever). At 12.6 t/s, costs 2.3 tokens of generation time.
  - Decision: Keyword-only sufficient. Model's own output provides 47x more n-gram material via prompt lookup self-matching than any re-query. All open questions on hybrid-lookup-spec-decode handoff now closed.
  - Files: `scripts/benchmark/q3_requery_ablation.py`, `benchmarks/results/runs/q3_requery/results.json`.

- **Context-window handoff closure + archival**:
  - Finalized and archived handoff to `handoffs/archived/context-window-management.md`.
  - Added deterministic live-trigger support for C1 validation:
    - `session_compaction_min_turns` config/env (`ORCHESTRATOR_CHAT_SESSION_COMPACTION_MIN_TURNS`, default `5`).
    - Compaction fallback index when `worker_explore` index generation fails/timeouts (compaction no longer aborts).
  - Production defaults now enable both:
    - `session_compaction=True` (already enabled)
    - `tool_result_clearing=True` (newly enabled)
  - Live validation evidence captured at `benchmarks/results/runs/compaction_validation/results_20260219_135956.json` (`c1_medium_context` triggered with `compaction_tokens_saved=2968`).
  - Updated docs/knowledgebase trackers:
    - `docs/chapters/10-orchestration-architecture.md`
    - `docs/reference/agent-config/ORCHESTRATION_DEBUG_PLAYBOOK.md`
    - `orchestration/BLOCKED_TASKS.md`

- **Q5: SoftMatcha v2 GloVe/FastText code vocabulary coverage evaluation**:
  - Evaluated whether GloVe (400K vocab) and FastText (2M known + subword) have meaningful coverage of code tokens for soft/fuzzy matching via SoftMatcha v2.
  - **Hypothesis disproved**: Expected <15% coverage, measured 79.2% (GloVe), 74.0% (FastText known), 86.1% (FastText subword) across 999 snippets (226K tokens) from V3 corpus.
  - **Nuance**: High coverage dominated by trivially matchable tokens — operators/punctuation (100%), English keywords (97-100%). Actual code-specific compound identifiers (`self.assertEqual`, `camelCase`) have <3% FastText known coverage.
  - Moses tokenizer artifact: Top OOV tokens are XML entities (`&quot;`, `&apos;`) from sacremoses, not real code vocabulary gaps.
  - Models: GloVe at `/mnt/raid0/llm/cache/gensim-data/`, FastText at `/mnt/raid0/llm/cache/fasttext/cc.en.300.bin`.
  - New: `scripts/benchmark/glove_code_coverage.py`, results at `benchmarks/results/runs/q5_coverage/results.json`.
  - **Step 2: SoftMatcha test index built + queried — Q5 CLOSED**:
    - Built HDF5 inverted file index from 10K V3 snippets (2.5M tokens, 19K vocab, 55.9MB, 23.6s build).
    - All 6 NL test queries return 0 matches at ALL thresholds (1.0 to 0.5). SoftMatcha requires consecutive token matches — NL phrases never appear consecutively in code.
    - Code-pattern diagnostics: `return` finds 9,955 exact but 57,691 soft matches (because `for` ≈ `return` at 0.53 in GloVe — meaningless). Soft matches are noise, not useful code retrieval.
    - Moses tokenizer destroys code structure: `BinarySearchTree` → `binarysearchtree`, `self.search` stays joined.
    - **Decision: Q5 CLOSED** — SoftMatcha v2 architecturally unsuitable for code retrieval. Exact n-gram matching via V3 SQLite remains the correct approach.
    - New: `scripts/benchmark/softmatcha_test_index.py`, results at `benchmarks/results/runs/q5_softmatcha/results.json`.

- **RLM roadmap R3 rollout-tuning closure**:
  - Captured live ON/OFF delegated probes for `depth_model_overrides` using explicit env toggles.
  - Verified telemetry toggle behavior in `budget_diagnostics.depth_override_enabled` and bounded delegated outcomes with explicit diagnostics in both modes.
  - Marked R3 closed for this roadmap cycle in `handoffs/active/rlm-orchestrator-roadmap.md`.

- **RLM roadmap Phase 6 load-validation closure**:
  - Ran targeted early-failure/generation-monitor validation suite:
    - `python3 -m pytest -n 0 tests/unit/test_chat_pipeline_stages.py tests/unit/test_stages.py tests/unit/test_generation_monitor.py -k "generation_monitor or early_abort" -q`
    - result: `47 passed`.
  - Ran concurrent live `/chat` probe with monitor enabled; observed bounded explicit outcomes (successful responses and explicit `504` timeout responses), with no silent hangs.
  - Marked Phase 6 validation complete in roadmap + docs/playbook.

- **Delethink research integration (Markovian Thinker, arXiv:2510.06557)**:
  - Compaction index prompt (`orchestration/prompts/compaction_index.md`) now generates a "Current Execution State" carryover block as the first section — captures active task, key values, and next action (semantic equivalent of Delethink's 100-token positional carryover, generated by 7B indexer).
  - Configurable retention ratio: `session_compaction_keep_recent_ratio` in `ChatPipelineConfig` (default 0.20, env: `ORCHESTRATOR_CHAT_SESSION_COMPACTION_KEEP_RECENT_RATIO`). Allows tuning how much recent context is kept verbatim after compaction.
  - Optional turn-based recompaction: `session_compaction_recompaction_interval` in `ChatPipelineConfig` (default 0, env: `ORCHESTRATOR_CHAT_SESSION_COMPACTION_RECOMPACTION_INTERVAL`). When > 0 and after first compaction, re-triggers every N turns to prevent context regrowth.
  - New `TaskState.last_compaction_turn` field tracks when compaction last fired for recompaction interval logic.
  - Design rationale documented in `docs/chapters/10-orchestration-architecture.md` (why virtual memory over lossy summarization, why execution state leads the index, why 20% default ratio).
  - Research references added to `handoffs/archived/context-window-management.md` (section 10).
  - 6 new unit tests for configurable ratio, recompaction interval, and turn tracking.

- **Context Window Management (C2→C3→C1→C4) — full implementation**:
  - **C2: Pre-flight Token Counter**: New `src/llm_primitives/tokenizer.py` — `LlamaTokenizer` class calling llama-server `/tokenize` with LRU cache (1000 entries, MD5 key on prefix+length). Fallback to `len//4` on timeout/error. Integrated into `TokensMixin` via `_tokenizer` attribute and new `_count_tokens()` convenience method. Feature flag: `accurate_token_counting`. 12 unit tests.
  - **C3: Tool Result Clearing**: New `_clear_stale_tool_outputs()` in `src/graph/helpers.py` — regex-based clearing of `<<<TOOL_OUTPUT>>>...<<<END_TOOL_OUTPUT>>>` blocks from `state.last_output`. Keeps last N blocks (default 2), replaces older with `[Tool result cleared]`. Gated on context size (40% of max_context or 12K chars). Feature flag: `tool_result_clearing`. 8 unit tests.
  - **C1: Enhanced Conversation Compactor**: Rewrote `_maybe_compact_context()` with "virtual memory" pattern — dumps full context to `/mnt/raid0/llm/tmp/session_{id}_ctx_{n}.md` (zero info loss), generates structured index via `worker_explore` with line coordinates for one-shot `read_file()` retrieval, keeps recent ~20% verbatim. Token-aware trigger at 60% of model max_context. Hot-swappable index prompt at `orchestration/prompts/compaction_index.md`. New `TaskState` fields: `compaction_tokens_saved`, `context_file_paths`. 8 unit tests.
  - **C4: Cache-Hit Telemetry**: Enriched `CachingBackend.get_stats()` with `slot_stats` (per-slot hit/miss/rate) and `token_savings_pct` (0-100 convenience field).
  - **ChatResponse** new fields: `tool_results_cleared`, `compaction_triggered`, `compaction_tokens_saved`.
  - New files: `src/llm_primitives/tokenizer.py`, `tests/unit/test_tokenizer.py`, `tests/unit/test_context_compactor.py`, `orchestration/prompts/compaction_index.md`.
  - Modified: `src/features.py` (+2 flags), `src/llm_primitives/tokens.py`, `src/llm_primitives/primitives.py`, `src/graph/helpers.py`, `src/graph/state.py`, `src/api/models/responses.py`, `src/api/routes/chat_pipeline/repl_executor.py`, `src/prefix_cache.py`.
  - Test results: 113 targeted tests pass, 0 regressions in full 1420-test suite.

- **Phase 2B-Sidecar: Corpus draft source in llama.cpp speculation loop — CLOSED** (`llama.cpp-experimental`, branch `feature/corpus-sidecar`):
  - Created pluggable `common_speculative_state_corpus_sidecar` with three modes: blocking (`--corpus-refresh N`), pre-query (`--corpus-refresh 0`), async (`--corpus-refresh -1`).
  - New files: `common/corpus-sidecar.h`, `common/corpus-sidecar.cpp`, `common/md5.h`.
  - Modified: `common/common.h`, `common/speculative.cpp`, `common/arg.cpp`, `common/CMakeLists.txt`.
  - Benchmarked on 30B with V3 corpus (76.6M snippets): all modes negative vs Phase 2A prompt injection.
    - Pre-query: 26.0 t/s (+1% vs baseline, -13% vs Phase 2A)
    - Async: 25.8 t/s (0% vs baseline, -14% vs Phase 2A)
    - Blocking: 24.9 t/s (-3% vs baseline, -17% vs Phase 2A)
    - Phase 2A (prompt injection): 29.9 t/s (+16% vs baseline)
  - Root cause: corpus n-grams injected into `nc_static` don't match draft model proposals effectively. Phase 2A works because injected tokens become part of the prompt context the model naturally matches.
  - Decision: Phase 2A prompt injection remains production approach. Branch preserved for reference, no cherry-pick.

- **RLM roadmap R4 (persistence protocol boundary) initial closure slice**:
  - Added explicit checkpoint payload protocol metadata (`Checkpoint.protocol_version`), with legacy missing-version decode support.
  - Added restore protocol normalization layer (`normalize_checkpoint_for_repl_restore`) with required/optional field contract and compatibility modes:
    - `exact`
    - `legacy_upgrade`
    - `forward_downgrade`
  - `/chat` REPL restore path now normalizes checkpoint payloads before restore and exposes protocol diagnostics in `session_persistence.restore_protocol`.
  - Added targeted coverage:
    - `tests/unit/test_session_models.py` (protocol version serialization/legacy decode),
    - `tests/unit/test_session_protocol.py` (legacy and forward compatibility normalization),
    - `tests/integration/test_chat_pipeline.py::TestChatEndpoint::test_session_restore_protocol_compat_diagnostics`.
  - Updated docs/tracking:
    - `docs/chapters/20-session-persistence.md`
    - `docs/reference/agent-config/ORCHESTRATION_DEBUG_PLAYBOOK.md`
    - `handoffs/active/rlm-orchestrator-roadmap.md`
    - `orchestration/BLOCKED_TASKS.md`

- **RLM roadmap R1 integration proof add-on**:
  - Added API integration test proving finalized `/chat` responses include `budget_diagnostics` from primitives:
    - `tests/integration/test_chat_pipeline.py::TestChatEndpoint::test_chat_response_includes_budget_diagnostics_from_primitives`
  - Validated alongside session restore compatibility tests in targeted integration run (3 passed).

- **RLM roadmap R2 integration proof add-on**:
  - Added API integration test proving `/chat` REPL responses include normalized chain wave diagnostics:
    - `tests/integration/test_chat_pipeline.py::TestChatEndpoint::test_chat_repl_response_surfaces_tool_chain_wave_diagnostics`
  - Validated in targeted integration run alongside session restore + budget diagnostics tests (4 passed).

- **RLM roadmap R3 config wiring hardening**:
  - Added explicit config field `LLMConfig.depth_role_overrides` with env loading (`ORCHESTRATOR_LLM_DEPTH_ROLE_OVERRIDES`).
  - `LLMPrimitives` depth-role override loader now reads config first, preserving env fallback compatibility.
  - Added targeted unit coverage in config + primitives tests (5 passed).

- **RLM roadmap R6 safe-default initial pass**:
  - Enabled `session_compaction` by default for `get_features(production=True)`.
  - Added rollback guidance (`ORCHESTRATOR_SESSION_COMPACTION=0`) and safe-default rationale in:
    - `docs/chapters/10-orchestration-architecture.md`
    - `docs/reference/agent-config/ORCHESTRATION_DEBUG_PLAYBOOK.md`
    - `handoffs/active/rlm-orchestrator-roadmap.md`
  - Updated feature default test coverage (`tests/unit/test_features.py`), validated with feature/concept suite (71 passed).

- **RLM roadmap R5 contention-evidence closure**:
  - Ran 5 consecutive contention-debug seeded probes (simpleqa, sample-size=1, seeds 70..74) with per-run heavy-lock holder checks.
  - Observed `LOCK_HOLDERS ... none` for all 5 runs (no stale-abandoned lock owner behavior).
  - Delegated architect paths remained bounded; one run hit a bounded infra-timeout branch (`seed=73`, `rc=124`) and still released lock cleanly.
  - Updated roadmap/progress/tracker entries to mark R5 evidence closure complete.

- **RLM roadmap R1 unit-coverage add-on**:
  - Added direct budget-clamp diagnostics tests for `LLMPrimitives.request_context` behavior (deadline and no-deadline paths) in `tests/unit/test_llm_primitives.py`.
  - Targeted validation passed (3 tests).

- **RLM roadmap R3 rollout-tuning diagnostics**:
  - Added depth-override telemetry fields to response budget diagnostics:
    - `depth_override_enabled`
    - `depth_override_events`
    - `depth_override_roles`
  - Added targeted unit coverage in `tests/unit/test_primitives_extended.py` and `tests/unit/test_llm_primitives.py`.

- **RLM roadmap R6 candidate-matrix refinement**:
  - Kept additional safe-default candidates off by default in this pass and documented explicit next-review order and evidence requirements in roadmap/playbook/architecture docs.

## 2026-02-18

- **Roadmap + agent knowledgebase refresh after orchestration stabilization**:
  - Rewrote `handoffs/active/rlm-orchestrator-roadmap.md` to reflect actual implemented state after lock-starvation and programmatic-tool-chaining work.
  - Reconciled remaining roadmap work into concrete tracks: end-to-end budget/deadline propagation (R1), trajectory visualization closure (R2), depth-aware overrides (R3), persistence protocol formalization (R4), and contention evidence closure (R5).
  - Added R6 as final roadmap task: safe-to-enable-by-default feature-set review (`src/features.py` defaults governance).
  - Updated agent workflow/playbook guidance for handoff lifecycle discipline and tool-chaining diagnostics:
    - `agents/shared/WORKFLOWS.md` (handoff closure + roadmap refresh workflow)
    - `docs/reference/agent-config/ORCHESTRATION_DEBUG_PLAYBOOK.md` (tool-chaining closure nuance: `tool_chains`, `session_persistence`, report handles)
  - Updated `orchestration/BLOCKED_TASKS.md` RLM status lines to match current implementation reality and remaining closure items.

- **R1 budget/deadline propagation (initial implementation slice)**:
  - Added shared request-budget clamping helpers and telemetry in `LLMPrimitives` (`_clamp_timeout_to_request_budget`, `_remaining_deadline_s`, `get_budget_diagnostics`).
  - Unified timeout clamping across real inference paths (`model_server`, `CachingBackend`, monitored streaming, worker-pool batch timeout wrapping).
  - Added response-level budget diagnostics field (`ChatResponse.budget_diagnostics`) and pipeline wiring so `/chat` responses include deadline/clamp metadata.
  - Lightweight validation only (no heavy inference runs): py_compile + targeted unit tests (`test_api_models_responses`, `test_chat_pipeline_stages`).

- **R2 trajectory/chain wave diagnostics (initial implementation slice)**:
  - Added canonical wave-level chain telemetry (`wave_timeline`) in structured chain execution metadata.
  - Response chain summaries now normalize/backfill wave diagnostics for compatibility (legacy `waves` count still supported).
  - ClaudeDebugger prompt rendering now prints wave-level rows (wave index, tools, mode, elapsed, fallback/parallel flags) for faster chain-loop root-cause analysis.
  - Lightweight validation only: py_compile + targeted unit tests (`test_repl_executor` chain diagnostics, `test_claude_debugger` wave rendering).

- **R3 depth-aware model override (initial implementation slice)**:
  - Added feature flag `depth_model_overrides` with env support (`ORCHESTRATOR_DEPTH_MODEL_OVERRIDES`).
  - Added depth-role override support in `LLMPrimitives` with configurable map (`ORCHESTRATOR_LLM_DEPTH_ROLE_OVERRIDES`) and safe fallback when override backend is unavailable.
  - Root depth preserves requested role; nested depth applies override mapping when feature is enabled.
  - Added/updated unit coverage in `test_primitives_extended`, `test_features`, and `test_concept_integration`.

- **Middleware Hardening Trio — all 3 gaps shipped** (from Clawzempic/OpenClaw gap analysis):
  - **Gap A — Credential Redaction**: `src/repl_environment/redaction.py` with 13 credential pattern categories; wired into ToolRegistry invoke(), ToolOutput serialization, and all 3 REPL execute paths. Feature flag `credential_redaction` (default True). 40 new tests.
  - **Gap B — Script Interception**: `src/api/routes/chat_pipeline/script_interceptor.py` with 4 built-in interceptors (timestamp, date, arithmetic, UUID); wired as Stage 0 in `_handle_chat()` before routing. Feature flag `script_interception` (default False). 29 new tests.
  - **Gap C — Cascading Tool Policy**: `src/tool_policy.py` with layered policy chain (Global→Role→Task→Delegation), group: prefixes, deny-always-wins invariant, and backward-compat `permissions_to_policy()` adapter; wired into `ToolRegistry.can_use_tool()` with `context` parameter support. Feature flag `cascading_tool_policy` (default False). 27 new tests.

- **Programmatic Tool Chaining Phase 3 (cross-request persistence) started**:
  - Added `session_id` to `ChatRequest` and wired `_execute_repl()` restore path to load latest session checkpoint globals at task start.
  - Extended checkpoint data model and storage with `user_globals`, `variable_lineage`, and `skipped_user_globals`.
  - Updated REPL checkpoint/restore flow to serialize user-defined globals (JSON-safe only), skip non-serializable values with warnings, and merge restored globals after `_build_globals()`.
  - Added SQLite additive migration for existing `checkpoints` tables (`user_globals`, `variable_lineage`, `skipped_user_globals` columns).
  - Added checkpoint globals size governance in `SessionPersister` (50MB warn, 100MB hard cap with oldest-variable eviction).
  - Expanded resume context rendering with "Variables (from previous request)" summaries.
  - Added/updated unit coverage in `test_api_models_requests.py`, `test_session_models.py`, `test_sqlite_store_extended.py`, `test_repl_state_extended.py`, `test_persister.py`, and `test_repl_executor.py`.

- **Phase 3 persistence diagnostics + roundtrip hardening**:
  - Added `ChatResponse.session_persistence` diagnostics (restore/save attempted/success, counts, ids, errors).
  - `_execute_repl()` now saves a fresh per-request session checkpoint when `session_id` is provided, enabling request-to-request global continuity without external checkpoint triggers.
  - Fixed restore compatibility for stored checkpoints missing explicit `version` by defaulting to v1.
  - Added REPL-executor cross-request roundtrip unit coverage (`request1` saves globals, `request2` restores).
  - Unblocked and enabled the HTTP `/chat` roundtrip integration test (`test_session_restore_roundtrip_repl_globals`) using async transport coverage.
  - Fixed a latent route-level contention issue in `_handle_chat`: delegated execution is now dispatched via `asyncio.to_thread(...)` only when mode is actually `delegated` (previously invoked unconditionally).
  - Updated touched persistence paths to timezone-aware UTC timestamps (`datetime.now(timezone.utc)`) in `repl_executor` and `SessionPersister`.

- **Phase 3 persistence tunables**:
  - Added centralized config/env knobs for checkpointed globals payload limits:
    - `ORCHESTRATOR_SESSION_PERSISTENCE_CHECKPOINT_GLOBALS_WARN_MB` (default `50`)
    - `ORCHESTRATOR_SESSION_PERSISTENCE_CHECKPOINT_GLOBALS_HARD_MB` (default `100`)
  - `SessionPersister` now reads these limits from `get_config().session_persistence` instead of hardcoded constants.

- **ClaudeDebugger wave-level chain rendering**: debugger prompts now include `tool_chains` execution metadata (chain ID, mode requested/used, wave count, fallback-to-seq, parallel-mutation flag) so delegation/chaining loops can be debugged at wave granularity. Added unit coverage in `tests/unit/test_claude_debugger.py`.

- **Programmatic Tool Chaining Phase 2 finalized**:
  - **Phase 2a complete**: structured-mode AST call detection, `allowed_callers` gating, invocation chain metadata (`caller_type`, `chain_id`, `chain_index`), and `tool_chains` API response summaries.
  - **Policy wiring complete**: `allowed_callers` explicitly populated across all 47 tools in `orchestration/tool_registry.yaml`.
  - **Phase 2b complete**: dependency-wave execution mode (`ORCHESTRATOR_TOOL_CHAIN_MODE=dep`) with safe sequential fallback and optional parallel mutation waves (`ORCHESTRATOR_TOOL_CHAIN_PARALLEL_MUTATIONS=1`) for conservative safe tools.
  - **Diagnostics complete**: `tool_chains` now includes execution-mode metadata (`mode_requested`, `mode_used`, `fallback_to_seq`, `parallel_mutations_enabled`, `waves`, `steps`).
  - **Evidence**:
    - targeted unit suite: 74 passing tests across chaining/dependency/allowed_callers/audit/response paths.
    - synthetic dep-mode contention check: `run_shell` pair latency improved from ~0.417s (parallel mutations OFF) to ~0.210s (ON), ~1.99x speedup.

- **Inference lock starvation root-cause closure**: eliminated cross-request cancellation/deadline context bleed, added request-scoped lock attribution (`request=<task_id>`), moved timeout clamping to post-lock-acquire paths, and added lock acquire/release trace telemetry (`ORCHESTRATOR_INFERENCE_LOCK_TRACE`). Contention sweeps now clear holders by +30s with no stale-abandoned lock behavior.
- **Delegation loop hardening + diagnostics**: introduced explicit delegation break reasons (including pre-delegation lock timeout), specialist-timeout/report exits, and richer delegation diagnostics in API + seeding debugger flows to reduce blind reruns.
- **Artifact-backed delegation report hydration**: added persisted report handles (`src/delegation_reports.py`), REPL/API lazy retrieval (`fetch_report(...)`, `GET /chat/delegation-report/{id}`), and prompt wiring so downstream turns can hydrate full specialist output on demand.
- **Worker runtime alignment**: restored `worker_coder`-first semantics end-to-end (`worker_code` kept as compatibility alias), aligned config/port defaults to fast worker pool (`8102`), and updated orchestration docs/prompts accordingly.
- **Stack operability improvements**: added `orchestrator_stack.py --profile contention-debug` and fixed mixed state serialization in `save_state()` to prevent startup/reload crashes during repeated debugging cycles.
- **Validation additions**: new unit coverage for inference lock + delegation reports and new integration roundtrip test for delegated handle emission/retrieval (`test_delegation_report_handle_roundtrip`; collected successfully; execution intermittently hangs in this environment).

## 2026-02-17

- **Perf: Parallel read-only tool dispatch (WS1)** — Multi-tool REPL turns now dispatch independent read-only tools via `ThreadPoolExecutor` instead of sequential `exec()`. AST-based extraction with conservative fallback (any dependency → sequential). Feature flag `parallel_tools=True`. Expected 2-4x speedup on multi-tool turns. New: `src/repl_environment/parallel_dispatch.py`, 22 unit tests.

- **Perf: Concurrent inference sweep script (WS2)** — New `scripts/benchmark/concurrent_inference_sweep.py`: asyncio + httpx benchmark for optimal `-np`/concurrency per model tier. Tests frontdoor/coder/worker/architect/fast_worker at varying concurrency levels. Incremental CSV output, TTFT streaming baseline, dry-run mode.

- **Perf: Wire id_slot for prefix cache routing (WS3A)** — Fixed dead code in `PrefixRouter`: computed optimal slots but never passed `id_slot` to llama-server. Added `slot_id` field to `InferenceRequest`, wired through `_build_payload()` and `CachingBackend.infer()`.

- **Perf: Escalation prompt compression (WS3B)** — Feature-flagged `escalation_compression=False`. When escalating with >16K char prompt, LLMLingua-2 BERT compresses to 50% preserving code tokens. ~1.67s saved per architect escalation at 1.2 t/s prefill.

- **Perf: Speculative architect pre-warming (WS3C)** — New `src/services/escalation_prewarmer.py`: at turn 1, if task classified COMPLEX, fires non-blocking `n_predict=0, cache_prompt=true` to architect server. Checks `/slots` first. ~417ms saved per architect escalation.

- **Fix: claude-mem 637 zombie workers (231 GB RAM leak)** — claude-mem v9.0.12 had no subprocess concurrency limit, allowing 637 Claude SDK agent workers to accumulate (228 GB RSS + 2.6 GB swap). Updated plugin from v9.0.12 → v10.2.3 (`git pull` on `~/.claude/plugins/marketplaces/thedotmack/`). New version enforces `CLAUDE_MEM_MAX_CONCURRENT_AGENTS=2` via `waitForSlot()` in ProcessRegistry. Killed workers, restarted daemon. RAM: 818→577 GB, swap: 7.5→0 GB.

- **Fix SimpleQA 0% pass rate (4-layer fix):**
  - **web_search**: Retry with 1s delay (2 attempts), DDG snippet extraction (`result__snippet`), Wikipedia opensearch fallback, typed error categories (timeout/rate_limit/network/parse_error).
  - **fetch_wikipedia**: Switched from REST summary API with bad `sentences*200` heuristic to query API with `exsentences` param for accurate sentence-level truncation.
  - **Scoring**: SimpleQA `exact_match` → `f1` (threshold=0.8). SQuAD-style normalization handles word reordering and prose wrapping. The `_score_f1` function already existed in `debug_scorer.py`.
  - **Architect prompt**: Removed blanket "NEVER delegate factual questions" rule. Now: answer directly when confident, delegate to `worker_explore` (which has web_search) when uncertain about obscure facts.
  - **Debugger agency**: Added Tool & Scorer Investigation section, Suite-Level Analysis section, action-biased editing rule ("wrong fixes are cheap to revert"), hot-swap reload guidance, suite-level failure alerts in prompt builder.

- **Nightshift first production run — 7 PRs reviewed and merged:**
  - **Dead code**: -4,038 lines (8 orphaned scripts + unused `ExecutorError` class)
  - **Test gap**: +189 unit tests for unicode sanitizer, LLM types, tokens mixin, API models
  - **Security**: Patched eval() RCE vector (stripped `__builtins__`), command injection in `git_status()` (`shlex.quote`), hardened `run_shell()` blocklist (control chars, `curl|sh`, `$()`, backticks). Removed `getattr`/`hasattr` from eval allowlist during review (class hierarchy escape).
  - **Perf**: N+1→batch queries in sqlite_store, pre-compiled symptom regex, O(n²)→O(n) SSE string concat, reusable `ThreadPoolExecutor` in parallel_embedder
  - **Doc drift detector**: New `validate_doc_drift.py` — cross-references CLAUDE.md ports/paths/make-targets against code. 15 tests. Integrated into `make check-agent-config`.
  - **Docs backfill**: Created missing `docs/SETUP.md` and `docs/MODEL_MANIFEST.md`, added module docstrings
  - **Skill groom**: Fixed stale refs to deleted files in governance/skills, corrected `research-update.md` and `benchmark.md` paths

- **Nightshift branch hygiene issue identified**: All PRs after first two had branch stacking (branching off previous task branches instead of `origin/main`), requiring cherry-picks for all affected PRs. 12 branches + worktree cleaned up.

## 2026-02-16

- **Fix: orchestrator_stack.py --hot-only preserves healthy servers** — `start --hot-only` was unconditionally killing all server processes before restarting. Now checks `/health` endpoint (3s timeout) before killing; healthy servers are preserved and skipped during launch.

- **REPL output spill-to-file with rolling summary** — Large REPL output (>8192 chars) no longer hard-truncated. Output spills to `{spill_dir}/{session_id}/turn_N.txt`; worker model generates a rolling summary that accumulates across turns (previous summary + new tail → updated summary). Model can `peek()`/`grep()` spill files on demand. `max_output_preview` bumped 500→1500. Static head/tail fallback when worker unavailable. 12 new tests.

- **Distillation pipeline latency instrumentation** — Fixed inter-model transition latency blindness in seeding pipeline:
  - **httpx client reuse** (`teachers.py`): `LocalLlamaTeacher` now creates a single `httpx.AsyncClient` lazily and reuses across batches (was creating new client per `distill()` call — TCP setup overhead on every batch). Supports async context manager for clean shutdown.
  - **Per-batch timing** (`pipeline.py`): Each teacher `distill()` call timed with `time.monotonic()`. New `batch_latencies` field on `DistillationReport` records `{skill_type, batch_index, batch_size, elapsed_ms, teacher}` per batch. Logged at INFO level.
  - **New anomaly signal `distill_batch_latency`** (`anomaly.py`): Fires when any distillation batch exceeds 5s threshold (vs `slow_delegation`'s 120s). Weight 0.5. Wired into `compute_anomaly_signals()` via backward-compatible `batch_latencies` kwarg.

- **Overnight run regression fixes** — 4 bugs from `seed_specialist_routing.py --evolve --debug-replay --continuous` run:
  - **Escalation loop guard** (`chat_delegation.py`): Replaced shallow brief dedup (first 200 chars) with 4-layer defense: semantic dedup (hash brief+target), thread-local re-entrance depth counter, role repetition guard (max 2 consecutive same target), cumulative token budget (20K cap). New constants in `src/constants.py`.
  - **Corpus injection in delegation** (`chat_delegation.py`): `_run_specialist_loop()` now calls `build_corpus_context()` on turn 0 and passes result to `build_root_lm_prompt()`. Previously delegated 32B/480B specialists never received corpus snippets — the +8.7pp A/B result was from direct REPL only.
  - **`--evolve`/`--debug-replay` silent no-op** (`seed_specialist_routing.py`): Removed `ORCHESTRATOR_SKILLBANK=1` env var gate on OutcomeTracker init. Added `_run_post_batch_hooks()` running evolve+replay every 10 batches in continuous mode (previously only ran at Ctrl+C exit, never on SIGTERM/kill).
  - **`no_skills_available` false positive** (`diagnostic.py`): Changed `skills_retrieved` default from `0` to `None` so "SkillBank not loaded" is distinguishable from "loaded but returned 0 results". Fixed truthiness check for output dict.
  - 13 new unit tests for loop guards. 3833 tests pass.

## 2026-02-15

- **Nightshift automated overnight maintenance** — Full integration of [nightshift](https://github.com/marcus/nightshift) for autonomous code maintenance via Claude Code CLI:
  - Architecture: systemd timer (02:30) → `run_wrapper.sh` → inference guard → nightshift → PATH shadow claude → devcontainer (bypassPermissions) → dedicated worktree.
  - 11 aggressive tasks: lint-fix, bug-finder, auto-dry, td-review, docs-backfill, skill-groom, dead-code, test-gap, security-footgun, perf-regression, doc-drift.
  - Inference guard: checks llama-server RSS via `/proc/*/status`, restricts to analysis-only tasks when >200GB RAM detected.
  - Dedicated worktree at `/mnt/raid0/llm/claude-nightshift` prevents branch switching from disrupting parallel agents on main.
  - Devcontainer routing solves permission issues: PATH shadow binary (`scripts/nightshift/bin/claude`) routes through `docker exec` into container with `bypassPermissions`.
  - Budget: 90% daily cap, 5% morning reserve.
  - New files: `nightshift.yaml`, `scripts/nightshift/{inference_guard,run_wrapper,claude_via_devc,claude-nightshift}.sh`, `scripts/nightshift/bin/claude`.

- **Phase 2A: A/B tested corpus-augmented prompt stuffing across all 5 models**:
  - **480B best result**: +15.6pp acceptance (74.9→90.5%), +17% speed (8.3→9.7 t/s), wall time decreased.
  - **32B solid result**: +8.7pp acceptance (84.6→93.3%), +6% speed (30.8→32.7 t/s).
  - **30B negative**: acceptance +2.1pp but speed -12% (overhead > gain). Corpus disabled.
  - **235B mixed**: +6.6pp on HTTP task, -12.1pp on BST task. Corpus disabled.
  - **7B saturated**: already 94-100% baseline, +5.3pp marginal. Corpus disabled.
  - Decision: enable corpus for Coder-family models (32B, 480B) only.

- **Telemetry key fix** (`src/backends/llama_server.py`): Wrong keys for spec decode stats (`drafted_n_tokens` → `draft_n`, `drafted_n_accepted` → `draft_n_accepted`). Both sync and streaming paths fixed.

- **Token normalization fix**: N-grams in index included punctuation but query n-grams didn't — 0 matches. Added `_normalize_token()` (strips non-alnum except underscore) in build_index.py, build_index_v2.py, corpus_retrieval.py, and test fixtures.

- **Corpus scaling v2** (`scripts/corpus/build_index_v2.py`): SQLite-backed index builder for The Stack v1 (v2 is metadata-only). HuggingFace streaming, 6 languages (Python/JS/TS/Rust/Go/C++), `--resume` for interrupted builds. Python build running: 67GB+ DB, ~12M+ snippets.

- **SQLite retriever** (`src/services/corpus_retrieval.py`): Auto-detects v1 (JSON) vs v2 (SQLite) index. SQLite uses mmap (~200KB RAM per query regardless of DB size).

- **Pruning tool** (`scripts/corpus/prune_index.py`): Optional post-build pruning by snippet count or target GB. Proportional per-language quotas, batch deletion for large sets, VACUUM.

- **Qwen3-TTS Phase 4: C++ native pipeline** (`llama.cpp-experimental`, branch `feature/qwen3-tts-support`):
  - Built `llama-tts-qwen3` binary: Talker GGUF + Code Predictor GGUF + sidecar weights → codec tokens at 1.5x RT.
  - Sidecar format v2 (QWTTS02): added `cp_vocab` field, fixed header size mismatch (32B vs 36B).
  - Multi-head Code Predictor: enabled `llama_set_embeddings()` + `llama_get_embeddings_ith()` to extract hidden states, apply correct per-step lm_head from sidecar.
  - Talker hidden state extraction for CP `past_hidden` input.
  - End-to-end pipeline: C++ → Tokenizer Decoder → WAV (24kHz). Pipeline works but **audio is unintelligible noise**.
  - Whisper round-trip test confirms garbled output. **BLOCKED** pending PyTorch reference token comparison.
  - New files: `scripts/voice/create_tts_sidecar.py`, `scripts/voice/validate_tts_e2e.py`.

- **Phase 2A: Corpus-augmented prompt stuffing implemented** (off by default):
  - New `scripts/corpus/build_index.py`: word-level 4-gram index from src/ + stdlib + numpy + torch (73K snippets, 5.5M n-grams, 14s build).
  - New `src/services/corpus_retrieval.py`: `CorpusRetriever` singleton — lazy index load, sub-ms query, graceful degradation.
  - Wired `corpus_context` into all 3 prompt paths: `chat.py`, `stream_adapter.py`, `nodes.py` (turn 0 only).
  - Added `reference_code` field to `RootLMPrompt` (renders as `## Reference Code` before `## Task`).
  - Added `corpus_retrieval: bool` to `AccelerationConfig` + `runtime_defaults` in registry YAML.
  - Added acceptance rate telemetry (`n_tokens_drafted`, `n_tokens_accepted`) to `InferenceResult` + extraction from llama-server timings.
  - 27 new tests (20 corpus retrieval + 7 prompt builder). All passing.
  - **Gate**: Feature stays off until A/B quality benchmark passes (max -0.5 score regression).

## 2026-02-14

- **Agent governance refactor (harness-aligned) completed**:
  - Finalized layered agent prompt architecture in `agents/` (thin execution contract, shared policy, lean role overlays).
  - Added operational depth docs in `docs/guides/agent-workflows/` to keep prompts concise.
  - Added CLAUDE coverage governance artifacts:
    - `docs/reference/agent-config/CLAUDE_MD_MATRIX.md`
    - `docs/reference/agent-config/claude_md_matrix.json`
    - Explicit governed scope for `CLAUDE.md` and `kernel-dev/llama-cpp-dev/CLAUDE.md`.
  - Added broad hook suite in `scripts/hooks/` and wired into `.claude/settings.json`:
    - `agents_schema_guard.sh`
    - `agents_reference_guard.sh`
    - `claude_accounting_context.sh`
    - `skills_context.sh`
  - Added dual skill surfaces:
    - Command skills: `.claude/commands/agent-files.md`, `.claude/commands/agent-governance.md`
    - Packaged local skills: `.claude/skills/agent-file-architecture/`, `.claude/skills/claude-md-accounting/`
  - Added lightweight validators and make target:
    - `scripts/validate/validate_agents_structure.py`
    - `scripts/validate/validate_agents_references.py`
    - `scripts/validate/validate_claude_md_matrix.py`
    - `make check-agent-config` (all checks passing)
  - Added explicit design logic doc: `docs/reference/agent-config/AGENT_FILE_LOGIC.md`
  - Folded skills-shell guidance into skill boundaries (`use when` / `do not use when`) for packaged local skills.

- **SkillBank End-to-End Integration**: Wired SkillBank infrastructure (122 tests, 10 files) into `seed_specialist_routing.py` and `ClaudeDebugger`. Five gaps closed:
  - **CLI bootstrap** (`scripts/skillbank/seed_skills.py`): Populates SkillBank from episodic memory or progress logs via `--teacher claude|codex|mock`.
  - **Debugger integration**: +2 anomaly signals (`skill_mismatch`, `no_skills_available`), skill retrieval data in diagnostics, `SkillAwareReplayEngine` in replay summary, skill health via `EvolutionMonitor`.
  - **API data flow**: `skill_ids` + `skills_retrieved` propagated through `RoutingResult → ChatResponse → RoleResult → diagnostic`. All 8 `ChatResponse` construction sites updated.
  - **Replay integration**: `--debug-replay` tries skill-aware replay first, prints skill metrics (coverage, avg/step).
  - **OutcomeTracker**: Records skill×task outcomes for evolution. Enabled via `ORCHESTRATOR_SKILLBANK=1`.
  - **Evolution trigger**: `--evolve` flag runs `EvolutionMonitor.run_evolution_cycle()` after seeding, prints promotion/decay/deprecation report.
  - **Tests**: 17 new tests (`test_skill_diagnostics.py`), 3525 total unit tests passing, 0 failures.
  - **Teacher fixes**: ClaudeTeacher rewrote from Anthropic SDK to `claude -p` CLI subprocess (no API key); CodexTeacher fixed CLI flags (`--full-auto`) and JSONL parser (`item.completed`/`agent_message`); both strip `CLAUDECODE` env var for nested invocation.
  - **Full seeding run**: 200 trajectories × 2 teachers → 138 skills stored (64 Claude + 58 Codex + 16 test). Zero merges, zero rejections. ~11 min total.

## 2026-02-13

- **Replay Evaluation Harness (MemRL meta-learning)**: Full 8-phase implementation of offline replay harness for meta-learned memory configurations. Motivated by ALMA (Xiong et al., 2026). 7 new modules (1,885 LOC production + 1,250 LOC tests = 3,135 LOC total):
  - **Trajectory extraction** (`replay/trajectory.py`): Reads progress logs, groups by task_id, builds complete Trajectory objects. Stratified sampling (default 1000), embedding pre-computation with cache.
  - **Replay engine** (`replay/engine.py`): Creates isolated EpisodicStore per candidate, replays chronologically, collects per-step routing accuracy and reward. NullEmbedder safety guard prevents live embedding calls.
  - **Metrics** (`replay/metrics.py`): Aggregate metrics — routing accuracy (overall + per-type), escalation precision/recall, Q-convergence step, cumulative/avg reward, cost efficiency.
  - **Design candidates + archive** (`replay/candidates.py`): DesignCandidate bundles (RetrievalConfig, ScoringConfig, StagedConfig) with lineage tracking. DesignArchive (SQLite) stores results, supports top-k queries, lineage traversal, diverse sampling for reflection.
  - **Warm-start protocol** (`replay/warm_start.py`): Detects model swap (majority model_id mismatch), resets Q-values to 0.5, doubles learning rate for 50-task warmup. RoleConfig for per-role memory schemas.
  - **model_id field**: Added `model_id TEXT` column to MemoryEntry + ALTER TABLE in episodic_store.py. Enables retrieval affinity (+15% same-model bonus) and model swap detection.
  - **Meta-agent workflow** (`replay/meta_agent.py`): Claude-as-meta-agent — builds reflection prompt, parses candidate proposals, runs replay evaluation, generates comparison report. Human-in-the-loop promotion (no auto-promote). Dual CLI + library interface.
  - **Prompt template** (`orchestration/prompts/meta_agent_reflect.md`): Structured prompt for Claude to propose memory config mutations.
  - **Baseline replay**: 1000 trajectories replayed in 0.18s. Routing accuracy 0% (expected: historical logs use mock routing). Cumulative reward 997.0 (nearly all success).
  - **Tests**: 75 new tests across 5 files, all passing. Full suite: 3386 passed, 0 failures.
  - **Shellcheck fix**: Fixed pre-existing SC2294 warning in `scripts/benchmark/deprecated/run_phase3_validation.sh` (`eval "$@"` → `"$@"`).

- **Speculative Decoding VERIFIED across all MoE models (Phases 0+0.5+1 + 235B)**:
  - **Phase 0 (480B prompt lookup)**: Works mechanically (18.4% acceptance), but net-negative on speed (-34%). Registry `forbid` was wrong (MoE ≠ SSM).
  - **Phase 0.5 (480B jukofyork draft)**: vocab transplant draft with matching BOS (comma token 11). 74-82% acceptance, full+spec = 9.00 t/s.
  - **Phase 1 (30B full matrix)**: **Best: MoE6 + spec + lookup = 47.11 t/s (2.58x over baseline)**. Lookup net-positive on 30B.
  - **235B spec decode (NEW)**: 0.6B Q8_0 draft, 53-55% acceptance. MoE4+spec = 8.21 t/s (fastest), full+spec = 6.08 t/s (production: quality). Previously untested — 0.6B draft dramatically outperforms 1.7B (55% vs 21% acceptance).
  - **Architect policy change**: Architect roles (235B, 480B) now use full experts + spec (no MoE reduction). Quality over speed for the hardest tasks.
  - **Per-role lookup flag**: `AccelerationConfig.lookup` field. 30B/coder_escalation: lookup=True; architects: lookup=False.
  - **Production shipped**: All models updated across `model_registry.yaml`, `orchestrator_stack.py`, `CLAUDE.md`, `RESULTS.md`.
  - **Tests**: 125/125 registry-related tests pass.

- **SoftMatcha v2 research + Corpus-Augmented Speculative Decoding plan**: Reviewed SoftMatcha v2 (arxiv 2602.10908) — fast fuzzy pattern matcher for trillion-scale corpora (Python+Rust, Apache 2.0). Identified corpus-augmented prompt lookup opportunity for models where spec decode isn't available. Expanded `handoffs/active/hybrid-lookup-spec-decode.md` (PROPOSAL→ACTIVE). Phase 2 (SoftMatcha corpus augmentation) remains pending — may help models without compatible draft models.

- **Orchestrator wiring: 4 scaffolded improvements connected to live pipeline**:
  - **#2 Think-Harder**: `_should_think_harder()` helper in `nodes.py` triggers on penultimate retry (before model escalation). All 7 graph node error paths updated to try same-model CoT (4096 tokens, "Think step by step" prefix) before escalating. Success/failure tracked in TaskState.
  - **#7 GBNF Grammar Enforcement**: `detect_tool_requirement()` wired into `_route_request()`. On first REPL turn when `tool_required=True`, `generate_gbnf_grammar()` constrains model output to valid tool call syntax via `llm_call(grammar=...)`.
  - **#9 Diagnostic Fields**: 8 fields populated end-to-end: ChatResponse → seeding_eval `_build_role_result()`. Fields: `cheap_first_attempted/passed`, `think_harder_attempted/succeeded`, `grammar_enforced`, `parallel_tools_used`, `cache_affinity_bonus`, `cost_dimensions`.
  - **#6 Streaming Tool Events**: `tool_start_event`/`tool_end_event` SSE events emitted after each `repl.execute()` in both `chat.py` legacy streaming and `stream_adapter.py` unified path. Invocation log delta tracking avoids double-emission.
  - **Test fixes**: `test_returns_all_20_signals` → `test_returns_all_22_signals` (stale count after signal additions). `test_contains_escalation_edges` assertion corrected: `FrontdoorNode → CoderNode` (not CoderEscalationNode).
  - **Files modified**: `nodes.py`, `state.py`, `responses.py`, `chat.py`, `routing.py`, `repl_executor.py`, `stream_adapter.py`, `seeding_eval.py`

- **NextPLAID Phase 5: LateOn-Code 130M + AST chunking + ColGrep**:
  - **Model upgrade**: LateOn-Code-edge (17M, 48-dim) → LateOn-Code (130M, 128-dim). +11.2% on MTEB Code benchmark (74.12 vs 66.64). Memory cost: 0.2GB → 1.2GB (trivial on 1.13TB machine).
  - **AST chunking**: `scripts/nextplaid/ast_chunker.py` — tree-sitter Python parser extracts semantic code units (functions, classes, methods with signatures + docstring detection) instead of naive 1800-char splits. `FallbackChunker` for non-Python files. Both `index_codebase.py` and `reindex_changed.py` updated.
  - **ColGrep CLI**: Installed colgrep 1.0.6 (LightOn agent-facing hybrid search). Storage paths configured on RAID.
  - **Search results enriched**: `code_search()` now returns `unit` (e.g. `class:EscalationPolicy`) and `signature` fields when AST metadata available.
  - **Files modified**: `orchestrator_stack.py`, `model_registry.yaml`, `index_codebase.py`, `reindex_changed.py`, `code_search.py`, `test_code_search.py`
  - **Files created**: `scripts/nextplaid/ast_chunker.py`, `handoffs/active/nextplaid-phase5-upgrade.md`
  - **Tests**: 20/20 code_search tests pass (18 existing + 2 new AST metadata tests)

- **Orchestrator Intelligence Improvements (Claude-Inspired)**: 7 improvements to the orchestration intelligence layer — routing, escalation, cost modeling, quality gating. Inspired by Anthropic's Claude architecture patterns. See `handoffs/active/orchestrator-intelligence-improvements.md` for full design.
  - **#8 Prefix Cache Expansion**: `prefix_length` 256→4096 in `model_registry.yaml`. Role prompts (1000-5000 tokens) now fully cacheable. All 9 role prompts audited for prefix stability (static first, variable last). Parallels Claude's prompt caching prefix stability.
  - **#3 Grammar-Constrained Structured Outputs**: `json_schema` and `grammar` (GBNF) fields added to `InferenceRequest` (`protocol.py`), threaded through `llama_server.py`, `primitives.py`, `inference.py`. Enables constrained generation without post-hoc formalization.
  - **#4 Cache Affinity Bonus**: Phase 2.5 in `TwoPhaseRetriever` gives 15% score bonus to memories matching last-used role. `_last_role_used` tracked by `HybridRouter.route()`. Improves KV cache hit rates. Parallels Claude's prompt caching TTL economics.
  - **#1 Multi-Dimensional Cost Model**: QScorer extended with 3 cost dimensions: latency (existing), quality gap penalty (`cost_lambda_quality_gap=0.10`, penalizes over-qualified model), memory tier penalty (`cost_lambda_memory=0.05`, penalizes WARM when HOT suffices). Per-role baselines from benchmarks.
  - **#7 Reliable Tool Use**: `generate_gbnf_grammar()` on ToolRegistry creates GBNF from registered tools. `get_read_only_tools()` identifies safe parallel tools. `_execute_structured()` relaxed for parallel read-only tool execution. `detect_tool_requirement()` in routing detects tool-needing tasks.
  - **#2 Think-Harder Escalation**: New `THINK_HARDER` action in `EscalationAction` enum. Fires on penultimate retry with `config_override: {n_tokens: 4096, cot_prefix: "Think step by step...", temperature: 0.5}`. Tries same model harder before expensive model swap. Parallels Claude's extended thinking.
  - **#5 Try-Cheap-First**: Speculative pre-filter in `chat.py`. 7B worker attempts answer, quality-gated. Phase A=all requests, B=MemRL Q-value gated, C=fully learned. Existing escalation chain untouched. Parallels Claude's Haiku→Sonnet→Opus routing.
  - **#6 Streaming Tool Use**: `llm_call_stream()` method for token-level streaming. `tool_start_event()` / `tool_end_event()` SSE types added.
  - **Debugger integration**: Diagnostic records extended with `cost_dimensions`, `think_harder_attempted/succeeded`, `cheap_first_attempted/passed`, `grammar_enforced`, `parallel_tools_used`, `cache_affinity_bonus`. ClaudeDebugger prompt builder surfaces these.
  - **Test infrastructure**: `pytest-timeout` installed, default timeout 120→30s per test. 3746 passed, 67 skipped, 42s with `-n 8`.

## 2026-02-12

- **`repl_no_tools` signal fix — direct-mode routing**: New `_should_use_direct()` heuristic in `chat_routing.py` short-circuits obvious simple questions (MCQ with 3+ choices <2000 chars, short factual <300 chars with question-word prefix) to direct mode before MemRL/REPL. Prevents false `repl_no_tools` signals on questions that don't need tools. Conservative — coding tasks, long context, research indicators always fall through to REPL.
- **`repl_no_tools` signal fix — max-turns answer rescue**: New `_rescue_from_last_output()` in `nodes.py` extracts answers (FINAL pattern → prose answer → code block) from `state.last_output` when max turns hit without FINAL(). Applied to all 7 graph node classes and as post-graph fallback in `repl_executor.py`. Recovers correct answers that models computed but failed to submit via FINAL().
- **`repl_no_tools` signal fix — graduated turn nudge**: Midpoint soft reminder at `remaining == max_turns // 2` ("Start converging on your answer") complements existing hard deadline at `remaining <= 3`. Reduces last-minute panic responses by giving models earlier awareness of turn budget.

## 2026-02-11

- **Slot erase timeout fix**: SELF:direct 600s timeout left server-side generation running (23k+ tokens), blocking all subsequent strategies. `_erase_slots` HTTP timeouts raised 3s→8s; new `_force_erase_and_verify()` resets capability cache + retries with verification; proactive slot erasure in polling loop at `timeout-15s`; inter-strategy cleanup between SELF:direct→SELF:repl. `_erase_slots(all_slots=True)` flushes stale KV cache between eval questions. All fixes applied to both monolithic file and v2 extracted modules.
- **Monolithic seed_specialist_routing.py retired**: Renamed to `deprecated/seed_specialist_routing_v1.py`. Former `seed_specialist_routing_v2.py` promoted to `seed_specialist_routing.py` (canonical entry point). All logic lives in extracted `seeding_*.py` modules; the hub file only re-exports and provides CLI.
- **NextPLAID multi-vector code & doc retrieval**: Deployed NextPLAID (Rust, Apache 2.0) on :8088 with LateOn-Code-edge ColBERT model (48-dim, ONNX INT8). Indexed 460 source files (4,599 chunks) + 140 doc files (1,345 chunks). New REPL tools `code_search()` and `doc_search()` provide token-level code retrieval — complementary to episodic memory `recall()`. 12/12 unit tests, 153/153 REPL regression tests pass. ~40ms p95 query latency, <200MB RAM overhead. See `handoffs/active/nextplaid-code-retrieval.md`.
- **NextPLAID Phase 4: Dedicated doc model**: Second container `nextplaid-docs` (:8089) with `answerai-colbert-small-v1-onnx` (96-dim, text-optimized). Code container (:8088) unchanged. `code_search.py` routes by index: code→:8088, docs→:8089 with fallback. Isolated volume mounts prevent embedding cross-contamination. `orchestrator_stack.py` manages both Docker containers (start/stop/status/reload). 18/18 tests pass.
- **Debugger infra health + service reload**: Claude Debugger can now detect degraded infrastructure and reload services. `check_infra_health()` probes orchestrator/:8000, nextplaid-code/:8088, nextplaid-docs/:8089. Each diagnostic batch prompt includes `INFRA DEGRADED: ...` or `all services healthy`. Claude can output `RELOAD_SERVICE: <name> reason=...` to restart services via `orchestrator_stack.py reload`. `_hot_restart_api()` refactored to use general `_reload_service()`. System prompt updated with Reloadable Services section. 42/42 debugger tests pass (11 new).
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
- **REPL prompt rewrite — few-shot examples replace instruction stacking**: 40-hour seeding analysis (1,673 records) revealed REPL mode 10 points behind direct mode (46.8% vs 56.7%), with 17% of REPL runs exhausting max turns without calling FINAL(). Root cause: models learn protocols from examples, not instruction lists. `rules.md` rewritten from 51 lines of rules to 8 concrete input/output examples covering factual, MCQ, math, web search, competitive programming, explanation, document reading, and architect consultation. `root_lm_system.md` and `builder.py` simplified to point at examples instead of repeating rules.
- **Wasteful delegation guard + signal**: Architect solves answer in `<think>`, delegates to coder anyway, coder round-trips unchanged. New runtime guard in `chat_delegation.py` intercepts short-answer delegations for non-code questions. New `wasteful_delegation` anomaly signal (weight 0.5).
- **REPL max-turns signal**: 76 records with `[Max turns (N) reached without FINAL()]` were invisible to all 17 detectors (score 0.0). New `repl_max_turns` signal (weight 1.0). Signal count 17→19.
- **Status-phrase set expanded**: `"code"`, `"explanation of code or reasoning"`, `"code execution complete. check output"`, `"your_computed_value"` added to both `anomaly.py` and `nodes.py` rejection sets. 10 records had these as final answers with zero anomaly signals.
- **Late-game FINAL() nudge**: When ≤3 REPL turns remain, DEADLINE message injected into prompt forcing immediate FINAL() submission. Targets the 69 max-turns failures.
- **Template echo prevention**: `FINAL(answer)` → `FINAL(value)` in tools.md, `FINAL(your_computed_value)` in rules.md. 5 records had literal `"answer"` as their final answer from echoing the prompt template.
- **Debugger system prompt hot-swap**: Extracted `DEBUGGER_SYSTEM_PROMPT` from `claude_debugger.py` into `orchestration/prompts/debugger_system.md`. Now resolved via `resolve_prompt()` — editable at runtime without restarting seeding script. Short `_DEBUGGER_SYSTEM_FALLBACK` constant kept as fallback.

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

## 2026-02-20

- **Refactor handoff completion (Waves 1-2) + final integration**:
  - Completed Wave 1 stabilization/extraction chain and Wave 2 config package split.
  - Added `src/config/models.py` and `src/config/validation.py`; `src/config/__init__.py` now serves as a compatibility facade preserving public imports.
  - Stabilized async chat endpoint tests by moving to direct async route invocation and threadpool shim where needed.
  - Added REPL environment protocol contract types for mixin/documentation clarity.
  - Final gate run (`make gates`) completed successfully at handoff end.

- **Integrated parallel debugging stream (seed_specialist_routing / orchestrator-adjacent wiring)**:
  - Included admission/timeout/lock-path hardening and prompt/tool-call translation updates from the concurrent debugging work.
  - Updated benchmark/config surfaces and progress notes to reflect backend saturation and delegation timeout diagnostics.

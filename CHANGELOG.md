# Changelog

## 2026-02-16

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

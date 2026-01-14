# Next Orchestration Tasks

**Created:** 2026-01-14
**Purpose:** Prioritized list of next implementation tasks for the orchestrator

---

## Recommended Priority Order

### Priority 1: Role-Based Generation Defaults ✅ COMPLETE

**Source:** Previous agent's plan (`glowing-splashing-eclipse.md`)
**Completed:** 2026-01-14

**What was done:**
- Added `generation_defaults` to 5 additional roles (ingest, thinking, coder_escalation, worker_summarize, toolrunner)
- Registry loading at API startup for mock mode
- Most infrastructure was already implemented by a previous agent

**Files modified:**
- `orchestration/model_registry.yaml` - Added generation_defaults to roles
- `src/api.py` - Registry loading at startup

---

### Priority 2: RLM Phase 1 - Backend Completion ✅ COMPLETE

**Source:** `orchestration/BLOCKED_TASKS.md`
**Verified:** 2026-01-14

**Status:** All infrastructure is complete:
- [x] LlamaServerBackend HTTP - Full implementation with streaming
- [x] CachingBackend init - Auto-wired in LLMPrimitives
- [x] Role→backend routing - Works via server_urls parameter
- [x] Real mode initialization - Creates CachingBackend automatically

**To test real inference:** Start llama-server, then call API with `real_mode=True`.

---

### Priority 3: MemRL Phase 4 - Escalation Learning ✅ COMPLETE

**Source:** `orchestration/BLOCKED_TASKS.md`
**Completed:** 2026-01-14

**What was done:**
- [x] Added `LearnedEscalationPolicy` class to query episodic memory
- [x] Added `LearnedEscalationResult` dataclass
- [x] Updated `FailureRouter` with `retriever` and `progress_logger` parameters
- [x] Hybrid routing: queries learned policy first, falls back to rules
- [x] Escalation decisions logged via `progress_logger.log_escalation()`
- [x] Strategy counts tracked for monitoring

**Files modified:**
- `src/failure_router.py` - LearnedEscalationPolicy, hybrid routing

**Note:** Retriever already had `retrieve_for_escalation()` method; embedder already had `embed_failure_context()`. Infrastructure was ready.

---

### Priority 4: RLM Phase 3 - Escalation Integration ✅ COMPLETE

**Source:** `handoffs/active/rlm-orchestrator-roadmap.md`
**Completed:** 2026-01-14

**What was done:**
- [x] Error classification (`_classify_error()` in api.py)
- [x] Wire FailureRouter into Root LM loop
- [x] Role switching on escalation
- [x] Gate execution integration (FailureContext supports gate_name)

**Implementation:**
- Root LM loop tracks current_role, consecutive_failures, role_history
- FailureRouter consulted on errors, returns RoutingDecision (retry/escalate/fail)
- On "escalate" action: switch role, build escalation prompt with failure context
- Escalations logged via `progress_logger.log_escalation()`

**Files modified:**
- `src/api.py` - Escalation integration in Root LM loop

---

### Priority 5: RLM Phase 2 - RLM Enhancements ✅ COMPLETE

**Source:** `handoffs/active/rlm-orchestrator-roadmap.md`
**Completed:** 2026-01-14

**What was done:**
- [x] Forced exploration validation (`REPLConfig.require_exploration_before_final`)
- [x] Async `llm_batch_async()` using asyncio.gather
- [x] Configurable recursion depth (`LLMPrimitivesConfig.max_recursion_depth`, default 5)
- [x] Per-query cost tracking (`QueryCost` dataclass, `start_query/end_query` methods)

**Files modified:**
- `src/repl_environment.py` - Exploration tracking, FINAL validation
- `src/llm_primitives.py` - Async batch, recursion depth, cost tracking

**Test results:** 80 tests pass (31 primitives + 49 REPL)

---

### Priority 6: MemRL Memory Seeding ✅ COMPLETE

**Source:** `handoffs/active/memrl-episodic-memory.md`
**Completed:** 2026-01-14

**What was done:**
- [x] Seeded ~5,000 episodic memories (67% success, 33% failure)
- [x] Hierarchical decomposition patterns (70 memories)
- [x] Coding failure patterns (100 memories)
- [x] Diverse cross-domain failures (240 memories)
- [x] Template-generated failures (~1,000 memories)
- [x] Probabilistic strategies (~450 memories with variable outcomes)

**Key anti-patterns encoded:**
- Worker for architecture tasks (Q=0.10)
- Frontdoor for complex code (Q=0.05)
- No escalation after failures (Q=0.0)
- Unsafe code execution (Q=0.0)
- Conservative > aggressive estimates

**Files created:**
- `scripts/seed_decomposition_memories.py`
- `scripts/seed_failure_memories.py`
- `scripts/seed_diverse_failures.py`
- `scripts/seed_probabilistic_memories.py`

---

### Priority 7: Tool Registry Infrastructure ✅ COMPLETE

**Source:** Previous agent's plan (`glowing-splashing-eclipse.md`)
**Completed:** 2026-01-14

**What was done:**
- [x] Created `orchestration/tool_registry.yaml` (20+ tools)
- [x] Created tool executor (`orchestration/tools/executor.py`)
- [x] Created tool implementations (web, data, math, system, code, llm)
- [x] Created mining script (`scripts/mine_tool_definitions.py`)
- [x] Mined 608 tools from BFCL v4, LangChain, OpenAI, HuggingFace

**Pending:**
- [ ] Wire `TOOL()` into REPLEnvironment (not yet connected)

---

## Lower Priority (When Time Permits)

### Phase 5: Tool/Script REPL Integration
- Wire TOOL() and SCRIPT() into REPLEnvironment
- Script invoke/find methods
- MCP client implementation (blocked on MCP server setup)

### Phase 6: Early Failure Detection
- GenerationMonitor integration
- Entropy thresholds in registry
- Early abort on high-entropy output

### Phase 7: REPL Exploration Learning
- Log exploration strategies in REPLEnvironment
- Implement `EpisodicREPL.suggest_exploration()`
- Track token efficiency metrics

### Phase 8: Trajectory Visualization
- Enhanced SSE events for debugging
- Gradio visualization tab

---

## Blocked Tasks

| Task | Blocked On | Priority |
|------|------------|----------|
| Hyperparameter Tuning | Need live benchmarks | Medium |
| MTP Testing (GLM-4.6) | PR #15225 merge | Low |
| Claude-as-Judge scoring | Run baseline benchmark | Low |
| Production validation | Start llama-server | High |

---

## Quick Wins (Can Do Anytime)

1. **Wire TOOL() into REPL** - Executor exists, just needs connection
2. **Run formalizer benchmark** - `nohup ./scripts/benchmark/run_all_formalizers.sh &`
3. **Run orchestrator_planning.yaml benchmark** - Get baseline scores for MemRL evaluation
4. **Production validation** - Start llama-server, test real_mode=True

---

## Recommendation

**Priorities 1-7 are COMPLETE.** Next: **Wire TOOL() into REPL** or **Production validation**.

Immediate options:
- **Wire TOOL() into REPL** - 30 min, enables tool use in REPL
- **Production validation** - Debug llama-server startup, test real inference
- **Phase 7: REPL Exploration Learning** - Log strategies, suggest exploration
- **Run benchmarks** - Formalizer eval or Claude-as-Judge

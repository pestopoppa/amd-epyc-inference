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

### Priority 4: RLM Phase 3 - Escalation Integration

**Source:** `handoffs/active/rlm-orchestrator-roadmap.md`
**Effort:** High
**Impact:** High - enables automatic recovery

**Tasks:**
- [ ] Error classification in `failure_router.py`
- [ ] Wire FailureRouter into Root LM loop
- [ ] Role switching on escalation
- [ ] Gate execution integration

---

### Priority 5: RLM Phase 2 - RLM Enhancements

**Source:** `handoffs/active/rlm-orchestrator-roadmap.md`
**Effort:** High
**Impact:** Medium - improves RLM capabilities

**Tasks:**
- [ ] Forced exploration validation (prevent premature FINAL)
- [ ] Async `llm_batch_async()` for parallel sub-LM calls
- [ ] Configurable recursion depth
- [ ] Per-query cost tracking

---

## Lower Priority (When Time Permits)

### Phase 5: Tool/Script Completion
- MCP client implementation (blocked on MCP server setup)
- Script invoke/find methods (already have basic wiring)

### Phase 6: Early Failure Detection
- GenerationMonitor integration
- Entropy thresholds in registry
- Early abort on high-entropy output

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

---

## Quick Wins (Can Do Anytime)

1. **Update BLOCKED_TASKS.md Phase 1** - Verify real inference is working, update checkboxes
2. **Create sample scripts** in `orchestration/script_registry/` for testing SCRIPT() function
3. **Run orchestrator_planning.yaml benchmark** - Get baseline scores for MemRL evaluation

---

## Recommendation

**Priorities 1-3 are COMPLETE.** Next: **Priority 4 (RLM Escalation Integration)** or **Priority 5 (RLM Enhancements)**.

Why RLM Escalation Integration:
- Wires FailureRouter into Root LM loop
- Enables automatic recovery with role switching
- Gate execution integration

Alternative: Priority 5 (RLM Enhancements) can run in parallel:
- Async llm_batch_async() for parallel sub-LM calls
- Configurable recursion depth
- Per-query cost tracking

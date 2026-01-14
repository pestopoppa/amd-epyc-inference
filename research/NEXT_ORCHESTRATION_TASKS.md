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

### Priority 3: MemRL Phase 4 - Escalation Learning

**Source:** `orchestration/BLOCKED_TASKS.md`
**Effort:** Medium
**Impact:** Medium - improves failure handling

**Tasks:**
- [ ] Store failure contexts with escalation decisions
- [ ] Implement `LearnedEscalationPolicy` in FailureRouter
- [ ] Wire ProgressLogger escalation logging (deferred from Phase 2)

**Files:**
- `src/failure_router.py`
- `orchestration/repl_memory/retriever.py` (add escalation retrieval)

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

**Priorities 1-2 are COMPLETE.** Next: **Priority 3 (MemRL Phase 4 - Escalation Learning)**.

Why escalation learning:
- Builds on completed MemRL Phases 1-3
- Enables learned failure recovery strategies
- Files are already identified: `src/failure_router.py`, `orchestration/repl_memory/retriever.py`

Alternative: Priority 4 (RLM Escalation Integration) could run in parallel if escalation learning is blocked.

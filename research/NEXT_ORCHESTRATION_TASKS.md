# Next Orchestration Tasks

**Created:** 2026-01-14
**Purpose:** Prioritized list of next implementation tasks for the orchestrator

---

## Recommended Priority Order

### Priority 1: Role-Based Generation Defaults (NEW)

**Source:** Previous agent's plan (`glowing-splashing-eclipse.md`)
**Effort:** Medium (4 files)
**Impact:** High - enables concise architect outputs, proper token budgets

**Why now:**
- Already planned and designed
- No external dependencies
- Improves orchestration quality immediately

**Tasks:**
1. Add `generation_defaults` section to roles in `model_registry.yaml`
2. Add `system_prompt_suffix` for architect roles ("Be maximally concise...")
3. Extend `RoleConfig` dataclass in `registry_loader.py`
4. Apply defaults in `llm_call()` based on role

**Files:**
- `orchestration/model_registry.yaml`
- `src/registry_loader.py`
- `src/llm_primitives.py`
- `src/api.py` (pass registry to primitives)

---

### Priority 2: RLM Phase 1 - Backend Completion

**Source:** `orchestration/BLOCKED_TASKS.md`
**Effort:** Medium
**Impact:** Critical - unblocks real inference

**Status Note:** Jan 13 progress says "Real Inference Wiring Fix" is complete with `n_tokens` parameter added. Verify if Phase 1 items are actually done.

**Tasks (verify/complete):**
- [ ] Complete LlamaServerBackend HTTP (`src/backends/llama_server.py`)
- [ ] Wire CachingBackend init (`src/llm_primitives.py`)
- [ ] Connect role→backend routing (`src/llm_primitives.py`)
- [ ] Fix real mode initialization (`src/api.py`)

**Test:** Start llama-server and run real inference through API.

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

**Start with Priority 1 (Role-Based Generation Defaults)** because:
- Already designed by previous agent
- Self-contained, no external dependencies
- Immediate quality improvement
- Low risk of breaking existing functionality

After that, verify Priority 2 (Backend Completion) status - if real inference already works, mark complete and move to Priority 3/4 (Escalation).

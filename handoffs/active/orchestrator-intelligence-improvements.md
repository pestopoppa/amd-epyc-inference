# Handoff: Orchestrator Intelligence Improvements (Claude-Inspired)

- **Created**: 2026-02-12
- **Status**: COMPLETE (scaffolding + wiring)
- **Priority**: High
- **Blocked by**: None

---

## Context

7 improvements to the orchestration intelligence layer — routing, escalation, cost modeling, quality gating — inspired by Claude's architecture (multi-tier routing, prompt caching economics, extended thinking, structured outputs).

**Expected impact**: 30-50% avg latency reduction on routine tasks, 15-25% cost-efficiency improvement, fewer wasteful escalations.

---

## Dependency Graph & Execution Order

```
#8 Prefix Cache Expansion ──────────────────────── (independent, quick)
#3 Grammar-Constrained Outputs ────┬────────────── (independent, quick)
#4 Cache Affinity in Retriever ────┤────────────── (independent, quick)
#7 Reliable Tool Use ─────────────┘ (needs #3 for grammar passthrough)
#1 Multi-Dimensional Cost Model ────┬───────────── (foundational)
#2 Think-Harder Escalation ─────────┤ (needs #1 for cost-benefit)
#5 Try-Cheap-First Quality Gate ────┘ (needs #1, benefits from #2)
#6 Streaming Tool Use ─────────────────────────── (independent, benefits from #5 & #7)
```

**Execution order**: #8 → #3 → #4 → #1 → #7 → #2 → #5 → #6

---

## Improvement #8: Prefix Cache Expansion

**Files**:
- `orchestration/model_registry.yaml:49-53` — prefix_length: 256 → 4096
- `src/prefix_cache.py:136-148` — PrefixRouter.__init__() accepts prefix_length
- `orchestration/prompts/roles/*.md` — audit for prefix stability

**Success**: Cache hit rate increases; role prompts have static content first.

---

## Improvement #3: Grammar-Constrained Structured Outputs

**Files**:
- `src/backends/protocol.py:31-55` — Add json_schema, grammar fields to InferenceRequest
- `src/backends/llama_server.py:545-583` — Pass json_schema/grammar in _build_payload()
- `src/llm_primitives/primitives.py` — Add json_schema param to llm_call()
- `src/services/toon_encoder.py:193-228` — Activate encode_escalation_context()
- `src/prompt_builders/builder.py` — Wire TOON escalation encoding
- `tests/unit/test_llm_primitives.py` — Test passthrough

**Success**: TaskIR output valid JSON without post-processing; grammar passthrough works.

---

## Improvement #4: Cache Affinity Bonus in Retriever

**Files**:
- `orchestration/repl_memory/retriever.py:30-43` — Add cache_affinity to RetrievalResult
- `orchestration/repl_memory/retriever.py:65-88` — Add Phase 2.5 after Q-value ranking
- `orchestration/repl_memory/retriever.py:286-511` — Track last_role_used in HybridRouter
- `tests/unit/test_research_context.py` — Test cache affinity

**Success**: Consecutive similar requests route to same role more often.

---

## Improvement #1: Multi-Dimensional Cost Model

**Files**:
- `orchestration/repl_memory/q_scorer.py:35-80` — Add baseline_quality_by_role, memory_cost_by_role, dimension weights
- `orchestration/repl_memory/q_scorer.py:240-317` — Extend _compute_reward() with quality gap + memory penalties
- `tests/unit/test_q_scorer.py` — Add tests for multi-dimensional cost

**Success**: Q-values for worker_explore on simple tasks converge higher than architect_general.

---

## Improvement #7: Reliable Tool Use

**Files**:
- `src/tool_registry.py` — Add generate_gbnf_grammar(role) method
- `src/repl_environment/environment.py:432-610` — Relax one-tool-per-turn for read-only parallel
- `src/api/routes/chat_pipeline/stages.py` — Wire grammar into ReAct llm_call
- `src/api/routes/chat_routing.py` — Add tool_required/tool_hint
- `src/parsing_config.py` — Add toolrunner_structured: GBNF

**Success**: Zero malformed tool calls in structured mode; parallel read-only tools.

---

## Improvement #2: Think-Harder Escalation

**Files**:
- `src/escalation.py:62-71` — Add THINK_HARDER to EscalationAction enum
- `src/escalation.py:116-132` — Add config_override to EscalationDecision
- `src/escalation.py:211-305` — Insert THINK_HARDER in decide() on penultimate retry
- `src/api/routes/chat_pipeline/stages.py` — Handle config_override in execution
- `tests/unit/test_escalation.py` — Test THINK_HARDER conditions

**Success**: THINK_HARDER fires before model escalation; config_override applied.

---

## Improvement #5: Try-Cheap-First Quality Gate

**Files**:
- `src/api/routes/chat_pipeline/routing.py` — Add try_cheap_first flag
- `src/api/routes/chat_pipeline/__init__.py` — New _execute_cheap_first() stage
- `src/api/routes/chat.py` — Wire cheap-first before expensive execution
- `src/config.py` — Add try_cheap_first config with phase A/B/C
- `tests/unit/test_pipeline_routing.py` — Test cheap-first flow

**Success**: Routine tasks answered by 7B at 44 t/s; hard tasks reach specialists.

---

## Improvement #6: Streaming Tool Use

**Files**:
- `src/llm_primitives/primitives.py` — Add llm_call_stream()
- `src/api/routes/chat_pipeline/stream_adapter.py` — Token-level streaming
- `src/sse_utils.py` — Add tool_start_event, tool_end_event
- `src/api/routes/chat_pipeline/stages.py` — Convert _execute_react() to async gen
- `tests/unit/test_stream_adapter.py` — Streaming tests

**Success**: Tokens arrive in real-time; tool execution overlaps where possible.

---

## Completion Checklist

- [x] #8: prefix_length 256→4096, prompts audited
- [x] #3: json_schema/grammar in InferenceRequest, passthrough in llama_server
- [x] #4: Cache affinity Phase 2.5, last_role tracked
- [x] #1: Multi-dimensional cost (quality, memory, latency) in _compute_reward()
- [x] #7: GBNF grammar for tools, forced tool use, parallel read-only
- [x] #2: THINK_HARDER action, config_override, decide() logic
- [x] #5: Try-cheap-first pipeline, quality gate, MemRL learning
- [x] #6: Token streaming, parallel prefetch, thinking events
- [x] All tests pass (3746 passed, 67 skipped, 42s)
- [x] Debugger integration (diagnostic records extended with new tunables)
- [x] Architecture docs updated (7 chapters)
- [x] CHANGELOG, progress report, BLOCKED_TASKS updated
- [x] Wiring session: #2 think-harder, #7 GBNF grammar, #9 diagnostics, #6 tool events connected to live pipeline
- [x] Test fixes: stale signal count (20→22), wrong escalation edge assertion

---

## Resume Commands

```bash
# Resume from any point:
cd /mnt/raid0/llm/claude
cat handoffs/active/orchestrator-intelligence-improvements.md  # Check status

# After each improvement:
make gates && pytest tests/ -x -q

# Full verification:
pytest tests/ -q
```

# Pydantic Graph Migration — Completed

> **Status**: Implementation complete, documentation updates pending
> **Created**: 2026-02-07
> **Completed**: 2026-02-07
> **Priority**: Medium-High

## Summary

Migrated orchestration escalation logic from scattered implicit routing (6+ files) to an explicit pydantic-graph structure. The graph encodes valid transitions in Union return types, MemRL functions are wired as deps, and `run_task()` replaces the manual state machine.

---

## What Was Done

### Phase 1: Documentation (Complete)
- [x] Generated Mermaid topology diagram (`docs/diagrams/orchestration_topology.md`)
- [x] Documented current routing flow
- [x] Created research handoff

### Phase 2: Graph Module (Complete)
- [x] Installed `pydantic-graph` dependency
- [x] Created `src/graph/state.py` — TaskState, TaskDeps, TaskResult, GraphConfig
- [x] Created `src/graph/nodes.py` — 7 node classes + shared helpers
- [x] Created `src/graph/graph.py` — orchestration_graph, run_task(), iter_task(), generate_mermaid()
- [x] Created `src/graph/persistence.py` — SQLiteStatePersistence adapter
- [x] Created `src/graph/__init__.py` and `src/graph/_compat.py`

### Phase 3: Integration (Complete)
- [x] Replaced manual for-loop in `repl_executor.py` with `run_task()`
- [x] Removed `failure_router` / `routing_facade` fields from AppState
- [x] Removed their initialization from lifespan
- [x] Removed LearnedEscalationPolicy wiring from `memrl.py`
- [x] Replaced `routing_facade.decide()` in `chat.py` with `EscalationPolicy().decide()`
- [x] Updated `builder.py` type hints (removed FailureContext/RoutingDecision)
- [x] Updated `code_utils.py` imports to `src.escalation`

### Phase 4: Cleanup (Complete)
- [x] Deleted `src/failure_router.py`
- [x] Deleted `src/routing_facade.py`
- [x] Deleted `tests/unit/test_failure_router.py`
- [x] Deleted `tests/unit/test_routing_facade.py`
- [x] Rewrote `test_repl_executor.py` (23 tests mock `run_task`)
- [x] Rewrote `test_generation_monitor.py` integration tests

### Phase 5: Tests (Complete)
- [x] 47 graph-specific tests pass (nodes, integration, persistence, mermaid)
- [x] 2677 total unit tests pass, 0 failures
- [x] `make gates` — no new failures (shellcheck failure is pre-existing in deprecated script)

---

## Bug Fixes Included

| Bug | Where Fixed |
|-----|-------------|
| `escalation_count` never incremented | `_handle_error()` in `src/graph/nodes.py` |
| `record_failure()` never called | `_handle_error()` in `src/graph/nodes.py` |
| `record_mitigation()` never called | `End` path after escalation in nodes |
| `add_evidence()` never called | `End` path on task completion in nodes |
| 3 hardcoded `EscalationPolicy()` fallbacks | Eliminated — graph is single code path |
| Arrow format parsing (`"escalate:from->to"`) | Fixed in research phase (now in graph nodes) |

---

## Architecture

### Node Classes

| Node | Role(s) | Escalates To |
|------|---------|-------------|
| `FrontdoorNode` | FRONTDOOR | CoderNode |
| `WorkerNode` | WORKER_* | CoderNode |
| `CoderNode` | CODER_PRIMARY, THINKING_REASONING | ArchitectNode |
| `CoderEscalationNode` | CODER_ESCALATION | ArchitectCodingNode |
| `IngestNode` | INGEST_LONG_CONTEXT | ArchitectNode |
| `ArchitectNode` | ARCHITECT_GENERAL | End (terminal) |
| `ArchitectCodingNode` | ARCHITECT_CODING | End (terminal) |

### MemRL Wiring

| Function | When Called |
|----------|------------|
| `failure_graph.record_failure()` | On every error in `_handle_error()` |
| `failure_graph.record_mitigation()` | When escalated role succeeds |
| `hypothesis_graph.add_evidence()` | On task success/failure |
| `retriever.retrieve_for_escalation()` | During `_check_memrl_suggestion()` |

---

## Files

### Created
| File | Purpose |
|------|---------|
| `src/graph/__init__.py` | Public exports |
| `src/graph/state.py` | TaskState, TaskDeps, TaskResult, GraphConfig |
| `src/graph/nodes.py` | 7 node classes + helpers |
| `src/graph/graph.py` | Graph construction, run_task, mermaid |
| `src/graph/persistence.py` | SQLiteStatePersistence adapter |
| `src/graph/_compat.py` | Bridge helpers for old API callers |
| `tests/unit/test_graph_nodes.py` | 24 node unit tests |
| `tests/unit/test_orchestration_graph.py` | 6 integration tests |
| `tests/unit/test_graph_persistence_adapter.py` | 9 persistence tests |
| `tests/unit/test_graph_mermaid.py` | 8 mermaid tests |

### Modified
| File | Change |
|------|--------|
| `pyproject.toml` | Added `pydantic-graph` dependency |
| `src/api/routes/chat_pipeline/repl_executor.py` | Replaced manual loop with `run_task()` |
| `src/api/state.py` | Removed failure_router/routing_facade fields |
| `src/api/__init__.py` | Removed FailureRouter/RoutingFacade initialization |
| `src/api/services/memrl.py` | Removed LearnedEscalationPolicy wiring |
| `src/api/routes/chat.py` | Direct `EscalationPolicy().decide()` |
| `src/prompt_builders/builder.py` | Updated type hints |
| `src/prompt_builders/code_utils.py` | Updated imports |
| `tests/unit/test_repl_executor.py` | Rewrote all tests to mock `run_task` |
| `tests/unit/test_generation_monitor.py` | Rewrote integration tests |

### Deleted
| File | Replacement |
|------|-------------|
| `src/failure_router.py` | Graph nodes |
| `src/routing_facade.py` | Graph nodes |
| `tests/unit/test_failure_router.py` | `test_graph_nodes.py` |
| `tests/unit/test_routing_facade.py` | Graph integration tests |

---

## Remaining Documentation Updates

These are non-blocking but should be done:

| Document | Change Needed |
|----------|--------------|
| `docs/chapters/18-escalation-and-routing.md` | Rewrite: replace RoutingFacade/FailureRouter docs with pydantic-graph architecture |
| `docs/chapters/10-orchestration-architecture.md` | Update component flow, add Mermaid from `generate_mermaid()` |
| `docs/chapters/15-memrl-system.md` | Document newly-wired `record_failure()`, `record_mitigation()`, `add_evidence()` |
| `docs/chapters/16-graph-reasoning.md` | Add pydantic-graph alongside Kuzu-based graphs |
| `docs/chapters/20-session-persistence.md` | Document SQLiteStatePersistence adapter |
| `src/escalation.py` | Update comments referencing `failure_router.py` |
| `src/roles.py` | Remove "backwards compatibility with failure_router" comment |
| `src/README.md` | Remove failure_router.py entry |
| `CLAUDE.md` | Update Component Flow section |
| `agents/AGENT_INSTRUCTIONS.md` | Remove RoutingFacade/FailureRouter references |

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|-----------|
| State persistence backend? | SQLiteStatePersistence wraps existing SQLiteSessionStore |
| Migration strategy? | Big-bang (all phases in one session) |
| Testing approach? | Full test rewrite to mock at `run_task` boundary |
| Rollback plan? | `_compat.py` bridge helpers exist; git revert if needed |

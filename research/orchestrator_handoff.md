> **DEPRECATED**: This handoff has moved to `handoffs/active/orchestrator.md`. This file is kept for historical reference only.

# Orchestrator Implementation - Handoff Document

**Goal**: Hierarchical LLM orchestration system using RLM (Recursive Language Models) pattern.

**Status**: CORE COMPONENTS COMPLETE (Mock Mode) + RadixAttention Ready + Tooling Architecture

**Last Updated**: 2026-01-08

---

## Quick Start

```bash
# Run all unit tests
python3 -m pytest tests/unit/test_repl_environment.py \
    tests/unit/test_llm_primitives.py \
    tests/unit/test_gate_runner.py \
    tests/unit/test_failure_router.py \
    tests/unit/test_api.py \
    tests/unit/test_tool_registry.py \
    tests/unit/test_script_registry.py -v

# Test tooling infrastructure (mock mode)
python scripts/test_tooling_mock.py

# Start API server (mock mode)
cd /mnt/raid0/llm/claude
uvicorn src.api:app --reload --port 8000

# Test API
curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello", "mock_mode": true}'
```

---

## Implementation Status

### ✅ Complete (Mock Mode Ready)

| Component | File | Tests | Purpose |
|-----------|------|-------|---------|
| REPL Environment | `src/repl_environment.py` | 49 | Sandboxed Python execution |
| LLM Primitives | `src/llm_primitives.py` | 31 | `llm_call()`, `llm_batch()` |
| Gate Runner | `src/gate_runner.py` | 22 | Quality gate execution |
| Failure Router | `src/failure_router.py` | 51 | Escalation routing |
| FastAPI | `src/api.py` | 26 | HTTP interface |
| System Prompts | `src/prompts/*.txt` | — | 5 role prompts |
| Gate Config | `config/gates.yaml` | — | 7 gate definitions |

### ✅ Previously Complete (Foundation)

| Component | File | Purpose |
|-----------|------|---------|
| Dispatcher | `src/dispatcher.py` | TaskIR routing |
| Registry Loader | `src/registry_loader.py` | Model registry parsing |
| Executor | `src/executor.py` | Step execution |
| Context Manager | `src/context_manager.py` | Inter-step context |
| Model Server | `src/model_server.py` | Inference abstraction |
| CLI | `src/cli.py` | Command-line interface |

### ✅ RadixAttention Infrastructure (2026-01-07)

| Component | File | Purpose |
|-----------|------|---------|
| LlamaServerBackend | `src/backends/llama_server.py` | HTTP client for llama-server with prefix caching |
| PrefixRouter | `src/prefix_cache.py` | Routes prompts to slots based on prefix hash |
| CachingBackend | `src/prefix_cache.py` | Wraps backend with automatic slot routing |
| canonicalize_prompt() | `src/prefix_cache.py` | Normalizes prompts for better cache hits |
| RadixCache | `src/radix_cache.py` | O(n) prefix lookup with LRU eviction |

**Tests**: 46/46 passing in `tests/unit/test_prefix_cache.py`

**Next Step**: Integration into `llm_primitives.py` - see `research/orchestration_integration_handoff.md`

### ✅ Tooling Architecture (2026-01-08)

| Component | File | Purpose |
|-----------|------|---------|
| Tool Registry | `src/tool_registry.py` | Role-based permission system (5 categories) |
| Script Registry | `src/script_registry.py` | Prepared scripts with fuzzy discovery |
| Source Registry | `orchestration/source_registry.yaml` | Website hierarchy by language/task |
| Parsing Config | `src/parsing_config.py` | GBNF vs Instructor by role |
| Web Tools | `src/tools/web/` | fetch_docs, web_search |
| File Tools | `src/tools/file/` | read_file, list_dir |
| Code Tools | `src/tools/code/` | run_tests, lint_python |
| Data Tools | `src/tools/data/` | json_parse, yaml_parse |

**REPL Integration**: `TOOL()`, `SCRIPT()`, `list_tools()`, `find_scripts()` injected into globals

**Tests**: 29 passing in `tests/unit/test_tool_registry.py` + `tests/unit/test_script_registry.py`

**Token Savings**: 92% via prepared scripts vs generating code from scratch

**Role Permissions**:
| Tier | Web Access | Categories |
|------|------------|------------|
| A (frontdoor) | Yes | web, file, data |
| B (specialists) | Yes | web, file, code, data, system |
| C (workers) | No | file, data (read-only) |
| D (draft) | No | None |

### ✅ Frontend Architecture (2026-01-08)

| Component | File | Purpose |
|-----------|------|---------|
| SSE Streaming | `src/api.py` | `/chat/stream` endpoint with routing metadata |
| OpenAI API | `src/api.py` | `/v1/chat/completions`, `/v1/models` endpoints |
| Gradio Web UI | `src/gradio_ui.py` | Web interface with chat, artifacts, routing viz |
| LiteLLM Config | `config/litellm_config.yaml` | Proxy config for OpenAI-compatible clients |

**Quick Start**:
```bash
# Start Gradio Web UI (local only)
python -m src.gradio_ui --port 7860

# With public URL (gradio.live)
python -m src.gradio_ui --share

# Start LiteLLM proxy (for LM Studio, etc.)
pip install 'litellm[proxy]'
litellm --config config/litellm_config.yaml --port 4000
```

**Endpoints**:
- `/chat/stream` - SSE streaming with turn/token/tool/file/thinking events
- `/v1/chat/completions` - OpenAI-compatible (streaming + non-streaming)
- `/v1/models` - Lists available roles
- `/sessions` - Session management (list, resume, rename)
- `/permission/{id}` - Permission request/response flow

**CLI** (separate repo): Plan at `/home/daniele/.claude/plans/twinkly-sniffing-crescent.md`

### ✅ CLI Parity Features (2026-01-08)

| Feature | Implementation | Purpose |
|---------|----------------|---------|
| Extended Thinking | `thinking_budget` param | Reserve tokens for reasoning (Claude Code `ultrathink`) |
| Thinking Events | `{"type": "thinking"}` SSE | Stream internal reasoning (Ctrl+O verbose mode) |
| Permission Modes | `permission_mode` param | normal/auto-accept/plan (Shift+Tab) |
| Session Management | `/sessions/*` endpoints | Resume, rename, list sessions |
| Permission Flow | `/permission/*` endpoints | Interactive tool approval |

**New SSE Event Types**:
```
{"type": "thinking", "content": "Analyzing..."}
{"type": "permission_request", "id": "...", "tool": "...", "args": {...}}
```

**Tests**: 23/23 integration tests in `tests/integration/test_frontend_integration.py`

### ❌ Not Implemented (Requires Models)

| Component | Blocker | Priority |
|-----------|---------|----------|
| Real inference mode | Need running model servers | High |
| Root LM loop | Need frontdoor model | High |
| Integration tests | Would load models | Medium |
| Prompt tuning | Need model feedback | Low |

---

## Architecture Overview

```
User Request
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI (/chat endpoint)                   │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                     REPL Environment                         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  context = "<user input>"  # NEVER sent to LLM          ││
│  │  artifacts = {}            # Step outputs                ││
│  │  Built-ins: peek(), grep(), FINAL(), llm_call()         ││
│  │  Tooling:   TOOL(), SCRIPT(), list_tools(), find_scripts()│
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
      │
      ├──────────────┬──────────────┬──────────────┐
      ▼              ▼              ▼              ▼
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│ LLM       │  │ Gate      │  │ Failure   │  │ System    │
│ Primitives│  │ Runner    │  │ Router    │  │ Prompts   │
│           │  │           │  │           │  │           │
│ llm_call  │  │ format    │  │ worker→   │  │ root_lm   │
│ llm_batch │  │ lint      │  │ coder→    │  │ coder     │
│           │  │ unit      │  │ architect │  │ worker    │
└───────────┘  └───────────┘  └───────────┘  └───────────┘
```

---

## Key Design Patterns

### 1. Context-as-Object (RLM Pattern)

```python
# Root LM writes code that manipulates context as a variable
# Context is NEVER sent directly to the LLM

# In REPL environment:
repl = REPLEnvironment(context="<large user input>")
repl.execute("""
# Root LM generated code:
print(f"Context is {len(context)} chars")
preview = peek(500)  # See first 500 chars
matches = grep(r"def \w+")  # Find function definitions
""")
```

### 2. Parallel Sub-LM Calls

```python
# llm_batch() for parallel processing
chunks = [context[i:i+4000] for i in range(0, len(context), 4000)]
summaries = llm_batch([f"Summarize:\n{c}" for c in chunks], role="worker")
# All chunks processed in parallel
```

### 3. Escalation Chains

```python
# Failure routing
router = FailureRouter()

# First failure → retry same role
# Second failure → escalate
context = FailureContext(role="worker", failure_count=2, error_category="code")
decision = router.route_failure(context)
# → action="escalate", next_role="coder"
```

### 4. Quality Gates

```python
# Gates run after code-producing steps
runner = GateRunner()
results = runner.run_all_gates(stop_on_first_failure=True)

# Failure info routed back to producing agent
for r in results:
    if not r.passed:
        print(f"{r.gate_name} failed: {r.errors}")
```

---

## File Locations

### Source Code
```
src/
├── repl_environment.py    # Sandboxed REPL with TOOL/SCRIPT
├── llm_primitives.py      # LLM call/batch
├── tool_registry.py       # Tool registry with permissions (NEW)
├── script_registry.py     # Script registry with fuzzy search (NEW)
├── parsing_config.py      # Parsing strategy by role (NEW)
├── gate_runner.py         # Gate execution
├── failure_router.py      # Escalation routing
├── api.py                 # FastAPI endpoints + SSE streaming + OpenAI compat
├── gradio_ui.py           # Gradio web interface (NEW)
├── tools/                 # Tool implementations
│   ├── __init__.py       # register_all_tools()
│   ├── base.py           # Base tool classes
│   ├── web/              # Web tools (fetch, search)
│   ├── file/             # File tools (read, list)
│   ├── code/             # Code tools (tests, lint)
│   └── data/             # Data tools (json, yaml)
├── prompts/
│   ├── root_lm_system.txt
│   ├── coder_system.txt
│   ├── worker_system.txt
│   ├── architect_system.txt
│   └── ingest_system.txt
├── dispatcher.py          # TaskIR routing (existing)
├── executor.py            # Step execution (existing)
├── context_manager.py     # Inter-step context (existing)
├── model_server.py        # Inference abstraction (existing)
└── cli.py                 # CLI entry point (existing)
```

### Configuration
```
config/
├── gates.yaml             # Gate definitions (7 gates)
└── litellm_config.yaml    # LiteLLM proxy configuration (NEW)

orchestration/
├── model_registry.yaml    # Role → model mapping (with tool_permissions)
├── source_registry.yaml   # Website hierarchy by language/task (NEW)
├── tool_registry.schema.json  # Tool schema (NEW)
├── task_ir.schema.json    # TaskIR JSON schema
├── formalization_ir.schema.json  # With script/web_sources fields
├── architecture_ir.schema.json
└── script_registry/       # Prepared scripts (NEW)
    ├── web/              # fetch_docs.json, search_arxiv.json
    ├── code/             # run_pytest.json, lint_python.json
    └── data/             # parse_json.json
```

### Tests
```
tests/unit/
├── test_repl_environment.py  # 49 tests
├── test_llm_primitives.py    # 31 tests
├── test_gate_runner.py       # 22 tests
├── test_failure_router.py    # 51 tests
├── test_api.py               # 26 tests
├── test_tool_registry.py     # 15 tests (NEW)
├── test_script_registry.py   # 14 tests (NEW)
├── test_prefix_cache.py      # 46 tests
├── test_dispatcher.py        # (existing)
├── test_executor.py          # (existing)
└── test_context_manager.py   # (existing)

scripts/
└── test_tooling_mock.py      # Integration test for tooling (NEW)
```

---

## API Reference

### POST /chat

```bash
curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Summarize this code",
        "context": "def foo(): pass",
        "mock_mode": true,
        "max_turns": 10
    }'
```

Response:
```json
{
    "answer": "[MOCK] Processed prompt...",
    "turns": 1,
    "tokens_used": 0,
    "elapsed_seconds": 0.001,
    "mock_mode": true
}
```

### POST /gates

```bash
curl -X POST http://localhost:8000/gates \
    -H "Content-Type: application/json" \
    -d '{"gate_names": ["format", "lint"]}'
```

### GET /health

```bash
curl http://localhost:8000/health
# {"status": "ok", "models_loaded": 0, ...}
```

---

## Integration Checklist

### When Benchmarks Complete

1. [ ] Enable real mode in LLM Primitives:
   ```python
   primitives = LLMPrimitives(model_server=server, mock_mode=False)
   ```

2. [ ] Start model servers for each role:
   ```bash
   # Frontdoor (Root LM)
   llama-server -m Qwen3-Coder-30B-A3B.gguf --port 8080

   # Coder
   llama-server -m Qwen2.5-Coder-32B.gguf --port 8081
   ```

3. [ ] Test real inference:
   ```python
   result = primitives.llm_call("Test prompt", role="frontdoor")
   assert "[MOCK]" not in result
   ```

4. [ ] Run integration tests:
   ```bash
   python3 -m pytest tests/integration/ -v
   ```

### Wire into Executor

1. [ ] Create `REPLEnvironment` in executor
2. [ ] Inject `llm_call`/`llm_batch` into REPL globals
3. [ ] Run Root LM loop:
   ```python
   for turn in range(max_turns):
       code = model_server.infer("frontdoor", prompt)
       result = repl.execute(code)
       if result.is_final:
           return result.final_answer
   ```

4. [ ] Connect gate failures to failure router
5. [ ] Implement escalation in executor

---

## Testing Without Models

All components have mock mode enabled by default:

```python
# REPL - always works (no model needed)
repl = REPLEnvironment(context="test")
result = repl.execute("print(len(context))")

# LLM Primitives - mock responses
primitives = LLMPrimitives(mock_mode=True)
result = primitives.llm_call("test")  # Returns "[MOCK] Response..."

# Gate Runner - runs real gates
runner = GateRunner()
results = runner.run_all_gates()  # Runs make format, lint, etc.

# Failure Router - pure logic
router = FailureRouter()
decision = router.route_failure(context)  # No model needed

# API - mock mode default
# POST /chat with mock_mode=true
```

---

## Known Limitations

| Limitation | Workaround |
|------------|------------|
| No real inference in mock mode | Expected - use for testing only |
| REPL timeout is process-based | Works on Linux, may need adjustment on other OS |
| Gate runner subprocess calls | May be slow on first run (cold cache) |
| FastAPI requires uvicorn | Install with `pip install uvicorn` |

---

## Performance Expectations

### Mock Mode (Current)

- API response: <10ms
- REPL execution: <100ms
- Gate execution: varies by gate (format ~2s, unit ~10s)

### Real Mode (When Enabled)

Based on model benchmarks:
- Frontdoor (Qwen3-Coder-30B): ~45 t/s with MoE reduction
- Coder (Qwen2.5-Coder-32B): ~33 t/s with speculative
- Worker (Llama-3-8B): ~85 t/s with prompt lookup

---

## Related Documents

| Document | Purpose |
|----------|---------|
| `research/Hierarchical_Orchestration_Methodology.md` | Full design spec |
| `research/amd_pace_testing.md` | AMD PACE benchmark handoff |
| `research/orchestration_integration_handoff.md` | RadixAttention integration |
| `orchestration/progress/PROGRESS_2026-01-08.md` | Tooling architecture progress |
| `orchestration/model_registry.yaml` | Role → model mapping |
| Plan file (`twinkly-sniffing-crescent.md`) | Tooling architecture phases |

---

## Contact Points

- **Orchestrator design**: See methodology document
- **Model performance**: See RESULTS_SUMMARY.md
- **Gate configuration**: Edit `config/gates.yaml`
- **Escalation rules**: Edit `src/failure_router.py`

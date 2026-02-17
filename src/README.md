# Source Code Overview

This directory contains the core orchestration system for hierarchical LLM inference.

## Architecture

```
User Request
    │
    ▼
┌─────────────────────────────────────────┐
│  API Layer (api/)                       │
│  - FastAPI endpoints                    │
│  - Request/response models              │
│  - SSE streaming                        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Orchestration Core                     │
│  - Root LM loop (api.py)                │
│  - REPL environment (repl_environment)  │
│  - Escalation policy (escalation.py)    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  LLM Primitives (llm_primitives.py)     │
│  - Role-based model routing             │
│  - Caching backend support              │
│  - Cost tracking                        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Backends (backends/)                   │
│  - LlamaServerBackend                   │
│  - CachingBackend (prefix cache)        │
│  - Mock backend (testing)               │
└─────────────────────────────────────────┘
```

## Directory Structure

```
src/
├── api/                    # FastAPI HTTP layer
│   ├── models/             # Pydantic request/response models
│   ├── routes/             # Endpoint implementations
│   │   ├── chat.py         # /chat endpoints
│   │   └── openai_compat.py # /v1/* OpenAI-compatible
│   ├── services/           # Business logic services
│   │   ├── orchestrator.py # Prompt building, code extraction
│   │   └── memrl.py        # MemRL integration
│   └── state.py            # AppState and dependency providers
│
├── backends/               # LLM backend implementations
│   ├── protocol.py         # Backend interface protocol
│   └── llama_server.py     # llama-server HTTP client
│
├── prompts/                # System prompt templates
│   ├── root_lm_system.txt  # Root LM orchestrator prompt
│   ├── coder_system.txt    # Coder specialist prompt
│   └── worker_system.txt   # Worker prompt
│
├── tools/                  # Tool implementations
│   └── executor.py         # Tool execution engine
│
└── [Core Modules]          # See below
```

## Core Modules

### Orchestration

| Module | Purpose |
|--------|---------|
| `api.py` | Main FastAPI app (legacy), Root LM loop implementation |
| `repl_environment.py` | Python REPL sandbox for Root LM code execution |
| `executor.py` | Full RLM execution loop with recursive calls |
| `escalation.py` | Unified escalation policy (EscalationPolicy, EscalationContext) |
| `graph/` | Pydantic-graph orchestration (nodes, state, persistence, mermaid) |
| `dispatcher.py` | TaskIR parsing and specialist routing |

### LLM Interface

| Module | Purpose |
|--------|---------|
| `llm_primitives.py` | High-level LLM API (`llm_call`, `llm_batch_async`) |
| `model_server.py` | Low-level model server interface |
| `registry_loader.py` | Load model configs from `model_registry.yaml` |
| `roles.py` | Role enum with escalation chains |

### Caching & Performance

| Module | Purpose |
|--------|---------|
| `prefix_cache.py` | PrefixRouter, CachingBackend for KV cache reuse |
| `radix_cache.py` | RadixCache for token-level prefix matching |
| `generation_monitor.py` | Early failure detection via entropy/repetition |

### Tools & Scripts

| Module | Purpose |
|--------|---------|
| `tool_registry.py` | Tool definitions and permission checking |
| `script_registry.py` | Script management for REPL |
| `builtin_tools.py` | Built-in tools (grep, glob, read) |

### Configuration

| Module | Purpose |
|--------|---------|
| `config.py` | Hierarchical configuration (OrchestratorConfig) |
| `features.py` | Feature flags (mock_mode, generation_monitor, etc.) |
| `parsing_config.py` | Parsing configurations for different roles |

### Utilities

| Module | Purpose |
|--------|---------|
| `prompt_builders.py` | Prompt construction for Root LM and specialists |
| `sse_utils.py` | SSE event helpers for streaming |
| `context_manager.py` | Context window management |
| `restricted_executor.py` | RestrictedPython sandbox |
| `proactive_delegation.py` | Architect-driven task decomposition (Phase 5) |

## Key Concepts

### Root LM Pattern (RLM)

The orchestrator uses a "Root LM" pattern where:
1. Root LM receives a task and REPL state
2. Root LM generates Python code to explore/solve
3. Code executes in sandboxed REPL
4. Root LM sees output, iterates until `FINAL(answer)` called

### Role-Based Routing

Requests route to different models based on task type:
- `frontdoor`: Interactive chat, task routing
- `coder_escalation`: Code generation
- `architect_general`: System design, escalation target
- `worker_*`: Parallel file-level tasks

See `src/roles.py` for the complete Role enum.

### Escalation Chain

```
worker → coder → architect
```

Failures escalate up the chain based on retry count and error category.
The escalation loop is driven by `src/graph/` (pydantic-graph), with MemRL
providing learned advisory via dependency injection.

## Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests (requires llama-server)
pytest tests/integration/ -v --run-server
```

## Related Documentation

- `orchestration/model_registry.yaml` - Model configurations
- `orchestration/task_ir.schema.json` - TaskIR JSON schema
- `handoffs/active/orchestration-refactoring.md` - Refactoring status
- `research/Orchestrator_Implementation_Plan.md` - Design document

# Orchestration

This folder contains the schemas, validators, and configuration for the hierarchical local-agent orchestration system.

## Contents

| File | Purpose |
|------|---------|
| `task_ir.schema.json` | JSON Schema for TaskIR (emitted by Front Door) |
| `architecture_ir.schema.json` | JSON Schema for ArchitectureIR (emitted by Tier-B3 Architect) |
| `validate_ir.py` | Python validator for both schemas |
| `model_registry.yaml` | Deterministic model → role mapping with acceleration configs |

## Quick Start

### 1. Validate a TaskIR file

```bash
# Validate a file
python orchestration/validate_ir.py task orchestration/last_task_ir.json

# Validate from stdin (e.g., Front Door output)
echo '{"task_id": "abc", ...}' | python orchestration/validate_ir.py task -
```

### 2. Validate an ArchitectureIR file

```bash
python orchestration/validate_ir.py arch architecture/architecture_ir.json
```

### 3. Run all gates

```bash
make gates
# or
just gates
```

## Workflow

1. **Front Door** receives user input and emits `TaskIR` JSON
2. Save output to `orchestration/last_task_ir.json` (gitignored)
3. **Dispatcher** reads `TaskIR` and routes to specialists/workers
4. Workers emit artifacts (code, docs, diffs)
5. Run `make gates` to validate
6. On failure: return gate report to producing agent (once)
7. On second failure: escalate one tier

## TaskIR Schema

Required fields:
- `task_id`: Unique identifier (UUID recommended)
- `task_type`: `chat | doc | code | ingest | manage`
- `priority`: `interactive | batch`
- `objective`: What success looks like
- `inputs`: Array of `{type, value}` objects
- `constraints`: Hard requirements
- `assumptions`: Decisions made when request was ambiguous
- `agents`: Which agent roles are needed
- `plan.steps`: Ordered steps with `{id, actor, action, outputs}`
- `gates`: Which verification gates to run
- `definition_of_done`: Human-readable success criteria
- `escalation`: When and how to escalate failures

See `task_ir.schema.json` for full specification.

## ArchitectureIR Schema

Emitted by Tier-B3 Architect for system design decisions. Includes:
- Goals and non-goals
- Global invariants
- Repository layout with ownership
- Module definitions with public APIs
- Contracts (OpenAPI, JSON Schema, etc.)
- Cross-cutting concerns (logging, errors, config, security)
- Acceptance criteria (tests, benchmarks)
- Architecture Decision Records (ADRs)

See `architecture_ir.schema.json` for full specification.

## Model Registry

`model_registry.yaml` maps agent roles to specific models:

```yaml
roles:
  frontdoor:
    model: Qwen3-Coder-30B-A3B-Instruct
    acceleration:
      type: moe_expert_reduction
      experts: 4
  
  coder_primary:
    model: Qwen2.5-Coder-32B-Instruct
    acceleration:
      type: speculative_decoding
      draft: Qwen2.5-Coder-0.5B-Instruct
      k: 24
```

The dispatcher should read this file and **never improvise** model selection.

## Files to Gitignore

Add to your `.gitignore`:

```
# Transient IR files
orchestration/last_task_ir.json
orchestration/last_architecture_ir.json

# Gate reports
orchestration/gate_report_*.json
```

## Dependencies

```bash
pip install jsonschema>=4.20
```

Or add to `requirements-dev.txt`.

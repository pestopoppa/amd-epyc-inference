# Orchestration

This folder contains the schemas, validators, configuration, and **self-management procedures** for the hierarchical local-agent orchestration system.

## Contents

| File | Purpose |
|------|---------|
| `task_ir.schema.json` | JSON Schema for TaskIR (emitted by Front Door) |
| `architecture_ir.schema.json` | JSON Schema for ArchitectureIR (emitted by Tier-B3 Architect) |
| `validate_ir.py` | Python validator for both schemas |
| `model_registry.yaml` | Deterministic model → role mapping with acceleration configs |
| `procedure.schema.json` | JSON Schema for procedure YAML validation |
| `procedure_registry.py` | Procedure loader, validator, and executor |
| `procedure_scheduler.py` | Background job scheduler with pause/resume |
| `procedures/` | YAML procedure definitions (11 procedures) |
| `procedures/state/` | Procedure execution state files |
| `checkpoints/` | Server config checkpoints for hot-swap |
| `patches/` | Patch queue for approval workflow |

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

## Tool Compliance Testing

Models must use REPL tools instead of Python imports. The sandbox blocks dangerous operations, so models that use `import os`, `pathlib`, etc. will fail.

### Run Compliance Tests

```bash
# Mock mode (no live models needed)
pytest tests/integration/test_model_tool_compliance.py -v

# Live model tests (requires orchestrator running)
pytest tests/integration/test_model_tool_compliance.py -v --run-live-models
```

### Benchmark Prompts

Tool compliance benchmark prompts are in `benchmarks/prompts/v1/tool_compliance.yaml`:
- **Tier 1**: Basic tool usage (list_dir, peek, grep)
- **Tier 2**: Combined tools and file metadata
- **Tier 3**: Document processing, LLM delegation, shell commands

### Tool Mapping (REPL → Forbidden Python)

| REPL Tool | Forbidden Alternative |
|-----------|----------------------|
| `list_dir(path)` | `os.listdir()`, `pathlib.Path().iterdir()` |
| `peek(n, file_path)` | `open().read()`, `Path().read_text()` |
| `grep(pattern)` | `re.findall()`, subprocess |
| `file_info(path)` | `os.stat()`, `Path().stat()` |
| `run_shell(cmd)` | `subprocess.run()`, `os.system()` |

See `docs/reference/models/QUIRKS.md` for model-specific compliance notes.

## Procedure Registry (Self-Management)

The Procedure Registry enables deterministic self-management operations with ~350 tokens per operation (vs 3000-5000 for manual execution).

### Available Procedures (11 total)

| Procedure | Category | Purpose |
|-----------|----------|---------|
| `benchmark_new_model` | benchmark | Run benchmark suite on new GGUF model |
| `check_draft_compatibility` | benchmark | Validate draft-target pairing for spec decode |
| `add_model_to_registry` | registry | Add new model entry with all fields |
| `update_registry_performance` | registry | Update t/s, speedup after benchmarks |
| `add_model_quirks` | registry | Document discovered model quirks |
| `deprecate_model` | registry | Mark deprecated (manual delete only) |
| `run_quality_gates` | codebase | Run full gate suite (lint, tests, etc.) |
| `create_handoff` | codebase | Generate handoff documents |
| `prepare_finetuning_dataset` | finetuning | Prepare/split datasets |
| `run_finetuning` | finetuning | Execute LoRA/QLoRA training |
| `evaluate_finetuned_model` | finetuning | Post-training evaluation |

### Using Procedures

```python
from orchestration.procedure_registry import ProcedureRegistry

# Initialize registry
registry = ProcedureRegistry()

# List all procedures
procedures = registry.list_procedures()

# List by category
benchmark_procs = registry.list_procedures(category="benchmark")

# Execute a procedure
result = registry.execute(
    "benchmark_new_model",
    model_path="/mnt/raid0/llm/models/NewModel.gguf",
    model_name="NewModel"
)
```

### REPL Tools for Procedures

| Tool | Purpose |
|------|---------|
| `run_procedure(id, **params)` | Execute procedure |
| `list_procedures(category=None)` | List available procedures |
| `get_procedure_status(id)` | Check execution status |
| `checkpoint_create(name)` | Save server configs |
| `checkpoint_restore(id)` | Restore from checkpoint |
| `prepare_patch(files, desc)` | Generate diff for approval |
| `list_patches(status)` | List pending/approved/rejected |
| `apply_approved_patch(name)` | Apply after owner approval |
| `reject_patch(name, reason)` | Reject with reason |

### Patch Approval Workflow

All registry modifications generate patches for owner approval:

```
orchestration/patches/
├── pending/     # Awaiting approval
├── approved/    # Applied patches (audit trail)
└── rejected/    # Rejected with reason
```

### Running Tests

```bash
# Procedure registry tests (25 tests)
python -m pytest tests/unit/test_procedure_registry.py -v
```

---

## REPL Memory (Episodic Learning)

The `repl_memory/` directory contains the MemRL episodic memory system for learning from REPL tool usage patterns.

### Files

| File | Purpose |
|------|---------|
| `repl_memory/seed_examples.json` | 56 canonical REPL tool usage examples |
| `repl_memory/seed_loader.py` | Script to load seeds into episodic memory |
| `repl_memory/episodic_store.py` | SQLite + numpy memory storage |
| `repl_memory/embedder.py` | Task embedding via Qwen2.5-0.5B |
| `repl_memory/retriever.py` | Two-phase retrieval + hybrid router |

### Seeding Episodic Memory

```bash
# First-time setup (load 56 canonical examples)
python orchestration/repl_memory/seed_loader.py

# Force reload (clear + reload)
python orchestration/repl_memory/seed_loader.py --force
```

### Category Coverage

| Category | Count | Tools |
|----------|-------|-------|
| filesystem | 8 | `list_dir`, `file_info`, `peek` |
| procedure | 8 | `run_procedure`, `list_procedures` |
| document | 6 | `ocr_document`, `extract_figure` |
| complex | 7 | Multi-step with `llm_call` |
| shell | 5 | git, ls, find |
| simple | 4 | Direct calculations |
| search | 3 | `grep` patterns |
| vision | 3 | `analyze_figure` |
| web | 3 | `web_fetch` |
| artifacts | 3 | Store/retrieve values |
| memory | 2 | `recall` past tasks |
| escalation | 2 | `escalate` to architect |
| parallel | 2 | `llm_batch` operations |

### Checking Memory Stats

```bash
python3 -c "from orchestration.repl_memory import EpisodicStore; print(EpisodicStore().get_stats())"
```

## Dependencies

```bash
pip install jsonschema>=4.20
```

Or add to `requirements-dev.txt`.

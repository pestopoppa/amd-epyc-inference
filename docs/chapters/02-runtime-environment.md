# Chapter 02: Runtime Environment & Configuration

## Introduction

The orchestration system's runtime environment provides a hierarchical configuration system with feature flags, environment-based overrides, and safety guardrails. This chapter documents the environment setup, configuration architecture, and runtime tuning parameters that control system behavior.

The configuration system uses **pydantic-settings** for type-safe environment variable parsing with fallback to manual parsing when unavailable. Feature flags enable modular system components to be toggled independently for testing and production deployment.

## Feature Flag System

The system implements 10 independent feature flags defined in `src/features.py` that control optional orchestration modules. All flags default to **False** in test mode for isolation, **True** in production mode for full functionality.

### Core Feature Flags

| Flag | Environment Variable | Purpose | Dependencies |
|------|---------------------|---------|--------------|
| `memrl` | `ORCHESTRATOR_MEMRL` | Memory-based RL (TaskEmbedder, QScorer) | numpy, sqlite3, sentence-transformers |
| `tools` | `ORCHESTRATOR_TOOLS` | TOOL() function in REPL | None |
| `scripts` | `ORCHESTRATOR_SCRIPTS` | SCRIPT() function for prepared scripts | `tools` |
| `streaming` | `ORCHESTRATOR_STREAMING` | SSE /chat/stream endpoint | None |
| `openai_compat` | `ORCHESTRATOR_OPENAI_COMPAT` | OpenAI-compatible /v1/* endpoints | None |
| `repl` | `ORCHESTRATOR_REPL` | Python REPL execution environment | None |
| `caching` | `ORCHESTRATOR_CACHING` | LLM response caching | None |
| `restricted_python` | `ORCHESTRATOR_RESTRICTED_PYTHON` | Use RestrictedPython sandbox | RestrictedPython>=7.0 |
| `generation_monitor` | `ORCHESTRATOR_GENERATION_MONITOR` | Early failure detection (Phase 6) | None |
| `mock_mode` | `ORCHESTRATOR_MOCK_MODE` | Mock responses (test safety) | None |

### Usage Pattern

```python
from src.features import features

# Check if a feature is enabled
if features().memrl:
    from orchestration.repl_memory import TaskEmbedder
    embedder = TaskEmbedder()

# Get feature summary
enabled = features().enabled_features()
# Returns: ['repl', 'tools', 'caching', 'mock_mode']
```

### Validation

Feature dependencies are checked at initialization:

```python
from src.features import get_features

features = get_features(production=True)
errors = features.validate()

if errors:
    # ['scripts feature requires tools feature']
    raise RuntimeError(f"Invalid configuration: {errors}")
```

## Hierarchical Configuration

The configuration system in `src/config.py` provides nested settings with environment variable support via **double-underscore nesting**:

```bash
# Top-level settings
ORCHESTRATOR_MOCK_MODE=0
ORCHESTRATOR_DEBUG=1

# Nested LLM settings
ORCHESTRATOR_LLM__OUTPUT_CAP=4096
ORCHESTRATOR_LLM__BATCH_PARALLELISM=8
ORCHESTRATOR_LLM__CALL_TIMEOUT=300

# Nested escalation settings
ORCHESTRATOR_ESCALATION__MAX_RETRIES=3
ORCHESTRATOR_ESCALATION__MAX_ESCALATIONS=2

# Nested REPL settings
ORCHESTRATOR_REPL__MAX_OUTPUT_LEN=10000
ORCHESTRATOR_REPL__TIMEOUT_SECONDS=30
```

### Configuration Hierarchy

```
OrchestratorConfig
├── mock_mode: bool = True
├── debug: bool = False
├── llm: LLMConfig
│   ├── output_cap: int = 8192
│   ├── batch_parallelism: int = 4
│   ├── call_timeout: int = 120
│   ├── max_recursion_depth: int = 5
│   ├── default_prompt_rate: float = 0.50
│   └── default_completion_rate: float = 1.50
├── escalation: EscalationConfig
│   ├── max_retries: int = 2
│   ├── max_escalations: int = 2
│   └── optional_gates: frozenset = {"typecheck", "integration", "shellcheck"}
├── repl: REPLConfig
│   ├── max_output_len: int = 10000
│   ├── timeout_seconds: int = 30
│   ├── forbidden_modules: frozenset = {"os", "sys", "subprocess", ...}
│   └── forbidden_builtins: frozenset = {"eval", "exec", "open", ...}
├── server: ServerConfig
│   ├── default_url: str = "http://localhost:8080"
│   ├── timeout: int = 300
│   ├── num_slots: int = 4
│   ├── connect_timeout: int = 5
│   ├── retry_count: int = 3
│   └── retry_backoff: float = 0.5
├── monitor: MonitorConfig
│   ├── entropy_threshold: float = 2.5
│   ├── repetition_window: int = 50
│   ├── repetition_threshold: float = 0.3
│   └── min_tokens_before_abort: int = 20
├── paths: PathsConfig
│   ├── models_dir: Path = /mnt/raid0/llm/models
│   ├── cache_dir: Path = /mnt/raid0/llm/cache
│   ├── tmp_dir: Path = /mnt/raid0/llm/tmp
│   └── registry_path: Path = .../model_registry.yaml
└── features: FeaturesConfig
    ├── memrl: bool = False
    ├── tools: bool = False
    └── ... (10 feature flags)
```

### Loading Configuration

```python
from src.config import get_config

# Load from environment (cached)
config = get_config()

# Access nested settings
max_output = config.llm.output_cap
max_retries = config.escalation.max_retries
repl_timeout = config.repl.timeout_seconds

# Reset cache if env changes
from src.config import reset_config
reset_config()
```

## Environment Variables

All LLM-related files MUST reside on `/mnt/raid0/` to prevent root filesystem exhaustion. These variables redirect all caches and temporary files:

```bash
# HuggingFace/Transformers caches
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/raid0/llm/cache/huggingface
export HF_DATASETS_CACHE=/mnt/raid0/llm/cache/huggingface/datasets

# Python package cache
export PIP_CACHE_DIR=/mnt/raid0/llm/cache/pip

# System temporary files
export TMPDIR=/mnt/raid0/llm/tmp

# XDG Base Directory Specification
export XDG_CACHE_HOME=/mnt/raid0/llm/claude/cache
export XDG_DATA_HOME=/mnt/raid0/llm/claude/share
export XDG_STATE_HOME=/mnt/raid0/llm/claude/state
```

### Path Verification (Mandatory)

Before any file write operation:

```bash
# Verify path starts with /mnt/raid0/
[[ "$TARGET_PATH" == /mnt/raid0/* ]] || { echo "ERROR: Path not on RAID!"; exit 1; }
```

**Forbidden paths**: `/home/`, `/tmp/` (except via bind mount), `/var/`, `~/.cache/`, any path not starting with `/mnt/raid0/`.

## OMP & NUMA Runtime Tuning

llama.cpp inference performance depends on thread binding and memory interleaving. Standard prefix for all inference commands:

```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m model.gguf -t 96 -p "prompt"
```

### Thread Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `OMP_NUM_THREADS` | 1 | Disable OpenMP parallelism (llama.cpp handles threading) |
| `-t` | 96 | Use all physical cores (96-core EPYC 9655) |
| `--interleave=all` | Required | Interleave memory across 12 DDR5 channels (~460 GB/s) |

### NUMA Architecture

The EPYC 9655 has **12 memory channels** (1.13TB DDR5-5600 ECC). Using `--interleave=all` ensures memory bandwidth is maximized across all channels rather than binding to a single NUMA node.

**Do NOT use** `numactl --cpunodebind=0` or similar node-specific bindings for large models - this restricts memory bandwidth to 2-4 channels and cuts throughput by 60-75%.

## Python Environment

The system uses **uv** (fast Python package installer) with a dedicated environment:

```bash
# Environment name
pace-env

# Activation
source /mnt/raid0/llm/pace-env/bin/activate

# Key packages
# - FastAPI + uvicorn (API server)
# - pydantic + pydantic-settings (config/validation)
# - httpx (async HTTP client)
# - numpy, sentence-transformers (MemRL)
# - RestrictedPython (REPL sandbox)
```

### Installation

```bash
# Create environment with uv
uv venv /mnt/raid0/llm/pace-env

# Install dependencies
uv pip install -r /mnt/raid0/llm/claude/requirements.txt
```

## Session Initialization

Every session MUST run the initialization script to verify environment, discover models, and check branch safety:

```bash
# Set environment variables
source /mnt/raid0/llm/claude/scripts/utils/agent_log.sh
agent_session_start "Session purpose"

# Discover models and verify llama.cpp branch
bash /mnt/raid0/llm/claude/scripts/session/session_init.sh
```

The initialization script:
1. Checks llama.cpp is on `production-consolidated` branch
2. Scans `/mnt/raid0/llm/models/` for GGUF files
3. Validates model registry against discovered models
4. Checks free memory (100GB minimum for tests)
5. Verifies environment variables point to `/mnt/raid0/`

## References

- `src/features.py` - Feature flag system implementation
- `src/config.py` - Hierarchical configuration with pydantic-settings
- `scripts/session/session_init.sh` - Environment initialization
- `scripts/utils/agent_log.sh` - Session logging utilities

---

*Previous: [Chapter 01: Hardware System](01-hardware-system.md)* | *Next: [Chapter 03: llama.cpp Toolchain](03-llama-cpp-toolchain.md)*

# Chapter 04: Storage Architecture & Safety

## Introduction

The system's storage architecture enforces a strict separation between the **120GB OS SSD** (root filesystem) and **4TB RAID0 NVMe array** (models, caches, temporary files). This chapter documents the storage rules, the 192-thread pytest danger that exhausted 1.13TB RAM, and the HOT/WARM/COLD memory pool design.

Violating the storage rules causes **system instability and crashes**. Writing large files to root triggers paging storms that freeze the machine and corrupt the OS. The environment variables and path verification checks in this chapter are **non-negotiable safety requirements**.

## Root Filesystem Crisis & Recovery

### What Happened (2025-12-18)

Claude Code filled `/tmp/claude` with 20GB of data, exhausting the 84GB root filesystem and crashing the system. The application creates `/tmp/claude` before AI prompt instructions are evaluated, bypassing the "write only to `/mnt/raid0/`" constraint.

**Root Cause**: Application-level cache directory creation happens before runtime configuration.

### Three-Layer Defense

**Layer 1: Bind Mount** (redirects `/tmp/claude` → `/mnt/raid0/llm/tmp/claude`)

```bash
# Create bind mount at session start
sudo mkdir -p /mnt/raid0/llm/tmp/claude
sudo mount --bind /mnt/raid0/llm/tmp/claude /tmp/claude

# Verify mount is active
mountpoint /tmp/claude
# Output: /tmp/claude is a mountpoint
```

The bind mount makes `/tmp/claude` a "portal" to the RAID array. Writes to `/tmp/claude` physically go to `/mnt/raid0/llm/tmp/claude`, preventing root FS exhaustion.

**Layer 2: Real-Time Monitoring** (`monitor_storage.sh`)

```bash
# Run in background during sessions
bash /mnt/raid0/llm/UTILS/monitor_storage.sh &

# Alerts:
# - 70% full: Warning logged
# - 85% full: Critical alert + system notification
```

**Layer 3: Emergency Recovery** (`emergency_cleanup.sh`)

```bash
# If system fills up:
sudo bash /mnt/raid0/llm/UTILS/emergency_cleanup.sh

# Actions:
# 1. Stop Claude processes
# 2. Unmount bind mount
# 3. Delete /tmp/claude
# 4. Report before/after usage
```

### Allowed vs Forbidden Paths

| ✅ ALLOWED (RAID Array) | ❌ FORBIDDEN (Root FS) |
|-------------------------|------------------------|
| `/mnt/raid0/llm/` | `/home/` (except symlinks) |
| `/mnt/raid0/llm/claude/` | `/tmp/` (except via bind mount) |
| `/mnt/raid0/llm/cache/` | `/var/` |
| `/mnt/raid0/llm/models/` | `~/.cache/` |
| `/mnt/raid0/llm/tmp/` | `~/.local/` |

**Mandatory Path Verification**:

```bash
# Before ANY file write operation
[[ "$TARGET_PATH" == /mnt/raid0/* ]] || { echo "ERROR: Path not on RAID!"; exit 1; }
```

## Storage Layout

### RAID0 NVMe Array (4TB)

| Directory | Purpose | Typical Size |
|-----------|---------|--------------|
| `/mnt/raid0/llm/models/` | GGUF quantized models | 2.1TB (90 models) |
| `/mnt/raid0/llm/hf/` | HuggingFace format models | 850GB (source models) |
| `/mnt/raid0/llm/cache/` | HF/pip caches | 120GB |
| `/mnt/raid0/llm/tmp/` | Temporary files (TMPDIR) | 50GB (cleaned daily) |
| `/mnt/raid0/llm/claude/` | Project docs & scripts | 8GB |
| `/mnt/raid0/llm/llama.cpp/` | Production toolchain | 2GB |
| `/mnt/raid0/llm/llama.cpp-experimental/` | Experimental worktree | 2GB |

**RAID Configuration**: 2× Solidigm P44 Pro 2TB NVMe in RAID0 (stripe size 64KB)

**Performance**:
- Sequential read: 12.5 GB/s
- Sequential write: 11.8 GB/s
- Random 4K read: 680K IOPS
- Random 4K write: 550K IOPS

### OS SSD (120GB)

**Used for**: OS, system packages, logs
**Free space required**: 30GB minimum (25% of capacity)
**No large files allowed**: Models, caches, and temporary files are forbidden

## 192-Thread Pytest Danger

### Memory Exhaustion Incident (2026-01-13)

An agent ran orchestration liveness tests with `pytest -n auto`, spawning ~192 worker processes (one per hardware thread). Each worker initialized the API, which loads:
- TaskEmbedder: 0.5B embedding model (~2GB)
- QScorer: Q-value scoring model (~1GB)

**Result**: 192 workers × 3GB = **576GB allocation**, exceeding the 1.13TB RAM budget when combined with existing HOT tier (~535GB). The machine exhausted memory and crashed.

### Safeguards Implemented

**1. Lazy MemRL Loading** (`src/api.py`)

TaskEmbedder and QScorer only load when `real_mode=True`:

```python
from src.features import features

if features().memrl and not features().mock_mode:
    # Load MemRL components only in production with real_mode=True
    embedder = TaskEmbedder()
    qscorer = QScorer()
else:
    # Tests use mock mode - no model loading
    embedder = None
    qscorer = None
```

**2. Memory Guard** (`tests/conftest.py`)

Tests fail early if < 100GB free RAM:

```python
import psutil

def pytest_configure(config):
    mem = psutil.virtual_memory()
    free_gb = mem.available / (1024**3)

    if free_gb < 100:
        raise RuntimeError(
            f"Insufficient memory for tests: {free_gb:.1f}GB free, need 100GB"
        )
```

**3. Makefile Check** (`make check-memory`)

Run before `test-all` to verify memory:

```makefile
check-memory:
    @python3 -c "import psutil; m = psutil.virtual_memory(); \
        assert m.available > 100*(1024**3), \
        f'Need 100GB free, have {m.available/(1024**3):.1f}GB'"
```

### Safe Test Commands

```bash
# ✅ Safe: Sequential execution
pytest tests/

# ✅ Safe: Limited parallelism (max 4 workers)
pytest tests/ -n 4

# ❌ DANGEROUS: Spawns ~192 workers!
pytest tests/ -n auto  # DO NOT USE
```

**Rule**: NEVER use `pytest -n auto` on this 192-thread machine. Limit to `-n 4` maximum.

## HOT/WARM/COLD Memory Architecture

The orchestrator uses a three-tier memory pool design optimized for the 1.13TB RAM capacity:

### HOT Tier (~535GB = 47% of RAM)

Always resident in memory, loaded at startup:

| Port | Role | Model | Size | Speed |
|------|------|-------|------|-------|
| 8080 | frontdoor, coder_primary | Qwen3-Coder-30B-A3B-Q4_K_M | ~17GB | 18 t/s |
| 8081 | coder_escalation | Qwen2.5-Coder-32B-Q4_K_M | ~19GB | 39 t/s (spec) |
| 8082 | worker_explore | Qwen2.5-7B-Instruct-f16 | ~14GB | 44 t/s (spec) |
| 8084 | architect_coding | Qwen3-Coder-480B-A35B-Q4_K_M | ~280GB | 10.3 t/s |
| 8086 | worker_vision | Qwen2.5-VL-7B-Q4_K_M | ~4GB | ~15 t/s |
| 8090-8095 | embedder (6x) | BGE-large-en-v1.5-F16 | ~4GB | probe-first |
| 9001 | document_formalizer | LightOnOCR-2-1B | ~2GB | 19x PDF speedup |

**Draft models** (shared by spec decode): ~0.5GB each

**Total HOT**: ~535GB (includes OS, buffers, KV caches)

### WARM Tier (~460GB, load on demand)

Models loaded via mmap when needed for specific tasks:

| Role | Model | Size | When Loaded |
|------|-------|------|-------------|
| architect_general | Qwen3-235B-A22B-Q4_K_M | ~140GB | Escalation to B3 tier |
| ingest_long_context | Qwen3-Next-80B-A3B-Q4_K_M | ~45GB | Long-context synthesis |
| vision_escalation | Qwen3-VL-30B-A3B-Q4_K_M | ~17GB | Complex vision tasks |

**Loading**: mmap() with `--no-mmap false` allows on-demand paging from NVMe (~12GB/s sequential read).

**Eviction**: Automatic via OS page cache when memory pressure increases.

### COLD Tier (Disk Only)

Models on disk, not loaded into memory:

- Benchmark test models
- Deprecated models
- Alternative quantizations (Q2_K, Q3_K_M, etc.)

**Total COLD**: ~1.5TB on `/mnt/raid0/llm/models/`

### Memory Budget Example

```
HOT tier (always loaded):     535GB (47%)
WARM tier (on-demand mmap):   460GB (41%)
OS + buffers + headroom:      135GB (12%)
Total capacity:              1130GB (100%)
```

**Safe margin**: Keep 100GB+ free for KV caches, tensor operations, and tests.

## Storage Monitoring

### Daily Cleanup

```bash
# Clean old temporary files (>24h)
find /mnt/raid0/llm/tmp/ -type f -mtime +1 -delete

# Clean old extraction directories
python3 -c "from src.services.archive_extractor import ArchiveExtractor; \
    ArchiveExtractor.cleanup_expired(max_age_hours=24)"

# Clean pytest cache
find /mnt/raid0/llm/claude -name ".pytest_cache" -type d -exec rm -rf {} +
```

### Health Check

```bash
# Check root FS usage
df -h /
# Must be < 70% (< 84GB used)

# Check RAID array usage
df -h /mnt/raid0
# Should have > 500GB free

# Check memory usage
free -h
# Should have > 100GB available
```

## References

- `docs/deprecated/RECOVERY_ACTION_PLAN.md` - Full incident analysis
- `research/ESCALATION_FLOW.md` - HOT/WARM/COLD memory architecture
- `tests/conftest.py` - Memory guard implementation
- `src/api.py` - Lazy MemRL loading

---

*Previous: [Chapter 03: llama.cpp Toolchain](03-llama-cpp-toolchain.md)* | *Next: [Chapter 05: Speculative Decoding](05-speculative-decoding.md)*

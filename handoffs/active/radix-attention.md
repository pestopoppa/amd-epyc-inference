# RadixAttention-Style Prefix KV Reuse - YOLO Agent Handoff

**Status**: ✅ VERIFIED (2026-01-09) - 80% cache hit rate, 110.5 t/s avg
**Target**: Autonomous agent in devcontainer
**Scope**: Full implementation (Phases A-E)
**Test Model**: Qwen2.5-Coder-0.5B-Q8_0

---

## Implementation Complete

| Phase | File | Lines | Status |
|-------|------|-------|--------|
| A | `src/backends/llama_server.py` | 477 | ✅ |
| B | `src/prefix_cache.py` (PrefixRouter) | 584 | ✅ |
| C | `src/prefix_cache.py` (canonicalize_prompt) | — | ✅ |
| D | `src/radix_cache.py` | 482 | ✅ |
| E | `src/prefix_cache.py` (persistence) | — | ✅ |

**Tests**: 46/46 passing in `tests/unit/test_prefix_cache.py`

**Next Steps**: Integration testing with live llama-server

---

## Quick Start

```bash
# 1. Launch devcontainer with YOLO agent
export PATH="/mnt/raid0/llm/npm-global/bin:/mnt/raid0/llm/tools/devc/bin:$PATH"
devc /mnt/raid0/llm/claude

# 2. Inside container - start test model server
/mnt/raid0/llm/llama.cpp/build/bin/llama-server \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 4096 -np 4 \
  -t 16

# 3. Verify server is running
curl http://localhost:8080/health

# 4. Run existing tests to confirm environment
cd /mnt/raid0/llm/claude && python -m pytest tests/unit/ -v
```

---

## Context

**What you're building**: Middleware to exploit llama-server's prefix caching for the orchestrator's `llm_batch()` workloads.

**Why it matters**: RLM (Recursive Language Machine) execution creates many sub-calls with shared prefixes. Currently each call re-processes the full prefix. With caching, subsequent calls skip prefill for the shared portion.

**Expected Impact**: 40-60% reduction in prefill time for RLM workloads.

**This is RadixAttention from SGLang**, adapted for CPU inference via llama-server.

---

## Key Files to Understand First

| File | Purpose |
|------|---------|
| `src/llm_primitives.py` | Current `llm_call()` and `llm_batch()` |
| `src/model_server.py` | Backend abstraction (currently per-inference subprocess) |
| `orchestration/model_registry.yaml` | Role→model mapping |
| `/mnt/raid0/llm/llama.cpp/tools/server/README.md` | llama-server API docs |

---

## llama-server Caching API (Already Available)

```bash
# Completion with caching
curl http://localhost:8080/completion -d '{
  "prompt": "Summarize: ...",
  "n_predict": 256,
  "cache_prompt": true,
  "id_slot": 0
}'

# Save slot state
curl -X POST "http://localhost:8080/slots/0?action=save" -d '{
  "filename": "/tmp/slot0.bin"
}'

# Restore slot state
curl -X POST "http://localhost:8080/slots/0?action=restore" -d '{
  "filename": "/tmp/slot0.bin"
}'

# Get slot info (shows cache stats)
curl http://localhost:8080/slots
```

---

## Implementation Phases

### Phase A: Persistent Server Mode (Foundation)

**Goal**: Switch from per-inference subprocess to persistent llama-server.

**Create**: `src/backends/llama_server.py`
```python
class LlamaServerBackend(ModelBackend):
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url
        self.session = requests.Session()

    def infer(self, role_config: RoleConfig, request: InferenceRequest) -> InferenceResult:
        payload = {
            "prompt": request.prompt,
            "n_predict": request.n_tokens,
            "temperature": request.temperature,
            "cache_prompt": True,  # ENABLE CACHING
        }
        resp = self.session.post(f"{self.base_url}/completion", json=payload)
        return self._parse_response(resp.json())
```

**Test**: Second identical prompt should be faster than first.

---

### Phase B: Sticky Slot Routing

**Goal**: Route requests with same prefix to same slot for cache hits.

**Create**: `src/prefix_cache.py`
```python
class PrefixRouter:
    def __init__(self, num_slots: int = 4):
        self.prefix_to_slot: dict[str, int] = {}
        self.slot_lru: list[int] = []

    def get_slot_for_prompt(self, prompt: str) -> int:
        prefix_hash = self._hash_prefix(prompt, n_tokens=64)
        if prefix_hash in self.prefix_to_slot:
            return self.prefix_to_slot[prefix_hash]
        return self._allocate_slot(prefix_hash)

    def _hash_prefix(self, prompt: str, n_tokens: int) -> str:
        return hashlib.md5(prompt[:256].encode()).hexdigest()
```

**Test**: Same-prefix requests hit same slot.

---

### Phase C: Prompt Canonicalization

**Goal**: Normalize prompts to maximize cache hits.

**Modify**: `src/llm_primitives.py`
```python
def canonicalize_prompt(prompt: str) -> str:
    prompt = prompt.rstrip()
    prompt = prompt.replace('\r\n', '\n')
    prompt = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '[TIMESTAMP]', prompt)
    return prompt
```

**Test**: >50% cache hit rate on RLM workloads.

---

### Phase D: Radix Tree Cache Manager

**Goal**: True radix tree for optimal prefix matching.

**Create**: `src/radix_cache.py`
```python
class RadixNode:
    def __init__(self):
        self.children: dict[int, RadixNode] = {}
        self.slot_id: int | None = None
        self.last_access: float = 0

class RadixCache:
    def find_longest_prefix(self, tokens: list[int]) -> tuple[int, int | None]:
        node = self.root
        length = 0
        best_slot = None
        for token in tokens:
            if token not in node.children:
                break
            node = node.children[token]
            length += 1
            if node.slot_id is not None:
                best_slot = node.slot_id
        return length, best_slot
```

**Test**: >70% cache hit rate on complex workloads.

---

### Phase E: Slot State Persistence

**Goal**: Persist hot prefix caches across server restarts.

```python
async def save_hot_prefixes(self):
    for slot_id, prefix_hash in self.hot_prefixes[:10]:
        await self.client.post(
            f"{self.base_url}/slots/{slot_id}?action=save",
            json={"filename": f"/cache/slot_{prefix_hash}.bin"}
        )
```

**Test**: Cache survives server restart.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/backends/llama_server.py` | HTTP client for llama-server |
| `src/prefix_cache.py` | PrefixRouter and canonicalization |
| `src/radix_cache.py` | Radix tree cache manager (Phase D) |
| `scripts/server/start_servers.sh` | Server launch script |
| `tests/unit/test_prefix_cache.py` | Unit tests |
| `tests/integration/test_cache_hits.py` | Integration tests |

---

## Files to Modify

| File | Change |
|------|--------|
| `src/llm_primitives.py` | Add `cache_policy` param, use server backend |
| `src/model_server.py` | Add `LlamaServerBackend` |
| `orchestration/model_registry.yaml` | Add server URLs, slot counts |

---

## Don't Touch These Files

- `src/repl_environment.py` - REPL is separate from caching
- `src/dispatcher.py` - Task routing unchanged
- `src/generation_monitor.py` - Monitoring is orthogonal

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Cache hit rate | >50% Phase C, >70% Phase D | Log hits/misses in PrefixRouter |
| Prefill speedup | >40% on cached | Compare TTFT with/without cache |
| HTTP overhead | <5ms | Profile request latency |

---

## Quick Cache Validation

```python
import requests
import time

url = "http://localhost:8080/completion"
scaffold = "You are helpful. Summarize:\n\n"

# First call - cache miss
t0 = time.time()
requests.post(url, json={
    "prompt": scaffold + "Text 1",
    "n_predict": 50,
    "cache_prompt": True,
    "id_slot": 0
})
t1 = time.time()
print(f"First call: {t1-t0:.3f}s")

# Second call - should cache hit on scaffold
t0 = time.time()
requests.post(url, json={
    "prompt": scaffold + "Text 2",
    "n_predict": 50,
    "cache_prompt": True,
    "id_slot": 0
})
t1 = time.time()
print(f"Second call: {t1-t0:.3f}s (should be faster)")
```

---

## When Done

1. Run full test suite: `python -m pytest tests/ -v`
2. Run benchmark showing cache hit rate
3. Update `orchestration/model_registry.yaml` with server mode config
4. Create PR or commit with results

---

## References

- Full plan: `/home/daniele/.claude/plans/twinkly-sniffing-crescent.md`
- Orchestrator Implementation Plan: `research/Orchestrator_Implementation_Plan.md`
- llama-server docs: `/mnt/raid0/llm/llama.cpp/tools/server/README.md`
- SGLang RadixAttention: https://arxiv.org/abs/2312.07104

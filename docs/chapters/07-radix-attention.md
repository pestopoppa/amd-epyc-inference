# Chapter 07: RadixAttention Prefix Caching

## Introduction

RadixAttention is a prefix caching optimization adapted from SGLang for CPU inference. The orchestrator's recursive execution (RLM - Recursive Language Machine) creates many sub-calls with shared prefixes. Without caching, each call re-processes the entire prefix. With caching, subsequent calls skip prefill for shared portions.

**Expected Impact**: 40-60% reduction in prefill time for RLM workloads.

**Status**: Implementation complete (46/46 tests passing). Awaiting integration testing with live servers.

## The Problem

Consider an orchestrator flow:

```
TaskIR → Dispatcher → [System prompt + Task context + Step 1]
                   → [System prompt + Task context + Step 2]
                   → [System prompt + Task context + Step 3]
```

Each step shares "System prompt + Task context" but processes it fresh. For a 4K token shared prefix processed 10 times, we waste 36K tokens of redundant computation.

## The Solution

Cache KV states for common prefixes:

```
First call:  [System prompt + Task context] → Cache KV → [Step 1]
Second call: [Cache lookup: hit!] → [Step 2]
Third call:  [Cache lookup: hit!] → [Step 3]
```

## Implementation Architecture

### Components

| File | Purpose | Lines |
|------|---------|-------|
| `src/backends/llama_server.py` | Server backend abstraction | 477 |
| `src/prefix_cache.py` | PrefixRouter, canonicalize_prompt | 584 |
| `src/radix_cache.py` | Radix tree for prefix matching | 482 |

### Key Classes

**LlamaServerBackend**: Manages connection to llama-server, handles slot allocation and cache operations.

**PrefixRouter**: Routes prompts to appropriate cached slots, decides when to cache vs fresh compute.

**RadixCache**: Radix tree data structure for efficient longest-prefix matching.

## llama-server Caching API

llama-server already supports slot-based KV caching:

```bash
# Completion with caching enabled
curl http://localhost:8080/completion -d '{
  "prompt": "System: ...",
  "n_predict": 256,
  "cache_prompt": true,
  "id_slot": 0
}'

# Save slot state to disk
curl -X POST "http://localhost:8080/slots/0?action=save" \
  -d '{"filename": "/tmp/slot0.bin"}'

# Restore slot state
curl -X POST "http://localhost:8080/slots/0?action=restore" \
  -d '{"filename": "/tmp/slot0.bin"}'
```

Our middleware routes prompts to slots with matching cached prefixes.

## Canonicalization

Before prefix matching, prompts are canonicalized:
- Normalize whitespace
- Strip ephemeral content (timestamps, UUIDs)
- Hash to fixed-length key

This ensures semantically equivalent prompts match even with superficial differences.

## Configuration

```yaml
# In model_registry.yaml
prefix_cache:
  enabled: true
  prefix_length: 256          # Min prefix length to cache
  canonicalize: true          # Enable prompt canonicalization
  cache_dir: /mnt/raid0/llm/cache/prefix
```

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Cache hit rate | >50% | RLM workloads have high prefix reuse |
| Prefill speedup | 40-60% | Skipping cached prefix computation |
| Memory overhead | <5% | Radix tree is memory-efficient |

## Integration with Orchestrator

The prefix cache integrates with `llm_batch()`:

```python
async def llm_batch(prompts: List[str], model: str) -> List[str]:
    router = get_prefix_router(model)

    results = []
    for prompt in prompts:
        # Router finds best slot with matching prefix
        slot, cache_hit = router.route(prompt)

        response = await backend.complete(
            prompt=prompt,
            slot=slot,
            cache_prompt=True  # Enable caching for this slot
        )
        results.append(response)

    return results
```

## Test Coverage

All 46 unit tests passing:
- Radix tree operations (insert, lookup, delete)
- Canonicalization edge cases
- Slot routing logic
- Cache persistence (save/restore)

```bash
python -m pytest tests/unit/test_prefix_cache.py -v
# 46/46 passed
```

## Next Steps

1. Integration testing with live llama-server
2. Benchmark cache hit rates on real orchestrator workloads
3. Tune prefix_length threshold based on measurements
4. Add cache eviction policy for memory management

## References

- [SGLang RadixAttention Paper](https://arxiv.org/abs/2312.07104)
- handoffs/active/radix-attention.md
- llama.cpp server documentation

---

*Previous: [Chapter 06: Orchestration Architecture](06-orchestration-architecture.md)*
*Next: [Chapter 08: Deprecated Approaches](08-deprecated-approaches.md)*

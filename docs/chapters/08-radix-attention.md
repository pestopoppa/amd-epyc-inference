# Chapter 08: RadixAttention Prefix Caching

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
  prefix_length: 4096         # Min prefix length to cache (was 256)
  canonicalize: true          # Enable prompt canonicalization
  cache_dir: /mnt/raid0/llm/cache/prefix
```

### Prefix Length Expansion (February 2026)

`prefix_length` increased from 256 to 4096 in `model_registry.yaml`. The original 256-token threshold was conservative -- most orchestrator role prompts span 1000-5000 tokens, meaning the cache was already capturing them, but the longer threshold ensures multi-turn system context also qualifies.

All role prompts were audited for prefix stability: each places static system instructions first and variable content (task description, user query) last. This layout maximizes RadixAttention cache hit rates because the shared prefix is a contiguous block at the start of the prompt. Prompts that mixed static and variable content (2 of 11) were restructured.

This parallels Claude's prompt caching prefix stability requirements, where Anthropic recommends placing static content at the beginning of the prompt so that the cacheable prefix is as long as possible. The same principle applies here: a stable 4K prefix cached across 10 sub-calls saves ~36K tokens of redundant prefill computation.

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

### RadixAttention and Prefix Caching

1. Zheng, L., Yin, L., Xie, Z., Huang, J., Sun, C., Yu, C. H., ... & Stoica, I. (2024). *SGLang: Efficient Execution of Structured Language Model Programs*. arXiv preprint. https://arxiv.org/abs/2312.07104

2. Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., ... & Stoica, I. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. https://arxiv.org/abs/2309.06180

### KV Cache Optimization

3. Liu, Z., Wang, J., Dao, T., Zhou, T., Yuan, B., Song, Z., ... & Chen, B. (2024). *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time*. NeurIPS 2023. https://arxiv.org/abs/2305.17118

4. Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., ... & Stoica, I. (2024). *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. NeurIPS 2023. https://arxiv.org/abs/2306.14048

5. Ge, S., Zhang, Y., Liu, L., Zhang, M., Han, J., & Gao, J. (2024). *Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs*. ICLR 2024. https://arxiv.org/abs/2310.01801

### LLM Serving Systems

6. Yu, G. I., Jeong, J. S., Kim, G. W., Kim, S., & Chun, B. G. (2022). *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/yu

7. Agrawal, A., Shanbhag, A., Bhandare, A., Ng, S., Amiri, S. H., Narayanan, S., ... & Tao, Z. (2024). *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*. OSDI 2024. https://arxiv.org/abs/2403.02310

### Radix Tree Data Structures

8. Morrison, D. R. (1968). *PATRICIA—Practical Algorithm To Retrieve Information Coded in Alphanumeric*. Journal of the ACM, 15(4), 514-534. https://doi.org/10.1145/321479.321481

9. Leis, V., Kemper, A., & Neumann, T. (2013). *The Adaptive Radix Tree: ARTful Indexing for Main-Memory Databases*. ICDE 2013. https://db.in.tum.de/~leis/papers/ART.pdf

### Implementation Resources

10. llama.cpp Contributors. (2024). *llama-server: Slot Management and KV Cache Persistence*. GitHub Documentation. https://github.com/ggml-org/llama.cpp/tree/master/examples/server

11. SGLang Team. (2024). *SGLang: A Structured Generation Language*. GitHub Repository. https://github.com/sgl-project/sglang

---

*Previous: [Chapter 07: Prompt Lookup](07-prompt-lookup.md)* | *Next: [Chapter 09: Deprecated Approaches](09-deprecated-approaches.md)*

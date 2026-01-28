# Chapter 14: TOON Encoding

## Introduction

TOON (Token-Oriented Object Notation) is a compact JSON-compatible format that achieves **52.5% average token reduction** on structured data while maintaining lossless round-trip fidelity. Integrated into the orchestrator's REPL tools and prompt builders, it delivers **50.8% average TTFT improvement** — 10x the original 5% target. The improvement scales with model size: 8B models see 54.2% TTFT gains, and extrapolation to 235B+ architect models suggests 60-70%+.

TOON eliminates the redundancy inherent in JSON-serialized arrays of uniform objects — the dominant data shape in orchestration contexts (file listings, grep results, memory recall, procedure registries).

## Format Specification

### JSON vs TOON

**JSON (147 tokens)**:
```json
{
  "files": [
    {"name": "main.py", "type": "file", "size": 1234},
    {"name": "utils.py", "type": "file", "size": 567},
    {"name": "tests", "type": "dir", "size": null}
  ]
}
```

**TOON (~88 tokens, 40% reduction)**:
```
files[3]{name,type,size}:
  main.py,file,1234
  utils.py,file,567
  tests,dir,
```

### Syntax Elements

| Feature | Syntax | Purpose |
|---------|--------|---------|
| Length declaration | `[N]` | Validates array bounds |
| Field headers | `{field1,field2}` | Declared once, not per row |
| CSV rows | `val1,val2,val3` | Compact tabular data |
| YAML nesting | `key: value` | Non-uniform objects |
| Empty cells | trailing `,` | Implicit null |

### Why It Compresses

1. **Field names eliminated from rows** — declared once in the header instead of per-object
2. **Whitespace minimized** — CSV-style rows, no indentation
3. **Null values implicit** — empty cells instead of `"field": null`
4. **Structure preserved** — validators can check array bounds via `[N]`

## Implementation

### Core API

**Source**: `src/services/toon_encoder.py`

```python
# Check availability
is_available() -> bool

# Encode Python object to TOON (falls back to JSON if unavailable)
encode(data: Any, fallback_to_json: bool = True) -> str

# Decode TOON back to Python (lossless round-trip)
decode(toon_str: str) -> Any

# Heuristic: should this data use TOON?
should_use_toon(data: Any, min_array_size: int = 3) -> bool
```

### Specialized Encoders

Each encoder targets a specific orchestration data shape:

| Encoder | Input | Reduction | Integration |
|---------|-------|-----------|-------------|
| `encode_list_dir()` | File listings | **64.6%** | `_list_dir()` REPL tool |
| `encode_escalation_context()` | Failure context | **46.2%** | Escalation chain |
| `encode_procedures()` | Procedure registry | **~55%** | `_list_procedures()` REPL tool |
| `encode_memory_results()` | Episodic recall | **~55%** | `_recall()` REPL tool |
| `encode_grep_hits()` | Grep results | **-18.6%** | Disabled (Markdown better) |

### Design Patterns

**Lazy loading**: The `toon_format` module loads only on first use, avoiding startup cost.

**Graceful fallback**: All encoders produce JSON when `toon_format` is unavailable. Zero API changes for callers.

**Heuristic gating**: `should_use_toon()` returns `True` only for arrays with 3+ uniform objects, preventing overhead on small or non-uniform data.

## Performance Results

### Token Reduction by Scenario

| Scenario | JSON Tokens | TOON Tokens | Reduction |
|----------|-------------|-------------|-----------|
| File listing (20 files) | 229 | 73 | **68.1%** |
| Architect complex (20 errors) | — | — | **69.0%** |
| Procedure listing (10+) | — | — | **56.7%** |
| Memory results (10+) | — | — | **56.3%** |
| Grep results | 177 | 84 | **52.5%** |
| Escalation context | 91 | 49 | **46.2%** |

### TTFT Impact by Model Size

Multi-model validation across 5 orchestration scenarios:

| Scenario | Token Reduction | 0.5B TTFT | 8B TTFT |
|----------|-----------------|-----------|---------|
| frontdoor_routing | 61.1% | +56.6% | +60.4% |
| coder_escalation | 51.9% | +46.4% | +49.5% |
| architect_complex | 69.0% | +58.7% | **+68.7%** |
| long_context_ingest | 54.4% | +48.2% | +56.5% |
| worker_batch | 25.9% | +27.4% | +35.9% |
| **Average** | **52.5%** | **47.5%** | **54.2%** |

TTFT improvement scales with model size — larger models benefit more from fewer input tokens because their per-token processing cost is higher.

### When TOON Excels vs Falls Short

| Excels | Falls Short |
|--------|-------------|
| Uniform arrays of objects | Deeply nested non-uniform structures |
| File listings, procedure lists | Highly variable schemas |
| Structured tool outputs | Semi-uniform data (<40% tabular) |
| Repeated field patterns | Single records or pure prose |
| Arrays with 3+ items | Grep results (Markdown is better) |

## Success Criteria

| Metric | Target | Kill Threshold | Actual | Status |
|--------|--------|----------------|--------|--------|
| TTFT improvement | >5% | <2% | **50.8%** | 10x target |
| Token reduction | >30% | <15% | **52.5%** | 1.75x target |
| Accuracy regression | <1% | >3% | **0%** | No regression |
| Unit test pass rate | >95% | <80% | **98%** | 51 tests |

## Test Coverage

**Unit tests**: `tests/unit/test_toon_encoder.py` — 51 tests at 98% pass rate:

| Suite | Tests | Coverage |
|-------|-------|----------|
| File listings | 13 | 28.6-69.4% reduction |
| Escalation context | 6 | 17.9-48.1% reduction |
| Procedures | 6 | 36.4-56.7% reduction |
| Memory results | 7 | 24.1-56.3% reduction |
| Edge cases | 11 | Unicode, nulls, special chars |
| Orchestration scenarios | 4 | 57.9-70.2% reduction |
| Round-trip validation | 4 | Lossless fidelity confirmed |

**Comprehensive suite**: `scripts/toon/comprehensive_toon_test.py` — ~120 test cases covering edge cases, unicode, non-uniform detection, and live TTFT measurement.

## References

### Project Files

- Source: `src/services/toon_encoder.py`
- Unit tests: `tests/unit/test_toon_encoder.py`
- Benchmark results: `benchmarks/results/ttft_toon_results.json`
- Comprehensive tests: `scripts/toon/comprehensive_toon_test.py`
- TTFT benchmark: `scripts/benchmark/ttft_toon_benchmark.py`
- Handoff: `handoffs/active/toon_format_integration.md`

### External

1. TOON Format Specification: https://github.com/toon-format/spec
2. Python Implementation: https://github.com/toon-format/toon (MIT license)

---

*Previous: [Chapter 13: Data Processing Pipelines](13-data-processing-pipelines.md)* | *Next: [Chapter 15: MemRL System](15-memrl-system.md)*

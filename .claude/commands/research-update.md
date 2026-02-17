# Research Summary Maintenance

## When Adding New Benchmark Results

1. **Add to summary.csv** with Claude-as-Judge scores
2. **Add to docs/reference/benchmarks/RESULTS.md** in the Complete Claude Score Table (line ~440)
3. **If optimization tested:** Add notes about baseline vs optimized speeds
4. **Update other relevant tables** (MoE, spec decode, etc.) if applicable

## Speed Reporting Rules

| Column | Meaning |
|--------|---------|
| `Baseline t/s` | Raw speed during benchmark (no optimization) |
| `Optimized t/s` | Speed with best optimization for that model |

## Optimization Configuration Tracking

When MoE reduction or other quality-affecting optimizations are tested:
- Document baseline AND optimized configurations separately
- Note quality impact (e.g., "2 experts = garbage" vs "4 experts = quality preserved")
- If optimization degrades quality, mark it explicitly

Example entry in docs/reference/benchmarks/RESULTS.md optimization tables:
```
| Model | Baseline | 4 Experts | 2 Experts | Quality Notes |
| Qwen3-235B | 3.6 t/s | 6.75 t/s | 3.80 t/s | 4 experts OK, 2 experts garbage |
```

## Keeping Tables in Sync

The docs/reference/benchmarks/RESULTS.md has multiple tables that may need updates:
- **Complete Claude Score Table** - ALL models with scores
- **Top Performers** - Summary of role recommendations
- **MoE Optimization Results** - Expert reduction benchmarks
- **Speculative Decoding Results** - Draft model combinations
- **Per-model Performance** - Baseline speeds

When benchmarking a new model, check if it belongs in any of these tables.

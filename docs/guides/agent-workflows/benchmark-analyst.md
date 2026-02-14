# Benchmark Analyst Workflow

## Baseline Procedure

1. Define controls (model, prompt set, threads, NUMA mode).
2. Run benchmark with explicit configuration capture.
3. Repeat runs to reduce noise.
4. Compare against baseline and prior best.
5. Publish decision-grade summary with caveats.

## Required Metrics

- Decode throughput (`TG t/s`)
- Prefill throughput (`PP t/s`)
- Acceptance rate (when speculative decode is enabled)
- Variance across repeated runs

## Reporting Standard

- Separate observations from inferences.
- Flag suspicious anomalies before making recommendations.
- Include exact commands and source result paths.

# Tool Chaining Patterns

This reference summarizes practical chain patterns for structured REPL mode.

## Config

- `ORCHESTRATOR_TOOL_CHAIN_MODE=seq`
  - Structured mode allows multi-tool turns but executes sequentially.
- `ORCHESTRATOR_TOOL_CHAIN_MODE=dep`
  - Dependency-aware wave execution.
  - Falls back to sequential execution when analysis is ambiguous.
- `ORCHESTRATOR_TOOL_CHAIN_PARALLEL_MUTATIONS=1`
  - In `dep` mode, allows parallel mutation waves for a conservative safe set:
    - `run_shell`, `file_write_safe`, `log_append`, `benchmark_run`

## Patterns

### Independent read tools in one turn

```python
a = peek(200)
b = grep("timeout")
c = list_dir("/mnt/raid0/llm/claude")
```

- In `dep` mode, executes as one parallel read wave.

### Read dependency chain

```python
a = peek(200)
b = grep(a)
```

- `b` depends on `a`; executes in separate waves.

### Mixed read + mutation

```python
a = grep("target.py")
b = run_shell("python3 -m pytest tests/unit/test_tool_chaining.py")
```

- Read wave can run first; mutation executes after dependency constraints.

### TOOL wrapper chain (registry tools)

```python
r1 = TOOL("search_wikipedia", query="Speculative decoding")
r2 = TOOL("get_wikipedia_article", title="Speculative decoding")
```

- Wrapped tool must be chain-enabled via `allowed_callers: ["chain"]`.

## Response Diagnostics (`tool_chains`)

`ChatResponse.tool_chains` includes per-chain diagnostics:

- `chain_id`
- `caller_type`
- `tools`
- `elapsed_ms`
- `success`
- `mode_requested`
- `mode_used`
- `fallback_to_seq`
- `parallel_mutations_enabled`
- `waves`
- `steps`

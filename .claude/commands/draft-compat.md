# Draft-Target Compatibility Validation

**Before adding ANY draft-target pairing to `model_registry.yaml`:**

```bash
python3 scripts/utils/check_draft_compatibility.py DRAFT.gguf TARGET.gguf
```

Checks: vocab size match, BOS/EOS token match, tokenizer model/pre match.

**Known pitfall**: SWA (Sliding Window Attention) models crash with spec decode (`llama_kv_cache::slot_info`). The script can't detect this — always test before adding to registry.

## Workflow

1. Run the check
2. If warnings: `llama-speculative -m TARGET -md DRAFT -p "test" -n 50`
3. If SIGSEGV or garbage → document in `runtime_quirks`, do NOT add pairing
4. If works → add to registry with `benchmark_date`

# Draft-Target Compatibility Validation

**Before adding ANY draft-target pairing to `model_registry.yaml`, run the compatibility check:**

```bash
python3 scripts/utils/check_draft_compatibility.py DRAFT.gguf TARGET.gguf
```

## What It Checks

1. **Vocab size match** - If draft has fewer tokens than target, spec decode may SIGSEGV when target generates a token ID the draft can't handle
2. **BOS/EOS token match** - Mismatch causes generation failures or garbage output
3. **Tokenizer model/pre match** - Different tokenizer families are usually incompatible

## Known Cases

| Draft | Target | Vocab Diff | Result |
|-------|--------|------------|--------|
| Gemma-3-1B-IT | Gemma-3-27B-QAT | 64 fewer | **std::bad_alloc crash** (SWA incompatibility, NOT vocab) |
| Qwen2.5-Coder-0.5B | Qwen2.5-Coder-32B | 128 fewer | **Works (11x speedup)** |

**Important**: The Gemma-3 crash is due to **Sliding Window Attention (SWA)** being incompatible with speculative decoding in llama.cpp, NOT vocab mismatch. The crash happens in `llama_kv_cache::slot_info` during KV cache initialization.

The script warns about vocab mismatch but can't detect SWA incompatibility. Always test before adding to registry.

## Workflow

1. **Run the check** before adding to registry
2. **If warnings**, run a test generation: `llama-speculative -m TARGET -md DRAFT -p "test" -n 50`
3. **If SIGSEGV or garbage**, do NOT add the pairing - document in `runtime_quirks`
4. **If works**, add to registry with `benchmark_date` and test results

## Example Output

```
Draft Model: gemma-3-1b-it-Q8_0.gguf
  vocab_size: 262,144
  bos_token_id: 2
  tokenizer_model: 108

Target Model: gemma-3-27B-it-QAT-Q4_0.gguf
  vocab_size: 262,208
  bos_token_id: 2
  tokenizer_model: 108

============================================================
RESULT: COMPATIBLE with warnings:
  VOCAB MISMATCH: draft=262,144, target=262,208 (64 fewer tokens in draft) - TESTING REQUIRED!
```

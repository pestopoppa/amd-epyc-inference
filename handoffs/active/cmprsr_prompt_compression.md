# Prompt Compression - Handoff Document

**Goal**: Add token compression layer to document pipeline, reducing tokens before Stage 1 (cursory summarization)

**Status**: BLOCKED - LLMLingua-2 causes quality regression. Waiting for Cmprsr weights.

**Priority**: LOW - Two-stage pipeline works well without compression

**Last Updated**: 2026-01-27

---

## ⚠️ EXPERIMENTAL FINDINGS (2026-01-27)

**LLMLingua-2 was implemented and tested. Results: QUALITY REGRESSION.**

### Test Setup
- Document: Twyne V1 Whitepaper (87K chars, ~22K tokens)
- Compression: LLMLingua-2 with target_ratio=0.5
- Pipeline: Stage 0 (compress) → Stage 1 (draft) → Stage 2 (review)

### Results Comparison

| Metric | Without Compression | With Compression |
|--------|---------------------|------------------|
| Time | 74s | 140s (**slower**) |
| Output quality | Clean, professional | Degraded |
| Hallucinations | None | Fake citations `[1]` |
| Typos | None | "tokenizeize", "cUSTODY" |
| Prompt leakage | None | Present |

### Root Cause
LLMLingua-2's **extractive** compression produces choppy, fragmented text:
```
Twyne Whitepaper Jakub Warmuz noncustodial risk modular credit delegation
protocol capital inefficiencies DeFi lending markets conservative passive...
```

This confuses downstream LLMs, causing them to hallucinate to fill semantic gaps.

### Conclusion
**DO NOT USE LLMLingua-2 for this pipeline.** Wait for Cmprsr (abstractive compression) weights.

### Files Created (kept for future use)
- `src/services/prompt_compressor.py` - LLMLingua-2 wrapper (disabled)
- `tests/unit/test_prompt_compressor.py` - Unit tests (12 passing)
- Config in `orchestration/model_registry.yaml` (compression.enabled=false)

---

**Sources**:
- Cmprsr: https://arxiv.org/abs/2511.12281
- LLMLingua: https://github.com/microsoft/LLMLingua

---

## Resume Command

```bash
# Install dependency
pip install llmlingua

# Test compression works
python -c "
from llmlingua import PromptCompressor
compressor = PromptCompressor(model_name='microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank')
result = compressor.compress_prompt('Your long text here. This is a test of the compression system.', target_token=10)
print(f'Compressed: {result[\"compressed_prompt\"]}')
"

# Read full implementation plan
cat /workspace/handoffs/active/cmprsr_prompt_compression.md
```

---

## Research Summary

### Cmprsr (Preferred - Unavailable)

**What it is**: Fine-tuned Qwen3-4B for abstractive prompt compression
- **Method**: SFT + GRPO (Group Relative Policy Optimization)
- **Key feature**: Accurately follows requested compression rates
- **Output**: Clean paraphrased text (abstractive)
- **Status**: ❌ Weights NOT publicly available (paper: Nov 2025)

**Why it's better**:
- Abstractive = rewrites text cleanly
- Native llama.cpp support (decoder model)
- Higher quality output

### LLMLingua-2 (Alternative - Available)

**What it is**: BERT encoder for token classification (keep/drop)
- **Method**: Distillation from GPT-4 on compression decisions
- **Key feature**: Fast (3-6x faster than v1), ~10-50ms on CPU
- **Output**: Extractive (drops tokens, may be grammatically choppy)
- **Status**: ✅ Open source, integrated in LangChain/LlamaIndex

### Comparison

| Aspect | Cmprsr | LLMLingua-2 |
|--------|--------|-------------|
| Method | Abstractive (rewrites) | Extractive (drops tokens) |
| Architecture | Qwen3-4B decoder | BERT encoder (~110M) |
| Quality | Higher (paraphrase) | Lower (choppy grammar) |
| Speed | ~50-100 t/s llama.cpp | ~10-50ms PyTorch |
| llama.cpp native | ✅ Yes | ❌ No (BERT classifier) |
| Availability | ❌ Not public | ✅ Open source |

### Decision

**~~Use LLMLingua-2 now, migrate to Cmprsr when available.~~**

**UPDATED 2026-01-27: Wait for Cmprsr weights. LLMLingua-2 causes quality regression.**

The extractive nature of LLMLingua-2 produces fragmentary text that confuses downstream LLMs. Abstractive compression (Cmprsr) rewrites text cleanly and is the only viable approach for this pipeline.

---

## Approved Architecture: 3-Tier Compression Pipeline

```
Document (N tokens)
       ↓
[DocumentCache] ← Store full OCR text (already exists)
       ↓
[LLMLingua-2] → Extractive compression (0.3-0.5N)  ← NEW "Stage 0"
       ↓
[Stage 1: Frontdoor] → Cursory summary (0.1N)
       ↓
[Stage 2: B2 Ingestion] → Validation + cleanup
       ↑                    (can reference full text from DocumentCache)
```

**Key insight**: DocumentCache already stores full OCR results. B2 can reference original text via REPL session when needed. LLMLingua-2 speeds up Stage 1.

---

## Implementation Plan

### Phase 1: Create Compression Service

**File**: `src/services/prompt_compressor.py` (CREATE ~150 LOC)

```python
from dataclasses import dataclass
from llmlingua import PromptCompressor as LLMLinguaCompressor

@dataclass
class CompressionResult:
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    actual_ratio: float
    latency_ms: float

class PromptCompressor:
    """LLMLingua-2 wrapper for extractive prompt compression."""

    def __init__(self, model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"):
        self.model = LLMLinguaCompressor(model_name=model_name)

    async def compress(self, text: str, target_ratio: float = 0.4) -> CompressionResult:
        """Compress text to target ratio (0.0-1.0)."""
        original_tokens = len(text.split())
        target_tokens = int(original_tokens * target_ratio)
        result = self.model.compress_prompt(text, target_token=target_tokens)
        return CompressionResult(
            compressed_text=result["compressed_prompt"],
            original_tokens=original_tokens,
            compressed_tokens=len(result["compressed_prompt"].split()),
            actual_ratio=len(result["compressed_prompt"]) / len(text),
            latency_ms=result.get("latency_ms", 0)
        )
```

### Phase 2: Integrate into Document Preprocessor

**File**: `src/services/document_preprocessor.py` (MODIFY +30 LOC)

```python
# In DocumentPreprocessor.preprocess()
async def preprocess(self, request: DocumentProcessRequest) -> DocumentPreprocessResult:
    # 1. OCR (existing)
    ocr_result = await self._run_ocr(request)

    # 2. Cache full text (existing - DocumentCache)
    await self._cache_document(ocr_result)

    # 3. NEW: Compress for Stage 1 if above threshold
    if self.compressor and len(ocr_result.text) > self.config.compression_threshold:
        compressed = await self.compressor.compress(
            ocr_result.text,
            target_ratio=request.compression_ratio or self.config.default_ratio
        )
        ocr_result.compressed_text = compressed.compressed_text
        ocr_result.compression_ratio = compressed.actual_ratio

    # 4. Chunk compressed text (existing)
    sections = self._chunk_text(ocr_result.compressed_text or ocr_result.text)
```

### Phase 3: Add Configuration

**File**: `orchestration/model_registry.yaml` (MODIFY +20 lines)

```yaml
compression:
  llmlingua2:
    enabled: true
    model: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
    default_ratio: 0.4          # 60% reduction
    min_tokens: 4000            # Below this, skip compression
    preserve_headers: true      # Keep markdown structure
    cache_model: true           # Keep model loaded

  # Per-task ratio overrides (dispatcher selects)
  task_ratios:
    urgent: 0.25                # Aggressive for speed
    balanced: 0.4               # Default
    quality: 0.6                # Conservative for quality
```

### Phase 4: REPL Integration

**File**: `src/repl_environment.py` (MODIFY +15 LOC)

```python
def get_full_document(self, doc_id: str) -> str:
    """Retrieve full uncompressed document from cache.

    Called by Stage 2 (B2 Ingestion) when validation needs
    original context not present in compressed summary.
    """
    return self.document_cache.get_full_text(doc_id)
```

### Phase 5: Tests & Benchmarks

**Files to create**:
- `tests/unit/test_prompt_compressor.py` (~100 LOC)
- `scripts/benchmark/bench_compression.py` (~80 LOC)

---

## Files Summary

| File | Action | LOC (est) |
|------|--------|-----------|
| `src/services/prompt_compressor.py` | CREATE | ~150 |
| `src/services/document_preprocessor.py` | MODIFY | +30 |
| `src/models/document.py` | MODIFY | +10 |
| `orchestration/model_registry.yaml` | MODIFY | +20 |
| `src/repl_environment.py` | MODIFY | +15 |
| `tests/unit/test_prompt_compressor.py` | CREATE | ~100 |
| `scripts/benchmark/bench_compression.py` | CREATE | ~80 |
| `pyproject.toml` | MODIFY | +1 |

---

## Verification Checklist

- [ ] `pip install llmlingua` succeeds
- [ ] Unit tests pass: `pytest tests/unit/test_prompt_compressor.py -v`
- [ ] Integration: Process sample PDF through full pipeline
- [ ] Benchmark compression ratios: 0.25, 0.4, 0.6
- [ ] Measure Stage 1 speedup with compression enabled
- [ ] Quality gate: Stage 2 output within 5% of uncompressed baseline

---

## Success Criteria

1. **Compression accuracy**: Actual ratio within ±15% of target
2. **Latency**: <100ms per document on EPYC 9655
3. **Stage 1 speedup**: >20% faster on documents >4K tokens
4. **Quality preservation**: <5% score drop on downstream tasks

---

## Future: Cmprsr Migration Path

When Cmprsr weights become available:
1. Check HuggingFace: `cmprsr/cmprsr-qwen3-4b`
2. Convert to GGUF: `convert_hf_to_gguf.py`
3. Benchmark against LLMLingua-2
4. If better: swap compressor backend (same interface)
5. Update model_registry.yaml

**Contact**: ivan@compresr.ai for weights availability

---

## Dependencies

```toml
# pyproject.toml
llmlingua = ">=0.2.0"
```

**Note**: LLMLingua-2 uses BERT encoder (~110M params), runs on CPU via PyTorch. First load ~2-3s, subsequent calls ~10-50ms depending on text length.

---

## References

- Cmprsr Paper: https://arxiv.org/abs/2511.12281
- LLMLingua GitHub: https://github.com/microsoft/LLMLingua
- LLMLingua-2 Paper: https://arxiv.org/abs/2310.05736
- Existing pipeline: `src/services/document_preprocessor.py`
- Two-stage config: `orchestration/model_registry.yaml:98-118`
- DocumentCache: `src/session/document_cache.py`

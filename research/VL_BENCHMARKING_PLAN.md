# Vision-Language Model Benchmarking Plan

**Created:** 2025-12-16
**Status:** Pending (awaiting model downloads)

---

## Current Vision Stack

| Role | Model | Speed | Acceleration |
|------|-------|-------|--------------|
| worker_vision | Qwen2.5-VL-7B-Instruct | 57.1 t/s | Spec Decode (K=8) |
| vision_escalation | Qwen3-VL-30B-A3B-Instruct | ~35 t/s | MoE (4 experts) |

---

## Models to Benchmark

### Downloaded (Ready)
- [ ] Qwen2.5-VL-7B-Instruct (baseline reference)
- [ ] Qwen3-VL-30B-A3B-Instruct (current escalation)

### Downloading
- [ ] (add models as they complete)

### Candidates to Consider
- Qwen2.5-VL-72B-Instruct (dense, ~45GB)
- Qwen3-VL-2B-Instruct (edge/draft candidate?)
- Qwen3-VL-4B-Instruct (fast worker candidate?)
- Other VL models as available

---

## Benchmark Metrics

### Quality Benchmarks (if available via prompt testing)

| Benchmark | What It Tests | Priority |
|-----------|---------------|----------|
| MMMU | Complex multi-step reasoning on images | HIGH |
| DocVQA | Document text extraction and QA | HIGH |
| MathVista | Mathematical reasoning in visual context | HIGH |
| ChartQA | Chart/graph understanding | MEDIUM |
| VideoMME | Long video comprehension | MEDIUM |
| ScreenSpot | GUI element identification | MEDIUM |
| OCRBench | Text recognition accuracy | LOW |

### Performance Benchmarks (measure on our system)

| Metric | Command/Method | Target |
|--------|----------------|--------|
| Baseline t/s | `llama-bench -m MODEL -t 96 -p 512 -n 128` | Record raw speed |
| MoE optimized t/s | `--override-kv MODEL.expert_used_count=int:4` | For Qwen3 VL only |
| Spec decode t/s | `llama-speculative -md DRAFT --draft-max 16` | For Qwen2.5 VL |
| Spec decode acceptance | Check output for acceptance rate | >20% useful |
| Cold load time | Time from start to first token | <30s for escalation |
| Memory usage | `nvidia-smi` or process memory | Fits in RAM tier |

---

## Test Prompts for Quality Comparison

### Math in Image
```
[Attach image with equation/diagram]
"Solve the equation shown in the image and explain each step."
```

### Document Understanding
```
[Attach PDF page or document screenshot]
"Extract the key information from this document and summarize."
```

### Complex Diagram
```
[Attach flowchart or architecture diagram]
"Describe the process shown in this diagram. What are the inputs and outputs?"
```

### Long Video (if supported)
```
[Attach 10+ minute video]
"Summarize the key events in this video with timestamps."
```

### GUI/Screenshot
```
[Attach application screenshot]
"Identify all interactive elements and describe their purpose."
```

---

## Acceleration Compatibility Matrix

| Model | MoE Reduction | Spec Decode | Prompt Lookup |
|-------|---------------|-------------|---------------|
| Qwen2.5-VL-7B | No (dense) | Yes (draft_qwen25) | Untested |
| Qwen2.5-VL-72B | No (dense) | Maybe (K=16?) | Untested |
| Qwen3-VL-30B-A3B | Yes (4 experts) | No (MoE) | Untested |
| Qwen3-VL-2B | Check arch | Check | Untested |
| Qwen3-VL-4B | Check arch | Check | Untested |

---

## Decision Criteria

### For worker_vision (fast, interactive)
1. Speed > 40 t/s with acceleration
2. Good DocVQA / OCR performance
3. Handles screenshots and simple images well
4. Memory < 8GB preferred

### For vision_escalation (quality, batch)
1. Speed > 20 t/s acceptable (cold load OK)
2. Strong MMMU / MathVista scores
3. Long video support preferred
4. Memory < 50GB (cold loadable)

### For potential vision_architect (if needed)
1. Best-in-class reasoning
2. Speed secondary concern
3. Memory can be large (split files OK)
4. Qwen3-VL-235B-A22B-Thinking candidate?

---

## Notes

- Qwen3-VL uses MoE architecture → benefits from expert reduction
- Qwen2.5-VL is dense → benefits from speculative decoding
- mmproj files required for all VL models
- Temperature 0.7 recommended for VL spec decode (per quirks)

---

## Results (fill in after benchmarking)

| Model | Baseline | Optimized | Best Acceleration | Quality Notes |
|-------|----------|-----------|-------------------|---------------|
| Qwen2.5-VL-7B | 15.28 t/s | 57.1 t/s | Spec Decode K=8 | Good for simple tasks |
| Qwen3-VL-30B-A3B | TBD | TBD | MoE 4 experts | TBD |
| ... | | | | |

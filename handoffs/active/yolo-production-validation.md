# Yolo Agent: Production Orchestrator Validation & Optimization

## EXECUTION MODE: FULLY AUTONOMOUS

**Do NOT pause between phases. Do NOT ask for permission. Execute all phases (0→5) sequentially without human intervention. If a phase fails, log the failure, attempt a fix, and continue to the next phase. Only stop if the entire stack is unrecoverable.**

## Status: PHASE 0 COMPLETE, PHASE 1 READY

**Prerequisites (ALL DONE):**
- Prompt lookup implemented and merged (commit 8e35dbc01)
- `--lookup` flag integrated into orchestrator_stack.py
- Vision server infrastructure added to orchestrator_stack.py (ports 8086, 8087)
- Model registry updated with combined spec+lookup numbers + vision server configs
- 243+ unit tests passing
- Optuna installed (`optuna 4.7.0`)

**Session 12 (2026-01-28) completed:**
- Phase 0.5: VL benchmark code changes (executor.py, run_benchmark.py, compare_orchestrator_direct.py)
- numactl fix: `_numa_prefix()` auto-detects NUMA support, falls back to direct launch in containers
- compare_orchestrator_direct.py fully rewritten: 8 suites, real baseline via architect_general, quality heuristics

**Note:** Servers are NOT running. You must launch them yourself.

**Launch command:**
```bash
./scripts/server/launch_production.sh --full
```

---

## Phase 0: Infrastructure — Add Vision Servers to Stack [COMPLETE]

**Goal:** Add dedicated vision server ports (8086, 8087) to the production stack. Currently VL models have NO server — only CLI via `llama-mtmd-cli`. Our `llama-server` supports `--mmproj` so we can run them as servers.

**Plan file with full details:** `/home/daniele/.claude/plans/jiggly-kindling-melody.md`

**Completion status (Session 12, 2026-01-28):**
- 0.1 orchestrator_stack.py: DONE (prior session)
- 0.2 launch_production.sh: DONE (prior session)
- 0.3 model_registry.yaml: DONE (prior session)
- 0.4 CLAUDE.md + vision config: DONE (prior session)
- 0.5 Benchmark VL server support: DONE (Session 12)
  - `scripts/lib/executor.py`: mmproj_path in ServerManager, `_run_vl_inference()` via `/v1/chat/completions`
  - `scripts/benchmark/run_benchmark.py`: mmproj wired through 3 start sites + image_path in inference
  - `scripts/benchmark/compare_orchestrator_direct.py`: Full rewrite — 8 suites, real baseline, quality heuristics
- **Bonus fix:** numactl made optional via `_numa_prefix()` in orchestrator_stack.py (container compatibility)

### 0.1 Add vision server support to `orchestrator_stack.py`

**File:** `scripts/server/orchestrator_stack.py`

Changes needed:

1. **Add VL model path constants** (after line 111):
```python
VISION_WORKER_MODEL = "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
VISION_WORKER_MMPROJ = "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf"
VISION_ESCALATION_MODEL = "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
VISION_ESCALATION_MMPROJ = "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"
```

2. **Update PORT_MAP**: Change `"worker_vision": 8082` → `"worker_vision": 8086`, add `"vision_escalation": 8087`

3. **Update HOT_ROLES**: Add `"worker_vision"`, `"vision_escalation"`

4. **Update HOT_SERVERS**: Remove `"worker_vision"` from port 8082 roles. Add:
```python
{"port": 8086, "roles": ["worker_vision"], "vision": True, "vision_type": "worker"},
{"port": 8087, "roles": ["vision_escalation"], "vision": True, "vision_type": "escalation"},
```

5. **Add vision branch to `build_server_command()`** (after embedding_mode block):
   - worker_vision: `-m MODEL --mmproj MMPROJ -np 2 -c 8192 -t 24 --flash-attn on`
   - vision_escalation: `-m MODEL --mmproj MMPROJ --override-kv qwen3vlmoe.expert_used_count=int:4 -np 2 -c 16384 -t 96 --flash-attn on`
   - NO spec decode, NO lookup (VL models incompatible)

6. **Update `validate_model_paths()`**: Add checks for all 4 VL files

### 0.2 Update launch script

**File:** `scripts/server/launch_production.sh`

Add to --full component list:
```bash
echo "  - worker_vision (8086): Qwen2.5-VL-7B + mmproj, ~15 t/s"
echo "  - vision_escalation (8087): Qwen3-VL-30B-A3B + mmproj, MoE4, ~10 t/s"
```
Update RAM: ~510GB → ~535GB

### 0.3 Update model registry

**File:** `orchestration/model_registry.yaml`
- worker_vision: port 8082 → 8086, add server config
- vision_escalation: add port 8087, add server config

### 0.4 Update documentation

**Files:** `CLAUDE.md`, `src/vision/config.py`
- Add 8086/8087 to CLAUDE.md server topology table
- Fix stale VL paths in `src/vision/config.py` (currently point to non-existent `/mnt/raid0/llm/models/` paths)

### 0.5 Fix benchmark VL server support

**File:** `scripts/benchmark/run_benchmark.py`
- Line 743: `mmproj_path is None  # VL models can't use server` — this is **STALE**. Our `llama-server` now supports `--mmproj`. Remove this guard so VL benchmarks can run against the server.
- The server VL endpoint uses `/v1/chat/completions` with OpenAI multimodal format:
```json
{"messages": [{"role": "user", "content": [
  {"type": "text", "text": "prompt"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]}], "max_tokens": 100}
```

**Definition of done:** `./scripts/server/launch_production.sh --full` starts all 11 servers (8080-8087, 8090, 9001, 8000) and health checks pass on all.

---

## Phase 1: Live Orchestrator Smoke Test

**Goal:** Verify the full production stack routes requests correctly to all 11 components.

### 1.1 Start the stack
```bash
cd /mnt/raid0/llm/claude
./scripts/server/launch_production.sh --full
```

### 1.2 Health-check polling loop
```bash
# Poll until all model servers are healthy (timeout 5 minutes)
PORTS="8080 8081 8082 8083 8084 8085 8086 8087 8090"
TIMEOUT=300
START=$(date +%s)
while true; do
  ALL_UP=true
  for port in $PORTS; do
    if ! curl -s --connect-timeout 2 http://localhost:$port/health >/dev/null 2>&1; then
      ALL_UP=false
      break
    fi
  done
  if $ALL_UP; then
    echo "All servers healthy"
    break
  fi
  ELAPSED=$(( $(date +%s) - START ))
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "TIMEOUT: Not all servers healthy after ${TIMEOUT}s"
    for port in $PORTS; do
      echo -n "  Port $port: "
      curl -s --connect-timeout 2 http://localhost:$port/health || echo "DOWN"
    done
    exit 1
  fi
  echo "Waiting... (${ELAPSED}s elapsed)"
  sleep 10
done
```

### 1.3 Test each server directly

```bash
# Frontdoor (8080) - task classification
curl -s -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Classify this task: Write a fibonacci function", "n_predict": 100, "temperature": 0}'

# Coder (8081) - code generation with spec+lookup (longer prompt for meaningful acceptance)
curl -s -X POST http://localhost:8081/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "# Implement a binary search tree with insert, delete, and search operations\n# Include balance checking and in-order traversal\n\nclass BSTNode:\n    def __init__(self, val):\n        self.val = val\n        self.left = None\n        self.right = None\n\nclass BST:\n    def __init__(self):\n        self.root = None\n\n    def insert(self, val):\n", "n_predict": 300, "temperature": 0, "lookup": true}'

# Worker explore (8082) - exploration with spec+lookup (longer prompt)
curl -s -X POST http://localhost:8082/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Analyze the following components of a modern web application architecture and explain how they interact:\n\n1. Load Balancer (nginx/HAProxy)\n2. Application Server (Node.js/Python)\n3. Database Layer (PostgreSQL + Redis cache)\n4. Message Queue (RabbitMQ/Kafka)\n5. CDN (CloudFront/Cloudflare)\n\nFor each component, describe its role, failure modes, and how it communicates with the others:\n", "n_predict": 300, "temperature": 0, "lookup": true}'

# Architect general (8083) - complex reasoning
curl -s -X POST http://localhost:8083/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Design a distributed cache invalidation strategy:", "n_predict": 200, "temperature": 0}'

# Architect coding (8084) - ultimate code escalation
curl -s -X POST http://localhost:8084/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Implement a lock-free concurrent hash map:", "n_predict": 200, "temperature": 0}'

# Ingest (8085) - long context synthesis (SSM, NO spec, NO lookup)
curl -s -X POST http://localhost:8085/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize the key themes:", "n_predict": 200, "temperature": 0}'

# Worker vision (8086) - VL model with mmproj (OpenAI multimodal format)
IMAGE_B64=$(base64 -w0 benchmarks/images/vl/ocrbench/sample_000.png)
curl -s -X POST http://localhost:8086/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"messages\": [{\"role\": \"user\", \"content\": [
    {\"type\": \"text\", \"text\": \"What text is visible in this image?\"},
    {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMAGE_B64\"}}
  ]}], \"max_tokens\": 100, \"temperature\": 0}"

# Vision escalation (8087) - MoE VL model
curl -s -X POST http://localhost:8087/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"messages\": [{\"role\": \"user\", \"content\": [
    {\"type\": \"text\", \"text\": \"Analyze this chart and describe the data trend:\"},
    {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMAGE_B64\"}}
  ]}], \"max_tokens\": 200, \"temperature\": 0}"

# Embedder (8090) - embedding endpoint
curl -s -X POST http://localhost:8090/embedding \
  -H "Content-Type: application/json" \
  -d '{"content": "Test embedding for episodic memory lookup"}'
# Verify: response contains "embedding": [...] array

# Document formalizer (9001) - health check
curl -s http://localhost:9001/health
```

### 1.4 Record speeds
For each response, extract `timings.predicted_per_second` and compare to expected:

| Port | Role | Expected t/s | Actual t/s | Status |
|------|------|-------------|-----------|--------|
| 8080 | frontdoor | 18 | | |
| 8081 | coder_escalation | 39 | | |
| 8082 | worker_explore | 44 | | |
| 8083 | architect_general | 6.75 | | |
| 8084 | architect_coding | 10.3 | | |
| 8085 | ingest_long_context | 6.3 | | |
| 8086 | worker_vision | ~15 | | |
| 8087 | vision_escalation | ~10 | | |
| 8090 | embedder | N/A (embedding) | | |
| 9001 | document_formalizer | N/A (OCR) | | |

### 1.5 Test orchestrator API routing
```bash
# Test via orchestrator API (port 8000)
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Write a Python function to sort a list"}], "max_tokens": 200}'
```

**Definition of done:** All 11 components respond. Model servers within 20% of expected speed.

**Note:** Fast workers (8102/8112) are WARM tier, only with `--with-burst`. Not in `--full` smoke test.

---

## Phase 2: End-to-End Orchestrator Benchmarking

**Goal:** Benchmark the full orchestrator pipeline (routing → delegation → specialist execution → escalation) by feeding ALL suite prompts through the frontdoor. Individual model benchmarks are already done — this tests the orchestrator as a system.

**Key tool:** `scripts/benchmark/compare_orchestrator_direct.py` — feeds prompts to orchestrator API (port 8000 `/chat`) and compares against pre-computed direct-model baselines.

### 2.1 Populate orchestrator baseline
The baseline file `orchestration/orchestrator_baseline.json` is currently **empty**. Populate it from existing direct-model benchmark results:
```bash
cd /mnt/raid0/llm/claude
# Generate baseline from existing individual model benchmark results
python3 scripts/benchmark/compare_orchestrator_direct.py --generate-baseline --suite all
```
If `--generate-baseline` doesn't exist, manually populate from `benchmarks/results/runs/` — each suite's best direct-model answers become the baseline.

### 2.2 Run orchestrator benchmark (all suites through frontdoor)
```bash
# Feed ALL suite prompts to orchestrator API at port 8000
# Frontdoor classifies, routes to specialist, escalates if needed
python3 scripts/benchmark/compare_orchestrator_direct.py --suite thinking
python3 scripts/benchmark/compare_orchestrator_direct.py --suite coder
python3 scripts/benchmark/compare_orchestrator_direct.py --suite general
python3 scripts/benchmark/compare_orchestrator_direct.py --suite math
python3 scripts/benchmark/compare_orchestrator_direct.py --suite agentic
python3 scripts/benchmark/compare_orchestrator_direct.py --suite instruction_precision
python3 scripts/benchmark/compare_orchestrator_direct.py --suite long_context
python3 scripts/benchmark/compare_orchestrator_direct.py --suite vl
```

**Vision routing note:** The orchestrator API `/chat` has NO multimodal input field. VL tasks work by the Frontdoor recognizing "this needs vision" and calling the `analyze_figure(image_path, prompt)` REPL tool, which routes to port 8086/8087 via the dispatcher (`src/dispatcher.py` maps `"vision" → "worker_vision"`). VL suite prompts must include the image file path in the text so the Frontdoor can delegate. Example:
```
"Analyze the image at /mnt/raid0/llm/claude/benchmarks/images/vl/ocrbench/sample_000.png — what text is visible?"
```

### 2.3 Compare orchestrator vs direct
```bash
python3 scripts/benchmark/compare_orchestrator_direct.py --suite all --use-baseline
```
Measures: speedup, latency, routing accuracy, quality retention vs direct model.

### 2.4 Save results
- Raw results → `benchmarks/results/runs/{timestamp}/`
- Update `benchmarks/results/index.jsonl`
- Update `docs/reference/benchmarks/RESULTS.md` with orchestrator scores

**Definition of done:** All 8 suites run through orchestrator. Quality within 10% of direct-model baseline. Routing accuracy ≥90% (correct specialist selected).

---

## Phase 3: Optuna Hyperparameter Optimization

**Goal:** Optimize orchestrator routing/escalation/learning parameters using Optuna.

**Blocked on:** Phase 2 completion (needs baseline scores)

**Tool:** `scripts/benchmark/optuna_orchestrator.py` — already fully implemented. Calls orchestrator `/chat` endpoint, tunes runtime-tunable params (no server restart required).

### 3.1 Routing layer optimization
```bash
python3 scripts/benchmark/optuna_orchestrator.py --layer routing --trials 50
```
Parameters tuned: `confidence_threshold`, `q_weight`, `min_q_value`, `min_similarity`

### 3.2 Escalation layer optimization
```bash
python3 scripts/benchmark/optuna_orchestrator.py --layer escalation --trials 25
```
Parameters tuned: `max_retries`, `max_escalations`

### 3.3 Learning layer optimization
```bash
python3 scripts/benchmark/optuna_orchestrator.py --layer learning --trials 25
```
Parameters tuned: `learning_rate`, `success_reward`, `failure_reward`

### 3.4 Apply optimal configs
- Script saves checkpoint to `orchestration/optimization_checkpoint.yaml`
- Uses cluster-based robust selection from top 20% of trials
- Re-run Phase 2 orchestrator benchmarks with winning params to verify improvement
- Document improvements in `docs/reference/benchmarks/RESULTS.md`

**Definition of done:** Optuna finds configs that improve at least one metric (speed or quality) without degrading the other.

---

## Phase 4: Prompt Lookup Validation

**Goal:** Validate the new prompt lookup feature across different workloads.

### 4.1 Code editing prompts (high n-gram overlap expected)
```bash
# Test with a prompt that includes source code to refactor
curl -s -X POST http://localhost:8081/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Refactor this code to use async/await:\n\ndef fetch_data():\n    response = requests.get(url)\n    data = response.json()\n    return data\n\nRefactored version:\n", "n_predict": 200, "temperature": 0, "lookup": true}' \
  | jq '.timings'
```

### 4.2 Summarization prompts (moderate n-gram overlap)
Test with document QA where response quotes source material.

### 4.3 Novel generation (low n-gram overlap)
Test creative writing where output is mostly novel — lookup should have minimal effect.

### 4.4 Record acceptance rates
For each test, record:
- `timings.predicted_per_second`
- Draft acceptance rate (from `timings.draft_n_accepted / timings.draft_n`)
- Compare lookup-enabled vs disabled

**Definition of done:** Acceptance rates documented for 3+ workload types, results added to RESULTS.md.

---

## Phase 5: Document & Clean Up

### 5.1 Update benchmark results
- `docs/reference/benchmarks/RESULTS.md` — add orchestrator benchmark scores (end-to-end, not per-model)
- `benchmarks/results/reviews/summary.csv` — add new entries
- If Optuna found improvements (Phase 3), merge `orchestration/optimization_checkpoint.yaml` winning params into `orchestration/model_registry.yaml`

### 5.2 Update progress log
- `progress/2026-01/` — create entry for the day this runs

### 5.3 Git commit
```bash
git add -A
git commit -m "feat: Production orchestrator validation + Optuna optimization + prompt lookup benchmarks"
git push origin main
```

---

## Full Production Stack (after Phase 0)

| Port | Role | Model | Accel | Speed | RAM |
|------|------|-------|-------|-------|-----|
| 8080 | frontdoor | Qwen3-Coder-30B-A3B | MoE6 | 18 t/s | 19 GB |
| 8081 | coder_escalation | Qwen2.5-Coder-32B + draft | spec K=24 + lookup | 39 t/s | 20 GB |
| 8082 | worker_explore | Qwen2.5-7B-Instruct + draft | spec K=24 + lookup | 44 t/s | 6 GB |
| 8083 | architect_general | Qwen3-235B-A22B | MoE4 | 6.75 t/s | 133 GB |
| 8084 | architect_coding | Qwen3-Coder-480B-A35B | MoE3 | 10.3 t/s | 271 GB |
| 8085 | ingest_long_context | Qwen3-Next-80B-A3B (SSM) | MoE4, NO spec | 6.3 t/s | 46 GB |
| 8086 | worker_vision | Qwen2.5-VL-7B + mmproj | None (no spec for VL) | ~15 t/s | 6 GB |
| 8087 | vision_escalation | Qwen3-VL-30B-A3B + mmproj | MoE4 | ~10 t/s | 19 GB |
| 8090 | embedder | Qwen2.5-Coder-0.5B | None | N/A | 1 GB |
| 9001 | document_formalizer | LightOnOCR-2-1B | None | N/A | 10 GB |
| 8000 | orchestrator API | uvicorn | N/A | N/A | — |

**Total: ~535 GB (47% of 1.13 TB), 11 components**

WARM tier (burst only, `--with-burst`):
| 8102 | worker_fast_1 | Qwen2.5-Coder-1.5B | None | 60 t/s | 1 GB |
| 8112 | worker_fast_2 | Qwen2.5-Coder-1.5B | None | 60 t/s | 1 GB |

---

## Environment Notes

- **Working dir:** `/mnt/raid0/llm/claude/`
- **Python env:** Use `python3` (pace-env)
- **NEVER set OMP_NUM_THREADS=1** — it disables parallel tensor repack
- **NEVER use pytest -n auto** — spawns 192 workers, crashes machine
- **SSM models (port 8085):** NO spec decode, NO prompt lookup
- **Qwen3-Coder-480B (port 8084):** MoE3 only, NO spec decode (BOS mismatch)
- **VL models (ports 8086, 8087):** NO spec decode (mmproj incompatible with llama-speculative)
- **VL API format:** Use `/v1/chat/completions` with OpenAI multimodal content (`image_url` type), NOT `/completion`

## Key Files

| File | Purpose |
|------|---------|
| `scripts/server/launch_production.sh` | Stack launcher |
| `scripts/server/orchestrator_stack.py` | Stack manager (ports, models, commands) |
| `scripts/benchmark/run_benchmark.py` | Benchmark runner |
| `scripts/benchmark/suites.py` | Suite definitions |
| `scripts/benchmark/optuna_orchestrator.py` | Optuna optimization |
| `orchestration/model_registry.yaml` | Model configs & routing |
| `docs/reference/benchmarks/RESULTS.md` | Master results table |
| `benchmarks/results/index.jsonl` | Raw results index |
| `src/vision/config.py` | Vision pipeline config (stale VL paths to fix) |

## Created

2026-01-28 — Consolidated from multiple handoffs (draft-benchmark, formalizer-evaluation, orchestration-integration, llama-server-prompt-lookup)
2026-01-28 — Phase 0 added: vision servers (8086, 8087) + infrastructure fixes

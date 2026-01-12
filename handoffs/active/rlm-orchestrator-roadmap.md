# RLM-Enhanced Orchestrator Development Roadmap

**Created**: 2026-01-13
**Status**: Active Development Handoff
**Depends On**: Benchmarks completing, RadixAttention verified (80% cache hit)
**Research**: See `research/rlm_analysis.md` for theoretical background

---

## Quick Status

| Phase | Description | Status | Blocking |
|-------|-------------|--------|----------|
| 1 | Backend Completion | READY | None |
| 2 | RLM Enhancements | READY | Phase 1 |
| 3 | Escalation Integration | READY | Phase 1 |
| 4 | Formalizer Integration | READY | Phase 3 |
| 5 | Tool/Script Completion | READY | None |
| 6 | Early Failure Detection | READY | Phase 3 |
| 7 | Hyperparameter Tuning | BLOCKED | Benchmarks |
| 8 | Trajectory Visualization | LOW | Phase 2 |

---

## Current Implementation State

### Complete (Mock Mode)

From `handoffs/active/orchestrator.md`:

| Component | File | Tests | Purpose |
|-----------|------|-------|---------|
| REPL Environment | `src/repl_environment.py` | 49 | Sandboxed Python execution |
| LLM Primitives | `src/llm_primitives.py` | 31 | `llm_call()`, `llm_batch()` |
| Gate Runner | `src/gate_runner.py` | 22 | Quality gate execution |
| Failure Router | `src/failure_router.py` | 51 | Escalation routing |
| FastAPI | `src/api.py` | 26 | HTTP interface |
| RadixAttention | `src/prefix_cache.py` | 46 | Prefix caching (80% hit verified) |
| Tool Registry | `src/tool_registry.py` | 15 | Role-based permissions |
| Script Registry | `src/script_registry.py` | 14 | Prepared scripts |

### Incomplete (Real Mode Blocked)

| Component | Current State | Blocker |
|-----------|---------------|---------|
| LlamaServerBackend HTTP | Stub only | Needs `infer()`, `infer_stream()` |
| CachingBackend init | Imported, never initialized | Needs wiring in `__init__()` |
| Role→Backend routing | Role param ignored | Needs URL selection |
| FailureRouter integration | Created but unused | Needs wiring in Root LM loop |
| GateRunner integration | Endpoint exists, not in loop | Needs wiring |
| MCP client | Raises NotImplementedError | Needs implementation |

---

## PHASE 1: Backend Completion (CRITICAL PATH)

**Goal**: Get real inference working

### Task 1.1: Complete LlamaServerBackend HTTP

**File**: `src/backends/llama_server.py`

**Current state**: ServerConfig, SlotInfo defined. HTTP methods stubbed.

**Work needed**:
```python
async def infer(self, prompt: str, **kwargs) -> str:
    """Make completion request to llama-server."""
    async with aiohttp.ClientSession() as session:
        payload = {
            "prompt": prompt,
            "n_predict": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.2),
            "cache_prompt": True,  # Enable RadixAttention
            "slot_id": self._get_slot_for_prefix(prompt),
        }
        async with session.post(
            f"{self.base_url}/completion",
            json=payload
        ) as resp:
            result = await resp.json()
            return result["content"]

async def infer_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
    """Stream completion tokens."""
    payload = {"prompt": prompt, "stream": True, ...}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{self.base_url}/completion", json=payload) as resp:
            async for line in resp.content:
                if line.startswith(b"data: "):
                    data = json.loads(line[6:])
                    if "content" in data:
                        yield data["content"]
```

### Task 1.2: Wire CachingBackend Initialization

**File**: `src/llm_primitives.py`

**Current state**: CachingBackend imported but never used.

**Work needed**:
```python
def __init__(self, model_server=None, mock_mode=True, server_urls=None, ...):
    self._mock_mode = mock_mode
    self._backends = {}

    if not mock_mode and server_urls:
        for role, url in server_urls.items():
            backend = LlamaServerBackend(base_url=url)
            self._backends[role] = CachingBackend(
                backend,
                PrefixRouter(prefix_length=256),
                canonicalize=True
            )
```

### Task 1.3: Connect Role→Backend Routing

**File**: `src/llm_primitives.py`

**Current state**: `role` parameter ignored in `_real_call()`.

**Work needed**:
```python
def _real_call(self, prompt: str, role: str = "worker") -> str:
    # Select backend by role
    if role in self._backends:
        backend = self._backends[role]
    elif "worker" in self._backends:
        backend = self._backends["worker"]  # Fallback
    else:
        raise ValueError(f"No backend for role: {role}")

    return backend.infer(prompt)
```

### Task 1.4: Fix Real Mode Initialization in API

**File**: `src/api.py`

**Current state**: Checks `_backends` but never populated.

**Work needed**:
```python
# In chat endpoint
if not request.mock_mode:
    server_urls = {
        "frontdoor": "http://localhost:8080",
        "coder": "http://localhost:8081",
        "worker": "http://localhost:8082",
    }
    primitives = LLMPrimitives(mock_mode=False, server_urls=server_urls)
```

### Tests for Phase 1

```bash
# Start test server
llama-server -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 -c 4096 -np 4 -t 16

# Run backend tests
python -m pytest tests/unit/test_prefix_cache.py tests/unit/test_llm_primitives.py -v

# Run integration tests
python -m pytest tests/integration/test_cache_hits.py -v --run-server
```

---

## PHASE 2: RLM Enhancements

**Goal**: Implement key features from RLM paper

### Task 2.1: Forced Exploration Validation

**File**: `src/repl_environment.py`

**Rationale**: Prevents models from calling `FINAL()` without actually exploring context.

```python
class REPLEnvironment:
    def __init__(self, context: str = "", ...):
        self._code_executed = False
        self._exploration_required = True  # Can be disabled for simple queries
        # ... existing init

    def execute(self, code: str) -> ExecutionResult:
        self._code_executed = True
        # ... existing execute logic

    def _final_handler(self, answer: str) -> None:
        if self._exploration_required and not self._code_executed:
            raise ExplorationRequired(
                "FINAL() called without code execution. "
                "Please explore the context first using peek(), grep(), or llm_call()."
            )
        self._final_answer = answer
        self._is_final = True
```

**Add exception class**:
```python
class ExplorationRequired(Exception):
    """Raised when FINAL() is called without exploration."""
    pass
```

### Task 2.2: Async llm_batch

**File**: `src/llm_primitives.py`

**Rationale**: True parallel execution of sub-LM calls.

```python
async def llm_batch_async(self, prompts: list[str], role: str = "worker") -> list[str]:
    """Async parallel execution of multiple LLM calls."""
    if self._mock_mode:
        return [f"[MOCK] Response for: {p[:50]}..." for p in prompts]

    tasks = [self._async_call(p, role) for p in prompts]
    return await asyncio.gather(*tasks)

async def _async_call(self, prompt: str, role: str) -> str:
    """Single async call."""
    backend = self._backends.get(role, self._backends.get("worker"))
    return await backend.infer_async(prompt)
```

**Inject into REPL**:
```python
# In repl_environment.py _build_globals()
"llm_batch_async": self.llm_primitives.llm_batch_async,
```

### Task 2.3: Configurable Recursion Depth

**File**: `src/llm_primitives.py`

**Rationale**: Enable depth-2+ experiments while defaulting to depth-1.

```python
class LLMPrimitives:
    def __init__(self, ..., max_recursion_depth: int = 1):
        self.max_recursion_depth = max_recursion_depth
        self._current_depth = 0

    def llm_call(self, prompt: str, role: str = "worker", allow_recursion: bool = True) -> str:
        """Call sub-LM with recursion depth tracking."""
        if self._current_depth >= self.max_recursion_depth:
            if allow_recursion:
                # At max depth, call without recursive capability
                return self._call_leaf(prompt, role)
            else:
                raise RecursionLimitReached(
                    f"Max recursion depth {self.max_recursion_depth} reached"
                )

        self._current_depth += 1
        try:
            return self._real_call(prompt, role)
        finally:
            self._current_depth -= 1

    def _call_leaf(self, prompt: str, role: str) -> str:
        """Call at leaf level (no further recursion allowed)."""
        # Sub-LM at leaf level doesn't get llm_call in its REPL
        return self._real_call_simple(prompt, role)
```

**API parameter**:
```python
class ChatRequest(BaseModel):
    max_recursion_depth: int = 1  # Default depth-1, configurable for research
```

### Task 2.4: Per-Query Cost Tracking

**File**: `src/llm_primitives.py`

```python
@dataclass
class QueryCost:
    prompt_tokens: int
    completion_tokens: int
    model: str
    role: str
    cost_usd: float
    timestamp: float

class LLMPrimitives:
    def __init__(self, ..., model_pricing: dict = None):
        self._query_costs: list[QueryCost] = []
        self._model_pricing = model_pricing or {}  # role -> cost_per_1k

    def _record_cost(self, role: str, prompt_tokens: int, completion_tokens: int):
        rate = self._model_pricing.get(role, 0.0)
        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1000) * rate
        self._query_costs.append(QueryCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=role,
            role=role,
            cost_usd=cost,
            timestamp=time.time()
        ))

    def get_total_cost(self) -> float:
        return sum(q.cost_usd for q in self._query_costs)

    def get_cost_breakdown(self) -> dict:
        by_role = {}
        for q in self._query_costs:
            by_role.setdefault(q.role, 0.0)
            by_role[q.role] += q.cost_usd
        return by_role
```

**Add to model_registry.yaml**:
```yaml
runtime_defaults:
  model_pricing:  # Cost per 1K tokens (input + output average)
    frontdoor: 0.002
    coder: 0.003
    worker: 0.0005
    architect: 0.01
```

### Tests for Phase 2

```bash
# Test forced exploration
python -c "
from src.repl_environment import REPLEnvironment, ExplorationRequired
repl = REPLEnvironment(context='test', exploration_required=True)
try:
    repl.execute('FINAL(\"premature\")')
except ExplorationRequired as e:
    print('Correctly blocked:', e)
"

# Test recursion depth
python -c "
from src.llm_primitives import LLMPrimitives
p = LLMPrimitives(mock_mode=True, max_recursion_depth=2)
print(f'Depth limit: {p.max_recursion_depth}')
"
```

---

## PHASE 3: Escalation Integration

**Goal**: Wire FailureRouter into the Root LM loop

### Task 3.1: Error Classification

**File**: `src/failure_router.py`

```python
def classify_error(exception: Exception) -> ErrorCategory:
    """Classify exception for routing decision."""
    error_msg = str(exception).lower()

    if isinstance(exception, SyntaxError):
        return ErrorCategory.CODE
    elif isinstance(exception, json.JSONDecodeError):
        return ErrorCategory.SCHEMA
    elif isinstance(exception, TimeoutError):
        return ErrorCategory.TIMEOUT
    elif isinstance(exception, ExplorationRequired):
        return ErrorCategory.LOGIC  # Model didn't follow protocol
    elif "format" in error_msg or "parse" in error_msg:
        return ErrorCategory.FORMAT
    elif "entropy" in error_msg or "abort" in error_msg:
        return ErrorCategory.EARLY_ABORT
    else:
        return ErrorCategory.LOGIC
```

### Task 3.2: Wire into Root LM Loop

**File**: `src/api.py`

```python
async def chat_endpoint(request: ChatRequest):
    repl = REPLEnvironment(context=request.context, ...)
    primitives = LLMPrimitives(...)
    failure_router = FailureRouter()

    current_role = "frontdoor"
    failure_count = 0

    for turn in range(request.max_turns):
        try:
            # Build prompt with current role
            prompt = _build_root_lm_prompt(repl.get_state(), request.prompt, current_role)

            # Call LLM
            code = primitives.llm_call(prompt, role=current_role)
            code = _extract_code_from_response(code)

            # Execute in REPL
            result = repl.execute(code)

            if result.is_final:
                return ChatResponse(answer=result.final_answer, ...)

            failure_count = 0  # Reset on success

        except Exception as e:
            failure_count += 1
            error_cat = classify_error(e)

            # Route failure
            ctx = FailureContext(
                role=current_role,
                failure_count=failure_count,
                error_category=error_cat,
                error_message=str(e),
            )
            decision = failure_router.route_failure(ctx)

            if decision.action == "escalate":
                current_role = decision.next_role
                # Continue to next turn with escalated role
            elif decision.action == "retry":
                # Continue to next turn with same role
                pass
            elif decision.action == "fail":
                return ChatResponse(error=str(e), ...)

    return ChatResponse(error="Max turns reached", ...)
```

### Task 3.3: Gate Execution Integration

**File**: `src/api.py`

```python
# After successful REPL execution, run quality gates
if result.has_code_output:
    gate_runner = GateRunner()
    gate_results = gate_runner.run_gates(["format", "lint"])

    for gr in gate_results:
        if not gr.passed:
            # Feed gate failure back to model
            failure_count += 1
            ctx = FailureContext(
                role=current_role,
                failure_count=failure_count,
                error_category=ErrorCategory.FORMAT,
                gate_name=gr.gate_name,
                error_message=gr.errors,
            )
            decision = failure_router.route_failure(ctx)
            # Handle decision...
```

---

## PHASE 4: Formalizer Integration

**Goal**: Enable automatic formalization for appropriate tasks

### Task 4.1: Formalizer Routing

**File**: `src/dispatcher.py`

```python
def should_formalize(task_ir: dict) -> bool:
    """Determine if task needs formalization preprocessing."""
    objective = task_ir.get("objective", "").lower()

    # Trigger words
    triggers = ["optimize", "constraint", "prove", "verify", "minimize", "maximize"]
    if any(t in objective for t in triggers):
        return True

    # High ambiguity
    if task_ir.get("ambiguity_score", 0) > 0.7:
        return True

    # Explicit flag
    if task_ir.get("formalization", {}).get("required", False):
        return True

    return False
```

### Task 4.2: Create Formalizer Module

**File**: `src/formalizer.py` (NEW)

```python
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class FormalizationResult:
    success: bool
    ir: Optional[dict]
    error: Optional[str]
    tokens_used: int

class Formalizer:
    def __init__(self, llm_primitives):
        self.primitives = llm_primitives

    def formalize(self, objective: str, context: str = "") -> FormalizationResult:
        """Generate FormalizationIR from natural language objective."""
        prompt = self._build_prompt(objective, context)

        try:
            response = self.primitives.llm_call(prompt, role="formalizer")
            ir = self._parse_response(response)
            return FormalizationResult(success=True, ir=ir, error=None, tokens_used=...)
        except Exception as e:
            return FormalizationResult(success=False, ir=None, error=str(e), tokens_used=...)

    def _build_prompt(self, objective: str, context: str) -> str:
        return f"""Formalize this task into structured JSON.

Objective: {objective}

Context: {context[:2000]}

Output FormalizationIR JSON with:
- problem_type: one of [optimization, constraint_satisfaction, proof, algorithm, ...]
- variables: list of variables with types and constraints
- constraints: list of constraints
- objective_function: if optimization
- acceptance_criteria: how to verify solution
"""

    def _parse_response(self, response: str) -> dict:
        # Extract JSON from response
        start = response.find("{")
        end = response.rfind("}") + 1
        return json.loads(response[start:end])
```

### Task 4.3: Inject into REPL Context

**File**: `src/api.py`

```python
# Before Root LM loop
if should_formalize(task_ir):
    formalizer = Formalizer(primitives)
    formal_result = formalizer.formalize(request.prompt, request.context)

    if formal_result.success:
        # Inject formalization into REPL context
        repl.set_artifact("formalization", formal_result.ir)

        # Modify prompt to reference formalization
        enhanced_prompt = f"""
Task has been formalized. See `artifacts['formalization']` for:
- problem_type: {formal_result.ir.get('problem_type')}
- variables: {len(formal_result.ir.get('variables', []))} defined
- constraints: {len(formal_result.ir.get('constraints', []))} defined

Original request: {request.prompt}
"""
```

---

## PHASE 5: Tool/Script Registry Completion

**Goal**: Full tool system operational

### Task 5.1: MCP Client Implementation

**File**: `src/tool_registry.py`

```python
import asyncio
from mcp import Client  # MCP client library

async def _invoke_mcp(self, tool: Tool, kwargs: dict) -> str:
    """Invoke tool via MCP server."""
    client = Client()
    await client.connect(tool.mcp_server)

    try:
        result = await client.call_tool(tool.mcp_tool, kwargs)
        return result.content
    finally:
        await client.disconnect()

def invoke(self, tool_name: str, role: str, **kwargs) -> str:
    """Invoke tool with permission check."""
    tool = self._tools.get(tool_name)
    if not tool:
        raise ValueError(f"Unknown tool: {tool_name}")

    # Permission check
    if not self._check_permission(tool, role):
        raise PermissionError(f"Role {role} cannot use tool {tool_name}")

    # Route to handler
    if tool.handler_type == "mcp":
        return asyncio.run(self._invoke_mcp(tool, kwargs))
    elif tool.handler_type == "python":
        return tool.handler(**kwargs)
    else:
        raise ValueError(f"Unknown handler type: {tool.handler_type}")
```

### Task 5.2: Script Registry invoke()

**File**: `src/script_registry.py`

```python
def invoke(self, script_id: str, **kwargs) -> str:
    """Execute prepared script."""
    script = self._scripts.get(script_id)
    if not script:
        raise ValueError(f"Unknown script: {script_id}")

    # Merge defaults with provided args
    args = {**script.default_args, **kwargs}

    # Route by execution mode
    if script.execution_mode == "code":
        return self._execute_code(script, args)
    elif script.execution_mode == "mcp":
        return self._execute_mcp(script, args)
    elif script.execution_mode == "command":
        return self._execute_command(script, args)

def _execute_code(self, script, args: dict) -> str:
    """Execute embedded Python code."""
    # Verify hash
    if script.code_hash:
        actual_hash = hashlib.sha256(script.code.encode()).hexdigest()
        if actual_hash != script.code_hash:
            raise SecurityError("Script code hash mismatch")

    # Execute in restricted environment
    globals_dict = {"args": args, "result": None}
    exec(script.code, globals_dict)
    return globals_dict.get("result", "")
```

### Task 5.3: Tool Result Capture

**File**: `src/repl_environment.py`

```python
def _invoke_tool(self, tool_name: str, **kwargs) -> str:
    """Invoke tool and capture result."""
    result = self.tool_registry.invoke(tool_name, self.role, **kwargs)

    # Capture in artifacts for later reference
    tool_key = f"tool_result_{tool_name}_{len(self._tool_results)}"
    self._tool_results.append({
        "tool": tool_name,
        "args": kwargs,
        "result": result,
        "timestamp": time.time()
    })
    self.artifacts[tool_key] = result

    return result
```

---

## PHASE 6: Early Failure Detection

**Goal**: Abort bad generations early to save compute

### Task 6.1: Wire GenerationMonitor

**File**: `src/llm_primitives.py`

```python
from src.generation_monitor import GenerationMonitor

def llm_call_monitored(self, prompt: str, role: str = "worker") -> str:
    """LLM call with early failure detection."""
    monitor = GenerationMonitor(
        entropy_threshold=self._get_entropy_threshold(role),
        spike_threshold=self._get_spike_threshold(role),
    )

    tokens = []
    for token in self._stream_call(prompt, role):
        tokens.append(token)
        monitor.observe(token)

        if monitor.should_abort():
            raise EarlyAbortError(
                f"Generation aborted: {monitor.abort_reason}",
                partial_output="".join(tokens)
            )

    return "".join(tokens)

def _get_entropy_threshold(self, role: str) -> float:
    """Per-tier entropy thresholds."""
    thresholds = {
        "frontdoor": 4.0,  # Tier A: strictest
        "coder": 5.0,       # Tier B
        "worker": 6.0,      # Tier C: most lenient
    }
    return thresholds.get(role, 5.0)
```

### Task 6.2: Add to model_registry.yaml

```yaml
runtime_defaults:
  early_failure:
    enabled: true
    entropy_thresholds:
      tier_a: 4.0
      tier_b: 5.0
      tier_c: 6.0
    spike_thresholds:
      tier_a: 2.0
      tier_b: 3.0
      tier_c: 4.0
    repetition_threshold: 0.3  # 30% 3-gram repetition
```

---

## PHASE 7: Hyperparameter Tuning Harness

**Goal**: Systematic optimization of model parameters

### Task 7.1: Sweep Framework

**File**: `scripts/benchmark/sweep_hyperparams.py` (NEW)

```python
#!/usr/bin/env python3
"""Hyperparameter sweep for orchestrator optimization."""

import itertools
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class SweepConfig:
    temperatures: list[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)
    top_p: list[float] = (0.9, 0.95, 1.0)
    top_k: list[int] = (40, 80, 0)  # 0 = disabled
    expert_counts: list[int] = (2, 3, 4, 6)  # For MoE models

def sweep_temperature(model: str, prompts: list[str], config: SweepConfig):
    """Sweep temperature for a model."""
    results = []
    for temp in config.temperatures:
        scores = run_evaluation(model, prompts, temperature=temp)
        results.append({
            "temperature": temp,
            "avg_score": sum(scores) / len(scores),
            "scores": scores
        })
    return results

def find_optimal(results: list[dict], metric: str = "avg_score") -> dict:
    """Find optimal configuration."""
    return max(results, key=lambda x: x[metric])

if __name__ == "__main__":
    # Run temperature sweep for frontdoor
    results = sweep_temperature(
        "Qwen3-Coder-30B-A3B",
        load_prompts("benchmarks/prompts/v1/orchestrator/"),
        SweepConfig()
    )

    # Save results
    Path("benchmarks/results/hyperparams/").mkdir(exist_ok=True)
    with open("benchmarks/results/hyperparams/temperature_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    optimal = find_optimal(results)
    print(f"Optimal temperature: {optimal['temperature']} (score: {optimal['avg_score']})")
```

---

## PHASE 8: Trajectory Visualization

**Goal**: Debug UI for recursive execution

### Task 8.1: Enhanced SSE Events

**File**: `src/api.py`

```python
# Add trajectory metadata to SSE events
async def stream_with_trajectory(repl, primitives, request):
    trajectory = []

    for turn in range(request.max_turns):
        # Record turn start
        trajectory.append({
            "type": "turn_start",
            "turn": turn,
            "role": current_role,
            "timestamp": time.time()
        })

        yield f"data: {json.dumps({'type': 'turn', 'turn': turn, 'role': current_role})}\n\n"

        # ... execution ...

        # Record LLM calls
        for call in primitives.get_recent_calls():
            trajectory.append({
                "type": "llm_call",
                "role": call.role,
                "prompt_preview": call.prompt[:100],
                "response_preview": call.response[:100],
                "tokens": call.tokens,
                "cost": call.cost
            })
            yield f"data: {json.dumps({'type': 'llm_call', **call.__dict__})}\n\n"
```

### Task 8.2: Trajectory Logging

**File**: `src/llm_primitives.py`

```python
from pathlib import Path
import json

class TrajectoryLogger:
    def __init__(self, log_dir: str = "logs/trajectories"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = None
        self.entries = []

    def start_session(self, query: str):
        self.session_id = f"{int(time.time())}_{hash(query) % 10000}"
        self.entries = [{"type": "session_start", "query": query, "ts": time.time()}]

    def log_call(self, role: str, prompt: str, response: str, tokens: int):
        self.entries.append({
            "type": "llm_call",
            "role": role,
            "prompt": prompt,
            "response": response,
            "tokens": tokens,
            "ts": time.time()
        })

    def end_session(self, final_answer: str):
        self.entries.append({"type": "session_end", "answer": final_answer, "ts": time.time()})

        # Write JSONL
        log_file = self.log_dir / f"{self.session_id}.jsonl"
        with open(log_file, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")
```

---

## Test Commands (Copy-Paste Ready)

```bash
# === Phase 1: Backend Tests ===
# Start test server
/mnt/raid0/llm/llama.cpp/build/bin/llama-server \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 -c 4096 -np 4 -t 16 &

# Run unit tests
python -m pytest tests/unit/test_prefix_cache.py tests/unit/test_llm_primitives.py -v

# Run integration tests
python -m pytest tests/integration/test_cache_hits.py -v --run-server

# === Phase 2-3: Integration Tests ===
python -m pytest tests/integration/test_cache_integration.py -v

# === E2E Validation ===
python scripts/test_recursive_orchestration.py -v

# === Start API Server ===
cd /mnt/raid0/llm/claude
uvicorn src.api:app --reload --port 8000

# === Test API ===
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize this text", "context": "Lorem ipsum...", "mock_mode": true}'
```

---

## Related Documents

| Document | Purpose |
|----------|---------|
| `handoffs/active/orchestrator.md` | Original orchestrator handoff |
| `handoffs/active/orchestration-integration.md` | RadixAttention integration |
| `research/rlm_analysis.md` | RLM paper analysis |
| `research/ESCALATION_FLOW.md` | Escalation chain documentation |
| `research/early_failure_prediction.md` | Early abort heuristics |
| `orchestration/model_registry.yaml` | Model configurations |
| `orchestration/BLOCKED_TASKS.md` | Blocked task tracking |

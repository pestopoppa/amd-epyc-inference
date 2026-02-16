# Chapter 18: Escalation, Failure Routing & Proactive Delegation

## Introduction

The orchestrator uses an **explicit pydantic-graph** to drive escalation decisions. Seven node classes encode valid transitions in Union return types. Rules from `escalation.py` are authoritative, and MemRL's learned escalation is advisory (injected via `TaskDeps`). Proactive delegation with complexity-aware routing (`proactive_delegation.py`) remains a separate execution strategy.

As of 2026-02-07, the legacy `FailureRouter` and `RoutingFacade` have been deleted. All escalation logic is now in `src/graph/nodes.py`.

## Unified Escalation Policy

### EscalationPolicy Architecture

```python
class EscalationAction(str, Enum):
    RETRY = "retry"          # Retry with same role
    THINK_HARDER = "think_harder"  # Same model, boosted config (CoT, 2x tokens)
    ESCALATE = "escalate"    # Escalate to next tier
    DELEGATE = "delegate"    # Route DOWN to a specific lower-tier role
    REVIEW = "review"        # Quality check by higher-tier model
    FAIL = "fail"            # Terminal failure
    SKIP = "skip"            # Skip the gate/step (optional gates only)
    EXPLORE = "explore"      # Fall back to REPL exploration (terminal roles)

class ErrorCategory(str, Enum):
    CODE = "code"              # Syntax/type errors, test failures
    LOGIC = "logic"            # Wrong output, failed assertions
    TIMEOUT = "timeout"        # Gate/execution timeout
    SCHEMA = "schema"          # IR/JSON schema violations
    FORMAT = "format"          # Style/format issues
    EARLY_ABORT = "early_abort"  # Model predicted failure, skip retries
    INFRASTRUCTURE = "infrastructure"  # Backend/network failure (seeding skips reward)
    UNKNOWN = "unknown"
```

**Early Abort**: When a model shows failure signs (incomplete generation, error patterns), immediately escalate instead of wasting retries. Detected via gate checks or output analysis.

### THINK_HARDER Action (February 2026)

Before escalating to a more expensive tier, the policy tries to squeeze more quality out of the current model. On the penultimate retry (`failure_count == max_retries - 1`), the decision is `THINK_HARDER` instead of `RETRY`. This returns the same role with a `config_override` that doubles the token budget, prepends a chain-of-thought prefix, and raises temperature slightly for diversity:

```python
EscalationDecision(
    action=EscalationAction.THINK_HARDER,
    target_role=context.current_role,   # Same model, not escalated
    config_override={
        "n_tokens": 4096,               # 2x default
        "cot_prefix": "Think step by step before answering.\n\n",
        "temperature": 0.5,             # Slightly higher for diversity
    },
)
```

The `config_override` field was added to `EscalationDecision` to carry these overrides. The calling node applies the overrides to the LLM call parameters for the retry turn.

**Rationale**: A model that failed once may succeed with explicit CoT prompting and more tokens to think. This is a "free" escalation axis -- often matching bigger-model quality without the cost or latency of tier escalation. Parallels Claude's extended thinking capability.

**Firing conditions**: Only fires for standard error categories (CODE, LOGIC, TIMEOUT on required gates, UNKNOWN). Format/schema errors and early-abort are excluded (they use retry-only or immediate-escalate paths respectively).

### Decision Rules

| Error Category | First Failure | Penultimate Retry | Final Retry | Post-Exhaustion |
|----------------|---------------|---------------------|-------------|-----------------|
| CODE, LOGIC | RETRY | THINK_HARDER | (exhausted) | ESCALATE |
| FORMAT, SCHEMA | RETRY | RETRY | FAIL | FAIL (never escalate) |
| TIMEOUT (optional gate) | SKIP | — | — | — |
| TIMEOUT (required gate) | RETRY | THINK_HARDER | (exhausted) | ESCALATE → FAIL |
| EARLY_ABORT | ESCALATE | — | — | — |
| INFRASTRUCTURE | — | — | — | Skip reward injection (seeding only); rules handle escalation |

**Config Defaults**:

```python
@dataclass
class EscalationConfig:
    max_retries: int = 2
    max_escalations: int = 2
    optional_gates: frozenset[str] = frozenset({
        "typecheck", "integration", "shellcheck"
    })
    no_escalate_categories: frozenset[ErrorCategory] = frozenset({
        ErrorCategory.FORMAT, ErrorCategory.SCHEMA
    })
```

**Rationale**: Format/schema errors indicate model instruction-following issues, not task complexity. Escalation won't help—just retry with clearer prompt.

### Escalation Chains

```python
class Role(Enum):
    WORKER_GENERAL = "worker_general"
    CODER_PRIMARY = "coder_primary"
    ARCHITECT_GENERAL = "architect_general"

    def escalates_to(self) -> Role | None:
        escalation_map = {
            Role.WORKER_GENERAL: Role.CODER_PRIMARY,
            Role.CODER_PRIMARY: Role.ARCHITECT_GENERAL,
            Role.ARCHITECT_GENERAL: None,  # Terminal
        }
        return escalation_map.get(self)
```

**Full Chains**:
- Worker → Coder → Architect (general tasks)
- Frontdoor → Coder → Architect (chat escalation)
- Ingest → Architect (long-context ingestion)
- Architect → FAIL (no further escalation)

## Pydantic-Graph Orchestration (February 2026)

The escalation loop is an explicit `pydantic_graph.Graph` with 7 node classes. Each node's `run()` method returns a Union of valid next nodes or `End[TaskResult]`, making transitions type-safe and visible.

### Node Classes

```python
from pydantic_graph import BaseNode, Graph, End, GraphRunContext

@dataclass
class FrontdoorNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    async def run(self, ctx) -> FrontdoorNode | CoderEscalationNode | WorkerNode | End[TaskResult]: ...

@dataclass
class WorkerNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    async def run(self, ctx) -> WorkerNode | CoderNode | End[TaskResult]: ...

@dataclass
class CoderNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    async def run(self, ctx) -> CoderNode | ArchitectNode | End[TaskResult]: ...

@dataclass
class CoderEscalationNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    async def run(self, ctx) -> CoderEscalationNode | ArchitectCodingNode | End[TaskResult]: ...

@dataclass
class IngestNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    async def run(self, ctx) -> IngestNode | ArchitectNode | End[TaskResult]: ...

@dataclass
class ArchitectNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    async def run(self, ctx) -> ArchitectNode | End[TaskResult]: ...  # Terminal

@dataclass
class ArchitectCodingNode(BaseNode[TaskState, TaskDeps, TaskResult]):
    async def run(self, ctx) -> ArchitectCodingNode | End[TaskResult]: ...  # Terminal

orchestration_graph = Graph(nodes=[all 7 classes])
```

### Node Execution Flow

Each node's `run()`:
1. Check `state.turns >= state.max_turns` → End(max turns)
2. Build prompt via `build_root_lm_prompt()` or use `state.escalation_prompt`
3. Call LLM via `deps.primitives.llm_call()` with role for this node
4. Extract code, auto-wrap FINAL, execute in REPL
5. If `is_final` → `End(TaskResult)`
6. If error → classify, record failure, decide retry/escalate/fail
7. If no error, no final → self-loop (return same node class)

### MemRL Integration via Dependencies

MemRL components are injected as immutable `TaskDeps`:

```python
@dataclass
class TaskDeps:
    primitives: LLMPrimitives | None
    repl: REPLEnvironment | None
    failure_graph: FailureGraphProtocol | None   # Anti-memory
    hypothesis_graph: HypothesisGraph | None      # Confidence tracking
    config: GraphConfig
    session_store: SessionStore | None

# Inside node error handler:
def _handle_error(ctx, error_cat, error):
    if ctx.deps.failure_graph:
        ctx.deps.failure_graph.record_failure(...)  # Anti-memory
    if ctx.deps.hypothesis_graph:
        ctx.deps.hypothesis_graph.add_evidence(...)  # Confidence
```

**Key change from RoutingFacade**: MemRL functions (`record_failure`, `record_mitigation`, `add_evidence`) are now actually called — they were dead code in the old architecture.

## 3-Way Confidence Routing (February 2026)

### Overview

The Unified Execution Model introduces 3-way confidence routing for faithful probability estimation. Instead of rigid mode-based routing, the frontdoor estimates P(success|action) for three approaches:

| Approach | Meaning | Maps To |
|----------|---------|---------|
| **SELF:direct** | Handle without tools | `frontdoor` with `mode=direct` |
| **SELF:repl** | Handle with tools, no delegation | `frontdoor` with `mode=repl`, `allow_delegation=False` |
| **ARCHITECT** | Escalate for complex reasoning | `architect_coding` or `architect_general` |
| **WORKER** | Delegate to faster workers | Scored via canonical `DelegationEvent` telemetry |

### Confidence Routing Prompt

```python
def build_confidence_estimation_prompt(question: str, context: str = "") -> str:
    return f"""Estimate your probability of correctly answering this question.

Question: {question[:500]}...

Rate your confidence (0.0-1.0) for each approach:
- SELF: You handle it (no escalation or delegation)
- ARCHITECT: Escalate to architect for complex reasoning
- WORKER: Delegate to faster worker models

Output ONLY this format:
CONF|SELF:X.XX|ARCHITECT:X.XX|WORKER:X.XX"""
```

### Cost-Adjusted Routing

At production routing time, Q-values are adjusted by cost tier:

```python
THREE_WAY_COST_TIER = {
    "SELF:direct": 2,   # Frontdoor speed
    "SELF:repl": 2,     # Frontdoor with tools
    "ARCHITECT": 4,     # Slow but capable
    "WORKER": 1,        # Fast workers
}

def route_with_cost(q_values: dict[str, float]) -> str:
    scores = {action: q / THREE_WAY_COST_TIER[action] for action, q in q_values.items()}
    return max(scores, key=scores.get)
```

**Key insight**: Cost is applied at routing time, not during Q-value updates. Q-values remain faithful P(success) estimates.

### General Delegation

Any role can now delegate (not just architects). Tier C restriction removed:

```python
_DELEGATABLE_ROLES = frozenset({
    "worker_explore", "worker_math", "worker_general",
    "worker_summarize", "worker_vision",
    "coder_escalation",
})

# In _delegate():
if target_role not in _DELEGATABLE_ROLES:
    raise ValueError(f"Cannot delegate to {target_role}")
```

### allow_delegation Parameter

The `ChatRequest` model now supports `allow_delegation` override for testing:

```python
class ChatRequest(BaseModel):
    allow_delegation: bool | None = Field(
        default=None,
        description="Override delegation. True=allow, False=disable, None=use feature flag.",
    )
```

Used by the 3-way evaluation script to test delegation value.

### Forced-role Semantics in 3-way Eval

For benchmark/seeding calls that set `force_role`, role identity is treated as an invariant for that call path:

- Quality-escalation role hopping is disabled under forced-role eval.
- Delegation is still allowed when `allow_delegation=True`.
- Result: eval keeps action identity stable (`SELF:*`, `ARCHITECT`) while still measuring delegation/tool value inside that action.

---

## Proactive Delegation

### Complexity Classification

```python
class TaskComplexity(Enum):
    TRIVIAL = "trivial"    # Frontdoor answers directly (factual, chat)
    SIMPLE = "simple"      # Frontdoor executes in REPL (single code task)
    MODERATE = "moderate"  # Frontdoor delegates to single specialist
    COMPLEX = "complex"    # Architect generates TaskIR, multi-specialist

def classify_task_complexity(objective: str) -> tuple[TaskComplexity, ComplexitySignals]:
    """Heuristic classifier based on objective text."""
    signals = ComplexitySignals()

    # TRIVIAL indicators
    if any(p in objective.lower() for p in ["what is", "who is", "define "]):
        return TaskComplexity.TRIVIAL, signals

    # CODE keywords
    if any(k in objective.lower() for k in ["implement", "code", "function"]):
        signals.has_code_keywords = True

    # MULTI-STEP keywords
    if any(k in objective.lower() for k in ["and then", "steps", "first"]):
        signals.has_multi_step_keywords = True

    # ARCHITECTURE keywords
    if any(k in objective.lower() for k in ["architecture", "design", "system"]):
        signals.has_architecture_keywords = True
        return TaskComplexity.COMPLEX, signals

    # Decision tree
    if signals.has_multi_step_keywords and signals.has_code_keywords:
        return TaskComplexity.MODERATE, signals
    elif signals.has_code_keywords:
        return TaskComplexity.SIMPLE, signals
    else:
        return TaskComplexity.SIMPLE, signals
```

**Escalation Triggers**: Override heuristics:

```python
ARCHITECT_TRIGGERS = ["/architect", "/plan", "break this down"]
THINKING_TRIGGERS = ["/think", "ultrathink", "think carefully"]

if has_architect_trigger(objective):
    return TaskComplexity.COMPLEX  # Force architect
if has_thinking_trigger(objective):
    signals.thinking_requested = True  # Route to thinking_reasoning model
```

### Delegation Paths

| Complexity | Action | Target | Token Efficiency |
|------------|--------|--------|------------------|
| TRIVIAL | direct | frontdoor | Free (no delegation) |
| SIMPLE | repl | frontdoor + REPL | 100x (context local) |
| MODERATE | specialist | coder_primary | Good (single model) |
| COMPLEX | architect | architect_general → TaskIR → multi-specialist | Lower (coordination overhead) |

**Design Goal**: Only invoke expensive architect (235B/480B models) when truly needed.

### Architect Review Loop

For COMPLEX tasks, the architect reviews specialist outputs:

```python
class ArchitectReviewService:
    # Concise prompts - minimize architect output tokens
    REVIEW_PROMPT_TEMPLATE = """Review specialist output. Be BRIEF.

    Objective: {objective}
    Subtask: {action}
    Output (truncated): {output}

    Reply JSON only (no explanation):
    {{"d":"approve|changes|escalate|reject","s":0.0-1.0,"f":"<10 words","c":["fix1"]}}

    d=decision, s=score, f=feedback, c=changes (optional, max 3 items)"""

    MAX_REVIEW_TOKENS = 128  # Strict limit for expensive model
```

**Abbreviated JSON**: Uses single-letter keys (`d`, `s`, `f`, `c`) to reduce output tokens from architect. Review typically 30-50 tokens vs 100+ for verbose format.

### Iteration Limits

```python
@dataclass
class IterationContext:
    max_iterations: int = 3          # Per subtask
    max_total_iterations: int = 10   # All subtasks combined

    def can_iterate(self, subtask_id: str) -> bool:
        subtask_count = self.subtask_iterations.get(subtask_id, 0)
        return (
            subtask_count < self.max_iterations
            and self.total_iterations < self.max_total_iterations
        )
```

Prevents infinite review-fix cycles. After max iterations, accept output or escalate.

## Performance Comparison

### Reactive vs Proactive (Estimated)

| Metric | Reactive Escalation | Proactive Delegation |
|--------|---------------------|----------------------|
| Cold start | 0s (rules only) | Architect call (2-5s) |
| Task decomposition | Manual in code | Architect generates TaskIR |
| Specialist coordination | Sequential escalation | Parallel execution |
| Token usage | Lower (no upfront planning) | Higher (architect review) |
| Quality | Good for well-defined | Better for complex/novel |

**When to Use**:
- **Reactive**: Well-defined tasks, known failure patterns, speed-critical
- **Proactive**: Novel tasks, complex multi-file changes, quality-critical

## Model Fallback (February 2026)

Model fallback handles **infrastructure failure** — distinct from task escalation which handles **task complexity**. When a backend is circuit-open, timed out, or OOM, same-tier alternatives are tried before failing the request.

### Fallback vs Escalation

| Concern | Mechanism | Trigger | Direction |
|---------|-----------|---------|-----------|
| Task complexity | Escalation (graph nodes) | Gate failure, retry exhaustion | Worker -> Coder -> Architect (upward) |
| Infrastructure failure | Fallback (`get_fallback_roles()`) | Circuit open, timeout, OOM | Same-tier alternatives (lateral) |

### Fallback Map

Defined in `src/roles.py` as `_FALLBACK_MAP`:

```python
_FALLBACK_MAP: dict[Role, list[Role]] = {
    Role.ARCHITECT_GENERAL: [Role.ARCHITECT_CODING, Role.CODER_PRIMARY],
    Role.ARCHITECT_CODING: [Role.ARCHITECT_GENERAL, Role.CODER_ESCALATION],
    Role.CODER_PRIMARY: [Role.CODER_ESCALATION],
    Role.CODER_ESCALATION: [Role.CODER_PRIMARY],
    Role.WORKER_MATH: [Role.WORKER_GENERAL],
    Role.INGEST_LONG_CONTEXT: [Role.ARCHITECT_GENERAL],
    Role.FRONTDOOR: [],            # Always-on, no fallback
    Role.WORKER_VISION: [],        # Hardware-specific, no fallback
}
```

### Failure Classification

`BackendHealthTracker.classify_failure()` maps error messages to `FailoverReason`:

| Error Pattern | FailoverReason |
|---------------|----------------|
| "circuit open" | `circuit_open` |
| "timed out", "timeout" | `timeout` |
| "out of memory", "oom", "kv cache" | `oom` |
| Everything else | `connection_error` |

### Integration

In `_real_call_impl()`: primary call via `_real_call_single()` catches `RuntimeError`. If `model_fallback` feature enabled, iterates `get_fallback_roles(role)` trying each alternative. Logs `FailoverReason` for observability.

Feature flag: `model_fallback`.

## Cache Affinity Phase 2.5 (February 2026)

The `TwoPhaseRetriever` has a Phase 2.5 step between Q-value ranking (Phase 2) and final sorting (Phase 3). It gives a 15% score bonus to memories whose role matches the last-used role, biasing routing toward warm KV caches.

### Mechanism

```python
class TwoPhaseRetriever:
    CACHE_AFFINITY_BONUS: float = 0.15

    def __init__(self, ...):
        self._last_role_used: Optional[str] = None

    def _retrieve(self, embedding, ...):
        # Phase 1: Semantic filtering → candidates
        # Phase 2: Q-value × similarity → combined_score

        # Phase 2.5: Cache affinity bonus
        if self._last_role_used:
            for result in results:
                if result.memory.metadata.get("role") == self._last_role_used:
                    result.combined_score *= (1.0 + self.CACHE_AFFINITY_BONUS)
                    result.cache_affinity = self.CACHE_AFFINITY_BONUS

        # Phase 3: Sort by combined_score → top-n
```

### Lifecycle

`HybridRouter.route()` calls `retriever.update_last_role(role)` after every routing decision (both learned and rule-based paths). The next retrieval then applies the bonus. The `RetrievalResult` dataclass carries `cache_affinity: float` and appends a warning string when the bonus fires.

### Why This Matters

When the same role serves consecutive requests, the KV cache from the previous request is warm -- prompt evaluation drops to near-zero. A 15% bonus is conservative; on this hardware (DDR5-5600, 460 GB/s), a warm-cache hit saves 50-200ms of prompt eval depending on context length. This parallels Claude's prompt caching TTL economics: recently-used contexts are cheap to re-enter.

**Files**: `orchestration/repl_memory/retriever.py` (`TwoPhaseRetriever._retrieve`, `HybridRouter.route`)

## Approval Gates (February 2026)

Human approval gates at escalation boundaries and destructive tool invocations. The graph halts, serializes state via resume token, waits for approval, then continues or rejects.

### Halt Triggers

| Trigger | HaltReason | Example |
|---------|------------|---------|
| Tier crossing | `ESCALATION` | Worker -> Coder, Worker -> Architect |
| Architect-tier model | `HIGH_COST` | Coder -> Architect (both Tier B, but expensive) |
| Destructive tool | `DESTRUCTIVE_TOOL` | `tool.destructive == True` |

### Protocol

```
Graph node -> should_halt(from_role, to_role)
  |
  +-- None -> continue normally
  +-- HaltReason -> build HaltState
       |
       +-- call deps.approval_callback.request_approval(halt)
            |
            +-- APPROVE -> continue to escalation/execution
            +-- REJECT -> End(success=False, answer="Rejected by user")
```

### Tier Classification

```python
_TIER_MAP = {
    "frontdoor": "A",
    "coder_primary": "B", "coder_escalation": "B",
    "architect_general": "B", "architect_coding": "B",
    "ingest_long_context": "B",
    "worker_general": "C", "worker_math": "C",
    "worker_explore": "C", "worker_summarize": "C",
    "worker_vision": "C",
}
```

Tier crossing (C->B, C->A) triggers `ESCALATION`. Same-tier to architect triggers `HIGH_COST`.

### Default Behavior

`AutoApproveCallback` auto-approves everything, preserving current behavior when `approval_gates=False` or no callback injected. The callback protocol (`ApprovalCallback`) can be implemented by API handlers for external approval flows.

Feature flag: `approval_gates` (requires `resume_tokens` + `side_effect_tracking`).

## Binding-Based Routing (February 2026)

Priority-ordered routing overrides from multiple sources. Same task type can route to different roles based on session state, user preference, or Q-values.

### Priority Levels

```python
class BindingPriority(IntEnum):
    DEFAULT = 0        # model_registry.yaml task_type -> role
    CLASSIFIER = 10    # _classify_and_route() keyword heuristic
    Q_VALUE = 20       # MemRL Q-value suggestion
    USER_PREF = 30     # ChatRequest.preferred_role header
    SESSION = 40       # Session-specific override (during conversation)
```

### Integration

After `_classify_and_route()` returns a role, `binding_router.resolve(task_type)` is checked. If a binding with higher priority exists and its backend is available, that role is used instead. Session bindings are cleared at conversation end.

Implementation: `src/routing_bindings.py` (`BindingRouter`), integrated in `src/api/routes/chat_routing.py`.

Feature flag: `binding_routing`.

## Tool Requirement Detection (February 2026)

`detect_tool_requirement()` in `chat_routing.py` is a keyword-based heuristic that identifies prompts requiring tool invocation. It populates `tool_required` and `tool_hint` fields on `RoutingResult`, which downstream execution stages can use to force first-turn tool use or select REPL mode.

### Keyword Map

```python
_TOOL_REQUIRED_KEYWORDS = {
    "search for": "grep",
    "find in": "grep",
    "grep": "grep",
    "look up": "peek",
    "read the file": "peek",
    "list the files": "list_dir",
    "calculate": None,       # Needs tools but no specific hint
    "compute": None,
    "run the": "run_shell",
}
```

### RoutingResult Integration

```python
@dataclass
class RoutingResult:
    # ... existing fields ...
    tool_required: bool = False   # True when task needs tools
    tool_hint: str | None = None  # Specific tool name if deterministic (e.g., "grep")
```

When `tool_required=True`, execution stages can: (a) force REPL mode even if the heuristic would have selected direct, and (b) inject the `tool_hint` into the system prompt so the model knows which tool to invoke first.

**Files**: `src/api/routes/chat_routing.py` (`detect_tool_requirement`, `_TOOL_REQUIRED_KEYWORDS`), `src/api/routes/chat_utils.py` (`RoutingResult`)

## REPL Defensive Mechanisms (February 2026)

The REPL execution loops have three defensive mechanisms that prevent infinite loops, wasted tokens, and unnecessary escalation.

### Comment-Only Guard

When the model generates Python code that is entirely comments (`# reasoning...`), the REPL executes it as valid Python — but produces no output, no error, no `FINAL()`. The turn loop continues indefinitely.

**Detection**: `_is_comment_only(code)` checks if all non-blank lines start with `#`.

**Response**: Returns explicit error to trigger `consecutive_failures` and eventually escalation: "Your output was all comments — no executable code ran."

Applied in all 4 REPL loops: `_execute_turn()` (graph), `_execute_react()` (ReAct), specialist delegation, and architect mini-REPL.

### FINAL() Rescue

When the model generates `FINAL("C")` inside code that has a syntax error before the FINAL line, the REPL crashes at the error and FINAL() never executes. The `_FINAL_RE` regex extracts the answer directly from the raw LLM output when REPL fails but FINAL() is present, preventing unnecessary escalation.

### Early-Stop Streaming

The model often writes `FINAL("D")` or `D|B` but continues generating hundreds of tokens of post-answer rambling. A three-layer `StopIteration`-based stream abort mechanism:

1. **`llama_server.py`**: `infer_stream_text()` catches `StopIteration` from `on_chunk` callback → breaks SSE loop
2. **`inference.py`**: When `_early_stop_check` is set on the primitives instance, creates composite callback that accumulates text and raises `StopIteration` when the check returns True
3. **Call sites**: Set `_early_stop_check` with appropriate regex before `llm_call()`, clear in `finally` block:
   - TOON regex for architect routing decisions (`D|[A-D]`, `D|.+`, `I|brief:`)
   - `_FINAL_RE` for REPL FINAL() detection in all 3 REPL loops

**Regex pitfall (fixed 2026-02-09)**: The original TOON regex used `D\|.{2,}` which required 2+ characters after `D|`. Single-character answers like `D|7` were missed because `.{2,}` needs 2+ chars and `.` doesn't match `\n`. Changed to `D\|.+` (1+ characters). Bare `D|` (partial streaming output) still correctly doesn't match.

### Architect Delegation Prompt Design

The architect investigate prompt must present `D|` and `I|` as **mutually exclusive alternatives**, not as a fill-in-the-blank template. MoE models (Qwen3-235B with expert reduction) are especially prone to echoing both formats when shown side by side.

**Anti-pattern** (causes template echoing):
```
D|<your answer>
I|brief:<spec>|to:coder_escalation
```

**Correct pattern** (bullet-list alternatives):
```
- Direct answer: D|<answer>
- Delegate to specialist: I|brief:<spec>|to:<role>
```

Combined with explicit instruction: "Output ONE line only. Do NOT output both D| and I|."

The architect prompt frames the role as "software architect" whose job is to design solutions (approach, data structures, algorithm, edge cases) for a coding specialist to implement. This produces architecturally useful briefs rather than problem restatements.

## Escalation Reduction via Skill Propagation (February 2026)

SkillBank creates a knowledge pipeline that monotonically reduces escalation rates for recurring task categories:

```
Architect solves task → trajectory stored in EpisodicStore
    → DistillationPipeline extracts escalation skill
    → SkillRetriever injects skill into worker prompts
    → Worker handles similar task directly (no escalation)
```

### Mechanism

When an architect solves a task that a worker failed, the distillation pipeline produces an `escalation` or `routing` skill encoding the architect's approach. On subsequent similar tasks:

1. `SkillRetriever` finds the skill via FAISS similarity
2. Skill principle is injected into the worker's prompt context
3. Worker applies the skill's guidance, potentially avoiding escalation

### Interaction with THINK_HARDER

Before escalating, the pipeline tries `THINK_HARDER` (same model, doubled token budget + CoT prefix). Skills complement this: the CoT prompt now includes relevant skill principles, improving the chance that THINK_HARDER succeeds and avoids escalation entirely.

### Expected Impact

For recurring task categories (e.g., specific debugging patterns, code generation patterns), escalation rate should decrease monotonically as skills accumulate. Novel tasks still escalate normally — skills only help with patterns seen before.

See [Chapter 27](27-skillbank-experience-distillation.md) for full SkillBank architecture.

## Vision Pipeline Routing (2026-02-09)

The vision pipeline has a critical routing requirement: VL models (Qwen2.5-VL-7B on port 8086, Qwen3-VL-30B on port 8087) need multimodal payloads with base64-encoded images. The standard text-only paths (`_execute_direct`, `_execute_repl`) discard `image_path` from the request.

**Stage 7.5 (`_execute_vision_multimodal`)** intercepts vision-role requests with image data and routes them to the appropriate multimodal handler:

| Mode | Handler | What It Does |
|------|---------|-------------|
| `direct` | `_handle_vision_request()` | OCR preprocessing → multimodal chat completion → VL answer |
| `repl` | `_vision_react_mode_answer()` | Multimodal ReAct loop with OCR/calculate tools |

Falls through to text-only mode on exception (graceful degradation).

**File**: `src/api/routes/chat_pipeline/vision_stage.py`

## Try-Cheap-First Speculative Pre-Filter (February 2026)

Before the normal routing pipeline runs, a speculative pre-filter attempts the task with the cheapest HOT model (7B worker at 44 t/s). If the answer passes a quality gate, it is returned 2-3x faster. On failure, the request falls through to the normal pipeline completely unchanged.

### Pipeline Position

```
Request → _route_request() → [Try-Cheap-First] → _execute_direct / _execute_repl / ...
                                    |
                                    +-- Pass quality gate → return ChatResponse (fast path)
                                    +-- Fail quality gate → fall through (normal pipeline)
```

The pre-filter inserts as Stage 7.9, BEFORE the existing execution stages. The downstream escalation chain (worker -> coder -> architect, `max_retries=3`, `max_escalations=2`) is completely untouched.

### Configuration

```python
# src/config.py — ChatConfig
try_cheap_first_enabled: bool = True
try_cheap_first_phase: str = "A"        # A=try all, B=Q-value guided, C=fully learned
try_cheap_first_role: str = "worker_explore"
try_cheap_first_max_tokens: int = 1024  # Keep short to minimize waste
try_cheap_first_quality_threshold: float = 0.6
try_cheap_first_q_threshold: float = 0.65  # Min Q-value for Phase B/C
```

### Skip Conditions

The pre-filter is bypassed when any of these hold:
- `force_mode` or `force_role` is set on the request
- The initial role is already a cheap worker (`worker_explore`, `worker_math`, `worker_vision`)
- Execution mode is `delegated`
- Phase B/C: MemRL Q-value for the cheap role is below `try_cheap_first_q_threshold`

### Quality Gate

The cheap answer must pass `try_cheap_first_quality_threshold` (default 0.6) via the existing `QScorer`. Answers that are too short, contain hedging, or fail structural checks are rejected and the normal pipeline handles the request.

**Files**: `src/api/routes/chat.py` (`_try_cheap_first`), `src/config.py` (`ChatConfig`)

## Early-Stop Timing Telemetry (2026-02-09)

When early-stop streaming aborts generation (REPL's `FINAL()` detection raises `StopIteration`), the SSE `stop: true` event (which carries `timings`) is never reached. This caused `generation_ms=0` for 79/99 REPL results while `tokens_generated` was correct.

**Fix**: On early-stop break, compute timing from wall clock elapsed time in `infer_stream_text()`. This is wall-clock time (includes prompt eval + HTTP overhead), not pure generation time, but far better than 0 for TPS estimation.

**File**: `src/backends/llama_server.py` (early-stop branch in `infer_stream_text`)

## References

### Implementation

1. `src/graph/nodes.py`: Pydantic-graph node classes with escalation logic
2. `src/graph/state.py`: TaskState, TaskDeps, TaskResult, GraphConfig
3. `src/graph/graph.py`: Graph singleton, `run_task()`, `generate_mermaid()`
4. `src/escalation.py`: Unified escalation policy (EscalationAction, ErrorCategory, EscalationConfig, THINK_HARDER)
5. `src/proactive_delegation/`: Complexity-aware routing package (types, complexity, review_service, delegator)
6. `src/roles.py`: Role definitions, escalation chains, and fallback map
7. `src/graph/approval_gate.py`: Halt/resume protocol types and approval gates
8. `src/routing_bindings.py`: Priority-ordered routing bindings
9. `orchestration/repl_memory/retriever.py`: TwoPhaseRetriever (Phase 2.5 cache affinity), HybridRouter
10. `src/api/routes/chat.py`: Try-cheap-first speculative pre-filter (`_try_cheap_first`)
11. `src/api/routes/chat_routing.py`: Tool requirement detection (`detect_tool_requirement`)
12. `src/api/routes/chat_utils.py`: RoutingResult (tool_required, tool_hint fields)

### Theoretical Foundations

5. Sutton, R. S., & Barto, A. G. (2018). *Hierarchical Reinforcement Learning*. In *Reinforcement Learning: An Introduction* (2nd ed., Chapter 13). MIT Press.

6. Russell, S., & Norvig, P. (2020). *Planning and Acting in the Real World*. In *Artificial Intelligence: A Modern Approach* (4th ed., Chapter 11). Pearson.

### Related Systems

7. Kubernetes Pod Disruption Budgets (failure budget): https://kubernetes.io/docs/concepts/workloads/pods/disruptions/

8. AWS Step Functions (state machine orchestration): https://aws.amazon.com/step-functions/

---

*Previous: [Chapter 17: Memory Seeding](17-memory-seeding.md)* | *Next: [Chapter 19: Procedure Registry](19-procedure-registry.md)*

## Conditional Schema Escalation (2026-02)

Schema failures now follow a conditional policy:

1. Retry while retry budget remains.
2. If retries are exhausted and failure signature indicates capability gap (schema/validation mismatch), allow escalation.
3. Parser/transient formatting signatures remain retry-only and do not escalate.

This replaced the previous hard block on all schema escalation and is implemented in:

- `src/escalation.py`
- `src/graph/helpers.py`

## Runtime Risk Gate and Prior/Posterior Blend (2026-02)

Routing now supports a strict runtime risk-gate contract in the hybrid router:

1. Compute effective threshold from calibrated/base confidence + conformal margin.
2. If enabled and confidence is below threshold, route uses `risk_abstain_escalate` to a configured abstain target role.
3. Emit risk provenance fields in telemetry:
   - `risk_gate_action`
   - `risk_gate_reason`
   - `risk_budget_id`

Heuristics are integrated as probabilistic priors in posterior scoring (instead of rules-only nudges):

- Posterior = learned selection score + prior term (`prior_strength`).
- Telemetry decomposition:
  - `prior_term_topk`
  - `posterior_score_topk`
  - `learned_evidence_topk`
  - `cost_term_topk`

THINK_HARDER regulation now uses an adaptive envelope in graph helpers:

- token budget and temperature scale by per-role expected ROI
- cooldown and EMA marginal-utility gating reduce repeated low-yield expansions
- decision artifacts emit ROI/token/temperature diagnostics for analysis

## Literature Mapping (Architecture Review Alignment)

This chapter's routing and escalation mechanics are grounded in the following research-backed ideas:

| Review Theme | Practical Interpretation | Code Anchors |
|--------------|--------------------------|--------------|
| Hierarchical/delegated routing | Architect envelope can decompose tasks and route to specialists | `src/proactive_delegation/delegator.py`, `src/api/routes/chat_pipeline/proactive_stage.py` |
| Conditional escalation by failure signature | Schema/format failures are treated differently from reasoning failures | `src/escalation.py`, `src/graph/helpers.py` |
| Risk-aware abstain/escalate behavior | Low-confidence paths can abstain and escalate under strict controls | `orchestration/repl_memory/retriever.py`, `src/api/routes/chat_pipeline/routing.py` |
| THINK_HARDER regulation | Controlled test-time compute expansion with ROI/cooldown governance | `src/graph/helpers.py`, `src/graph/state.py` |
| Workspace-centered coordination | Shared state reduces multi-agent drift during long chains | `src/graph/state.py`, `src/graph/helpers.py` |

## Additional Literature (From Architecture Review)

These references motivate the escalation/delegation strategy and failure taxonomy:

1. DeepSeek-AI (2025). Router-R1: Multi-round routing/aggregation behavior. https://openreview.net/forum?id=DWf4vroKWJ
2. Xue et al. (2025). Conformal Risk-Controlled Routing for Large Language Model. https://openreview.net/forum?id=lLR61sHcS5
3. SHIELDA (2025). Structured exception taxonomy for LLM agent workflows. https://arxiv.org/html/2508.07935v1
4. Baars (1988). A Cognitive Theory of Consciousness (Global Workspace Theory foundation). https://pmc.ncbi.nlm.nih.gov/articles/PMC12310485/
5. Dai, Yang, Si (2025). S-GRPO for reasoning-length regulation. https://arxiv.org/abs/2505.07686
6. Ong et al. (2024). RouteLLM (quality/cost routing tradeoffs). https://arxiv.org/abs/2406.18665

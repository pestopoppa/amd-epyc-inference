# Chapter 18: Escalation, Failure Routing & Proactive Delegation

## Introduction

The orchestrator uses three complementary systems for task routing and failure handling: unified escalation policy (`escalation.py`), legacy failure router with MemRL integration (`failure_router.py`), and proactive delegation with complexity-aware routing (`proactive_delegation.py`). Together they implement reactive escalation chains (worker → coder → architect) and proactive task decomposition.

## Unified Escalation Policy

### EscalationPolicy Architecture

```python
class EscalationAction(str, Enum):
    RETRY = "retry"      # Retry with same role
    ESCALATE = "escalate"  # Escalate to next tier
    FAIL = "fail"        # Terminal failure
    SKIP = "skip"        # Skip the gate/step (optional gates only)

class ErrorCategory(str, Enum):
    CODE = "code"              # Syntax/type errors, test failures
    LOGIC = "logic"            # Wrong output, failed assertions
    TIMEOUT = "timeout"        # Gate/execution timeout
    SCHEMA = "schema"          # IR/JSON schema violations
    FORMAT = "format"          # Style/format issues
    EARLY_ABORT = "early_abort"  # Model predicted failure, skip retries
    UNKNOWN = "unknown"
```

**Early Abort**: When a model shows failure signs (incomplete generation, error patterns), immediately escalate instead of wasting retries. Detected via gate checks or output analysis.

### Decision Rules

| Error Category | First Failure | Second Failure | Third+ Failure |
|----------------|---------------|----------------|----------------|
| CODE, LOGIC | RETRY | RETRY | ESCALATE |
| FORMAT, SCHEMA | RETRY | RETRY | FAIL (never escalate) |
| TIMEOUT (optional gate) | SKIP | — | — |
| TIMEOUT (required gate) | RETRY | ESCALATE | FAIL |
| EARLY_ABORT | ESCALATE | — | — |

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

## Failure Router with Learned Escalation

### Legacy Failure Router

The `FailureRouter` class predates `EscalationPolicy` but remains for MemRL integration:

```python
class FailureRouter:
    """Routes failures with optional learned escalation (Phase 4 MemRL)."""

    ESCALATION_CHAINS: dict[str, EscalationChain] = {
        "worker": EscalationChain("worker", "coder", max_retries=2, max_escalations=2),
        "coder": EscalationChain("coder", "architect", max_retries=2, max_escalations=1),
        "architect": EscalationChain("architect", None, max_retries=3, max_escalations=0),
    }

    # Map specific roles to generic chains
    ROLE_TO_CHAIN: dict[str, str] = {
        "worker_general": "worker",
        "coder_primary": "coder",
        "architect_general": "architect",
    }
```

### MemRL Integration (Phase 4)

```python
class LearnedEscalationPolicy:
    """Queries episodic memory for similar failures."""

    def query(self, context: FailureContext) -> LearnedEscalationResult:
        failure_dict = {
            "role": context.role,
            "error_category": context.error_category.value,
            "gate_name": context.gate_name,
            "error_message": context.error_message[:500],
        }

        # Two-phase retrieval from episodic memory
        results = self.retriever.retrieve_for_escalation(failure_dict)

        if not self.retriever.should_use_learned(results, min_samples=3):
            return LearnedEscalationResult(should_use_learned=False)

        # Parse best action from memory
        best = results[0]
        action_parts = best.memory.action.split(":")
        suggested_action = action_parts[0]  # "retry", "escalate", "fail"
        suggested_role = action_parts[1] if len(action_parts) > 1 else None

        return LearnedEscalationResult(
            should_use_learned=True,
            suggested_action=suggested_action,
            suggested_role=suggested_role,
            confidence=best.combined_score,
        )
```

**Hybrid Strategy**: Try learned escalation first (if confident), fall back to rule-based when cold-starting or low confidence.

### Strategy Tracking

```python
# Track usage for monitoring
self._strategy_counts = {"learned": 0, "rules": 0}

# In route_failure():
if learned_result.should_use_learned:
    self._strategy_counts["learned"] += 1
    return learned_decision
else:
    self._strategy_counts["rules"] += 1
    return rule_based_decision
```

Monitor convergence: As MemRL matures, `learned / (learned + rules)` should increase.

## 3-Way Confidence Routing (February 2026)

### Overview

The Unified Execution Model introduces 3-way confidence routing for faithful probability estimation. Instead of rigid mode-based routing, the frontdoor estimates P(success|action) for three approaches:

| Approach | Meaning | Maps To |
|----------|---------|---------|
| **SELF:direct** | Handle without tools | `frontdoor` with `mode=direct` |
| **SELF:repl** | Handle with tools, no delegation | `frontdoor` with `mode=repl`, `allow_delegation=False` |
| **ARCHITECT** | Escalate for complex reasoning | `architect_coding` or `architect_general` |
| **WORKER** | Delegate to faster workers | Scored via delegation chain attribution |

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

## References

### Implementation

1. `src/escalation.py`: Unified escalation policy (336 lines)
2. `src/failure_router.py`: Legacy router with MemRL (787 lines)
3. `src/proactive_delegation/`: Complexity-aware routing package (types, complexity, review_service, delegator)
4. `src/roles.py`: Role definitions and escalation chains

### Theoretical Foundations

5. Sutton, R. S., & Barto, A. G. (2018). *Hierarchical Reinforcement Learning*. In *Reinforcement Learning: An Introduction* (2nd ed., Chapter 13). MIT Press.

6. Russell, S., & Norvig, P. (2020). *Planning and Acting in the Real World*. In *Artificial Intelligence: A Modern Approach* (4th ed., Chapter 11). Pearson.

### Related Systems

7. Kubernetes Pod Disruption Budgets (failure budget): https://kubernetes.io/docs/concepts/workloads/pods/disruptions/

8. AWS Step Functions (state machine orchestration): https://aws.amazon.com/step-functions/

---

*Previous: [Chapter 17: Memory Seeding](17-memory-seeding.md)* | *Next: [Chapter 19: Procedure Registry](19-procedure-registry.md)*

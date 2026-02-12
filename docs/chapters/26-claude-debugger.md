# Chapter 26: Claude-in-the-Loop Debugger

Automatic pipeline debugging during 3-way evaluation. A persistent Claude Code session monitors inference diagnostics, identifies anomalies, applies prompt/code fixes via hot-swap, and verifies fixes with a 3-part mini regression suite.

## Architecture Overview

```
seed_specialist_routing.py (eval loop)
  ├── evaluate_question_3way()  → role_results, rewards, metadata
  ├── build_diagnostic()        → anomaly signals + raw answer + tap offset
  ├── debugger.add_diagnostic() → batches diagnostics (default: 5 per batch)
  ├── debugger.end_question()   → flushes urgent batches early
  │
  │   [Background: Claude subprocess]
  │   ├── snapshot files (SHA-1 of first 8KB)
  │   ├── Claude analyzes diagnostics + tap output
  │   ├── Claude edits .md prompts or .py code
  │   ├── diff snapshots → detect actual changes
  │   ├── hot-restart API if .py changed
  │   ├── queue failed questions for retry
  │   └── record to change_log (JSONL audit trail)
  │
  └── pop_retries() → mini regression suite
      ├── VERIFY:     exact failed questions
      ├── GENERALIZE: 2 fresh unseen per affected suite
      └── REGRESS:    2 previously-passing per affected suite
```

## 12 Anomaly Signals

Computed per answer in `src/pipeline_monitor/anomaly.py`.

| Signal | Weight | Trigger |
|--------|--------|---------|
| `repetition_loop` | 1.0 | Trigram unique ratio < 0.4 (degeneration) |
| `template_echo` | 1.0 | Both `D\|` AND `I\|` prefixes in same answer |
| `format_violation` | 1.0 | Architect role, no `D\|` or `I\|`, answer > 50 chars |
| `near_empty` | 1.0 | Answer < 5 tokens, no error (except MCQ) |
| `delegation_format_error` | 1.0 | `I\|` present but missing `brief:` field |
| `vision_blindness` | 1.0 | Vision role, < 10 tokens |
| `comment_only` | 0.5 | All code lines are `#`-prefixed or blank |
| `self_doubt_loop` | 0.5 | > 3 restart phrases ("Actually", "Wait", "Let me reconsider"...) |
| `think_tag_leak` | 0.5 | `<think>` in answer text |
| `excessive_tokens` | 0.5 | MCQ with > 2000 tokens generated |
| `self_escalation` | 0.5 | Consecutive duplicate roles in role_history |
| `silent_execution` | 0.5 | Tools used >= 1, no error, empty answer |

**Scoring**: `anomaly_score = max(triggered weights)`. Range [0, 1]. Score >= 1.0 sets urgent flag.

## Batching & Background Execution

The debugger accumulates diagnostics and dispatches them to Claude in batches:

1. `add_diagnostic(diag)` appends to batch, collects any finished background result
2. When `len(batch) >= batch_size` (default 5), calls `_dispatch()`:
   - Captures file snapshot (SHA-1 hashes of dirty files)
   - Launches `claude --session-id {id} --json` as background subprocess
   - First invocation includes full system prompt; subsequent reuse session for context accumulation
3. `end_question()` triggers early flush if `_urgent=True` (any signal with weight 1.0)
4. Background Claude runs concurrently — eval loop continues scoring the next question

### Prompt Construction

Each batch prompt includes per-diagnostic:
- Question ID, pass/fail, config, role, mode
- Expected answer, scoring method, tokens generated, error
- Triggered anomalies + overall score
- Full role history and tool calls
- **Inference log** (inlined, up to 12K chars): raw prompt/response for every LLM call in the delegation chain — architect TOON decisions, specialist outputs, escalation triggers
- **REPL execution log** (inlined, up to 4K chars): code executed and stdout/stderr — NameErrors, SyntaxErrors, import failures, silent execution issues
- Full answer text (truncated at 2000 chars)

Both logs are captured via byte-range offsets recorded around each `/chat` API call in `seeding_eval.py`, then read inline by `_read_tap_inline()` in the debugger. If a tap file is missing or offset is stale (e.g., TUI restart), the debugger gracefully omits the log section.

## Fix Application & Hot-Restart

When Claude's subprocess completes:

1. **Snapshot diff**: Compare file hashes before/after — detects what Claude actually changed vs pre-existing dirty state
2. **Prompt hot-swap**: If `.md` files changed in `orchestration/prompts/`, next inference request picks up the edit automatically (no restart needed, ~1ms disk read)
3. **Code hot-restart**: If `.py` files changed, runs `orchestrator_stack.py reload orchestrator` (uvicorn restart, ~10s)
4. **Auto-commit** (optional, `--debug-auto-commit`): `git add -A && git commit` per batch, enabling easy `git revert`

## Retry-After-Fix: Mini Regression Suite

After Claude applies fixes that modify files, the eval loop runs a 3-part verification to prevent overfitting:

### Phase 1: VERIFY
Re-test the exact questions that failed and triggered the fix. Confirms the fix addresses the original failure. Each question retried at most once (tracked by `_retried` set).

### Phase 2: GENERALIZE
Sample 2 fresh unseen questions per affected suite. These questions were never seen by Claude — it cannot tailor fixes to them. If verify passes but generalize fails, the fix was too narrow.

### Phase 3: REGRESS
Sample up to 2 previously-passing questions per affected suite. If these now fail, the fix introduced a regression.

### Metadata

All retry results carry:
```python
meta["is_retry"] = True
meta["retry_tag"] = "verify" | "generalize" | "regress"
meta["retry_batch_id"] = N  # which batch triggered the fix
```

Retry diagnostics feed back into the debugger. Generalize/regress failures can trigger new fix cycles. Verify questions are capped at 1 retry to prevent infinite loops.

## MemRL Interaction

Debug mode injects rewards into episodic memory (`--debug` is orthogonal to `--dry-run`). This means debug runs simultaneously seed MemRL while fixing bugs.

### Q-Value Convergence on Retried Questions

When a question is retried after a fix, the same `(task_description, action)` pair gets a second reward injection:

1. **Pre-fix** (failed): `_inject_3way_rewards_http(reward=0.0)` → `score_external_result()` creates memory with `initial_q = 0.5`
2. **Post-fix** (passes): `_inject_3way_rewards_http(reward=1.0)` → `score_external_result()` finds existing memory (cosine similarity >= 0.85, same action) → TD update

The TD-learning update:
```
Q_decayed = 0.5 + (Q_old - 0.5) * decay_rate ^ days_elapsed
Q_new = Q_decayed + α(reward - Q_decayed)
```

With α=0.1 and both updates in the same run (days_elapsed ≈ 0):
- After fail:  Q = 0.50
- After pass:  Q = 0.50 + 0.1(1.0 - 0.50) = 0.55

This is **intentional**: a question that required a pipeline fix to pass should have a lower Q-value than one that passed on first attempt (Q ≈ 1.0). MemRL correctly learns "this task type is harder for this route."

### Generalize Questions

Generalize questions are fresh — they create new memories entirely. If they pass post-fix, their Q-values start at `0.5 + 1.0 * 0.5 = 1.0` — clean signals uncontaminated by pre-fix failures.

## Audit Trail

All debugger activity is logged to `/mnt/raid0/llm/claude/logs/debug_changes.jsonl`:

```json
{
  "ts": "2026-02-10T03:14:15.926535",
  "session_id": "abc123",
  "batch_id": 7,
  "questions_analyzed": ["math/q42", "coder/q17"],
  "anomalies_seen": {"math/q42": {"template_echo": true}},
  "claude_reasoning": "Template echo: architect outputs both D| and I| ...",
  "files_modified": ["orchestration/prompts/architect_investigate.md"],
  "git_diff": "diff --git a/...",
  "git_commit_sha": "a1b2c3d"
}
```

File-locked (POSIX `fcntl.flock`), append-only. Full git diff stored for rewind via `git apply --reverse`.

Diagnostics per-answer are logged separately to `/mnt/raid0/llm/claude/logs/seeding_diagnostics.jsonl`.

## CLI Flags

```bash
# Live debugging (Claude analyzes every 5 answers, applies fixes)
python scripts/benchmark/seed_specialist_routing.py --3way --continuous --debug

# Dry run (log diagnostics, don't invoke Claude)
--debug --debug-dry-run

# Auto-commit each fix batch
--debug --debug-auto-commit

# Custom batch size and threshold
--debug --debug-batch-size 10 --debug-threshold 0.5
```

## Extended Observation Patterns (February 2026)

Diagnostic records now include additional tunable fields that the ClaudeDebugger prompt builder surfaces when present. These fields provide Claude with richer context about the orchestrator's decision-making during each inference cycle.

| Field | Type | Description |
|-------|------|-------------|
| `cost_dimensions` | dict | Breakdown of cost signals: model tier, memory tier, elapsed ratio, tokens |
| `think_harder_attempted` | bool | Whether the pipeline invoked extended thinking mode |
| `think_harder_succeeded` | bool | Whether extended thinking produced a correct answer after standard mode failed |
| `cheap_first_attempted` | bool | Whether cheap-first cascade was tried (frontdoor before escalation) |
| `cheap_first_passed` | bool | Whether cheap-first produced a correct answer (no escalation needed) |
| `grammar_enforced` | bool | Whether json_schema/grammar constraint was active on the request |
| `parallel_tools_used` | int | Number of parallel tool invocations in REPL mode |
| `cache_affinity_bonus` | float | RadixAttention cache hit benefit (0.0 = miss, 1.0 = full prefix cached) |

The ClaudeDebugger prompt builder (`_build_batch_prompt()`) conditionally includes these fields only when they carry non-default values. This keeps batch prompts compact for simple cases while giving Claude full visibility into complex routing and cost decisions when debugging failures. For example, a `think_harder_attempted=True, think_harder_succeeded=False` pair signals that the pipeline already tried its escalation strategy and still failed -- Claude should look for prompt or tool issues rather than suggesting "try harder."

## File Locations

| File | Purpose |
|------|---------|
| `src/pipeline_monitor/claude_debugger.py` | ClaudeDebugger class |
| `src/pipeline_monitor/anomaly.py` | 12 anomaly signal detectors |
| `src/pipeline_monitor/change_log.py` | JSONL audit trail writer |
| `src/pipeline_monitor/diagnostic.py` | Diagnostic record builder |
| `src/inference_tap.py` | TapWriter: append-only inference log (prompt/response per LLM call) |
| `src/graph/nodes.py` | REPL tap writers: `_tap_write_repl_exec()`, `_tap_write_repl_result()` |
| `/mnt/raid0/llm/tmp/inference_tap.log` | Inference tap file (created by TUI, deleted on exit) |
| `/mnt/raid0/llm/tmp/repl_tap.log` | REPL tap file (truncated on TUI start, deleted on exit) |
| `logs/debug_changes.jsonl` | Audit trail (fixes + reasoning) |
| `logs/seeding_diagnostics.jsonl` | Per-answer diagnostics |
| `orchestration/prompts/` | Hot-swappable prompt templates |

# Agent Instructions for Orchestration Codebase

**Purpose**: This document provides detailed guidance for AI agents working on this codebase. It captures lessons learned, common errors to avoid, and best practices discovered during development and refactoring.

---

## Critical Rules

### 1. NEVER Write to Root Filesystem

```bash
# WRONG - Will crash the system
/home/user/models/...
/tmp/large_file.bin
~/.cache/...

# CORRECT - Always use RAID
/mnt/raid0/llm/models/...
/mnt/raid0/llm/tmp/...
/mnt/raid0/llm/cache/...
```

### 2. NEVER Use `pytest -n auto`

This machine has 192 threads. `pytest -n auto` spawns ~192 workers → OOM crash.

```bash
# Safe
pytest tests/
pytest tests/ -n 4

# DANGEROUS - DO NOT USE
pytest tests/ -n auto
```

### 3. ALWAYS Use Feature Flags for Optional Modules

```python
# WRONG - Imports execute at module load
from orchestration.repl_memory import TaskEmbedder
embedder = TaskEmbedder()  # Loads 0.5B model!

# CORRECT - Check feature flag first
from src.features import features
if features().memrl:
    from orchestration.repl_memory import TaskEmbedder
    embedder = TaskEmbedder()
else:
    embedder = None
```

### 4. ALWAYS Use Role Enum, Not Strings

```python
# WRONG - Typo causes silent bug
role = "codre_primary"  # Typo!

# CORRECT - Compile-time error if wrong
from src.roles import Role
role = Role.CODER_PRIMARY
```

### 5. ALWAYS Log Exceptions (Never Silently Swallow)

```python
# WRONG - Error disappears
try:
    do_something()
except Exception:
    pass

# CORRECT - Error is logged
try:
    do_something()
except Exception as e:
    logger.warning(f"Operation failed: {e}", exc_info=True)
```

---

## Code Modification Guidelines

### Adding a New Feature

1. **Add feature flag** in `src/features.py`:
   ```python
   my_feature: bool = False
   ```

2. **Add environment variable** in `get_features()`:
   ```python
   "my_feature": _env_bool("MY_FEATURE", defaults["my_feature"]),
   ```

3. **Guard all feature code**:
   ```python
   if features().my_feature:
       # Feature-specific code
   ```

4. **Add tests for both enabled/disabled states**

5. **Document in docs/ARCHITECTURE.md**

### Adding a New Role

1. **Add to `src/roles.py`**:
   ```python
   MY_ROLE = "my_role"
   """Description."""
   ```

2. **Add tier mapping**:
   ```python
   Role.MY_ROLE: Tier.B,
   ```

3. **Add escalation mapping** (if applicable):
   ```python
   Role.MY_ROLE: Role.ARCHITECT_GENERAL,
   ```

4. **Add to `orchestration/model_registry.yaml`**

### Modifying Escalation Logic

**Use `src/escalation.py` and `src/graph/nodes.py`.** Do not modify:
- `api.py` ESCALATION_ROLES (deprecated)
- `executor.py` escalation methods (should delegate to escalation.py)

```python
# CORRECT way to make escalation decisions
from src.escalation import EscalationPolicy, EscalationContext
policy = EscalationPolicy()
decision = policy.decide(context)
```

### Modifying the API

**Current structure** (will be split in future refactoring):
- All endpoints in `api.py`
- Pydantic models inline
- State in global `_state`

**When modifying:**
1. Keep changes atomic (one concern per commit)
2. Verify syntax: `python3 -m py_compile src/api.py`
3. Test mock mode first
4. Test real mode if llama-server available

---

## Common Errors and Fixes

### Error: Module Import Fails in Tests

**Symptom**: `ModuleNotFoundError` for MemRL components

**Cause**: Test runs with features disabled but code imports unconditionally

**Fix**: Use conditional imports:
```python
if features().memrl:
    from orchestration.repl_memory import TaskEmbedder
```

### Error: Silent Failure in Background Task

**Symptom**: Q-scoring never happens, no errors visible

**Cause**: Exception swallowed in `_background_cleanup()`

**Fix**: Always log exceptions:
```python
except Exception as e:
    logger.warning(f"Background cleanup error: {e}", exc_info=True)
```

### Error: Race Condition in Statistics

**Symptom**: Inconsistent request counts under load

**Cause**: Non-atomic increment on shared state

**Fix**: Use thread-safe methods:
```python
_state.increment_request(mock_mode, turns)  # Thread-safe
# NOT: _state.total_requests += 1  # Race condition!
```

### Error: REPL Sandbox Escape

**Symptom**: User code accesses forbidden operations

**Cause**: Regex-based validation bypassed

**Fix**: Already fixed with AST-based validation in `ASTSecurityVisitor`

### Error: Role String Typo

**Symptom**: Unknown role error or silent routing failure

**Cause**: Magic string typo: `"codre_primary"` instead of `"coder_primary"`

**Fix**: Use Role enum:
```python
from src.roles import Role
role = Role.CODER_PRIMARY  # IDE catches typos
```

### Error: Memory Exhaustion

**Symptom**: System crash during testing

**Cause**: Parallel test execution loading models in each worker

**Fix**: Never use `pytest -n auto`. Use sequential or limited parallelism.

---

## File Locations

### Where to Find Things

| Need | Location |
|------|----------|
| Feature flags | `src/features.py` |
| Role definitions | `src/roles.py` |
| Escalation logic | `src/escalation.py` |
| API app factory | `src/api/__init__.py` |
| API routes | `src/api/routes/` |
| API models | `src/api/models/` |
| API services | `src/api/services/` |
| API state | `src/api/state.py` |
| LLM abstraction | `src/llm_primitives.py` |
| REPL sandbox | `src/repl_environment.py` |
| Model registry | `orchestration/model_registry.yaml` |
| Architecture docs | `docs/ARCHITECTURE.md` |
| Open source recommendations | `docs/reference/OPEN_SOURCE_RECOMMENDATIONS.md` |

### Where to Put New Things

| Creating | Put In |
|----------|--------|
| New feature flag | `src/features.py` |
| New role | `src/roles.py` |
| New escalation rule | `src/escalation.py` |
| New API endpoint | `src/api/routes/<domain>.py` |
| New request model | `src/api/models/requests.py` |
| New response model | `src/api/models/responses.py` |
| New API service | `src/api/services/<service>.py` |
| New backend | `src/backends/` |
| New tool | `src/tools/` |
| Unit tests | `tests/unit/` |
| Integration tests | `tests/integration/` |

---

## Testing Checklist

Before committing changes:

- [ ] `python3 -m py_compile <modified_files>` - Syntax check
- [ ] Feature works with flag enabled
- [ ] Feature works with flag disabled (or gracefully degrades)
- [ ] No silent exception swallowing
- [ ] Role enum used (not magic strings)
- [ ] Thread-safe if touching shared state
- [ ] Documentation updated if adding features/roles

---

## Quick Commands

```bash
# Verify syntax of all modified files
python3 -m py_compile src/api.py src/features.py src/roles.py

# Run tests (safe)
pytest tests/ -v

# Start API in mock mode
ORCHESTRATOR_MOCK_MODE=1 uvicorn src.api:app --port 8000

# Start API in production mode (requires llama-server)
ORCHESTRATOR_MEMRL=1 ORCHESTRATOR_TOOLS=1 uvicorn src.api:app --port 8000

# Check feature flags
python3 -c "from src.features import features; print(features().summary())"

# List all roles
python3 -c "from src.roles import Role; print([r.value for r in Role])"
```

---

## Asking for Help

If stuck:

1. **Check docs/ARCHITECTURE.md** for system overview
2. **Check CLAUDE.md** for project-specific rules
3. **Check this file** for common errors
4. **Search existing code** for similar patterns
5. **Document what you tried** before escalating

---

## Summary

| Do | Don't |
|----|-------|
| Use Role enum | Use magic role strings |
| Use features() | Import optional modules unconditionally |
| Log exceptions | Silent `except: pass` |
| Use thread-safe methods | Direct state mutation |
| Write to /mnt/raid0/ | Write to root filesystem |
| Use limited parallelism | Use `pytest -n auto` |
| Check feature flag | Assume module available |
| Use escalation.py / graph nodes | Modify escalation inline |

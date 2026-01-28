# Open Source Tool Recommendations

**Created**: 2026-01-14
**Purpose**: Document custom-built code that could be replaced with established open-source libraries for improved reliability, security, and maintainability.

## High Priority Replacements

### 1. REPL Sandboxing → RestrictedPython ✅ IMPLEMENTED

**Status**: Implemented in `src/restricted_executor.py` with feature flag `restricted_python`

**Implementation**:
- Created `src/restricted_executor.py` with `RestrictedExecutor` class
- Added `restricted_python` feature flag to `src/features.py`
- Added factory function `create_repl_environment()` to `src/repl_environment.py`
- Maintains backward compatibility with original `REPLEnvironment`

**Usage**:
```python
# Option 1: Use factory function (respects feature flag)
from src.repl_environment import create_repl_environment
repl = create_repl_environment(context="...", use_restricted_python=True)

# Option 2: Enable via environment variable
# ORCHESTRATOR_RESTRICTED_PYTHON=1

# Option 3: Direct use of RestrictedExecutor
from src.restricted_executor import RestrictedExecutor, is_available
if is_available():
    executor = RestrictedExecutor(context="...")
    result = executor.execute("print(peek(100))")
```

**Effort**: Medium (requires testing all REPL use cases) ✅ DONE
**Risk**: Low (more secure than current approach)

---

### 2. SSE Streaming → sse-starlette ✅ IMPLEMENTED

**Status**: Implemented in `src/sse_utils.py` with automatic fallback

**Implementation**:
- Created `src/sse_utils.py` with SSE helper functions
- Updated `src/api/routes/chat.py` to use helper functions
- Automatic fallback to manual SSE when sse-starlette not installed
- Works with existing streaming feature flag

**Usage**:
```python
from src.sse_utils import (
    create_sse_response,
    token_event,
    turn_start_event,
    error_event,
    done_event,
)

async def generate():
    yield turn_start_event(turn=1, role="frontdoor")
    yield token_event("Hello")
    yield done_event()

return create_sse_response(generate())
```

**Effort**: Low (straightforward replacement) ✅ DONE
**Risk**: Low (drop-in replacement)

---

### 3. Markdown Code Extraction → markdown-it-py or mistune

**Current**: Custom regex in `src/api/services/orchestrator.py:extract_code_from_response()`

**Recommended**: [markdown-it-py](https://github.com/executablebooks/markdown-it-py) or [mistune](https://github.com/lepture/mistune)

**Why**:
- Proper CommonMark parsing
- Handles edge cases (nested blocks, escaped backticks)
- AST access for reliable code block extraction

**Migration**:
```python
# Current (regex)
code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
matches = re.findall(code_block_pattern, response, re.DOTALL)

# Recommended
from markdown_it import MarkdownIt

md = MarkdownIt()
tokens = md.parse(response)
code_blocks = [t.content for t in tokens if t.type == "fence" and t.info in ("", "python")]
```

**Effort**: Low
**Risk**: Low

---

## Medium Priority Replacements

### 4. Configuration Management → python-decouple or pydantic-settings

**Current**: Custom `src/features.py` with manual environment variable parsing

**Recommended**: [pydantic-settings](https://github.com/pydantic/pydantic-settings) (since we already use Pydantic)

**Why**:
- Type validation built-in
- Nested settings support
- .env file loading
- Consistent with existing Pydantic usage

**Migration**:
```python
# Current
def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    # ...

# Recommended
from pydantic_settings import BaseSettings

class Features(BaseSettings):
    memrl: bool = False
    tools: bool = False

    class Config:
        env_prefix = "ORCHESTRATOR_"
```

**Effort**: Low
**Risk**: Low

---

### 5. Error Classification → Custom Exception Hierarchy

**Current**: String matching in `classify_error()` function

**Recommended**: Structured exception hierarchy with error codes

**Why**:
- Type-safe error handling
- IDE autocomplete for error types
- Easier to extend and maintain
- Can use [result](https://github.com/rustedpy/result) library for Result types

**Migration**:
```python
# Current
if "syntaxerror" in error_lower:
    return ErrorCategory.CODE

# Recommended
class OrchestratorError(Exception):
    category: ErrorCategory

class SyntaxExecutionError(OrchestratorError):
    category = ErrorCategory.CODE

class LogicError(OrchestratorError):
    category = ErrorCategory.LOGIC
```

**Effort**: Medium (requires updating error handling throughout)
**Risk**: Low

---

## Low Priority (Consider for Future)

### 6. Task Scheduling → APScheduler or Celery

**Current**: `asyncio.create_task()` for background Q-scoring

**Recommendation**: For more complex scheduling needs, consider [APScheduler](https://github.com/agronholm/apscheduler)

**Current Status**: Adequate for current use case (simple periodic task)

---

### 7. Retry Logic → tenacity

**Current**: Manual retry loops with counters

**Recommended**: [tenacity](https://github.com/jd/tenacity) for retry with backoff

**Why**:
- Exponential backoff
- Configurable retry conditions
- Logging integration

**Current Status**: Low priority, current implementation is simple and works

---

## Implementation Priority

| Item | Priority | Effort | Security Impact | Status |
|------|----------|--------|-----------------|--------|
| RestrictedPython | HIGH | Medium | HIGH | ✅ DONE |
| sse-starlette | HIGH | Low | Low | ✅ DONE |
| markdown-it-py | MEDIUM | Low | Low | Pending |
| pydantic-settings | MEDIUM | Low | Low | ✅ Added to config.py |
| Exception hierarchy | MEDIUM | Medium | Low | Pending |
| APScheduler | LOW | Medium | None | Pending |
| tenacity | LOW | Low | None | Pending |

## Action Items

1. ~~**Immediate**: Evaluate RestrictedPython for REPL sandboxing (security critical)~~ ✅ DONE
2. ~~**Short-term**: Replace manual SSE with sse-starlette~~ ✅ DONE
3. **Short-term**: Replace regex markdown parsing with proper parser
4. ~~**Medium-term**: Migrate features.py to pydantic-settings~~ ✅ Added to config.py
5. **Long-term**: Refactor error handling with proper exception hierarchy

## Notes

- All recommended libraries are well-maintained with active communities
- Most replacements are drop-in with minimal code changes
- RestrictedPython is the highest priority due to security implications
- SSE replacement will improve reliability of streaming endpoints

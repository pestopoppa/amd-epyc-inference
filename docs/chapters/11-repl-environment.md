# Chapter 11: REPL Environment & Sandboxing

## Introduction

The REPL (Read-Eval-Print Loop) environment provides sandboxed Python execution for orchestrator agents. It enables agents to explore large contexts without transmitting them through the LLM, reducing token costs by orders of magnitude. Built-in functions (`peek`, `grep`, `FINAL`) manipulate context locally, while AST-based security prevents sandbox escapes.

This architecture implements RLM-style (Retrieval-augmented Language Model) orchestration where the large document remains in local memory and the agent uses code to extract relevant portions.

## Security Architecture

### AST-Based Validation

The `ASTSecurityVisitor` class analyzes parsed syntax trees before execution, making it immune to string concatenation tricks:

```python
class ASTSecurityVisitor(ast.NodeVisitor):
    FORBIDDEN_MODULES = frozenset({
        "os", "sys", "subprocess", "socket", "shutil", "pathlib",
        "tempfile", "multiprocessing", "threading", "ctypes", "pickle",
        "importlib", "builtins", "code", "codeop", "runpy", "pkgutil",
    })

    FORBIDDEN_CALLS = frozenset({
        "__import__", "eval", "exec", "compile", "open",
        "getattr", "setattr", "delattr", "hasattr",
        "globals", "locals", "vars", "dir",
    })

    FORBIDDEN_ATTRS = frozenset({
        "__class__", "__bases__", "__subclasses__", "__mro__",
        "__dict__", "__globals__", "__locals__", "__code__",
        "__builtins__", "__closure__",
    })
```

**Why AST over Regex**: String patterns like `getattr(__builtins__, '__im' + 'port__')('os')` bypass regex checks but are caught during AST analysis.

### Dual-Layer Sandboxing

The system offers two execution backends:

| Backend | Security | Performance | Dependencies |
|---------|----------|-------------|--------------|
| **Custom AST** | AST validation + timeout | Fast | Built-in only |
| **RestrictedPython** | `compile_restricted` + guards | Medium | RestrictedPython >= 7.0 |

RestrictedPython (optional) provides battle-tested sandbox used by Zope/Plone, with `PrintCollector` for stdout capture and guarded attribute access.

## Built-In Functions

### Context Exploration

| Function | Purpose | Cost | Use Case |
|----------|---------|------|----------|
| `peek(n)` | First n chars | Free | Preview document structure |
| `grep(pattern)` | Regex search | Free | Find specific content |
| `FINAL(answer)` | Terminate with result | Free | Return final answer |

### Extended Functions (Archive/Web)

| Function | Purpose | Example |
|----------|---------|---------|
| `archive_open(path)` | Extract ZIP/TAR/PDF | Process research papers |
| `archive_file(name)` | Get specific file | Read extracted document |
| `archive_search(query)` | Search across archive | Find all references |
| `web_fetch(url)` | Fetch HTTP content | Download documentation |

### Tool Calls from REPL

```python
# Call external tools via tool_call()
result = tool_call("math_simplify", {"expression": "x^2 + 2x + 1"})
# Returns: {"simplified": "(x+1)^2", "steps": [...]}
```

Agents can invoke the tool registry (41 deterministic tools) from within REPL code.

## Execution Model

### Resource Limits

```python
@dataclass
class REPLConfig:
    timeout_seconds: int = 600  # 10 min for document processing
    output_cap: int = 8192      # Max output characters
    max_grep_results: int = 100 # Prevent DoS via grep
    require_exploration_before_final: bool = False  # Force peek/grep
    min_exploration_calls: int = 1  # Minimum calls before FINAL
```

**Timeout Enforcement**: UNIX `SIGALRM` signal terminates runaway executions (600s default for document ingestion, 120s for general use).

### Trusted vs User Code Layers

| Layer | Execution Context | Restrictions |
|-------|-------------------|--------------|
| **Trusted** | Built-in functions (`peek`, `grep`, `archive_open`) | Full system access |
| **User** | Agent-generated code | AST validation, no file I/O |

Built-in functions execute in the trusted layer with access to file system (for archives) and network (for `web_fetch`), but user code is sandboxed.

## Exploration Logging

### Strategy Classification

The `ExplorationLog` tracks which primitives the agent used:

```python
strategy_types = {
    "scan": peek() > grep(),      # Sequential scanning
    "search": grep() > peek(),    # Targeted searching
    "delegated": llm_call() > 0,  # Sub-agent delegation
    "mixed": Multiple strategies
}
```

**Token Efficiency**: Calculated as `result_tokens / exploration_tokens`. Higher is better—indicates effective exploration with minimal LLM calls.

### MemRL Integration

Exploration logs feed into episodic memory for Q-learning:
- **Phase 1**: Log exploration strategy (scan/search/delegated)
- **Phase 2**: Compute token efficiency
- **Phase 3**: Update Q-values based on final outcome
- **Phase 4**: Retrieve similar explorations for future tasks

## Performance Characteristics

### Token Reduction

| Approach | Tokens | Speedup | Use Case |
|----------|--------|---------|----------|
| **Full context to LLM** | 50,000 | 1x | Baseline (avoid) |
| **REPL with peek/grep** | 500 | 100x | Document QA |
| **REPL with archive tools** | 2,000 | 25x | Multi-file analysis |
| **REPL with TOON encoding** | 890 | 56x | Structured data (55.6% reduction) |

**TOON Encoding**: Enabled by default (`use_toon_encoding=True`). Reduces tokens by ~55% on structured tool outputs with 41.8% latency improvement (TTFT benchmark).

## Security Considerations

### Attack Surface

1. **Import Bypass**: Blocked by AST analysis of `Import` and `ImportFrom` nodes
2. **Dunder Escapes**: `obj.__class__.__bases__[0]` caught by `visit_Attribute`
3. **String Subscript**: `obj['__globals__']` caught by `visit_Subscript`
4. **Eval/Exec**: Direct calls blocked in `FORBIDDEN_CALLS`

### Known Limitations

- **CPU DoS**: Infinite loops terminate at timeout, but consume CPU until then
- **Memory Exhaustion**: Large string concatenations not limited (rely on OS)
- **Regex DoS**: Complex patterns in `grep()` can cause slowdown

**Mitigation**: Production deployments should use cgroups for hard resource limits.

## References

### Implementation

1. `src/repl_environment.py`: Main REPL environment (2700+ lines)
2. `src/restricted_executor.py`: RestrictedPython backend (425 lines)
3. Python AST module documentation: https://docs.python.org/3/library/ast.html

### Security Frameworks

4. RestrictedPython project: https://github.com/zopefoundation/RestrictedPython
5. Plone CMS security model (uses RestrictedPython): https://plone.org/security

### Related Approaches

6. Jupyter Notebook sandboxing: https://jupyter-notebook.readthedocs.io/en/stable/security.html
7. PyPy sandboxing (deprecated): https://doc.pypy.org/en/latest/sandbox.html

---

*Previous: [Chapter 10: Orchestration Architecture](10-orchestration-architecture.md)* | *Next: [Chapter 12: Production Server Stack](12-production-server-stack.md)*

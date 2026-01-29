# Chapter 22: Tool Registry & Permission Model

## Introduction

The orchestration system defines **40+ callable tools** across 8 categories with role-scoped permissions. Tools are declared in YAML (`orchestration/tool_registry.yaml`) and enforced at runtime by `src/tool_registry.py`. Each local orchestrator role (frontdoor, coder, architect, worker, etc.) receives scoped tool access via allow/deny lists.

This chapter covers the tool inventory, permission model, and invocation patterns.

## Tool Registry

### Tool Categories (40+ Tools)

All tools are defined in `orchestration/tool_registry.yaml`:

#### Web Tools (4)

| Tool | Description | Permission |
|------|-------------|------------|
| `http_get` | Fetch content via HTTP GET | `network` |
| `http_post` | Send data via HTTP POST | `network` |
| `web_search` | DuckDuckGo search (no API key) | `network` |
| `fetch_wikipedia` | Wikipedia article summary | `network` |

#### Data Tools (9)

| Tool | Description | Permission |
|------|-------------|------------|
| `json_query` | JMESPath queries on JSON | none |
| `csv_to_json` | CSV to JSON array conversion | none |
| `json_to_csv` | JSON array to CSV conversion | none |
| `sql_query` | In-memory SQLite queries | none |
| `plot_braille` | Terminal braille character plots (C++) | `compute` |
| `plot_function` | Math function plotting (sin, cos, exp) | `compute` |
| `histogram` | Histogram from data (C++) | `compute` |
| `plot_sixel` | High-resolution sixel graphics (C++) | `compute` |
| `render_math` | LaTeX to Unicode/ASCII rendering (C++) | `compute` |

#### Code Tools (4)

| Tool | Description | Permission |
|------|-------------|------------|
| `python_eval` | Safe Python expression evaluation | `compute` |
| `run_shell` | Sandboxed shell command execution | `shell` |
| `git_status` | Git repository status | `filesystem` |
| `lint_python` | Python linting with ruff | `compute` |

#### Math Tools — Python (4)

| Tool | Description | Permission |
|------|-------------|------------|
| `calculate` | Math expression evaluation (numpy) | `compute` |
| `statistics` | Dataset statistics (mean, std, min, max, median) | `compute` |
| `monte_carlo` | Monte Carlo simulation | `compute` |
| `symbolic_solve` | Symbolic equation solving (SymPy) | `compute` |

#### Math Tools — C++ Native (8)

All use the compiled `llama-math-tools` binary with Eigen and Boost:

| Tool | Description | Library |
|------|-------------|---------|
| `matrix_solve` | Solve Ax=b with QR decomposition | Eigen |
| `matrix_eigenvalues` | Eigenvalues/eigenvectors | Eigen |
| `matrix_svd` | Singular Value Decomposition | Eigen |
| `solve_ode` | ODE solver (RK45) | Boost.Odeint |
| `optimize` | Nelder-Mead minimization | Custom |
| `monte_carlo_native` | OpenMP Monte Carlo | Custom |
| `mcmc` | Metropolis-Hastings MCMC sampler | Custom |
| `bayesopt` | Bayesian optimization (GP) | Custom |

#### System Tools (3)

| Tool | Description | Permission |
|------|-------------|------------|
| `read_file` | File contents (max 1MB) | `filesystem` |
| `write_file` | Write/append to file | `filesystem` |
| `list_directory` | Directory listing with glob | `filesystem` |

#### Archive Tools (4)

| Tool | Description | Permission |
|------|-------------|------------|
| `archive_open` | Open archive manifest (zip, tar, 7z) | `filesystem` |
| `archive_extract` | Extract files from archive | `filesystem` |
| `archive_file` | Get specific file from archive | `filesystem` |
| `archive_search` | Search archive contents | `filesystem` |

#### LLM Tools (3)

| Tool | Description | Permission |
|------|-------------|------------|
| `embed_text` | Generate text embedding | `compute` |
| `similarity_search` | Find similar items by embedding | `compute` |
| `classify_text` | Classify text into categories | `compute` |

## Permission Model

### Permission Types

```yaml
permissions:
  network:
    description: Can make network requests
    requires_approval: false
  filesystem:
    description: Can read/write files
    requires_approval: true
    allowed_paths: ["/mnt/raid0/llm/", "/tmp/"]
  shell:
    description: Can execute shell commands
    requires_approval: true
  compute:
    description: Can execute computation
    requires_approval: false
```

### Enforcement

Implemented in `src/tool_registry.py`:

```python
class ToolPermissions:
    web_access: bool
    allowed_categories: list[ToolCategory]
    allowed_tools: list[str]       # Explicit allow list
    forbidden_tools: list[str]     # Explicit deny list

def can_use_tool(self, tool: Tool) -> bool:
    # 1. Check forbidden list (deny wins)
    # 2. Check explicit allow list
    # 3. Check category + web_access flag
```

The deny list takes priority: a tool on the `forbidden_tools` list is blocked even if its category is otherwise allowed.

### Path Validation

All filesystem tools validate paths against the whitelist:

```python
ALLOWED_FILE_PATHS = ["/mnt/raid0/llm/", "/tmp/"]

def _validate_file_path(path: str) -> bool:
    resolved = os.path.realpath(path)  # Resolve symlinks
    return any(resolved.startswith(p) for p in ALLOWED_FILE_PATHS)
```

Uses `os.path.realpath()` to defeat symlink-based escape attempts.

## Tool Invocation Pattern

```python
from src.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.load_from_yaml("orchestration/tool_registry.yaml")

if registry.can_use_tool("frontdoor", "fetch_docs"):
    result = registry.invoke("fetch_docs", role="frontdoor", url="...")
```

## References

### Project Files

- Tool definitions: `orchestration/tool_registry.yaml`
- Python implementation: `src/tool_registry.py`

### Related Chapters

1. [Chapter 10: Orchestration Architecture](10-orchestration-architecture.md) — TaskIR and agent tiers
2. [Chapter 18: Escalation & Routing](18-escalation-and-routing.md) — how tools route between agents
3. [Chapter 23: Security & Monitoring](23-security-and-monitoring.md) — runtime security enforcement

---

*Previous: [Chapter 21: Benchmarking Framework](21-benchmarking-framework.md)* | *Next: [Chapter 23: Security & Monitoring](23-security-and-monitoring.md)*

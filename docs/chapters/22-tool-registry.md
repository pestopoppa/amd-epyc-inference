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

## Plugin Architecture

### Overview

Tools can also be loaded via a **plugin-based architecture** with manifest files (`src/tool_loader.py`). Each tool directory contains a `manifest.json` that declares:
- Plugin metadata (name, version, description)
- Tool definitions (module, function, parameters)
- Dependencies and settings schema

### Manifest Schema

```json
{
  "schema_version": "1.0",
  "name": "canvas_tools",
  "version": "1.0.0",
  "description": "Canvas tools for reasoning visualization",
  "enabled": true,
  "dependencies": ["kuzu"],
  "settings_schema": {
    "canvas_directory": {"type": "string", "default": "/mnt/raid0/llm/claude/logs/canvases"}
  },
  "tools": [
    {
      "name": "export_reasoning_canvas",
      "description": "Export hypothesis/failure graphs to JSON Canvas",
      "module": "src.tools.canvas_tools",
      "function": "export_reasoning_canvas",
      "category": "data",
      "parameters": {
        "graph_type": {"type": "string", "required": false}
      }
    }
  ]
}
```

### Plugin Discovery

```python
from src.tool_loader import ToolPluginLoader

loader = ToolPluginLoader()
count = loader.discover_plugins(Path("src/tools"))  # Scans for manifest.json files

tools = loader.list_tools(enabled_only=True)
# Returns: [{"name": "export_reasoning_canvas", "plugin": "canvas_tools", ...}, ...]
```

### Hot Reload

Plugins can be reloaded without restarting the server:

```python
changed = loader.check_for_changes()  # Returns list of modified plugins
count = loader.reload_changed()        # Reloads them
```

Via MCP: `reload_plugins()` tool.

### Current Plugins

| Plugin | Tools | Description |
|--------|-------|-------------|
| `web` | `fetch_docs`, `web_search` | Web content retrieval |
| `file` | `read_file`, `list_dir` | File system operations |
| `code` | `run_tests`, `lint_code` | Code quality tools |
| `data` | `json_parse` | Data transformation |
| `canvas_tools` | `export_reasoning_canvas`, `import_canvas_edits`, `list_canvases` | JSON Canvas integration |

### Per-Tool Settings

User-specific settings stored in `src/tool_settings/{plugin_name}.json` (gitignored):

```json
{
  "enabled": true,
  "tool_overrides": {
    "export_reasoning_canvas": {"enabled": false}
  },
  "custom_config": {"canvas_directory": "/custom/path"}
}
```

## References

### Project Files

- Tool definitions: `orchestration/tool_registry.yaml`
- Python implementation: `src/tool_registry.py`
- Plugin loader: `src/tool_loader.py`
- Plugin manifests: `src/tools/*/manifest.json`

### Related Chapters

1. [Chapter 10: Orchestration Architecture](10-orchestration-architecture.md) — TaskIR and agent tiers
2. [Chapter 18: Escalation & Routing](18-escalation-and-routing.md) — how tools route between agents
3. [Chapter 23: Security & Monitoring](23-security-and-monitoring.md) — runtime security enforcement

---

*Previous: [Chapter 21: Benchmarking Framework](21-benchmarking-framework.md)* | *Next: [Chapter 23: Security & Monitoring](23-security-and-monitoring.md)*

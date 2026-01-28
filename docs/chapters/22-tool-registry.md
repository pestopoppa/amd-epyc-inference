# Chapter 22: Tool Registry & Agent Roles

## Introduction

The orchestration system defines **40+ callable tools** across 8 categories and **8 agent roles** with hierarchical permissions. Tools are declared in YAML (`orchestration/tool_registry.yaml`) and enforced at runtime by `src/tool_registry.py`. Agent roles range from the Lead Developer (strategic coordination) to specialized support agents (Build Engineer, Sysadmin), each with scoped tool access and model selection based on task complexity rather than role identity.

This chapter covers the tool inventory, permission model, agent role definitions, and the coordination flow between agents.

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

## Agent Role Definitions

All 8 roles are defined in `/mnt/raid0/llm/claude/agents/`:

### Primary Agents (4)

| Role | File | Default Model | Responsibility |
|------|------|---------------|----------------|
| **Lead Developer** | `lead-developer.md` | Sonnet (Opus for novel) | Architecture, coordination, strategic decisions |
| **Research Engineer** | `research-engineer.md` | Sonnet (Opus for novel) | C++ implementation, debugging, novel approaches |
| **Research Writer** | `research-writer.md` | Sonnet (Opus for analysis) | Report synthesis, documentation, literature |
| **Benchmark Analyst** | `benchmark-analyst.md` | Haiku | Benchmark execution, data collection, metrics |

### Support Agents (4)

| Role | File | Default Model | Responsibility |
|------|------|---------------|----------------|
| **Sysadmin** | `sysadmin.md` | Sonnet | System config, NUMA, CPU governor |
| **Build Engineer** | `build-engineer.md` | Sonnet | CMake, compiler flags, build issues |
| **Model Engineer** | `model-engineer.md` | Sonnet | GGUF conversion, quantization formats |
| **Safety Reviewer** | `safety-reviewer.md` | Opus | Risk assessment, security review |

### Task-Based Model Selection

Model selection depends on **task complexity**, not agent identity:

```
NOVEL/COMPLEX (Opus 4.5)
  Novel architecture, complex debugging, security assessment

RESEARCH/SYNTHESIS (Sonnet 4.5)
  Code implementation, report writing, comparison analysis

ROUTINE/EXECUTION (Haiku 4.5)
  Benchmark runs, CSV parsing, status checks, known commands
```

This means the Research Engineer might use Opus for a novel KV cache debugging session but Sonnet for routine code implementation.

## Agent Coordination

### Hierarchy

```
              Lead Developer
              (coordinates)
                    |
    +---------------+---------------+
    |               |               |
Research       Benchmark        Research
Engineer       Analyst           Writer
(implements)   (measures)       (documents)
    |               |               |
    +---------------+---------------+
                    |
             Support Agents
             (as needed)
            +-- Sysadmin
            +-- Build Engineer
            +-- Model Engineer
            +-- Safety Reviewer
```

### Decision Flow

The Lead Developer makes routing decisions:

1. **Novel/complex** task arrives -> escalate to Opus, assign Research Engineer
2. **Benchmark needed** -> assign Benchmark Analyst (Haiku for speed)
3. **Documentation** -> assign Research Writer
4. **System tuning** -> delegate to Sysadmin
5. **Build issue** -> delegate to Build Engineer
6. **New model** -> delegate to Model Engineer
7. **Risk detected** -> invoke Safety Reviewer (always Opus)

### Critical Rules

From `agents/AGENT_INSTRUCTIONS.md`:

1. **Never write to root filesystem** — all paths must start with `/mnt/raid0/`
2. **Never use `pytest -n auto`** — 192 threads would OOM
3. **Always use feature flags** — `from src.features import features`
4. **Always use Role enum** — `Role.CODER_PRIMARY`, not string `"coder_primary"`
5. **Always log exceptions** — no bare `except: pass`
6. **Max 3 retries** — then document blocker and stop

### Tool Invocation Pattern

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
- Agent overview: `agents/README.md`
- Critical rules: `agents/AGENT_INSTRUCTIONS.md`
- Role definitions: `agents/{lead-developer,research-engineer,research-writer,benchmark-analyst,sysadmin,build-engineer,model-engineer,safety-reviewer}.md`

### Related Chapters

1. [Chapter 10: Orchestration Architecture](10-orchestration-architecture.md) — TaskIR and agent tiers
2. [Chapter 18: Escalation & Routing](18-escalation-and-routing.md) — how tools route between agents
3. [Chapter 23: Security & Monitoring](23-security-and-monitoring.md) — runtime security enforcement

---

*Previous: [Chapter 21: Benchmarking Framework](21-benchmarking-framework.md)* | *Next: [Chapter 23: Security & Monitoring](23-security-and-monitoring.md)*

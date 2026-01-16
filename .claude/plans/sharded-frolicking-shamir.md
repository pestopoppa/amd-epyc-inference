# Native Computational Tools - Comprehensive Handoff

**Goal:** Build a comprehensive toolkit that offloads computation from LLM to fast native tools
**Updated:** 2026-01-15
**Status:** Phase 1 COMPLETE, Phases 2-6 IN PROGRESS
**Related:** `handoffs/active/rlm-orchestrator-roadmap.md` (Phase 8: Visualization)

---

## CURRENT STATUS (for YOLO agent)

### ✅ COMPLETED
- Directory structure created at `/mnt/raid0/llm/llama.cpp/tools/math-tools/`
- CMakeLists.txt with Eigen + nlohmann/json + OpenMP
- Command dispatcher (main.cpp)
- Python wrapper (`orchestration/tools/cpp_tools.py`)
- Working commands: `matrix_op`, `solve_ode`, `optimize`, `monte_carlo`, `plot`
- **[2026-01-15]** Added 14 C++ tools to `tool_registry.yaml`
- **[2026-01-15]** Added Python wrappers for: `mcmc`, `bayesopt`, `plot_sixel`, `render_math`
- **[2026-01-15]** Registered new tools in `register_math_tools()` function

### 🔧 WORKING BINARY
```bash
/mnt/raid0/llm/llama.cpp/build/bin/llama-math-tools
# or rebuild from source:
cd /mnt/raid0/llm/llama.cpp/tools/math-tools && cmake -B build && cmake --build build -j 96
```

### 📋 REMAINING WORK (Priority Order)

**HIGH PRIORITY - C++ Implementation Needed (Python wrappers ready):**
1. `mcmc` - Metropolis-Hastings sampler (commands/statistical/mcmc.cpp) ⚠️ wrapper ready
2. `bayesopt` - Gaussian process optimization (commands/statistical/bayesopt.cpp) ⚠️ wrapper ready
3. `plot_sixel` - High-res terminal graphics (commands/visualization/plot_sixel.cpp) ⚠️ wrapper ready
4. `render_math` - LaTeX → Unicode (commands/visualization/render_math.cpp) ⚠️ wrapper ready

**MEDIUM PRIORITY:**
5. `symbolic_diff` - Differentiation via SymEngine (commands/symbolic/differentiate.cpp)
6. `simplify` - Expression simplification (commands/symbolic/simplify.cpp)
7. `integrate` - Numerical integration (commands/numerical/integrate.cpp)

**LOW PRIORITY:**
8. `latex_compile` - Document generation (subprocess to pdflatex)
9. `sat_solve` - Z3 bindings
10. `parse_ast` - tree-sitter integration

**NOTE:** The Python integration layer is complete. Next step is implementing the C++ commands.

### 🚀 HOW TO ADD A NEW COMMAND
```bash
cd /mnt/raid0/llm/llama.cpp/tools/math-tools

# 1. Create command file following existing pattern (e.g., monte_carlo.cpp)
# 2. Add to CMakeLists.txt source list
# 3. Add factory function declaration in main.cpp
# 4. Register in CommandRegistry in main()
# 5. Rebuild: cmake --build build -j 96
# 6. Test: echo '{"command":"NEW_CMD",...}' | ./build/llama-math-tools
```

---

## 1. ARCHITECTURAL PHILOSOPHY

### Core Principle

> **LLMs reason. Tools execute.**
>
> Every tool call should save 100-5000 tokens of LLM generation.
> The LLM specifies WHAT (high-level intent), tools execute HOW (computation).

### Tool Selection Criteria

A tool is worth building/integrating if it meets **2+ criteria**:

| Criterion | Example |
|-----------|---------|
| **Token savings > 200** | ODE solver vs LLM explaining Runge-Kutta steps |
| **Deterministic correctness** | Matrix inversion must be exact, not "approximately" |
| **Speed critical** | Monte Carlo needs 10,000 samples in <100ms |
| **Visual output** | Plotting data the LLM cannot "see" |
| **External state** | File I/O, database queries, API calls |
| **Domain expertise** | LaTeX compilation, proof checking |

### Anti-Patterns (Don't Build)

- Tools the LLM can do in 1-2 sentences
- Tools requiring complex natural language parsing
- Tools that just wrap LLM calls (use roles instead)

---

## 2. TOOL DISCOVERY FRAMEWORK

### Sources to Mine

| Source | URL | What to Extract |
|--------|-----|-----------------|
| **BFCL v4** | berkeley-function-call-leaderboard | 2000+ function schemas |
| **LangChain Tools** | langchain.com/docs/integrations/tools | 100+ tool patterns |
| **OpenAI Cookbook** | github.com/openai/openai-cookbook | Function calling examples |
| **HuggingFace Agents** | huggingface.co/docs/transformers/agents | Multimodal tool patterns |
| **Gorilla API Zoo** | github.com/ShishirPatil/gorilla | 1600+ API schemas |
| **Awesome-LLM-Tools** | github.com/topics/llm-tools | Community tools |

### Evaluation Rubric

For each candidate tool, score 1-5:

```
IMPACT = (tokens_saved × frequency_of_use) / implementation_complexity
```

| Score | Meaning |
|-------|---------|
| 5 | Must have - high impact, easy to implement |
| 4 | Should have - good impact, moderate complexity |
| 3 | Nice to have - moderate impact |
| 2 | Low priority - niche use case |
| 1 | Skip - not worth the complexity |

### Discovery Process

1. **Scrape schemas** from sources above
2. **Cluster by domain** (math, code, web, data, etc.)
3. **Score each cluster** using rubric
4. **Prototype top 5** from each cluster
5. **Benchmark token savings** vs pure LLM
6. **Integrate winners** into tool registry

---

## 3. INTEGRATION ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Python)                     │
│  src/tool_registry.py → TOOL("solve_ode", params={...})     │
└──────────────────────────┬──────────────────────────────────┘
                           │ subprocess + JSON
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              llama-math-tools (C++ binary)                   │
│  /mnt/raid0/llm/llama.cpp/build/bin/llama-math-tools        │
│                                                              │
│  Commands: solve_ode, matrix_op, monte_carlo, bayesopt,     │
│            symbolic_diff, plot_data, latex_render, ...      │
└─────────────────────────────────────────────────────────────┘
```

### Communication Patterns

| Pattern | Use When | Latency |
|---------|----------|---------|
| **Subprocess + JSON stdout** | One-shot computation | ~50ms startup |
| **HTTP server (persistent)** | Tight loops, stateful | ~1ms per call |
| **Shared library (FFI)** | Ultra-low latency | ~0.1ms |
| **Julia subprocess** | PySR-style symbolic regression | ~200ms |

**Default:** Subprocess + JSON (like llama-cli)
**Upgrade to HTTP:** When same tool called >10x in loop

### JSON Interface Standard

```json
// Request (stdin)
{
  "command": "solve_ode",
  "params": {
    "system": "dy/dt = -0.5*y",
    "y0": [1.0],
    "t_span": [0, 10]
  }
}

// Response (stdout)
{
  "status": "success",  // or "error"
  "result": {...},      // command-specific
  "stats": {            // optional metadata
    "elapsed_ms": 12,
    "iterations": 47
  },
  "error": null         // or error message
}
```

---

## 4. SYMBOLIC REGRESSION (PySR-Style)

### Inspiration: PySR/SymbolicRegression.jl

[PySR](https://github.com/MilesCranmer/PySR) uses genetic algorithms to discover symbolic expressions from data. Key features:

- **Genetic search**: Mutation, crossover, migration across populations
- **Parsimony pressure**: Balance accuracy vs complexity
- **Julia backend**: SymbolicRegression.jl for performance
- **Multi-format export**: SymPy, PyTorch, JAX, LaTeX

### Implementation Options

| Option | Pros | Cons |
|--------|------|------|
| **Wrap PySR directly** | Full-featured, maintained | Julia dependency, ~200ms startup |
| **Port to C++** | No Julia, fast startup | Significant effort, lose updates |
| **Use SymEngine + custom GA** | Pure C++, moderate effort | Less sophisticated than PySR |

**Recommendation**: Wrap PySR via subprocess for full functionality, with fallback to SymEngine for simple cases.

### Tool: `symbolic_regress`

```json
// Input
{
  "command": "symbolic_regress",
  "X": [[1,2], [2,3], [3,4], ...],
  "y": [5, 8, 11, ...],
  "max_complexity": 20,
  "populations": 20,
  "iterations": 100
}

// Output
{
  "status": "success",
  "equations": [
    {"expr": "x0 + 2*x1 - 1", "complexity": 5, "mse": 0.0},
    {"expr": "x0 + x1 + x1", "complexity": 5, "mse": 0.0}
  ],
  "pareto_front": [...],  // Complexity vs accuracy tradeoff
  "best": "x0 + 2*x1 - 1"
}
```

---

## 5. VISUALIZATION SYSTEM

### Dual-Mode Rendering

| Mode | Terminal Support | Quality | Use Case |
|------|------------------|---------|----------|
| **Braille (Unicode)** | Universal | Low-res (2x4 per char) | SSH, basic terminals |
| **Sixel** | xterm, mlterm, mintty, WezTerm | High-res | Local dev, rich output |

### Auto-Detection

```cpp
bool supports_sixel() {
    // Check TERM, query terminal via DA1 escape sequence
    const char* term = getenv("TERM");
    if (strstr(term, "xterm") || strstr(term, "mlterm")) {
        // Send DA1 query: ESC [ c
        // Parse response for sixel support (code 4)
        return query_terminal_capabilities();
    }
    return false;
}
```

### Tool: `plot_ascii`

```json
// Input
{
  "command": "plot_ascii",
  "x": [0, 1, 2, 3, 4, 5],
  "y": [0, 1, 4, 9, 16, 25],
  "width": 80,
  "height": 24,
  "title": "y = x²",
  "mode": "auto"  // "braille", "sixel", or "auto"
}

// Output (braille mode)
{
  "status": "success",
  "format": "braille",
  "plot": "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣤⣶⣾⣿\n⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣶⣿⣿\n..."
}

// Output (sixel mode)
{
  "status": "success",
  "format": "sixel",
  "plot": "\x1bPq#0;2;0;0;0#1;2;100;100;100...~;\x1b\\"
}
```

### Tool: `render_math`

LaTeX → Unicode/ASCII math rendering:

```json
// Input
{
  "command": "render_math",
  "latex": "\\frac{d^2y}{dx^2} + \\omega^2 y = 0",
  "format": "unicode"  // or "ascii"
}

// Output
{
  "status": "success",
  "rendered": "d²y/dx² + ω²y = 0"
}
```

### Connection to Frontend (Phase 8)

The visualization tools feed into the Trajectory Visualization system in `handoffs/active/rlm-orchestrator-roadmap.md`:

```
Tool Output → SSE Stream → Gradio/Web UI
     ↓
plot_ascii/sixel → {"type": "plot", "data": ...} → Render in browser
render_math → {"type": "math", "latex": ...} → KaTeX rendering
```

---

## 6. COMPLETE TOOL INVENTORY

### Tier 1: Numerical (C++ - Eigen/Boost.Odeint)

| Tool | Tokens Saved | Priority |
|------|--------------|----------|
| `solve_ode` | 500-2000 | 5 |
| `matrix_op` | 200-1000 | 5 |
| `optimize` | 300-1500 | 5 |
| `integrate` | 200-800 | 4 |
| `interpolate` | 100-500 | 4 |
| `fft` | 200-800 | 4 |
| `roots` | 100-500 | 4 |

### Tier 2: Statistical (C++ - Xoshiro/Custom)

| Tool | Tokens Saved | Priority |
|------|--------------|----------|
| `monte_carlo` | 500-2000 | 5 |
| `mcmc` | 1000-5000 | 4 |
| `bayesopt` | 500-3000 | 4 |
| `markov_bridge` | 300-1000 | 4 |
| `kde` | 200-800 | 3 |
| `bootstrap` | 300-1000 | 3 |
| `importance_sample` | 400-1500 | 3 |

### Tier 3: Symbolic (C++ - SymEngine, Julia - PySR)

| Tool | Tokens Saved | Priority |
|------|--------------|----------|
| `symbolic_diff` | 100-500 | 4 |
| `symbolic_int` | 200-1000 | 4 |
| `simplify` | 100-400 | 4 |
| `taylor` | 200-800 | 3 |
| `symbolic_regress` | 1000-5000 | 4 |
| `solve_symbolic` | 300-1500 | 3 |

### Tier 4: Visualization (C++ - Custom)

| Tool | Tokens Saved | Priority |
|------|--------------|----------|
| `plot_ascii` | 200-1000 | 5 |
| `plot_sixel` | 200-1000 | 4 |
| `render_math` | 300-1500 | 4 |
| `graph_viz` | 200-800 | 3 |
| `table_format` | 100-400 | 3 |

### Tier 5: Documents (Subprocess - pdflatex/pandoc)

| Tool | Tokens Saved | Priority |
|------|--------------|----------|
| `latex_compile` | 500-2000 | 4 |
| `latex_check` | 100-300 | 4 |
| `markdown_render` | 100-400 | 3 |
| `bibtex_format` | 100-500 | 2 |
| `typst_compile` | 300-1000 | 3 |

### Tier 6: Verification (C++ - Z3/MiniSat)

| Tool | Tokens Saved | Priority |
|------|--------------|----------|
| `sat_solve` | 300-2000 | 3 |
| `smt_check` | 500-3000 | 3 |
| `type_check` | 200-1000 | 3 |
| `constraint_solve` | 300-1500 | 3 |

### Tier 7: Data/Code (C++ - tree-sitter/simdjson)

| Tool | Tokens Saved | Priority |
|------|--------------|----------|
| `parse_ast` | 200-800 | 4 |
| `semantic_grep` | 300-1500 | 4 |
| `json_query` | 100-400 | 4 |
| `csv_stats` | 200-800 | 4 |
| `diff_semantic` | 300-1000 | 3 |

---

## 7. IMPLEMENTATION PHASES

### Phase 1: Infrastructure ✅ COMPLETE
- [x] Create `tools/math-tools/` in llama.cpp
- [x] CMakeLists.txt with Eigen + nlohmann/json + OpenMP
- [x] Command dispatcher (main.cpp)
- [x] Python wrapper (`orchestration/tools/cpp_tools.py`)
- [x] Register in `tool_registry.yaml` (2026-01-15: Added 14 C++ tools)
- [ ] Basic test suite

### Phase 2: Numerical Core (Partial)
- [x] `solve_ode` - RK45 adaptive solver
- [x] `matrix_op` - solve, inverse, det, eigen, SVD, LU, QR, rank
- [x] `optimize` - Nelder-Mead, gradient descent
- [ ] `integrate` - adaptive quad, MC integration

### Phase 3: Statistical (Partial)
- [x] `monte_carlo` - parallel simulation with OpenMP
- [x] `mcmc` - Metropolis-Hastings (Python wrapper ready, C++ pending)
- [x] `bayesopt` - Gaussian process optimization (Python wrapper ready, C++ pending)
- [ ] `markov_bridge` - Brownian, OU bridges

### Phase 4: Visualization (Partial)
- [x] `plot` - braille character plotting (scatter, line, histogram, function)
- [x] `plot_sixel` - sixel graphics (Python wrapper ready, C++ pending)
- [x] `render_math` - LaTeX → Unicode (Python wrapper ready, C++ pending)
- [ ] Auto-detection of terminal capabilities

### Phase 5: Symbolic (Week 5)
- [ ] Install SymEngine
- [ ] `symbolic_diff`, `symbolic_int`, `simplify`
- [ ] PySR wrapper for `symbolic_regress`
- [ ] `taylor` series expansion

### Phase 6: Documents & Verification (Week 6)
- [ ] `latex_compile`, `latex_check`
- [ ] Z3 bindings for `sat_solve`, `smt_check`
- [ ] tree-sitter for `parse_ast`, `semantic_grep`

---

## 8. FILE STRUCTURE

```
/mnt/raid0/llm/llama.cpp/tools/math-tools/
├── CMakeLists.txt
├── main.cpp                    # Command dispatcher
├── commands/
│   ├── numerical/
│   │   ├── solve_ode.cpp
│   │   ├── matrix_op.cpp
│   │   ├── optimize.cpp
│   │   └── integrate.cpp
│   ├── statistical/
│   │   ├── monte_carlo.cpp
│   │   ├── mcmc.cpp
│   │   └── bayesopt.cpp
│   ├── symbolic/
│   │   ├── differentiate.cpp
│   │   └── simplify.cpp
│   ├── visualization/
│   │   ├── plot_braille.cpp
│   │   ├── plot_sixel.cpp
│   │   └── render_math.cpp
│   └── verification/
│       └── sat_solve.cpp
├── include/
│   ├── json_io.hpp
│   ├── common.hpp
│   └── xoshiro.hpp
├── external/
│   ├── eigen/                  # Header-only
│   └── boost_odeint/           # Header-only
└── tests/
    ├── test_ode.cpp
    ├── test_matrix.cpp
    └── test_plot.cpp

/mnt/raid0/llm/claude/orchestration/tools/
├── cpp_tools.py                # Python subprocess wrapper
├── pysr_wrapper.py             # PySR/Julia wrapper
└── tool_registry.yaml          # Updated with C++ tools
```

---

## 9. DEPENDENCIES

### Header-Only (No Build Required)
- **Eigen 3.4+** - Linear algebra (MPL2)
- **Boost.Odeint** - ODE solvers (BSD)
- **nlohmann/json** - Already in llama.cpp vendor/
- **Xoshiro-cpp** - Fast PRNG (Public Domain)

### Optional Compiled
- **SymEngine** - Symbolic math (MIT, ~100MB)
- **Z3** - SMT solver (MIT, ~50MB)
- **tree-sitter** - AST parsing (MIT)
- **simdjson** - Fast JSON (Apache 2.0)

### External (Subprocess)
- **PySR/Julia** - Symbolic regression
- **pdflatex** - LaTeX compilation
- **gnuplot** - Advanced plotting
- **pandoc** - Document conversion

---

## 10. SUCCESS METRICS

| Metric | Target |
|--------|--------|
| Token reduction per tool call | 200-2000 |
| Tool latency (p50) | <50ms |
| Tool latency (p99) | <500ms |
| Memory overhead | <100MB total |
| Coverage of numerical tasks | 80% offloadable |
| Test coverage | >90% |

---

## 11. OPEN QUESTIONS

1. **Julia bundling**: Ship Julia with PySR, or require user install?
2. **GPU acceleration**: CUDA for Monte Carlo? (192 CPU threads may be enough)
3. **Caching**: Cache expensive tool results (e.g., symbolic regression)?
4. **Streaming**: Stream large outputs (e.g., ODE trajectories) via SSE?
5. **Sandboxing**: Run tools in separate process/container for safety?

---

## 12. RESUME COMMANDS

```bash
# ============ BUILD (already works) ============
cd /mnt/raid0/llm/llama.cpp/tools/math-tools
cmake -B build && cmake --build build -j 96

# ============ VERIFY EXISTING COMMANDS ============
echo '{"command":"help"}' | ./build/llama-math-tools

# Matrix solve: Ax = b
echo '{"command":"matrix_op","operation":"solve","A":[[3,1],[1,2]],"b":[9,8]}' | ./build/llama-math-tools
# Expected: x = [2, 3]

# ODE: dy/dt = -0.5y, y(0)=1, t=[0,5]
echo '{"command":"solve_ode","y0":[1.0],"t_span":[0,5],"system":"-0.5*y"}' | ./build/llama-math-tools
# Expected: y(5) ≈ 0.082 = e^{-2.5}

# Monte Carlo pi
echo '{"command":"monte_carlo","operation":"pi","samples":100000}' | ./build/llama-math-tools
# Expected: pi ≈ 3.14

# Braille plot
echo '{"command":"plot","type":"function","function":"sin"}' | ./build/llama-math-tools
# Expected: ASCII sine wave

# ============ ADD NEW COMMAND (TEMPLATE) ============
# 1. Copy an existing command as template:
cp commands/statistical/monte_carlo.cpp commands/statistical/mcmc.cpp

# 2. Implement the command class (follow the pattern)

# 3. Add to CMakeLists.txt:
#    commands/statistical/mcmc.cpp

# 4. Add factory declaration to main.cpp:
#    std::unique_ptr<Command> create_mcmc();

# 5. Register in main():
#    registry.register_command("mcmc", create_mcmc);

# 6. Rebuild and test:
cmake --build build -j 96
echo '{"command":"mcmc","samples":10000}' | ./build/llama-math-tools
```

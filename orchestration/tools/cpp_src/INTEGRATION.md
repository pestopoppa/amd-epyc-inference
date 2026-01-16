# Integration Instructions for New Math Tools Commands

This directory contains new C++ command implementations ready to be integrated into `llama-math-tools`.

## Files Created

### Headers (include/)
- `command.hpp` - Base Command class (may already exist in target)
- `expression.hpp` - Mathematical expression parser for mcmc/bayesopt

### Statistical Commands (commands/statistical/)
- `mcmc.cpp` - Metropolis-Hastings MCMC sampler
- `bayesopt.cpp` - Bayesian Optimization with Gaussian Process surrogate

### Visualization Commands (commands/visualization/)
- `render_math.cpp` - LaTeX to Unicode/ASCII converter
- `plot_sixel.cpp` - Sixel graphics with braille fallback

## Integration Steps

### 1. Copy Files to Build Directory

```bash
# Navigate to math-tools directory
cd /mnt/raid0/llm/llama.cpp/tools/math-tools

# Copy header files (expression.hpp is new, command.hpp may need merging)
cp /workspace/orchestration/tools/cpp_src/include/expression.hpp include/

# Copy statistical commands
cp /workspace/orchestration/tools/cpp_src/commands/statistical/mcmc.cpp commands/statistical/
cp /workspace/orchestration/tools/cpp_src/commands/statistical/bayesopt.cpp commands/statistical/

# Copy visualization commands
cp /workspace/orchestration/tools/cpp_src/commands/visualization/render_math.cpp commands/visualization/
cp /workspace/orchestration/tools/cpp_src/commands/visualization/plot_sixel.cpp commands/visualization/
```

### 2. Update CMakeLists.txt

Add these source files to the existing `llama-math-tools` target in CMakeLists.txt:

```cmake
# Add to existing source list:
set(MATH_TOOLS_SOURCES
    # ... existing sources ...
    commands/statistical/mcmc.cpp
    commands/statistical/bayesopt.cpp
    commands/visualization/render_math.cpp
    commands/visualization/plot_sixel.cpp
)
```

### 3. Update main.cpp

Add factory function declarations near the top:

```cpp
// Statistical commands
std::unique_ptr<Command> create_mcmc();
std::unique_ptr<Command> create_bayesopt();

// Visualization commands
std::unique_ptr<Command> create_render_math();
std::unique_ptr<Command> create_plot_sixel();
```

Register commands in main():

```cpp
int main() {
    CommandRegistry registry;

    // ... existing registrations ...

    // Statistical
    registry.register_command("mcmc", create_mcmc);
    registry.register_command("bayesopt", create_bayesopt);

    // Visualization
    registry.register_command("render_math", create_render_math);
    registry.register_command("plot_sixel", create_plot_sixel);

    // ... rest of main ...
}
```

### 4. Rebuild

```bash
cd /mnt/raid0/llm/llama.cpp/tools/math-tools
cmake --build build -j 96
```

### 5. Test

```bash
# Test MCMC (2D standard normal)
echo '{"command":"mcmc","log_density":"-0.5*(x0**2 + x1**2)","x0":[0,0],"n_samples":5000}' | ./build/llama-math-tools

# Test BayesOpt (find max of -(x-2)^2 on [0,5])
echo '{"command":"bayesopt","bounds":[[0,5]],"objective":"-(x0-2)**2","n_iter":20}' | ./build/llama-math-tools

# Test render_math
echo '{"command":"render_math","latex":"\\\\frac{dy}{dx} = \\\\alpha x","format":"unicode"}' | ./build/llama-math-tools

# Test plot_sixel (falls back to braille if sixel unavailable)
echo '{"command":"plot_sixel","x":[0,1,2,3,4,5],"y":[0,1,4,9,16,25],"title":"y=x^2"}' | ./build/llama-math-tools
```

## Dependencies

All dependencies are already included in the existing build:
- **Eigen** - Required by bayesopt for GP computations
- **nlohmann/json** - Already in llama.cpp vendor/
- **OpenMP** - Optional, for parallel sampling in MCMC

## Python Integration

Python wrappers are already implemented in `/workspace/orchestration/tools/cpp_tools.py`:
- `MathTools.mcmc()`
- `MathTools.bayesopt()`
- `MathTools.plot_sixel()`
- `MathTools.render_math()`

These are registered in `register_math_tools()` function.

## Command Reference

### mcmc
```json
{
  "command": "mcmc",
  "log_density": "-0.5*(x0**2 + x1**2)",  // Expression for log-density
  "x0": [0, 0],                            // Initial state
  "n_samples": 10000,                      // Number of samples
  "proposal_std": 1.0,                     // Proposal distribution std
  "burnin": 1000,                          // Burn-in period
  "thin": 1,                               // Thinning factor
  "seed": 42                               // Random seed
}
```

### bayesopt
```json
{
  "command": "bayesopt",
  "bounds": [[0, 5], [0, 5]],              // Parameter bounds
  "objective": "-(x0-2)**2-(x1-3)**2",     // Objective to maximize
  "n_init": 5,                             // Initial random samples
  "n_iter": 25,                            // Optimization iterations
  "acquisition": "ei",                     // "ei", "ucb", or "pi"
  "noise": 1e-6,                           // Observation noise
  "seed": 42                               // Random seed
}
```

### render_math
```json
{
  "command": "render_math",
  "latex": "\\frac{dy}{dx}",               // LaTeX expression
  "format": "unicode"                      // "unicode" or "ascii"
}
```

### plot_sixel
```json
{
  "command": "plot_sixel",
  "x": [0, 1, 2, 3],
  "y": [0, 1, 4, 9],
  "plot_type": "line",                     // "line", "scatter", or "bar"
  "width": 800,                            // Image width in pixels
  "height": 400,                           // Image height in pixels
  "title": "Plot Title",
  "x_label": "X Axis",
  "y_label": "Y Axis",
  "color": "blue"                          // "blue", "red", "green", "orange"
}
```

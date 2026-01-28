# CMakeLists.txt for new math-tools commands
#
# INTEGRATION: These files should be copied to /mnt/raid0/llm/llama.cpp/tools/math-tools/
# and the sources added to the existing CMakeLists.txt there.
#
# Commands in this directory:
#
# Phase 4 (Complete):
#   - commands/statistical/mcmc.cpp (Metropolis-Hastings MCMC sampler)
#   - commands/statistical/bayesopt.cpp (Bayesian Optimization with GP)
#   - commands/visualization/render_math.cpp (LaTeX to Unicode converter)
#   - commands/visualization/plot_sixel.cpp (Sixel graphics with braille fallback)
#
# Phase 5 (New - 2026-01-27):
#   - commands/numerical/integrate.cpp (Adaptive Simpson numerical integration)
#   - commands/symbolic/symbolic_diff.cpp (Symbolic differentiation - requires SymEngine)
#   - commands/symbolic/simplify.cpp (Expression simplification - requires SymEngine)
#
# Headers:
#   - include/command.hpp (base command class - may exist in parent)
#   - include/expression.hpp (mathematical expression parser)

# Source files to add to main CMakeLists.txt
set(NEW_COMMAND_SOURCES
    # Phase 4 - Statistical
    commands/statistical/mcmc.cpp
    commands/statistical/bayesopt.cpp
    # Phase 4 - Visualization
    commands/visualization/render_math.cpp
    commands/visualization/plot_sixel.cpp
    # Phase 5 - Numerical
    commands/numerical/integrate.cpp
)

# Optional: SymEngine commands (compile only if SymEngine available)
# set(SYMENGINE_COMMAND_SOURCES
#     commands/symbolic/symbolic_diff.cpp
#     commands/symbolic/simplify.cpp
# )

# Dependencies:
#   - Eigen (required, already in build)
#   - SymEngine (optional, for symbolic commands)
#     Install: apt install libsymengine-dev
#     CMake: find_package(SymEngine REQUIRED)
#            target_link_libraries(llama-math-tools SymEngine::symengine)
#            target_compile_definitions(llama-math-tools PRIVATE HAVE_SYMENGINE)

# Factory functions to register in main.cpp:
#
#   // Add declarations at top of main.cpp:
#   std::unique_ptr<Command> create_mcmc();
#   std::unique_ptr<Command> create_bayesopt();
#   std::unique_ptr<Command> create_render_math();
#   std::unique_ptr<Command> create_plot_sixel();
#   std::unique_ptr<Command> create_integrate();
#   // Uncomment if SymEngine available:
#   // std::unique_ptr<Command> create_symbolic_diff();
#   // std::unique_ptr<Command> create_simplify();
#
#   // Add registrations in main():
#   registry.register_command("mcmc", create_mcmc);
#   registry.register_command("bayesopt", create_bayesopt);
#   registry.register_command("render_math", create_render_math);
#   registry.register_command("plot_sixel", create_plot_sixel);
#   registry.register_command("integrate", create_integrate);
#   // Uncomment if SymEngine available:
#   // registry.register_command("symbolic_diff", create_symbolic_diff);
#   // registry.register_command("simplify", create_simplify);

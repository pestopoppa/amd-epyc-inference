# CMakeLists.txt for new math-tools commands
#
# INTEGRATION: These files should be copied to /mnt/raid0/llm/llama.cpp/tools/math-tools/
# and the sources added to the existing CMakeLists.txt there.
#
# Commands in this directory:
#   - commands/statistical/mcmc.cpp (Metropolis-Hastings MCMC sampler)
#   - commands/statistical/bayesopt.cpp (Bayesian Optimization with GP)
#   - commands/visualization/render_math.cpp (LaTeX to Unicode converter)
#   - commands/visualization/plot_sixel.cpp (Sixel graphics with braille fallback)
#
# Headers:
#   - include/command.hpp (base command class - may exist in parent)
#   - include/expression.hpp (mathematical expression parser)

# Source files to add to main CMakeLists.txt
set(NEW_COMMAND_SOURCES
    commands/statistical/mcmc.cpp
    commands/statistical/bayesopt.cpp
    commands/visualization/render_math.cpp
    commands/visualization/plot_sixel.cpp
)

# These commands require Eigen (already a dependency)
# No additional dependencies needed

# Factory functions to register in main.cpp:
#
#   // Add declarations at top of main.cpp:
#   std::unique_ptr<Command> create_mcmc();
#   std::unique_ptr<Command> create_bayesopt();
#   std::unique_ptr<Command> create_render_math();
#   std::unique_ptr<Command> create_plot_sixel();
#
#   // Add registrations in main():
#   registry.register_command("mcmc", create_mcmc);
#   registry.register_command("bayesopt", create_bayesopt);
#   registry.register_command("render_math", create_render_math);
#   registry.register_command("plot_sixel", create_plot_sixel);

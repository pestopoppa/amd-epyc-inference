#!/usr/bin/env python3
"""Python wrapper for llama-math-tools C++ binary.

This module provides a Pythonic interface to the native computational tools,
integrating with the orchestrator's tool registry.

Philosophy: LLMs reason. Tools execute.
Every tool call should save 100-5000 tokens of LLM generation.

Usage:
    from orchestration.tools.cpp_tools import MathTools

    tools = MathTools()

    # Matrix operations
    result = tools.matrix_solve([[3, 1], [1, 2]], [9, 8])
    print(result["x"])  # [2.0, 3.0]

    # ODE solving
    result = tools.solve_ode(y0=[1.0], t_span=[0, 10], coefficients={"A": [[-0.5]]})

    # Monte Carlo integration
    result = tools.monte_carlo_integrate(bounds=[[-1, 1], [-1, 1]], function="sphere")

    # Plotting
    result = tools.plot([1, 2, 3, 4], [1, 4, 9, 16], plot_type="line")
    print(result["plot"])  # Braille plot string
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to the compiled C++ binary
MATH_TOOLS_BINARY = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-math-tools")


class MathToolsError(Exception):
    """Error from math tools execution."""

    pass


class MathTools:
    """Python interface to llama-math-tools C++ binary.

    Attributes:
        binary_path: Path to the llama-math-tools executable.
        timeout: Default timeout for tool execution in seconds.
    """

    def __init__(
        self,
        binary_path: Path | str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize the math tools wrapper.

        Args:
            binary_path: Path to llama-math-tools binary.
                        Defaults to /mnt/raid0/llm/llama.cpp/build/bin/llama-math-tools
            timeout: Default timeout in seconds.
        """
        self.binary_path = Path(binary_path) if binary_path else MATH_TOOLS_BINARY
        self.timeout = timeout

        if not self.binary_path.exists():
            logger.warning(
                f"Math tools binary not found at {self.binary_path}. "
                "Build with: cd /mnt/raid0/llm/llama.cpp/tools/math-tools && "
                "cmake -B build && cmake --build build"
            )

    def _call(self, command: str, **params: Any) -> dict[str, Any]:
        """Execute a command via the C++ binary.

        Args:
            command: Command name (e.g., "matrix_op", "solve_ode").
            **params: Command parameters as keyword arguments.

        Returns:
            Parsed JSON response from the tool.

        Raises:
            MathToolsError: If the tool returns an error or times out.
            FileNotFoundError: If the binary doesn't exist.
        """
        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"Math tools binary not found: {self.binary_path}. "
                "Build with: cmake -B build && cmake --build build -j"
            )

        # Build request JSON
        request = {"command": command, **params}
        input_json = json.dumps(request)

        try:
            result = subprocess.run(
                [str(self.binary_path)],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0 and not result.stdout:
                raise MathToolsError(
                    f"Tool failed with return code {result.returncode}: "
                    f"{result.stderr}"
                )

            response = json.loads(result.stdout)

            if response.get("status") == "error":
                raise MathToolsError(response.get("error", "Unknown error"))

            return response

        except subprocess.TimeoutExpired:
            raise MathToolsError(f"Tool timed out after {self.timeout}s")
        except json.JSONDecodeError as e:
            raise MathToolsError(f"Invalid JSON response: {e}")

    # =========================================================================
    # Matrix Operations
    # =========================================================================

    def matrix_solve(
        self,
        A: list[list[float]],
        b: list[float],
    ) -> dict[str, Any]:
        """Solve linear system Ax = b.

        Args:
            A: Coefficient matrix (n x n).
            b: Right-hand side vector (n).

        Returns:
            {"x": solution_vector, "relative_error": float}

        Example:
            result = tools.matrix_solve([[3, 1], [1, 2]], [9, 8])
            # x = [2.0, 3.0]
        """
        response = self._call("matrix_op", operation="solve", A=A, b=b)
        return response.get("result", {})

    def matrix_inverse(self, A: list[list[float]]) -> dict[str, Any]:
        """Compute matrix inverse.

        Args:
            A: Square matrix (n x n).

        Returns:
            {"inverse": matrix, "condition_number": float}
        """
        response = self._call("matrix_op", operation="inverse", A=A)
        return response.get("result", {})

    def matrix_eigenvalues(
        self,
        A: list[list[float]],
        vectors: bool = False,
    ) -> dict[str, Any]:
        """Compute eigenvalues (and optionally eigenvectors).

        Args:
            A: Square matrix (n x n).
            vectors: If True, also compute eigenvectors.

        Returns:
            {"eigenvalues": [...], "eigenvectors": [...] (if vectors=True)}
        """
        response = self._call("matrix_op", operation="eigen", A=A, vectors=vectors)
        return response.get("result", {})

    def matrix_svd(
        self,
        A: list[list[float]],
        full: bool = False,
    ) -> dict[str, Any]:
        """Compute Singular Value Decomposition.

        Args:
            A: Matrix (m x n).
            full: If True, return full U and V matrices.

        Returns:
            {"singular_values": [...], "U": matrix, "V": matrix, "rank": int}
        """
        response = self._call("matrix_op", operation="svd", A=A, full=full)
        return response.get("result", {})

    def matrix_determinant(self, A: list[list[float]]) -> float:
        """Compute matrix determinant.

        Args:
            A: Square matrix (n x n).

        Returns:
            Determinant value.
        """
        response = self._call("matrix_op", operation="det", A=A)
        return response.get("result", {}).get("determinant")

    # =========================================================================
    # ODE Solving
    # =========================================================================

    def solve_ode(
        self,
        y0: list[float],
        t_span: list[float],
        coefficients: dict[str, Any] | None = None,
        system: str | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_steps: int = 10000,
    ) -> dict[str, Any]:
        """Solve initial value problem dy/dt = f(t, y).

        Args:
            y0: Initial condition vector.
            t_span: [t_start, t_end].
            coefficients: For linear systems {"A": matrix, "b": vector (optional)}.
            system: String expression like "-0.5*y".
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            max_steps: Maximum integration steps.

        Returns:
            {"t": times, "y": solutions, "stats": {...}}

        Example:
            # Exponential decay: dy/dt = -0.5*y
            result = tools.solve_ode(
                y0=[1.0],
                t_span=[0, 10],
                coefficients={"A": [[-0.5]]}
            )
        """
        params = {
            "y0": y0,
            "t_span": t_span,
            "rtol": rtol,
            "atol": atol,
            "max_steps": max_steps,
        }

        if coefficients:
            params["coefficients"] = coefficients
        if system:
            params["system"] = system

        response = self._call("solve_ode", **params)
        return response.get("result", {})

    # =========================================================================
    # Optimization
    # =========================================================================

    def optimize(
        self,
        x0: list[float],
        objective: dict[str, Any] | str,
        method: str = "nelder_mead",
        mode: str = "minimize",
        max_iter: int = 1000,
        tolerance: float = 1e-8,
    ) -> dict[str, Any]:
        """Minimize or maximize a function.

        Args:
            x0: Initial guess.
            objective: One of:
                - {"quadratic": {"A": matrix, "b": vector, "c": scalar}}
                - "rosenbrock"
                - "sphere"
            method: "nelder_mead" or "gradient_descent".
            mode: "minimize" or "maximize".
            max_iter: Maximum iterations.
            tolerance: Convergence tolerance.

        Returns:
            {"x": optimal_point, "f": optimal_value, "iterations": int, "converged": bool}

        Example:
            # Minimize f(x) = (x-2)^2 + (y-3)^2
            result = tools.optimize(
                x0=[0, 0],
                objective={"quadratic": {"A": [[2, 0], [0, 2]], "b": [-4, -6], "c": 13}},
            )
            # x ≈ [2, 3]
        """
        params = {
            "x0": x0,
            "method": method,
            "mode": mode,
            "max_iter": max_iter,
            "tolerance": tolerance,
        }

        if isinstance(objective, str):
            params[objective] = True  # e.g., "rosenbrock": True
        else:
            params.update(objective)

        response = self._call("optimize", **params)
        return response.get("result", {})

    # =========================================================================
    # Monte Carlo
    # =========================================================================

    def monte_carlo_integrate(
        self,
        bounds: list[list[float]],
        function: str = "sphere",
        n_samples: int = 10000,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Monte Carlo integration over rectangular domain.

        Args:
            bounds: [[a1, b1], [a2, b2], ...] for each dimension.
            function: "sphere" (indicator), "gaussian", "polynomial", or "constant".
            n_samples: Number of samples.
            seed: Random seed.

        Returns:
            {"integral": value, "standard_error": error, ...}

        Example:
            # Estimate volume of unit circle
            result = tools.monte_carlo_integrate(
                bounds=[[-1, 1], [-1, 1]],
                function="sphere"
            )
            # integral ≈ π
        """
        response = self._call(
            "monte_carlo",
            operation="integrate",
            bounds=bounds,
            function=function,
            n_samples=n_samples,
            seed=seed,
        )
        return response.get("result", {})

    def monte_carlo_pi(self, n_samples: int = 100000, seed: int = 42) -> dict[str, Any]:
        """Estimate π using Monte Carlo.

        Args:
            n_samples: Number of random points.
            seed: Random seed.

        Returns:
            {"pi_estimate": value, "absolute_error": error, ...}
        """
        response = self._call(
            "monte_carlo",
            operation="pi",
            n_samples=n_samples,
            seed=seed,
        )
        return response.get("result", {})

    def bootstrap(
        self,
        data: list[float],
        statistic: str = "mean",
        n_samples: int = 10000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Bootstrap confidence interval estimation.

        Args:
            data: Sample data.
            statistic: "mean", "median", or "std".
            n_samples: Number of bootstrap samples.
            confidence: Confidence level (e.g., 0.95 for 95% CI).
            seed: Random seed.

        Returns:
            {"original_value": v, "confidence_interval": {"lower": l, "upper": u}, ...}
        """
        response = self._call(
            "monte_carlo",
            operation="bootstrap",
            data=data,
            statistic=statistic,
            n_samples=n_samples,
            confidence=confidence,
            seed=seed,
        )
        return response.get("result", {})

    # =========================================================================
    # Visualization
    # =========================================================================

    def plot(
        self,
        x: list[float],
        y: list[float],
        plot_type: str = "line",
        width: int = 80,
        height: int = 20,
        title: str = "Plot",
    ) -> dict[str, Any]:
        """Create a braille character plot.

        Args:
            x: X coordinates.
            y: Y coordinates.
            plot_type: "scatter" or "line".
            width: Plot width in characters.
            height: Plot height in characters.
            title: Plot title.

        Returns:
            {"plot": braille_string, "x_range": [...], "y_range": [...], ...}

        Example:
            result = tools.plot([1, 2, 3, 4], [1, 4, 9, 16], title="y = x²")
            print(result["plot"])
        """
        response = self._call(
            "plot",
            type=plot_type,
            x=x,
            y=y,
            width=width,
            height=height,
            title=title,
        )
        return response.get("result", {})

    def plot_function(
        self,
        function: str = "sin",
        x_min: float = -3.14159,
        x_max: float = 3.14159,
        width: int = 80,
        height: int = 20,
    ) -> dict[str, Any]:
        """Plot a mathematical function.

        Args:
            function: One of: sin, cos, exp, log, x^2, x^3, sqrt, tanh, sigmoid, gaussian.
            x_min: Minimum x value.
            x_max: Maximum x value.
            width: Plot width in characters.
            height: Plot height in characters.

        Returns:
            {"plot": braille_string, ...}
        """
        response = self._call(
            "plot",
            type="function",
            function=function,
            x_min=x_min,
            x_max=x_max,
            width=width,
            height=height,
        )
        return response.get("result", {})

    def histogram(
        self,
        data: list[float],
        bins: int = 20,
        width: int = 80,
        height: int = 20,
        title: str = "Histogram",
    ) -> dict[str, Any]:
        """Create a histogram plot.

        Args:
            data: Data values.
            bins: Number of bins.
            width: Plot width in characters.
            height: Plot height in characters.
            title: Plot title.

        Returns:
            {"plot": braille_string, "n_bins": int, ...}
        """
        response = self._call(
            "plot",
            type="histogram",
            data=data,
            bins=bins,
            width=width,
            height=height,
            title=title,
        )
        return response.get("result", {})

    # =========================================================================
    # MCMC Sampling
    # =========================================================================

    def mcmc(
        self,
        log_density: str,
        x0: list[float],
        n_samples: int = 10000,
        proposal_std: float = 1.0,
        burnin: int = 1000,
        thin: int = 1,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Metropolis-Hastings MCMC sampler.

        Args:
            log_density: Log density expression (vars: x0, x1, ...).
                        e.g., "-0.5*(x0**2 + x1**2)" for 2D normal.
            x0: Initial state vector.
            n_samples: Number of samples after burnin.
            proposal_std: Standard deviation of Gaussian proposal.
            burnin: Number of burnin samples to discard.
            thin: Thinning interval (keep every nth sample).
            seed: Random seed.

        Returns:
            {
                "samples": [[x0, x1, ...], ...],  # Shape: (n_samples, dim)
                "acceptance_rate": float,
                "mean": [float, ...],
                "covariance": [[float, ...], ...],
                "stats": {...}
            }

        Example:
            # Sample from 2D standard normal
            result = tools.mcmc(
                log_density="-0.5*(x0**2 + x1**2)",
                x0=[0.0, 0.0],
                n_samples=5000,
            )
        """
        response = self._call(
            "mcmc",
            log_density=log_density,
            x0=x0,
            n_samples=n_samples,
            proposal_std=proposal_std,
            burnin=burnin,
            thin=thin,
            seed=seed,
        )
        return response.get("result", {})

    # =========================================================================
    # Bayesian Optimization
    # =========================================================================

    def bayesopt(
        self,
        bounds: list[list[float]],
        objective: str,
        n_init: int = 5,
        n_iter: int = 25,
        acquisition: str = "ei",
        noise: float = 1e-6,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Bayesian optimization with Gaussian process surrogate.

        Args:
            bounds: Parameter bounds [[min, max], ...] for each dimension.
            objective: Expression to maximize (vars: x0, x1, ...).
                      e.g., "-(x0-2)**2 - (x1-3)**2" for quadratic.
            n_init: Number of initial random samples.
            n_iter: Number of optimization iterations.
            acquisition: Acquisition function: "ei" (expected improvement),
                        "ucb" (upper confidence bound), "pi" (prob. improvement).
            noise: Observation noise for GP.
            seed: Random seed.

        Returns:
            {
                "x_best": [float, ...],  # Best parameters found
                "y_best": float,         # Best objective value
                "history": {
                    "X": [[x0, x1, ...], ...],
                    "y": [float, ...],
                },
                "model": {...}  # GP hyperparameters
            }

        Example:
            # Find maximum of -(x-2)² - (y-3)²
            result = tools.bayesopt(
                bounds=[[0, 5], [0, 5]],
                objective="-(x0-2)**2 - (x1-3)**2",
                n_iter=20,
            )
            # x_best ≈ [2, 3], y_best ≈ 0
        """
        response = self._call(
            "bayesopt",
            bounds=bounds,
            objective=objective,
            n_init=n_init,
            n_iter=n_iter,
            acquisition=acquisition,
            noise=noise,
            seed=seed,
        )
        return response.get("result", {})

    # =========================================================================
    # Advanced Visualization
    # =========================================================================

    def plot_sixel(
        self,
        x: list[float],
        y: list[float],
        plot_type: str = "line",
        width: int = 800,
        height: int = 400,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        colors: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create high-resolution sixel graphics plot.

        Sixel is a terminal graphics protocol supported by xterm, mlterm,
        WezTerm, and other terminals. Falls back to braille if unsupported.

        Args:
            x: X coordinates.
            y: Y coordinates (or list of y arrays for multi-series).
            plot_type: "line", "scatter", "bar".
            width: Image width in pixels.
            height: Image height in pixels.
            title: Plot title.
            x_label: X-axis label.
            y_label: Y-axis label.
            colors: Color palette (hex codes).

        Returns:
            {
                "sixel": str,      # Sixel escape sequence
                "format": "sixel", # or "braille" if fallback
                "dimensions": {"width": int, "height": int},
            }

        Example:
            result = tools.plot_sixel(
                [0, 1, 2, 3, 4],
                [0, 1, 4, 9, 16],
                title="y = x²",
            )
            print(result["sixel"])  # Prints image in terminal
        """
        params = {
            "x": x,
            "y": y,
            "type": plot_type,
            "width": width,
            "height": height,
        }
        if title:
            params["title"] = title
        if x_label:
            params["x_label"] = x_label
        if y_label:
            params["y_label"] = y_label
        if colors:
            params["colors"] = colors

        response = self._call("plot_sixel", **params)
        return response.get("result", {})

    def render_math(
        self,
        latex: str,
        format: str = "unicode",
    ) -> dict[str, Any]:
        """Render LaTeX expression to Unicode or ASCII.

        Args:
            latex: LaTeX expression (e.g., "\\frac{a}{b}", "x^2 + y^2").
            format: "unicode" for pretty symbols, "ascii" for plain text.

        Returns:
            {
                "rendered": str,   # Rendered expression
                "original": str,   # Original LaTeX
            }

        Examples:
            # Fractions
            tools.render_math("\\frac{dy}{dx}")
            # → {"rendered": "dy/dx"}

            # Greek letters
            tools.render_math("\\alpha + \\beta = \\gamma")
            # → {"rendered": "α + β = γ"}

            # Differential equation
            tools.render_math("\\frac{d^2y}{dx^2} + \\omega^2 y = 0")
            # → {"rendered": "d²y/dx² + ω²y = 0"}
        """
        response = self._call(
            "render_math",
            latex=latex,
            format=format,
        )
        return response.get("result", {})

    # =========================================================================
    # Utilities
    # =========================================================================

    def help(self) -> dict[str, Any]:
        """Get list of available commands.

        Returns:
            {"commands": [{"name": str, "description": str}, ...]}
        """
        response = self._call("help")
        return response.get("result", {})

    def is_available(self) -> bool:
        """Check if the math tools binary is available.

        Returns:
            True if the binary exists and is executable.
        """
        return self.binary_path.exists()


# =============================================================================
# Tool Registry Integration
# =============================================================================

def register_math_tools(registry) -> int:
    """Register C++ math tools with the tool registry.

    Args:
        registry: ToolRegistry instance.

    Returns:
        Number of tools registered.
    """
    from src.tool_registry import Tool, ToolCategory

    tools = MathTools()

    if not tools.is_available():
        logger.warning("C++ math tools not available, skipping registration")
        return 0

    count = 0

    # Matrix operations
    registry.register_tool(Tool(
        name="matrix_solve",
        description="Solve linear system Ax=b using QR decomposition",
        category=ToolCategory.MATH,
        parameters={
            "A": {"type": "array", "required": True, "description": "Coefficient matrix"},
            "b": {"type": "array", "required": True, "description": "Right-hand side vector"},
        },
        handler=lambda A, b: tools.matrix_solve(A, b),
    ), update=True)
    count += 1

    registry.register_tool(Tool(
        name="matrix_eigenvalues",
        description="Compute eigenvalues of a square matrix",
        category=ToolCategory.MATH,
        parameters={
            "A": {"type": "array", "required": True, "description": "Square matrix"},
            "vectors": {"type": "boolean", "required": False, "description": "Also compute eigenvectors"},
        },
        handler=lambda A, vectors=False: tools.matrix_eigenvalues(A, vectors),
    ), update=True)
    count += 1

    # ODE solver
    registry.register_tool(Tool(
        name="solve_ode",
        description="Solve ODE dy/dt=f(t,y) with adaptive RK45",
        category=ToolCategory.MATH,
        parameters={
            "y0": {"type": "array", "required": True, "description": "Initial conditions"},
            "t_span": {"type": "array", "required": True, "description": "[t_start, t_end]"},
            "coefficients": {"type": "object", "required": False, "description": "Linear system {A, b}"},
        },
        handler=lambda y0, t_span, coefficients=None: tools.solve_ode(y0, t_span, coefficients),
    ), update=True)
    count += 1

    # Monte Carlo
    registry.register_tool(Tool(
        name="monte_carlo_integrate",
        description="Monte Carlo numerical integration",
        category=ToolCategory.MATH,
        parameters={
            "bounds": {"type": "array", "required": True, "description": "Integration bounds"},
            "function": {"type": "string", "required": False, "description": "Function type"},
            "n_samples": {"type": "integer", "required": False, "description": "Number of samples"},
        },
        handler=lambda bounds, function="constant", n_samples=10000: tools.monte_carlo_integrate(bounds, function, n_samples),
    ), update=True)
    count += 1

    # Plotting
    registry.register_tool(Tool(
        name="plot_data",
        description="Create braille character plot from x,y data",
        category=ToolCategory.DATA,
        parameters={
            "x": {"type": "array", "required": True, "description": "X coordinates"},
            "y": {"type": "array", "required": True, "description": "Y coordinates"},
            "plot_type": {"type": "string", "required": False, "description": "scatter or line"},
        },
        handler=lambda x, y, plot_type="line": tools.plot(x, y, plot_type),
    ), update=True)
    count += 1

    # MCMC
    registry.register_tool(Tool(
        name="mcmc",
        description="Metropolis-Hastings MCMC sampler",
        category=ToolCategory.MATH,
        parameters={
            "log_density": {"type": "string", "required": True, "description": "Log density expression"},
            "x0": {"type": "array", "required": True, "description": "Initial state"},
            "n_samples": {"type": "integer", "required": False, "description": "Number of samples"},
            "proposal_std": {"type": "number", "required": False, "description": "Proposal std dev"},
        },
        handler=lambda log_density, x0, n_samples=10000, proposal_std=1.0: tools.mcmc(
            log_density, x0, n_samples, proposal_std
        ),
    ), update=True)
    count += 1

    # Bayesian Optimization
    registry.register_tool(Tool(
        name="bayesopt",
        description="Bayesian optimization with Gaussian process",
        category=ToolCategory.MATH,
        parameters={
            "bounds": {"type": "array", "required": True, "description": "Parameter bounds"},
            "objective": {"type": "string", "required": True, "description": "Objective expression"},
            "n_iter": {"type": "integer", "required": False, "description": "Optimization iterations"},
        },
        handler=lambda bounds, objective, n_iter=25: tools.bayesopt(bounds, objective, n_iter=n_iter),
    ), update=True)
    count += 1

    # Sixel Plotting
    registry.register_tool(Tool(
        name="plot_sixel",
        description="Create high-resolution sixel graphics plot",
        category=ToolCategory.DATA,
        parameters={
            "x": {"type": "array", "required": True, "description": "X coordinates"},
            "y": {"type": "array", "required": True, "description": "Y coordinates"},
            "title": {"type": "string", "required": False, "description": "Plot title"},
        },
        handler=lambda x, y, title="": tools.plot_sixel(x, y, title=title),
    ), update=True)
    count += 1

    # Math Rendering
    registry.register_tool(Tool(
        name="render_math",
        description="Render LaTeX to Unicode/ASCII",
        category=ToolCategory.DATA,
        parameters={
            "latex": {"type": "string", "required": True, "description": "LaTeX expression"},
            "format": {"type": "string", "required": False, "description": "unicode or ascii"},
        },
        handler=lambda latex, format="unicode": tools.render_math(latex, format),
    ), update=True)
    count += 1

    logger.info(f"Registered {count} C++ math tools")
    return count

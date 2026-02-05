"""Math tools - numerical, symbolic, statistics."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def calculate(expression: str) -> float | complex:
    """Evaluate mathematical expression safely.

    Available: np.*, math.*, basic operators
    """
    # Safe evaluation namespace
    namespace = {
        "np": np,
        "numpy": np,
        "math": math,
        "pi": np.pi,
        "e": np.e,
        "inf": np.inf,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "sum": np.sum,
        "prod": np.prod,
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }

    # Block dangerous builtins
    namespace["__builtins__"] = {}

    try:
        result = eval(expression, namespace)
        if isinstance(result, np.ndarray):
            if result.size == 1:
                return float(result)
            return result.tolist()
        return result
    except Exception as e:
        return {"error": str(e)}


def statistics(data: list[float], metrics: list[str] | None = None) -> dict:
    """Calculate statistics for a dataset."""
    if metrics is None:
        metrics = ["mean", "std", "min", "max", "median", "count"]

    arr = np.array(data)
    result = {}

    metric_funcs = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
        "median": np.median,
        "count": len,
        "sum": np.sum,
        "var": np.var,
        "q25": lambda x: np.percentile(x, 25),
        "q75": lambda x: np.percentile(x, 75),
        "iqr": lambda x: np.percentile(x, 75) - np.percentile(x, 25),
    }

    for metric in metrics:
        if metric in metric_funcs:
            try:
                value = metric_funcs[metric](arr)
                result[metric] = float(value) if isinstance(value, (np.floating, np.integer)) else value
            except Exception as e:
                result[metric] = {"error": str(e)}

    return result


def monte_carlo(expression: str, samples: int = 10000,
                distribution: str = "uniform") -> dict:
    """Run Monte Carlo simulation.

    Args:
        expression: Expression to evaluate (use 'x' for random variable)
        samples: Number of samples
        distribution: uniform, normal, or exponential
    """
    # Generate random samples
    if distribution == "uniform":
        x = np.random.uniform(0, 1, samples)
    elif distribution == "normal":
        x = np.random.normal(0, 1, samples)
    elif distribution == "exponential":
        x = np.random.exponential(1, samples)
    else:
        return {"error": f"Unknown distribution: {distribution}"}

    # Safe evaluation namespace
    namespace = {
        "x": x,
        "np": np,
        "pi": np.pi,
        "e": np.e,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "__builtins__": {},
    }

    try:
        results = eval(expression, namespace)
        return {
            "mean": float(np.mean(results)),
            "std": float(np.std(results)),
            "percentiles": {
                "5": float(np.percentile(results, 5)),
                "25": float(np.percentile(results, 25)),
                "50": float(np.percentile(results, 50)),
                "75": float(np.percentile(results, 75)),
                "95": float(np.percentile(results, 95)),
            },
            "samples": samples,
        }
    except Exception as e:
        return {"error": str(e)}


def symbolic_solve(equation: str, variable: str = "x") -> list:
    """Solve equation symbolically using SymPy."""
    try:
        from sympy import Symbol, solve, sympify
        from sympy.parsing.sympy_parser import parse_expr

        x = Symbol(variable)
        # Parse the equation (assume it equals 0 if no '=' present)
        if "=" in equation:
            left, right = equation.split("=", 1)
            expr = parse_expr(left) - parse_expr(right)
        else:
            expr = parse_expr(equation)

        solutions = solve(expr, x)
        return [str(sol) for sol in solutions]
    except ImportError:
        return ["error: sympy not installed"]
    except Exception as e:
        return [f"error: {e}"]


def integrate(expression: str, variable: str = "x",
              lower: float | None = None, upper: float | None = None) -> str | float:
    """Integrate expression symbolically or numerically."""
    try:
        from sympy import Symbol, integrate as sym_integrate, sympify
        from sympy.parsing.sympy_parser import parse_expr

        x = Symbol(variable)
        expr = parse_expr(expression)

        if lower is not None and upper is not None:
            # Definite integral
            result = sym_integrate(expr, (x, lower, upper))
            try:
                return float(result.evalf())
            except Exception as e:
                logger.debug("Could not evaluate integral to float: %s", e)
                return str(result)
        else:
            # Indefinite integral
            result = sym_integrate(expr, x)
            return str(result)
    except ImportError:
        return "error: sympy not installed"
    except Exception as e:
        return f"error: {e}"


def differentiate(expression: str, variable: str = "x", order: int = 1) -> str:
    """Differentiate expression symbolically."""
    try:
        from sympy import Symbol, diff
        from sympy.parsing.sympy_parser import parse_expr

        x = Symbol(variable)
        expr = parse_expr(expression)
        result = diff(expr, x, order)
        return str(result)
    except ImportError:
        return "error: sympy not installed"
    except Exception as e:
        return f"error: {e}"

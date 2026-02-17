"""Parallel dispatch for independent read-only tool calls.

When the LLM generates multiple independent read-only tool calls in one turn,
this module extracts them via AST analysis and dispatches them via
ThreadPoolExecutor instead of sequential exec().

The dispatch is conservative: any ambiguity (dependencies between calls,
non-literal args, non-read-only tools) causes a fallback to sequential exec().

Expected gain: 2-4x speedup on multi-tool turns (10-450ms saved per turn).
"""

from __future__ import annotations

import ast
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _ParallelCall:
    """A single tool call extracted from AST for parallel dispatch."""

    func_name: str
    args: list[Any]
    kwargs: dict[str, Any]
    target_var: str | None  # Variable assigned to, or None for bare calls
    index: int  # Original order in source code


def _eval_ast_arg(node: ast.expr, globals_dict: dict[str, Any]) -> Any | None:
    """Safely evaluate an AST argument node to a Python value.

    Only handles literals and simple name lookups (no function calls,
    attribute access, or complex expressions).

    Returns None if the argument cannot be safely evaluated, which signals
    the caller to abort parallel dispatch.
    """
    # Literal values: strings, numbers, booleans, None
    if isinstance(node, ast.Constant):
        return node.value

    # Simple variable reference (e.g., a name already in globals)
    if isinstance(node, ast.Name):
        if node.id in globals_dict:
            return globals_dict[node.id]
        return None  # Unknown variable — abort

    # List literal: [a, b, c]
    if isinstance(node, ast.List):
        items = []
        for elt in node.elts:
            val = _eval_ast_arg(elt, globals_dict)
            if val is None and not isinstance(elt, ast.Constant):
                return None
            items.append(val)
        return items

    # Tuple literal: (a, b)
    if isinstance(node, ast.Tuple):
        items = []
        for elt in node.elts:
            val = _eval_ast_arg(elt, globals_dict)
            if val is None and not isinstance(elt, ast.Constant):
                return None
            items.append(val)
        return tuple(items)

    # Dict literal: {"key": val}
    if isinstance(node, ast.Dict):
        result = {}
        for k, v in zip(node.keys, node.values):
            if k is None:
                return None  # ** unpacking
            key = _eval_ast_arg(k, globals_dict)
            val = _eval_ast_arg(v, globals_dict)
            if key is None:
                return None
            result[key] = val
        return result

    # Unary operator: -1, not x
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _eval_ast_arg(node.operand, globals_dict)
        if operand is not None:
            return -operand

    return None  # Complex expression — abort


def _extract_parallel_calls(
    code: str,
    globals_dict: dict[str, Any],
    read_only_tools: set[str],
) -> list[_ParallelCall] | None:
    """Extract independent read-only calls from code via AST analysis.

    Returns a list of _ParallelCall if ALL calls are:
    - Top-level statements (assignments or bare expressions)
    - Calls to functions in read_only_tools
    - Arguments are literals or pre-existing globals (no cross-references)

    Returns None to signal fallback to sequential exec() if:
    - Parse fails
    - Any non-call statement found
    - Any dependency between calls
    - Any non-read-only tool
    - Any complex expression in arguments
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    calls: list[_ParallelCall] = []
    assigned_vars: set[str] = set()

    for i, stmt in enumerate(tree.body):
        # Pattern 1: var = tool(args)
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                return None  # Multi-assignment
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                return None  # Complex target
            if not isinstance(stmt.value, ast.Call):
                return None  # Not a call
            call_node = stmt.value
            target_var = target.id
            assigned_vars.add(target_var)

        # Pattern 2: bare tool(args)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call_node = stmt.value
            target_var = None

        else:
            return None  # Non-call statement

        # Extract function name
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
        else:
            return None  # Method call or complex callable

        # Must be read-only
        if func_name not in read_only_tools:
            return None

        # Must be a callable in globals
        if func_name not in globals_dict or not callable(globals_dict[func_name]):
            return None

        # Evaluate positional args
        args = []
        for arg_node in call_node.args:
            # Check for dependency on vars assigned in this block
            if isinstance(arg_node, ast.Name) and arg_node.id in assigned_vars:
                return None  # Cross-dependency
            val = _eval_ast_arg(arg_node, globals_dict)
            if val is None and not (isinstance(arg_node, ast.Constant) and arg_node.value is None):
                return None  # Unevaluable arg
            args.append(val)

        # Evaluate keyword args
        kwargs = {}
        for kw in call_node.keywords:
            if kw.arg is None:
                return None  # **kwargs unpacking
            if isinstance(kw.value, ast.Name) and kw.value.id in assigned_vars:
                return None  # Cross-dependency
            val = _eval_ast_arg(kw.value, globals_dict)
            if val is None and not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                return None
            kwargs[kw.arg] = val

        calls.append(_ParallelCall(
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            target_var=target_var,
            index=i,
        ))

    return calls if len(calls) > 1 else None


def execute_parallel_calls(
    calls: list[_ParallelCall],
    globals_dict: dict[str, Any],
    state_lock,
) -> dict[str, Any]:
    """Execute extracted calls in parallel via ThreadPoolExecutor.

    Args:
        calls: List of extracted parallel calls.
        globals_dict: The REPL globals dict (for looking up callables).
        state_lock: Lock for thread-safe state mutations in tool implementations.

    Returns:
        Dict mapping target_var (or f"_result_{index}") to call results.
    """
    max_workers = min(4, len(calls))
    results: dict[str, Any] = {}

    # Pre-allocate result slots indexed by order
    result_list: list[Any] = [None] * len(calls)

    def _run_call(idx: int, call: _ParallelCall) -> None:
        func = globals_dict[call.func_name]
        try:
            result_list[idx] = func(*call.args, **call.kwargs)
        except Exception as e:
            logger.warning(f"Parallel call {call.func_name} failed: {e}")
            result_list[idx] = f"[ERROR: {e}]"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, call in enumerate(calls):
            futures.append(executor.submit(_run_call, idx, call))

        # Wait for all to complete
        for f in futures:
            f.result()

    # Map results to variable names
    for idx, call in enumerate(calls):
        key = call.target_var or f"_result_{call.index}"
        results[key] = result_list[idx]

    return results

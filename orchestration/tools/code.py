"""Code tools - linting, formatting, git, execution."""

import ast
import subprocess
import time
from typing import Any


def python_eval(expression: str, variables: dict | None = None) -> Any:
    """Evaluate Python expression safely."""
    # Build safe namespace
    import math
    namespace = {
        "math": math,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "sorted": sorted,
        "reversed": reversed,
        "any": any,
        "all": all,
        "True": True,
        "False": False,
        "None": None,
    }

    if variables:
        namespace.update(variables)

    # Block dangerous builtins
    namespace["__builtins__"] = {}

    try:
        # Parse to check for dangerous operations
        tree = ast.parse(expression, mode="eval")
        # Could add AST validation here

        result = eval(expression, namespace)
        return result
    except Exception as e:
        return {"error": str(e)}


def run_shell(command: str, timeout: int = 30, cwd: str | None = None) -> dict:
    """Execute shell command with sandbox restrictions."""
    # Block dangerous commands
    dangerous = ["rm -rf /", "mkfs", "dd if=", "> /dev/", "chmod -R 777 /"]
    if any(d in command for d in dangerous):
        return {"error": "Command blocked for safety"}

    start = time.time()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "stdout": result.stdout[:50000],
            "stderr": result.stderr[:10000],
            "returncode": result.returncode,
            "elapsed_ms": int((time.time() - start) * 1000),
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "returncode": -1,
            "elapsed_ms": timeout * 1000,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "elapsed_ms": int((time.time() - start) * 1000),
        }


def git_status(repo_path: str = ".") -> dict:
    """Get git repository status."""
    result = run_shell(f"cd {repo_path} && git status --porcelain -b", timeout=10)
    if result.get("returncode") != 0:
        return {"error": result.get("stderr", "git status failed")}

    lines = result["stdout"].strip().split("\n")
    status = {
        "branch": "",
        "modified": [],
        "staged": [],
        "untracked": [],
    }

    for line in lines:
        if line.startswith("##"):
            status["branch"] = line[3:].split("...")[0]
        elif line.startswith("??"):
            status["untracked"].append(line[3:])
        elif line.startswith(" M"):
            status["modified"].append(line[3:])
        elif line.startswith("M "):
            status["staged"].append(line[3:])
        elif line.startswith("A "):
            status["staged"].append(line[3:])

    return status


def lint_python(code: str, fix: bool = False) -> dict:
    """Lint Python code using ruff if available, else basic checks."""
    issues = []

    # Try ruff first
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        cmd = f"ruff check {temp_path} --output-format=json"
        if fix:
            cmd += " --fix"

        result = run_shell(cmd, timeout=10)

        import os
        if fix:
            with open(temp_path) as f:
                fixed_code = f.read()
        else:
            fixed_code = code

        os.unlink(temp_path)

        if result["stdout"]:
            import json
            try:
                issues = json.loads(result["stdout"])
            except:
                issues = [{"message": result["stdout"]}]

        return {
            "issues": issues,
            "fixed_code": fixed_code if fix else None,
            "summary": f"{len(issues)} issues found",
        }
    except Exception:
        pass

    # Fallback: basic AST check
    try:
        ast.parse(code)
        return {
            "issues": [],
            "fixed_code": None,
            "summary": "No syntax errors (basic check only, ruff not available)",
        }
    except SyntaxError as e:
        return {
            "issues": [{
                "line": e.lineno,
                "message": str(e.msg),
                "code": "E999",
            }],
            "fixed_code": None,
            "summary": f"Syntax error at line {e.lineno}",
        }


def format_python(code: str) -> dict:
    """Format Python code using black if available."""
    try:
        import black
        formatted = black.format_str(code, mode=black.Mode())
        return {
            "formatted_code": formatted,
            "changed": formatted != code,
        }
    except ImportError:
        return {"error": "black not installed", "formatted_code": code, "changed": False}
    except Exception as e:
        return {"error": str(e), "formatted_code": code, "changed": False}


def parse_python(code: str) -> dict:
    """Parse Python code and return AST summary."""
    try:
        tree = ast.parse(code)

        functions = []
        classes = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "line": node.lineno,
                })
            elif isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "bases": [ast.unparse(b) if hasattr(ast, 'unparse') else str(b) for b in node.bases],
                    "line": node.lineno,
                })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"from {node.module}")

        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "lines": len(code.split("\n")),
        }
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

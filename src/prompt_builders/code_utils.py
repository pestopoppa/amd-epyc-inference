"""Code extraction and error classification utilities."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.escalation import ErrorCategory


def _strip_import_lines(code: str) -> str:
    """Strip top-level import/from lines since safe modules are pre-loaded in REPL globals.

    Models frequently generate 'import math' or 'import os' even when told not to.
    Safe modules (math, json, re, collections, numpy, scipy, etc.) are pre-loaded
    in _build_globals(); unsafe modules would be blocked by _safe_import() anyway.

    Skips lines inside triple-quoted strings to avoid corrupting embedded code
    (e.g., USACO solutions wrapped in solution = \"\"\"import sys...\"\"\" ).
    """
    lines = code.split("\n")
    filtered = []
    in_triple_quote = False
    triple_char = None
    for line in lines:
        # Track triple-quoted string boundaries
        stripped = line.strip()
        if not in_triple_quote:
            # Check if this line opens a triple-quoted string
            for tq in ('"""', "'''"):
                count = stripped.count(tq)
                if count % 2 == 1:  # Odd number = we entered/exited
                    in_triple_quote = True
                    triple_char = tq
                    break
            # Only strip imports at top level (outside strings)
            if not in_triple_quote and (
                stripped.startswith("import ") or stripped.startswith("from ")
            ):
                continue
        else:
            # Inside a triple-quoted string — check if it closes
            if triple_char and triple_char in stripped:
                count = stripped.count(triple_char)
                if count % 2 == 1:
                    in_triple_quote = False
                    triple_char = None
        filtered.append(line)
    return "\n".join(filtered).strip()


def extract_code_from_response(response: str) -> str:
    """Extract Python code from an LLM response.

    Handles responses that may be wrapped in markdown code blocks
    or contain explanatory text. Also strips import lines since
    all needed modules are pre-loaded in the REPL globals.
    """
    response = response.strip()

    # Remove trailing backticks that aren't properly paired
    # (model sometimes outputs code followed by ``` without opening)
    if response.endswith("```"):
        # Check if there's a matching opening
        if response.count("```") % 2 == 1:
            response = response[:-3].rstrip()

    # Try to extract from markdown code block
    code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)

    if matches:
        return _strip_import_lines(matches[0].strip())

    # If no code block, try to find code-like content
    lines = response.split("\n")
    code_lines = []
    in_code = False

    # Include REPL tool functions as code starters
    code_starters = [
        "import ",
        "from ",
        "def ",
        "class ",
        "if ",
        "for ",
        "while ",
        "try:",
        "except",
        "with ",
        "return ",
        "print(",
        "FINAL(",
        "artifacts[",
        "result =",
        "answer =",
        "output =",
        # REPL tools
        "peek(",
        "grep(",
        "list_dir(",
        "file_info(",
        "ocr_document(",
        "analyze_figure(",
        "extract_figure(",
        "web_fetch(",
        "run_shell(",
        "recall(",
        "escalate(",
        "llm_call(",
        "llm_batch(",
    ]

    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(kw) for kw in code_starters):
            in_code = True

        if not in_code and ("=" in stripped or "()" in stripped):
            # Assignment or call expression — treat as code even before
            # seeing a formal code starter keyword.
            in_code = True

        if in_code:
            # Once we've seen a code starter, include continuation lines
            # (comments, assignments, expressions)
            code_lines.append(line)
        elif stripped.startswith("#"):
            # Standalone comments before code — include as potential preamble
            code_lines.append(line)

    if code_lines:
        # Strip common leading whitespace from all lines
        code = "\n".join(code_lines)
        # Dedent the code to remove consistent leading whitespace
        import textwrap

        code = textwrap.dedent(code).strip()
        # Strip import lines - modules like json are pre-loaded in REPL globals
        code = _strip_import_lines(code)
        return code

    # Fallback: return the whole response, dedented
    import textwrap

    code = textwrap.dedent(response).strip()
    code = _strip_import_lines(code)
    return code


def auto_wrap_final(code: str) -> str:
    """Auto-wrap code in FINAL() if it looks like a final answer.

    This is a deterministic wrapper that detects when the model has generated
    complete code but didn't wrap it in FINAL(). This allows models to generate
    code naturally while still signaling completion to the orchestrator.

    Args:
        code: Extracted code from the model's response.

    Returns:
        Code wrapped in FINAL() if it's a final answer, otherwise unchanged.
    """
    # Already has FINAL - check if it's reachable at module level.
    # Models often define main() with FINAL() inside, plus
    # `if __name__ == "__main__": main()` — but in a REPL exec() context
    # __name__ is NOT "__main__", so FINAL() is never reached.
    if "FINAL(" in code:
        # Check if FINAL() only appears inside indented (function) blocks
        has_toplevel_final = any(
            ln.startswith("FINAL(") for ln in code.split("\n")
            if ln.strip() and not ln.strip().startswith("#")
        )
        if has_toplevel_final:
            return code
        # FINAL() is only inside a function.  Ensure the function is called.
        # Common pattern: def main(): ... FINAL(...) + if __name__=="__main__": main()
        # In exec() context, __name__ != "__main__" so add a bare main() call.
        if re.search(r"^def main\s*\(", code, re.MULTILINE):
            has_main_call = re.search(
                r"^main\s*\(", code, re.MULTILINE
            )
            if not has_main_call:
                code = code.rstrip() + "\nmain()\n"
        return code

    # Has exploration/continuation functions - not a final answer
    exploration_patterns = [
        "peek(",  # Exploring context
        "grep(",  # Searching context
        "llm_call(",  # Delegating to sub-LM
        "llm_batch(",  # Batch delegation
        "artifacts[",  # Storing intermediate results
    ]
    for pattern in exploration_patterns:
        if pattern in code:
            return code

    # Get non-empty, non-comment lines
    lines = [
        line.strip()
        for line in code.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return code

    # Code starting with def/class is likely a complete answer —
    # UNLESS it also calls print() or has a trailing expression that
    # invokes the function.  In that case the model wants the code to
    # execute and produce output, not be returned as a string.
    if lines[0].startswith(("def ", "class ")):
        has_print = any("print(" in ln for ln in lines)
        # Trailing call: check original (unstripped) lines for a top-level
        # statement after the def/class (not indented = not part of the body)
        raw_nonempty = [
            ln for ln in code.split("\n")
            if ln.strip() and not ln.strip().startswith("#")
        ]
        # Check if ANY top-level line (after the opening def/class) calls
        # or assigns — indicating the model wants the code to execute.
        has_trailing_call = False
        for raw_ln in raw_nonempty[1:]:
            if raw_ln.startswith((" ", "\t")):
                continue  # inside a function/class body
            s = raw_ln.strip()
            if s.startswith(("def ", "class ")):
                continue  # another definition, not a call
            if "(" in s or "=" in s:
                has_trailing_call = True
                break
        if not has_print and not has_trailing_call:
            # Pure definition — the code IS the answer
            escaped_code = code.replace("'''", r"\'\'\'")
            return f"FINAL('''{escaped_code}''')"

    # Single expression/value is likely a final answer
    # Exclude control flow and imports
    if len(lines) == 1:
        first_line = lines[0]
        # Error payloads from upper layers often come as bracketed strings
        # like "[ERROR: ...]". Wrapping them as FINAL([ERROR: ...]) yields
        # invalid Python syntax, so coerce to a quoted string.
        if first_line.startswith("[ERROR:") and first_line.endswith("]"):
            safe = first_line.replace("\\", "\\\\").replace('"', '\\"')
            return f'FINAL("{safe}")'
        non_final_patterns = [
            "import ",
            "from ",
            "for ",
            "while ",
            "if ",
            "try:",
            "with ",
        ]
        if not any(first_line.startswith(p) for p in non_final_patterns):
            return f"FINAL({first_line})"

    return code


# Error classification utilities
def classify_error(error_message: str, gate_name: str = "") -> ErrorCategory:
    """Classify an error message into an ErrorCategory.

    Args:
        error_message: The error message to classify.
        gate_name: Optional gate name if error came from a gate.

    Returns:
        ErrorCategory for the error.
    """
    # Import here to avoid circular imports
    from src.escalation import ErrorCategory

    error_lower = error_message.lower()

    # Schema/format errors (from gates or parsing)
    if gate_name in ("schema", "format", "lint", "mdformat", "shfmt"):
        return ErrorCategory.FORMAT
    if "schema" in error_lower or "validation" in error_lower:
        return ErrorCategory.SCHEMA
    if "format" in error_lower or "style" in error_lower:
        return ErrorCategory.FORMAT

    # Code errors (syntax, type, import)
    code_keywords = [
        "syntaxerror",
        "indentationerror",
        "typeerror",
        "nameerror",
        "importerror",
        "modulenotfound",
        "attributeerror",
    ]
    if any(kw in error_lower for kw in code_keywords):
        return ErrorCategory.CODE

    # Logic errors (test failures, assertions)
    logic_keywords = ["assertionerror", "test failed", "expected", "actual"]
    if any(kw in error_lower for kw in logic_keywords):
        return ErrorCategory.LOGIC

    # Timeout errors
    if "timeout" in error_lower or "timed out" in error_lower:
        return ErrorCategory.TIMEOUT

    # Early abort (from generation monitor)
    if "early abort" in error_lower or "high entropy" in error_lower:
        return ErrorCategory.EARLY_ABORT

    return ErrorCategory.UNKNOWN

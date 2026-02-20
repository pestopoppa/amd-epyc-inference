"""Code extraction and error classification utilities."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.escalation import ErrorCategory

_log = logging.getLogger(__name__)

# Regex matching OpenAI-format tool_call JSON objects.  The model sometimes
# emits these instead of the REPL ``CALL()`` syntax because Qwen3-Coder's
# instruct training activates the chat-completion tool-calling format even
# in raw-completion mode.
_TOOL_CALL_RE = re.compile(
    r'"function"\s*:\s*\{\s*"(?:arguments|name)"\s*:'
    r'|'
    r'"id"\s*:\s*"call_[^"]+"\s*,\s*"function"\s*:',
)


def _extract_json_arrays(text: str) -> list[list[dict]]:
    """Extract JSON arrays from text by scanning for balanced brackets."""
    arrays: list[list[dict]] = []
    i = 0
    while i < len(text):
        if text[i] == '[':
            # Scan forward tracking depth and string literals
            depth = 0
            in_str = False
            escape = False
            j = i
            while j < len(text):
                c = text[j]
                if escape:
                    escape = False
                elif c == '\\' and in_str:
                    escape = True
                elif c == '"' and not escape:
                    in_str = not in_str
                elif not in_str:
                    if c == '[':
                        depth += 1
                    elif c == ']':
                        depth -= 1
                        if depth == 0:
                            try:
                                arr = json.loads(text[i:j + 1])
                                if isinstance(arr, list):
                                    arrays.append(arr)
                            except (json.JSONDecodeError, ValueError):
                                pass
                            i = j + 1
                            break
                j += 1
            else:
                i += 1
        else:
            i += 1
    return arrays


def translate_openai_tool_calls(text: str) -> str | None:
    """Detect OpenAI-format tool_call JSON in raw LLM output and translate to CALL() code.

    When a model emits ``[{"id":"call_...","function":{"name":"web_search",
    "arguments":"{\"query\":\"...\"}"},"type":"function"}]`` instead of
    ``CALL("web_search", query="...")``, the REPL cannot execute it.

    This function extracts the *unique* tool calls, deduplicates them, and
    returns equivalent Python code using ``CALL()`` syntax.  Returns ``None``
    if no tool_call JSON is detected.
    """
    if not _TOOL_CALL_RE.search(text):
        return None

    # Extract all JSON arrays that look like tool_call lists.
    calls_seen: list[tuple[str, dict]] = []  # (name, kwargs) deduped
    seen_keys: set[str] = set()

    # Find JSON array boundaries robustly.  The model emits space-separated
    # ``[{...}] [{...}]`` blocks; simple regex can't handle nested braces
    # in the "arguments" field, so we scan for ``[`` and find the matching
    # ``]`` by tracking brace/bracket depth.
    for arr in _extract_json_arrays(text):
        for item in arr:
            if not isinstance(item, dict):
                continue
            func = item.get("function") or {}
            name = func.get("name")
            if not name:
                continue
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, ValueError):
                args = {}
            # Dedup key: (name, sorted args)
            dedup_key = (name, tuple(sorted(args.items())))
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            calls_seen.append((name, args))

    if not calls_seen:
        return None

    _log.info(
        "Translated %d OpenAI-format tool_call(s) to CALL() syntax: %s",
        len(calls_seen),
        [c[0] for c in calls_seen],
    )

    # Build CALL() code lines.  Do NOT auto-wrap with FINAL() — the model
    # wants to inspect tool results and continue reasoning in the next turn.
    lines: list[str] = []
    for i, (name, kwargs) in enumerate(calls_seen):
        kw_parts = ", ".join(f'{k}={json.dumps(v)}' for k, v in kwargs.items())
        var = f"result_{i}" if len(calls_seen) > 1 else "result"
        lines.append(f'{var} = CALL("{name}", {kw_parts})')

    # Print results so the REPL captures output for the next turn
    if len(calls_seen) == 1:
        lines.append("print(result)")
    else:
        for i in range(len(calls_seen)):
            lines.append(f"print(result_{i})")

    return "\n".join(lines)


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

    # Intercept OpenAI-format tool_call JSON (Qwen3-Coder instruct artifact)
    # and translate to CALL() syntax before normal code extraction.
    translated = translate_openai_tool_calls(response)
    if translated is not None:
        return translated

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
        "CALL(",
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

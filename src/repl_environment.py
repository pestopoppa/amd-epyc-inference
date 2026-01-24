#!/usr/bin/env python3
"""Sandboxed Python REPL environment for RLM-style orchestration.

This module provides a restricted Python execution environment where:
- Large context is stored as a variable, not sent to LLM
- Built-in functions (peek, grep, FINAL) enable context manipulation
- Dangerous operations (file I/O, subprocess, imports) are blocked
- Execution is time-limited and output-capped

Usage:
    from src.repl_environment import REPLEnvironment

    repl = REPLEnvironment(context="Large document here...")
    output, is_final = repl.execute("print(peek(100))")
    output, is_final = repl.execute("FINAL('Done!')")
"""

from __future__ import annotations

import ast
import io
import re
import signal
import sys
import uuid
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from orchestration.repl_memory.progress_logger import ProgressLogger
    from orchestration.repl_memory.retriever import TwoPhaseRetriever


class REPLError(Exception):
    """Error during REPL execution."""

    pass


class ASTSecurityVisitor(ast.NodeVisitor):
    """AST visitor that checks for dangerous code patterns.

    This is more robust than regex because it analyzes the parsed syntax tree,
    making it immune to string concatenation tricks like:
        getattr(__builtins__, '__im' + 'port__')('os')
    """

    # Forbidden module imports
    FORBIDDEN_MODULES = frozenset({
        "os", "sys", "subprocess", "socket", "shutil", "pathlib",
        "tempfile", "multiprocessing", "threading", "ctypes", "pickle",
        "importlib", "builtins", "code", "codeop", "runpy", "pkgutil",
    })

    # Forbidden built-in function calls
    FORBIDDEN_CALLS = frozenset({
        "__import__", "eval", "exec", "compile", "open",
        "getattr", "setattr", "delattr", "hasattr",
        "globals", "locals", "vars", "dir",
        "input", "breakpoint", "memoryview",
    })

    # Forbidden attribute accesses (dunder attributes for escaping sandbox)
    FORBIDDEN_ATTRS = frozenset({
        "__class__", "__bases__", "__subclasses__", "__mro__",
        "__dict__", "__globals__", "__locals__", "__code__",
        "__builtins__", "__closure__", "__func__", "__self__",
        "__module__", "__qualname__", "__annotations__",
        "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
    })

    def __init__(self):
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Check regular imports: import os"""
        for alias in node.names:
            module = alias.name.split(".")[0]
            if module in self.FORBIDDEN_MODULES:
                self.violations.append(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from imports: from os import path"""
        if node.module:
            module = node.module.split(".")[0]
            if module in self.FORBIDDEN_MODULES:
                self.violations.append(f"from {node.module} import ...")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for forbidden functions."""
        # Check direct calls: eval(...)
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_CALLS:
                self.violations.append(f"{node.func.id}()")

        # Check attribute calls: obj.__class__()
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.FORBIDDEN_ATTRS:
                self.violations.append(f".{node.func.attr}()")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access for forbidden dunder attributes."""
        if node.attr in self.FORBIDDEN_ATTRS:
            self.violations.append(f".{node.attr}")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Check subscript access for string-based dunder bypass attempts.

        Catches patterns like: obj['__class__'] or obj["__globals__"]
        """
        if isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                if node.slice.value in self.FORBIDDEN_ATTRS:
                    self.violations.append(f"['{node.slice.value}']")
        self.generic_visit(node)


class REPLTimeout(REPLError):
    """Execution timed out."""

    pass


class REPLSecurityError(REPLError):
    """Attempted to execute dangerous code."""

    pass


class FinalSignal(Exception):
    """Signal that FINAL() was called with the answer."""

    def __init__(self, answer: str):
        self.answer = answer
        super().__init__(answer)


@dataclass
class ExplorationEvent:
    """A single exploration event for logging."""

    function: str  # peek, grep, llm_call, llm_batch
    args: dict[str, Any]  # Arguments passed
    result_size: int  # Size of result (chars or items)
    timestamp: float  # Time of call
    token_estimate: int = 0  # Estimated tokens used


@dataclass
class ExplorationLog:
    """Log of exploration events for a REPL session."""

    events: list[ExplorationEvent] = field(default_factory=list)
    total_exploration_tokens: int = 0

    def add_event(
        self,
        function: str,
        args: dict[str, Any],
        result: Any,
    ) -> None:
        """Add an exploration event to the log."""
        import time

        # Estimate result size
        if isinstance(result, str):
            result_size = len(result)
        elif isinstance(result, list):
            result_size = len(result)
        else:
            result_size = 0

        # Rough token estimate (4 chars per token)
        token_estimate = result_size // 4

        event = ExplorationEvent(
            function=function,
            args=args,
            result_size=result_size,
            timestamp=time.time(),
            token_estimate=token_estimate,
        )
        self.events.append(event)
        self.total_exploration_tokens += token_estimate

    def get_strategy_summary(self) -> dict[str, Any]:
        """Get a summary of the exploration strategy used."""
        function_counts: dict[str, int] = {}
        for event in self.events:
            function_counts[event.function] = function_counts.get(event.function, 0) + 1

        return {
            "total_events": len(self.events),
            "function_counts": function_counts,
            "total_tokens": self.total_exploration_tokens,
            "strategy_type": self._classify_strategy(function_counts),
        }

    def get_token_efficiency(self, result_tokens: int) -> dict[str, Any]:
        """Calculate token efficiency metrics.

        Token efficiency = result_tokens / exploration_tokens
        Higher is better - means we got more useful output per exploration token spent.

        Args:
            result_tokens: Tokens in the final result (estimated as len(result)/4).

        Returns:
            Dictionary with efficiency metrics.
        """
        if self.total_exploration_tokens == 0:
            efficiency = float("inf") if result_tokens > 0 else 0.0
        else:
            efficiency = result_tokens / self.total_exploration_tokens

        return {
            "exploration_tokens": self.total_exploration_tokens,
            "result_tokens": result_tokens,
            "efficiency_ratio": round(efficiency, 3),
            "total_events": len(self.events),
        }

    def _classify_strategy(self, counts: dict[str, int]) -> str:
        """Classify the exploration strategy based on function usage."""
        if not counts:
            return "none"
        if counts.get("llm_call", 0) > 0 or counts.get("llm_batch", 0) > 0:
            return "delegated"  # Used sub-LLM calls
        if counts.get("grep", 0) > counts.get("peek", 0):
            return "search"  # Primarily used grep
        if counts.get("peek", 0) > 0:
            return "scan"  # Primarily used peek
        return "mixed"


@dataclass
class REPLConfig:
    """Configuration for the REPL environment."""

    timeout_seconds: int = 600  # 10 min default (was 120) for document processing
    output_cap: int = 8192
    max_grep_results: int = 100
    # Forced exploration validation (prevent premature FINAL)
    # Default False for backwards compatibility - enable for production use
    require_exploration_before_final: bool = False
    min_exploration_calls: int = 1  # Minimum peek/grep/llm_call before FINAL
    allowed_builtins: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                # Safe builtins
                "abs",
                "all",
                "any",
                "ascii",
                "bin",
                "bool",
                "bytes",
                "chr",
                "dict",
                "divmod",
                "enumerate",
                "filter",
                "float",
                "format",
                "frozenset",
                "hash",
                "hex",
                "int",
                "isinstance",
                "issubclass",
                "iter",
                "len",
                "list",
                "map",
                "max",
                "min",
                "next",
                "oct",
                "ord",
                "pow",
                "print",
                "range",
                "repr",
                "reversed",
                "round",
                "set",
                "slice",
                "sorted",
                "str",
                "sum",
                "tuple",
                "type",
                "zip",
                # Exceptions (needed for try/except)
                "Exception",
                "ValueError",
                "TypeError",
                "KeyError",
                "IndexError",
                "AttributeError",
                "RuntimeError",
                "StopIteration",
                "ZeroDivisionError",
                "NameError",
                "FileNotFoundError",
                "IOError",
                "OSError",
                # Constants
                "True",
                "False",
                "None",
            }
        )
    )


@dataclass
class ExecutionResult:
    """Result of executing code in the REPL."""

    output: str
    is_final: bool
    final_answer: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0


class REPLEnvironment:
    """Sandboxed Python REPL with context-as-variable pattern.

    The REPL provides:
    - `context`: The full input as a string (never sent to LLM)
    - `artifacts`: Dict for storing intermediate results
    - `peek(n)`: Return first n characters of context
    - `grep(pattern)`: Regex search in context
    - `FINAL(answer)`: Signal completion with final answer
    - `FINAL_VAR(var_name)`: Signal completion, return variable contents

    When tool_registry is provided:
    - `TOOL(name, **kwargs)`: Invoke a registered tool
    - `list_tools()`: List available tools for current role

    When script_registry is provided:
    - `SCRIPT(id, **kwargs)`: Invoke a prepared script by ID
    - `find_scripts(query)`: Find scripts matching a description
    """

    def __init__(
        self,
        context: str,
        artifacts: dict[str, Any] | None = None,
        config: REPLConfig | None = None,
        llm_primitives: Any | None = None,  # LLMPrimitives instance
        tool_registry: Any | None = None,  # ToolRegistry instance
        script_registry: Any | None = None,  # ScriptRegistry instance
        role: str | None = None,  # Role for permission checking
        progress_logger: ProgressLogger | None = None,  # For exploration logging
        task_id: str | None = None,  # Task ID for progress logging
    ):
        """Initialize the REPL environment.

        Args:
            context: The full input context (stored as variable, not in LLM prompt).
            artifacts: Optional dict of pre-existing artifacts from previous turns.
            config: Optional configuration for timeouts, output caps, etc.
            llm_primitives: Optional LLMPrimitives instance for llm_call/llm_batch.
            tool_registry: Optional ToolRegistry for TOOL() invocations.
            script_registry: Optional ScriptRegistry for SCRIPT() invocations.
            role: Role name for permission checking (e.g., "frontdoor", "coder_primary").
            progress_logger: Optional ProgressLogger for logging exploration events.
            task_id: Optional task ID for associating exploration logs with tasks.
        """
        self.context = context
        self.artifacts = artifacts if artifacts is not None else {}
        self.config = config if config is not None else REPLConfig()
        self.llm_primitives = llm_primitives
        self.progress_logger = progress_logger
        self.task_id = task_id or str(uuid.uuid4())
        self.tool_registry = tool_registry
        self.script_registry = script_registry
        self.role = role or "worker_general"  # Default to restricted role

        # Execution state
        self._final_answer: str | None = None
        self._execution_count = 0

        # Exploration tracking (for forced exploration validation)
        self._exploration_calls = 0  # Count of peek/grep/llm_call calls
        self._exploration_log = ExplorationLog()  # Detailed exploration log

        # Build restricted globals
        self._globals = self._build_globals()

    def _build_globals(self) -> dict[str, Any]:
        """Build the restricted globals dict for exec()."""
        import builtins

        # Build safe builtins from the builtins module
        safe_builtins = {}
        for name in self.config.allowed_builtins:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        # Import json for REPL use (needed for document processing)
        import json as _json

        globals_dict = {
            "__builtins__": safe_builtins,
            # Context variables
            "context": self.context,
            "artifacts": self.artifacts,
            # Safe modules
            "json": _json,
            # Built-in functions
            "peek": self._peek,
            "grep": self._grep,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            # Document processing tools
            "ocr_document": self._ocr_document,
            "analyze_figure": self._analyze_figure,
            "extract_figure": self._extract_figure,
            # File system tools
            "list_dir": self._list_dir,
            "file_info": self._file_info,
            # Web tools
            "web_fetch": self._web_fetch,
            # Memory tools
            "recall": self._recall,
            # Orchestration tools
            "escalate": self._escalate,
            # Shell tools (sandboxed)
            "run_shell": self._run_shell,
            # Self-management procedure tools
            "run_procedure": self._run_procedure,
            "list_procedures": self._list_procedures,
            "get_procedure_status": self._get_procedure_status,
            "checkpoint_create": self._checkpoint_create,
            "checkpoint_restore": self._checkpoint_restore,
            "registry_lookup": self._registry_lookup,
            "registry_update": self._registry_update,
            "benchmark_run": self._benchmark_run,
            "benchmark_compare": self._benchmark_compare,
            "gate_run": self._gate_run,
            "log_append": self._log_append,
            "file_write_safe": self._file_write_safe,
        }

        # Add LLM primitives if available (wrapped for exploration tracking)
        if self.llm_primitives is not None:
            globals_dict["llm_call"] = self._tracked_llm_call
            globals_dict["llm_batch"] = self._tracked_llm_batch

        # Add tool registry functions if available
        if self.tool_registry is not None:
            globals_dict["TOOL"] = self._invoke_tool
            globals_dict["list_tools"] = self._list_tools

        # Add script registry functions if available
        if self.script_registry is not None:
            globals_dict["SCRIPT"] = self._invoke_script
            globals_dict["find_scripts"] = self._find_scripts

        return globals_dict

    # Allowed file paths for file operations (security)
    ALLOWED_FILE_PATHS = [
        "/mnt/raid0/llm/",
        "/tmp/",
    ]

    def _validate_file_path(self, path: str) -> tuple[bool, str | None]:
        """Validate that a file path is allowed.

        Args:
            path: Path to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        import os
        resolved = os.path.realpath(path)
        for allowed in self.ALLOWED_FILE_PATHS:
            if resolved.startswith(allowed):
                return True, None
        return False, f"Path not in allowed locations: {self.ALLOWED_FILE_PATHS}"

    def _peek(self, n: int = 500, file_path: str | None = None) -> str:
        """Return first n characters of context or file.

        Args:
            n: Number of characters to return (default 500).
            file_path: Optional file path to read from instead of context.

        Returns:
            First n characters of the context or file.
        """
        self._exploration_calls += 1

        if file_path is not None:
            # Read from file
            is_valid, error = self._validate_file_path(file_path)
            if not is_valid:
                return f"[ERROR: {error}]"
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    result = f.read(n)
                self._exploration_log.add_event("peek", {"n": n, "file_path": file_path}, result)
                return result
            except FileNotFoundError:
                return f"[ERROR: File not found: {file_path}]"
            except Exception as e:
                return f"[ERROR: {type(e).__name__}: {e}]"

        # Read from context
        result = self.context[:n]
        self._exploration_log.add_event("peek", {"n": n}, result)
        return result

    def _ocr_document(self, path: str) -> str:
        """Extract text and figure bounding boxes from a PDF document.

        Uses the LightOnOCR-2 server for fast, accurate OCR processing.
        The server processes pages in parallel for maximum throughput.

        Args:
            path: Absolute path to the PDF file.

        Returns:
            JSON string with extracted text and figure locations:
            {
                "full_text": "combined text from all pages",
                "pages": [{"page": 1, "text": "...", "figures": [...]}],
                "total_pages": N,
                "figures": [{"page": P, "id": N, "bbox": [x1,y1,x2,y2]}]
            }
        """
        self._exploration_calls += 1
        import json
        import requests

        # Validate path
        is_valid, error = self._validate_file_path(path)
        if not is_valid:
            return f"[ERROR: {error}]"

        if not path.lower().endswith(".pdf"):
            return "[ERROR: Only PDF files are supported. Use peek() for text files.]"

        try:
            with open(path, "rb") as f:
                files = {"file": (path.split("/")[-1], f, "application/pdf")}
                resp = requests.post(
                    "http://localhost:9001/v1/document/pdf",
                    files=files,
                    data={"max_pages": 100, "dpi": 200},
                    timeout=600,  # 10 min timeout for large documents
                )

            if resp.status_code != 200:
                return f"[ERROR: OCR server returned {resp.status_code}: {resp.text[:200]}]"

            data = resp.json()

            # Combine text from all pages
            full_text = ""
            all_figures = []

            for page_data in data.get("pages", []):
                page_num = page_data.get("page", 0)
                page_text = page_data.get("text", "")
                full_text += f"\n--- Page {page_num} ---\n{page_text}"

                # Collect figure bounding boxes
                for bbox in page_data.get("bboxes", []):
                    all_figures.append({
                        "page": page_num,
                        "id": bbox.get("id"),
                        "bbox": [bbox.get("x1"), bbox.get("y1"),
                                 bbox.get("x2"), bbox.get("y2")],
                    })

            result = {
                "full_text": full_text.strip()[:50000],  # Cap at 50K chars
                "total_pages": data.get("total_pages", 0),
                "figures": all_figures,
                "elapsed_sec": data.get("elapsed_sec", 0),
            }

            self._exploration_log.add_event("ocr_document", {"path": path}, result)
            return json.dumps(result, indent=2)

        except requests.exceptions.ConnectionError:
            return "[ERROR: LightOnOCR server not running on port 9001. Start with: python src/services/lightonocr_llama_server.py]"
        except FileNotFoundError:
            return f"[ERROR: File not found: {path}]"
        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _analyze_figure(self, image_path: str, prompt: str = "Describe this figure in detail") -> str:
        """Analyze an image or extracted figure with the vision model.

        Uses the vision pipeline for image analysis including:
        - General description via VL model
        - Face detection (optional)
        - OCR of text in images

        Args:
            image_path: Path to the image file (PNG, JPG, etc.)
            prompt: Analysis prompt/question about the image.

        Returns:
            Description of the image content.
        """
        self._exploration_calls += 1
        import json
        import requests

        # Validate path
        is_valid, error = self._validate_file_path(image_path)
        if not is_valid:
            return f"[ERROR: {error}]"

        try:
            resp = requests.post(
                "http://localhost:8000/v1/vision/analyze",
                json={
                    "image_path": image_path,
                    "vl_prompt": prompt,
                    "analyzers": ["vl_describe"],
                },
                timeout=120,
            )

            if resp.status_code != 200:
                return f"[ERROR: Vision API returned {resp.status_code}: {resp.text[:200]}]"

            data = resp.json()
            description = data.get("vl_description", data.get("description", ""))

            self._exploration_log.add_event(
                "analyze_figure",
                {"image_path": image_path, "prompt": prompt},
                description
            )
            return description

        except requests.exceptions.ConnectionError:
            return "[ERROR: Vision API not available on localhost:8000]"
        except FileNotFoundError:
            return f"[ERROR: File not found: {image_path}]"
        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _list_dir(self, path: str) -> str:
        """List contents of a directory.

        Args:
            path: Absolute path to the directory.

        Returns:
            JSON string with directory contents:
            {
                "path": "/path/to/dir",
                "files": [{"name": "file.txt", "type": "file", "size": 1234}],
                "total": N
            }
        """
        self._exploration_calls += 1
        import json
        import os

        # Validate path
        is_valid, error = self._validate_file_path(path)
        if not is_valid:
            return f"[ERROR: {error}]"

        try:
            entries = []
            for entry in os.scandir(path):
                entry_info = {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                }
                if entry.is_file():
                    try:
                        entry_info["size"] = entry.stat().st_size
                    except Exception:
                        entry_info["size"] = 0
                entries.append(entry_info)

            # Sort: directories first, then files
            entries.sort(key=lambda x: (x["type"] == "file", x["name"]))

            result = {
                "path": path,
                "files": entries[:100],  # Cap at 100 entries
                "total": len(entries),
            }

            self._exploration_log.add_event("list_dir", {"path": path}, result)
            return json.dumps(result, indent=2)

        except FileNotFoundError:
            return f"[ERROR: Directory not found: {path}]"
        except NotADirectoryError:
            return f"[ERROR: Not a directory: {path}]"
        except PermissionError:
            return f"[ERROR: Permission denied: {path}]"
        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _file_info(self, path: str) -> str:
        """Get metadata about a file.

        Args:
            path: Absolute path to the file.

        Returns:
            JSON string with file metadata:
            {
                "path": "/path/to/file",
                "exists": true,
                "type": "file"|"dir"|"symlink",
                "size": 1234,
                "modified": "2024-01-15T10:30:00",
                "extension": ".pdf"
            }
        """
        self._exploration_calls += 1
        import json
        import os
        from datetime import datetime

        # Validate path
        is_valid, error = self._validate_file_path(path)
        if not is_valid:
            return f"[ERROR: {error}]"

        try:
            stat_info = os.stat(path)
            is_dir = os.path.isdir(path)
            is_link = os.path.islink(path)

            result = {
                "path": path,
                "exists": True,
                "type": "symlink" if is_link else ("dir" if is_dir else "file"),
                "size": stat_info.st_size,
                "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "extension": os.path.splitext(path)[1] if not is_dir else None,
            }

            self._exploration_log.add_event("file_info", {"path": path}, result)
            return json.dumps(result, indent=2)

        except FileNotFoundError:
            return json.dumps({"path": path, "exists": False})
        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _web_fetch(self, url: str, max_chars: int = 10000) -> str:
        """Fetch content from a URL.

        Fetches web content and converts HTML to plain text.
        Only allows http/https URLs.

        Args:
            url: URL to fetch (must be http or https).
            max_chars: Maximum characters to return (default 10000).

        Returns:
            Plain text content from the URL, truncated to max_chars.
        """
        self._exploration_calls += 1
        import requests

        # Validate URL
        if not url.startswith(("http://", "https://")):
            return "[ERROR: Only http/https URLs are allowed]"

        try:
            resp = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (compatible; OrchestratorBot/1.0)"},
            )
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")

            # Handle HTML - extract text
            if "text/html" in content_type:
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # Remove script and style elements
                    for elem in soup(["script", "style", "nav", "footer"]):
                        elem.decompose()
                    text = soup.get_text(separator="\n", strip=True)
                except ImportError:
                    # Fallback: basic tag stripping
                    import re
                    text = re.sub(r"<[^>]+>", "", resp.text)
            else:
                text = resp.text

            result = text[:max_chars]
            if len(text) > max_chars:
                result += f"\n[... truncated at {max_chars} chars, total: {len(text)}]"

            self._exploration_log.add_event("web_fetch", {"url": url}, result)
            return result

        except requests.exceptions.Timeout:
            return "[ERROR: Request timed out after 30s]"
        except requests.exceptions.RequestException as e:
            return f"[ERROR: Request failed: {e}]"
        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _extract_figure(
        self,
        pdf_path: str,
        page: int,
        bbox: list[int],
        output_path: str | None = None,
    ) -> str:
        """Extract a figure from a PDF given its bounding box.

        Crops a region from a PDF page and saves it as an image.
        Bounding box coordinates are in 0-1000 normalized range.

        Args:
            pdf_path: Path to the PDF file.
            page: Page number (1-indexed).
            bbox: Bounding box as [x1, y1, x2, y2] in 0-1000 range.
            output_path: Optional output path. If None, saves to /tmp.

        Returns:
            Path to the extracted figure image.
        """
        self._exploration_calls += 1
        import os
        import tempfile

        # Validate paths
        is_valid, error = self._validate_file_path(pdf_path)
        if not is_valid:
            return f"[ERROR: {error}]"

        if output_path:
            is_valid, error = self._validate_file_path(output_path)
            if not is_valid:
                return f"[ERROR: {error}]"

        try:
            import pypdfium2 as pdfium
            from PIL import Image
        except ImportError:
            return "[ERROR: pypdfium2 or Pillow not installed]"

        try:
            pdf = pdfium.PdfDocument(pdf_path)
            if page < 1 or page > len(pdf):
                return f"[ERROR: Page {page} out of range (1-{len(pdf)})]"

            # Render page at high DPI
            pdf_page = pdf[page - 1]
            scale = 300 / 72  # 300 DPI
            bitmap = pdf_page.render(scale=scale)
            img = bitmap.to_pil()

            # Convert normalized coords (0-1000) to pixel coords
            width, height = img.size
            x1 = int(bbox[0] * width / 1000)
            y1 = int(bbox[1] * height / 1000)
            x2 = int(bbox[2] * width / 1000)
            y2 = int(bbox[3] * height / 1000)

            # Crop the figure
            cropped = img.crop((x1, y1, x2, y2))

            # Save to output path or temp
            if output_path:
                save_path = output_path
            else:
                fd, save_path = tempfile.mkstemp(suffix=".png", dir="/mnt/raid0/llm/tmp")
                os.close(fd)

            cropped.save(save_path, "PNG")

            self._exploration_log.add_event(
                "extract_figure",
                {"pdf_path": pdf_path, "page": page, "bbox": bbox},
                save_path
            )
            return save_path

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _recall(self, query: str, limit: int = 5) -> str:
        """Search episodic memory for similar past tasks.

        Retrieves relevant past task executions that may help with the current task.
        Returns task descriptions, outcomes, and what strategies worked.

        Args:
            query: Natural language description of what you're looking for.
            limit: Maximum number of results to return (default 5).

        Returns:
            JSON string with similar past tasks:
            {
                "results": [
                    {
                        "task": "description",
                        "outcome": "success"|"failure",
                        "strategy": "what approach was used",
                        "similarity": 0.85
                    }
                ]
            }
        """
        self._exploration_calls += 1
        import json

        # Check if episodic memory is available
        try:
            from orchestration.repl_memory.episodic_store import EpisodicStore
            from orchestration.repl_memory.embedder import TaskEmbedder
        except ImportError:
            return json.dumps({"results": [], "error": "Episodic memory not available"})

        try:
            store = EpisodicStore()
            embedder = TaskEmbedder()

            # Embed the query
            query_embedding = embedder.embed(query)

            # Search for similar tasks
            memories = store.search_similar(
                embedding=query_embedding,
                limit=limit,
                min_similarity=0.3,
            )

            results = []
            for mem in memories:
                results.append({
                    "task": mem.task_description[:200] if mem.task_description else "",
                    "outcome": mem.outcome,
                    "strategy": mem.context.get("exploration_strategy", {}).get("strategy_type", "unknown")
                              if mem.context else "unknown",
                    "similarity": round(mem.similarity, 3) if hasattr(mem, "similarity") else 0.0,
                })

            result = {"results": results}
            self._exploration_log.add_event("recall", {"query": query}, result)
            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"results": [], "error": str(e)})

    def _escalate(self, reason: str) -> str:
        """Request escalation to a higher-tier model.

        Call this when the current task is too complex or requires
        capabilities beyond the current role. The orchestrator will
        route to a more capable model.

        Args:
            reason: Why escalation is needed (be specific).

        Returns:
            Acknowledgment message. The actual escalation happens
            after this REPL turn completes.
        """
        # Store escalation request in artifacts for the orchestrator to handle
        self.artifacts["_escalation_requested"] = True
        self.artifacts["_escalation_reason"] = reason

        return f"[ESCALATION REQUESTED: {reason}] - The orchestrator will route to a higher-tier model after this turn."

    def _run_shell(self, cmd: str, timeout: int = 30) -> str:
        """Run a sandboxed shell command (read-only operations only).

        Only allows safe, non-destructive commands:
        - ls, find, wc, du, file, head, tail, cat (read-only)
        - git status, git log, git diff (read-only git)
        - python -c (with restrictions)

        Blocked: rm, mv, cp, chmod, chown, dd, mkfs, and all write operations.

        Args:
            cmd: Shell command to execute.
            timeout: Maximum execution time in seconds (default 30, max 120).

        Returns:
            Command output (stdout + stderr combined).
        """
        self._exploration_calls += 1
        import subprocess
        import shlex

        # Parse command
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            return f"[ERROR: Invalid command syntax: {e}]"

        if not parts:
            return "[ERROR: Empty command]"

        # Allowlist of safe commands
        SAFE_COMMANDS = {
            "ls", "find", "wc", "du", "file", "head", "tail", "cat",
            "grep", "awk", "sed", "sort", "uniq", "tr", "cut",
            "git", "pwd", "whoami", "date", "echo", "printf",
            "python", "python3",
        }

        # Commands that are always blocked
        BLOCKED_COMMANDS = {
            "rm", "mv", "cp", "chmod", "chown", "chgrp", "dd", "mkfs",
            "mount", "umount", "kill", "pkill", "killall",
            "sudo", "su", "bash", "sh", "zsh", "csh",
            "wget", "curl",  # Blocked to prevent downloads
            "nc", "netcat", "ncat",  # Network tools
        }

        base_cmd = parts[0].split("/")[-1]  # Handle /usr/bin/ls -> ls

        if base_cmd in BLOCKED_COMMANDS:
            return f"[ERROR: Command '{base_cmd}' is blocked for security]"

        if base_cmd not in SAFE_COMMANDS:
            return f"[ERROR: Command '{base_cmd}' not in allowlist: {sorted(SAFE_COMMANDS)}]"

        # Additional git restrictions
        if base_cmd == "git":
            if len(parts) > 1:
                git_subcmd = parts[1]
                safe_git = {"status", "log", "diff", "branch", "show", "ls-files", "rev-parse"}
                if git_subcmd not in safe_git:
                    return f"[ERROR: git {git_subcmd} not allowed. Safe: {sorted(safe_git)}]"

        # Timeout cap
        timeout = min(timeout, 120)

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="/mnt/raid0/llm/claude",  # Always run from project root
            )

            output = result.stdout
            if result.stderr:
                output += "\n[STDERR]\n" + result.stderr

            # Cap output
            if len(output) > 8000:
                output = output[:8000] + f"\n[... truncated at 8000 chars]"

            self._exploration_log.add_event("run_shell", {"cmd": cmd}, output)
            return output

        except subprocess.TimeoutExpired:
            return f"[ERROR: Command timed out after {timeout}s]"
        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    # =========================================================================
    # Self-Management Procedure Tools (~350 tokens per operation vs 3000-5000)
    # =========================================================================

    def _run_procedure(self, procedure_id: str, **kwargs) -> str:
        """Execute a self-management procedure by ID.

        Procedures are deterministic, pre-defined sequences of operations for
        common self-management tasks like benchmarking, registry updates, etc.

        Args:
            procedure_id: ID of the procedure to execute (e.g., 'benchmark_model').
            **kwargs: Input parameters for the procedure.

        Returns:
            JSON string with execution result including success status and outputs.
        """
        self._exploration_calls += 1
        import json

        try:
            from orchestration.procedure_registry import ProcedureRegistry

            registry = ProcedureRegistry()
            result = registry.execute(procedure_id, role=self.role, **kwargs)

            output = {
                "success": result.success,
                "procedure_id": result.procedure_id,
                "error": result.error,
                "elapsed_seconds": round(result.elapsed_seconds, 2),
                "outputs": result.outputs,
                "steps_completed": sum(1 for s in result.step_results if s.success),
                "steps_total": len(result.step_results),
            }

            self._exploration_log.add_event("run_procedure", {"procedure_id": procedure_id, **kwargs}, output)
            return json.dumps(output, indent=2)

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _list_procedures(self, category: str | None = None) -> str:
        """List available self-management procedures.

        Args:
            category: Optional category filter ('benchmark', 'registry', 'gate', etc.).

        Returns:
            JSON string with list of available procedures.
        """
        self._exploration_calls += 1
        import json

        try:
            from orchestration.procedure_registry import ProcedureRegistry

            registry = ProcedureRegistry()
            procedures = registry.list_procedures(category=category, role=self.role)

            self._exploration_log.add_event("list_procedures", {"category": category}, procedures)
            return json.dumps(procedures, indent=2)

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _get_procedure_status(self, procedure_id: str) -> str:
        """Get the most recent execution status of a procedure.

        Args:
            procedure_id: ID of the procedure.

        Returns:
            JSON string with last execution status or 'never_run'.
        """
        self._exploration_calls += 1
        import json
        import os
        from pathlib import Path

        try:
            state_dir = Path("/mnt/raid0/llm/claude/orchestration/procedures/state")
            if not state_dir.exists():
                return json.dumps({"status": "never_run", "procedure_id": procedure_id})

            # Find most recent state file for this procedure
            state_files = sorted(state_dir.glob(f"{procedure_id}_*.json"), reverse=True)
            if not state_files:
                return json.dumps({"status": "never_run", "procedure_id": procedure_id})

            with open(state_files[0], encoding="utf-8") as f:
                state = json.load(f)

            return json.dumps(state, indent=2)

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _checkpoint_create(self, name: str) -> str:
        """Create a checkpoint of current system state.

        Args:
            name: Descriptive name for the checkpoint.

        Returns:
            Checkpoint ID that can be used for restore.
        """
        self._exploration_calls += 1
        return self._run_procedure("checkpoint_create", name=name)

    def _checkpoint_restore(self, checkpoint_id: str) -> str:
        """Restore system state from a checkpoint.

        Args:
            checkpoint_id: ID returned from checkpoint_create.

        Returns:
            Restoration status.
        """
        self._exploration_calls += 1
        import json
        from pathlib import Path

        try:
            checkpoint_dir = Path("/mnt/raid0/llm/claude/orchestration/checkpoints")
            checkpoint_path = checkpoint_dir / f"{checkpoint_id}.json"

            if not checkpoint_path.exists():
                return f"[ERROR: Checkpoint not found: {checkpoint_id}]"

            with open(checkpoint_path, encoding="utf-8") as f:
                checkpoint = json.load(f)

            return json.dumps({
                "restored": True,
                "checkpoint_id": checkpoint_id,
                "created_at": checkpoint.get("created_at"),
            }, indent=2)

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _prepare_patch(self, files: list[str], description: str) -> str:
        """Generate unified diff for owner review.

        Creates a patch file in orchestration/patches/pending/ for approval.

        Args:
            files: List of file paths to include in the patch.
            description: Short description of the changes.

        Returns:
            Path to the generated patch file.
        """
        self._exploration_calls += 1
        import subprocess
        from datetime import datetime
        from pathlib import Path

        try:
            patches_dir = Path("/mnt/raid0/llm/claude/orchestration/patches/pending")
            patches_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_desc = description.replace(" ", "_")[:30]
            patch_name = f"{timestamp}_{safe_desc}.patch"
            patch_path = patches_dir / patch_name

            # Generate unified diff
            result = subprocess.run(
                ["git", "diff", "--"] + files,
                capture_output=True,
                text=True,
                cwd="/mnt/raid0/llm/claude"
            )

            if not result.stdout.strip():
                return "[INFO: No changes to create patch from]"

            # Write patch with metadata header
            with open(patch_path, "w", encoding="utf-8") as f:
                f.write(f"# Patch: {description}\n")
                f.write(f"# Created: {datetime.now().isoformat()}\n")
                f.write(f"# Files: {', '.join(files)}\n")
                f.write(f"# Status: PENDING APPROVAL\n")
                f.write("#\n")
                f.write(result.stdout)

            return f"Patch created: {patch_path}\nReview with: cat {patch_path}"

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _list_patches(self, status: str = "pending") -> str:
        """List patches by status.

        Args:
            status: One of 'pending', 'approved', 'rejected', or 'all'.

        Returns:
            List of patches with metadata.
        """
        self._exploration_calls += 1
        import json
        from pathlib import Path

        try:
            patches_base = Path("/mnt/raid0/llm/claude/orchestration/patches")
            results = []

            statuses = ["pending", "approved", "rejected"] if status == "all" else [status]

            for s in statuses:
                status_dir = patches_base / s
                if not status_dir.exists():
                    continue

                for patch_file in sorted(status_dir.glob("*.patch")):
                    # Read first few lines for metadata
                    with open(patch_file, encoding="utf-8") as f:
                        lines = f.readlines()[:5]

                    metadata = {"file": str(patch_file), "status": s}
                    for line in lines:
                        if line.startswith("# Patch:"):
                            metadata["description"] = line.split(":", 1)[1].strip()
                        elif line.startswith("# Created:"):
                            metadata["created"] = line.split(":", 1)[1].strip()
                        elif line.startswith("# Files:"):
                            metadata["files"] = line.split(":", 1)[1].strip()

                    results.append(metadata)

            if not results:
                return f"[INFO: No {status} patches found]"

            return json.dumps(results, indent=2)

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _apply_approved_patch(self, patch_name: str) -> str:
        """Apply a patch after owner approval.

        IMPORTANT: Only applies patches from the 'pending' directory.
        Moves to 'approved' after successful application.

        Args:
            patch_name: Name of the patch file (e.g., '20260124_update_registry.patch').

        Returns:
            Application status.
        """
        self._exploration_calls += 1
        import shutil
        import subprocess
        from datetime import datetime
        from pathlib import Path

        try:
            patches_base = Path("/mnt/raid0/llm/claude/orchestration/patches")
            pending_path = patches_base / "pending" / patch_name
            approved_path = patches_base / "approved" / patch_name

            if not pending_path.exists():
                return f"[ERROR: Patch not found in pending: {patch_name}]"

            # Dry run first
            result = subprocess.run(
                ["git", "apply", "--check", str(pending_path)],
                capture_output=True,
                text=True,
                cwd="/mnt/raid0/llm/claude"
            )

            if result.returncode != 0:
                return f"[ERROR: Patch cannot be applied cleanly: {result.stderr}]"

            # Apply the patch
            result = subprocess.run(
                ["git", "apply", str(pending_path)],
                capture_output=True,
                text=True,
                cwd="/mnt/raid0/llm/claude"
            )

            if result.returncode != 0:
                return f"[ERROR: Failed to apply patch: {result.stderr}]"

            # Move to approved
            shutil.move(str(pending_path), str(approved_path))

            # Add approval timestamp
            with open(approved_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Applied: {datetime.now().isoformat()}\n")

            return f"Patch applied successfully and moved to approved: {approved_path}"

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _reject_patch(self, patch_name: str, reason: str) -> str:
        """Reject a pending patch with reason.

        Moves patch to 'rejected' directory with rejection metadata.

        Args:
            patch_name: Name of the patch file.
            reason: Why the patch was rejected.

        Returns:
            Rejection status.
        """
        self._exploration_calls += 1
        import shutil
        from datetime import datetime
        from pathlib import Path

        try:
            patches_base = Path("/mnt/raid0/llm/claude/orchestration/patches")
            pending_path = patches_base / "pending" / patch_name
            rejected_path = patches_base / "rejected" / patch_name

            if not pending_path.exists():
                return f"[ERROR: Patch not found in pending: {patch_name}]"

            # Move to rejected
            shutil.move(str(pending_path), str(rejected_path))

            # Add rejection metadata
            with open(rejected_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Rejected: {datetime.now().isoformat()}\n")
                f.write(f"# Reason: {reason}\n")

            return f"Patch rejected and moved to: {rejected_path}"

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _registry_lookup(self, key_path: str) -> str:
        """Look up a value in the model registry.

        Args:
            key_path: Dot-separated path (e.g., 'roles.coder_primary.model.name').

        Returns:
            The value at that path, or error if not found.
        """
        self._exploration_calls += 1
        import json

        try:
            # Try yaml first, fall back to json parsing
            registry_path = "/mnt/raid0/llm/claude/orchestration/model_registry.yaml"
            try:
                import yaml
                with open(registry_path, encoding="utf-8") as f:
                    registry = yaml.safe_load(f)
            except ImportError:
                # Fallback: can't parse yaml without the module
                return "[ERROR: YAML support not available for registry lookup]"

            # Navigate to key
            keys = key_path.split(".")
            obj = registry
            for key in keys:
                if isinstance(obj, dict) and key in obj:
                    obj = obj[key]
                else:
                    return f"[ERROR: Key not found: {key_path}]"

            result = json.dumps(obj, indent=2, default=str)
            self._exploration_log.add_event("registry_lookup", {"key_path": key_path}, result)
            return result

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _registry_update(self, key_path: str, value: Any) -> str:
        """Update a value in the model registry.

        This is a privileged operation that requires appropriate role.

        Args:
            key_path: Dot-separated path to update.
            value: New value to set.

        Returns:
            Success/failure status.
        """
        self._exploration_calls += 1
        return self._run_procedure("update_registry", key_path=key_path, value=value)

    def _benchmark_run(
        self,
        model_path: str,
        suite: str = "quick",
        n_tokens: int = 256,
    ) -> str:
        """Run a benchmark on a model.

        Args:
            model_path: Path to the GGUF model file.
            suite: Benchmark suite ('quick', 'thinking', 'coder', 'general', 'all').
            n_tokens: Number of tokens to generate per prompt.

        Returns:
            JSON with benchmark results including tokens/second.
        """
        self._exploration_calls += 1
        return self._run_procedure(
            "benchmark_model",
            model_path=model_path,
            benchmark_suite=suite,
            n_tokens=n_tokens,
        )

    def _benchmark_compare(
        self,
        model_a: str,
        model_b: str,
    ) -> str:
        """Compare benchmark results between two models.

        Args:
            model_a: First model path or name.
            model_b: Second model path or name.

        Returns:
            JSON with comparison of benchmark results.
        """
        self._exploration_calls += 1
        import json
        from pathlib import Path

        try:
            results_dir = Path("/mnt/raid0/llm/claude/benchmarks/results/runs")

            # Find results for both models
            results = {}
            for model in [model_a, model_b]:
                model_name = Path(model).stem if "/" in model else model
                result_files = list(results_dir.glob(f"*{model_name}*.json"))
                if result_files:
                    with open(sorted(result_files)[-1], encoding="utf-8") as f:
                        results[model_name] = json.load(f)
                else:
                    results[model_name] = {"error": "No benchmark results found"}

            comparison = {
                "models": results,
                "comparison": {
                    "note": "Compare 'tps' values for throughput"
                }
            }

            self._exploration_log.add_event("benchmark_compare", {"model_a": model_a, "model_b": model_b}, comparison)
            return json.dumps(comparison, indent=2)

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _gate_run(
        self,
        gates: list[str] | None = None,
        path: str = "src/",
        fix: bool = False,
    ) -> str:
        """Run verification gates (lint, format, tests).

        Args:
            gates: List of gates to run (default: ['lint', 'format']).
            path: Path to check.
            fix: Whether to auto-fix issues.

        Returns:
            JSON with gate results.
        """
        self._exploration_calls += 1
        return self._run_procedure(
            "gate_runner",
            gates=gates or ["lint", "format"],
            path=path,
            fix=fix,
        )

    def _log_append(self, log_name: str, message: str) -> str:
        """Append a message to a log file.

        Args:
            log_name: Name of the log file (without path).
            message: Message to append.

        Returns:
            Confirmation message.
        """
        self._exploration_calls += 1
        from datetime import datetime

        try:
            log_path = f"/mnt/raid0/llm/claude/logs/{log_name}"

            # Validate path
            is_valid, error = self._validate_file_path(log_path)
            if not is_valid:
                return f"[ERROR: {error}]"

            timestamp = datetime.now().isoformat()
            log_entry = f"[{timestamp}] {message}\n"

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)

            return f"Appended to {log_name}"

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _file_write_safe(
        self,
        path: str,
        content: str,
        backup: bool = True,
    ) -> str:
        """Safely write content to a file with optional backup.

        Only allows writing to /mnt/raid0/ paths.

        Args:
            path: Absolute path to write to.
            content: Content to write.
            backup: Whether to create backup of existing file.

        Returns:
            Success/failure status.
        """
        self._exploration_calls += 1
        import os
        from datetime import datetime
        from pathlib import Path as P

        try:
            # Validate path
            is_valid, error = self._validate_file_path(path)
            if not is_valid:
                return f"[ERROR: {error}]"

            # Create backup if file exists and backup requested
            if backup and os.path.exists(path):
                backup_path = f"{path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(path, "r", encoding="utf-8") as src:
                    with open(backup_path, "w", encoding="utf-8") as dst:
                        dst.write(src.read())

            # Ensure parent directory exists
            P(path).parent.mkdir(parents=True, exist_ok=True)

            # Write content
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            self._exploration_log.add_event("file_write_safe", {"path": path, "size": len(content)}, "success")
            return f"Wrote {len(content)} bytes to {path}"

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _grep(self, pattern: str, file_path: str | None = None) -> list[str]:
        """Search context or file with regex and return matching lines.

        Args:
            pattern: Regular expression pattern to search for.
            file_path: Optional file path to search instead of context.

        Returns:
            List of lines containing matches (capped at max_grep_results).
        """
        self._exploration_calls += 1
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return [f"[REGEX ERROR: {e}]"]

        # Determine source text
        if file_path is not None:
            is_valid, error = self._validate_file_path(file_path)
            if not is_valid:
                return [f"[ERROR: {error}]"]
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    source_text = f.read()
            except FileNotFoundError:
                return [f"[ERROR: File not found: {file_path}]"]
            except Exception as e:
                return [f"[ERROR: {type(e).__name__}: {e}]"]
        else:
            source_text = self.context

        matches = []
        for line in source_text.split("\n"):
            if regex.search(line):
                matches.append(line)
                if len(matches) >= self.config.max_grep_results:
                    matches.append(f"[... truncated at {self.config.max_grep_results} results]")
                    break

        self._exploration_log.add_event("grep", {"pattern": pattern, "file_path": file_path}, matches)
        return matches

    def _tracked_llm_call(self, *args, **kwargs) -> str:
        """Wrapper for llm_call that tracks exploration.

        Args:
            *args: Positional arguments for llm_call.
            **kwargs: Keyword arguments for llm_call.

        Returns:
            llm_call result.
        """
        self._exploration_calls += 1
        result = self.llm_primitives.llm_call(*args, **kwargs)
        self._exploration_log.add_event("llm_call", {"args": args, "kwargs": kwargs}, result)
        return result

    def _tracked_llm_batch(self, *args, **kwargs) -> list[str]:
        """Wrapper for llm_batch that tracks exploration.

        Args:
            *args: Positional arguments for llm_batch.
            **kwargs: Keyword arguments for llm_batch.

        Returns:
            llm_batch result.
        """
        self._exploration_calls += 1
        result = self.llm_primitives.llm_batch(*args, **kwargs)
        self._exploration_log.add_event("llm_batch", {"args": args, "kwargs": kwargs}, result)
        return result

    def _final(self, answer: str) -> None:
        """Signal completion with final answer.

        Args:
            answer: The final answer to return.

        Raises:
            FinalSignal: Raised to terminate execution (after validation).
            ValueError: If exploration requirement not met.
        """
        # Check forced exploration validation
        if self.config.require_exploration_before_final:
            if self._exploration_calls < self.config.min_exploration_calls:
                raise ValueError(
                    f"Premature FINAL: Must call at least {self.config.min_exploration_calls} "
                    f"exploration function(s) (peek, grep, llm_call, llm_batch) before FINAL(). "
                    f"Current exploration calls: {self._exploration_calls}. "
                    "Use peek() or grep() to examine the context first."
                )
        raise FinalSignal(str(answer))

    def _final_var(self, var_name: str) -> None:
        """Signal completion, returning contents of a variable.

        Args:
            var_name: Name of variable in artifacts dict to return.

        Raises:
            FinalSignal: Raised to terminate execution (after validation).
            KeyError: If variable not found in artifacts.
            ValueError: If exploration requirement not met.
        """
        # Check forced exploration validation
        if self.config.require_exploration_before_final:
            if self._exploration_calls < self.config.min_exploration_calls:
                raise ValueError(
                    f"Premature FINAL_VAR: Must call at least {self.config.min_exploration_calls} "
                    f"exploration function(s) (peek, grep, llm_call, llm_batch) before FINAL_VAR(). "
                    f"Current exploration calls: {self._exploration_calls}. "
                    "Use peek() or grep() to examine the context first."
                )
        if var_name not in self.artifacts:
            raise KeyError(f"Variable '{var_name}' not found in artifacts")
        raise FinalSignal(str(self.artifacts[var_name]))

    def _invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """Invoke a registered tool.

        Args:
            tool_name: Name of the tool to invoke.
            **kwargs: Tool arguments.

        Returns:
            Tool result.

        Raises:
            ValueError: If tool doesn't exist.
            PermissionError: If role cannot use this tool.
        """
        if self.tool_registry is None:
            raise RuntimeError("No tool registry configured")

        return self.tool_registry.invoke(tool_name, self.role, **kwargs)

    def _list_tools(self) -> list[dict[str, Any]]:
        """List available tools for the current role.

        Returns:
            List of tool info dicts.
        """
        if self.tool_registry is None:
            return []

        return self.tool_registry.list_tools(role=self.role)

    def _invoke_script(self, script_id: str, **kwargs) -> Any:
        """Invoke a prepared script by ID.

        Args:
            script_id: Script identifier.
            **kwargs: Script arguments.

        Returns:
            Script result.

        Raises:
            ValueError: If script doesn't exist.
        """
        if self.script_registry is None:
            raise RuntimeError("No script registry configured")

        # Pass sandbox globals for code execution
        return self.script_registry.invoke(
            script_id,
            sandbox_globals=self._globals,
            **kwargs,
        )

    def _find_scripts(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find scripts matching a natural language query.

        Args:
            query: Search query (e.g., "fetch python documentation").
            limit: Maximum results to return.

        Returns:
            List of matching script info dicts.
        """
        if self.script_registry is None:
            return []

        matches = self.script_registry.find_scripts(query, limit=limit)
        return [
            {
                "id": m.script.id,
                "description": m.script.description,
                "score": round(m.score, 2),
                "matched_on": m.matched_on,
            }
            for m in matches
        ]

    def _validate_code(self, code: str) -> None:
        """Validate code for dangerous patterns using AST analysis.

        Uses AST-based validation which is more robust than regex because it
        analyzes the parsed syntax tree, making it immune to string tricks like:
            getattr(__builtins__, '__im' + 'port__')('os')

        Args:
            code: Python code to validate.

        Raises:
            REPLSecurityError: If dangerous patterns detected or syntax is invalid.
        """
        # First, try to parse the code to get an AST
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            # Let the actual execution handle syntax errors for better messages
            return

        # Run AST-based security analysis
        visitor = ASTSecurityVisitor()
        visitor.visit(tree)

        if visitor.violations:
            # Report first violation (avoids info disclosure about all checks)
            raise REPLSecurityError(
                f"Dangerous operation not allowed: {visitor.violations[0]}"
            )

    def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in the sandboxed environment.

        Args:
            code: Python code to execute.

        Returns:
            ExecutionResult with output, is_final flag, and optional error.
        """
        import time

        self._execution_count += 1
        start_time = time.perf_counter()

        # Validate code for dangerous patterns
        try:
            self._validate_code(code)
        except REPLSecurityError as e:
            return ExecutionResult(
                output="",
                is_final=False,
                error=str(e),
                elapsed_seconds=time.perf_counter() - start_time,
            )

        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise REPLTimeout(f"Execution timed out after {self.config.timeout_seconds}s")

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Set timeout (Unix only)
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.config.timeout_seconds)

            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, self._globals)

                output = stdout_capture.getvalue()
                if stderr_capture.getvalue():
                    output += "\n[STDERR]\n" + stderr_capture.getvalue()

                # Cap output
                if len(output) > self.config.output_cap:
                    output = output[: self.config.output_cap] + f"\n[... truncated at {self.config.output_cap} chars]"

                return ExecutionResult(
                    output=output,
                    is_final=False,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except FinalSignal as e:
                output = stdout_capture.getvalue()
                return ExecutionResult(
                    output=output,
                    is_final=True,
                    final_answer=e.answer,
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except SyntaxError as e:
                return ExecutionResult(
                    output="",
                    is_final=False,
                    error=f"SyntaxError: {e}",
                    elapsed_seconds=time.perf_counter() - start_time,
                )

            except Exception as e:
                return ExecutionResult(
                    output=stdout_capture.getvalue(),
                    is_final=False,
                    error=f"{type(e).__name__}: {e}",
                    elapsed_seconds=time.perf_counter() - start_time,
                )

        finally:
            # Clear timeout
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def get_state(self) -> str:
        """Get a summary of current REPL state for the Root LM.

        Returns:
            String describing available variables and artifacts.
        """
        state_lines = [
            f"context: str ({len(self.context)} chars)",
            f"artifacts: {list(self.artifacts.keys()) if self.artifacts else '{}'}",
        ]

        # Show artifact previews
        for key, value in self.artifacts.items():
            preview = str(value)[:100]
            if len(str(value)) > 100:
                preview += "..."
            state_lines.append(f"  artifacts['{key}']: {preview}")

        return "\n".join(state_lines)

    def get_exploration_log(self) -> ExplorationLog:
        """Get the detailed exploration log.

        Returns:
            ExplorationLog containing all exploration events.
        """
        return self._exploration_log

    def get_exploration_strategy(self) -> dict[str, Any]:
        """Get a summary of the exploration strategy used.

        Returns:
            Dictionary with strategy summary including event counts and type.
        """
        return self._exploration_log.get_strategy_summary()

    def log_exploration_completed(
        self,
        success: bool,
        result: str = "",
    ) -> dict[str, Any]:
        """Log exploration completion to ProgressLogger.

        This should be called when the REPL task is complete (after FINAL).
        Logs the exploration strategy and token efficiency for Q-learning.

        Args:
            success: Whether the task completed successfully.
            result: The final result (used for token efficiency calculation).

        Returns:
            Dictionary with the logged exploration data.
        """
        strategy = self.get_exploration_strategy()
        result_tokens = len(result) // 4  # Rough token estimate
        efficiency = self._exploration_log.get_token_efficiency(result_tokens)

        exploration_data = {
            "strategy": strategy,
            "efficiency": efficiency,
            "success": success,
        }

        # Log to ProgressLogger if available
        if self.progress_logger is not None:
            query_preview = self.context[:100] if self.context else ""
            self.progress_logger.log_exploration(
                task_id=self.task_id,
                query=query_preview,
                strategy_used=strategy.get("strategy_type", "unknown"),
                tokens_spent=strategy.get("total_tokens", 0),
                success=success,
            )

        return exploration_data

    def suggest_exploration(
        self,
        task_description: str,
        retriever: TwoPhaseRetriever | None = None,
    ) -> list[str]:
        """Suggest exploration strategies based on similar past tasks.

        Uses episodic memory (if available) to suggest effective exploration
        strategies based on successful past tasks.

        Args:
            task_description: Description of the current task.
            retriever: TwoPhaseRetriever from orchestration.repl_memory (optional).

        Returns:
            List of suggested exploration function calls as strings.
        """
        suggestions = []
        episodic_suggestions = []

        # If retriever available, query for similar successful exploration tasks
        if retriever is not None:
            try:
                context_preview = self.context[:500] if self.context else ""
                results = retriever.retrieve_for_exploration(
                    query=task_description,
                    context_preview=context_preview,
                )

                if results:
                    # Extract suggestions from successful similar tasks
                    for r in results[:3]:
                        # Only use high-quality memories (Q > 0.6, successful)
                        if r.q_value < 0.6 or r.memory.outcome != "success":
                            continue

                        context = r.memory.context or {}
                        strategy = context.get("exploration_strategy", {})
                        function_counts = strategy.get("function_counts", {})
                        strategy_type = strategy.get("strategy_type", "")

                        # Generate specific suggestions based on what worked
                        if function_counts.get("grep", 0) > 0:
                            # Extract grep patterns that worked
                            episodic_suggestions.append(
                                f"grep('pattern')  # Similar task (q={r.q_value:.2f}) used grep"
                            )
                        if function_counts.get("llm_call", 0) > 0:
                            episodic_suggestions.append(
                                f"llm_call('summarize key points')  # Similar task delegated effectively"
                            )
                        if strategy_type == "scan" and function_counts.get("peek", 0) > 0:
                            peek_count = function_counts["peek"]
                            episodic_suggestions.append(
                                f"# Scan strategy worked: {peek_count} peek() calls"
                            )

            except Exception:
                pass  # Silently ignore retrieval errors

        # Default suggestions based on context characteristics
        context_len = len(self.context)

        if context_len < 500:
            suggestions.append("peek(500)  # Context is short, read it all")
        elif context_len < 2000:
            suggestions.append("peek(1000)  # Scan the beginning")
        else:
            suggestions.append("peek(500)  # Preview context")
            suggestions.append("grep('keyword')  # Search for specific patterns")

        # Prepend episodic suggestions (learned patterns first)
        return episodic_suggestions + suggestions

    def reset(self) -> None:
        """Reset the REPL state (clear artifacts, keep context)."""
        self.artifacts.clear()
        self._final_answer = None
        self._execution_count = 0
        self._exploration_calls = 0
        self._exploration_log = ExplorationLog()  # Reset exploration log
        self._globals = self._build_globals()


def create_repl_environment(
    context: str,
    artifacts: dict[str, Any] | None = None,
    config: REPLConfig | None = None,
    llm_primitives: Any | None = None,
    tool_registry: Any | None = None,
    script_registry: Any | None = None,
    role: str | None = None,
    use_restricted_python: bool | None = None,
    progress_logger: ProgressLogger | None = None,
    task_id: str | None = None,
) -> REPLEnvironment:
    """Factory function to create REPL environment with optional RestrictedPython.

    When use_restricted_python is True (or None with feature flag enabled),
    creates a REPLEnvironment that delegates to RestrictedExecutor for execution.

    Args:
        context: The full input context.
        artifacts: Optional pre-existing artifacts.
        config: Optional REPL configuration.
        llm_primitives: Optional LLMPrimitives for llm_call/llm_batch.
        tool_registry: Optional ToolRegistry for TOOL() invocations.
        script_registry: Optional ScriptRegistry for SCRIPT() invocations.
        role: Role name for permission checking.
        use_restricted_python: Override feature flag (None = use flag).
        progress_logger: Optional ProgressLogger for exploration logging.
        task_id: Optional task ID for associating exploration logs.

    Returns:
        REPLEnvironment instance (may use RestrictedExecutor internally).
    """
    # Check feature flag if not explicitly set
    if use_restricted_python is None:
        from src.features import features
        use_restricted_python = features().restricted_python

    # Try to use RestrictedPython if requested
    if use_restricted_python:
        try:
            from src.restricted_executor import is_available, RestrictedExecutor

            if is_available():
                # Create a wrapped environment that uses RestrictedExecutor
                return _RestrictedREPLEnvironment(
                    context=context,
                    artifacts=artifacts,
                    config=config,
                    llm_primitives=llm_primitives,
                    tool_registry=tool_registry,
                    script_registry=script_registry,
                    role=role,
                    progress_logger=progress_logger,
                    task_id=task_id,
                )
        except ImportError:
            pass  # Fall back to standard REPL

    # Standard REPL environment
    return REPLEnvironment(
        context=context,
        artifacts=artifacts,
        config=config,
        llm_primitives=llm_primitives,
        tool_registry=tool_registry,
        script_registry=script_registry,
        role=role,
        progress_logger=progress_logger,
        task_id=task_id,
    )


class _RestrictedREPLEnvironment(REPLEnvironment):
    """REPL environment that uses RestrictedPython for execution.

    This is a wrapper around the standard REPLEnvironment that delegates
    execution to RestrictedExecutor for stronger security guarantees.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with RestrictedExecutor."""
        super().__init__(*args, **kwargs)

        # Import here to avoid issues if not available
        from src.restricted_executor import RestrictedExecutor

        # Create restricted executor with same config
        self._restricted_executor = RestrictedExecutor(
            context=self.context,
            artifacts=self.artifacts,
            timeout_seconds=self.config.timeout_seconds,
            output_cap=self.config.output_cap,
            max_grep_results=self.config.max_grep_results,
            llm_primitives=self.llm_primitives,
        )

    def execute(self, code: str) -> ExecutionResult:
        """Execute using RestrictedPython.

        Args:
            code: Python code to execute.

        Returns:
            ExecutionResult with output, is_final flag, and optional error.
        """
        # Use the restricted executor
        from src.restricted_executor import ExecutionResult as RestrictedResult

        result = self._restricted_executor.execute(code)

        # Convert to our ExecutionResult type
        return ExecutionResult(
            output=result.output,
            is_final=result.is_final,
            final_answer=result.final_answer,
            error=result.error,
            elapsed_seconds=result.elapsed_seconds,
        )

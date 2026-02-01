"""File exploration tools for the REPL environment.

Read-only file system inspection: peek, grep, list_dir, file_info.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.repl_environment.types import wrap_tool_output

logger = logging.getLogger(__name__)


class _FileExplorationMixin:
    """Mixin providing read-only file system exploration tools (_peek, _grep, _list_dir, _file_info).

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig — environment configuration
        context: str — full input context
        artifacts: dict — collected artifacts
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
        _grep_hits_buffer: list — grep results buffer for two-stage summarization
        _validate_file_path: Callable[[str], tuple[bool, str | None]] — path validation method
    """

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
                logger.debug("peek failed", exc_info=True)
                return f"[ERROR: {type(e).__name__}: {e}]"

        # Read from context
        result = self.context[:n]
        self._exploration_log.add_event("peek", {"n": n}, result)
        return result

    def _grep(
        self,
        pattern: str,
        file_path: str | None = None,
        context_lines: int = 2,
    ) -> list[str]:
        """Search context or file with regex and return matching lines.

        Also captures hits to _grep_hits_buffer for two-stage summarization.

        Args:
            pattern: Regular expression pattern to search for.
            file_path: Optional file path to search instead of context.
            context_lines: Number of context lines before/after match (default 2).

        Returns:
            List of lines containing matches (capped at max_grep_results).
        """
        self._exploration_calls += 1
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return [f"[REGEX ERROR: {e}]"]

        # Determine source text
        source_name = "context"
        if file_path is not None:
            is_valid, error = self._validate_file_path(file_path)
            if not is_valid:
                return [f"[ERROR: {error}]"]
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    source_text = f.read()
                source_name = file_path
            except FileNotFoundError:
                return [f"[ERROR: File not found: {file_path}]"]
            except Exception as e:
                logger.debug("grep failed", exc_info=True)
                return [f"[ERROR: {type(e).__name__}: {e}]"]
        else:
            source_text = self.context

        lines = source_text.split("\n")
        matches = []
        match_details = []  # For grep hits buffer

        for i, line in enumerate(lines):
            if regex.search(line):
                matches.append(line)

                # Capture context for grep hits buffer
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                context_snippet = "\n".join(lines[start:end])

                match_details.append({
                    "line_num": i + 1,
                    "match": line[:500],  # Cap line length
                    "context": context_snippet[:1000],  # Cap context
                })

                if len(matches) >= self.config.max_grep_results:
                    matches.append(f"[... truncated at {self.config.max_grep_results} results]")
                    break

        # Store in grep hits buffer for two-stage pipeline
        if match_details:
            self._grep_hits_buffer.append({
                "pattern": pattern,
                "source": source_name,
                "match_count": len(match_details),
                "hits": match_details[:20],  # Cap at 20 detailed hits
            })

        self._exploration_log.add_event("grep", {"pattern": pattern, "file_path": file_path}, matches)
        return matches

    def _list_dir(self, path: str) -> str:
        """List contents of a directory.

        Args:
            path: Absolute path to the directory.

        Returns:
            JSON string with directory contents.
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
                        logger.debug("Failed to stat %s", entry.name, exc_info=True)
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

            # Use TOON encoding for token efficiency if enabled
            if self.config.use_toon_encoding:
                from src.services.toon_encoder import encode_list_dir
                output = encode_list_dir(path, entries[:100], len(entries))
            else:
                output = json.dumps(result, indent=2)
            self.artifacts.setdefault("_tool_outputs", []).append(output)
            return wrap_tool_output(output)

        except FileNotFoundError:
            return f"[ERROR: Directory not found: {path}]"
        except NotADirectoryError:
            return f"[ERROR: Not a directory: {path}]"
        except PermissionError:
            return f"[ERROR: Permission denied: {path}]"
        except Exception as e:
            logger.debug("list_dir failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _file_info(self, path: str) -> str:
        """Get metadata about a file.

        Args:
            path: Absolute path to the file.

        Returns:
            JSON string with file metadata.
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
            logger.debug("file_info failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

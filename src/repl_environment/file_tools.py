"""File I/O tool methods for the REPL environment.

Provides mixin with: peek, grep, list_dir, file_info, archive tools,
web_fetch, run_shell, log_append, file_write_safe, and patch tools.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.repl_environment.types import wrap_tool_output

logger = logging.getLogger(__name__)


class _FileToolsMixin:
    """Mixin providing file I/O tools for REPLEnvironment.

    Expects the following attributes from the concrete class:
    - config: REPLConfig
    - context: str
    - artifacts: dict
    - _exploration_calls: int
    - _exploration_log: ExplorationLog
    - progress_logger: ProgressLogger | None
    - task_id: str
    - _grep_hits_buffer: list
    - ALLOWED_FILE_PATHS: list[str]
    - _validate_file_path(path) -> tuple[bool, str | None]
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

    # =========================================================================
    # Archive Tools (for document ingestion from compressed files)
    # =========================================================================

    def _archive_open(self, path: str) -> str:
        """Open an archive and return its manifest.

        Supported formats: .zip, .tar, .tar.gz, .tgz, .tar.bz2, .tar.xz, .7z

        Args:
            path: Absolute path to the archive file.

        Returns:
            JSON string with archive manifest.
        """
        self._exploration_calls += 1
        import json
        from pathlib import Path

        # Validate path
        is_valid, error = self._validate_file_path(path)
        if not is_valid:
            return f"[ERROR: {error}]"

        try:
            from src.services.archive_extractor import ArchiveExtractor

            extractor = ArchiveExtractor()
            archive_path = Path(path)

            # Validate archive first
            validation = extractor.validate(archive_path)
            if not validation.is_safe:
                return json.dumps({
                    "error": f"Archive validation failed: {validation.status.value}",
                    "issues": validation.issues,
                }, indent=2)

            # Get manifest
            manifest = extractor.list_contents(archive_path)

            # Store in artifacts for later use
            if "_archives" not in self.artifacts:
                self.artifacts["_archives"] = {}

            archive_name = archive_path.name
            self.artifacts["_archives"][archive_name] = {
                "manifest": manifest,
                "path": path,
                "extracted_to": None,
                "processed_files": {},
            }
            # Also track "current" archive for easier access
            self.artifacts["_archives"]["current"] = archive_name

            result = manifest.to_summary_dict()
            self._exploration_log.add_event("archive_open", {"path": path}, result)
            return json.dumps(result, indent=2)

        except ImportError:
            return "[ERROR: Archive extractor not available]"
        except Exception as e:
            logger.debug("archive_open failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _archive_extract(
        self,
        archive_name: str | None = None,
        pattern: str | None = None,
        files: list[str] | None = None,
        process_documents: bool = True,
    ) -> str:
        """Extract files from an opened archive.

        Must call archive_open() first.

        Args:
            archive_name: Name of the archive (from archive_open). If None,
                         uses the most recently opened archive.
            pattern: Glob pattern to match (e.g., "*.pdf", "docs/*.md").
            files: List of specific files to extract.
            process_documents: If True, route PDFs through OCR pipeline.

        Returns:
            JSON string with extraction results.
        """
        self._exploration_calls += 1
        import json
        from pathlib import Path

        try:
            from src.services.archive_extractor import ArchiveExtractor

            # Get archive info from artifacts
            if "_archives" not in self.artifacts or not self.artifacts["_archives"]:
                return "[ERROR: No archive opened. Call archive_open() first.]"

            if archive_name is None:
                # Use the "current" marker if available
                current = self.artifacts["_archives"].get("current")
                if current:
                    archive_name = current
                else:
                    # Fall back to last opened archive (excluding "current" key)
                    real_archives = [k for k in self.artifacts["_archives"].keys() if k != "current"]
                    if not real_archives:
                        return "[ERROR: No archive opened. Call archive_open() first.]"
                    archive_name = real_archives[-1]

            if archive_name not in self.artifacts["_archives"]:
                real_archives = [k for k in self.artifacts["_archives"].keys() if k != "current"]
                return f"[ERROR: Archive not found: {archive_name}. Opened: {real_archives}]"

            archive_info = self.artifacts["_archives"][archive_name]
            if not isinstance(archive_info, dict):
                return f"[ERROR: Invalid archive info for: {archive_name}]"
            archive_path = Path(archive_info["path"])
            manifest = archive_info["manifest"]

            extractor = ArchiveExtractor()

            # Determine what to extract
            if pattern:
                result = extractor.extract_pattern(archive_path, pattern)
            elif files:
                result = extractor.extract_files(archive_path, files)
            else:
                # Extract all
                result = extractor.extract_all(archive_path)

            if not result.success and not result.extracted_files:
                return json.dumps({
                    "error": "Extraction failed",
                    "errors": result.errors,
                }, indent=2)

            # Update artifacts with extracted files
            archive_info["extracted_to"] = str(result.extracted_files.get(
                list(result.extracted_files.keys())[0], ""
            ).parent if result.extracted_files else None)

            # Process documents if requested
            sections_total = 0
            figures_total = 0

            if process_documents:
                # Route document files through preprocessing
                doc_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

                for filename, file_path in result.extracted_files.items():
                    ext = Path(filename).suffix.lower()

                    if ext in doc_extensions:
                        # Try to process via OCR
                        try:
                            ocr_result = self._ocr_document(str(file_path))
                            ocr_data = json.loads(ocr_result)

                            if "error" not in ocr_result.lower():
                                archive_info["processed_files"][filename] = {
                                    "type": "document",
                                    "sections": ocr_data.get("total_pages", 0),
                                    "figures": len(ocr_data.get("figures", [])),
                                    "text_preview": ocr_data.get("full_text", "")[:500],
                                }
                                sections_total += ocr_data.get("total_pages", 0)
                                figures_total += len(ocr_data.get("figures", []))
                        except Exception:
                            logger.debug("Archive document processing failed: %s", filename, exc_info=True)
                            # Store as unprocessed document
                            archive_info["processed_files"][filename] = {
                                "type": "document",
                                "error": "Processing failed",
                            }
                    else:
                        # Text/code files - read content directly
                        try:
                            content = file_path.read_text(encoding="utf-8", errors="replace")
                            archive_info["processed_files"][filename] = {
                                "type": "text",
                                "content": content[:50000],  # Cap at 50K
                                "lines": content.count("\n") + 1,
                            }
                        except Exception:
                            logger.debug("Archive binary processing failed: %s", filename, exc_info=True)
                            archive_info["processed_files"][filename] = {
                                "type": "binary",
                                "size": file_path.stat().st_size,
                            }

            summary = {
                "success": True,
                "extracted": len(result.extracted_files),
                "processed": len(archive_info["processed_files"]),
                "sections_total": sections_total,
                "figures_total": figures_total,
                "skipped": len(result.skipped_files),
                "stored_in": f"artifacts['_archives']['{archive_name}']",
            }

            self._exploration_log.add_event("archive_extract", {
                "archive_name": archive_name,
                "pattern": pattern,
                "files": files,
            }, summary)

            return json.dumps(summary, indent=2)

        except ImportError:
            return "[ERROR: Archive extractor not available]"
        except Exception as e:
            logger.debug("archive_extract failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _archive_file(self, filename: str, archive_name: str | None = None) -> str:
        """Get content of a specific file from an extracted archive.

        Args:
            filename: Path of the file within the archive.
            archive_name: Name of the archive. If None, searches all.

        Returns:
            File content (text) or document metadata (for processed PDFs).
        """
        self._exploration_calls += 1
        import json

        if "_archives" not in self.artifacts:
            return "[ERROR: No archives opened]"

        # Find the file
        archives_to_search = [archive_name] if archive_name else list(self.artifacts["_archives"].keys())

        for arch_name in archives_to_search:
            if arch_name not in self.artifacts["_archives"]:
                continue

            archive_info = self.artifacts["_archives"][arch_name]
            processed = archive_info.get("processed_files", {})

            if filename in processed:
                file_info = processed[filename]

                if file_info.get("type") == "text":
                    return file_info.get("content", "")
                elif file_info.get("type") == "document":
                    return json.dumps(file_info, indent=2)
                else:
                    return json.dumps(file_info, indent=2)

        return f"[ERROR: File not found: {filename}]"

    def _archive_search(self, query: str, archive_name: str | None = None) -> str:
        """Search across all extracted archive content.

        Args:
            query: Search query string.
            archive_name: Name of archive to search. If None, searches all.

        Returns:
            JSON string with search results.
        """
        self._exploration_calls += 1
        import json
        import re

        if "_archives" not in self.artifacts:
            return json.dumps([])

        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        if archive_name:
            archives_to_search = [archive_name]
        else:
            # Skip the "current" marker key
            archives_to_search = [k for k in self.artifacts["_archives"].keys() if k != "current"]

        for arch_name in archives_to_search:
            if arch_name not in self.artifacts["_archives"]:
                continue

            archive_info = self.artifacts["_archives"][arch_name]
            if not isinstance(archive_info, dict):
                continue
            processed = archive_info.get("processed_files", {})

            for filename, file_info in processed.items():
                file_type = file_info.get("type", "")

                if file_type == "text":
                    content = file_info.get("content", "")
                    for i, line in enumerate(content.splitlines(), 1):
                        if pattern.search(line):
                            results.append({
                                "archive": arch_name,
                                "file": filename,
                                "line": i,
                                "match": line[:200],
                            })
                            if len(results) >= 50:  # Cap results
                                break

                elif file_type == "document":
                    text = file_info.get("text_preview", "")
                    if pattern.search(text):
                        # Find match context
                        match = pattern.search(text)
                        if match:
                            start = max(0, match.start() - 50)
                            end = min(len(text), match.end() + 50)
                            results.append({
                                "archive": arch_name,
                                "file": filename,
                                "section": "document",
                                "match": text[start:end],
                            })

                if len(results) >= 50:
                    break

            if len(results) >= 50:
                break

        self._exploration_log.add_event("archive_search", {"query": query}, results)
        return json.dumps(results, indent=2)

    def _web_fetch(self, url: str, max_chars: int = 10000) -> str:
        """Fetch content from a URL.

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
            logger.debug("web_fetch failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _run_shell(self, cmd: str, timeout: int = 30) -> str:
        """Run a sandboxed shell command (read-only operations only).

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
            import shlex
            result = subprocess.run(
                shlex.split(cmd),
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
            logger.debug("run_shell failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

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
            logger.debug("log_append failed", exc_info=True)
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
            logger.debug("file_write_safe failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    # =========================================================================
    # Patch Tools
    # =========================================================================

    def _prepare_patch(self, files: list[str], description: str) -> str:
        """Generate unified diff for owner review.

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
            logger.debug("prepare_patch failed", exc_info=True)
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
            logger.debug("list_patches failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _apply_approved_patch(self, patch_name: str) -> str:
        """Apply a patch after owner approval.

        Args:
            patch_name: Name of the patch file.

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
            logger.debug("apply_approved_patch failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _reject_patch(self, patch_name: str, reason: str) -> str:
        """Reject a pending patch with reason.

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
            logger.debug("reject_patch failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

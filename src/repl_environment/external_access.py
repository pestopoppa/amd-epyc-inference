"""External access tools for the REPL environment.

Network access (web_fetch) and sandboxed shell execution (run_shell).
"""

from __future__ import annotations

import logging
import subprocess
import shlex
from typing import Any

logger = logging.getLogger(__name__)


class _ExternalAccessMixin:
    """Mixin providing network and shell access tools (_web_fetch, _run_shell).

    Required attributes (provided by REPLEnvironment.__init__):
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
    """

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

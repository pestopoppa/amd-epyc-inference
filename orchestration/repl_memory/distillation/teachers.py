"""
Teacher model interfaces for skill distillation.

Each teacher implements a common protocol. The distillation pipeline
selects the teacher per skill type.

Available teachers:
- ClaudeTeacher: Anthropic API (Opus 4.6) — best reasoning
- CodexTeacher: OpenAI Codex CLI (gpt-5.3-codex) via `codex exec --json`
- LocalLlamaTeacher: llama.cpp server (e.g., Qwen3-235B at :8083)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TeacherModel(Protocol):
    """Abstract interface for distillation teachers."""

    async def distill(
        self,
        prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a distillation prompt, return the teacher's response.

        Args:
            prompt: Complete prompt with trajectory data embedded
            max_tokens: Maximum response length

        Returns:
            Raw text response containing JSON skill blocks
        """
        ...

    @property
    def model_id(self) -> str:
        """Identifier for provenance tracking."""
        ...


class ClaudeTeacher:
    """
    Uses Claude Code CLI (claude -p) for distillation.

    Invokes `claude -p <prompt> --output-format text --model <model>`
    as a subprocess — same pattern as ClaudeDebugger.  No API key needed;
    auth is handled by the Claude Code CLI session.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        timeout: int = 300,
    ):
        self._model = model
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model

    async def distill(self, prompt: str, max_tokens: int = 4096) -> str:
        import asyncio
        import tempfile

        # Write prompt to temp file to avoid shell escaping issues
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
            dir="/mnt/raid0/llm/tmp",
        ) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            cmd = [
                "claude", "-p",
                f"Read the file {prompt_file} and follow its instructions exactly. "
                f"Return ONLY the JSON output requested, no commentary.",
                "--output-format", "text",
                "--model", self._model,
                "--allowedTools", "Read",
            ]

            # Strip CLAUDECODE env var to allow nested invocation
            import os
            env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout,
            )

            if proc.returncode != 0:
                error = stderr.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"claude -p failed (rc={proc.returncode}): {error[:500]}"
                )

            return stdout.decode("utf-8", errors="replace")

        finally:
            import os
            try:
                os.unlink(prompt_file)
            except OSError:
                pass


class LocalLlamaTeacher:
    """
    Uses a local llama.cpp server (OpenAI-compatible API).

    Default: Qwen3-235B-A22B at port 8083.

    Reuses a single httpx.AsyncClient across calls to avoid per-request
    TCP/TLS setup overhead.  Use as async context manager or call
    ``close()`` explicitly when done::

        async with LocalLlamaTeacher() as teacher:
            result = await teacher.distill(prompt)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8083",
        model_id: str = "qwen3-235b-a22b",
        timeout: int = 300,
    ):
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout = timeout
        self._client: Optional[Any] = None  # lazy httpx.AsyncClient

    async def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    @property
    def model_id(self) -> str:
        return self._model_id

    async def distill(self, prompt: str, max_tokens: int = 4096) -> str:
        try:
            import httpx  # noqa: F811 — needed for ImportError path
        except ImportError as e:
            raise ImportError(
                "httpx not installed. Run: pip install httpx"
            ) from e

        client = await self._get_client()
        response = await client.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class CodexTeacher:
    """
    Uses OpenAI Codex CLI (gpt-5.3-codex) for distillation.

    Binary: ~/.nvm/versions/node/v22.14.0/bin/codex
    Package: @openai/codex v0.98.0
    Auth: ChatGPT OAuth (~/.codex/auth.json)
    Config: ~/.codex/config.toml (model = "gpt-5.3-codex")

    Uses `codex exec --json` for non-interactive batch processing.
    The --json flag outputs JSONL events to stdout; we collect the
    assistant message content.
    """

    # Default binary path (nvm-managed)
    DEFAULT_BINARY = "/home/daniele/.nvm/versions/node/v22.14.0/bin/codex"

    def __init__(
        self,
        binary_path: Optional[str] = None,
        model: str = "gpt-5.3-codex",
        timeout: int = 300,
    ):
        self._binary = binary_path or self.DEFAULT_BINARY
        self._model = model
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model

    async def distill(self, prompt: str, max_tokens: int = 4096) -> str:
        import asyncio
        import tempfile

        # Write prompt to a temp file to avoid shell escaping issues
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
            dir="/mnt/raid0/llm/tmp",
        ) as f:
            f.write(prompt)
            prompt_file = f.name

        try:
            # codex exec reads prompt from args, outputs JSONL with --json
            cmd = [
                self._binary, "exec",
                "--json",
                "-m", self._model,
                "-s", "read-only",  # No filesystem writes needed for distillation
                "--full-auto",  # No approval prompts
                f"Read the file {prompt_file} and follow its instructions exactly.",
            ]

            # Strip CLAUDECODE env var to allow invocation from Claude Code
            import os
            env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )

            if proc.returncode != 0:
                error = stderr.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Codex exec failed (rc={proc.returncode}): {error[:500]}"
                )

            # Parse JSONL output — collect assistant message content
            # Codex JSONL format:
            #   {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
            output = stdout.decode("utf-8", errors="replace")
            content_parts = []
            for line in output.strip().splitlines():
                try:
                    event = json.loads(line)
                    if event.get("type") == "item.completed":
                        item = event.get("item", {})
                        if item.get("type") == "agent_message":
                            content_parts.append(item.get("text", ""))
                except json.JSONDecodeError:
                    continue

            return "".join(content_parts) if content_parts else output

        finally:
            import os
            try:
                os.unlink(prompt_file)
            except OSError:
                pass


class MockTeacher:
    """
    Mock teacher for testing. Returns pre-configured skill JSON.

    Usage:
        teacher = MockTeacher(responses=["```json\\n{...}\\n```"])
    """

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        model_id: str = "mock-teacher",
    ):
        self._responses = list(responses or [])
        self._call_count = 0
        self._model_id = model_id
        self._prompts: List[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def prompts(self) -> List[str]:
        """Prompts received (for test assertions)."""
        return self._prompts

    async def distill(self, prompt: str, max_tokens: int = 4096) -> str:
        self._prompts.append(prompt)
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = "No skills to extract from these trajectories."
        self._call_count += 1
        return response


def parse_skills_from_response(response: str) -> List[Dict[str, Any]]:
    """
    Extract JSON skill blocks from a teacher's response.

    Tries:
    1. ```json ... ``` fenced code blocks
    2. Bare {...} JSON objects

    Returns:
        List of parsed skill dicts
    """
    skills = []

    # Try fenced JSON blocks first
    fenced = re.findall(r"```json\s*\n(.*?)```", response, re.DOTALL)
    for block in fenced:
        block = block.strip()
        try:
            parsed = json.loads(block)
            if isinstance(parsed, list):
                skills.extend(parsed)
            elif isinstance(parsed, dict):
                skills.append(parsed)
        except json.JSONDecodeError:
            logger.debug("Failed to parse fenced JSON block: %.100s...", block)

    if skills:
        return skills

    # Fallback: bare JSON objects
    bare = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response)
    for block in bare:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict) and "title" in parsed:
                skills.append(parsed)
        except json.JSONDecodeError:
            continue

    return skills

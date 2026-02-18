"""Context management, completion signals, and tool dispatch.

Provides mixin with: context_len, chunk_context, summarize_chunks,
FINAL/FINAL_VAR, mark_finding, list_findings, get/clear_findings,
tracked LLM calls, and tool/script invocation.
"""

from __future__ import annotations

from typing import Any

from src.repl_environment.types import FinalSignal


class _ContextMixin:
    """Mixin providing context management and tool dispatch.

    Includes: _context_len, _chunk_context, _summarize_chunks, _final, _final_var,
    _mark_finding, _list_findings, get_findings, clear_findings, _tracked_llm_call,
    _tracked_llm_batch, _invoke_tool, _call_tool, _list_tools, _invoke_script, _find_scripts.

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig — environment configuration
        context: str — full input context
        artifacts: dict — collected artifacts
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
        _execution_count: int — number of execute() calls
        _findings_buffer: list — key findings buffer for session persistence
        _tool_invocations: int — tool invocation counter
        llm_primitives: Any | None — LLM primitives for delegation
        tool_registry: Any | None — registry of available tools
        script_registry: Any | None — registry of prepared scripts
        role: str — current agent role
        _globals: dict — restricted globals for REPL execution
    """

    # ------------------------------------------------------------------
    # Long context exploration tools
    # ------------------------------------------------------------------

    def _context_len(self) -> int:
        """Return the character length of the context.

        Returns:
            Number of characters in the context string.
        """
        return len(self.context)

    def _chunk_context(self, n_chunks: int = 4, overlap: int = 200) -> list[dict]:
        """Split context into roughly equal chunks with metadata.

        Args:
            n_chunks: Number of chunks to split into (default 4).
            overlap: Characters of overlap between chunks (default 200).

        Returns:
            List of dicts with index, start, end, text, char_count.
        """
        self._exploration_calls += 1
        text = self.context
        total = len(text)
        if total == 0:
            return []

        n_chunks = max(1, min(n_chunks, 20))  # Cap at 20
        chunk_size = total // n_chunks
        chunks = []

        for i in range(n_chunks):
            start = max(0, i * chunk_size - (overlap if i > 0 else 0))
            end = min(total, (i + 1) * chunk_size + (overlap if i < n_chunks - 1 else 0))
            # Avoid splitting mid-word: extend to next whitespace
            if end < total:
                ws = text.find("\n", end)
                if ws != -1 and ws - end < 200:
                    end = ws

            chunk_text = text[start:end]
            chunks.append(
                {
                    "index": i,
                    "start": start,
                    "end": end,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
            )

        self._exploration_log.add_event(
            "chunk_context",
            {"n_chunks": n_chunks, "overlap": overlap, "total_chars": total},
            f"{len(chunks)} chunks",
        )
        return chunks

    def _summarize_chunks(
        self,
        task: str = "Summarize the key content",
        n_chunks: int = 4,
        role: str = "worker_general",
    ) -> list[dict]:
        """Chunk context and summarize each chunk in parallel using workers.

        Args:
            task: The task/question to apply to each chunk.
            n_chunks: Number of chunks (default 4).
            role: Worker role to use (default "worker_general").

        Returns:
            List of dicts with index, chunk_start, chunk_end, chunk_chars, summary.
        """
        self._exploration_calls += 1

        if self.llm_primitives is None:
            return [{"error": "LLM primitives not available"}]

        chunks = self._chunk_context(n_chunks)
        if not chunks:
            return [{"error": "Empty context"}]

        # Build prompts for each chunk
        prompts = []
        for chunk in chunks:
            prompt = (
                f"You are analyzing section {chunk['index'] + 1} of {len(chunks)} "
                f"from a larger document (chars {chunk['start']}-{chunk['end']}).\n\n"
                f"## Task\n{task}\n\n"
                f"## Section Content\n{chunk['text'][:15000]}\n\n"  # Cap per-chunk
                f"## Instructions\nAnalyze this section for the task above. "
                f"Be specific about what you find. Note any references to other "
                f"sections or information that may require cross-referencing."
            )
            prompts.append(prompt)

        # Dispatch to workers in batch
        try:
            summaries = self.llm_primitives.llm_batch(
                prompts,
                role=role,
                n_tokens=512,
            )
        except Exception as e:
            return [{"error": f"Batch call failed: {e}"}]

        results = []
        for i, (chunk, summary) in enumerate(zip(chunks, summaries)):
            results.append(
                {
                    "index": i,
                    "chunk_start": chunk["start"],
                    "chunk_end": chunk["end"],
                    "chunk_chars": chunk["char_count"],
                    "summary": summary,
                }
            )

        self._exploration_log.add_event(
            "summarize_chunks",
            {"task": task[:100], "n_chunks": n_chunks, "role": role},
            f"{len(results)} summaries",
        )
        return results

    # ------------------------------------------------------------------
    # Completion signals
    # ------------------------------------------------------------------

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
        # Guard: model passed a function/class object instead of source text.
        # str() on callables produces '<function foo at 0x...>' which is useless.
        if callable(answer):
            raise ValueError(
                f"FINAL() received a {type(answer).__name__} object, not a string. "
                "Pass the source code as a string: FINAL('''def foo(): ...''')"
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

    # ------------------------------------------------------------------
    # Session persistence (findings)
    # ------------------------------------------------------------------

    def _mark_finding(
        self,
        content: str,
        tags: list[str] | None = None,
        source_file: str | None = None,
        source_page: int | None = None,
        source_section: str | None = None,
    ) -> dict[str, Any]:
        """Mark a key finding for session persistence.

        Args:
            content: The finding text (required).
            tags: Optional tags for categorization.
            source_file: Optional source file path.
            source_page: Optional page number (for PDFs).
            source_section: Optional section ID or title.

        Returns:
            Dict with finding info including ID.
        """
        import time
        import uuid

        finding = {
            "id": str(uuid.uuid4()),
            "content": content,
            "tags": tags or [],
            "source": {
                "file": source_file,
                "page": source_page,
                "section": source_section,
            },
            "turn": self._execution_count,
            "timestamp": time.time(),
        }

        self._findings_buffer.append(finding)

        return {
            "id": finding["id"],
            "content": content[:100] + "..." if len(content) > 100 else content,
            "tags": finding["tags"],
            "status": "marked",
        }

    def _list_findings(self) -> list[dict[str, Any]]:
        """List all findings marked in this session.

        Returns:
            List of finding summaries with id, content preview, tags, and turn.
        """
        return [
            {
                "id": f["id"],
                "content": f["content"][:80] + "..." if len(f["content"]) > 80 else f["content"],
                "tags": f["tags"],
                "turn": f["turn"],
            }
            for f in self._findings_buffer
        ]

    def get_findings(self) -> list[dict[str, Any]]:
        """Get all findings (full content) for external access.

        Returns:
            List of full finding dicts.
        """
        return self._findings_buffer.copy()

    def clear_findings(self) -> int:
        """Clear the findings buffer after syncing.

        Returns:
            Number of findings cleared.
        """
        count = len(self._findings_buffer)
        self._findings_buffer = []
        return count

    # ------------------------------------------------------------------
    # LLM call wrappers (exploration tracked)
    # ------------------------------------------------------------------

    def _tracked_llm_call(self, *args, **kwargs) -> str:
        """Wrapper for llm_call that tracks exploration.

        Returns:
            llm_call result.
        """
        self._exploration_calls += 1
        result = self.llm_primitives.llm_call(*args, **kwargs)
        self._exploration_log.add_event("llm_call", {"args": args, "kwargs": kwargs}, result)
        return result

    def _tracked_llm_batch(self, *args, **kwargs) -> list[str]:
        """Wrapper for llm_batch that tracks exploration.

        Returns:
            llm_batch result.
        """
        self._exploration_calls += 1
        result = self.llm_primitives.llm_batch(*args, **kwargs)
        self._exploration_log.add_event("llm_batch", {"args": args, "kwargs": kwargs}, result)
        return result

    # ------------------------------------------------------------------
    # Tool / script dispatch
    # ------------------------------------------------------------------

    def _invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """Invoke a registered tool.

        Args:
            tool_name: Name of the tool to invoke.
            **kwargs: Tool arguments.

        Returns:
            Tool result.
        """
        if self.tool_registry is None:
            raise RuntimeError("No tool registry configured")

        self._tool_invocations += 1
        chain_id = getattr(self, "_active_tool_chain_id", None)
        chain_index = int(getattr(self, "_active_tool_chain_index", 0))
        caller_type = "chain" if chain_id else "direct"

        result = self.tool_registry.invoke(
            tool_name,
            self.role,
            caller_type=caller_type,
            chain_id=chain_id,
            chain_index=chain_index,
            **kwargs,
        )
        if chain_id:
            self._active_tool_chain_index = chain_index + 1

        # Track in research context
        if hasattr(self, "_research_context"):
            try:
                node_id = self._research_context.add(
                    tool="TOOL",
                    query=f"{tool_name}({', '.join(f'{k}={v!r}' for k, v in list(kwargs.items())[:3])})",
                    content=str(result)[:8000],
                    parent_id=getattr(self, "_last_research_node", None),
                )
                self._last_research_node = node_id
            except Exception:
                pass  # Silently ignore research tracking failures

        return result

    def _call_tool(self, tool_name: str, **kwargs) -> str:
        """Invoke a registered tool and return JSON-serialized result.

        Args:
            tool_name: Name of the tool to invoke.
            **kwargs: Tool arguments.

        Returns:
            JSON-serialized string of the tool result.
        """
        import json as _json

        result = self._invoke_tool(tool_name, **kwargs)
        try:
            return _json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            return str(result)

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
            query: Search query.
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

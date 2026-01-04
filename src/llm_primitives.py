#!/usr/bin/env python3
"""LLM primitives for RLM-style orchestration.

This module provides llm_call() and llm_batch() functions for spawning
sub-LM calls from the Root LM. Supports both mock mode (for testing)
and real inference via the ModelServer.

Usage:
    from src.llm_primitives import LLMPrimitives

    # Mock mode (for testing)
    primitives = LLMPrimitives(mock_mode=True)
    result = primitives.llm_call("Summarize this:", "Some text")

    # Real mode (with model server)
    from src.model_server import ModelServer
    server = ModelServer()
    primitives = LLMPrimitives(model_server=server, mock_mode=False)
    result = primitives.llm_call("Summarize this:", "Some text", role="worker")
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CallLogEntry:
    """Entry in the call log for debugging/testing."""

    timestamp: float
    call_type: str  # "call" or "batch"
    prompt: str | None = None
    prompts: list[str] | None = None
    context_slice: str | None = None
    role: str = "worker"
    result: str | list[str] | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass
class LLMPrimitivesConfig:
    """Configuration for LLM primitives."""

    output_cap: int = 8192  # Max chars per sub-LM output
    batch_parallelism: int = 4  # Max parallel calls in llm_batch
    call_timeout: int = 120  # Timeout per call in seconds
    mock_response_prefix: str = "[MOCK]"  # Prefix for mock responses


class LLMPrimitives:
    """LLM primitives for sub-LM spawning.

    Provides llm_call() for single calls and llm_batch() for parallel calls.
    Can operate in mock mode for testing or with a real ModelServer.
    """

    def __init__(
        self,
        model_server: Any | None = None,
        mock_mode: bool = True,
        config: LLMPrimitivesConfig | None = None,
        mock_responses: dict[str, str] | None = None,
    ):
        """Initialize LLM primitives.

        Args:
            model_server: Optional ModelServer instance for real inference.
            mock_mode: If True, return mock responses instead of real inference.
            config: Optional configuration for output caps, parallelism, etc.
            mock_responses: Optional dict mapping prompts to mock responses.
        """
        self.model_server = model_server
        self.mock_mode = mock_mode
        self.config = config if config is not None else LLMPrimitivesConfig()
        self.mock_responses = mock_responses if mock_responses is not None else {}

        # Call log for debugging and testing
        self.call_log: list[CallLogEntry] = []

        # Stats
        self.total_calls = 0
        self.total_batch_calls = 0
        self.total_tokens_generated = 0

    def llm_call(
        self,
        prompt: str,
        context_slice: str = "",
        role: str = "worker",
    ) -> str:
        """Call a sub-LM with optional context slice.

        Args:
            prompt: The instruction/prompt for the sub-LM.
            context_slice: Optional context to include (appended to prompt).
            role: Role determining which model to use (e.g., "worker", "coder").

        Returns:
            Sub-LM response (capped at output_cap chars).
        """
        start_time = time.perf_counter()
        self.total_calls += 1

        # Build full prompt
        if context_slice:
            full_prompt = f"{prompt}\n\nContext:\n{context_slice}"
        else:
            full_prompt = prompt

        # Create log entry
        log_entry = CallLogEntry(
            timestamp=time.time(),
            call_type="call",
            prompt=prompt,
            context_slice=context_slice[:500] if context_slice else None,
            role=role,
        )

        try:
            if self.mock_mode:
                result = self._mock_call(full_prompt, role)
            else:
                result = self._real_call(full_prompt, role)

            # Cap output
            if len(result) > self.config.output_cap:
                result = result[: self.config.output_cap] + f"\n[... truncated at {self.config.output_cap} chars]"

            log_entry.result = result[:500]  # Truncate for log
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

            return result

        except Exception as e:
            log_entry.error = str(e)
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)
            return f"[ERROR: {e}]"

    def llm_batch(
        self,
        prompts: list[str],
        role: str = "worker",
    ) -> list[str]:
        """Call multiple sub-LMs in parallel.

        Args:
            prompts: List of prompts to send to sub-LMs.
            role: Role determining which model to use.

        Returns:
            List of responses in the same order as prompts.
        """
        start_time = time.perf_counter()
        self.total_batch_calls += 1

        # Create log entry
        log_entry = CallLogEntry(
            timestamp=time.time(),
            call_type="batch",
            prompts=prompts[:5] if len(prompts) <= 5 else prompts[:5] + ["..."],
            role=role,
        )

        try:
            if self.mock_mode:
                results = self._mock_batch(prompts, role)
            else:
                results = self._real_batch(prompts, role)

            # Cap each output
            capped_results = []
            for result in results:
                if len(result) > self.config.output_cap:
                    result = result[: self.config.output_cap] + f"\n[... truncated at {self.config.output_cap} chars]"
                capped_results.append(result)

            log_entry.result = [r[:200] for r in capped_results[:3]]  # Truncate for log
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

            return capped_results

        except Exception as e:
            log_entry.error = str(e)
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)
            return [f"[ERROR: {e}]" for _ in prompts]

    def _mock_call(self, prompt: str, role: str) -> str:
        """Generate a mock response for testing.

        Args:
            prompt: The full prompt.
            role: The role being called.

        Returns:
            Mock response string.
        """
        # Check for custom mock responses
        for key, response in self.mock_responses.items():
            if key in prompt:
                return response

        # Default mock response
        prompt_preview = prompt[:50].replace("\n", " ")
        return f"{self.config.mock_response_prefix} Response for role='{role}': {prompt_preview}..."

    def _mock_batch(self, prompts: list[str], role: str) -> list[str]:
        """Generate mock responses for batch testing.

        Args:
            prompts: List of prompts.
            role: The role being called.

        Returns:
            List of mock responses.
        """
        return [
            f"{self.config.mock_response_prefix} Batch[{i}] for role='{role}': {p[:30]}..."
            for i, p in enumerate(prompts)
        ]

    def _real_call(self, prompt: str, role: str) -> str:
        """Make a real inference call via the model server.

        Args:
            prompt: The full prompt.
            role: The role determining which model to use.

        Returns:
            Model response.

        Raises:
            RuntimeError: If model server not configured.
        """
        if self.model_server is None:
            raise RuntimeError("ModelServer not configured for real inference")

        # Use the model server's infer method
        from src.model_server import InferenceRequest

        request = InferenceRequest(
            role=role,
            prompt=prompt,
            timeout=self.config.call_timeout,
        )
        result = self.model_server.infer(role, request)

        self.total_tokens_generated += result.tokens_generated
        return result.output

    def _real_batch(self, prompts: list[str], role: str) -> list[str]:
        """Make real inference calls in parallel.

        Args:
            prompts: List of prompts.
            role: The role determining which model to use.

        Returns:
            List of model responses in order.
        """
        if self.model_server is None:
            raise RuntimeError("ModelServer not configured for real inference")

        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=self.config.batch_parallelism) as executor:
            # Submit all calls
            future_to_idx = {
                executor.submit(self._real_call, prompt, role): i
                for i, prompt in enumerate(prompts)
            }

            # Collect results in order
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"[ERROR: {e}]"

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about LLM calls.

        Returns:
            Dict with call counts, token counts, etc.
        """
        return {
            "total_calls": self.total_calls,
            "total_batch_calls": self.total_batch_calls,
            "total_tokens_generated": self.total_tokens_generated,
            "call_log_size": len(self.call_log),
            "mock_mode": self.mock_mode,
        }

    def get_recent_calls(self, n: int = 10) -> list[CallLogEntry]:
        """Get the most recent call log entries.

        Args:
            n: Number of entries to return.

        Returns:
            List of recent CallLogEntry objects.
        """
        return self.call_log[-n:]

    def clear_log(self) -> None:
        """Clear the call log."""
        self.call_log.clear()

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.total_calls = 0
        self.total_batch_calls = 0
        self.total_tokens_generated = 0
        self.call_log.clear()

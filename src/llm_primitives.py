#!/usr/bin/env python3
"""LLM primitives for RLM-style orchestration.

This module provides llm_call() and llm_batch() functions for spawning
sub-LM calls from the Root LM. Supports both mock mode (for testing)
and real inference via persistent llama-server backends with RadixAttention
prefix caching.

Includes optional GenerationMonitor integration for early failure detection.
See research/early_failure_prediction.md for literature references.

Usage:
    from src.llm_primitives import LLMPrimitives

    # Mock mode (for testing)
    primitives = LLMPrimitives(mock_mode=True)
    result = primitives.llm_call("Summarize this:", "Some text")

    # Real mode (with server URLs - RadixAttention enabled)
    primitives = LLMPrimitives(
        mock_mode=False,
        server_urls={
            "frontdoor": "http://localhost:8080",
            "coder": "http://localhost:8081",
            "worker": "http://localhost:8082",
        }
    )
    result = primitives.llm_call("Summarize this:", "Some text", role="worker")

    # Legacy mode (with model server - per-inference subprocess)
    from src.model_server import ModelServer
    server = ModelServer()
    primitives = LLMPrimitives(model_server=server, mock_mode=False)
    result = primitives.llm_call("Summarize this:", "Some text", role="worker")

    # With generation monitoring for early abort
    from src.generation_monitor import GenerationMonitor, MonitorConfig
    monitor = GenerationMonitor(config=MonitorConfig.for_tier("worker"))
    result = primitives.llm_call_monitored("Solve this:", "", role="worker", monitor=monitor)
    if result.aborted:
        # Handle early abort - escalate to higher tier
        pass
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
class LLMResult:
    """Result of an LLM call with optional abort information.

    Used by llm_call_monitored when generation monitoring is enabled.

    Attributes:
        text: The generated text (may be partial if aborted).
        aborted: Whether generation was aborted early.
        abort_reason: Reason for abort (if aborted).
        tokens_generated: Number of tokens generated.
        tokens_saved: Estimated tokens saved by early abort.
        failure_probability: Estimated failure probability at abort.
        elapsed_seconds: Time taken for generation.
    """

    text: str
    aborted: bool = False
    abort_reason: str = ""
    tokens_generated: int = 0
    tokens_saved: int = 0
    failure_probability: float = 0.0
    elapsed_seconds: float = 0.0


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
    Can operate in mock mode for testing, with CachingBackend (RadixAttention),
    or with a legacy ModelServer.
    """

    # Default server URLs for orchestrator roles
    DEFAULT_SERVER_URLS = {
        "frontdoor": "http://localhost:8080",
        "coder": "http://localhost:8081",
        "worker": "http://localhost:8082",
    }

    def __init__(
        self,
        model_server: Any | None = None,
        mock_mode: bool = True,
        config: LLMPrimitivesConfig | None = None,
        mock_responses: dict[str, str] | None = None,
        server_urls: dict[str, str] | None = None,
        num_slots: int = 4,
    ):
        """Initialize LLM primitives.

        Args:
            model_server: Optional ModelServer instance for legacy inference.
            mock_mode: If True, return mock responses instead of real inference.
            config: Optional configuration for output caps, parallelism, etc.
            mock_responses: Optional dict mapping prompts to mock responses.
            server_urls: Dict mapping role names to llama-server URLs.
                        If provided and not mock_mode, uses CachingBackend.
            num_slots: Number of slots per server for prefix caching.
        """
        self.model_server = model_server
        self.mock_mode = mock_mode
        self.config = config if config is not None else LLMPrimitivesConfig()
        self.mock_responses = mock_responses if mock_responses is not None else {}
        self.server_urls = server_urls
        self.num_slots = num_slots

        # CachingBackend instances per role (RadixAttention)
        self._backends: dict[str, Any] = {}

        # Initialize backends if server URLs provided and not mock mode
        if not mock_mode and server_urls:
            self._init_caching_backends(server_urls, num_slots)

        # Call log for debugging and testing
        self.call_log: list[CallLogEntry] = []

        # Stats
        self.total_calls = 0
        self.total_batch_calls = 0
        self.total_tokens_generated = 0

    def _init_caching_backends(self, server_urls: dict[str, str], num_slots: int) -> None:
        """Initialize CachingBackend instances for each role.

        Args:
            server_urls: Dict mapping role names to llama-server URLs.
            num_slots: Number of slots per server.
        """
        try:
            from src.backends.llama_server import LlamaServerBackend, ServerConfig
            from src.prefix_cache import CachingBackend, PrefixRouter

            for role, url in server_urls.items():
                config = ServerConfig(base_url=url, num_slots=num_slots)
                backend = LlamaServerBackend(config)
                router = PrefixRouter(num_slots=num_slots)
                self._backends[role] = CachingBackend(backend, router)

        except ImportError as e:
            # If RadixAttention modules not available, log and continue
            import logging
            logging.warning(f"CachingBackend not available: {e}. Using legacy mode.")

    def get_backend(self, role: str) -> Any | None:
        """Get the CachingBackend for a role.

        Args:
            role: Role name (e.g., "worker", "coder", "frontdoor").

        Returns:
            CachingBackend instance or None if not configured.
        """
        return self._backends.get(role)

    def get_cache_stats(self) -> dict[str, dict[str, Any]]:
        """Get cache statistics for all backends.

        Returns:
            Dict mapping role to cache stats dict.
        """
        stats = {}
        for role, backend in self._backends.items():
            if hasattr(backend, "get_stats"):
                stats[role] = backend.get_stats()
        return stats

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

    def llm_call_monitored(
        self,
        prompt: str,
        context_slice: str = "",
        role: str = "worker",
        monitor: Any = None,
        expected_length: int | None = None,
    ) -> LLMResult:
        """Call a sub-LM with generation monitoring for early abort.

        This method integrates with GenerationMonitor to detect likely
        failures during generation and abort early to save compute.

        Args:
            prompt: The instruction/prompt for the sub-LM.
            context_slice: Optional context to include.
            role: Role determining which model to use.
            monitor: GenerationMonitor instance (required).
            expected_length: Expected output length (for runaway detection).

        Returns:
            LLMResult with text and abort information.

        Raises:
            ValueError: If monitor is None.
        """
        if monitor is None:
            raise ValueError("monitor required for llm_call_monitored")

        start_time = time.perf_counter()
        self.total_calls += 1

        # Build full prompt
        if context_slice:
            full_prompt = f"{prompt}\n\nContext:\n{context_slice}"
        else:
            full_prompt = prompt

        # Set expected length if provided
        if expected_length:
            monitor.expected_length = expected_length

        # Create log entry
        log_entry = CallLogEntry(
            timestamp=time.time(),
            call_type="call_monitored",
            prompt=prompt,
            context_slice=context_slice[:500] if context_slice else None,
            role=role,
        )

        try:
            if self.mock_mode:
                result = self._mock_call_monitored(full_prompt, role, monitor)
            else:
                result = self._real_call_monitored(full_prompt, role, monitor)

            log_entry.result = result.text[:500] if result.text else None
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

            result.elapsed_seconds = log_entry.elapsed_seconds
            return result

        except Exception as e:
            log_entry.error = str(e)
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)
            return LLMResult(
                text=f"[ERROR: {e}]",
                aborted=False,
                elapsed_seconds=log_entry.elapsed_seconds,
            )

    def _mock_call_monitored(
        self,
        prompt: str,
        role: str,
        monitor: Any,
    ) -> LLMResult:
        """Generate mock response with monitoring simulation.

        Args:
            prompt: The full prompt.
            role: The role being called.
            monitor: GenerationMonitor instance.

        Returns:
            LLMResult with mock text and abort status.
        """
        # Reset monitor for this generation
        monitor.reset()

        # Simulate token-by-token generation
        tokens = []
        max_tokens = 200  # Mock generation limit

        for i in range(max_tokens):
            # Simulate a token
            token_id = hash(prompt + str(i)) % 50000
            tokens.append(token_id)

            # Update monitor (in mock mode, it generates synthetic metrics)
            monitor.update(token_id, logits=None)

            # Check if we should abort
            should_abort, abort_reason = monitor.should_abort()
            if should_abort:
                health = monitor.get_health()
                return LLMResult(
                    text=f"{self.config.mock_response_prefix} Partial response (aborted at token {i})",
                    aborted=True,
                    abort_reason=abort_reason.value,
                    tokens_generated=i + 1,
                    tokens_saved=max_tokens - i - 1,
                    failure_probability=health.estimated_failure_prob,
                )

        # Completed without abort
        health = monitor.get_health()
        prompt_preview = prompt[:50].replace("\n", " ")
        return LLMResult(
            text=f"{self.config.mock_response_prefix} Response for role='{role}': {prompt_preview}...",
            aborted=False,
            tokens_generated=max_tokens,
            failure_probability=health.estimated_failure_prob,
        )

    def _real_call_monitored(
        self,
        prompt: str,
        role: str,
        monitor: Any,
    ) -> LLMResult:
        """Make real inference call with monitoring.

        Args:
            prompt: The full prompt.
            role: The role determining which model to use.
            monitor: GenerationMonitor instance.

        Returns:
            LLMResult with text and abort status.

        Raises:
            RuntimeError: If model server not configured.
        """
        if self.model_server is None:
            raise RuntimeError("ModelServer not configured for real inference")

        # Reset monitor for this generation
        monitor.reset()

        # Use the model server's streaming infer method
        from src.model_server import InferenceRequest

        request = InferenceRequest(
            role=role,
            prompt=prompt,
            timeout=self.config.call_timeout,
            stream=True,  # Enable streaming for per-token monitoring
        )

        output_tokens = []
        for token_id, logits in self.model_server.infer_stream(role, request):
            output_tokens.append(token_id)

            # Update monitor with real logits
            monitor.update(token_id, logits)

            # Check if we should abort
            should_abort, abort_reason = monitor.should_abort()
            if should_abort:
                health = monitor.get_health()
                # Decode partial output
                partial_text = self.model_server.decode_tokens(output_tokens)
                return LLMResult(
                    text=partial_text,
                    aborted=True,
                    abort_reason=abort_reason.value,
                    tokens_generated=len(output_tokens),
                    tokens_saved=0,  # Unknown for real inference
                    failure_probability=health.estimated_failure_prob,
                )

        # Completed without abort
        health = monitor.get_health()
        full_text = self.model_server.decode_tokens(output_tokens)
        self.total_tokens_generated += len(output_tokens)

        return LLMResult(
            text=full_text,
            aborted=False,
            tokens_generated=len(output_tokens),
            failure_probability=health.estimated_failure_prob,
        )

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
        """Make a real inference call via CachingBackend or legacy ModelServer.

        Args:
            prompt: The full prompt.
            role: The role determining which model to use.

        Returns:
            Model response.

        Raises:
            RuntimeError: If no backend configured for this role.
        """
        # Try CachingBackend first (RadixAttention)
        backend = self._backends.get(role)
        if backend is not None:
            return self._call_caching_backend(backend, prompt, role)

        # Fall back to legacy ModelServer
        if self.model_server is None:
            raise RuntimeError(
                f"No backend configured for role '{role}'. "
                "Provide server_urls or model_server."
            )

        from src.model_server import InferenceRequest

        request = InferenceRequest(
            role=role,
            prompt=prompt,
            timeout=self.config.call_timeout,
        )
        result = self.model_server.infer(role, request)

        self.total_tokens_generated += result.tokens_generated
        return result.output

    def _call_caching_backend(self, backend: Any, prompt: str, role: str) -> str:
        """Call a CachingBackend with RadixAttention prefix caching.

        Args:
            backend: CachingBackend instance.
            prompt: The full prompt.
            role: The role name.

        Returns:
            Model response.
        """
        from src.model_server import InferenceRequest
        from src.registry_loader import RoleConfig, AccelerationConfig

        # Create minimal RoleConfig for backend
        role_config = RoleConfig(
            name=role,
            model_path="",  # Backend already knows the model
            acceleration=AccelerationConfig(type="baseline"),
        )

        request = InferenceRequest(
            role=role,
            prompt=prompt,
            timeout=self.config.call_timeout,
        )

        result = backend.infer(role_config, request)

        if not result.success:
            raise RuntimeError(f"Inference failed: {result.error_message}")

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
        # Check if we have a backend for this role
        backend = self._backends.get(role)
        if backend is None and self.model_server is None:
            raise RuntimeError(
                f"No backend configured for role '{role}'. "
                "Provide server_urls or model_server."
            )

        results: list[str | None] = [None] * len(prompts)

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

        return [r if r is not None else "" for r in results]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about LLM calls.

        Returns:
            Dict with call counts, token counts, cache stats, etc.
        """
        stats = {
            "total_calls": self.total_calls,
            "total_batch_calls": self.total_batch_calls,
            "total_tokens_generated": self.total_tokens_generated,
            "call_log_size": len(self.call_log),
            "mock_mode": self.mock_mode,
        }

        # Add cache stats if using CachingBackend
        if self._backends:
            cache_stats = self.get_cache_stats()
            stats["cache_stats"] = cache_stats

            # Calculate aggregate hit rate
            total_routes = 0
            total_hits = 0
            for role_stats in cache_stats.values():
                total_routes += role_stats.get("router_total_routes", 0)
                total_hits += role_stats.get("router_hit_rate", 0) * role_stats.get("router_total_routes", 0)
            if total_routes > 0:
                stats["aggregate_cache_hit_rate"] = total_hits / total_routes

        return stats

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

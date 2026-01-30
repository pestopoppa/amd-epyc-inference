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

import asyncio
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
class QueryCost:
    """Cost tracking for a single query/task.

    Tracks tokens used and estimates cost based on model rates.
    """

    query_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls_made: int = 0
    batch_calls_made: int = 0
    elapsed_seconds: float = 0.0

    # Cost estimation (per 1M tokens, configurable)
    prompt_rate: float = 0.0  # $ per 1M prompt tokens
    completion_rate: float = 0.0  # $ per 1M completion tokens

    @property
    def estimated_cost(self) -> float:
        """Estimate cost in dollars based on token counts and rates."""
        prompt_cost = (self.prompt_tokens / 1_000_000) * self.prompt_rate
        completion_cost = (self.completion_tokens / 1_000_000) * self.completion_rate
        return prompt_cost + completion_cost

    def __str__(self) -> str:
        return (
            f"QueryCost(id={self.query_id}, tokens={self.total_tokens}, "
            f"calls={self.calls_made}, batch={self.batch_calls_made}, "
            f"cost=${self.estimated_cost:.6f})"
        )


@dataclass
class LLMPrimitivesConfig:
    """Configuration for LLM primitives."""

    output_cap: int = 8192  # Max chars per sub-LM output
    batch_parallelism: int = 4  # Max parallel calls in llm_batch
    call_timeout: int = 120  # Timeout per call in seconds
    mock_response_prefix: str = "[MOCK]"  # Prefix for mock responses
    max_recursion_depth: int = 5  # Maximum nesting depth for sub-LM calls
    # Cost estimation rates (per 1M tokens)
    default_prompt_rate: float = 0.50  # $ per 1M input tokens
    default_completion_rate: float = 1.50  # $ per 1M output tokens


class LLMPrimitives:
    """LLM primitives for sub-LM spawning.

    Provides llm_call() for single calls and llm_batch() for parallel calls.
    Can operate in mock mode for testing, with CachingBackend (RadixAttention),
    or with a legacy ModelServer.
    """

    # Default server URLs for orchestrator roles
    DEFAULT_SERVER_URLS = {
        "frontdoor": "http://localhost:8080",
        "coder_primary": "http://localhost:8080",
        "coder": "http://localhost:8081",
        "coder_escalation": "http://localhost:8081",
        # Worker pool servers (heterogeneous)
        "worker": "http://localhost:8082",
        "worker_general": "http://localhost:8082",
        "worker_explore": "http://localhost:8082",
        "worker_math": "http://localhost:8082",
        "worker_vision": "http://localhost:8086",
        "vision_escalation": "http://localhost:8087",
        "worker_code": "http://localhost:8092",
        "worker_fast": "http://localhost:8102",
        "worker_fast_1": "http://localhost:8102",
        "worker_fast_2": "http://localhost:8112",
        # Architects
        "architect_general": "http://localhost:8083",
        "architect_coding": "http://localhost:8084",
        "ingest_long_context": "http://localhost:8085",
    }

    # Task type to worker role mapping (for pool routing)
    WORKER_TASK_ROUTING = {
        "explore": "worker_explore",
        "summarize": "worker_explore",
        "understand": "worker_explore",
        "code": "worker_code",
        "code_impl": "worker_code",
        "refactor": "worker_code",
        "test_gen": "worker_code",
        "fast": "worker_fast",
        "boilerplate": "worker_fast",
        "transform": "worker_fast",
    }

    def __init__(
        self,
        model_server: Any | None = None,
        mock_mode: bool = True,
        config: LLMPrimitivesConfig | None = None,
        mock_responses: dict[str, str] | None = None,
        server_urls: dict[str, str] | None = None,
        num_slots: int = 4,
        registry: Any | None = None,
        worker_pool: Any | None = None,
        use_worker_pool: bool = False,
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
            registry: Optional RegistryLoader for role-based generation defaults.
            worker_pool: Optional WorkerPoolManager for worker role routing.
            use_worker_pool: If True and worker_pool provided, route worker calls through pool.
        """
        self.model_server = model_server
        self.mock_mode = mock_mode
        self.config = config if config is not None else LLMPrimitivesConfig()
        self.mock_responses = mock_responses if mock_responses is not None else {}
        self.server_urls = server_urls
        self.num_slots = num_slots
        self.registry = registry
        self.worker_pool = worker_pool
        self.use_worker_pool = use_worker_pool and worker_pool is not None

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

        # Recursion depth tracking
        self._recursion_depth = 0
        self._max_recursion_depth_reached = 0

        # Per-query cost tracking
        self._current_query: QueryCost | None = None
        self._completed_queries: list[QueryCost] = []

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

    def start_query(self, query_id: str) -> None:
        """Start tracking costs for a new query.

        Args:
            query_id: Unique identifier for this query (e.g., task_id).
        """
        # End any existing query first
        if self._current_query is not None:
            self.end_query()

        self._current_query = QueryCost(
            query_id=query_id,
            prompt_rate=self.config.default_prompt_rate,
            completion_rate=self.config.default_completion_rate,
        )

    def end_query(self) -> QueryCost | None:
        """End current query tracking and return the cost.

        Returns:
            QueryCost for the completed query, or None if no query active.
        """
        if self._current_query is None:
            return None

        query = self._current_query
        self._completed_queries.append(query)
        self._current_query = None
        return query

    def get_current_query_cost(self) -> QueryCost | None:
        """Get cost for the current query (if active).

        Returns:
            QueryCost for current query, or None if no query active.
        """
        return self._current_query

    def get_completed_queries(self, last_n: int | None = None) -> list[QueryCost]:
        """Get completed query costs.

        Args:
            last_n: Only return last N queries (None = all).

        Returns:
            List of QueryCost for completed queries.
        """
        if last_n is None:
            return list(self._completed_queries)
        return self._completed_queries[-last_n:]

    def get_total_cost(self) -> float:
        """Get total estimated cost for all completed queries.

        Returns:
            Total cost in dollars.
        """
        return sum(q.estimated_cost for q in self._completed_queries)

    def _estimate_prompt_tokens(self, prompt: str) -> int:
        """Estimate number of tokens in a prompt.

        Uses a simple heuristic of ~4 characters per token.
        More accurate tokenization would require the model's tokenizer.

        Args:
            prompt: The prompt text.

        Returns:
            Estimated token count.
        """
        return len(prompt) // 4

    def _estimate_completion_tokens(self, response: str) -> int:
        """Estimate number of tokens in a response.

        Args:
            response: The response text.

        Returns:
            Estimated token count.
        """
        return len(response) // 4

    def llm_call(
        self,
        prompt: str,
        context_slice: str = "",
        role: str = "worker",
        n_tokens: int | None = None,
        skip_suffix: bool = False,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Call a sub-LM with optional context slice.

        Args:
            prompt: The instruction/prompt for the sub-LM.
            context_slice: Optional context to include (appended to prompt).
            role: Role determining which model to use (e.g., "worker", "coder").
            n_tokens: Max tokens to generate. If None, uses role default from registry.
            skip_suffix: If True, skip the registry system_prompt_suffix for this role.
                Used in direct-answer mode where any suffix (like "Elaborate on
                specialist outputs") degrades instruction precision quality.
            stop_sequences: Optional list of stop sequences to halt generation early.

        Returns:
            Sub-LM response (capped at output_cap chars).

        Raises:
            RecursionError: If max recursion depth exceeded.
        """
        # Check recursion depth
        if self._recursion_depth >= self.config.max_recursion_depth:
            raise RecursionError(
                f"Maximum recursion depth ({self.config.max_recursion_depth}) exceeded. "
                f"Sub-LM calls cannot be nested more than {self.config.max_recursion_depth} levels deep."
            )

        self._recursion_depth += 1
        self._max_recursion_depth_reached = max(self._max_recursion_depth_reached, self._recursion_depth)

        try:
            return self._llm_call_impl(prompt, context_slice, role, n_tokens, skip_suffix, stop_sequences)
        finally:
            self._recursion_depth -= 1

    def _llm_call_impl(
        self,
        prompt: str,
        context_slice: str = "",
        role: str = "worker",
        n_tokens: int | None = None,
        skip_suffix: bool = False,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Internal implementation of llm_call (after recursion check)."""
        start_time = time.perf_counter()
        self.total_calls += 1

        # Get role defaults from registry if available
        system_prompt_suffix = None
        if self.registry and not skip_suffix:
            default_n, _default_temp, system_prompt_suffix = self.registry.get_role_defaults(role)
            if n_tokens is None:
                n_tokens = default_n
        elif self.registry:
            # Still get default n_tokens even when skipping suffix
            default_n, _default_temp, _ = self.registry.get_role_defaults(role)
            if n_tokens is None:
                n_tokens = default_n
        if n_tokens is None:
            n_tokens = 512  # Fallback default

        # Apply system prompt suffix if configured for this role
        if system_prompt_suffix:
            prompt = f"{prompt}\n\n{system_prompt_suffix}"

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
                result = self._real_call(full_prompt, role, n_tokens, stop_sequences)

            # Cap output
            if len(result) > self.config.output_cap:
                result = result[: self.config.output_cap] + f"\n[... truncated at {self.config.output_cap} chars]"

            log_entry.result = result[:500]  # Truncate for log
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

            # Track tokens for current query cost
            if self._current_query is not None:
                prompt_tokens = self._estimate_prompt_tokens(full_prompt)
                completion_tokens = self._estimate_completion_tokens(result)
                self._current_query.prompt_tokens += prompt_tokens
                self._current_query.completion_tokens += completion_tokens
                self._current_query.total_tokens += prompt_tokens + completion_tokens
                self._current_query.calls_made += 1
                self._current_query.elapsed_seconds += log_entry.elapsed_seconds

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

            # Track tokens for current query cost
            if self._current_query is not None:
                total_prompt_tokens = sum(self._estimate_prompt_tokens(p) for p in prompts)
                total_completion_tokens = sum(self._estimate_completion_tokens(r) for r in capped_results)
                self._current_query.prompt_tokens += total_prompt_tokens
                self._current_query.completion_tokens += total_completion_tokens
                self._current_query.total_tokens += total_prompt_tokens + total_completion_tokens
                self._current_query.batch_calls_made += 1
                self._current_query.elapsed_seconds += log_entry.elapsed_seconds

            return capped_results

        except Exception as e:
            log_entry.error = str(e)
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)
            return [f"[ERROR: {e}]" for _ in prompts]

    async def llm_batch_async(
        self,
        prompts: list[str],
        role: str = "worker",
    ) -> list[str]:
        """Call multiple sub-LMs in parallel using asyncio.

        This is the async version of llm_batch() for use in async contexts.
        Uses asyncio.gather for parallel execution.

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
            call_type="batch_async",
            prompts=prompts[:5] if len(prompts) <= 5 else prompts[:5] + ["..."],
            role=role,
        )

        try:
            if self.mock_mode:
                # Mock mode: simulate async calls
                results = self._mock_batch(prompts, role)
            else:
                # Real mode: run calls in parallel using asyncio
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(None, self._real_call, prompt, role)
                    for prompt in prompts
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Convert exceptions to error strings
                results = [
                    str(r) if isinstance(r, Exception) else r
                    for r in results
                ]

            # Cap each output
            capped_results = []
            for result in results:
                if len(result) > self.config.output_cap:
                    result = result[: self.config.output_cap] + f"\n[... truncated at {self.config.output_cap} chars]"
                capped_results.append(result)

            log_entry.result = [r[:200] for r in capped_results[:3]]  # Truncate for log
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

            # Track tokens for current query cost
            if self._current_query is not None:
                total_prompt_tokens = sum(self._estimate_prompt_tokens(p) for p in prompts)
                total_completion_tokens = sum(self._estimate_completion_tokens(r) for r in capped_results)
                self._current_query.prompt_tokens += total_prompt_tokens
                self._current_query.completion_tokens += total_completion_tokens
                self._current_query.total_tokens += total_prompt_tokens + total_completion_tokens
                self._current_query.batch_calls_made += 1
                self._current_query.elapsed_seconds += log_entry.elapsed_seconds

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

    def _real_call(
        self, prompt: str, role: str, n_tokens: int = 512,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Make a real inference call via CachingBackend or legacy ModelServer.

        Args:
            prompt: The full prompt.
            role: The role determining which model to use.
            n_tokens: Maximum tokens to generate.
            stop_sequences: Optional stop sequences to halt generation.

        Returns:
            Model response.

        Raises:
            RuntimeError: If no backend configured for this role.
        """
        # Try CachingBackend first (RadixAttention)
        backend = self._backends.get(role)
        if backend is not None:
            return self._call_caching_backend(backend, prompt, role, n_tokens, stop_sequences)

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
            n_tokens=n_tokens,
            timeout=self.config.call_timeout,
            stop_sequences=stop_sequences,
        )
        result = self.model_server.infer(role, request)

        self.total_tokens_generated += result.tokens_generated
        return result.output

    def _call_caching_backend(
        self, backend: Any, prompt: str, role: str, n_tokens: int = 512,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Call a CachingBackend with RadixAttention prefix caching.

        Args:
            backend: CachingBackend instance.
            prompt: The full prompt.
            role: The role name.
            n_tokens: Maximum tokens to generate.
            stop_sequences: Optional stop sequences to halt generation.

        Returns:
            Model response.
        """
        from src.model_server import InferenceRequest
        from src.registry_loader import (
            RoleConfig,
            AccelerationConfig,
            ModelConfig,
            PerformanceMetrics,
            MemoryConfig,
        )

        # Create minimal RoleConfig for backend
        role_config = RoleConfig(
            name=role,
            tier="C",
            description=f"Dynamic role for {role}",
            model=ModelConfig(
                name="dynamic-model",
                path="",  # Backend already knows the model
                quant="Q4_K_M",
                size_gb=0.0,
            ),
            acceleration=AccelerationConfig(type="baseline", temperature=0.0),
            performance=PerformanceMetrics(),
            memory=MemoryConfig(residency="warm"),
        )

        request = InferenceRequest(
            role=role,
            prompt=prompt,
            n_tokens=n_tokens,
            timeout=self.config.call_timeout,
            stop_sequences=stop_sequences,
        )

        result = backend.infer(role_config, request)

        if not result.success:
            raise RuntimeError(f"Inference failed: {result.error_message}")

        self.total_tokens_generated += result.tokens_generated
        return result.output

    # Pool of fast worker roles for round-robin dispatch
    WORKER_FAST_POOL = ["worker_fast_1", "worker_fast_2"]
    _worker_fast_counter: int = 0

    def _real_batch(self, prompts: list[str], role: str) -> list[str]:
        """Make real inference calls in parallel.

        Args:
            prompts: List of prompts.
            role: The role determining which model to use.

        Returns:
            List of model responses in order.
        """
        # Use worker pool for worker roles if configured
        if self.use_worker_pool and role.startswith("worker"):
            return self._worker_pool_batch(prompts, role)

        # Check if we have a backend for this role
        backend = self._backends.get(role)
        if backend is None and self.model_server is None:
            raise RuntimeError(
                f"No backend configured for role '{role}'. "
                "Provide server_urls or model_server."
            )

        results: list[str | None] = [None] * len(prompts)

        # Round-robin dispatch for worker_fast across 8102+8112 pool
        use_fast_pool = (role == "worker_fast" and
                         any(self._backends.get(r) for r in self.WORKER_FAST_POOL))

        with ThreadPoolExecutor(max_workers=self.config.batch_parallelism) as executor:
            future_to_idx = {}
            for i, prompt in enumerate(prompts):
                if use_fast_pool:
                    # Distribute across fast worker pool (round-robin)
                    pool_role = self.WORKER_FAST_POOL[
                        LLMPrimitives._worker_fast_counter % len(self.WORKER_FAST_POOL)
                    ]
                    LLMPrimitives._worker_fast_counter += 1
                    future = executor.submit(self._real_call, prompt, pool_role)
                else:
                    future = executor.submit(self._real_call, prompt, role)
                future_to_idx[future] = i

            # Collect results in order
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"[ERROR: {e}]"

        return [r if r is not None else "" for r in results]

    def _worker_pool_batch(self, prompts: list[str], role: str) -> list[str]:
        """Execute batch using the heterogeneous worker pool.

        Routes to appropriate worker type based on role.

        Args:
            prompts: List of prompts.
            role: Worker role (determines task type routing).

        Returns:
            List of model responses in order.
        """
        import asyncio

        # Determine task type from role
        # worker_explore -> explore, worker_code -> code, etc.
        task_type = "explore"  # default
        if "_" in role:
            suffix = role.split("_", 1)[1]
            task_type = self.WORKER_TASK_ROUTING.get(suffix, suffix)

        try:
            # Run async batch in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                import nest_asyncio
                nest_asyncio.apply()
                results = loop.run_until_complete(
                    self.worker_pool.batch(prompts, task_type=task_type)
                )
            else:
                results = asyncio.run(
                    self.worker_pool.batch(prompts, task_type=task_type)
                )
            return results
        except Exception as e:
            # Fall back to standard batch if worker pool fails
            import logging
            logging.getLogger(__name__).warning(
                f"Worker pool batch failed, falling back to standard: {e}"
            )
            return self._fallback_batch(prompts, role)

    def _fallback_batch(self, prompts: list[str], role: str) -> list[str]:
        """Fallback batch implementation using ThreadPoolExecutor.

        Used when worker pool is unavailable or fails.
        """
        results: list[str | None] = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=self.config.batch_parallelism) as executor:
            future_to_idx = {
                executor.submit(self._real_call, prompt, role): i
                for i, prompt in enumerate(prompts)
            }

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
            "max_recursion_depth_reached": self._max_recursion_depth_reached,
            "current_recursion_depth": self._recursion_depth,
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
        self._max_recursion_depth_reached = 0
        # Note: _recursion_depth is not reset as it tracks current call stack

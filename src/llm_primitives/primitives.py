"""Main LLMPrimitives class that combines all mixins."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Any

from .backend import BackendMixin
from .config import LLMPrimitivesConfig
from .cost_tracking import CostTrackingMixin
from .inference import InferenceMixin
from .mock import MockMixin
from .persona import PersonaMixin
from .stats import StatsMixin
from .tokens import TokensMixin
from .types import CallLogEntry, LLMResult


class LLMPrimitives(
    BackendMixin,
    CostTrackingMixin,
    TokensMixin,
    PersonaMixin,
    MockMixin,
    InferenceMixin,
    StatsMixin,
):
    """LLM primitives for sub-LM spawning.

    Provides llm_call() for single calls and llm_batch() for parallel calls.
    Can operate in mock mode for testing, with CachingBackend (RadixAttention),
    or with a legacy ModelServer.
    """

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
        health_tracker: Any | None = None,
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
            health_tracker: Optional BackendHealthTracker for circuit breaker integration.
        """
        self.model_server = model_server
        self.mock_mode = mock_mode
        self.health_tracker = health_tracker
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
        # Clean timing accumulators from llama.cpp timings
        self.total_prompt_eval_ms = 0.0
        self.total_generation_ms = 0.0
        self._last_predicted_tps = 0.0  # Most recent call's clean t/s

        # Recursion depth tracking
        self._recursion_depth = 0
        self._max_recursion_depth_reached = 0

        # HTTP overhead tracking (server-side overhead not captured in inference timings)
        self.total_http_overhead_ms = 0.0

        # Per-request cache_prompt override (None = backend default)
        self.cache_prompt: bool | None = None

        # Per-query cost tracking
        self._current_query = None
        self._completed_queries: list = []

        # Concurrency policy (small workers only)
        from src.concurrency import get_role_max_concurrency

        self._role_limits = {
            role: get_role_max_concurrency(role) for role in (server_urls or {}).keys()
        }
        self._role_semaphores: dict[str, threading.Semaphore] = {}

    def _get_role_limit(self, role: str) -> int:
        """Return per-role concurrency limit (defaults to 1)."""
        from src.concurrency import get_role_max_concurrency

        return self._role_limits.get(role) or get_role_max_concurrency(role)

    @contextlib.contextmanager
    def _acquire_role(self, role: str):
        """Acquire per-role concurrency gate."""
        limit = self._get_role_limit(role)
        if limit <= 1:
            sem = self._role_semaphores.setdefault(role, threading.Semaphore(1))
        else:
            sem = self._role_semaphores.setdefault(role, threading.Semaphore(limit))
        sem.acquire()
        try:
            yield
        finally:
            sem.release()

    def llm_call(
        self,
        prompt: str,
        context_slice: str = "",
        role: str = "worker",
        n_tokens: int | None = None,
        skip_suffix: bool = False,
        stop_sequences: list[str] | None = None,
        persona: str | None = None,
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
            persona: Optional persona name (e.g., "security_auditor"). When set and
                the personas feature is enabled, injects the persona's system prompt
                before the user prompt.

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
        self._max_recursion_depth_reached = max(
            self._max_recursion_depth_reached, self._recursion_depth
        )

        try:
            return self._llm_call_impl(
                prompt, context_slice, role, n_tokens, skip_suffix, stop_sequences, persona
            )
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
        persona: str | None = None,
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

        # Inject persona system prompt if specified and feature enabled
        if persona and not skip_suffix:
            prompt = self._apply_persona_prefix(prompt, persona)

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
            persona=persona,
        )

        try:
            if self.mock_mode:
                result = self._mock_call(full_prompt, role)
            else:
                result = self._real_call(full_prompt, role, n_tokens, stop_sequences)

            # Cap output
            if len(result) > self.config.output_cap:
                result = (
                    result[: self.config.output_cap]
                    + f"\n[... truncated at {self.config.output_cap} chars]"
                )

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
        persona: str | None = None,
    ) -> list[str]:
        """Call multiple sub-LMs in parallel.

        Args:
            prompts: List of prompts to send to sub-LMs.
            role: Role determining which model to use.
            persona: Optional persona name for system prompt injection.

        Returns:
            List of responses in the same order as prompts.
        """
        start_time = time.perf_counter()
        self.total_batch_calls += 1

        # Apply persona prefix to all prompts if specified
        if persona:
            prompts = [self._apply_persona_prefix(p, persona) for p in prompts]

        # Create log entry
        log_entry = CallLogEntry(
            timestamp=time.time(),
            call_type="batch",
            prompts=prompts[:5] if len(prompts) <= 5 else prompts[:5] + ["..."],
            role=role,
            persona=persona,
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
                    result = (
                        result[: self.config.output_cap]
                        + f"\n[... truncated at {self.config.output_cap} chars]"
                    )
                capped_results.append(result)

            log_entry.result = [r[:200] for r in capped_results[:3]]  # Truncate for log
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

            # Track tokens for current query cost
            if self._current_query is not None:
                total_prompt_tokens = sum(self._estimate_prompt_tokens(p) for p in prompts)
                total_completion_tokens = sum(
                    self._estimate_completion_tokens(r) for r in capped_results
                )
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
        persona: str | None = None,
    ) -> list[str]:
        """Call multiple sub-LMs in parallel using asyncio.

        This is the async version of llm_batch() for use in async contexts.
        Uses asyncio.gather for parallel execution.

        Args:
            prompts: List of prompts to send to sub-LMs.
            role: Role determining which model to use.
            persona: Optional persona name for system prompt injection.

        Returns:
            List of responses in the same order as prompts.
        """
        start_time = time.perf_counter()
        self.total_batch_calls += 1

        # Apply persona prefix to all prompts if specified
        if persona:
            prompts = [self._apply_persona_prefix(p, persona) for p in prompts]

        # Create log entry
        log_entry = CallLogEntry(
            timestamp=time.time(),
            call_type="batch_async",
            prompts=prompts[:5] if len(prompts) <= 5 else prompts[:5] + ["..."],
            role=role,
            persona=persona,
        )

        try:
            if self.mock_mode:
                # Mock mode: simulate async calls
                results = self._mock_batch(prompts, role)
            else:
                role_limit = self._get_role_limit(role)
                if role_limit <= 1:
                    results = []
                    for prompt in prompts:
                        results.append(
                            await asyncio.to_thread(self._real_call, prompt, role)
                        )
                else:
                    # Real mode: run calls in parallel using asyncio
                    loop = asyncio.get_event_loop()
                    tasks = [
                        loop.run_in_executor(None, self._real_call, prompt, role)
                        for prompt in prompts
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    # Convert exceptions to error strings
                    results = [str(r) if isinstance(r, Exception) else r for r in results]

            # Cap each output
            capped_results = []
            for result in results:
                if len(result) > self.config.output_cap:
                    result = (
                        result[: self.config.output_cap]
                        + f"\n[... truncated at {self.config.output_cap} chars]"
                    )
                capped_results.append(result)

            log_entry.result = [r[:200] for r in capped_results[:3]]  # Truncate for log
            log_entry.elapsed_seconds = time.perf_counter() - start_time
            self.call_log.append(log_entry)

            # Track tokens for current query cost
            if self._current_query is not None:
                total_prompt_tokens = sum(self._estimate_prompt_tokens(p) for p in prompts)
                total_completion_tokens = sum(
                    self._estimate_completion_tokens(r) for r in capped_results
                )
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
                with self._acquire_role(role):
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

"""Real inference methods using backends."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .types import LLMResult


class InferenceMixin:
    """Mixin for real inference methods."""

    def _real_call(
        self,
        prompt: str,
        role: str,
        n_tokens: int = 512,
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
        acquire = getattr(self, "_acquire_role", None)
        if acquire:
            with acquire(role):
                return self._real_call_impl(prompt, role, n_tokens, stop_sequences)
        return self._real_call_impl(prompt, role, n_tokens, stop_sequences)

    def _real_call_impl(
        self,
        prompt: str,
        role: str,
        n_tokens: int = 512,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Internal real call implementation (no concurrency gating)."""
        # Content-addressable cache check
        from src.features import features as _get_features

        cache = getattr(self, "_content_cache", None)
        cache_key = None
        if cache is not None and _get_features().content_cache and stop_sequences is None:
            from src.llm_cache import ContentAddressableCache

            cache_key = ContentAddressableCache.make_key(
                prompt, role, n_tokens,
                model_hash=getattr(self, "_model_hash", ""),
            )
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        result = None
        try:
            result = self._real_call_single(prompt, role, n_tokens, stop_sequences)
        except RuntimeError as primary_error:
            # Model fallback: try same-tier alternatives on infrastructure failure
            if not _get_features().model_fallback:
                raise

            from src.roles import get_fallback_roles

            fallback_roles = get_fallback_roles(role)
            if not fallback_roles:
                raise

            # Classify the failure
            reason = "unknown"
            if self.health_tracker:
                reason = self.health_tracker.classify_failure(primary_error)

            log = logging.getLogger(__name__)
            for fallback_role in fallback_roles:
                fb_role_str = str(fallback_role)
                log.warning(
                    "Model fallback: %s → %s (reason: %s)",
                    role, fb_role_str, reason,
                )
                try:
                    result = self._real_call_single(
                        prompt, fb_role_str, n_tokens, stop_sequences
                    )
                    break
                except RuntimeError:
                    continue
            else:
                # All fallbacks failed
                raise

        # Store in content cache on success
        if result is not None and cache_key is not None and cache is not None:
            cache.put(cache_key, result, metadata={"role": role})

        return result

    def _real_call_single(
        self,
        prompt: str,
        role: str,
        n_tokens: int = 512,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Execute a single inference call against one role's backend."""
        # Try CachingBackend first (RadixAttention)
        backend = self._backends.get(role)
        if backend is not None:
            return self._call_caching_backend(backend, prompt, role, n_tokens, stop_sequences)

        # Fall back to legacy ModelServer
        if self.model_server is None:
            raise RuntimeError(
                f"No backend configured for role '{role}'. Provide server_urls or model_server."
            )

        from src.model_server import InferenceRequest
        from src.config import get_config

        role_timeout = get_config().timeouts.role_timeouts_dict().get(
            role, self.config.call_timeout
        )

        request = InferenceRequest(
            role=role,
            prompt=prompt,
            n_tokens=n_tokens,
            timeout=role_timeout,
            stop_sequences=stop_sequences,
            cache_prompt=self.cache_prompt,
        )
        from src.inference_lock import inference_lock

        with inference_lock(role):
            result = self.model_server.infer(role, request)

        self.total_tokens_generated += result.tokens_generated
        self.total_prompt_eval_ms += result.prompt_eval_ms
        self.total_generation_ms += result.generation_ms
        if result.predicted_per_second > 0:
            self._last_predicted_tps = result.predicted_per_second
        return result.output

    def _call_caching_backend(
        self,
        backend: Any,
        prompt: str,
        role: str,
        n_tokens: int = 512,
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

        from src.config import get_config

        role_timeout = get_config().timeouts.role_timeouts_dict().get(
            role, self.config.call_timeout
        )

        request = InferenceRequest(
            role=role,
            prompt=prompt,
            n_tokens=n_tokens,
            timeout=role_timeout,
            stop_sequences=stop_sequences,
            cache_prompt=self.cache_prompt,
        )

        # Circuit breaker: fast-fail if backend is known to be down
        backend_url = self.server_urls.get(role, "") if self.server_urls else ""
        if backend_url and self.health_tracker:
            if not self.health_tracker.is_available(backend_url):
                raise RuntimeError(f"Backend unavailable (circuit open): {backend_url}")

        from src.inference_lock import inference_lock

        with inference_lock(role):
            from src.inference_tap import is_active as _tap_active

            if _tap_active() and hasattr(backend, "infer_stream_text"):
                from src.inference_tap import tap_section

                with tap_section(role, prompt, n_tokens) as tap:
                    result = backend.infer_stream_text(
                        role_config, request, on_chunk=tap.write_chunk
                    )
                    tap.write_timings(
                        result.tokens_generated,
                        result.prompt_eval_ms,
                        result.generation_ms,
                        result.predicted_per_second,
                    )
            else:
                result = backend.infer(role_config, request)

        # Record success/failure for circuit breaker
        if backend_url and self.health_tracker:
            if result.success:
                self.health_tracker.record_success(backend_url)
            else:
                self.health_tracker.record_failure(backend_url)

        if not result.success:
            raise RuntimeError(f"Inference failed: {result.error_message}")

        self.total_tokens_generated += result.tokens_generated
        self.total_prompt_eval_ms += result.prompt_eval_ms
        self.total_generation_ms += result.generation_ms
        self.total_http_overhead_ms += result.http_overhead_ms
        if result.predicted_per_second > 0:
            self._last_predicted_tps = result.predicted_per_second
        return result.output

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
                f"No backend configured for role '{role}'. Provide server_urls or model_server."
            )

        role_limit = getattr(self, "_get_role_limit", lambda _r: self.config.batch_parallelism)(
            role
        )
        if role_limit <= 1:
            results = []
            for prompt in prompts:
                try:
                    results.append(self._real_call(prompt, role))
                except Exception as e:
                    results.append(f"[ERROR: {e}]")
            return results

        results: list[str | None] = [None] * len(prompts)
        max_workers = min(self.config.batch_parallelism, role_limit)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for i, prompt in enumerate(prompts):
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
                results = asyncio.run(self.worker_pool.batch(prompts, task_type=task_type))
            return results
        except Exception as e:
            # Fall back to standard batch if worker pool fails
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
        from src.inference_lock import inference_lock

        with inference_lock(role):
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

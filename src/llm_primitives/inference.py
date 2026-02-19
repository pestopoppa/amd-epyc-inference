"""Real inference methods using backends."""

import asyncio
import contextvars
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .types import LLMResult

log = logging.getLogger(__name__)


def _frontdoor_trace_enabled() -> bool:
    raw = os.environ.get("ORCHESTRATOR_FRONTDOOR_TRACE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_frontdoor_role(role: str) -> bool:
    norm = (role or "").strip().lower()
    return norm == "frontdoor" or norm.endswith(".frontdoor")


def _detect_streaming_repetition(text: str, min_block: int = 60, min_repeats: int = 3) -> bool:
    """Detect paragraph-level repetition in streaming text.

    Scans the most recent output for any line-aligned block of >= *min_block*
    chars that appears *min_repeats* or more times.  Designed to catch
    degenerate generation loops where the model repeats the same paragraph.

    Only called periodically (every ~50 chunks) to avoid overhead.
    """
    if len(text) < min_block * min_repeats:
        return False
    # Work on a reasonable tail to limit cost.
    tail = text[-4000:] if len(text) > 4000 else text
    # Split into lines and check line-aligned blocks.
    lines = tail.split("\n")
    if len(lines) < 6:
        return False
    # Slide a window of 3 consecutive lines over the tail.  If the same
    # 3-line block appears min_repeats times, it's a repetition loop.
    for i in range(len(lines) - 2):
        block = "\n".join(lines[i : i + 3])
        if len(block) < min_block or not block.strip():
            continue
        if tail.count(block) >= min_repeats:
            return True
    return False


class InferenceMixin:
    """Mixin for real inference methods."""

    def _real_call(
        self,
        prompt: str,
        role: str,
        n_tokens: int = -1,
        stop_sequences: list[str] | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
    ) -> str:
        """Make a real inference call via CachingBackend or legacy ModelServer.

        Args:
            prompt: The full prompt.
            role: The role determining which model to use.
            n_tokens: Maximum tokens to generate.
            stop_sequences: Optional stop sequences to halt generation.
            json_schema: Optional JSON schema to constrain output structure.
            grammar: Optional GBNF grammar for constrained generation.

        Returns:
            Model response.

        Raises:
            RuntimeError: If no backend configured for this role.
        """
        acquire = getattr(self, "_acquire_role", None)
        if acquire:
            with acquire(role):
                return self._real_call_impl(
                    prompt, role, n_tokens, stop_sequences,
                    json_schema=json_schema, grammar=grammar,
                )
        return self._real_call_impl(
            prompt, role, n_tokens, stop_sequences,
            json_schema=json_schema, grammar=grammar,
        )

    def _real_call_impl(
        self,
        prompt: str,
        role: str,
        n_tokens: int = -1,
        stop_sequences: list[str] | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
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
            result = self._real_call_single(
                prompt, role, n_tokens, stop_sequences,
                json_schema=json_schema, grammar=grammar,
            )
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
                        prompt, fb_role_str, n_tokens, stop_sequences,
                        json_schema=json_schema, grammar=grammar,
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
        n_tokens: int = -1,
        stop_sequences: list[str] | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
    ) -> str:
        """Execute a single inference call against one role's backend."""
        # Try CachingBackend first (RadixAttention)
        backend = self._backends.get(role)
        if backend is not None:
            return self._call_caching_backend(
                backend, prompt, role, n_tokens, stop_sequences,
                json_schema=json_schema, grammar=grammar,
            )

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
        role_timeout = self._clamp_timeout_to_request_budget(role_timeout)

        request = InferenceRequest(
            role=role,
            prompt=prompt,
            n_tokens=n_tokens,
            timeout=role_timeout,
            stop_sequences=stop_sequences,
            cache_prompt=self.cache_prompt,
            json_schema=json_schema,
            grammar=grammar,
        )
        req_started = time.perf_counter()
        from src.inference_lock import inference_lock

        try:
            with inference_lock(
                role,
                cancel_check=self.get_request_cancel_check(),
                deadline_s=self.get_request_deadline_s(),
                request_tag=self.get_request_task_id(),
            ):
                request.timeout = self._clamp_timeout_to_request_budget(request.timeout)
                result = self.model_server.infer(role, request)
        except Exception as exc:
            req_elapsed_ms = (time.perf_counter() - req_started) * 1000
            self._last_inference_meta = {
                "role": role,
                "transport": "model_server",
                "elapsed_ms": req_elapsed_ms,
                "completion_reason": "exception",
                "error": str(exc),
            }
            if "lock timeout" in str(exc).lower() or "cancelled" in str(exc).lower():
                log.warning(
                    "Inference aborted before model call (role=%s, transport=model_server, elapsed_ms=%.1f): %s",
                    role,
                    req_elapsed_ms,
                    exc,
                )
            raise

        req_elapsed_ms = (time.perf_counter() - req_started) * 1000
        self._last_inference_meta = {
            "role": role,
            "transport": "model_server",
            "elapsed_ms": req_elapsed_ms,
            "first_token_ms": getattr(result, "first_token_ms", 0.0),
            "stream_chunks": getattr(result, "stream_chunks", 0),
            "completion_reason": getattr(result, "completion_reason", "") or "unknown",
            "tokens": result.tokens_generated,
            "prompt_ms": result.prompt_eval_ms,
            "gen_ms": result.generation_ms,
            "overhead_ms": result.http_overhead_ms,
        }
        if _is_frontdoor_role(role) and _frontdoor_trace_enabled():
            log.warning(
                "Frontdoor inference telemetry: transport=model_server elapsed_ms=%.1f "
                "first_token_ms=%.1f chunks=%d completion_reason=%s "
                "tokens=%d prompt_ms=%.1f gen_ms=%.1f overhead_ms=%.1f",
                req_elapsed_ms,
                getattr(result, "first_token_ms", 0.0),
                getattr(result, "stream_chunks", 0),
                getattr(result, "completion_reason", "") or "unknown",
                result.tokens_generated,
                result.prompt_eval_ms,
                result.generation_ms,
                result.http_overhead_ms,
            )

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
        n_tokens: int = -1,
        stop_sequences: list[str] | None = None,
        json_schema: dict | None = None,
        grammar: str | None = None,
    ) -> str:
        """Call a CachingBackend with RadixAttention prefix caching.

        Args:
            backend: CachingBackend instance.
            prompt: The full prompt.
            role: The role name.
            n_tokens: Maximum tokens to generate.
            stop_sequences: Optional stop sequences to halt generation.
            json_schema: Optional JSON schema to constrain output structure.
            grammar: Optional GBNF grammar for constrained generation.

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
        role_timeout = self._clamp_timeout_to_request_budget(role_timeout)

        request = InferenceRequest(
            role=role,
            prompt=prompt,
            n_tokens=n_tokens,
            timeout=role_timeout,
            stop_sequences=stop_sequences,
            cache_prompt=self.cache_prompt,
            json_schema=json_schema,
            grammar=grammar,
        )
        req_started = time.perf_counter()

        # Admission control: reject early if backend queue is full
        backend_url = self.server_urls.get(role, "") if self.server_urls else ""
        admission = getattr(self, "admission_controller", None)
        admitted = False
        if backend_url and admission:
            if not admission.try_acquire(backend_url):
                raise RuntimeError(
                    f"[ERROR: admission] Backend queue full for {backend_url}"
                )
            admitted = True

        try:
            # Circuit breaker: fast-fail if backend is known to be down
            if backend_url and self.health_tracker:
                if not self.health_tracker.is_available(backend_url):
                    raise RuntimeError(f"Backend unavailable (circuit open): {backend_url}")

            from src.inference_lock import inference_lock
            can_stream = False

            try:
                with inference_lock(
                    role,
                    cancel_check=self.get_request_cancel_check(),
                    deadline_s=self.get_request_deadline_s(),
                    request_tag=self.get_request_task_id(),
                ):
                    request.timeout = self._clamp_timeout_to_request_budget(request.timeout)
                    from src.inference_tap import (
                        is_active as _tap_active,
                        should_stream_role as _tap_should_stream_role,
                        tap_section,
                    )

                    tap_enabled = _tap_active()
                    can_stream = (
                        tap_enabled
                        and hasattr(backend, "infer_stream_text")
                        and _tap_should_stream_role(role)
                    )
                    if tap_enabled:
                        with tap_section(role, prompt) as tap:
                            if can_stream:
                                # Early-stop: if _early_stop_check is set, wrap the
                                # tap callback to also check accumulated output and
                                # raise StopIteration to abort streaming.
                                _stop_check = getattr(self, "_early_stop_check", None)
                                # Always accumulate for repetition detection,
                                # even when no _early_stop_check is set.
                                _acc: list[str] = []
                                _chunk_count = 0
                                _REP_CHECK_INTERVAL = 50  # check every ~50 chunks

                                def _on_chunk_guarded(content: str) -> None:
                                    nonlocal _chunk_count
                                    tap.write_chunk(content)
                                    _acc.append(content)
                                    _chunk_count += 1
                                    # Caller-provided early-stop (FINAL, TOON, etc.)
                                    if _stop_check is not None and _stop_check("".join(_acc)):
                                        raise StopIteration
                                    # Repetition guard: check periodically
                                    if _chunk_count % _REP_CHECK_INTERVAL == 0:
                                        if _detect_streaming_repetition("".join(_acc)):
                                            log.warning(
                                                "Repetition loop detected after %d chunks, "
                                                "aborting generation",
                                                _chunk_count,
                                            )
                                            raise StopIteration

                                result = backend.infer_stream_text(
                                    role_config, request, on_chunk=_on_chunk_guarded
                                )
                            else:
                                result = backend.infer(role_config, request)
                                tap.write_response(result.output)
                            tap.write_timings(
                                result.tokens_generated,
                                result.prompt_eval_ms,
                                result.generation_ms,
                                result.predicted_per_second,
                            )
                    else:
                        result = backend.infer(role_config, request)
            except Exception as exc:
                req_elapsed_ms = (time.perf_counter() - req_started) * 1000
                transport = "stream" if can_stream else "batch"
                self._last_inference_meta = {
                    "role": role,
                    "transport": transport,
                    "elapsed_ms": req_elapsed_ms,
                    "completion_reason": "exception",
                    "error": str(exc),
                }
                if "lock timeout" in str(exc).lower() or "cancelled" in str(exc).lower():
                    log.warning(
                        "Inference aborted before backend response (role=%s, transport=%s, elapsed_ms=%.1f): %s",
                        role,
                        transport,
                        req_elapsed_ms,
                        exc,
                    )
                raise

            req_elapsed_ms = (time.perf_counter() - req_started) * 1000
            transport = "stream" if can_stream else "batch"
            self._last_inference_meta = {
                "role": role,
                "transport": transport,
                "elapsed_ms": req_elapsed_ms,
                "first_token_ms": getattr(result, "first_token_ms", 0.0),
                "stream_chunks": getattr(result, "stream_chunks", 0),
                "completion_reason": getattr(result, "completion_reason", "") or "unknown",
                "tokens": result.tokens_generated,
                "prompt_ms": result.prompt_eval_ms,
                "gen_ms": result.generation_ms,
                "overhead_ms": result.http_overhead_ms,
            }
            if _is_frontdoor_role(role) and _frontdoor_trace_enabled():
                log.warning(
                    "Frontdoor inference telemetry: transport=%s elapsed_ms=%.1f "
                    "first_token_ms=%.1f chunks=%d completion_reason=%s "
                    "tokens=%d prompt_ms=%.1f gen_ms=%.1f overhead_ms=%.1f",
                    transport,
                    req_elapsed_ms,
                    getattr(result, "first_token_ms", 0.0),
                    getattr(result, "stream_chunks", 0),
                    getattr(result, "completion_reason", "") or "unknown",
                    result.tokens_generated,
                    result.prompt_eval_ms,
                    result.generation_ms,
                    result.http_overhead_ms,
                )

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
        finally:
            if admitted and admission:
                admission.release(backend_url)

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
                ctx = contextvars.copy_context()
                future = executor.submit(ctx.run, self._real_call, prompt, role)
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
            timeout_s = self._remaining_deadline_s()
            batch_coro = self.worker_pool.batch(prompts, task_type=task_type)
            if timeout_s is not None:
                timeout_s = max(1.0, timeout_s)
                batch_coro = asyncio.wait_for(batch_coro, timeout=timeout_s)
                self._budget_diagnostics["budget_applied"] = True
                self._budget_diagnostics["timeout_clamp_events"] += 1
            if loop.is_running():
                # If we're already in an async context, create a new task
                import nest_asyncio

                nest_asyncio.apply()
                results = loop.run_until_complete(batch_coro)
            else:
                results = asyncio.run(batch_coro)
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
                executor.submit(contextvars.copy_context().run, self._real_call, prompt, role): i
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
        request.timeout = self._clamp_timeout_to_request_budget(request.timeout)

        output_tokens = []
        from src.inference_lock import inference_lock

        with inference_lock(
            role,
            cancel_check=self.get_request_cancel_check(),
            deadline_s=self.get_request_deadline_s(),
            request_tag=self.get_request_task_id(),
        ):
            request.timeout = self._clamp_timeout_to_request_budget(request.timeout)
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

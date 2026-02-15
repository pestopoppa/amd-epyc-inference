#!/usr/bin/env python3
"""LlamaServerBackend - Persistent HTTP server mode for llama.cpp inference.

This backend connects to a running llama-server instance instead of spawning
per-inference subprocesses. Enables:
- KV cache reuse across requests (prefix caching)
- Lower latency (no model load per inference)
- Multi-slot parallel processing

Usage:
    from src.backends.llama_server import LlamaServerBackend

    backend = LlamaServerBackend(base_url="http://localhost:8080")
    result = backend.infer(role_config, request)

See research/radix_attention_handoff.md for implementation plan.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

from src.model_server import InferenceRequest, InferenceResult, ModelBackend
from src.registry_loader import RoleConfig

logger = logging.getLogger(__name__)


def _server_cfg():
    from src.config import get_config

    return get_config().server


@dataclass
class ServerConfig:
    """Configuration for a llama-server instance."""

    base_url: str = field(default_factory=lambda: _server_cfg().default_url)
    timeout: int = field(default_factory=lambda: _server_cfg().timeout)
    num_slots: int = field(default_factory=lambda: _server_cfg().num_slots)
    connect_timeout: int = field(default_factory=lambda: _server_cfg().connect_timeout)
    retry_count: int = field(default_factory=lambda: _server_cfg().retry_count)
    retry_backoff: float = field(default_factory=lambda: _server_cfg().retry_backoff)


@dataclass
class SlotInfo:
    """Information about a server slot."""

    slot_id: int
    state: str  # "idle", "processing"
    prompt_tokens: int = 0
    cache_tokens: int = 0  # Tokens served from cache
    last_prompt_hash: str = ""
    last_access: float = 0.0


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_prompt_tokens: int = 0
    cached_prompt_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100

    @property
    def token_savings_rate(self) -> float:
        """Percentage of prompt tokens served from cache."""
        if self.total_prompt_tokens == 0:
            return 0.0
        return (self.cached_prompt_tokens / self.total_prompt_tokens) * 100


class LlamaServerError(Exception):
    """Error communicating with llama-server."""

    pass


class LlamaServerBackend(ModelBackend):
    """Backend using persistent llama-server HTTP API.

    This backend connects to a running llama-server instance and uses its
    HTTP API for inference. Enables KV cache reuse for prefix caching.

    Key features:
    - cache_prompt=True enables automatic prefix caching
    - id_slot parameter enables sticky slot routing for cache hits
    - Slot state management for optimal cache utilization

    Attributes:
        config: Server configuration.
        session: HTTP session with retry logic.
        slots: Current slot information.
        cache_stats: Cache performance statistics.
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        base_url: str = "http://localhost:8080",
    ):
        """Initialize the llama-server backend.

        Args:
            config: Server configuration. If None, creates default config.
            base_url: Base URL for the server (used if config is None).
        """
        if config is not None:
            self.config = config
        else:
            self.config = ServerConfig(base_url=base_url)

        # Create httpx client with connection pooling for ~6x latency reduction
        # Connection pool keeps persistent connections to reduce per-request overhead
        self.client = httpx.Client(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(
                connect=self.config.connect_timeout,
                read=self.config.timeout,
                write=self.config.timeout,
                pool=self.config.timeout,
            ),
            limits=httpx.Limits(
                max_connections=20,  # Total connections in pool
                max_keepalive_connections=10,  # Persistent idle connections
                keepalive_expiry=60.0,  # Seconds before closing idle connection
            ),
            transport=httpx.HTTPTransport(retries=self.config.retry_count),
        )

        # Slot tracking
        self.slots: dict[int, SlotInfo] = {}

        # Cache statistics
        self.cache_stats = CacheStats()

        # Server health
        self._healthy = False
        self._last_health_check = 0.0

    def load(self, role_config: RoleConfig) -> int:
        """Verify server is running and healthy.

        In server mode, models are pre-loaded. This method validates connectivity.

        Args:
            role_config: Role configuration (used for logging).

        Returns:
            0 (server mode doesn't track PIDs).

        Raises:
            LlamaServerError: If server is not reachable.
        """
        if not self.health_check(0):
            raise LlamaServerError(
                f"Cannot reach llama-server at {self.config.base_url}. "
                "Start the server with: llama-server -m MODEL.gguf --host 0.0.0.0 --port 8080"
            )

        logger.info(f"Connected to llama-server for role {role_config.name}")
        return 0

    def unload(self, pid: int) -> bool:
        """No-op in server mode (model stays loaded).

        Args:
            pid: Ignored in server mode.

        Returns:
            True (always succeeds).
        """
        return True

    def health_check(self, pid: int) -> bool:
        """Check if the llama-server is healthy.

        Args:
            pid: Ignored in server mode.

        Returns:
            True if server is reachable and healthy.
        """
        # Rate limit health checks
        now = time.time()
        if now - self._last_health_check < 1.0 and self._healthy:
            return self._healthy

        try:
            response = self.client.get(
                "/health",
                timeout=self.config.connect_timeout,
            )
            self._healthy = response.status_code == 200
            self._last_health_check = now
            return self._healthy
        except httpx.RequestError:
            self._healthy = False
            return False

    def infer(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Run inference via llama-server HTTP API.

        Args:
            role_config: Configuration for the role/model.
            request: Inference request parameters.

        Returns:
            InferenceResult with output and metrics.
        """
        start_time = time.time()
        self.cache_stats.total_requests += 1

        # Build request payload
        payload = self._build_payload(role_config, request)

        try:
            http_start = time.perf_counter()
            response = self.client.post(
                "/completion",
                json=payload,
                timeout=request.timeout or self.config.timeout,
            )
            http_elapsed_ms = (time.perf_counter() - http_start) * 1000
            response.raise_for_status()
            result_data = response.json()

            elapsed = time.time() - start_time

            # Extract metrics
            output = result_data.get("content", "")
            tokens_generated = result_data.get("tokens_predicted", 0)
            prompt_tokens = result_data.get("tokens_evaluated", 0)
            cached_tokens = result_data.get("tokens_cached", 0)

            # Extract clean timing data from llama.cpp timings object
            timings = result_data.get("timings", {})
            prompt_eval_ms = timings.get("prompt_ms", 0.0)
            generation_ms = timings.get("predicted_ms", 0.0)
            predicted_per_second = timings.get("predicted_per_second", 0.0)

            # Compute server-side overhead (HTTP round-trip minus reported inference)
            inference_ms = prompt_eval_ms + generation_ms
            http_overhead_ms = max(0.0, http_elapsed_ms - inference_ms)

            # Log overhead when significant (> 5s)
            if http_overhead_ms > 5000:
                logger.warning(
                    f"Server overhead: {http_overhead_ms:.0f}ms "
                    f"(HTTP={http_elapsed_ms:.0f}ms, inference={inference_ms:.0f}ms) "
                    f"for {role_config.name}"
                )

            # Update cache stats
            self.cache_stats.total_prompt_tokens += prompt_tokens
            self.cache_stats.cached_prompt_tokens += cached_tokens
            if cached_tokens > 0:
                self.cache_stats.cache_hits += 1
            else:
                self.cache_stats.cache_misses += 1

            # Use clean predicted_per_second if available, else fall back to elapsed
            speed = (
                predicted_per_second
                if predicted_per_second > 0
                else (tokens_generated / elapsed if elapsed > 0 else 0.0)
            )

            # Log cache performance
            if cached_tokens > 0:
                logger.debug(
                    f"Cache hit: {cached_tokens}/{prompt_tokens} tokens "
                    f"({100 * cached_tokens / prompt_tokens:.1f}%) from cache"
                )

            # Extract speculative decoding acceptance telemetry
            n_drafted = timings.get("drafted_n_tokens", 0)
            n_accepted = timings.get("drafted_n_accepted", 0)
            accept_rate = (n_accepted / n_drafted) if n_drafted > 0 else 0.0
            if n_drafted > 0:
                logger.info(
                    "Spec accept: %d/%d (%.1f%%) for %s",
                    n_accepted, n_drafted, accept_rate * 100, role_config.name,
                )

            return InferenceResult(
                role=role_config.name,
                output=output,
                tokens_generated=tokens_generated,
                generation_speed=speed,
                elapsed_time=elapsed,
                success=True,
                prompt_eval_ms=prompt_eval_ms,
                generation_ms=generation_ms,
                predicted_per_second=predicted_per_second,
                http_overhead_ms=http_overhead_ms,
                n_tokens_drafted=n_drafted,
                n_tokens_accepted=n_accepted,
                acceptance_rate=accept_rate,
            )

        except httpx.TimeoutException:
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=request.timeout or self.config.timeout,
                success=False,
                error_message=f"Request timed out after {request.timeout}s",
            )

        except httpx.RequestError as e:
            elapsed = time.time() - start_time
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=elapsed,
                success=False,
                error_message=f"Server request failed: {e}",
            )

    def infer_stream(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> Iterator[tuple[int, list[float] | None]]:
        """Stream inference tokens for monitoring support.

        Yields token IDs and optional logits for each generated token.
        Enables integration with GenerationMonitor for early abort.

        Args:
            role_config: Configuration for the role/model.
            request: Inference request parameters (must have stream=True).

        Yields:
            Tuple of (token_id, logits) for each token.
            logits may be None if not requested.
        """
        payload = self._build_payload(role_config, request)
        payload["stream"] = True

        try:
            with self.client.stream(
                "POST",
                "/completion",
                json=payload,
                timeout=request.timeout or self.config.timeout,
            ) as response:
                response.raise_for_status()

                # Parse streaming response (SSE format)
                for line in response.iter_lines():
                    if not line:
                        continue

                    line_str = line if isinstance(line, str) else line.decode("utf-8")
                    if not line_str.startswith("data: "):
                        continue

                    import json

                    try:
                        data = json.loads(line_str[6:])
                    except json.JSONDecodeError:
                        continue

                    # Extract token info
                    if "tokens" in data:
                        for token_id in data["tokens"]:
                            yield (token_id, None)
                    elif "content" in data:
                        # Single token mode - no direct ID available
                        # This is a limitation - ideally server returns token IDs
                        yield (0, None)

                    # Check for completion
                    if data.get("stop", False):
                        break

        except httpx.RequestError as e:
            logger.error(f"Stream error: {e}")
            return

    def infer_stream_text(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
        on_chunk: Any | None = None,
    ) -> InferenceResult:
        """Stream inference and return text result (with optional per-chunk callback).

        Like ``infer()`` but uses SSE streaming so that an ``on_chunk``
        callback can observe tokens as they arrive (e.g. for the
        inference tap).  The return value is identical to ``infer()``.

        Args:
            role_config: Configuration for the role/model.
            request: Inference request parameters.
            on_chunk: Optional callback ``(str) -> None`` called for each
                text chunk received from the server.

        Returns:
            InferenceResult with output and metrics (same shape as batch).
        """
        import json as _json

        start_time = time.time()
        self.cache_stats.total_requests += 1

        payload = self._build_payload(role_config, request)
        payload["stream"] = True

        try:
            http_start = time.perf_counter()

            with self.client.stream(
                "POST",
                "/completion",
                json=payload,
                timeout=request.timeout or self.config.timeout,
            ) as response:
                response.raise_for_status()

                chunks: list[str] = []
                timings: dict[str, Any] = {}
                tokens_generated = 0
                prompt_tokens = 0
                cached_tokens = 0

                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line if isinstance(line, str) else line.decode("utf-8")
                    if not line_str.startswith("data: "):
                        continue

                    try:
                        data = _json.loads(line_str[6:])
                    except _json.JSONDecodeError:
                        continue

                    content = data.get("content", "")
                    if content:
                        chunks.append(content)
                    early_stopped = False
                    if content and on_chunk is not None:
                        try:
                            on_chunk(content)
                        except StopIteration:
                            early_stopped = True
                    if early_stopped:
                        tokens_generated = len(chunks)  # approximate
                        # Early-stop breaks before the stop=True event which
                        # carries timings. Compute generation_ms from wall
                        # clock so timing telemetry isn't lost.
                        _es_elapsed = (time.perf_counter() - http_start) * 1000
                        timings = {
                            "predicted_ms": _es_elapsed,
                            "predicted_per_second": (
                                tokens_generated / (_es_elapsed / 1000)
                                if _es_elapsed > 0
                                else 0.0
                            ),
                        }
                        break

                    if data.get("stop", False):
                        timings = data.get("timings", {})
                        tokens_generated = data.get("tokens_predicted", 0)
                        prompt_tokens = data.get("tokens_evaluated", 0)
                        cached_tokens = data.get("tokens_cached", 0)
                        break

            http_elapsed_ms = (time.perf_counter() - http_start) * 1000
            elapsed = time.time() - start_time

            prompt_eval_ms = timings.get("prompt_ms", 0.0)
            generation_ms = timings.get("predicted_ms", 0.0)
            predicted_per_second = timings.get("predicted_per_second", 0.0)

            inference_ms = prompt_eval_ms + generation_ms
            http_overhead_ms = max(0.0, http_elapsed_ms - inference_ms)

            # Update cache stats
            self.cache_stats.total_prompt_tokens += prompt_tokens
            self.cache_stats.cached_prompt_tokens += cached_tokens
            if cached_tokens > 0:
                self.cache_stats.cache_hits += 1
            else:
                self.cache_stats.cache_misses += 1

            speed = (
                predicted_per_second
                if predicted_per_second > 0
                else (tokens_generated / elapsed if elapsed > 0 else 0.0)
            )

            # Extract speculative decoding acceptance telemetry (streaming)
            n_drafted = timings.get("drafted_n_tokens", 0)
            n_accepted = timings.get("drafted_n_accepted", 0)
            accept_rate = (n_accepted / n_drafted) if n_drafted > 0 else 0.0
            if n_drafted > 0:
                logger.info(
                    "Spec accept (stream): %d/%d (%.1f%%) for %s",
                    n_accepted, n_drafted, accept_rate * 100, role_config.name,
                )

            return InferenceResult(
                role=role_config.name,
                output="".join(chunks),
                tokens_generated=tokens_generated,
                generation_speed=speed,
                elapsed_time=elapsed,
                success=True,
                prompt_eval_ms=prompt_eval_ms,
                generation_ms=generation_ms,
                predicted_per_second=predicted_per_second,
                http_overhead_ms=http_overhead_ms,
                n_tokens_drafted=n_drafted,
                n_tokens_accepted=n_accepted,
                acceptance_rate=accept_rate,
            )

        except httpx.TimeoutException:
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=request.timeout or self.config.timeout,
                success=False,
                error_message=f"Request timed out after {request.timeout}s",
            )

        except httpx.RequestError as e:
            elapsed = time.time() - start_time
            return InferenceResult(
                role=role_config.name,
                output="",
                tokens_generated=0,
                generation_speed=0.0,
                elapsed_time=elapsed,
                success=False,
                error_message=f"Server request failed: {e}",
            )

    def _build_payload(
        self,
        role_config: RoleConfig,
        request: InferenceRequest,
    ) -> dict[str, Any]:
        """Build the JSON payload for the completion endpoint.

        Args:
            role_config: Role configuration with acceleration settings.
            request: Inference request parameters.

        Returns:
            Dictionary payload for POST request.
        """
        # Honor per-request cache_prompt override; default to True
        cache_prompt = request.cache_prompt if request.cache_prompt is not None else True
        payload: dict[str, Any] = {
            "prompt": request.prompt or "",
            "n_predict": request.n_tokens,
            "cache_prompt": cache_prompt,
        }

        # Temperature
        temp = role_config.acceleration.temperature
        if temp is None:
            temp = request.temperature
        payload["temperature"] = temp

        # Add sampling parameters
        payload["top_k"] = 40
        payload["top_p"] = 0.95
        payload["repeat_penalty"] = 1.1

        # Forward stop sequences to llama-server
        stop_seqs = getattr(request, "stop_sequences", None)
        if stop_seqs:
            payload["stop"] = stop_seqs

        # Grammar-constrained generation (llama-server native support)
        if request.json_schema:
            payload["json_schema"] = request.json_schema
        if request.grammar:
            payload["grammar"] = request.grammar

        return payload

    def get_slots(self) -> list[SlotInfo]:
        """Get current slot information from the server.

        Returns:
            List of SlotInfo for each server slot.
        """
        try:
            response = self.client.get(
                "/slots",
                timeout=self.config.connect_timeout,
            )
            response.raise_for_status()
            slots_data = response.json()

            result = []
            for slot in slots_data:
                info = SlotInfo(
                    slot_id=slot.get("id", 0),
                    state=slot.get("state", "unknown"),
                    prompt_tokens=slot.get("n_past", 0),
                    cache_tokens=slot.get("n_cache", 0),
                    last_access=time.time(),
                )
                result.append(info)
                self.slots[info.slot_id] = info

            return result

        except httpx.RequestError as e:
            logger.warning(f"Failed to get slot info: {e}")
            return []

    def get_cache_stats(self) -> CacheStats:
        """Get cache performance statistics.

        Returns:
            CacheStats with hit rates and token savings.
        """
        return self.cache_stats

    def reset_cache_stats(self) -> None:
        """Reset cache statistics."""
        self.cache_stats = CacheStats()

    def save_slot(self, slot_id: int, filename: str) -> bool:
        """Save a slot's KV cache state to disk.

        Args:
            slot_id: Slot to save.
            filename: Absolute path to save file.

        Returns:
            True if save succeeded.
        """
        try:
            response = self.client.post(
                f"/slots/{slot_id}?action=save",
                json={"filename": filename},
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            logger.info(f"Saved slot {slot_id} to {filename}")
            return True
        except httpx.RequestError as e:
            logger.error(f"Failed to save slot {slot_id}: {e}")
            return False

    def restore_slot(self, slot_id: int, filename: str) -> bool:
        """Restore a slot's KV cache state from disk.

        Args:
            slot_id: Slot to restore.
            filename: Path to saved state file.

        Returns:
            True if restore succeeded.
        """
        try:
            response = self.client.post(
                f"/slots/{slot_id}?action=restore",
                json={"filename": filename},
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            logger.info(f"Restored slot {slot_id} from {filename}")
            return True
        except httpx.RequestError as e:
            logger.error(f"Failed to restore slot {slot_id}: {e}")
            return False

    def close(self) -> None:
        """Close the HTTP client and release connections.

        Call this when done with the backend to properly clean up resources.
        """
        self.client.close()

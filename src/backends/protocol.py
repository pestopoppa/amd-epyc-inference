"""Backend protocol definitions for LLM inference.

This module defines the interfaces for LLM backends using Python's Protocol
for structural subtyping. This allows any class that implements the required
methods to be used as a backend, without requiring inheritance.

Protocols defined:
- LLMBackend: Core inference interface
- StreamingBackend: Extension for streaming inference
- CachingBackend: Extension for prefix caching

Usage:
    from src.backends.protocol import LLMBackend, InferenceRequest, InferenceResult

    class MyBackend:
        def infer(self, request: InferenceRequest) -> InferenceResult:
            ...

    # MyBackend satisfies LLMBackend without explicit inheritance
    backend: LLMBackend = MyBackend()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass
class InferenceRequest:
    """Request for LLM inference.

    Attributes:
        prompt: The input prompt/context for generation.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic).
        stop_sequences: Optional list of stop sequences.
        role: Role hint for routing (e.g., "coder", "worker").
        stream: Whether to stream the response.
        timeout: Request timeout in seconds.
        extra: Additional backend-specific parameters.
    """

    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    role: str = "worker"
    stream: bool = False
    timeout: int = 120
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result from LLM inference.

    Attributes:
        text: Generated text.
        tokens_generated: Number of tokens generated.
        prompt_tokens: Number of tokens in the prompt.
        elapsed_seconds: Time taken for inference.
        success: Whether inference completed successfully.
        error: Error message if not successful.
        tokens_per_second: Generation speed.
        cache_hit: Whether prefix cache was hit (if applicable).
        cached_tokens: Number of tokens served from cache.
        extra: Additional backend-specific metrics.
    """

    text: str
    tokens_generated: int = 0
    prompt_tokens: int = 0
    elapsed_seconds: float = 0.0
    success: bool = True
    error: str = ""
    tokens_per_second: float = 0.0
    cache_hit: bool = False
    cached_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamToken:
    """A single token from streaming inference.

    Attributes:
        text: Token text.
        token_id: Numeric token ID (if available).
        logits: Token logits/probabilities (if available).
        is_stop: Whether this is a stop token.
    """

    text: str
    token_id: int | None = None
    logits: list[float] | None = None
    is_stop: bool = False


@dataclass
class BackendStats:
    """Statistics for a backend.

    Attributes:
        total_requests: Total number of requests processed.
        total_tokens_generated: Total tokens generated.
        total_prompt_tokens: Total prompt tokens processed.
        total_elapsed_seconds: Total time spent on inference.
        cache_hits: Number of cache hits (if caching).
        cache_misses: Number of cache misses (if caching).
        errors: Number of failed requests.
    """

    total_requests: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    total_elapsed_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0

    @property
    def average_tokens_per_second(self) -> float:
        """Average generation speed across all requests."""
        if self.total_elapsed_seconds == 0:
            return 0.0
        return self.total_tokens_generated / self.total_elapsed_seconds

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as percentage (0-100)."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM inference backends.

    This is the core interface that all backends must implement.
    Use @runtime_checkable to enable isinstance() checks.

    Example:
        class MyBackend:
            def infer(self, request: InferenceRequest) -> InferenceResult:
                # implementation
                ...

            def health_check(self) -> bool:
                return True

        # Type checking works without inheritance
        backend: LLMBackend = MyBackend()
        assert isinstance(backend, LLMBackend)
    """

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference on the backend.

        Args:
            request: Inference request parameters.

        Returns:
            InferenceResult with generated text and metrics.
        """
        ...

    def health_check(self) -> bool:
        """Check if the backend is healthy and ready.

        Returns:
            True if backend is ready for inference.
        """
        ...


@runtime_checkable
class StreamingBackend(Protocol):
    """Protocol for streaming inference backends.

    Extends LLMBackend with streaming support. Backends implementing
    this protocol can yield tokens as they're generated.
    """

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference (non-streaming)."""
        ...

    def health_check(self) -> bool:
        """Check if backend is healthy."""
        ...

    def infer_stream(self, request: InferenceRequest) -> Iterator[StreamToken]:
        """Run streaming inference.

        Args:
            request: Inference request (request.stream should be True).

        Yields:
            StreamToken for each generated token.
        """
        ...


@runtime_checkable
class CachingBackend(Protocol):
    """Protocol for backends with prefix caching support.

    Extends LLMBackend with cache management. Backends implementing
    this protocol support RadixAttention-style prefix caching.
    """

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference (with caching)."""
        ...

    def health_check(self) -> bool:
        """Check if backend is healthy."""
        ...

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dict with cache hit rate, tokens saved, etc.
        """
        ...

    def reset_cache_stats(self) -> None:
        """Reset cache statistics."""
        ...


class MockBackend:
    """Mock backend for testing.

    Implements LLMBackend protocol without inheritance for demonstration.
    """

    def __init__(self, response_prefix: str = "[MOCK]"):
        """Initialize mock backend.

        Args:
            response_prefix: Prefix for mock responses.
        """
        self.response_prefix = response_prefix
        self.stats = BackendStats()

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Generate mock response.

        Args:
            request: Inference request.

        Returns:
            Mock InferenceResult.
        """
        import time

        start = time.perf_counter()

        # Generate mock response
        prompt_preview = request.prompt[:50].replace("\n", " ")
        text = f"{self.response_prefix} Response for {request.role}: {prompt_preview}..."

        elapsed = time.perf_counter() - start
        tokens = len(text) // 4  # Rough token estimate

        # Update stats
        self.stats.total_requests += 1
        self.stats.total_tokens_generated += tokens
        self.stats.total_prompt_tokens += len(request.prompt) // 4
        self.stats.total_elapsed_seconds += elapsed

        return InferenceResult(
            text=text,
            tokens_generated=tokens,
            prompt_tokens=len(request.prompt) // 4,
            elapsed_seconds=elapsed,
            success=True,
            tokens_per_second=tokens / elapsed if elapsed > 0 else 0,
        )

    def health_check(self) -> bool:
        """Mock is always healthy."""
        return True

    def get_stats(self) -> BackendStats:
        """Get mock backend stats."""
        return self.stats


# Type aliases for convenience
BackendType = LLMBackend | StreamingBackend | CachingBackend


def is_streaming_backend(backend: Any) -> bool:
    """Check if a backend supports streaming.

    Args:
        backend: Backend instance to check.

    Returns:
        True if backend implements StreamingBackend.
    """
    return isinstance(backend, StreamingBackend)


def is_caching_backend(backend: Any) -> bool:
    """Check if a backend supports caching.

    Args:
        backend: Backend instance to check.

    Returns:
        True if backend implements CachingBackend.
    """
    return isinstance(backend, CachingBackend)

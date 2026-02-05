"""OpenAI API backend for LLM inference.

Implements the LLMBackend and StreamingBackend protocols for OpenAI's API.

Usage:
    from src.config import get_config
    from src.backends.openai import OpenAIBackend

    config = get_config()
    backend = OpenAIBackend(config.external_backends.openai)

    request = InferenceRequest(prompt="Hello!")
    result = backend.infer(request)
    print(result.text)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import httpx

logger = logging.getLogger(__name__)

from src.backends.protocol import (
    BackendStats,
    InferenceRequest,
    InferenceResult,
    StreamToken,
)
from src.config import ExternalAPIConfig


class OpenAIBackend:
    """Backend for OpenAI's Chat Completions API.

    Implements LLMBackend and StreamingBackend protocols.
    Compatible with OpenAI API and OpenAI-compatible endpoints.

    Attributes:
        config: API configuration (key, base URL, timeouts).
        stats: Cumulative statistics for this backend instance.
    """

    def __init__(self, config: ExternalAPIConfig) -> None:
        """Initialize the OpenAI backend.

        Args:
            config: External API configuration with API key and settings.

        Raises:
            ValueError: If API key is not provided.
        """
        if not config.api_key:
            raise ValueError("OpenAI API key is required")

        self.config = config
        self.stats = BackendStats()
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(config.timeout, connect=10.0),
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference via OpenAI API.

        Args:
            request: Inference request with prompt and parameters.

        Returns:
            InferenceResult with generated text and metrics.
        """
        start_time = time.perf_counter()

        try:
            response = self._make_request(request)
            elapsed = time.perf_counter() - start_time

            # Extract text from response
            text = ""
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                text = message.get("content", "")

            # Extract usage metrics
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Update stats
            self.stats.total_requests += 1
            self.stats.total_tokens_generated += completion_tokens
            self.stats.total_prompt_tokens += prompt_tokens
            self.stats.total_elapsed_seconds += elapsed

            # Extract finish reason
            finish_reason = ""
            if choices:
                finish_reason = choices[0].get("finish_reason", "")

            return InferenceResult(
                text=text,
                tokens_generated=completion_tokens,
                prompt_tokens=prompt_tokens,
                elapsed_seconds=elapsed,
                success=True,
                tokens_per_second=completion_tokens / elapsed if elapsed > 0 else 0,
                extra={
                    "model": response.get("model", ""),
                    "finish_reason": finish_reason,
                },
            )

        except httpx.HTTPStatusError as e:
            elapsed = time.perf_counter() - start_time
            self.stats.errors += 1
            return InferenceResult(
                text="",
                elapsed_seconds=elapsed,
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )

        except httpx.RequestError as e:
            elapsed = time.perf_counter() - start_time
            self.stats.errors += 1
            return InferenceResult(
                text="",
                elapsed_seconds=elapsed,
                success=False,
                error=f"Request error: {str(e)}",
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            self.stats.errors += 1
            return InferenceResult(
                text="",
                elapsed_seconds=elapsed,
                success=False,
                error=f"Unexpected error: {str(e)}",
            )

    def infer_stream(self, request: InferenceRequest) -> Iterator[StreamToken]:
        """Run streaming inference via OpenAI API.

        Args:
            request: Inference request (stream flag is set automatically).

        Yields:
            StreamToken for each generated token/chunk.
        """
        payload = self._build_payload(request, stream=True)

        try:
            with self._client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield StreamToken(text="", is_stop=True)
                            break

                        try:
                            import json
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield StreamToken(text=content)

                                finish_reason = choices[0].get("finish_reason")
                                if finish_reason:
                                    yield StreamToken(text="", is_stop=True)

                        except Exception as e:
                            logger.debug("Failed to parse streaming chunk: %s", e)
                            continue

        except httpx.HTTPStatusError:
            yield StreamToken(text="", is_stop=True)

    def health_check(self) -> bool:
        """Check if the OpenAI API is reachable.

        Returns:
            True if API responds, False otherwise.
        """
        try:
            # Use models endpoint for lightweight health check
            response = self._client.get("/models", timeout=10.0)
            return response.status_code in (200, 401)
        except Exception as e:
            logger.debug("OpenAI health check failed: %s", e)
            return False

    def get_stats(self) -> BackendStats:
        """Get cumulative statistics for this backend.

        Returns:
            BackendStats with request counts and performance metrics.
        """
        return self.stats

    def _build_payload(
        self, request: InferenceRequest, stream: bool = False
    ) -> dict[str, Any]:
        """Build API request payload from InferenceRequest.

        Args:
            request: The inference request.
            stream: Whether to enable streaming.

        Returns:
            Dict payload for the API.
        """
        # Determine model
        model = request.extra.get("model", self.config.default_model)

        # Build messages from prompt
        messages = []
        if request.extra.get("messages"):
            messages = request.extra["messages"]
        else:
            # Add system message if provided
            if request.extra.get("system"):
                messages.append({
                    "role": "system",
                    "content": request.extra["system"],
                })
            messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }

        # Add optional parameters
        if request.temperature >= 0:
            payload["temperature"] = request.temperature

        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        if stream:
            payload["stream"] = True

        return payload

    def _make_request(self, request: InferenceRequest) -> dict[str, Any]:
        """Make a non-streaming API request.

        Args:
            request: The inference request.

        Returns:
            Parsed JSON response.

        Raises:
            httpx.HTTPStatusError: On HTTP errors.
            httpx.RequestError: On network errors.
        """
        payload = self._build_payload(request, stream=False)

        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()

        return response.json()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "OpenAIBackend":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

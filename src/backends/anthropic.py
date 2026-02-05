"""Anthropic API backend for LLM inference.

Implements the LLMBackend and StreamingBackend protocols for Anthropic's API.

Usage:
    from src.config import get_config
    from src.backends.anthropic import AnthropicBackend

    config = get_config()
    backend = AnthropicBackend(config.external_backends.anthropic)

    request = InferenceRequest(prompt="Hello, Claude!")
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


class AnthropicBackend:
    """Backend for Anthropic's Claude API.

    Implements LLMBackend and StreamingBackend protocols.

    Attributes:
        config: API configuration (key, base URL, timeouts).
        stats: Cumulative statistics for this backend instance.
    """

    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, config: ExternalAPIConfig) -> None:
        """Initialize the Anthropic backend.

        Args:
            config: External API configuration with API key and settings.

        Raises:
            ValueError: If API key is not provided.
        """
        if not config.api_key:
            raise ValueError("Anthropic API key is required")

        self.config = config
        self.stats = BackendStats()
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": self.ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(config.timeout, connect=10.0),
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference via Anthropic API.

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
            if response.get("content"):
                for block in response["content"]:
                    if block.get("type") == "text":
                        text += block.get("text", "")

            # Extract usage metrics
            usage = response.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # Update stats
            self.stats.total_requests += 1
            self.stats.total_tokens_generated += output_tokens
            self.stats.total_prompt_tokens += input_tokens
            self.stats.total_elapsed_seconds += elapsed

            return InferenceResult(
                text=text,
                tokens_generated=output_tokens,
                prompt_tokens=input_tokens,
                elapsed_seconds=elapsed,
                success=True,
                tokens_per_second=output_tokens / elapsed if elapsed > 0 else 0,
                extra={
                    "model": response.get("model", ""),
                    "stop_reason": response.get("stop_reason", ""),
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
        """Run streaming inference via Anthropic API.

        Args:
            request: Inference request (stream flag is set automatically).

        Yields:
            StreamToken for each generated token/chunk.
        """
        payload = self._build_payload(request, stream=True)

        try:
            with self._client.stream(
                "POST",
                "/v1/messages",
                json=payload,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            import json
                            event = json.loads(data)
                            event_type = event.get("type", "")

                            if event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        yield StreamToken(text=text)

                            elif event_type == "message_stop":
                                yield StreamToken(text="", is_stop=True)

                        except Exception as e:
                            logger.debug("Failed to parse streaming event: %s", e)
                            continue

        except httpx.HTTPStatusError:
            yield StreamToken(
                text="",
                is_stop=True,
            )

    def health_check(self) -> bool:
        """Check if the Anthropic API is reachable.

        Returns:
            True if API responds, False otherwise.
        """
        try:
            # Make a minimal request to check connectivity
            response = self._client.post(
                "/v1/messages",
                json={
                    "model": self.config.default_model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                timeout=10.0,
            )
            return response.status_code in (200, 400, 401)
        except Exception as e:
            logger.debug("Anthropic health check failed: %s", e)
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
            messages = [{"role": "user", "content": request.prompt}]

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }

        # Add optional parameters
        if request.temperature > 0:
            payload["temperature"] = request.temperature

        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences

        if stream:
            payload["stream"] = True

        # Add system prompt if provided
        if request.extra.get("system"):
            payload["system"] = request.extra["system"]

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

        response = self._client.post("/v1/messages", json=payload)
        response.raise_for_status()

        return response.json()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "AnthropicBackend":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

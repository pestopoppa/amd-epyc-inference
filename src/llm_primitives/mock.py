"""Mock inference methods for testing."""

from typing import Any

from .types import LLMResult


class MockMixin:
    """Mixin for mock inference methods."""

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

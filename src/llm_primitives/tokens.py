"""Token estimation utilities."""


class TokensMixin:
    """Mixin for token estimation methods."""

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

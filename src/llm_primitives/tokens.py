"""Token estimation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tokenizer import LlamaTokenizer


class TokensMixin:
    """Mixin for token estimation methods.

    When ``accurate_token_counting`` is enabled and a ``_tokenizer``
    attribute is set (by LLMPrimitives.__init__), uses the llama-server
    /tokenize endpoint for exact counts.  Otherwise falls back to the
    ``len(text) // 4`` heuristic.
    """

    _tokenizer: LlamaTokenizer | None = None

    def _estimate_prompt_tokens(self, prompt: str) -> int:
        """Estimate number of tokens in a prompt.

        Args:
            prompt: The prompt text.

        Returns:
            Token count (exact if tokenizer available, estimated otherwise).
        """
        if self._tokenizer is not None:
            return self._tokenizer.count_tokens(prompt)
        return len(prompt) // 4

    def _estimate_completion_tokens(self, response: str) -> int:
        """Estimate number of tokens in a response.

        Args:
            response: The response text.

        Returns:
            Token count (exact if tokenizer available, estimated otherwise).
        """
        if self._tokenizer is not None:
            return self._tokenizer.count_tokens(response)
        return len(response) // 4

    def _count_tokens(self, text: str) -> int:
        """Count tokens — accurate if tokenizer available, heuristic otherwise.

        Convenience method for context-window management code (C1/C3).

        Args:
            text: Input text.

        Returns:
            Token count.
        """
        if self._tokenizer is not None:
            return self._tokenizer.count_tokens(text)
        return len(text) // 4

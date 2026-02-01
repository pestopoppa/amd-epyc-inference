"""Type definitions for LLM primitives."""

from dataclasses import dataclass


@dataclass
class CallLogEntry:
    """Entry in the call log for debugging/testing."""

    timestamp: float
    call_type: str  # "call" or "batch"
    prompt: str | None = None
    prompts: list[str] | None = None
    context_slice: str | None = None
    role: str = "worker"
    persona: str | None = None
    result: str | list[str] | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass
class LLMResult:
    """Result of an LLM call with optional abort information.

    Used by llm_call_monitored when generation monitoring is enabled.

    Attributes:
        text: The generated text (may be partial if aborted).
        aborted: Whether generation was aborted early.
        abort_reason: Reason for abort (if aborted).
        tokens_generated: Number of tokens generated.
        tokens_saved: Estimated tokens saved by early abort.
        failure_probability: Estimated failure probability at abort.
        elapsed_seconds: Time taken for generation.
    """

    text: str
    aborted: bool = False
    abort_reason: str = ""
    tokens_generated: int = 0
    tokens_saved: int = 0
    failure_probability: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class QueryCost:
    """Cost tracking for a single query/task.

    Tracks tokens used and estimates cost based on model rates.
    """

    query_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls_made: int = 0
    batch_calls_made: int = 0
    elapsed_seconds: float = 0.0

    # Cost estimation (per 1M tokens, configurable)
    prompt_rate: float = 0.0  # $ per 1M prompt tokens
    completion_rate: float = 0.0  # $ per 1M completion tokens

    @property
    def estimated_cost(self) -> float:
        """Estimate cost in dollars based on token counts and rates."""
        prompt_cost = (self.prompt_tokens / 1_000_000) * self.prompt_rate
        completion_cost = (self.completion_tokens / 1_000_000) * self.completion_rate
        return prompt_cost + completion_cost

    def __str__(self) -> str:
        return (
            f"QueryCost(id={self.query_id}, tokens={self.total_tokens}, "
            f"calls={self.calls_made}, batch={self.batch_calls_made}, "
            f"cost=${self.estimated_cost:.6f})"
        )

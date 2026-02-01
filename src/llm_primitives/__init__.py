"""LLM primitives for RLM-style orchestration.

This module provides llm_call() and llm_batch() functions for spawning
sub-LM calls from the Root LM. Supports both mock mode (for testing)
and real inference via persistent llama-server backends with RadixAttention
prefix caching.

Includes optional GenerationMonitor integration for early failure detection.
See research/early_failure_prediction.md for literature references.

Usage:
    from src.llm_primitives import LLMPrimitives

    # Mock mode (for testing)
    primitives = LLMPrimitives(mock_mode=True)
    result = primitives.llm_call("Summarize this:", "Some text")

    # Real mode (with server URLs - RadixAttention enabled)
    primitives = LLMPrimitives(
        mock_mode=False,
        server_urls={
            "frontdoor": "http://localhost:8080",
            "coder": "http://localhost:8081",
            "worker": "http://localhost:8082",
        }
    )
    result = primitives.llm_call("Summarize this:", "Some text", role="worker")

    # Legacy mode (with model server - per-inference subprocess)
    from src.model_server import ModelServer
    server = ModelServer()
    primitives = LLMPrimitives(model_server=server, mock_mode=False)
    result = primitives.llm_call("Summarize this:", "Some text", role="worker")

    # With generation monitoring for early abort
    from src.generation_monitor import GenerationMonitor, MonitorConfig
    monitor = GenerationMonitor(config=MonitorConfig.for_tier("worker"))
    result = primitives.llm_call_monitored("Solve this:", "", role="worker", monitor=monitor)
    if result.aborted:
        # Handle early abort - escalate to higher tier
        pass
"""

from .config import LLMPrimitivesConfig
from .primitives import LLMPrimitives
from .types import CallLogEntry, LLMResult, QueryCost

__all__ = [
    "LLMPrimitives",
    "LLMPrimitivesConfig",
    "LLMResult",
    "CallLogEntry",
    "QueryCost",
]

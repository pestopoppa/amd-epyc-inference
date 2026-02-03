"""Backend modules for model inference.

This package provides different inference backends:
- LlamaServerBackend: Persistent HTTP server mode with prefix caching (local)
- AnthropicBackend: Anthropic Claude API backend (external)
- OpenAIBackend: OpenAI API backend (external)
- MockBackend: Mock backend for testing

Protocols:
- LLMBackend: Core inference interface
- StreamingBackend: Extension for streaming inference
- CachingBackend: Extension for prefix caching
"""

from src.backends.protocol import (
    LLMBackend,
    StreamingBackend,
    CachingBackend,
    InferenceRequest,
    InferenceResult,
    StreamToken,
    BackendStats,
    MockBackend,
    is_streaming_backend,
    is_caching_backend,
)
from src.backends.llama_server import LlamaServerBackend
from src.backends.anthropic import AnthropicBackend
from src.backends.openai import OpenAIBackend

__all__ = [
    # Protocols
    "LLMBackend",
    "StreamingBackend",
    "CachingBackend",
    # Data classes
    "InferenceRequest",
    "InferenceResult",
    "StreamToken",
    "BackendStats",
    # Local backends
    "LlamaServerBackend",
    "MockBackend",
    # External API backends
    "AnthropicBackend",
    "OpenAIBackend",
    # Utilities
    "is_streaming_backend",
    "is_caching_backend",
]

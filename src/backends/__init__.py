"""Backend modules for model inference.

This package provides different inference backends:
- LlamaCppBackend: Per-inference subprocess via llama.cpp CLI tools
- LlamaServerBackend: Persistent HTTP server mode with prefix caching
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
    # Backends
    "LlamaServerBackend",
    "MockBackend",
    # Utilities
    "is_streaming_backend",
    "is_caching_backend",
]

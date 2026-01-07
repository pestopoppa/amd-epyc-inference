"""Backend modules for model inference.

This package provides different inference backends:
- LlamaCppBackend: Per-inference subprocess via llama.cpp CLI tools
- LlamaServerBackend: Persistent HTTP server mode with prefix caching
"""

from src.backends.llama_server import LlamaServerBackend

__all__ = ["LlamaServerBackend"]

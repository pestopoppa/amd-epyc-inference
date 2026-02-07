"""
TaskEmbedder: Generate embeddings for TaskIR and related content.

Uses BGE-large-en-v1.5 via llama-server /embedding endpoint for efficient embedding.
Falls back to subprocess (llama-embedding) if server unavailable.
Falls back to hash-based pseudo-embeddings if model is unavailable.

For parallel fan-out to multiple servers, see parallel_embedder.py.

Performance:
- HTTP server: 2-5ms per embedding (40x faster)
- Subprocess: 50-200ms per embedding
- Hash fallback: <1ms but no semantic similarity
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from src.inference_lock import inference_lock

# Default model path (BGE-large-en-v1.5 for embeddings)
# BGE-large produces 1024-dim embeddings, purpose-built for similarity search
DEFAULT_MODEL_PATH = Path("/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf")
DEFAULT_EMBEDDING_BINARY = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-embedding")
DEFAULT_SERVER_URL = "http://127.0.0.1:8090"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_path: Path = DEFAULT_MODEL_PATH
    embedding_binary: Path = DEFAULT_EMBEDDING_BINARY
    embedding_dim: int = 1024  # BGE-large embedding dimension
    threads: int = 8  # Use fewer threads for embedding (fast operation)
    timeout: int = 30  # Seconds
    use_fallback: bool = True  # Fall back to hash-based if model unavailable
    server_url: str = DEFAULT_SERVER_URL  # Embedding server URL
    use_server: bool = True  # Try HTTP server first (40x faster)
    use_parallel: bool = True  # Use parallel embedder client (probe-first to 6 servers)


class TaskEmbedder:
    """
    Generate embeddings for TaskIR and other orchestration content.

    Embedding strategy (in priority order):
    1. Parallel HTTP servers (probe-first to ports 8090-8095) - 5-15ms with redundancy
    2. Single HTTP server fallback (port 8090) - 2-5ms
    3. Subprocess (llama-embedding binary) - 50-200ms
    4. Hash-based pseudo-embeddings - fallback

    Fallback:
    - If model unavailable, use hash-based pseudo-embeddings
    - These preserve semantic locality for identical inputs
    - But lose generalization (similar != identical)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model_available = self._check_model()
        self._server_available: Optional[bool] = None  # Lazy check
        self._http_client = None  # Lazy httpx client
        self._parallel_client = None  # Lazy parallel client

    def _check_model(self) -> bool:
        """Check if embedding model and binary are available."""
        return (
            self.config.model_path.exists()
            and self.config.embedding_binary.exists()
        )

    def _check_server(self) -> bool:
        """Check if embedding server is available (lazy, cached)."""
        if self._server_available is not None:
            return self._server_available

        if not self.config.use_server:
            self._server_available = False
            return False

        try:
            import httpx
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.config.server_url}/health")
                self._server_available = resp.status_code == 200
        except Exception as e:
            logger.debug("Embedding server health check failed: %s", e)
            self._server_available = False

        if self._server_available:
            logger.debug("Embedding server available at %s", self.config.server_url)
        else:
            logger.debug("Embedding server not available, falling back to subprocess")

        return self._server_available

    def _get_http_client(self):
        """Get or create httpx client (lazy initialization)."""
        if self._http_client is None:
            import httpx
            self._http_client = httpx.Client(timeout=10.0)
        return self._http_client

    def _generate_embedding_http(self, text: str) -> np.ndarray:
        """Generate embedding using HTTP server (2-5ms)."""
        client = self._get_http_client()

        # llama-server /embedding endpoint format
        resp = client.post(
            f"{self.config.server_url}/embedding",
            json={"content": text},
        )
        resp.raise_for_status()
        data = resp.json()

        # llama-server returns:
        # - {"embedding": [[...]], ...}
        # - {"data": [{"embedding": [...]}], ...}
        # - or list payload: [{"embedding": [[...]], ...}]
        if isinstance(data, list) and data:
            data = data[0]

        if "embedding" in data:
            # Direct embedding format
            embedding_data = data["embedding"]
            if isinstance(embedding_data[0], list):
                embedding = np.array(embedding_data[0], dtype=np.float32)
            else:
                embedding = np.array(embedding_data, dtype=np.float32)
        elif "data" in data and len(data["data"]) > 0:
            # OpenAI-compatible format
            embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
        else:
            raise ValueError(
                "Unexpected embedding response format: "
                f"{list(data.keys()) if hasattr(data, 'keys') else type(data)}"
            )

        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _serialize_task_ir(self, task_ir: Dict[str, Any]) -> str:
        """
        Serialize TaskIR to embedding-friendly text.

        Focus on semantically meaningful fields:
        - task_type
        - objective
        - constraints
        - inputs (types only, not content)
        """
        parts = []

        # Task type
        if "task_type" in task_ir:
            parts.append(f"type:{task_ir['task_type']}")

        # Objective (most important)
        if "objective" in task_ir:
            parts.append(f"objective:{task_ir['objective']}")

        # Priority
        if "priority" in task_ir:
            parts.append(f"priority:{task_ir['priority']}")

        # Constraints
        if "constraints" in task_ir and task_ir["constraints"]:
            constraints_str = ",".join(task_ir["constraints"][:5])  # Limit
            parts.append(f"constraints:{constraints_str}")

        # Input types (not content)
        if "inputs" in task_ir:
            input_types = [inp.get("type", "unknown") for inp in task_ir["inputs"]]
            parts.append(f"input_types:{','.join(input_types)}")

        return " | ".join(parts)

    def _serialize_failure_context(self, failure_context: Dict[str, Any]) -> str:
        """Serialize failure context for escalation memory."""
        parts = []

        if "error_type" in failure_context:
            parts.append(f"error:{failure_context['error_type']}")

        if "gate_name" in failure_context:
            parts.append(f"gate:{failure_context['gate_name']}")

        if "agent_tier" in failure_context:
            parts.append(f"tier:{failure_context['agent_tier']}")

        if "failure_message" in failure_context:
            # Truncate message to avoid embedding noise
            msg = failure_context["failure_message"][:200]
            parts.append(f"message:{msg}")

        return " | ".join(parts)

    def _serialize_exploration(self, query: str, context_preview: str) -> str:
        """Serialize exploration context for REPL memory."""
        # Truncate to focus on structure
        preview_truncated = context_preview[:500] if context_preview else ""
        return f"query:{query} | preview:{preview_truncated}"

    def _serialize_classification_prompt(self, prompt: str, classification_type: str) -> str:
        """Serialize a prompt for classification lookup.

        Args:
            prompt: User prompt to classify.
            classification_type: Type of classification (e.g., "routing", "summarization").

        Returns:
            Serialized string for embedding.
        """
        # Truncate long prompts (first 300 chars capture intent)
        prompt_truncated = prompt[:300] if len(prompt) > 300 else prompt
        return f"classify:{classification_type} | prompt:{prompt_truncated}"

    def embed_classification_prompt(
        self,
        prompt: str,
        classification_type: str = "routing",
    ) -> np.ndarray:
        """Generate embedding for a classification prompt.

        Used by ClassificationRetriever to find similar classification exemplars.

        Args:
            prompt: User prompt to classify.
            classification_type: Type of classification (routing, summarization, etc.).

        Returns:
            Embedding vector (1024-dim for BGE-large).
        """
        text = self._serialize_classification_prompt(prompt, classification_type)
        return self._generate_embedding(text)

    def _generate_embedding_llama(self, text: str) -> np.ndarray:
        """Generate embedding using llama-embedding binary."""
        try:
            result = subprocess.run(
                [
                    str(self.config.embedding_binary),
                    "-m", str(self.config.model_path),
                    "-p", text,
                    "-t", str(self.config.threads),
                    "--embd-output-format", "json",
                ],
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            if result.returncode != 0:
                raise RuntimeError(f"llama-embedding failed: {result.stderr}")

            # Parse JSON output
            output = json.loads(result.stdout)

            # llama-embedding returns embeddings in different formats:
            # Format 1: {'object': 'list', 'data': [{'embedding': [...]}]}
            # Format 2: [{'embedding': [...]}]
            if isinstance(output, dict) and "data" in output:
                # Format 1: OpenAI-compatible format
                data = output["data"]
                if data and len(data) > 0:
                    embedding = np.array(data[0]["embedding"], dtype=np.float32)
                else:
                    raise ValueError(f"Empty data in output: {output}")
            elif isinstance(output, list) and len(output) > 0:
                # Format 2: Simple list format
                embedding = np.array(output[0]["embedding"], dtype=np.float32)
            else:
                raise ValueError(f"Unexpected output format: {output}")

            # Normalize to unit vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"llama-embedding timed out after {self.config.timeout}s")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse embedding output: {e}")

    def _generate_embedding_fallback(self, text: str) -> np.ndarray:
        """
        Generate hash-based pseudo-embedding as fallback.

        Uses SHA-256 to create deterministic embeddings.
        Preserves identity (same input = same embedding).
        Does NOT preserve similarity (similar != close).
        """
        # Create deterministic hash
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Expand hash to embedding dimension
        # Repeat hash bytes to fill embedding
        repeats = (self.config.embedding_dim * 4 // len(hash_bytes)) + 1
        expanded = (hash_bytes * repeats)[: self.config.embedding_dim * 4]

        # Convert to float32 array
        embedding = np.frombuffer(expanded, dtype=np.float32)[: self.config.embedding_dim]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_task_ir(self, task_ir: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding for a TaskIR.

        Args:
            task_ir: TaskIR dictionary

        Returns:
            Embedding vector (normalized)
        """
        text = self._serialize_task_ir(task_ir)
        return self._generate_embedding(text)

    def embed_failure_context(self, failure_context: Dict[str, Any]) -> np.ndarray:
        """
        Generate embedding for failure context (escalation memory).

        Args:
            failure_context: Failure context dictionary

        Returns:
            Embedding vector (normalized)
        """
        text = self._serialize_failure_context(failure_context)
        return self._generate_embedding(text)

    def embed_exploration(self, query: str, context_preview: str) -> np.ndarray:
        """
        Generate embedding for exploration context (REPL memory).

        Args:
            query: User query
            context_preview: Preview of context being explored

        Returns:
            Embedding vector (normalized)
        """
        text = self._serialize_exploration(query, context_preview)
        return self._generate_embedding(text)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for arbitrary text.

        Args:
            text: Input text

        Returns:
            Embedding vector (normalized)
        """
        return self._generate_embedding(text)

    def _get_parallel_client(self):
        """Get or create parallel embedder client (lazy initialization)."""
        if self._parallel_client is None:
            from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient
            self._parallel_client = ParallelEmbedderClient()
        return self._parallel_client

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using available method (parallel > single HTTP > subprocess > hash)."""
        with inference_lock("embedder", shared=True):
            # Try parallel embedder first (probe-first to 6 servers)
            if self.config.use_parallel and self.config.use_server:
                try:
                    client = self._get_parallel_client()
                    return client.embed_sync(text)
                except Exception as e:
                    logger.warning(
                        "Parallel embedder failed (%s), falling back to single server", e
                    )

            # Try single HTTP server (2-5ms)
            if self._check_server():
                try:
                    return self._generate_embedding_http(text)
                except Exception as e:
                    logger.warning(
                        "Embedding server failed (%s), falling back to subprocess", e
                    )
                    self._server_available = False  # Reset to retry later

            # Try subprocess (50-200ms)
            if self._model_available:
                try:
                    return self._generate_embedding_llama(text)
                except Exception as e:
                    if self.config.use_fallback:
                        logger.warning("llama-embedding failed (%s), using hash fallback", e)
                        return self._generate_embedding_fallback(text)
                    raise

            # Hash fallback
            if self.config.use_fallback:
                return self._generate_embedding_fallback(text)
            else:
                raise RuntimeError("Embedding model not available and fallback disabled")

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Array of embedding vectors (N x embedding_dim)
        """
        embeddings = []
        for text in texts:
            embeddings.append(self._generate_embedding(text))
        return np.array(embeddings, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.config.embedding_dim

    @property
    def is_model_available(self) -> bool:
        """Check if the neural embedding model is available."""
        return self._model_available

    @property
    def is_server_available(self) -> bool:
        """Check if the embedding server is available."""
        return self._check_server()

    def close(self) -> None:
        """Close HTTP client connections."""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
        if self._parallel_client is not None:
            self._parallel_client.close_sync()
            self._parallel_client = None

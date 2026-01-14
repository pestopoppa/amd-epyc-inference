"""
TaskEmbedder: Generate embeddings for TaskIR and related content.

Uses Qwen2.5-Coder-0.5B via llama-embedding for efficient embedding generation.
Falls back to hash-based pseudo-embeddings if model is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Default model path (Qwen2.5-Coder-0.5B for embeddings)
DEFAULT_MODEL_PATH = Path("/mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf")
DEFAULT_EMBEDDING_BINARY = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-embedding")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_path: Path = DEFAULT_MODEL_PATH
    embedding_binary: Path = DEFAULT_EMBEDDING_BINARY
    embedding_dim: int = 896  # Qwen2.5-0.5B hidden dim
    threads: int = 8  # Use fewer threads for embedding (fast operation)
    timeout: int = 30  # Seconds
    use_fallback: bool = True  # Fall back to hash-based if model unavailable


class TaskEmbedder:
    """
    Generate embeddings for TaskIR and other orchestration content.

    Embedding strategy:
    1. Serialize content to structured text
    2. Generate embedding via llama-embedding
    3. Normalize to unit vector

    Fallback:
    - If model unavailable, use hash-based pseudo-embeddings
    - These preserve semantic locality for identical inputs
    - But lose generalization (similar != identical)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model_available = self._check_model()

    def _check_model(self) -> bool:
        """Check if embedding model and binary are available."""
        return (
            self.config.model_path.exists()
            and self.config.embedding_binary.exists()
        )

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

            # llama-embedding returns list of embeddings (one per prompt)
            if isinstance(output, list) and len(output) > 0:
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

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using available method."""
        if self._model_available:
            try:
                return self._generate_embedding_llama(text)
            except Exception as e:
                if self.config.use_fallback:
                    # Log warning and fall back
                    print(f"Warning: llama-embedding failed ({e}), using fallback")
                    return self._generate_embedding_fallback(text)
                raise
        elif self.config.use_fallback:
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

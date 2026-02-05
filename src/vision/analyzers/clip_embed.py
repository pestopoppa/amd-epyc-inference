"""CLIP embedding analyzer for visual similarity search."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from src.vision.analyzers.base import Analyzer, AnalyzerResult
from src.vision.config import VISION_CACHE_DIR
from src.db.chroma_client import add_image_embedding

logger = logging.getLogger(__name__)


class ClipEmbedAnalyzer(Analyzer):
    """Generate CLIP embeddings for visual similarity search.

    Uses sentence-transformers CLIP model for image embeddings.
    Embeddings are stored in ChromaDB for "find similar images" queries.
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        store_embeddings: bool = True,
        **config: Any,
    ):
        """Initialize CLIP embedding analyzer.

        Args:
            model_name: CLIP model name from sentence-transformers.
            store_embeddings: Whether to store embeddings in ChromaDB.
            **config: Additional configuration.
        """
        super().__init__(**config)
        self.model_name = model_name
        self.store_embeddings = store_embeddings
        self._model = None

    @property
    def name(self) -> str:
        return "clip_embed"

    def initialize(self) -> None:
        """Load CLIP model."""
        try:
            import os

            from sentence_transformers import SentenceTransformer

            # Set cache directory
            os.environ.setdefault("TRANSFORMERS_CACHE", str(VISION_CACHE_DIR))

            self._model = SentenceTransformer(self.model_name)
            logger.info(f"CLIP model '{self.model_name}' loaded")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

        super().initialize()

    def analyze(self, image: Image.Image, path: Path | None = None) -> AnalyzerResult:
        """Generate CLIP embedding for image.

        Args:
            image: PIL Image to embed.
            path: Optional path to original file (for metadata).

        Returns:
            AnalyzerResult with embedding and storage status.
        """
        self.ensure_initialized()
        start = time.perf_counter()

        try:
            # Ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Generate embedding
            embedding = self._model.encode(image)
            embedding_list = embedding.tolist()

            result_data = {
                "embedding_dim": len(embedding_list),
                "stored": False,
            }

            # Store in ChromaDB
            if self.store_embeddings:
                image_id = str(uuid.uuid4())
                metadata = {}
                if path:
                    metadata["path"] = str(path)
                    image_id = str(path)  # Use path as ID for dedup

                add_image_embedding(image_id, embedding_list, metadata)
                result_data["stored"] = True
                result_data["image_id"] = image_id

            elapsed = (time.perf_counter() - start) * 1000

            return AnalyzerResult(
                analyzer_name=self.name,
                success=True,
                data=result_data,
                processing_time_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"CLIP embedding failed: {e}")
            return AnalyzerResult(
                analyzer_name=self.name,
                success=False,
                error=str(e),
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

    def embed_text(self, text: str) -> list[float]:
        """Generate CLIP embedding for text query.

        This allows searching images by text description.

        Args:
            text: Text query to embed.

        Returns:
            CLIP text embedding vector.
        """
        self.ensure_initialized()
        embedding = self._model.encode(text)
        return embedding.tolist()

    def cleanup(self) -> None:
        """Release model resources."""
        self._model = None
        self._initialized = False


class TextEmbedAnalyzer(Analyzer):
    """Generate text embeddings for description search.

    Uses sentence-transformers for text embedding.
    Separate from CLIP since we want optimized text models for description search.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        **config: Any,
    ):
        """Initialize text embedding analyzer.

        Args:
            model_name: Sentence transformer model name.
            **config: Additional configuration.
        """
        super().__init__(**config)
        self.model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return "text_embed"

    def initialize(self) -> None:
        """Load text embedding model."""
        try:
            import os

            from sentence_transformers import SentenceTransformer

            # Set cache directory
            os.environ.setdefault("TRANSFORMERS_CACHE", str(VISION_CACHE_DIR))

            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Text embedding model '{self.model_name}' loaded")
        except ImportError:
            raise ImportError("sentence-transformers not installed")

        super().initialize()

    def analyze(self, image: Image.Image, path: Path | None = None) -> AnalyzerResult:
        """Not applicable for text embedder."""
        return AnalyzerResult(
            analyzer_name=self.name,
            success=False,
            error="TextEmbedAnalyzer does not analyze images. Use embed_text() for text.",
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Text embedding vector.
        """
        self.ensure_initialized()
        embedding = self._model.encode(text)
        return embedding.tolist()

    def cleanup(self) -> None:
        """Release model resources."""
        self._model = None
        self._initialized = False

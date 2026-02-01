"""LLMLingua-2 prompt compression service.

Provides extractive token selection to reduce document size before LLM summarization.
Key benefit: representative sampling from entire document vs truncation to first N chars.

Architecture:
    Document (N chars)
           ↓
    [LLMLingua-2] → Select best tokens (ratio * N)
           ↓
    [Stage 1: Frontdoor] → Fast draft summary
           ↓
    [Stage 2: Large model] → Quality review
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_compressor_lock = threading.Lock()


@dataclass
class CompressionResult:
    """Result of prompt compression."""

    compressed_text: str
    original_chars: int
    compressed_chars: int
    original_tokens: int  # Approximate word count
    compressed_tokens: int
    actual_ratio: float
    latency_ms: float


class PromptCompressor:
    """LLMLingua-2 wrapper for extractive prompt compression.

    Uses BERT encoder (~110M params) for token classification (keep/drop).
    Runs on CPU, ~10-50ms for typical documents after model warmup.

    Example:
        compressor = PromptCompressor()
        result = compressor.compress(long_document, target_ratio=0.5)
        # result.compressed_text contains ~50% of original tokens
    """

    _instance: Optional["PromptCompressor"] = None
    _model = None

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        device: str = "cpu",
    ):
        """Initialize compressor.

        Args:
            model_name: HuggingFace model ID for LLMLingua-2
            device: Device to run on ('cpu' for this server)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._load_time: Optional[float] = None

    @classmethod
    def get_instance(cls) -> "PromptCompressor":
        """Get singleton instance (thread-safe, keeps model loaded)."""
        if cls._instance is None:
            with _compressor_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the LLMLingua-2 model."""
        if self._model is not None:
            return

        try:
            from llmlingua import PromptCompressor as LLMLinguaCompressor
        except ImportError as e:
            raise ImportError(
                "llmlingua not installed. Run: pip install llmlingua"
            ) from e

        logger.info(f"Loading LLMLingua-2 model: {self.model_name}")
        start = time.perf_counter()

        self._model = LLMLinguaCompressor(
            model_name=self.model_name,
            use_llmlingua2=True,
            device_map=self.device,
        )

        self._load_time = (time.perf_counter() - start) * 1000
        logger.info(f"LLMLingua-2 loaded in {self._load_time:.1f}ms")

    def compress(
        self,
        text: str,
        target_ratio: float = 0.5,
        force_tokens: Optional[list[str]] = None,
    ) -> CompressionResult:
        """Compress text to target ratio.

        Args:
            text: Input text to compress
            target_ratio: Target compression ratio (0.0-1.0)
                         0.3 = aggressive (30% of original)
                         0.5 = balanced (50% of original)
                         0.7 = conservative (70% of original)
            force_tokens: Tokens/phrases to always preserve

        Returns:
            CompressionResult with compressed text and metrics
        """
        self._ensure_model_loaded()

        original_chars = len(text)
        # Approximate token count (words)
        original_tokens = len(text.split())
        target_tokens = max(1, int(original_tokens * target_ratio))

        start = time.perf_counter()

        # Build compression kwargs
        compress_kwargs = {
            "target_token": target_tokens,
        }
        if force_tokens:
            compress_kwargs["force_tokens"] = force_tokens

        result = self._model.compress_prompt(text, **compress_kwargs)

        latency_ms = (time.perf_counter() - start) * 1000

        compressed_text = result["compressed_prompt"]
        compressed_chars = len(compressed_text)
        compressed_tokens = len(compressed_text.split())

        actual_ratio = compressed_chars / original_chars if original_chars > 0 else 1.0

        logger.debug(
            f"Compressed {original_chars} → {compressed_chars} chars "
            f"({actual_ratio:.1%}) in {latency_ms:.1f}ms"
        )

        return CompressionResult(
            compressed_text=compressed_text,
            original_chars=original_chars,
            compressed_chars=compressed_chars,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            actual_ratio=actual_ratio,
            latency_ms=latency_ms,
        )

    def compress_if_needed(
        self,
        text: str,
        min_chars: int = 30000,
        target_ratio: float = 0.5,
    ) -> tuple[str, Optional[CompressionResult]]:
        """Compress only if text exceeds threshold.

        Args:
            text: Input text
            min_chars: Minimum chars before compression kicks in
            target_ratio: Target compression ratio

        Returns:
            (possibly_compressed_text, compression_result_or_none)
        """
        if len(text) < min_chars:
            logger.debug(f"Skipping compression: {len(text)} chars < {min_chars} threshold")
            return text, None

        result = self.compress(text, target_ratio=target_ratio)
        return result.compressed_text, result

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def load_time_ms(self) -> Optional[float]:
        """Get model load time in milliseconds."""
        return self._load_time

"""Base analyzer interface for vision pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class AnalyzerResult:
    """Result from an analyzer.

    Analyzers return this structure with their specific output in the `data` dict.
    The pipeline aggregates results from all enabled analyzers.
    """
    analyzer_name: str
    success: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    processing_time_ms: float = 0.0


class Analyzer(ABC):
    """Abstract base class for vision analyzers.

    Each analyzer implements a specific analysis capability:
    - Face detection
    - Face embedding
    - VL description
    - EXIF extraction
    - CLIP embedding
    - etc.

    Analyzers are stateless and can be reused across multiple images.
    Heavy resources (models) are loaded lazily on first use.
    """

    def __init__(self, **config: Any):
        """Initialize analyzer with optional configuration.

        Args:
            **config: Analyzer-specific configuration options.
        """
        self.config = config
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique analyzer name."""
        pass

    def initialize(self) -> None:
        """Load models and resources. Called lazily on first analyze()."""
        self._initialized = True

    def ensure_initialized(self) -> None:
        """Ensure analyzer is initialized."""
        if not self._initialized:
            self.initialize()

    @property
    def is_initialized(self) -> bool:
        """Check if analyzer is initialized (public accessor)."""
        return self._initialized

    def _error_result(
        self,
        error: str | Exception,
        start_time: float,
    ) -> AnalyzerResult:
        """Create a standardized error result.

        Args:
            error: Error message or exception.
            start_time: Start time from time.perf_counter() for elapsed calculation.

        Returns:
            AnalyzerResult with success=False and error details.
        """
        import time
        return AnalyzerResult(
            analyzer_name=self.name,
            success=False,
            error=str(error),
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    def _success_result(
        self,
        data: dict[str, Any],
        start_time: float,
    ) -> AnalyzerResult:
        """Create a standardized success result.

        Args:
            data: Analysis output data.
            start_time: Start time from time.perf_counter() for elapsed calculation.

        Returns:
            AnalyzerResult with success=True and data.
        """
        import time
        return AnalyzerResult(
            analyzer_name=self.name,
            success=True,
            data=data,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    @abstractmethod
    def analyze(self, image: Image.Image, path: Path | None = None) -> AnalyzerResult:
        """Analyze an image.

        Args:
            image: PIL Image to analyze.
            path: Optional path to the original file (for metadata).

        Returns:
            AnalyzerResult with analysis output.
        """
        pass

    def analyze_path(self, path: Path | str) -> AnalyzerResult:
        """Analyze an image from path.

        Args:
            path: Path to image file.

        Returns:
            AnalyzerResult with analysis output.
        """
        path = Path(path)
        try:
            image = Image.open(path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return self.analyze(image, path)
        except Exception as e:
            return AnalyzerResult(
                analyzer_name=self.name,
                success=False,
                error=str(e),
            )

    def cleanup(self) -> None:
        """Release resources. Override if cleanup is needed."""
        pass


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (RGB, HWC format).

    Args:
        image: PIL Image.

    Returns:
        numpy array with shape (H, W, 3) and dtype uint8.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image.

    Args:
        array: numpy array with shape (H, W, 3) and dtype uint8.

    Returns:
        PIL Image in RGB mode.
    """
    return Image.fromarray(array.astype(np.uint8))

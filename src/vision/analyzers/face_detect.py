"""Face detection analyzer using InsightFace/RetinaFace."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from PIL import Image

from src.vision.analyzers.base import Analyzer, AnalyzerResult, pil_to_numpy
from src.vision.config import FACE_MIN_CONFIDENCE
from src.vision.analyzers.insightface_loader import get_face_app

logger = logging.getLogger(__name__)


class FaceDetectAnalyzer(Analyzer):
    """Detect faces in images using InsightFace.

    Returns bounding boxes and confidence scores for each detected face.
    Crops can optionally be extracted for downstream processing.
    """

    def __init__(
        self,
        min_confidence: float = FACE_MIN_CONFIDENCE,
        return_crops: bool = False,
        **config: Any,
    ):
        """Initialize face detector.

        Args:
            min_confidence: Minimum detection confidence (0-1).
            return_crops: Whether to extract and return face crops.
            **config: Additional configuration.
        """
        super().__init__(**config)
        self.min_confidence = min_confidence
        self.return_crops = return_crops
        self._app = None

    @property
    def name(self) -> str:
        return "face_detect"

    def initialize(self) -> None:
        """Load InsightFace model (shared with face_embed)."""
        self._app = get_face_app()
        super().initialize()

    def analyze(self, image: Image.Image, path: Path | None = None) -> AnalyzerResult:
        """Detect faces in image.

        Args:
            image: PIL Image to analyze.
            path: Optional path to original file.

        Returns:
            AnalyzerResult with faces list containing:
            - bbox: {x, y, width, height}
            - confidence: float 0-1
            - crop: base64 PNG (if return_crops=True)
            - landmarks: facial landmarks (if available)
        """
        self.ensure_initialized()
        start = time.perf_counter()

        try:
            # Convert to numpy (InsightFace expects BGR, but works with RGB)
            img_array = pil_to_numpy(image)

            # Detect faces
            faces = self._app.get(img_array)

            results = []
            for face in faces:
                confidence = float(face.det_score)
                if confidence < self.min_confidence:
                    continue

                # Extract bounding box (x1, y1, x2, y2 format)
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                face_data = {
                    "bbox": {
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                    },
                    "confidence": confidence,
                }

                # Extract landmarks if available
                if hasattr(face, "kps") and face.kps is not None:
                    face_data["landmarks"] = face.kps.tolist()

                # Extract crop if requested
                if self.return_crops:
                    import base64
                    import io

                    crop = image.crop((x1, y1, x2, y2))
                    buffer = io.BytesIO()
                    crop.save(buffer, format="PNG")
                    face_data["crop"] = base64.b64encode(buffer.getvalue()).decode()

                # Store embedding reference for face_embed analyzer
                if hasattr(face, "embedding") and face.embedding is not None:
                    face_data["_embedding"] = face.embedding.tolist()

                results.append(face_data)

            elapsed = (time.perf_counter() - start) * 1000

            return AnalyzerResult(
                analyzer_name=self.name,
                success=True,
                data={"faces": results, "face_count": len(results)},
                processing_time_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return AnalyzerResult(
                analyzer_name=self.name,
                success=False,
                error=str(e),
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

    def cleanup(self) -> None:
        """Release InsightFace reference (shared model not released)."""
        self._app = None
        self._initialized = False
        # Note: Shared model released via release_face_app() if needed

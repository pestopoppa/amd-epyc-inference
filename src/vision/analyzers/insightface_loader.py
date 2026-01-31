"""Shared InsightFace model loader for face analyzers.

This module provides a singleton loader for the InsightFace FaceAnalysis model,
which is shared between face_detect and face_embed analyzers to avoid duplicate
model loading and memory usage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.vision.config import ARCFACE_MODEL_NAME, ONNX_PROVIDERS, VISION_MODELS_DIR

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

# Singleton instance
_face_app: "FaceAnalysis | None" = None


def get_face_app() -> "FaceAnalysis":
    """Get or create the shared InsightFace FaceAnalysis instance.

    The FaceAnalysis model is loaded lazily on first call and cached
    for subsequent calls. This avoids loading the same model twice
    when both face_detect and face_embed are used together.

    Returns:
        Initialized FaceAnalysis instance ready for inference.

    Raises:
        ImportError: If insightface is not installed.
        Exception: If model loading fails.
    """
    global _face_app

    if _face_app is not None:
        return _face_app

    try:
        from insightface.app import FaceAnalysis

        _face_app = FaceAnalysis(
            name=ARCFACE_MODEL_NAME,
            root=str(VISION_MODELS_DIR),
            providers=ONNX_PROVIDERS,
        )
        _face_app.prepare(ctx_id=-1)  # CPU mode
        logger.info(f"InsightFace model '{ARCFACE_MODEL_NAME}' loaded (shared)")
        return _face_app

    except ImportError:
        logger.error("insightface not installed. Run: pip install insightface")
        raise

    except Exception as e:
        logger.error(f"Failed to load InsightFace: {e}")
        raise


def release_face_app() -> None:
    """Release the shared InsightFace instance to free memory.

    This should be called during cleanup if the face analyzers are no longer needed.
    """
    global _face_app
    _face_app = None
    logger.info("InsightFace model released")

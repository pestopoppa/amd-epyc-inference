"""Vision processing pipeline for the orchestrator.

This package provides:
- Modular analyzer plugins (face detection, VL description, OCR, etc.)
- Batch processing with job management
- ChromaDB-backed semantic search
- Video frame extraction and analysis
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from src.vision.config import (
    VISION_DATA_DIR,
    VISION_THUMBS_DIR,
    VISION_MODELS_DIR,
    VISION_CACHE_DIR,
    VISION_LOGS_DIR,
    CHROMA_PATH,
    SQLITE_PATH,
)

__all__ = [
    "VISION_DATA_DIR",
    "VISION_THUMBS_DIR",
    "VISION_MODELS_DIR",
    "VISION_CACHE_DIR",
    "VISION_LOGS_DIR",
    "CHROMA_PATH",
    "SQLITE_PATH",
]

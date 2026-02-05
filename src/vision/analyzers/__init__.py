"""Vision analyzers for the pipeline.

Each analyzer is a modular plugin that processes images and returns structured results.
Analyzers can be enabled/disabled per job configuration.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from src.vision.analyzers.base import Analyzer, AnalyzerResult
from src.vision.analyzers.face_detect import FaceDetectAnalyzer
from src.vision.analyzers.face_embed import FaceEmbedAnalyzer
from src.vision.analyzers.vl_describe import VLDescribeAnalyzer
from src.vision.analyzers.exif import ExifAnalyzer
from src.vision.analyzers.clip_embed import ClipEmbedAnalyzer

__all__ = [
    "Analyzer",
    "AnalyzerResult",
    "FaceDetectAnalyzer",
    "FaceEmbedAnalyzer",
    "VLDescribeAnalyzer",
    "ExifAnalyzer",
    "ClipEmbedAnalyzer",
]

"""Vision processing pipeline that orchestrates analyzers."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from src.vision.config import (
    MAX_IMAGE_DIMENSION,
    THUMB_SIZE,
    THUMB_QUALITY,
    VISION_THUMBS_DIR,
    ensure_directories,
)
from src.vision.models import AnalyzerType, AnalyzeResult, BoundingBox, FaceResult, ExifData
from src.vision.analyzers.base import Analyzer, AnalyzerResult
from src.vision.analyzers.face_detect import FaceDetectAnalyzer
from src.vision.analyzers.face_embed import FaceEmbedAnalyzer
from src.vision.analyzers.vl_describe import VLDescribeAnalyzer, VLOCRAnalyzer, VLStructuredAnalyzer
from src.vision.analyzers.exif import ExifAnalyzer
from src.vision.analyzers.clip_embed import ClipEmbedAnalyzer
from src.vision.analyzers.insightface_loader import release_face_app
from src.db.models.vision import Photo, Face, get_session

logger = logging.getLogger(__name__)


class VisionPipeline:
    """Orchestrates vision analyzers for image processing.

    The pipeline:
    1. Validates and preprocesses the image
    2. Runs enabled analyzers in optimal order
    3. Aggregates results into a unified response
    4. Optionally stores results in SQLite/ChromaDB
    """

    def __init__(self):
        """Initialize pipeline with available analyzers."""
        self._analyzers: dict[AnalyzerType, Analyzer] = {}
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if pipeline is initialized (public accessor)."""
        return self._initialized

    def initialize(self, analyzers: list[AnalyzerType] | None = None) -> None:
        """Initialize specified analyzers.

        Args:
            analyzers: List of analyzer types to initialize. If None, all are available.
        """
        ensure_directories()

        analyzer_map = {
            AnalyzerType.FACE_DETECT: FaceDetectAnalyzer,
            AnalyzerType.FACE_EMBED: FaceEmbedAnalyzer,
            AnalyzerType.VL_DESCRIBE: VLDescribeAnalyzer,
            AnalyzerType.VL_OCR: VLOCRAnalyzer,
            AnalyzerType.VL_STRUCTURED: VLStructuredAnalyzer,
            AnalyzerType.EXIF_EXTRACT: ExifAnalyzer,
            AnalyzerType.CLIP_EMBED: ClipEmbedAnalyzer,
        }

        types_to_init = analyzers or list(analyzer_map.keys())

        for analyzer_type in types_to_init:
            if analyzer_type in analyzer_map:
                self._analyzers[analyzer_type] = analyzer_map[analyzer_type]()

        self._initialized = True

    def get_analyzer(self, analyzer_type: AnalyzerType) -> Analyzer | None:
        """Get an analyzer instance by type."""
        return self._analyzers.get(analyzer_type)

    def analyze(
        self,
        image: Image.Image | Path | str,
        analyzers: list[AnalyzerType] | None = None,
        vl_prompt: str | None = None,
        store_results: bool = True,
        return_crops: bool = False,
    ) -> AnalyzeResult:
        """Analyze an image with specified analyzers.

        Args:
            image: PIL Image, path, or path string.
            analyzers: Analyzers to run (default: all initialized).
            vl_prompt: Custom prompt for VL description.
            store_results: Whether to store in database.
            return_crops: Whether to return face crops.

        Returns:
            AnalyzeResult with aggregated analysis.
        """
        if not self._initialized:
            self.initialize()

        start_time = time.perf_counter()
        errors: list[str] = []

        # Load image
        path: Path | None = None
        if isinstance(image, (str, Path)):
            path = Path(image)
            try:
                image = Image.open(path)
            except Exception as e:
                return AnalyzeResult(
                    errors=[f"Failed to open image: {e}"],
                    processing_time_ms=0,
                )

        # Validate and preprocess
        image, preprocess_error = self._preprocess_image(image)
        if preprocess_error:
            errors.append(preprocess_error)

        # Determine analyzers to run
        analyzer_types = analyzers or list(self._analyzers.keys())

        # Initialize result
        result = AnalyzeResult(
            image_id=str(uuid.uuid4()),
            path=str(path) if path else None,
        )

        # Compute hash for dedup
        if path:
            result.image_id = self._compute_hash(path)

        # Run analyzers in optimal order
        face_detections: list[dict[str, Any]] = []

        # 1. EXIF first (fast, provides metadata)
        if AnalyzerType.EXIF_EXTRACT in analyzer_types:
            exif_result = self._run_analyzer(AnalyzerType.EXIF_EXTRACT, image, path)
            if exif_result.success and "exif" in exif_result.data:
                result.exif = ExifData(**exif_result.data["exif"])
            elif exif_result.error:
                errors.append(f"EXIF: {exif_result.error}")

        # 2. Face detection (needed for face embedding)
        if AnalyzerType.FACE_DETECT in analyzer_types:
            face_analyzer = self._analyzers.get(AnalyzerType.FACE_DETECT)
            if face_analyzer:
                if return_crops:
                    face_analyzer.return_crops = True
                face_result = face_analyzer.analyze(image, path)

                if face_result.success:
                    face_detections = face_result.data.get("faces", [])
                    for fd in face_detections:
                        result.faces.append(
                            FaceResult(
                                bbox=BoundingBox(**fd["bbox"]),
                                confidence=fd["confidence"],
                                crop_base64=fd.get("crop"),
                            )
                        )
                elif face_result.error:
                    errors.append(f"Face detect: {face_result.error}")

        # 3. Face embedding (uses face detections)
        if AnalyzerType.FACE_EMBED in analyzer_types and face_detections:
            embed_analyzer = self._analyzers.get(AnalyzerType.FACE_EMBED)
            if embed_analyzer:
                embed_result = embed_analyzer.analyze(image, path, face_detections)

                if embed_result.success:
                    embeddings = embed_result.data.get("embeddings", [])
                    for i, emb in enumerate(embeddings):
                        if i < len(result.faces):
                            result.faces[i].embedding_stored = emb.get("stored", False)
                            if "identified_as" in emb:
                                result.faces[i].person_id = emb["identified_as"]
                elif embed_result.error:
                    errors.append(f"Face embed: {embed_result.error}")

        # 4. VL description
        if AnalyzerType.VL_DESCRIBE in analyzer_types:
            vl_analyzer = self._analyzers.get(AnalyzerType.VL_DESCRIBE)
            if vl_analyzer:
                if vl_prompt:
                    vl_analyzer.prompt = vl_prompt
                vl_result = vl_analyzer.analyze(image, path)

                if vl_result.success:
                    result.description = vl_result.data.get("description")
                elif vl_result.error:
                    errors.append(f"VL describe: {vl_result.error}")

        # 5. VL OCR
        if AnalyzerType.VL_OCR in analyzer_types:
            ocr_result = self._run_analyzer(AnalyzerType.VL_OCR, image, path)
            if ocr_result.success:
                result.ocr_text = ocr_result.data.get("description")
            elif ocr_result.error:
                errors.append(f"VL OCR: {ocr_result.error}")

        # 6. VL Structured extraction
        if AnalyzerType.VL_STRUCTURED in analyzer_types:
            struct_result = self._run_analyzer(AnalyzerType.VL_STRUCTURED, image, path)
            if struct_result.success:
                result.structured_data = struct_result.data.get("structured")
            elif struct_result.error:
                errors.append(f"VL structured: {struct_result.error}")

        # 7. CLIP embedding (for visual similarity)
        if AnalyzerType.CLIP_EMBED in analyzer_types:
            clip_result = self._run_analyzer(AnalyzerType.CLIP_EMBED, image, path)
            if clip_result.success:
                result.clip_embedding_stored = clip_result.data.get("stored", False)
            elif clip_result.error:
                errors.append(f"CLIP: {clip_result.error}")

        # Store in database
        if store_results and path:
            try:
                self._store_results(result, path)
            except Exception as e:
                errors.append(f"Database: {e}")

        result.errors = errors
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000

        return result

    def _run_analyzer(
        self,
        analyzer_type: AnalyzerType,
        image: Image.Image,
        path: Path | None,
    ) -> AnalyzerResult:
        """Run a single analyzer."""
        analyzer = self._analyzers.get(analyzer_type)
        if not analyzer:
            return AnalyzerResult(
                analyzer_name=analyzer_type.value,
                success=False,
                error=f"Analyzer {analyzer_type.value} not initialized",
            )
        return analyzer.analyze(image, path)

    def _preprocess_image(self, image: Image.Image) -> tuple[Image.Image, str | None]:
        """Validate and preprocess image.

        Args:
            image: PIL Image.

        Returns:
            Tuple of (processed image, error message if any).
        """
        error = None

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Check dimensions and resize if needed
        if max(image.size) > MAX_IMAGE_DIMENSION:
            ratio = MAX_IMAGE_DIMENSION / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            error = f"Image resized from {image.size} to fit {MAX_IMAGE_DIMENSION}px limit"

        return image, error

    def _compute_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file for deduplication."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _store_results(self, result: AnalyzeResult, path: Path) -> None:
        """Store analysis results in SQLite database."""
        session = get_session()
        try:
            # Check if photo already exists
            existing = session.query(Photo).filter(Photo.hash == result.image_id).first()
            if existing:
                logger.debug(f"Photo already indexed: {path}")
                return

            # Create photo record
            photo = Photo(
                id=result.image_id,
                path=str(path),
                hash=result.image_id,
                description=result.description,
            )

            if result.exif:
                photo.width = result.exif.width
                photo.height = result.exif.height
                photo.camera = result.exif.camera_model
                photo.location_lat = result.exif.gps_lat
                photo.location_lon = result.exif.gps_lon
                if result.exif.taken_at:
                    from datetime import datetime

                    try:
                        photo.taken_at = datetime.fromisoformat(str(result.exif.taken_at))
                    except ValueError:
                        pass

            session.add(photo)

            # Create face records
            for face in result.faces:
                face_record = Face(
                    id=face.face_id,
                    photo_id=photo.id,
                    person_id=face.person_id,
                    bbox_x=face.bbox.x,
                    bbox_y=face.bbox.y,
                    bbox_w=face.bbox.width,
                    bbox_h=face.bbox.height,
                    confidence=face.confidence,
                    embedding_id=face.face_id if face.embedding_stored else None,
                )
                session.add(face_record)

            session.commit()
            logger.debug(f"Stored results for {path}")

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_thumbnail(self, image: Image.Image, path: Path) -> Path | None:
        """Create and save a thumbnail.

        Args:
            image: PIL Image.
            path: Original image path (for naming).

        Returns:
            Path to thumbnail or None if failed.
        """
        try:
            VISION_THUMBS_DIR.mkdir(parents=True, exist_ok=True)

            thumb = image.copy()
            thumb.thumbnail(THUMB_SIZE, Image.Resampling.LANCZOS)

            thumb_name = f"{path.stem}_thumb.jpg"
            thumb_path = VISION_THUMBS_DIR / thumb_name

            thumb.save(thumb_path, "JPEG", quality=THUMB_QUALITY)
            return thumb_path

        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return None

    def cleanup(self) -> None:
        """Release all analyzer resources including shared models."""
        for analyzer in self._analyzers.values():
            analyzer.cleanup()
        self._analyzers.clear()
        self._initialized = False
        # Release shared InsightFace model
        release_face_app()


# Global pipeline instance
_pipeline: VisionPipeline | None = None


def get_pipeline() -> VisionPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = VisionPipeline()
    return _pipeline

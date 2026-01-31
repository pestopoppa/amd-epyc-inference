"""Face embedding analyzer using ArcFace."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from src.vision.analyzers.base import Analyzer, AnalyzerResult, pil_to_numpy
from src.vision.analyzers.insightface_loader import get_face_app
from src.db.chroma_client import add_face_embedding, search_faces

logger = logging.getLogger(__name__)


class FaceEmbedAnalyzer(Analyzer):
    """Generate ArcFace embeddings for detected faces.

    This analyzer should run AFTER face_detect. It uses the face detections
    to generate 512-dimensional embeddings for each face, which are stored
    in ChromaDB for similarity search.
    """

    def __init__(
        self,
        store_embeddings: bool = True,
        identify_faces: bool = True,
        identification_threshold: float = 0.6,
        **config: Any,
    ):
        """Initialize face embedding analyzer.

        Args:
            store_embeddings: Whether to store embeddings in ChromaDB.
            identify_faces: Whether to attempt face identification.
            identification_threshold: Cosine similarity threshold for identification.
            **config: Additional configuration.
        """
        super().__init__(**config)
        self.store_embeddings = store_embeddings
        self.identify_faces = identify_faces
        self.identification_threshold = identification_threshold
        self._app = None

    @property
    def name(self) -> str:
        return "face_embed"

    def initialize(self) -> None:
        """Load InsightFace model (shared with face_detect)."""
        self._app = get_face_app()
        super().initialize()

    def analyze(
        self,
        image: Image.Image,
        path: Path | None = None,
        face_detections: list[dict[str, Any]] | None = None,
    ) -> AnalyzerResult:
        """Generate embeddings for faces in image.

        Args:
            image: PIL Image.
            path: Optional path to original file.
            face_detections: Optional pre-computed face detections from face_detect.
                If provided with _embedding field, skips re-detection.

        Returns:
            AnalyzerResult with embeddings list containing:
            - face_id: UUID for this face
            - embedding: 512-dim vector (only if not stored)
            - stored: whether embedding was stored in ChromaDB
            - identified_as: person_id if identified
            - confidence: identification confidence
        """
        self.ensure_initialized()
        start = time.perf_counter()

        try:
            results = []
            img_array = pil_to_numpy(image)

            # If we have pre-computed embeddings from face_detect, use them
            if face_detections:
                for fd in face_detections:
                    if "_embedding" in fd:
                        embedding = fd["_embedding"]
                        face_id = str(uuid.uuid4())
                        result = self._process_embedding(
                            face_id=face_id,
                            embedding=embedding,
                            bbox=fd.get("bbox"),
                            image_path=str(path) if path else None,
                        )
                        results.append(result)
            else:
                # Run face detection ourselves
                faces = self._app.get(img_array)
                for face in faces:
                    if face.embedding is None:
                        continue

                    face_id = str(uuid.uuid4())
                    bbox = face.bbox.astype(int)

                    result = self._process_embedding(
                        face_id=face_id,
                        embedding=face.embedding.tolist(),
                        bbox={
                            "x": int(bbox[0]),
                            "y": int(bbox[1]),
                            "width": int(bbox[2] - bbox[0]),
                            "height": int(bbox[3] - bbox[1]),
                        },
                        image_path=str(path) if path else None,
                    )
                    results.append(result)

            elapsed = (time.perf_counter() - start) * 1000

            return AnalyzerResult(
                analyzer_name=self.name,
                success=True,
                data={"embeddings": results, "embedding_count": len(results)},
                processing_time_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"Face embedding failed: {e}")
            return AnalyzerResult(
                analyzer_name=self.name,
                success=False,
                error=str(e),
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

    def _process_embedding(
        self,
        face_id: str,
        embedding: list[float],
        bbox: dict[str, int] | None,
        image_path: str | None,
    ) -> dict[str, Any]:
        """Process a single face embedding.

        Args:
            face_id: UUID for this face.
            embedding: 512-dim embedding vector.
            bbox: Bounding box dict.
            image_path: Path to source image.

        Returns:
            Dict with face_id, stored status, identification results.
        """
        result = {"face_id": face_id, "stored": False}

        # Try to identify the face
        if self.identify_faces:
            search_result = search_faces(embedding, n_results=1)
            if search_result["ids"] and search_result["ids"][0]:
                # Convert distance to similarity (ChromaDB uses cosine distance)
                distance = search_result["distances"][0][0]
                similarity = 1 - distance

                if similarity >= self.identification_threshold:
                    metadata = search_result["metadatas"][0][0]
                    result["identified_as"] = metadata.get("person_id")
                    result["identification_confidence"] = similarity
                    result["matched_face_id"] = search_result["ids"][0][0]

        # Store embedding
        if self.store_embeddings:
            metadata = {
                "image_path": image_path or "",
            }
            if bbox:
                metadata.update({
                    "bbox_x": bbox["x"],
                    "bbox_y": bbox["y"],
                    "bbox_w": bbox["width"],
                    "bbox_h": bbox["height"],
                })
            if "identified_as" in result:
                metadata["person_id"] = result["identified_as"]

            add_face_embedding(face_id, embedding, metadata)
            result["stored"] = True

        return result

    def cleanup(self) -> None:
        """Release InsightFace reference (shared model not released)."""
        self._app = None
        self._initialized = False
        # Note: Shared model released via release_face_app() if needed

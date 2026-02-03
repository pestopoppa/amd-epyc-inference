"""Unified search interface for vision data."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from sqlalchemy import and_

from src.vision.config import VISION_THUMBS_DIR
from src.vision.models import SearchResult, SearchResponse
from src.db.chroma_client import (
    search_descriptions,
    search_faces,
    search_images,
)
from src.db.models.vision import Photo, Face, Person, get_session
from src.vision.analyzers.clip_embed import ClipEmbedAnalyzer, TextEmbedAnalyzer

logger = logging.getLogger(__name__)


class VisionSearch:
    """Unified search across vision data.

    Supports:
    - Text search on descriptions (semantic)
    - Face search by embedding similarity
    - Visual similarity via CLIP
    - Metadata filters (date, location, person)
    """

    def __init__(self):
        """Initialize search with lazy-loaded embedding models."""
        self._text_embedder: TextEmbedAnalyzer | None = None
        self._clip_embedder: ClipEmbedAnalyzer | None = None

    def _get_text_embedder(self) -> TextEmbedAnalyzer:
        """Get or create text embedder."""
        if self._text_embedder is None:
            self._text_embedder = TextEmbedAnalyzer()
            self._text_embedder.initialize()
        return self._text_embedder

    def _get_clip_embedder(self) -> ClipEmbedAnalyzer:
        """Get or create CLIP embedder."""
        if self._clip_embedder is None:
            self._clip_embedder = ClipEmbedAnalyzer(store_embeddings=False)
            self._clip_embedder.initialize()
        return self._clip_embedder

    def search(
        self,
        query: str,
        search_type: str = "description",
        filters: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> SearchResponse:
        """Search indexed images.

        Args:
            query: Text query for description/visual search, or face_id for face search.
            search_type: One of "description", "face", "visual".
            filters: Optional metadata filters:
                - date_range: [start, end] ISO dates
                - has_faces: bool
                - person_ids: list of person IDs
                - location_box: [lat_min, lat_max, lon_min, lon_max]
            limit: Maximum results.

        Returns:
            SearchResponse with matched images.
        """
        start = time.perf_counter()

        if search_type == "description":
            results = self._search_by_description(query, filters, limit)
        elif search_type == "face":
            results = self._search_by_face(query, filters, limit)
        elif search_type == "visual":
            results = self._search_visual_similarity(query, filters, limit)
        else:
            results = []

        elapsed = (time.perf_counter() - start) * 1000

        return SearchResponse(
            query=query,
            results=results,
            total_found=len(results),
            search_time_ms=elapsed,
        )

    def _search_by_description(
        self,
        query: str,
        filters: dict[str, Any] | None,
        limit: int,
    ) -> list[SearchResult]:
        """Search by text description using semantic similarity."""
        # Embed query text
        embedder = self._get_text_embedder()
        query_embedding = embedder.embed_text(query)

        # Search ChromaDB
        chroma_results = search_descriptions(
            query_embedding=query_embedding,
            n_results=limit,
        )

        if not chroma_results["ids"] or not chroma_results["ids"][0]:
            return []

        results = []
        for i, image_id in enumerate(chroma_results["ids"][0]):
            metadata = chroma_results["metadatas"][0][i] if chroma_results["metadatas"] else {}
            distance = chroma_results["distances"][0][i] if chroma_results["distances"] else 0
            document = (
                chroma_results["documents"][0][i] if chroma_results.get("documents") else None
            )

            # Convert distance to similarity score (cosine distance)
            score = 1 - distance

            result = SearchResult(
                image_id=image_id,
                path=metadata.get("path", ""),
                score=score,
                description=document,
                metadata=metadata,
            )

            # Check for thumbnail
            if result.path:
                from pathlib import Path

                thumb_path = VISION_THUMBS_DIR / f"{Path(result.path).stem}_thumb.jpg"
                if thumb_path.exists():
                    result.thumbnail_path = str(thumb_path)

            results.append(result)

        # Apply additional filters from SQLite
        if filters:
            results = self._apply_filters(results, filters)

        return results[:limit]

    def _search_by_face(
        self,
        face_id_or_embedding: str | list[float],
        filters: dict[str, Any] | None,
        limit: int,
    ) -> list[SearchResult]:
        """Search for images containing similar faces.

        Args:
            face_id_or_embedding: Either a face_id to look up, or a raw embedding.
        """
        # If it's a face_id, look up the embedding
        if isinstance(face_id_or_embedding, str):
            # Get face embedding from database
            from src.db.chroma_client import get_faces_collection

            collection = get_faces_collection()
            try:
                result = collection.get(ids=[face_id_or_embedding], include=["embeddings"])
                if not result["embeddings"]:
                    return []
                query_embedding = result["embeddings"][0]
            except Exception:
                return []
        else:
            query_embedding = face_id_or_embedding

        # Search ChromaDB faces
        chroma_results = search_faces(
            query_embedding=query_embedding,
            n_results=limit * 2,  # Get more to account for grouping
        )

        if not chroma_results["ids"] or not chroma_results["ids"][0]:
            return []

        # Group by image and take best match per image
        image_scores: dict[str, tuple[float, dict]] = {}

        for i, face_id in enumerate(chroma_results["ids"][0]):
            metadata = chroma_results["metadatas"][0][i] if chroma_results["metadatas"] else {}
            distance = chroma_results["distances"][0][i] if chroma_results["distances"] else 0
            score = 1 - distance

            image_path = metadata.get("image_path", "")
            if image_path:
                if image_path not in image_scores or score > image_scores[image_path][0]:
                    image_scores[image_path] = (score, metadata)

        results = []
        for image_path, (score, metadata) in sorted(
            image_scores.items(), key=lambda x: x[1][0], reverse=True
        )[:limit]:
            result = SearchResult(
                image_id=image_path,
                path=image_path,
                score=score,
                metadata=metadata,
            )

            # Get description from SQLite
            session = get_session()
            try:
                photo = session.query(Photo).filter(Photo.path == image_path).first()
                if photo:
                    result.description = photo.description
            finally:
                session.close()

            results.append(result)

        if filters:
            results = self._apply_filters(results, filters)

        return results

    def _search_visual_similarity(
        self,
        query: str,
        filters: dict[str, Any] | None,
        limit: int,
    ) -> list[SearchResult]:
        """Search for visually similar images using CLIP.

        Args:
            query: Text description of what to find.
        """
        # Embed query text with CLIP
        embedder = self._get_clip_embedder()
        query_embedding = embedder.embed_text(query)

        # Search CLIP embeddings
        chroma_results = search_images(
            query_embedding=query_embedding,
            n_results=limit,
        )

        if not chroma_results["ids"] or not chroma_results["ids"][0]:
            return []

        results = []
        for i, image_id in enumerate(chroma_results["ids"][0]):
            metadata = chroma_results["metadatas"][0][i] if chroma_results["metadatas"] else {}
            distance = chroma_results["distances"][0][i] if chroma_results["distances"] else 0
            score = 1 - distance

            result = SearchResult(
                image_id=image_id,
                path=metadata.get("path", image_id),
                score=score,
                metadata=metadata,
            )
            results.append(result)

        if filters:
            results = self._apply_filters(results, filters)

        return results

    def _apply_filters(
        self,
        results: list[SearchResult],
        filters: dict[str, Any],
    ) -> list[SearchResult]:
        """Apply metadata filters from SQLite.

        Args:
            results: Initial results from vector search.
            filters: Filter dict with date_range, has_faces, person_ids, etc.
        """
        if not filters:
            return results

        session = get_session()
        try:
            # Build set of valid paths based on filters
            query = session.query(Photo.path)

            # Date range filter
            if "date_range" in filters:
                start, end = filters["date_range"]
                if start:
                    query = query.filter(Photo.taken_at >= datetime.fromisoformat(start))
                if end:
                    query = query.filter(Photo.taken_at <= datetime.fromisoformat(end))

            # Has faces filter
            if filters.get("has_faces"):
                query = query.filter(Photo.faces.any())

            # Person IDs filter
            if "person_ids" in filters:
                person_ids = filters["person_ids"]
                query = query.join(Face).filter(Face.person_id.in_(person_ids))

            # Location box filter
            if "location_box" in filters:
                lat_min, lat_max, lon_min, lon_max = filters["location_box"]
                query = query.filter(
                    and_(
                        Photo.location_lat >= lat_min,
                        Photo.location_lat <= lat_max,
                        Photo.location_lon >= lon_min,
                        Photo.location_lon <= lon_max,
                    )
                )

            valid_paths = {row.path for row in query.all()}

            # Filter results
            return [r for r in results if r.path in valid_paths]

        finally:
            session.close()

    def list_persons(self, limit: int = 100) -> list[dict[str, Any]]:
        """List all known persons.

        Returns:
            List of person info dicts.
        """
        session = get_session()
        try:
            persons = session.query(Person).limit(limit).all()
            return [
                {
                    "person_id": p.id,
                    "name": p.name,
                    "photo_count": p.photo_count,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                }
                for p in persons
            ]
        finally:
            session.close()

    def update_person(
        self,
        person_id: str,
        name: str | None = None,
        merge_with: str | None = None,
    ) -> bool:
        """Update person name or merge with another person.

        Args:
            person_id: Person to update.
            name: New name (optional).
            merge_with: Person ID to merge into (optional).

        Returns:
            True if successful.
        """
        session = get_session()
        try:
            person = session.query(Person).filter(Person.id == person_id).first()
            if not person:
                return False

            if name:
                person.name = name

            if merge_with:
                # Move all faces to target person
                session.query(Face).filter(Face.person_id == person_id).update(
                    {"person_id": merge_with}
                )
                # Update photo count
                target = session.query(Person).filter(Person.id == merge_with).first()
                if target:
                    target.photo_count += person.photo_count
                # Delete old person
                session.delete(person)

            session.commit()
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update person: {e}")
            return False
        finally:
            session.close()


# Global search instance
_search: VisionSearch | None = None


def get_search() -> VisionSearch:
    """Get or create global search instance."""
    global _search
    if _search is None:
        _search = VisionSearch()
    return _search

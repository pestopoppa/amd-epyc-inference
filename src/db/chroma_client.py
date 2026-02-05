"""ChromaDB client for vision embeddings.

Manages three collections:
- faces: 512-dim ArcFace embeddings for face matching
- descriptions: Text embeddings from descriptions for semantic search
- images: CLIP embeddings for visual similarity

Note: chromadb is an optional dependency. Functions will raise ImportError
if chromadb is not installed when called.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.vision.config import (
    CHROMA_PATH,
    COLLECTION_FACES,
    COLLECTION_DESCRIPTIONS,
    COLLECTION_IMAGES,
    FACE_EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)

# Optional chromadb import - not required for CI tests
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment,misc]
    CHROMADB_AVAILABLE = False
    logger.debug("chromadb not installed - face embedding features disabled")

if TYPE_CHECKING:
    import chromadb as chromadb_types

_client: "chromadb_types.PersistentClient | None" = None


def _require_chromadb() -> None:
    """Raise ImportError if chromadb is not available."""
    if not CHROMADB_AVAILABLE:
        raise ImportError(
            "chromadb is required for face embedding features. "
            "Install with: pip install chromadb"
        )


def get_chroma_client() -> "chromadb_types.PersistentClient":
    """Get or create the ChromaDB persistent client."""
    _require_chromadb()
    global _client
    if _client is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        logger.info(f"ChromaDB client initialized at {CHROMA_PATH}")
    return _client


def get_faces_collection() -> "chromadb_types.Collection":
    """Get or create the faces collection for ArcFace embeddings."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_FACES,
        metadata={"hnsw:space": "cosine", "dimension": FACE_EMBEDDING_DIM},
    )


def get_descriptions_collection() -> "chromadb_types.Collection":
    """Get or create the descriptions collection for text embeddings."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_DESCRIPTIONS,
        metadata={"hnsw:space": "cosine"},
    )


def get_images_collection() -> "chromadb_types.Collection":
    """Get or create the images collection for CLIP embeddings."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_IMAGES,
        metadata={"hnsw:space": "cosine"},
    )


def add_face_embedding(
    face_id: str,
    embedding: list[float],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Add a face embedding to the faces collection.

    Args:
        face_id: Unique face identifier
        embedding: 512-dim ArcFace embedding
        metadata: Optional metadata (photo_id, person_id, bbox, etc.)
    """
    collection = get_faces_collection()
    collection.add(
        ids=[face_id],
        embeddings=[embedding],
        metadatas=[metadata or {}],
    )


def add_description_embedding(
    image_id: str,
    embedding: list[float],
    document: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Add a description embedding to the descriptions collection.

    Args:
        image_id: Unique image identifier
        embedding: Text embedding of the description
        document: Original description text
        metadata: Optional metadata (path, taken_at, etc.)
    """
    collection = get_descriptions_collection()
    collection.add(
        ids=[image_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata or {}],
    )


def add_image_embedding(
    image_id: str,
    embedding: list[float],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Add a CLIP embedding to the images collection.

    Args:
        image_id: Unique image identifier
        embedding: CLIP embedding
        metadata: Optional metadata
    """
    collection = get_images_collection()
    collection.add(
        ids=[image_id],
        embeddings=[embedding],
        metadatas=[metadata or {}],
    )


def search_faces(
    query_embedding: list[float],
    n_results: int = 10,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search for similar faces by embedding.

    Args:
        query_embedding: 512-dim face embedding to match
        n_results: Number of results to return
        where: Optional metadata filter

    Returns:
        ChromaDB query results dict with ids, distances, metadatas
    """
    collection = get_faces_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )


def search_descriptions(
    query_embedding: list[float],
    n_results: int = 50,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search for images by description similarity.

    Args:
        query_embedding: Text embedding of search query
        n_results: Number of results to return
        where: Optional metadata filter

    Returns:
        ChromaDB query results dict
    """
    collection = get_descriptions_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def search_images(
    query_embedding: list[float],
    n_results: int = 50,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search for visually similar images by CLIP embedding.

    Args:
        query_embedding: CLIP embedding
        n_results: Number of results to return
        where: Optional metadata filter

    Returns:
        ChromaDB query results dict
    """
    collection = get_images_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )


def delete_face(face_id: str) -> None:
    """Delete a face embedding."""
    collection = get_faces_collection()
    collection.delete(ids=[face_id])


def delete_image_embeddings(image_id: str) -> None:
    """Delete all embeddings associated with an image."""
    try:
        get_descriptions_collection().delete(ids=[image_id])
    except Exception as e:
        logger.debug("Failed to delete description embedding for %s: %s", image_id, e)
    try:
        get_images_collection().delete(ids=[image_id])
    except Exception as e:
        logger.debug("Failed to delete image embedding for %s: %s", image_id, e)


def get_collection_stats() -> dict[str, int]:
    """Get counts for all collections."""
    return {
        "faces": get_faces_collection().count(),
        "descriptions": get_descriptions_collection().count(),
        "images": get_images_collection().count(),
    }

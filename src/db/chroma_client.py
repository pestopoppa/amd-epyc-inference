"""ChromaDB client for vision embeddings.

Manages three collections:
- faces: 512-dim ArcFace embeddings for face matching
- descriptions: Text embeddings from descriptions for semantic search
- images: CLIP embeddings for visual similarity
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings

from src.vision.config import (
    CHROMA_PATH,
    COLLECTION_FACES,
    COLLECTION_DESCRIPTIONS,
    COLLECTION_IMAGES,
    FACE_EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create the ChromaDB persistent client."""
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


def get_faces_collection() -> chromadb.Collection:
    """Get or create the faces collection for ArcFace embeddings."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_FACES,
        metadata={"hnsw:space": "cosine", "dimension": FACE_EMBEDDING_DIM},
    )


def get_descriptions_collection() -> chromadb.Collection:
    """Get or create the descriptions collection for text embeddings."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_DESCRIPTIONS,
        metadata={"hnsw:space": "cosine"},
    )


def get_images_collection() -> chromadb.Collection:
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
    except Exception:
        pass
    try:
        get_images_collection().delete(ids=[image_id])
    except Exception:
        pass


def get_collection_stats() -> dict[str, int]:
    """Get counts for all collections."""
    return {
        "faces": get_faces_collection().count(),
        "descriptions": get_descriptions_collection().count(),
        "images": get_images_collection().count(),
    }

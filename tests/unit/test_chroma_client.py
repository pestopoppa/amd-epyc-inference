"""Tests for ChromaDB client (src/db/chroma_client.py).

Tests CRUD operations on faces, descriptions, and images collections
using mocked chromadb client.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.db import chroma_client


@pytest.fixture(autouse=True)
def reset_client():
    """Reset the module-level singleton before each test."""
    original = chroma_client._client
    chroma_client._client = None
    yield
    chroma_client._client = original


@pytest.fixture
def mock_chroma():
    """Create a mock chromadb PersistentClient with collections."""
    mock_client = MagicMock()
    mock_faces = MagicMock()
    mock_descriptions = MagicMock()
    mock_images = MagicMock()

    def get_or_create(name, **kwargs):
        if name == "faces":
            return mock_faces
        elif name == "descriptions":
            return mock_descriptions
        elif name == "images":
            return mock_images
        return MagicMock()

    mock_client.get_or_create_collection.side_effect = get_or_create

    with patch("src.db.chroma_client.CHROMADB_AVAILABLE", True):
        with patch("src.db.chroma_client.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            with patch("src.db.chroma_client.CHROMA_PATH") as mock_path:
                mock_path.mkdir = MagicMock()
                yield {
                    "client": mock_client,
                    "faces": mock_faces,
                    "descriptions": mock_descriptions,
                    "images": mock_images,
                }


class TestGetChromaClient:
    """Test client singleton."""

    def test_creates_client_on_first_call(self, mock_chroma):
        client = chroma_client.get_chroma_client()
        assert client is mock_chroma["client"]

    def test_reuses_client_on_second_call(self, mock_chroma):
        client1 = chroma_client.get_chroma_client()
        client2 = chroma_client.get_chroma_client()
        assert client1 is client2

    def test_raises_when_chromadb_unavailable(self):
        """Should raise ImportError when chromadb is not installed."""
        with patch("src.db.chroma_client.CHROMADB_AVAILABLE", False):
            with pytest.raises(ImportError) as exc_info:
                chroma_client.get_chroma_client()
            assert "chromadb is required" in str(exc_info.value)


class TestCollections:
    """Test collection getters."""

    def test_get_faces_collection(self, mock_chroma):
        coll = chroma_client.get_faces_collection()
        assert coll is mock_chroma["faces"]

    def test_get_descriptions_collection(self, mock_chroma):
        coll = chroma_client.get_descriptions_collection()
        assert coll is mock_chroma["descriptions"]

    def test_get_images_collection(self, mock_chroma):
        coll = chroma_client.get_images_collection()
        assert coll is mock_chroma["images"]


class TestAddOperations:
    """Test add operations."""

    def test_add_face_embedding(self, mock_chroma):
        chroma_client.add_face_embedding("face1", [0.1] * 512, {"person": "alice"})
        mock_chroma["faces"].add.assert_called_once_with(
            ids=["face1"],
            embeddings=[[0.1] * 512],
            metadatas=[{"person": "alice"}],
        )

    def test_add_face_embedding_no_metadata(self, mock_chroma):
        chroma_client.add_face_embedding("face2", [0.2] * 512)
        mock_chroma["faces"].add.assert_called_once_with(
            ids=["face2"],
            embeddings=[[0.2] * 512],
            metadatas=[{}],
        )

    def test_add_description_embedding(self, mock_chroma):
        chroma_client.add_description_embedding(
            "img1", [0.3] * 384, "A blue sky", {"path": "/photo.jpg"}
        )
        mock_chroma["descriptions"].add.assert_called_once_with(
            ids=["img1"],
            embeddings=[[0.3] * 384],
            documents=["A blue sky"],
            metadatas=[{"path": "/photo.jpg"}],
        )

    def test_add_image_embedding(self, mock_chroma):
        chroma_client.add_image_embedding("img2", [0.4] * 512, {"taken_at": "2026-01-01"})
        mock_chroma["images"].add.assert_called_once_with(
            ids=["img2"],
            embeddings=[[0.4] * 512],
            metadatas=[{"taken_at": "2026-01-01"}],
        )


class TestSearchOperations:
    """Test search operations."""

    def test_search_faces(self, mock_chroma):
        mock_chroma["faces"].query.return_value = {"ids": [["face1"]], "distances": [[0.1]]}
        result = chroma_client.search_faces([0.1] * 512, n_results=5)
        mock_chroma["faces"].query.assert_called_once_with(
            query_embeddings=[[0.1] * 512],
            n_results=5,
            where=None,
        )
        assert result["ids"] == [["face1"]]

    def test_search_faces_with_filter(self, mock_chroma):
        chroma_client.search_faces([0.1] * 512, n_results=3, where={"person": "alice"})
        mock_chroma["faces"].query.assert_called_once_with(
            query_embeddings=[[0.1] * 512],
            n_results=3,
            where={"person": "alice"},
        )

    def test_search_descriptions(self, mock_chroma):
        mock_chroma["descriptions"].query.return_value = {
            "ids": [["img1"]],
            "documents": [["A blue sky"]],
        }
        result = chroma_client.search_descriptions([0.3] * 384, n_results=10)
        assert result["ids"] == [["img1"]]

    def test_search_images(self, mock_chroma):
        mock_chroma["images"].query.return_value = {"ids": [["img2"]], "distances": [[0.2]]}
        result = chroma_client.search_images([0.4] * 512, n_results=20)
        assert result["ids"] == [["img2"]]


class TestDeleteOperations:
    """Test delete operations."""

    def test_delete_face(self, mock_chroma):
        chroma_client.delete_face("face1")
        mock_chroma["faces"].delete.assert_called_once_with(ids=["face1"])

    def test_delete_image_embeddings(self, mock_chroma):
        chroma_client.delete_image_embeddings("img1")
        mock_chroma["descriptions"].delete.assert_called_once_with(ids=["img1"])
        mock_chroma["images"].delete.assert_called_once_with(ids=["img1"])

    def test_delete_image_embeddings_handles_missing(self, mock_chroma):
        """Deleting non-existent embeddings should not raise."""
        mock_chroma["descriptions"].delete.side_effect = Exception("not found")
        mock_chroma["images"].delete.side_effect = Exception("not found")
        # Should not raise
        chroma_client.delete_image_embeddings("nonexistent")


class TestCollectionStats:
    """Test stats."""

    def test_get_collection_stats(self, mock_chroma):
        mock_chroma["faces"].count.return_value = 100
        mock_chroma["descriptions"].count.return_value = 500
        mock_chroma["images"].count.return_value = 500

        stats = chroma_client.get_collection_stats()
        assert stats == {"faces": 100, "descriptions": 500, "images": 500}

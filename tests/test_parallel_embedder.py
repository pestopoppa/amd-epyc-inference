"""
Unit tests for ParallelEmbedderClient.

Tests cover:
- Basic embedding generation (sync and async)
- Parallel fan-out behavior
- Fallback to hash when all servers fail
- Server health tracking and backoff
- Batch embedding
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


class TestEmbedderPoolConfig:
    """Tests for EmbedderPoolConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from orchestration.repl_memory.parallel_embedder import EmbedderPoolConfig

        config = EmbedderPoolConfig()

        assert len(config.server_urls) == 6
        assert config.server_urls[0] == "http://127.0.0.1:8090"
        assert config.server_urls[5] == "http://127.0.0.1:8095"
        assert config.embedding_dim == 1024
        assert config.request_timeout == 2.0
        assert config.connect_timeout == 1.0
        assert config.use_fallback is True

    def test_custom_config(self):
        """Test custom configuration."""
        from orchestration.repl_memory.parallel_embedder import EmbedderPoolConfig

        config = EmbedderPoolConfig(
            server_urls=["http://localhost:9000"],
            embedding_dim=768,
            request_timeout=5.0,
        )

        assert len(config.server_urls) == 1
        assert config.embedding_dim == 768
        assert config.request_timeout == 5.0


class TestParallelEmbedderClient:
    """Tests for ParallelEmbedderClient class."""

    @pytest.fixture
    def mock_config(self):
        """Create a config with mock server URLs."""
        from orchestration.repl_memory.parallel_embedder import EmbedderPoolConfig

        return EmbedderPoolConfig(
            server_urls=["http://mock1:8090", "http://mock2:8091"],
            embedding_dim=1024,
            use_fallback=True,
        )

    @pytest.fixture
    def client(self, mock_config):
        """Create a ParallelEmbedderClient with mock config."""
        from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient

        return ParallelEmbedderClient(config=mock_config)

    def test_init(self, client, mock_config):
        """Test client initialization."""
        assert client.config == mock_config
        assert client._http_client is None
        assert client._server_health == {}

    def test_embedding_dim_property(self, client):
        """Test embedding_dim property."""
        assert client.embedding_dim == 1024

    def test_fallback_generates_correct_dimension(self, client):
        """Test hash fallback generates correct dimension."""
        embedding = client._generate_fallback("test text")

        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32
        # Should be normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_fallback_deterministic(self, client):
        """Test hash fallback is deterministic."""
        emb1 = client._generate_fallback("test text")
        emb2 = client._generate_fallback("test text")

        np.testing.assert_array_equal(emb1, emb2)

    def test_fallback_different_texts(self, client):
        """Test different texts produce different fallback embeddings."""
        emb1 = client._generate_fallback("text one")
        emb2 = client._generate_fallback("text two")

        assert not np.allclose(emb1, emb2)

    def test_server_health_tracking(self, client):
        """Test server health marking."""
        url = "http://mock1:8090"

        # Initially healthy
        assert client._is_server_healthy(url)

        # Mark as failed
        client._mark_server_failed(url)
        # Should be unhealthy immediately after (within backoff period)
        # Note: depends on backoff_base config
        assert url in client._server_health

        # Mark as healthy
        client._mark_server_healthy(url)
        assert client._is_server_healthy(url)
        assert url not in client._server_health


class TestParallelEmbedderAsync:
    """Async tests for ParallelEmbedderClient."""

    @pytest.fixture
    def mock_config(self):
        """Create a config with mock server URLs."""
        from orchestration.repl_memory.parallel_embedder import EmbedderPoolConfig

        return EmbedderPoolConfig(
            server_urls=["http://mock1:8090", "http://mock2:8091"],
            embedding_dim=1024,
            use_fallback=True,
        )

    @pytest.mark.asyncio
    async def test_embed_async_with_mock_server(self, mock_config):
        """Test embed_async with mocked HTTP response."""
        from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient

        client = ParallelEmbedderClient(config=mock_config)

        # Create a mock embedding response
        mock_embedding = np.random.randn(1024).astype(np.float32).tolist()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": mock_embedding}]
        }

        # Mock the HTTP client
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        client._http_client = mock_client
        client._closed = False

        embedding = await client.embed_async("test text")

        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32
        # Should be normalized
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

        await client.close()

    @pytest.mark.asyncio
    async def test_embed_async_fallback_on_all_failures(self, mock_config):
        """Test fallback when all servers fail."""
        from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient

        client = ParallelEmbedderClient(config=mock_config)

        # Mock HTTP client to always fail
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection failed"))
        client._http_client = mock_client
        client._closed = False

        embedding = await client.embed_async("test text")

        # Should fall back to hash
        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32

        await client.close()

    @pytest.mark.asyncio
    async def test_embed_async_no_fallback_raises(self):
        """Test that error is raised when fallback disabled and all fail."""
        from orchestration.repl_memory.parallel_embedder import (
            EmbedderPoolConfig,
            ParallelEmbedderClient,
        )

        config = EmbedderPoolConfig(
            server_urls=["http://mock:8090"],
            use_fallback=False,
        )
        client = ParallelEmbedderClient(config=config)

        # Mock HTTP client to fail
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection failed"))
        client._http_client = mock_client
        client._closed = False

        with pytest.raises(RuntimeError, match="All embedding servers failed"):
            await client.embed_async("test text")

        await client.close()

    @pytest.mark.asyncio
    async def test_embed_batch_async(self, mock_config):
        """Test batch embedding."""
        from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient

        client = ParallelEmbedderClient(config=mock_config)

        # Mock HTTP client to return embeddings
        def make_response():
            mock_embedding = np.random.randn(1024).astype(np.float32).tolist()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "data": [{"embedding": mock_embedding}]
            }
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=lambda *args, **kwargs: make_response())
        client._http_client = mock_client
        client._closed = False

        texts = ["text one", "text two", "text three"]
        embeddings = await client.embed_batch_async(texts)

        assert embeddings.shape == (3, 1024)
        assert embeddings.dtype == np.float32

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_all(self, mock_config):
        """Test health check across all servers."""
        from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient

        client = ParallelEmbedderClient(config=mock_config)

        # Mock health responses
        async def mock_get(url, **kwargs):
            response = MagicMock()
            if "mock1" in url:
                response.status_code = 200
            else:
                response.status_code = 500
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        client._http_client = mock_client
        client._closed = False

        health = await client.health_check_all()

        assert health["http://mock1:8090"] is True
        assert health["http://mock2:8091"] is False

        await client.close()


class TestParallelEmbedderSync:
    """Sync wrapper tests for ParallelEmbedderClient."""

    @pytest.fixture
    def mock_config(self):
        """Create a config with mock server URLs."""
        from orchestration.repl_memory.parallel_embedder import EmbedderPoolConfig

        return EmbedderPoolConfig(
            server_urls=["http://mock:8090"],
            embedding_dim=1024,
            use_fallback=True,
        )

    def test_embed_sync_uses_fallback(self, mock_config):
        """Test sync wrapper falls back to hash when servers unavailable."""
        from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient

        # Create client with no running servers (will use fallback)
        client = ParallelEmbedderClient(config=mock_config)

        embedding = client.embed_sync("test text")

        assert embedding.shape == (1024,)
        assert embedding.dtype == np.float32


class TestIntegrationWithFAISSStore:
    """Integration tests with FAISSEmbeddingStore."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for test data."""
        return tmp_path

    def test_parallel_embedder_with_faiss_store(self, temp_dir):
        """Test that parallel embedder embeddings work with FAISS store."""
        from orchestration.repl_memory.faiss_store import FAISSEmbeddingStore
        from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient

        # Create FAISS store with 1024 dims (matching BGE-large)
        store = FAISSEmbeddingStore(path=temp_dir, dim=1024)

        # Create embedder (will use fallback since no servers)
        client = ParallelEmbedderClient()

        # Generate embedding and add to store
        embedding = client.embed_sync("test text for FAISS")
        memory_id = store.add("memory_1", embedding)

        assert memory_id == 0
        assert store.count == 1

        # Search should work
        results = store.search(embedding, k=1)
        assert len(results) == 1
        assert results[0][0] == "memory_1"
        assert results[0][1] > 0.9  # High similarity for same embedding

    def test_dimension_consistency(self, temp_dir):
        """Test dimension consistency between embedder and store."""
        from orchestration.repl_memory.faiss_store import FAISSEmbeddingStore
        from orchestration.repl_memory.parallel_embedder import ParallelEmbedderClient

        client = ParallelEmbedderClient()
        store = FAISSEmbeddingStore(path=temp_dir, dim=client.embedding_dim)

        # Add multiple embeddings
        texts = ["first text", "second text", "third text"]
        for i, text in enumerate(texts):
            embedding = client.embed_sync(text)
            store.add(f"memory_{i}", embedding)

        assert store.count == 3
        assert store.dim == 1024
        assert client.embedding_dim == 1024

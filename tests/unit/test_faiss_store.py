"""
Unit tests for FAISS embedding store and EpisodicStore FAISS backend.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestFAISSEmbeddingStore:
    """Tests for FAISSEmbeddingStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create a FAISSEmbeddingStore instance."""
        from orchestration.repl_memory.faiss_store import FAISSEmbeddingStore

        return FAISSEmbeddingStore(path=temp_dir, dim=128)

    def test_init_creates_empty_index(self, store):
        """Test that initialization creates an empty index."""
        assert store.count == 0
        assert store.index.ntotal == 0
        assert len(store.id_map) == 0

    def test_add_single_embedding(self, store):
        """Test adding a single embedding."""
        embedding = np.random.randn(128).astype(np.float32)
        idx = store.add("memory_1", embedding)

        assert idx == 0
        assert store.count == 1
        assert store.id_map[0] == "memory_1"

    def test_add_multiple_embeddings(self, store):
        """Test adding multiple embeddings."""
        for i in range(10):
            embedding = np.random.randn(128).astype(np.float32)
            idx = store.add(f"memory_{i}", embedding)
            assert idx == i

        assert store.count == 10
        assert len(store.id_map) == 10

    def test_add_dimension_mismatch_raises(self, store):
        """Test that dimension mismatch raises ValueError."""
        wrong_dim_embedding = np.random.randn(64).astype(np.float32)

        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add("memory_1", wrong_dim_embedding)

    def test_search_empty_index(self, store):
        """Test searching an empty index returns empty list."""
        query = np.random.randn(128).astype(np.float32)
        results = store.search(query, k=10)

        assert results == []

    def test_search_returns_correct_format(self, store):
        """Test search returns list of (memory_id, score) tuples."""
        # Add some embeddings
        for i in range(5):
            embedding = np.random.randn(128).astype(np.float32)
            store.add(f"memory_{i}", embedding)

        query = np.random.randn(128).astype(np.float32)
        results = store.search(query, k=3)

        assert len(results) == 3
        for memory_id, score in results:
            assert isinstance(memory_id, str)
            assert memory_id.startswith("memory_")
            assert isinstance(score, float)
            # Cosine similarity after L2 norm is in [-1, 1]
            assert -1.0 <= score <= 1.0

    def test_search_finds_exact_match(self, store):
        """Test that searching for an identical embedding returns it first."""
        # Add a specific embedding
        target_embedding = np.array([1.0] * 64 + [0.0] * 64, dtype=np.float32)
        store.add("target", target_embedding)

        # Add some noise embeddings
        for i in range(10):
            noise = np.random.randn(128).astype(np.float32)
            store.add(f"noise_{i}", noise)

        # Search with the same embedding
        results = store.search(target_embedding, k=5)

        # Target should be first with high similarity
        assert results[0][0] == "target"
        assert results[0][1] > 0.9  # High similarity for identical embedding

    def test_search_k_larger_than_index(self, store):
        """Test searching with k larger than index size."""
        for i in range(3):
            embedding = np.random.randn(128).astype(np.float32)
            store.add(f"memory_{i}", embedding)

        query = np.random.randn(128).astype(np.float32)
        results = store.search(query, k=100)

        # Should return all 3, not error
        assert len(results) == 3

    def test_search_dimension_mismatch_raises(self, store):
        """Test that query dimension mismatch raises ValueError."""
        embedding = np.random.randn(128).astype(np.float32)
        store.add("memory_1", embedding)

        wrong_dim_query = np.random.randn(64).astype(np.float32)

        with pytest.raises(ValueError, match="dimension mismatch"):
            store.search(wrong_dim_query, k=1)

    def test_save_and_load(self, temp_dir):
        """Test persistence: save and reload."""
        from orchestration.repl_memory.faiss_store import FAISSEmbeddingStore

        # Create and populate store
        store1 = FAISSEmbeddingStore(path=temp_dir, dim=128)
        embeddings = []
        for i in range(5):
            emb = np.random.randn(128).astype(np.float32)
            embeddings.append(emb)
            store1.add(f"memory_{i}", emb)

        store1.save()

        # Reload in new instance
        store2 = FAISSEmbeddingStore(path=temp_dir, dim=128)

        assert store2.count == 5
        assert store2.id_map == store1.id_map

        # Search should work
        results = store2.search(embeddings[0], k=1)
        assert results[0][0] == "memory_0"

    def test_get_embedding(self, store):
        """Test retrieving embedding by index."""
        original = np.array([1.0, 2.0] + [0.0] * 126, dtype=np.float32)
        store.add("memory_1", original)

        retrieved = store.get_embedding(0)

        assert retrieved is not None
        # After L2 normalization, the values will be different but direction same
        assert retrieved.shape == (128,)

    def test_get_embedding_invalid_index(self, store):
        """Test get_embedding with invalid index returns None."""
        assert store.get_embedding(-1) is None
        assert store.get_embedding(100) is None

    def test_get_memory_id(self, store):
        """Test get_memory_id retrieves correct ID."""
        store.add("memory_abc", np.random.randn(128).astype(np.float32))
        store.add("memory_xyz", np.random.randn(128).astype(np.float32))

        assert store.get_memory_id(0) == "memory_abc"
        assert store.get_memory_id(1) == "memory_xyz"
        assert store.get_memory_id(99) is None

    def test_get_index(self, store):
        """Test get_index retrieves correct index."""
        store.add("memory_abc", np.random.randn(128).astype(np.float32))
        store.add("memory_xyz", np.random.randn(128).astype(np.float32))

        assert store.get_index("memory_abc") == 0
        assert store.get_index("memory_xyz") == 1
        assert store.get_index("nonexistent") is None


class TestNumpyEmbeddingStore:
    """Tests for NumpyEmbeddingStore (legacy backend)."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create a NumpyEmbeddingStore instance."""
        from orchestration.repl_memory.faiss_store import NumpyEmbeddingStore

        return NumpyEmbeddingStore(path=temp_dir, dim=128)

    def test_add_and_search(self, store):
        """Test basic add and search functionality."""
        for i in range(5):
            embedding = np.random.randn(128).astype(np.float32)
            store.add(f"memory_{i}", embedding)

        query = np.random.randn(128).astype(np.float32)
        results = store.search(query, k=3)

        assert len(results) == 3
        for memory_id, score in results:
            assert isinstance(memory_id, str)
            assert isinstance(score, float)

    def test_save_and_load(self, temp_dir):
        """Test persistence."""
        from orchestration.repl_memory.faiss_store import NumpyEmbeddingStore

        store1 = NumpyEmbeddingStore(path=temp_dir, dim=128)
        for i in range(3):
            store1.add(f"memory_{i}", np.random.randn(128).astype(np.float32))
        store1.save()

        store2 = NumpyEmbeddingStore(path=temp_dir, dim=128)
        assert store2.count == 3


class TestEpisodicStoreWithFAISS:
    """Integration tests for EpisodicStore with FAISS backend."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def faiss_store(self, temp_dir):
        """Create an EpisodicStore with FAISS backend."""
        from orchestration.repl_memory import EpisodicStore

        return EpisodicStore(db_path=temp_dir, embedding_dim=128, use_faiss=True)

    @pytest.fixture
    def numpy_store(self, temp_dir):
        """Create an EpisodicStore with NumPy backend."""
        from orchestration.repl_memory import EpisodicStore

        return EpisodicStore(db_path=temp_dir, embedding_dim=128, use_faiss=False)

    def test_store_and_retrieve(self, faiss_store):
        """Test storing and retrieving memories."""
        embedding = np.random.randn(128).astype(np.float32)

        memory_id = faiss_store.store(
            embedding=embedding,
            action="route_to_coder",
            action_type="routing",
            context={"task_type": "code", "objective": "Fix bug"},
        )

        assert memory_id is not None
        assert faiss_store.count() == 1

        # Retrieve by ID
        memory = faiss_store.get_by_id(memory_id)
        assert memory is not None
        assert memory.action == "route_to_coder"
        assert memory.action_type == "routing"

    def test_retrieve_by_similarity(self, faiss_store):
        """Test similarity-based retrieval."""
        # Store some memories with different embeddings
        base_embedding = np.zeros(128, dtype=np.float32)

        for i in range(5):
            embedding = base_embedding.copy()
            embedding[i] = 1.0  # Make each embedding slightly different
            faiss_store.store(
                embedding=embedding,
                action=f"action_{i}",
                action_type="routing",
                context={"i": i},
            )

        # Query with embedding similar to first one
        query = base_embedding.copy()
        query[0] = 1.0

        results = faiss_store.retrieve_by_similarity(query, k=3)

        assert len(results) == 3
        # First result should be most similar
        assert results[0].action == "action_0"
        assert results[0].similarity_score > 0

    def test_retrieve_with_filters(self, faiss_store):
        """Test retrieval with action_type and q_value filters."""
        embedding = np.random.randn(128).astype(np.float32)

        # Store memories with different types and q-values
        faiss_store.store(
            embedding=embedding,
            action="action_1",
            action_type="routing",
            context={},
            initial_q=0.8,
        )
        faiss_store.store(
            embedding=embedding,
            action="action_2",
            action_type="escalation",
            context={},
            initial_q=0.3,
        )

        # Filter by action_type
        results = faiss_store.retrieve_by_similarity(embedding, k=10, action_type="routing")
        assert len(results) == 1
        assert results[0].action == "action_1"

        # Filter by min_q_value
        results = faiss_store.retrieve_by_similarity(embedding, k=10, min_q_value=0.5)
        assert len(results) == 1
        assert results[0].q_value >= 0.5

    def test_update_q_value(self, faiss_store):
        """Test Q-value updates."""
        embedding = np.random.randn(128).astype(np.float32)

        memory_id = faiss_store.store(
            embedding=embedding,
            action="action_1",
            action_type="routing",
            context={},
            initial_q=0.5,
        )

        # Update with high reward
        new_q = faiss_store.update_q_value(memory_id, reward=1.0, learning_rate=0.1)

        assert new_q > 0.5
        memory = faiss_store.get_by_id(memory_id)
        assert memory.q_value == new_q
        assert memory.update_count == 1

    def test_get_stats(self, faiss_store):
        """Test statistics reporting."""
        embedding = np.random.randn(128).astype(np.float32)

        faiss_store.store(
            embedding=embedding,
            action="action_1",
            action_type="routing",
            context={},
        )

        stats = faiss_store.get_stats()

        assert stats["total_memories"] == 1
        assert stats["backend"] == "faiss"
        assert stats["embeddings_count"] == 1
        assert "routing" in stats["by_action_type"]

    def test_backend_flag(self, temp_dir):
        """Test that use_faiss flag selects correct backend."""
        from orchestration.repl_memory import EpisodicStore

        faiss_store = EpisodicStore(db_path=temp_dir / "faiss", use_faiss=True)
        numpy_store = EpisodicStore(db_path=temp_dir / "numpy", use_faiss=False)

        assert faiss_store.use_faiss is True
        assert numpy_store.use_faiss is False

        faiss_stats = faiss_store.get_stats()
        numpy_stats = numpy_store.get_stats()

        assert faiss_stats["backend"] == "faiss"
        assert numpy_stats["backend"] == "numpy"

    def test_similarity_score_populated(self, faiss_store):
        """Test that similarity_score is populated in results."""
        embedding = np.random.randn(128).astype(np.float32)

        faiss_store.store(
            embedding=embedding,
            action="action_1",
            action_type="routing",
            context={},
        )

        results = faiss_store.retrieve_by_similarity(embedding, k=1)

        assert len(results) == 1
        assert results[0].similarity_score > 0.9  # Should be very high for same embedding


class TestBackendEquivalence:
    """Tests to verify FAISS and NumPy backends produce equivalent results."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_search_results_overlap(self, temp_dir):
        """Test that both backends return similar results for the same data."""
        from orchestration.repl_memory import EpisodicStore

        # Create both stores
        faiss_store = EpisodicStore(db_path=temp_dir / "faiss", embedding_dim=128, use_faiss=True)
        numpy_store = EpisodicStore(db_path=temp_dir / "numpy", embedding_dim=128, use_faiss=False)

        # Add same data to both
        np.random.seed(42)
        embeddings = [np.random.randn(128).astype(np.float32) for _ in range(20)]

        for i, emb in enumerate(embeddings):
            faiss_store.store(
                embedding=emb,
                action=f"action_{i}",
                action_type="routing",
                context={"i": i},
            )
            numpy_store.store(
                embedding=emb,
                action=f"action_{i}",
                action_type="routing",
                context={"i": i},
            )

        # Query with random embedding
        query = np.random.randn(128).astype(np.float32)

        faiss_results = faiss_store.retrieve_by_similarity(query, k=10)
        numpy_results = numpy_store.retrieve_by_similarity(query, k=10)

        # Extract actions for comparison
        faiss_actions = {r.action for r in faiss_results}
        numpy_actions = {r.action for r in numpy_results}

        # Should have significant overlap (>80%)
        overlap = len(faiss_actions & numpy_actions) / 10
        assert overlap >= 0.8, f"Only {overlap * 100}% overlap between backends"

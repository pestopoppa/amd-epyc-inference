"""Unit tests for MemRL service module.

Tests cover:
- Lazy import loading based on feature flags
- MemRL initialization
- Failure recovery
- Q-scoring with and without scorer
"""

from unittest.mock import MagicMock, patch

import pytest

from src.api.services import memrl


class TestLoadOptionalImports:
    """Test optional import loading."""

    def test_load_optional_imports_with_memrl_enabled(self):
        """Test loading imports when memrl feature is enabled."""
        # Reset module globals
        memrl.ProgressLogger = None
        memrl.EpisodicStore = None
        memrl.TaskEmbedder = None

        with patch("src.api.services.memrl.features") as mock_features:
            mock_feat = MagicMock()
            mock_feat.memrl = True
            mock_feat.tools = False
            mock_feat.scripts = False
            mock_features.return_value = mock_feat

            # Mock the imports
            with patch.dict("sys.modules", {
                "orchestration.repl_memory.progress_logger": MagicMock(),
                "orchestration.repl_memory.episodic_store": MagicMock(),
                "orchestration.repl_memory.embedder": MagicMock(),
                "orchestration.repl_memory.q_scorer": MagicMock(),
                "orchestration.repl_memory.retriever": MagicMock(),
            }):
                memrl.load_optional_imports()

        # Imports should be populated (though with mocks)
        # The actual test is that no exception was raised

    def test_load_optional_imports_with_memrl_disabled(self):
        """Test loading imports when memrl feature is disabled."""
        # Reset module globals
        memrl.ProgressLogger = None
        memrl.EpisodicStore = None

        with patch("src.api.services.memrl.features") as mock_features:
            mock_feat = MagicMock()
            mock_feat.memrl = False
            mock_feat.tools = False
            mock_feat.scripts = False
            mock_features.return_value = mock_feat

            with patch.dict("sys.modules", {
                "orchestration.repl_memory.progress_logger": MagicMock(),
            }):
                memrl.load_optional_imports()

        # Only ProgressLogger should be loaded (minimal mode)
        # EpisodicStore should remain None
        # The test is that no exception was raised

    def test_load_optional_imports_with_tools_enabled(self):
        """Test loading imports when tools feature is enabled."""
        memrl.ToolRegistry = None

        with patch("src.api.services.memrl.features") as mock_features:
            mock_feat = MagicMock()
            mock_feat.memrl = False
            mock_feat.tools = True
            mock_feat.scripts = False
            mock_features.return_value = mock_feat

            with patch.dict("sys.modules", {
                "orchestration.repl_memory.progress_logger": MagicMock(),
                "src.tool_registry": MagicMock(),
            }):
                memrl.load_optional_imports()

        # No exception should be raised

    def test_load_optional_imports_with_scripts_enabled(self):
        """Test loading imports when scripts feature is enabled."""
        memrl.ScriptRegistry = None

        with patch("src.api.services.memrl.features") as mock_features:
            mock_feat = MagicMock()
            mock_feat.memrl = False
            mock_feat.tools = False
            mock_feat.scripts = True
            mock_features.return_value = mock_feat

            with patch.dict("sys.modules", {
                "orchestration.repl_memory.progress_logger": MagicMock(),
                "src.script_registry": MagicMock(),
            }):
                memrl.load_optional_imports()

        # No exception should be raised


class TestEnsureMemRLInitialized:
    """Test MemRL initialization."""

    def test_ensure_memrl_initialized_feature_disabled(self):
        """Test initialization returns False when feature is disabled."""
        mock_state = MagicMock()
        mock_state._memrl_initialized = False

        with patch("src.api.services.memrl.features") as mock_features:
            mock_feat = MagicMock()
            mock_feat.memrl = False
            mock_features.return_value = mock_feat

            result = memrl.ensure_memrl_initialized(mock_state)

        assert result is False
        # Should not attempt initialization
        assert mock_state._memrl_initialized is False

    def test_ensure_memrl_initialized_already_initialized(self):
        """Test initialization is idempotent."""
        mock_state = MagicMock()
        mock_state._memrl_initialized = True
        mock_state.q_scorer = MagicMock()

        with patch("src.api.services.memrl.features") as mock_features:
            mock_feat = MagicMock()
            mock_feat.memrl = True
            mock_features.return_value = mock_feat

            result = memrl.ensure_memrl_initialized(mock_state)

        assert result is True
        # Should return existing q_scorer status

    def test_ensure_memrl_initialized_missing_imports(self):
        """Test initialization fails gracefully when imports missing."""
        mock_state = MagicMock()
        mock_state._memrl_initialized = False

        # Set imports to None to simulate missing
        original_episodic_store = memrl.EpisodicStore
        original_task_embedder = memrl.TaskEmbedder
        original_q_scorer = memrl.QScorer

        try:
            memrl.EpisodicStore = None
            memrl.TaskEmbedder = None
            memrl.QScorer = None

            with patch("src.api.services.memrl.features") as mock_features:
                mock_feat = MagicMock()
                mock_feat.memrl = True
                mock_features.return_value = mock_feat

                result = memrl.ensure_memrl_initialized(mock_state)

            assert result is False
            assert mock_state._memrl_initialized is True  # Marked to prevent retries

        finally:
            # Restore
            memrl.EpisodicStore = original_episodic_store
            memrl.TaskEmbedder = original_task_embedder
            memrl.QScorer = original_q_scorer

    def test_ensure_memrl_initialized_success(self):
        """Test successful initialization."""
        mock_state = MagicMock()
        mock_state._memrl_initialized = False
        mock_state.registry = MagicMock()
        mock_state.registry.routing_hints = {}
        mock_state.progress_logger = MagicMock()

        # Mock the classes
        mock_episodic = MagicMock()
        mock_embedder = MagicMock()
        mock_q_scorer_class = MagicMock()
        mock_scorer_instance = MagicMock()
        mock_q_scorer_class.return_value = mock_scorer_instance
        mock_scoring_config = MagicMock()
        mock_retriever_class = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever_class.return_value = mock_retriever
        mock_hybrid_router_class = MagicMock()
        mock_rule_router_class = MagicMock()
        mock_progress_reader = MagicMock()

        original_vals = (
            memrl.EpisodicStore,
            memrl.TaskEmbedder,
            memrl.QScorer,
            memrl.ScoringConfig,
            memrl.TwoPhaseRetriever,
            memrl.HybridRouter,
            memrl.RuleBasedRouter,
            memrl.ProgressReader,
        )

        try:
            memrl.EpisodicStore = mock_episodic
            memrl.TaskEmbedder = mock_embedder
            memrl.QScorer = mock_q_scorer_class
            memrl.ScoringConfig = mock_scoring_config
            memrl.TwoPhaseRetriever = mock_retriever_class
            memrl.HybridRouter = mock_hybrid_router_class
            memrl.RuleBasedRouter = mock_rule_router_class
            memrl.ProgressReader = mock_progress_reader

            with patch("src.api.services.memrl.features") as mock_features:
                mock_feat = MagicMock()
                mock_feat.memrl = True
                mock_feat.specialist_routing = False
                mock_features.return_value = mock_feat

                result = memrl.ensure_memrl_initialized(mock_state)

            assert result is True
            assert mock_state._memrl_initialized is True
            # Q-scorer should be created
            mock_q_scorer_class.assert_called_once()

        finally:
            # Restore
            (
                memrl.EpisodicStore,
                memrl.TaskEmbedder,
                memrl.QScorer,
                memrl.ScoringConfig,
                memrl.TwoPhaseRetriever,
                memrl.HybridRouter,
                memrl.RuleBasedRouter,
                memrl.ProgressReader,
            ) = original_vals

    def test_ensure_memrl_initialized_handles_exception(self):
        """Test initialization handles exceptions gracefully."""
        mock_state = MagicMock()
        mock_state._memrl_initialized = False
        mock_state.progress_logger = MagicMock()

        # Mock classes that will raise on instantiation
        mock_episodic = MagicMock(side_effect=RuntimeError("Init failed"))

        original_episodic = memrl.EpisodicStore
        original_embedder = memrl.TaskEmbedder
        original_scorer = memrl.QScorer

        try:
            memrl.EpisodicStore = mock_episodic
            memrl.TaskEmbedder = MagicMock()
            memrl.QScorer = MagicMock()

            with patch("src.api.services.memrl.features") as mock_features:
                mock_feat = MagicMock()
                mock_feat.memrl = True
                mock_features.return_value = mock_feat

                result = memrl.ensure_memrl_initialized(mock_state)

            assert result is False
            # State should be marked initialized to prevent retry loops
            assert mock_state._memrl_initialized is True
            # Components should be cleared
            assert mock_state.q_scorer is None
            assert mock_state.episodic_store is None

        finally:
            memrl.EpisodicStore = original_episodic
            memrl.TaskEmbedder = original_embedder
            memrl.QScorer = original_scorer


class TestScoreCompletedTask:
    """Test task scoring functions."""

    def setup_method(self):
        """Ensure _score_pool is alive (may be shut down by app lifespan tests)."""
        from concurrent.futures import ThreadPoolExecutor

        if memrl._score_pool._shutdown:
            memrl._score_pool = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="q-scorer-test"
            )

    def test_score_completed_task_no_scorer(self):
        """Test scoring is skipped when q_scorer is None."""
        mock_state = MagicMock()
        mock_state.q_scorer = None
        mock_state.q_scorer_enabled = True

        # Should not raise
        memrl.score_completed_task(mock_state, "task123")

        # No scoring should occur

    def test_score_completed_task_scorer_disabled(self):
        """Test scoring is skipped when scorer is disabled."""
        mock_state = MagicMock()
        mock_state.q_scorer = MagicMock()
        mock_state.q_scorer_enabled = False

        # Should not raise
        memrl.score_completed_task(mock_state, "task123")

        # Scorer should not be called
        mock_state.q_scorer._score_task.assert_not_called()

    def test_score_completed_task_with_valid_scorer(self):
        """Test scoring is submitted when scorer is available."""
        mock_state = MagicMock()
        mock_state.q_scorer = MagicMock()
        mock_state.q_scorer_enabled = True
        mock_state.progress_logger = MagicMock()

        # Call should submit to thread pool (we can't easily test the pool)
        # Just verify no exceptions
        memrl.score_completed_task(mock_state, "task123")

    def test_score_completed_task_with_mode(self):
        """Test scoring with execution mode context."""
        mock_state = MagicMock()
        mock_state.q_scorer = MagicMock()
        mock_state.q_scorer_enabled = True
        mock_state.progress_logger = MagicMock()

        # Should not raise
        memrl.score_completed_task_with_mode(mock_state, "task123", mode="react")


class TestExternalReward:
    """Test external reward storage."""

    def test_store_external_reward_feature_disabled(self):
        """Test external reward storage when feature is disabled."""
        mock_state = MagicMock()

        with patch("src.api.services.memrl.features") as mock_features:
            mock_feat = MagicMock()
            mock_feat.memrl = False
            mock_features.return_value = mock_feat

            result = memrl.store_external_reward(
                mock_state,
                "Test task",
                "action",
                0.8,
            )

        assert result is False

    def test_store_external_reward_no_scorer(self):
        """Test external reward storage when scorer is None."""
        mock_state = MagicMock()
        mock_state.q_scorer = None

        with patch("src.api.services.memrl.features") as mock_features:
            mock_feat = MagicMock()
            mock_feat.memrl = True
            mock_features.return_value = mock_feat

            result = memrl.store_external_reward(
                mock_state,
                "Test task",
                "action",
                0.8,
            )

        assert result is False

    def test_store_external_reward_success(self):
        """Test successful external reward storage."""
        mock_state = MagicMock()
        mock_state.q_scorer = MagicMock()
        mock_state.q_scorer.score_external_result.return_value = {
            "memories_created": 1,
            "memories_updated": 0,
        }
        mock_state.episodic_store = MagicMock()
        mock_state.progress_logger = MagicMock()

        with patch("src.api.services.memrl.features") as mock_features:
            mock_feat = MagicMock()
            mock_feat.memrl = True
            mock_features.return_value = mock_feat

            result = memrl.store_external_reward(
                mock_state,
                "Test task",
                "frontdoor:direct",
                0.8,
                context={"extra": "info"},
            )

        assert result is True
        mock_state.q_scorer.score_external_result.assert_called_once_with(
            task_description="Test task",
            action="frontdoor:direct",
            reward=0.8,
            context={"extra": "info"},
        )

    def test_store_external_reward_handles_exception(self):
        """Test external reward storage handles exceptions."""
        mock_state = MagicMock()
        mock_state.q_scorer = MagicMock()
        mock_state.q_scorer.score_external_result.side_effect = RuntimeError("DB error")
        mock_state.episodic_store = MagicMock()

        with patch("src.api.services.memrl.features") as mock_features:
            mock_feat = MagicMock()
            mock_feat.memrl = True
            mock_features.return_value = mock_feat

            result = memrl.store_external_reward(
                mock_state,
                "Test task",
                "action",
                0.5,
            )

        assert result is False


class TestBackgroundCleanup:
    """Test background cleanup task."""

    async def test_background_cleanup_respects_idle_state(self):
        """Test that cleanup only runs when idle."""
        mock_state = MagicMock()
        mock_state.active_requests = 1  # Not idle
        mock_state.q_scorer = MagicMock()
        mock_state.q_scorer_enabled = True

        # Run cleanup for a short time
        task = None
        try:
            import asyncio
            task = asyncio.create_task(memrl.background_cleanup(mock_state))
            await asyncio.sleep(0.1)
            task.cancel()
            await task
        except asyncio.CancelledError:
            pass

        # Should not have called scorer (not idle)
        mock_state.q_scorer.score_pending_tasks.assert_not_called()

    async def test_background_cleanup_handles_cancellation(self):
        """Test that cleanup handles cancellation gracefully."""
        mock_state = MagicMock()
        mock_state.active_requests = 0
        mock_state.q_scorer = MagicMock()
        mock_state.q_scorer_enabled = True

        import asyncio
        task = asyncio.create_task(memrl.background_cleanup(mock_state))

        # Cancel immediately
        await asyncio.sleep(0.01)
        task.cancel()

        # Should not raise
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

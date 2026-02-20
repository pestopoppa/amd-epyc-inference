"""Unit tests for MemRL service module.

Tests cover:
- Lazy import loading based on feature flags
- MemRL initialization
- Failure recovery
- Q-scoring with and without scorer
"""

from unittest.mock import patch
from dataclasses import dataclass
from typing import Any


from src.api.services import memrl
from src.api.state import AppState


# Stub classes for testing (minimal implementation)
@dataclass
class StubQScorer:
    """Stub Q-scorer for testing."""

    store: Any = None
    embedder: Any = None
    logger: Any = None
    reader: Any = None
    config: Any = None

    def _score_task(self, task_id: str, mode_context: str | None = None) -> None:
        """Stub scoring method."""
        pass

    def score_external_result(
        self,
        task_description: str,
        action: str,
        reward: float,
        context: dict,
        embedding: list[float] | None = None,
    ) -> dict:
        """Stub external result scoring."""
        return {"memories_created": 1, "memories_updated": 0}

    def score_pending_tasks(self) -> dict:
        """Stub pending task scoring."""
        return {"tasks_processed": 5, "skipped": False}


@dataclass
class StubEpisodicStore:
    """Stub episodic store for testing."""

    pass


@dataclass
class StubTaskEmbedder:
    """Stub task embedder for testing."""

    pass


@dataclass
class StubProgressReader:
    """Stub progress reader for testing."""

    pass


@dataclass
class StubProgressLogger:
    """Stub progress logger for testing."""

    def flush(self) -> None:
        """Stub flush method."""
        pass


@dataclass
class StubScoringConfig:
    """Stub scoring config for testing."""

    use_claude_judge: bool = False
    min_score_interval_seconds: int = 30
    batch_size: int = 10


@dataclass
class StubTwoPhaseRetriever:
    """Stub retriever for testing."""

    store: Any = None
    embedder: Any = None
    config: Any = None


@dataclass
class StubHybridRouter:
    """Stub hybrid router for testing."""

    retriever: Any = None
    rule_based_router: Any = None
    graph_router: Any = None


@dataclass
class StubRuleBasedRouter:
    """Stub rule-based router for testing."""

    routing_hints: dict = None


@dataclass
class StubRetrievalConfig:
    """Stub retrieval config for testing."""

    semantic_k: int = 20
    min_similarity: float = 0.3
    q_weight: float = 0.7
    confidence_threshold: float = 0.6


@dataclass
class StubRegistryLoader:
    """Stub registry loader for testing."""

    routing_hints: dict = None

    def __init__(self, validate_paths: bool = False):
        self.routing_hints = {}


class StubFeatures:
    """Stub features for testing."""

    def __init__(
        self,
        memrl: bool = False,
        tools: bool = False,
        scripts: bool = False,
        specialist_routing: bool = False,
        skillbank: bool = False,
        graph_router: bool = False,
    ):
        self.memrl = memrl
        self.tools = tools
        self.scripts = scripts
        self.specialist_routing = specialist_routing
        self.skillbank = skillbank
        self.graph_router = graph_router


class TestLoadOptionalImports:
    """Test optional import loading."""

    def test_load_optional_imports_with_memrl_enabled(self):
        """Test loading imports when memrl feature is enabled."""
        # Reset module globals
        memrl.ProgressLogger = None
        memrl.EpisodicStore = None
        memrl.TaskEmbedder = None

        with patch("src.api.services.memrl.features") as mock_features:
            mock_features.return_value = StubFeatures(memrl=True)

            # Mock the imports
            with patch.dict(
                "sys.modules",
                {
                    "orchestration.repl_memory.progress_logger": type(
                        "module",
                        (),
                        {
                            "ProgressLogger": StubProgressLogger,
                            "ProgressReader": StubProgressReader,
                        },
                    ),
                    "orchestration.repl_memory.episodic_store": type(
                        "module",
                        (),
                        {
                            "EpisodicStore": StubEpisodicStore,
                        },
                    ),
                    "orchestration.repl_memory.embedder": type(
                        "module",
                        (),
                        {
                            "TaskEmbedder": StubTaskEmbedder,
                        },
                    ),
                    "orchestration.repl_memory.q_scorer": type(
                        "module",
                        (),
                        {
                            "QScorer": StubQScorer,
                            "ScoringConfig": StubScoringConfig,
                        },
                    ),
                    "orchestration.repl_memory.retriever": type(
                        "module",
                        (),
                        {
                            "TwoPhaseRetriever": StubTwoPhaseRetriever,
                            "HybridRouter": StubHybridRouter,
                            "RuleBasedRouter": StubRuleBasedRouter,
                            "RetrievalConfig": StubRetrievalConfig,
                        },
                    ),
                },
            ):
                memrl.load_optional_imports()

        # Imports should be populated
        assert memrl.ProgressLogger is not None

    def test_load_optional_imports_with_memrl_disabled(self):
        """Test loading imports when memrl feature is disabled."""
        # Reset module globals
        memrl.ProgressLogger = None
        memrl.EpisodicStore = None

        with patch("src.api.services.memrl.features") as mock_features:
            mock_features.return_value = StubFeatures(memrl=False)

            with patch.dict(
                "sys.modules",
                {
                    "orchestration.repl_memory.progress_logger": type(
                        "module",
                        (),
                        {
                            "ProgressLogger": StubProgressLogger,
                            "ProgressReader": StubProgressReader,
                        },
                    ),
                },
            ):
                memrl.load_optional_imports()

        # Only ProgressLogger should be loaded
        assert memrl.ProgressLogger is not None

    def test_load_optional_imports_with_tools_enabled(self):
        """Test loading imports when tools feature is enabled."""
        memrl.ToolRegistry = None

        with patch("src.api.services.memrl.features") as mock_features:
            mock_features.return_value = StubFeatures(tools=True)

            with patch.dict(
                "sys.modules",
                {
                    "orchestration.repl_memory.progress_logger": type(
                        "module",
                        (),
                        {
                            "ProgressLogger": StubProgressLogger,
                            "ProgressReader": StubProgressReader,
                        },
                    ),
                    "src.tool_registry": type(
                        "module",
                        (),
                        {
                            "ToolRegistry": type("ToolRegistry", (), {}),
                        },
                    ),
                },
            ):
                memrl.load_optional_imports()

        # No exception should be raised

    def test_load_optional_imports_with_scripts_enabled(self):
        """Test loading imports when scripts feature is enabled."""
        memrl.ScriptRegistry = None

        with patch("src.api.services.memrl.features") as mock_features:
            mock_features.return_value = StubFeatures(scripts=True)

            with patch.dict(
                "sys.modules",
                {
                    "orchestration.repl_memory.progress_logger": type(
                        "module",
                        (),
                        {
                            "ProgressLogger": StubProgressLogger,
                            "ProgressReader": StubProgressReader,
                        },
                    ),
                    "src.script_registry": type(
                        "module",
                        (),
                        {
                            "ScriptRegistry": type("ScriptRegistry", (), {}),
                        },
                    ),
                },
            ):
                memrl.load_optional_imports()

        # No exception should be raised


class TestEnsureMemRLInitialized:
    """Test MemRL initialization."""

    def test_ensure_memrl_initialized_feature_disabled(self):
        """Test initialization returns False when feature is disabled."""
        state = AppState()
        state._memrl_initialized = False

        with patch("src.api.services.memrl.features") as mock_features:
            mock_features.return_value = StubFeatures(memrl=False)

            result = memrl.ensure_memrl_initialized(state)

        assert result is False
        # Should not attempt initialization
        assert state._memrl_initialized is False

    def test_ensure_memrl_initialized_already_initialized(self):
        """Test initialization is idempotent."""
        state = AppState()
        state._memrl_initialized = True
        state.q_scorer = StubQScorer()

        with patch("src.api.services.memrl.features") as mock_features:
            mock_features.return_value = StubFeatures(memrl=True)

            result = memrl.ensure_memrl_initialized(state)

        assert result is True

    def test_ensure_memrl_initialized_missing_imports(self):
        """Test initialization fails gracefully when imports missing."""
        state = AppState()
        state._memrl_initialized = False

        # Set imports to None to simulate missing
        original_episodic_store = memrl.EpisodicStore
        original_task_embedder = memrl.TaskEmbedder
        original_q_scorer = memrl.QScorer

        try:
            memrl.EpisodicStore = None
            memrl.TaskEmbedder = None
            memrl.QScorer = None

            with patch("src.api.services.memrl.features") as mock_features:
                mock_features.return_value = StubFeatures(memrl=True)

                result = memrl.ensure_memrl_initialized(state)

            assert result is False
            assert state._memrl_initialized is True  # Marked to prevent retries

        finally:
            # Restore
            memrl.EpisodicStore = original_episodic_store
            memrl.TaskEmbedder = original_task_embedder
            memrl.QScorer = original_q_scorer

    def test_ensure_memrl_initialized_success(self):
        """Test successful initialization."""
        state = AppState()
        state._memrl_initialized = False
        state.registry = StubRegistryLoader()
        state.progress_logger = StubProgressLogger()

        original_vals = (
            memrl.EpisodicStore,
            memrl.TaskEmbedder,
            memrl.QScorer,
            memrl.ScoringConfig,
            memrl.TwoPhaseRetriever,
            memrl.HybridRouter,
            memrl.RuleBasedRouter,
            memrl.ProgressReader,
            memrl.RetrievalConfig,
        )

        try:
            memrl.EpisodicStore = StubEpisodicStore
            memrl.TaskEmbedder = StubTaskEmbedder
            memrl.QScorer = StubQScorer
            memrl.ScoringConfig = StubScoringConfig
            memrl.TwoPhaseRetriever = StubTwoPhaseRetriever
            memrl.HybridRouter = StubHybridRouter
            memrl.RuleBasedRouter = StubRuleBasedRouter
            memrl.ProgressReader = StubProgressReader
            memrl.RetrievalConfig = StubRetrievalConfig

            with patch("src.api.services.memrl.features") as mock_features:
                mock_features.return_value = StubFeatures(memrl=True, specialist_routing=False)

                result = memrl.ensure_memrl_initialized(state)

            assert result is True
            assert state._memrl_initialized is True
            # Q-scorer should be created
            assert state.q_scorer is not None

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
                memrl.RetrievalConfig,
            ) = original_vals

    def test_ensure_memrl_initialized_handles_exception(self):
        """Test initialization handles exceptions gracefully."""
        state = AppState()
        state._memrl_initialized = False
        state.progress_logger = StubProgressLogger()

        # Create a class that raises on instantiation
        class FailingEpisodicStore:
            def __init__(self):
                raise RuntimeError("Init failed")

        original_episodic = memrl.EpisodicStore
        original_embedder = memrl.TaskEmbedder
        original_scorer = memrl.QScorer

        try:
            memrl.EpisodicStore = FailingEpisodicStore
            memrl.TaskEmbedder = StubTaskEmbedder
            memrl.QScorer = StubQScorer

            with patch("src.api.services.memrl.features") as mock_features:
                mock_features.return_value = StubFeatures(memrl=True)

                result = memrl.ensure_memrl_initialized(state)

            assert result is False
            # State should be marked initialized to prevent retry loops
            assert state._memrl_initialized is True
            # Components should be cleared
            assert state.q_scorer is None
            assert state.episodic_store is None

        finally:
            memrl.EpisodicStore = original_episodic
            memrl.TaskEmbedder = original_embedder
            memrl.QScorer = original_scorer


class TestScoreCompletedTask:
    """Test task scoring functions."""

    def setup_method(self):
        """Ensure score pool is available (lazily recreated after shutdown)."""
        # With the lazy-init pattern, shutdown_scoring() sets _score_pool=None
        # and _get_score_pool() recreates it on next use. No manual reset needed.

    def test_score_completed_task_no_scorer(self):
        """Test scoring is skipped when q_scorer is None."""
        state = AppState()
        state.q_scorer = None
        state.q_scorer_enabled = True

        # Should not raise
        memrl.score_completed_task(state, "task123")

    def test_score_completed_task_scorer_disabled(self):
        """Test scoring is skipped when scorer is disabled."""
        state = AppState()
        state.q_scorer = StubQScorer()
        state.q_scorer_enabled = False

        # Should not raise
        memrl.score_completed_task(state, "task123")

    def test_score_completed_task_with_valid_scorer(self):
        """Test scoring is submitted when scorer is available."""
        state = AppState()
        state.q_scorer = StubQScorer()
        state.q_scorer_enabled = True
        state.progress_logger = StubProgressLogger()

        # Call should submit to thread pool (we can't easily test the pool)
        # Just verify no exceptions
        memrl.score_completed_task(state, "task123")

    def test_score_completed_task_with_mode(self):
        """Test scoring with execution mode context."""
        state = AppState()
        state.q_scorer = StubQScorer()
        state.q_scorer_enabled = True
        state.progress_logger = StubProgressLogger()

        # Should not raise
        memrl.score_completed_task_with_mode(state, "task123", mode="react")


class TestExternalReward:
    """Test external reward storage."""

    def test_store_external_reward_feature_disabled(self):
        """Test external reward storage when feature is disabled."""
        state = AppState()

        with patch("src.api.services.memrl.features") as mock_features:
            mock_features.return_value = StubFeatures(memrl=False)

            result = memrl.store_external_reward(
                state,
                "Test task",
                "action",
                0.8,
            )

        assert result is False

    def test_store_external_reward_no_scorer(self):
        """Test external reward storage when scorer is None."""
        state = AppState()
        state.q_scorer = None

        with patch("src.api.services.memrl.features") as mock_features:
            mock_features.return_value = StubFeatures(memrl=True)

            result = memrl.store_external_reward(
                state,
                "Test task",
                "action",
                0.8,
            )

        assert result is False

    def test_store_external_reward_success(self):
        """Test successful external reward storage."""
        state = AppState()
        state.q_scorer = StubQScorer()
        state.episodic_store = StubEpisodicStore()
        state.progress_logger = StubProgressLogger()

        with patch("src.api.services.memrl.features") as mock_features:
            mock_features.return_value = StubFeatures(memrl=True)

            result = memrl.store_external_reward(
                state,
                "Test task",
                "frontdoor:direct",
                0.8,
                context={"extra": "info"},
            )

        assert result is True

    def test_store_external_reward_handles_exception(self):
        """Test external reward storage handles exceptions."""

        # Create a scorer that raises on score_external_result
        class FailingScorer:
            def score_external_result(self, **kwargs):
                raise RuntimeError("DB error")

        state = AppState()
        state.q_scorer = FailingScorer()
        state.episodic_store = StubEpisodicStore()

        with patch("src.api.services.memrl.features") as mock_features:
            mock_features.return_value = StubFeatures(memrl=True)

            result = memrl.store_external_reward(
                state,
                "Test task",
                "action",
                0.5,
            )

        assert result is False


class TestBackgroundCleanup:
    """Test background cleanup task."""

    async def test_background_cleanup_respects_idle_state(self):
        """Test that cleanup only runs when idle."""
        state = AppState()
        state.active_requests = 1  # Not idle
        state.q_scorer = StubQScorer()
        state.q_scorer_enabled = True

        # Run cleanup for a short time
        import asyncio

        task = asyncio.create_task(memrl.background_cleanup(state))
        await asyncio.sleep(0.1)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should not have called scorer (not idle)
        # No easy way to verify this without adding tracking to stub

    async def test_background_cleanup_handles_cancellation(self):
        """Test that cleanup handles cancellation gracefully."""
        state = AppState()
        state.active_requests = 0
        state.q_scorer = StubQScorer()
        state.q_scorer_enabled = True

        import asyncio

        task = asyncio.create_task(memrl.background_cleanup(state))

        # Cancel immediately
        await asyncio.sleep(0.01)
        task.cancel()

        # Should not raise
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

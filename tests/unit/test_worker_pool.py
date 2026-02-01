"""Unit tests for the heterogeneous worker pool.

Tests cover:
- Worker configuration and initialization
- Task type routing
- Round-robin load balancing
- HOT/WARM tier management
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.worker_pool import (
    TaskType,
    WorkerConfig,
    WorkerInstance,
    WorkerPoolConfig,
    WorkerPoolManager,
    WorkerTier,
    TASK_ROUTING,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def worker_config():
    """Create a basic worker configuration."""
    return WorkerConfig(
        name="test_worker",
        port=8082,
        model_path="/mnt/raid0/llm/models/test.gguf",
        tier=WorkerTier.HOT,
        threads=24,
        slots=2,
        task_types=["explore", "summarize"],
    )


@pytest.fixture
def pool_config():
    """Create a basic pool configuration."""
    return WorkerPoolConfig(
        enabled=True,
        prompt_lookup=True,
        workers={
            "explore": WorkerConfig(
                name="explore",
                port=8082,
                model_path="/mnt/raid0/llm/models/explore.gguf",
                tier=WorkerTier.HOT,
                threads=24,
                slots=2,
                task_types=["explore", "summarize"],
            ),
            "code": WorkerConfig(
                name="code",
                port=8092,
                model_path="/mnt/raid0/llm/models/code.gguf",
                tier=WorkerTier.HOT,
                threads=24,
                slots=2,
                task_types=["code_impl", "refactor"],
            ),
            "fast": WorkerConfig(
                name="fast",
                port=8102,
                model_path="/mnt/raid0/llm/models/fast.gguf",
                tier=WorkerTier.WARM,
                threads=16,
                slots=4,
                task_types=["boilerplate", "transform"],
            ),
        },
    )


@pytest.fixture
def pool_manager(pool_config):
    """Create a pool manager instance."""
    return WorkerPoolManager(config=pool_config)


# =============================================================================
# WorkerConfig Tests
# =============================================================================


class TestWorkerConfig:
    """Tests for WorkerConfig dataclass."""

    def test_worker_config_creation(self, worker_config):
        """Test basic WorkerConfig creation."""
        assert worker_config.name == "test_worker"
        assert worker_config.port == 8082
        assert worker_config.tier == WorkerTier.HOT
        assert worker_config.threads == 24
        assert worker_config.slots == 2
        assert "explore" in worker_config.task_types

    def test_worker_config_defaults(self):
        """Test WorkerConfig with minimal args."""
        config = WorkerConfig(
            name="minimal",
            port=9000,
            model_path="/path/to/model.gguf",
            tier=WorkerTier.WARM,
        )
        assert config.threads == 24  # default
        assert config.slots == 2  # default
        assert config.task_types == []  # default empty list


# =============================================================================
# WorkerInstance Tests
# =============================================================================


class TestWorkerInstance:
    """Tests for WorkerInstance dataclass."""

    def test_worker_instance_creation(self, worker_config):
        """Test WorkerInstance creation."""
        instance = WorkerInstance(config=worker_config)
        assert instance.config == worker_config
        assert instance.process is None
        assert instance.request_count == 0
        assert instance.error_count == 0

    def test_is_running_no_process(self, worker_config):
        """Test is_running when no process."""
        instance = WorkerInstance(config=worker_config)
        assert instance.is_running is False

    def test_is_running_with_process(self, worker_config):
        """Test is_running with mock process."""
        instance = WorkerInstance(config=worker_config)
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Still running
        instance.process = mock_process
        assert instance.is_running is True

    def test_is_running_process_exited(self, worker_config):
        """Test is_running when process exited."""
        instance = WorkerInstance(config=worker_config)
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Exited
        instance.process = mock_process
        assert instance.is_running is False

    def test_url_property(self, worker_config):
        """Test URL property."""
        instance = WorkerInstance(config=worker_config)
        assert instance.url == "http://localhost:8082"


# =============================================================================
# Task Routing Tests
# =============================================================================


class TestTaskRouting:
    """Tests for task type routing."""

    def test_explore_tasks_route_to_explore(self):
        """Test explore-related tasks route correctly."""
        assert TASK_ROUTING[TaskType.EXPLORE] == "explore"
        assert TASK_ROUTING[TaskType.SUMMARIZE] == "explore"
        assert TASK_ROUTING[TaskType.UNDERSTAND] == "explore"

    def test_code_tasks_route_to_code(self):
        """Test code-related tasks route correctly."""
        assert TASK_ROUTING[TaskType.CODE] == "code"
        assert TASK_ROUTING[TaskType.CODE_IMPL] == "code"
        assert TASK_ROUTING[TaskType.REFACTOR] == "code"
        assert TASK_ROUTING[TaskType.TEST_GEN] == "code"

    def test_fast_tasks_route_to_fast(self):
        """Test fast/boilerplate tasks route correctly."""
        assert TASK_ROUTING[TaskType.FAST] == "fast"
        assert TASK_ROUTING[TaskType.BOILERPLATE] == "fast"
        assert TASK_ROUTING[TaskType.TRANSFORM] == "fast"


# =============================================================================
# WorkerPoolManager Tests
# =============================================================================


class TestWorkerPoolManager:
    """Tests for WorkerPoolManager."""

    def test_pool_manager_creation(self, pool_manager, pool_config):
        """Test pool manager creation."""
        assert pool_manager.config == pool_config
        assert pool_manager._initialized is False

    @pytest.mark.asyncio
    async def test_initialize(self, pool_manager):
        """Test pool initialization."""
        await pool_manager.initialize()
        assert pool_manager._initialized is True
        assert len(pool_manager._workers) == 3
        assert len(pool_manager._hot_workers) == 2
        assert len(pool_manager._warm_workers) == 1
        assert pool_manager._http_session is not None

        # Cleanup
        await pool_manager.stop_all()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, pool_manager):
        """Test initialize is idempotent."""
        await pool_manager.initialize()
        session1 = pool_manager._http_session
        await pool_manager.initialize()
        session2 = pool_manager._http_session
        assert session1 is session2

        # Cleanup
        await pool_manager.stop_all()

    def test_get_worker_role_explore(self, pool_manager):
        """Test worker role mapping for explore tasks."""
        assert pool_manager._get_worker_role("explore") == "explore"
        assert pool_manager._get_worker_role("summarize") == "explore"
        assert pool_manager._get_worker_role("EXPLORE") == "explore"

    def test_get_worker_role_code(self, pool_manager):
        """Test worker role mapping for code tasks."""
        assert pool_manager._get_worker_role("code") == "code"
        assert pool_manager._get_worker_role("code_impl") == "code"
        assert pool_manager._get_worker_role("refactor") == "code"

    def test_get_worker_role_fast(self, pool_manager):
        """Test worker role mapping for fast tasks."""
        assert pool_manager._get_worker_role("fast") == "fast"
        assert pool_manager._get_worker_role("boilerplate") == "fast"
        assert pool_manager._get_worker_role("transform") == "fast"

    def test_get_worker_role_unknown(self, pool_manager):
        """Test worker role defaults to explore for unknown."""
        assert pool_manager._get_worker_role("unknown_task") == "explore"

    def test_round_robin(self, pool_manager):
        """Test round-robin worker selection."""
        # Create mock workers
        w1 = MagicMock()
        w1.config.name = "w1"
        w2 = MagicMock()
        w2.config.name = "w2"
        workers = [w1, w2]

        # Should cycle through workers
        result1 = pool_manager._get_round_robin(workers)
        result2 = pool_manager._get_round_robin(workers)
        result3 = pool_manager._get_round_robin(workers)

        assert result1.config.name == "w1"
        assert result2.config.name == "w2"
        assert result3.config.name == "w1"

    def test_build_launch_command_explore(self, pool_manager):
        """Test launch command for explore worker."""
        config = pool_manager.config.workers["explore"]
        cmd = pool_manager._build_launch_command(config)

        assert pool_manager.config.llama_server_path in cmd
        assert "-m" in cmd
        assert config.model_path in cmd
        assert "--port" in cmd
        assert "8082" in cmd
        assert "--flash-attn" in cmd
        assert "--lookup-ngram-min" in cmd  # prompt lookup enabled

    def test_build_launch_command_fast(self, pool_manager):
        """Test launch command for fast worker."""
        config = pool_manager.config.workers["fast"]
        cmd = pool_manager._build_launch_command(config)

        assert "--port" in cmd
        assert "8102" in cmd
        assert "-t" in cmd
        assert "16" in cmd  # threads for consolidated fast worker

    @pytest.mark.asyncio
    async def test_stop_all(self, pool_manager):
        """Test stopping all workers."""
        await pool_manager.initialize()
        await pool_manager.stop_all()

        assert pool_manager._http_session is None
        assert pool_manager._initialized is False

    def test_get_status_not_initialized(self, pool_manager):
        """Test status when not initialized."""
        status = pool_manager.get_status()
        assert status["initialized"] is False
        assert status["enabled"] is True
        assert status["hot_workers"] == {}
        assert status["warm_workers"] == {}

    @pytest.mark.asyncio
    async def test_get_status_initialized(self, pool_manager):
        """Test status when initialized."""
        await pool_manager.initialize()
        status = pool_manager.get_status()

        assert status["initialized"] is True
        assert "explore" in status["hot_workers"]
        assert "code" in status["hot_workers"]
        assert "fast" in status["warm_workers"]

        # Cleanup
        await pool_manager.stop_all()


# =============================================================================
# HTTP Call Tests (Mocked)
# =============================================================================


class TestWorkerPoolHTTP:
    """Tests for HTTP calls with mocked responses."""

    @pytest.mark.asyncio
    async def test_http_call_success(self, pool_manager):
        """Test successful HTTP call."""
        await pool_manager.initialize()

        # Create mock worker instance
        worker = pool_manager._workers["explore"]
        worker._healthy = True

        # Mock the HTTP response
        with patch.object(pool_manager._http_session, "post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"content": "test response"})
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await pool_manager._http_call(
                worker, "test prompt", temperature=0.2, max_tokens=100
            )

            assert result == "test response"

        await pool_manager.stop_all()

    @pytest.mark.asyncio
    async def test_http_call_error(self, pool_manager):
        """Test HTTP call with error response."""
        await pool_manager.initialize()

        worker = pool_manager._workers["explore"]
        worker._healthy = True

        with patch.object(pool_manager._http_session, "post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(RuntimeError, match="Worker returned 500"):
                await pool_manager._http_call(
                    worker, "test prompt", temperature=0.2, max_tokens=100
                )

        await pool_manager.stop_all()


# =============================================================================
# Integration Tests (No real servers)
# =============================================================================


class TestWorkerPoolIntegration:
    """Integration tests without actual server connections."""

    @pytest.mark.asyncio
    async def test_call_routes_correctly(self, pool_manager):
        """Test that call routes to correct worker type."""
        await pool_manager.initialize()

        # Mark workers as healthy
        for worker in pool_manager._workers.values():
            worker._healthy = True

        # Mock HTTP call
        with patch.object(pool_manager, "_http_call", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "response"

            await pool_manager.call("test prompt", task_type="explore")

            # Should have called with explore worker
            call_args = mock_call.call_args
            worker = call_args[0][0]
            assert "explore" in worker.config.name or "explore" in worker.config.task_types

        await pool_manager.stop_all()

    @pytest.mark.asyncio
    async def test_batch_parallel_execution(self, pool_manager):
        """Test that batch executes prompts in parallel."""
        await pool_manager.initialize()

        for worker in pool_manager._workers.values():
            worker._healthy = True

        with patch.object(pool_manager, "_http_call", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "response"

            prompts = ["prompt1", "prompt2", "prompt3"]
            results = await pool_manager.batch(prompts, task_type="explore")

            # Should have made 3 calls
            assert mock_call.call_count == 3
            assert len(results) == 3

        await pool_manager.stop_all()

    @pytest.mark.asyncio
    async def test_batch_maintains_order(self, pool_manager):
        """Test that batch results maintain prompt order."""
        await pool_manager.initialize()

        for worker in pool_manager._workers.values():
            worker._healthy = True

        # Return different responses based on prompt
        async def mock_response(worker, prompt, temp, max_tok):
            return f"response_for_{prompt}"

        with patch.object(pool_manager, "_http_call", side_effect=mock_response):
            prompts = ["a", "b", "c"]
            results = await pool_manager.batch(prompts, task_type="explore")

            assert results[0] == "response_for_a"
            assert results[1] == "response_for_b"
            assert results[2] == "response_for_c"

        await pool_manager.stop_all()

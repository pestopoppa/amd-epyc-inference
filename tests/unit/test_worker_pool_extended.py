"""Extended unit tests for worker pool to improve coverage.

Tests uncovered paths:
- Port checking logic
- Health check timeout scenarios
- Batch exception handling
- Round-robin distribution
- Warm shutdown cancellation
- HTTP error paths

Note: Requires aiohttp for WorkerPoolManager async HTTP operations.
"""

import asyncio
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip entire module if aiohttp is not available
aiohttp = pytest.importorskip("aiohttp", reason="aiohttp required for worker_pool tests")

from src.services.worker_pool import (
    WorkerConfig,
    WorkerInstance,
    WorkerPoolConfig,
    WorkerPoolManager,
    WorkerTier,
)


@pytest.fixture
def pool_config():
    """Create a basic pool configuration."""
    return WorkerPoolConfig(
        enabled=True,
        prompt_lookup=True,
        warm_timeout_seconds=10,  # Short timeout for tests
        workers={
            "explore": WorkerConfig(
                name="explore",
                port=8082,
                model_path="/mnt/raid0/llm/models/explore.gguf",
                tier=WorkerTier.HOT,
                threads=24,
                slots=2,
                task_types=["explore"],
            ),
            "fast": WorkerConfig(
                name="fast",
                port=8102,
                model_path="/mnt/raid0/llm/models/fast.gguf",
                tier=WorkerTier.WARM,
                threads=16,
                slots=4,
                task_types=["transform"],
            ),
        },
    )


class TestWorkerPoolInitialization:
    """Test worker pool initialization logic."""

    async def test_initialize_creates_http_session(self, pool_config):
        """Test that initialization creates HTTP session."""
        pool = WorkerPoolManager(config=pool_config)
        assert pool._http_session is None

        await pool.initialize()

        assert pool._http_session is not None
        assert isinstance(pool._http_session, aiohttp.ClientSession)
        assert pool._initialized is True

        # Cleanup
        await pool.stop_all()

    async def test_initialize_idempotent(self, pool_config):
        """Test that calling initialize multiple times is safe."""
        pool = WorkerPoolManager(config=pool_config)

        await pool.initialize()
        first_session = pool._http_session

        await pool.initialize()
        second_session = pool._http_session

        # Should be the same session
        assert first_session is second_session

        await pool.stop_all()

    async def test_initialize_creates_worker_instances(self, pool_config):
        """Test that initialization creates WorkerInstance objects."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        assert "explore" in pool._workers
        assert "fast" in pool._workers
        assert isinstance(pool._workers["explore"], WorkerInstance)
        assert pool._workers["explore"].config.name == "explore"

        await pool.stop_all()

    async def test_initialize_separates_hot_and_warm(self, pool_config):
        """Test that HOT and WARM workers are separated correctly."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        assert "explore" in pool._hot_workers
        assert "fast" in pool._warm_workers
        assert "fast" not in pool._hot_workers
        assert "explore" not in pool._warm_workers

        await pool.stop_all()


class TestPortChecking:
    """Test port availability checking."""

    async def test_check_port_in_use_when_free(self, pool_config):
        """Test port checking when port is free."""
        pool = WorkerPoolManager(config=pool_config)

        # Use an unlikely port
        is_used = await pool._check_port_in_use(59999)

        # Port should be free
        assert is_used is False

    async def test_check_port_in_use_when_occupied(self, pool_config):
        """Test port checking when port is occupied."""
        pool = WorkerPoolManager(config=pool_config)

        # Create a temporary socket to occupy a port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))  # Bind to any available port
        sock.listen(1)
        port = sock.getsockname()[1]

        try:
            is_used = await pool._check_port_in_use(port)
            assert is_used is True
        finally:
            sock.close()

    @patch("socket.socket")
    async def test_check_port_handles_connection_refused(self, mock_socket_class, pool_config):
        """Test port checking handles connection refused."""
        pool = WorkerPoolManager(config=pool_config)

        # Mock socket to simulate connection refused
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 111  # Connection refused
        mock_sock.__enter__.return_value = mock_sock
        mock_sock.__exit__.return_value = False
        mock_socket_class.return_value = mock_sock

        is_used = await pool._check_port_in_use(8082)

        # Connection refused means port is not in use
        assert is_used is False


class TestHealthCheck:
    """Test health check functionality."""

    async def test_wait_for_health_success(self, pool_config):
        """Test successful health check."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        # Mock successful health response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        with patch.object(pool._http_session, "get", return_value=mock_response):
            result = await pool._wait_for_health(8082, timeout=5)

        assert result is True

        await pool.stop_all()

    async def test_wait_for_health_timeout(self, pool_config):
        """Test health check timeout."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        # Mock failing health response
        async def mock_get_raises(*args, **kwargs):
            raise aiohttp.ClientError("Connection failed")

        with patch.object(pool._http_session, "get", side_effect=mock_get_raises):
            result = await pool._wait_for_health(8082, timeout=1)

        assert result is False

        await pool.stop_all()

    async def test_wait_for_health_eventual_success(self, pool_config):
        """Test health check succeeds after retries."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        # Mock responses that fail then succeed
        call_count = [0]  # Use list to avoid closure issues

        class MockResponse:
            def __init__(self, status):
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        class MockGet:
            def __call__(self, *args, **kwargs):
                call_count[0] += 1
                if call_count[0] < 3:
                    raise aiohttp.ClientError("Not ready yet")
                return MockResponse(200)

        with patch.object(pool._http_session, "get", new=MockGet()):
            result = await pool._wait_for_health(8082, timeout=10)

        assert result is True
        assert call_count[0] >= 3

        await pool.stop_all()


class TestBatchOperations:
    """Test batch operation handling."""

    async def test_batch_with_exceptions(self, pool_config):
        """Test batch handling when some tasks fail."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        # Create a mock worker
        mock_worker = WorkerInstance(config=pool_config.workers["explore"])
        mock_worker._healthy = True
        pool._workers["explore"] = mock_worker
        pool._hot_workers["explore"] = mock_worker

        # Mock _http_call to fail on second call
        call_count = 0

        async def mock_http_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated failure")
            return f"Response {call_count}"

        with patch.object(pool, "_http_call", side_effect=mock_http_call):
            results = await pool.batch(
                ["prompt1", "prompt2", "prompt3"],
                task_type="explore",
            )

        # Should have 3 results, with second one empty due to error
        assert len(results) == 3
        assert results[0] == "Response 1"
        assert results[1] == ""  # Failed task
        assert results[2] == "Response 3"

        await pool.stop_all()

    async def test_batch_round_robin_distribution(self, pool_config):
        """Test that batch distributes across multiple workers."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        # Create two healthy workers
        worker1 = WorkerInstance(config=pool_config.workers["explore"])
        worker1._healthy = True
        worker2_config = WorkerConfig(
            name="explore2",
            port=8083,
            model_path="/mnt/raid0/llm/models/explore.gguf",
            tier=WorkerTier.HOT,
            threads=24,
            slots=2,
            task_types=["explore"],
        )
        worker2 = WorkerInstance(config=worker2_config)
        worker2._healthy = True

        pool._workers["explore"] = worker1
        pool._workers["explore2"] = worker2
        pool._hot_workers["explore"] = worker1
        pool._hot_workers["explore2"] = worker2

        # Track which worker was called
        called_workers = []

        async def mock_http_call(worker, *args, **kwargs):
            called_workers.append(worker.config.name)
            return f"Response from {worker.config.name}"

        with patch.object(pool, "_http_call", side_effect=mock_http_call):
            with patch.object(pool, "_select_workers", return_value=[worker1, worker2]):
                await pool.batch(["p1", "p2", "p3", "p4"], task_type="explore")

        # Should alternate between workers (round-robin)
        assert "explore" in called_workers
        assert "explore2" in called_workers


class TestWarmShutdown:
    """Test WARM worker shutdown logic."""

    async def test_schedule_warm_shutdown(self, pool_config):
        """Test scheduling WARM worker shutdown."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        # Create a WARM worker instance
        worker = WorkerInstance(config=pool_config.workers["fast"])
        worker._healthy = True
        worker.process = MagicMock()
        worker.process.poll.return_value = None  # Running
        pool._workers["fast"] = worker
        pool._warm_workers["fast"] = worker

        # Schedule shutdown
        await pool._schedule_warm_shutdown(worker)

        # Task should be created
        assert "fast" in pool._warm_shutdown_tasks
        task = pool._warm_shutdown_tasks["fast"]
        assert isinstance(task, asyncio.Task)

        # Cancel to clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        await pool.stop_all()

    async def test_schedule_warm_shutdown_cancels_previous(self, pool_config):
        """Test that scheduling shutdown cancels previous task."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        worker = WorkerInstance(config=pool_config.workers["fast"])
        worker._healthy = True
        pool._workers["fast"] = worker
        pool._warm_workers["fast"] = worker

        # Schedule first shutdown
        await pool._schedule_warm_shutdown(worker)
        first_task = pool._warm_shutdown_tasks["fast"]

        # Give the first task a moment to start
        await asyncio.sleep(0.01)

        # Schedule second shutdown (should cancel first)
        await pool._schedule_warm_shutdown(worker)
        second_task = pool._warm_shutdown_tasks["fast"]

        # Wait for cancellation to propagate
        await asyncio.sleep(0.01)

        # First task should be cancelled or done
        assert first_task.cancelled() or first_task.done()
        assert not second_task.done()

        # Cleanup
        second_task.cancel()
        try:
            await second_task
        except asyncio.CancelledError:
            pass

        await pool.stop_all()

    async def test_warm_shutdown_task_handles_cancellation(self, pool_config):
        """Test that shutdown task gracefully handles cancellation."""
        pool = WorkerPoolManager(config=pool_config)
        pool.config.warm_timeout_seconds = 1  # Short timeout
        await pool.initialize()

        worker = WorkerInstance(config=pool_config.workers["fast"])
        worker._healthy = True
        pool._workers["fast"] = worker
        pool._warm_workers["fast"] = worker

        await pool._schedule_warm_shutdown(worker)
        task = pool._warm_shutdown_tasks["fast"]

        # Cancel immediately
        task.cancel()

        # Should not raise
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

        await pool.stop_all()


class TestHTTPCallErrors:
    """Test HTTP call error handling."""

    async def test_http_call_non_200_status(self, pool_config):
        """Test HTTP call with non-200 status."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        worker = WorkerInstance(config=pool_config.workers["explore"])

        # Mock 500 error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        with patch.object(pool._http_session, "post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Worker returned 500"):
                await pool._http_call(worker, "test prompt", 0.2, 1024)

        await pool.stop_all()

    async def test_http_call_client_error(self, pool_config):
        """Test HTTP call with ClientError."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        worker = WorkerInstance(config=pool_config.workers["explore"])

        # Mock ClientError - need to make it raise when used as async context manager
        class MockPost:
            async def __aenter__(self):
                raise aiohttp.ClientError("Connection refused")

            async def __aexit__(self, *args):
                return None

        def mock_post_factory(*args, **kwargs):
            return MockPost()

        with patch.object(pool._http_session, "post", side_effect=mock_post_factory):
            with pytest.raises(RuntimeError, match="HTTP call to explore failed"):
                await pool._http_call(worker, "test prompt", 0.2, 1024)

        # Error count should be incremented
        assert worker.error_count == 1

        await pool.stop_all()

    async def test_http_call_json_parse_error(self, pool_config):
        """Test HTTP call when JSON parsing fails."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        worker = WorkerInstance(config=pool_config.workers["explore"])

        # Mock response with invalid JSON
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        with patch.object(pool._http_session, "post", return_value=mock_response):
            with pytest.raises(ValueError, match="Invalid JSON"):
                await pool._http_call(worker, "test prompt", 0.2, 1024)

        await pool.stop_all()

    async def test_http_call_success_with_content(self, pool_config):
        """Test successful HTTP call returns content."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        worker = WorkerInstance(config=pool_config.workers["explore"])

        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"content": "Generated text"})
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        with patch.object(pool._http_session, "post", return_value=mock_response):
            result = await pool._http_call(worker, "test prompt", 0.2, 1024)

        assert result == "Generated text"

        await pool.stop_all()

    async def test_http_call_missing_content_field(self, pool_config):
        """Test HTTP call when response lacks content field."""
        pool = WorkerPoolManager(config=pool_config)
        await pool.initialize()

        worker = WorkerInstance(config=pool_config.workers["explore"])

        # Mock response without content field
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "something else"})
        mock_response.__aenter__.return_value = mock_response
        mock_response.__aexit__.return_value = None

        with patch.object(pool._http_session, "post", return_value=mock_response):
            result = await pool._http_call(worker, "test prompt", 0.2, 1024)

        # Should return empty string when content is missing
        assert result == ""

        await pool.stop_all()

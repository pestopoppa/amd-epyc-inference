#!/usr/bin/env python3
"""Heterogeneous worker pool for parallel task execution.

Manages multiple llama-server instances with different models optimized
for different task types:
- explore (7B): Directory crawling, file summaries, codebase understanding
- code (7B): Code implementation, following architect instructions
- fast (1.5B): Simple transformations, boilerplate, high-volume parallel

Architecture:
- HOT workers: Always resident, immediate availability
- WARM workers: Loaded on demand for burst capacity

Usage:
    pool = WorkerPoolManager()
    await pool.start_hot_workers()

    # Single call
    result = await pool.call("Summarize this file", task_type="explore")

    # Batch call (auto-routes and parallelizes)
    results = await pool.batch(prompts, task_type="explore")
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class WorkerTier(Enum):
    """Worker availability tier."""
    HOT = "hot"    # Always resident
    WARM = "warm"  # Load on demand


class TaskType(Enum):
    """Task types for routing."""
    EXPLORE = "explore"       # Understanding, summarization
    CODE = "code"             # Code implementation
    FAST = "fast"             # Simple transformations
    SUMMARIZE = "summarize"   # Alias for explore
    UNDERSTAND = "understand" # Alias for explore
    CODE_IMPL = "code_impl"   # Alias for code
    REFACTOR = "refactor"     # Alias for code
    TEST_GEN = "test_gen"     # Alias for code
    BOILERPLATE = "boilerplate"  # Alias for fast
    TRANSFORM = "transform"   # Alias for fast


# Task type to worker role mapping
TASK_ROUTING = {
    TaskType.EXPLORE: "explore",
    TaskType.SUMMARIZE: "explore",
    TaskType.UNDERSTAND: "explore",
    TaskType.CODE: "code",
    TaskType.CODE_IMPL: "code",
    TaskType.REFACTOR: "code",
    TaskType.TEST_GEN: "code",
    TaskType.FAST: "fast",
    TaskType.BOILERPLATE: "fast",
    TaskType.TRANSFORM: "fast",
}


@dataclass
class WorkerConfig:
    """Configuration for a single worker instance."""
    name: str
    port: int
    model_path: str
    tier: WorkerTier
    threads: int = 24
    slots: int = 2
    task_types: list[str] = field(default_factory=list)
    launch_flags: list[str] = field(default_factory=list)


@dataclass
class WorkerInstance:
    """Running worker instance state."""
    config: WorkerConfig
    process: Optional[subprocess.Popen] = None
    started_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    request_count: int = 0
    error_count: int = 0
    _healthy: bool = False

    @property
    def is_running(self) -> bool:
        """Check if worker process is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    @property
    def url(self) -> str:
        """Get worker HTTP URL."""
        return f"http://localhost:{self.config.port}"


@dataclass
class WorkerPoolConfig:
    """Configuration for the entire worker pool."""
    enabled: bool = True
    prompt_lookup: bool = True
    warm_timeout_seconds: int = 300  # 5 minutes
    expansion_threshold: int = 4  # Concurrent tasks to trigger WARM expansion
    health_check_interval: int = 30
    llama_server_path: str = "/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
    log_dir: str = "/mnt/raid0/llm/claude/logs"

    workers: dict[str, WorkerConfig] = field(default_factory=dict)


class WorkerPoolManager:
    """Manages a heterogeneous pool of llama-server worker instances.

    Provides intelligent routing based on task type and load balancing
    across multiple workers.
    """

    def __init__(self, config: Optional[WorkerPoolConfig] = None):
        """Initialize the worker pool manager.

        Args:
            config: Pool configuration. If None, uses defaults from registry.
        """
        self.config = config or self._load_default_config()
        self._workers: dict[str, WorkerInstance] = {}
        self._hot_workers: dict[str, WorkerInstance] = {}
        self._warm_workers: dict[str, WorkerInstance] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._round_robin_idx: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._warm_shutdown_tasks: dict[str, asyncio.Task] = {}
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._initialized = False

    def _load_default_config(self) -> WorkerPoolConfig:
        """Load default configuration (can be overridden by registry)."""
        model_base = "/mnt/raid0/llm/lmstudio/models"
        return WorkerPoolConfig(
            workers={
                "explore": WorkerConfig(
                    name="explore",
                    port=8082,
                    model_path=f"{model_base}/Qwen/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                    tier=WorkerTier.HOT,
                    threads=24,
                    slots=2,
                    task_types=["explore", "summarize", "understand"],
                ),
                "code": WorkerConfig(
                    name="code",
                    port=8092,
                    model_path=f"{model_base}/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
                    tier=WorkerTier.HOT,
                    threads=24,
                    slots=2,
                    task_types=["code_impl", "refactor", "test_gen"],
                ),
                "fast_1": WorkerConfig(
                    name="fast_1",
                    port=8102,
                    model_path=f"{model_base}/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf",
                    tier=WorkerTier.WARM,
                    threads=12,
                    slots=2,
                    task_types=["boilerplate", "transform", "parallel_burst"],
                ),
                "fast_2": WorkerConfig(
                    name="fast_2",
                    port=8112,
                    model_path=f"{model_base}/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/Qwen2.5-Coder-1.5B-Instruct-Q8_0.gguf",
                    tier=WorkerTier.WARM,
                    threads=12,
                    slots=2,
                    task_types=["boilerplate", "transform", "parallel_burst"],
                ),
            }
        )

    async def initialize(self) -> None:
        """Initialize the pool (call once before use)."""
        if self._initialized:
            return

        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=300, connect=10)
        self._http_session = aiohttp.ClientSession(timeout=timeout)

        # Create worker instances (not started yet)
        for name, worker_config in self.config.workers.items():
            instance = WorkerInstance(config=worker_config)
            self._workers[name] = instance

            if worker_config.tier == WorkerTier.HOT:
                self._hot_workers[name] = instance
            else:
                self._warm_workers[name] = instance

            self._round_robin_idx[name] = 0

        # Calculate total concurrent capacity
        total_slots = sum(w.config.slots for w in self._hot_workers.values())
        self._semaphore = asyncio.Semaphore(total_slots * 2)  # 2x for buffering

        self._initialized = True
        logger.info(f"WorkerPool initialized: {len(self._hot_workers)} HOT, "
                    f"{len(self._warm_workers)} WARM workers")

    async def start_hot_workers(self) -> dict[str, bool]:
        """Start all HOT tier workers.

        Returns:
            Dict mapping worker name to success status.
        """
        await self.initialize()

        results = {}
        for name, instance in self._hot_workers.items():
            success = await self._start_worker(instance)
            results[name] = success

        return results

    async def _start_worker(self, instance: WorkerInstance) -> bool:
        """Start a single worker instance.

        Args:
            instance: Worker instance to start.

        Returns:
            True if started successfully.
        """
        config = instance.config

        if instance.is_running:
            logger.info(f"Worker {config.name} already running on port {config.port}")
            return True

        # Check if port is already in use
        if await self._check_port_in_use(config.port):
            logger.warning(f"Port {config.port} already in use, attempting cleanup")
            await self._kill_port(config.port)
            await asyncio.sleep(1)

        # Build command
        cmd = self._build_launch_command(config)

        # Create log file
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"worker-{config.name}-{config.port}.log"

        logger.info(f"Starting worker {config.name} on port {config.port}")
        logger.debug(f"Command: {' '.join(cmd[:8])}...")

        try:
            with open(log_file, "w") as log:
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = "1"

                instance.process = subprocess.Popen(
                    ["numactl", "--interleave=all"] + cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

            instance.started_at = datetime.now()

            # Wait for health
            if await self._wait_for_health(config.port, timeout=120):
                instance._healthy = True
                logger.info(f"Worker {config.name} ready on port {config.port}")
                return True
            else:
                logger.error(f"Worker {config.name} failed health check")
                await self._stop_worker(instance)
                return False

        except Exception as e:
            logger.error(f"Failed to start worker {config.name}: {e}")
            return False

    def _build_launch_command(self, config: WorkerConfig) -> list[str]:
        """Build llama-server launch command."""
        cmd = [
            self.config.llama_server_path,
            "-m", config.model_path,
            "--host", "0.0.0.0",
            "--port", str(config.port),
            "-np", str(config.slots),
            "-c", "8192",  # 4K per slot with np=2
            "-t", str(config.threads),
            "--flash-attn", "on",
        ]

        # Add prompt lookup for all workers
        if self.config.prompt_lookup:
            cmd.extend(["--lookup-ngram-min", "3"])

        # Add any extra flags
        cmd.extend(config.launch_flags)

        return cmd

    async def _wait_for_health(self, port: int, timeout: int = 120) -> bool:
        """Wait for worker health endpoint.

        Args:
            port: Worker port.
            timeout: Timeout in seconds.

        Returns:
            True if healthy within timeout.
        """
        url = f"http://localhost:{port}/health"
        start = time.time()

        while time.time() - start < timeout:
            try:
                async with self._http_session.get(url) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)

        return False

    async def _check_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    async def _kill_port(self, port: int) -> None:
        """Kill any process using a port."""
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                for pid_str in result.stdout.strip().split("\n"):
                    try:
                        os.kill(int(pid_str), signal.SIGKILL)
                    except Exception:
                        pass
        except Exception:
            pass

    async def _stop_worker(self, instance: WorkerInstance) -> None:
        """Stop a worker instance."""
        if instance.process is None:
            return

        try:
            instance.process.terminate()
            try:
                instance.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                instance.process.kill()
                instance.process.wait(timeout=2)
        except Exception as e:
            logger.warning(f"Error stopping worker {instance.config.name}: {e}")
        finally:
            instance.process = None
            instance._healthy = False

    async def stop_all(self) -> None:
        """Stop all workers and cleanup."""
        # Cancel warm shutdown tasks
        for task in self._warm_shutdown_tasks.values():
            task.cancel()

        # Stop all workers
        for instance in self._workers.values():
            await self._stop_worker(instance)

        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        self._initialized = False
        logger.info("WorkerPool stopped")

    # =========================================================================
    # Request Routing
    # =========================================================================

    def _get_worker_role(self, task_type: str) -> str:
        """Map task type to worker role.

        Args:
            task_type: Task type string or TaskType enum.

        Returns:
            Worker role name (explore, code, fast).
        """
        # Handle string task types
        try:
            tt = TaskType(task_type.lower())
        except ValueError:
            # Unknown task type, default to explore
            return "explore"

        return TASK_ROUTING.get(tt, "explore")

    async def _select_workers(
        self,
        task_type: str,
        parallelism: int = 1,
    ) -> list[WorkerInstance]:
        """Select workers for a task.

        Args:
            task_type: Type of task.
            parallelism: Number of parallel tasks.

        Returns:
            List of worker instances to use.
        """
        role = self._get_worker_role(task_type)

        # Find matching workers
        candidates = []
        for instance in self._workers.values():
            if role in instance.config.name or role in instance.config.task_types:
                if instance._healthy:
                    candidates.append(instance)

        if not candidates:
            # Fallback to any healthy HOT worker
            candidates = [w for w in self._hot_workers.values() if w._healthy]

        if not candidates:
            raise RuntimeError("No healthy workers available")

        # For high parallelism, consider spinning up WARM workers
        if parallelism > self.config.expansion_threshold:
            await self._ensure_warm_workers_running(role)
            # Re-scan for fast workers
            for instance in self._warm_workers.values():
                if instance._healthy and instance not in candidates:
                    candidates.append(instance)

        return candidates

    async def _ensure_warm_workers_running(self, role: str = "fast") -> None:
        """Ensure WARM workers are running for high parallelism.

        Args:
            role: Preferred worker role (used for logging).
        """
        async with self._lock:
            for name, instance in self._warm_workers.items():
                if not instance._healthy:
                    logger.info(f"Expanding pool: starting WARM worker {name}")
                    await self._start_worker(instance)

                # Cancel any pending shutdown
                if name in self._warm_shutdown_tasks:
                    self._warm_shutdown_tasks[name].cancel()
                    del self._warm_shutdown_tasks[name]

    async def _schedule_warm_shutdown(self, instance: WorkerInstance) -> None:
        """Schedule a WARM worker for shutdown after idle timeout."""
        name = instance.config.name

        async def _shutdown_after_timeout():
            await asyncio.sleep(self.config.warm_timeout_seconds)
            async with self._lock:
                # Check if still idle
                if instance.last_used:
                    idle_time = (datetime.now() - instance.last_used).total_seconds()
                    if idle_time >= self.config.warm_timeout_seconds:
                        logger.info(f"Shutting down idle WARM worker {name}")
                        await self._stop_worker(instance)

        # Cancel existing shutdown task if any
        if name in self._warm_shutdown_tasks:
            self._warm_shutdown_tasks[name].cancel()

        # Schedule new shutdown
        task = asyncio.create_task(_shutdown_after_timeout())
        self._warm_shutdown_tasks[name] = task

    def _get_round_robin(self, workers: list[WorkerInstance]) -> WorkerInstance:
        """Get next worker using round-robin.

        Args:
            workers: List of candidate workers.

        Returns:
            Selected worker instance.
        """
        # Use first worker's name as the key for round-robin state
        key = workers[0].config.name if workers else "default"

        idx = self._round_robin_idx.get(key, 0)
        worker = workers[idx % len(workers)]
        self._round_robin_idx[key] = (idx + 1) % len(workers)

        return worker

    # =========================================================================
    # Public API
    # =========================================================================

    async def call(
        self,
        prompt: str,
        task_type: str = "explore",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Make a single call to a worker.

        Args:
            prompt: The prompt to send.
            task_type: Task type for routing.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Model response text.
        """
        await self.initialize()

        workers = await self._select_workers(task_type, parallelism=1)
        worker = self._get_round_robin(workers)

        async with self._semaphore:
            result = await self._http_call(worker, prompt, temperature, max_tokens)
            worker.last_used = datetime.now()
            worker.request_count += 1

            # Schedule WARM shutdown if applicable
            if worker.config.tier == WorkerTier.WARM:
                await self._schedule_warm_shutdown(worker)

            return result

    async def batch(
        self,
        prompts: list[str],
        task_type: str = "explore",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> list[str]:
        """Execute batch in parallel across pool.

        Args:
            prompts: List of prompts.
            task_type: Task type for routing.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            List of responses in order.
        """
        await self.initialize()

        workers = await self._select_workers(task_type, parallelism=len(prompts))

        async def _call_with_worker(prompt: str, idx: int) -> tuple[int, str]:
            worker = workers[idx % len(workers)]
            async with self._semaphore:
                result = await self._http_call(worker, prompt, temperature, max_tokens)
                worker.last_used = datetime.now()
                worker.request_count += 1
                return idx, result

        # Execute all calls in parallel
        tasks = [
            asyncio.create_task(_call_with_worker(prompt, i))
            for i, prompt in enumerate(prompts)
        ]

        results_unordered = await asyncio.gather(*tasks, return_exceptions=True)

        # Reorder results
        results = [""] * len(prompts)
        for item in results_unordered:
            if isinstance(item, Exception):
                logger.error(f"Batch call failed: {item}")
                continue
            idx, text = item
            results[idx] = text

        # Schedule WARM shutdown for any warm workers used
        for worker in workers:
            if worker.config.tier == WorkerTier.WARM:
                await self._schedule_warm_shutdown(worker)

        return results

    async def _http_call(
        self,
        worker: WorkerInstance,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make HTTP call to worker.

        Args:
            worker: Worker instance.
            prompt: Prompt text.
            temperature: Sampling temperature.
            max_tokens: Max tokens.

        Returns:
            Response text.
        """
        url = f"{worker.url}/completion"

        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            "stream": False,
        }

        try:
            async with self._http_session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Worker returned {resp.status}: {error_text}")

                data = await resp.json()
                return data.get("content", "")

        except aiohttp.ClientError as e:
            worker.error_count += 1
            raise RuntimeError(f"HTTP call to {worker.config.name} failed: {e}")

    # =========================================================================
    # Status & Monitoring
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get pool status summary.

        Returns:
            Dict with pool status information.
        """
        return {
            "initialized": self._initialized,
            "enabled": self.config.enabled,
            "hot_workers": {
                name: {
                    "port": w.config.port,
                    "healthy": w._healthy,
                    "running": w.is_running,
                    "requests": w.request_count,
                    "errors": w.error_count,
                    "model": Path(w.config.model_path).stem,
                }
                for name, w in self._hot_workers.items()
            },
            "warm_workers": {
                name: {
                    "port": w.config.port,
                    "healthy": w._healthy,
                    "running": w.is_running,
                    "requests": w.request_count,
                    "errors": w.error_count,
                    "model": Path(w.config.model_path).stem,
                }
                for name, w in self._warm_workers.items()
            },
        }

    async def health_check(self) -> dict[str, bool]:
        """Check health of all workers.

        Returns:
            Dict mapping worker name to health status.
        """
        results = {}
        for name, instance in self._workers.items():
            if instance.is_running:
                instance._healthy = await self._wait_for_health(
                    instance.config.port, timeout=5
                )
            else:
                instance._healthy = False
            results[name] = instance._healthy

        return results


# Singleton instance for module-level access
_pool_instance: Optional[WorkerPoolManager] = None


def get_worker_pool() -> WorkerPoolManager:
    """Get or create the global worker pool instance."""
    global _pool_instance
    if _pool_instance is None:
        _pool_instance = WorkerPoolManager()
    return _pool_instance


async def shutdown_worker_pool() -> None:
    """Shutdown the global worker pool."""
    global _pool_instance
    if _pool_instance is not None:
        await _pool_instance.stop_all()
        _pool_instance = None

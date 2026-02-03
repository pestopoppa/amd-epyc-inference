"""Batch processing for vision pipeline."""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterator

from src.vision.config import (
    DEFAULT_BATCH_SIZE,
    MAX_CONCURRENT_WORKERS,
    ensure_directories,
)
from src.vision.models import (
    AnalyzerType,
    JobStatus,
    BatchStatusResponse,
)
from src.vision.pipeline import VisionPipeline, get_pipeline

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Tracks state of a batch processing job."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    errors: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    _lock: Lock = field(default_factory=Lock)

    def increment_processed(self) -> None:
        with self._lock:
            self.processed_items += 1

    def increment_failed(self, error: str | None = None) -> None:
        with self._lock:
            self.failed_items += 1
            if error:
                self.errors.append(error)

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    @property
    def estimated_remaining_seconds(self) -> float | None:
        if self.processed_items == 0 or self.elapsed_seconds == 0:
            return None
        rate = self.processed_items / self.elapsed_seconds
        remaining = self.total_items - self.processed_items
        return remaining / rate if rate > 0 else None


class BatchProcessor:
    """Manages batch processing jobs.

    Provides job queue, progress tracking, and parallel execution.
    """

    def __init__(self, max_workers: int = MAX_CONCURRENT_WORKERS):
        """Initialize batch processor.

        Args:
            max_workers: Maximum concurrent workers.
        """
        self.max_workers = max_workers
        self._jobs: dict[str, BatchJob] = {}

    def create_job(
        self,
        input_directory: Path | str | None = None,
        input_paths: list[Path | str] | None = None,
        recursive: bool = True,
        extensions: list[str] | None = None,
        analyzers: list[AnalyzerType] | None = None,
        vl_prompt: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> BatchJob:
        """Create a new batch processing job.

        Args:
            input_directory: Directory to process.
            input_paths: List of specific paths to process.
            recursive: Whether to process subdirectories.
            extensions: File extensions to include.
            analyzers: Analyzers to run.
            vl_prompt: Custom VL prompt.
            batch_size: Number of items per batch.

        Returns:
            BatchJob with job_id.
        """
        ensure_directories()

        job_id = str(uuid.uuid4())
        job = BatchJob(job_id=job_id)

        # Collect input files
        files = list(
            self._collect_files(
                input_directory=Path(input_directory) if input_directory else None,
                input_paths=[Path(p) for p in input_paths] if input_paths else None,
                recursive=recursive,
                extensions=extensions or ["jpg", "jpeg", "png", "heic", "webp"],
            )
        )

        job.total_items = len(files)

        if job.total_items == 0:
            job.status = JobStatus.COMPLETED
            logger.warning("No files found for batch job")
            self._jobs[job_id] = job
            return job

        self._jobs[job_id] = job

        # Start processing in background
        self._start_job(
            job=job,
            files=files,
            analyzers=analyzers,
            vl_prompt=vl_prompt,
            batch_size=batch_size,
        )

        return job

    def get_job(self, job_id: str) -> BatchJob | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def get_job_status(self, job_id: str) -> BatchStatusResponse | None:
        """Get job status response."""
        job = self._jobs.get(job_id)
        if not job:
            return None

        return BatchStatusResponse(
            job_id=job.job_id,
            status=job.status,
            total_items=job.total_items,
            processed_items=job.processed_items,
            failed_items=job.failed_items,
            elapsed_seconds=job.elapsed_seconds,
            estimated_remaining_seconds=job.estimated_remaining_seconds,
            errors=job.errors[:10],  # Limit errors returned
        )

    def shutdown(self) -> None:
        """Shutdown the batch processor, cancelling all running jobs."""
        for job in self._jobs.values():
            if job.status == JobStatus.RUNNING:
                job.status = JobStatus.CANCELLED
        self._jobs.clear()

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.RUNNING:
            return False

        job.status = JobStatus.CANCELLED
        return True

    def _collect_files(
        self,
        input_directory: Path | None,
        input_paths: list[Path] | None,
        recursive: bool,
        extensions: list[str],
    ) -> Iterator[Path]:
        """Collect files to process."""
        ext_set = {f".{ext.lower().lstrip('.')}" for ext in extensions}

        if input_paths:
            for p in input_paths:
                if p.exists() and p.suffix.lower() in ext_set:
                    yield p

        if input_directory and input_directory.exists():
            pattern = "**/*" if recursive else "*"
            for p in input_directory.glob(pattern):
                if p.is_file() and p.suffix.lower() in ext_set:
                    yield p

    def _start_job(
        self,
        job: BatchJob,
        files: list[Path],
        analyzers: list[AnalyzerType] | None,
        vl_prompt: str | None,
        batch_size: int,
    ) -> None:
        """Start job processing in background."""
        import threading

        def run():
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()

            pipeline = get_pipeline()
            if not pipeline._initialized:
                pipeline.initialize(analyzers)

            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}

                    for file_path in files:
                        if job.status == JobStatus.CANCELLED:
                            break

                        future = executor.submit(
                            self._process_file,
                            pipeline=pipeline,
                            file_path=file_path,
                            analyzers=analyzers,
                            vl_prompt=vl_prompt,
                        )
                        futures[future] = file_path

                    for future in as_completed(futures):
                        if job.status == JobStatus.CANCELLED:
                            break

                        file_path = futures[future]
                        try:
                            result = future.result()
                            if result.errors:
                                job.increment_failed(f"{file_path}: {result.errors[0]}")
                            else:
                                job.increment_processed()
                        except Exception as e:
                            job.increment_failed(f"{file_path}: {e}")

                job.completed_at = datetime.utcnow()
                if job.status != JobStatus.CANCELLED:
                    job.status = JobStatus.COMPLETED if job.failed_items == 0 else JobStatus.FAILED

            except Exception as e:
                logger.error(f"Batch job {job.job_id} failed: {e}")
                job.status = JobStatus.FAILED
                job.errors.append(str(e))
                job.completed_at = datetime.utcnow()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _process_file(
        self,
        pipeline: VisionPipeline,
        file_path: Path,
        analyzers: list[AnalyzerType] | None,
        vl_prompt: str | None,
    ):
        """Process a single file."""
        return pipeline.analyze(
            image=file_path,
            analyzers=analyzers,
            vl_prompt=vl_prompt,
            store_results=True,
        )


# Global batch processor
_processor: BatchProcessor | None = None


def get_batch_processor() -> BatchProcessor:
    """Get or create global batch processor."""
    global _processor
    if _processor is None:
        _processor = BatchProcessor()
    return _processor

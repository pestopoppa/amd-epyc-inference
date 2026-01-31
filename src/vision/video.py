"""Video processing for vision pipeline."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from src.vision.config import (
    DEFAULT_VIDEO_FPS,
    VISION_CACHE_DIR,
    VISION_THUMBS_DIR,
    THUMB_SIZE,
    THUMB_QUALITY,
    FFMPEG_VERSION_TIMEOUT,
    FFMPEG_PROBE_TIMEOUT,
    FFMPEG_EXTRACT_TIMEOUT,
    ensure_directories,
)
from src.vision.models import (
    AnalyzerType,
    VideoAnalyzeResponse,
    VideoFrameResult,
    FaceResult,
    BoundingBox,
)
from src.vision.pipeline import VisionPipeline, get_pipeline
from src.db.models.vision import Video, VideoFrame, get_session

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process videos by extracting and analyzing frames.

    Uses ffmpeg for frame extraction and the vision pipeline for analysis.
    """

    def __init__(self):
        """Initialize video processor."""
        self._ffmpeg_available = False
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> None:
        """Check if ffmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=FFMPEG_VERSION_TIMEOUT,
            )
            self._ffmpeg_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._ffmpeg_available = False
            logger.warning("ffmpeg not available. Video processing disabled.")

    def get_video_info(self, video_path: Path) -> dict[str, Any] | None:
        """Get video metadata using ffprobe.

        Args:
            video_path: Path to video file.

        Returns:
            Dict with duration, width, height, fps.
        """
        if not self._ffmpeg_available:
            return None

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=FFMPEG_PROBE_TIMEOUT,
            )

            if result.returncode != 0:
                return None

            import json
            data = json.loads(result.stdout)

            # Find video stream
            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if not video_stream:
                return None

            # Parse fps
            fps_str = video_stream.get("r_frame_rate", "0/1")
            if "/" in fps_str:
                num, den = map(float, fps_str.split("/"))
                fps = num / den if den != 0 else 0
            else:
                fps = float(fps_str)

            return {
                "duration": float(data.get("format", {}).get("duration", 0)),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": fps,
            }

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None

    def extract_frames(
        self,
        video_path: Path,
        fps: float = DEFAULT_VIDEO_FPS,
        output_dir: Path | None = None,
    ) -> list[tuple[int, Path]]:
        """Extract frames from video at specified fps.

        Args:
            video_path: Path to video file.
            fps: Frames per second to extract.
            output_dir: Directory for frame images.

        Returns:
            List of (timestamp_ms, frame_path) tuples.
        """
        if not self._ffmpeg_available:
            raise RuntimeError("ffmpeg not available")

        ensure_directories()

        if output_dir is None:
            output_dir = VISION_CACHE_DIR / f"frames_{uuid.uuid4().hex[:8]}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames
        frame_pattern = str(output_dir / "frame_%06d.jpg")
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",  # High quality JPEG
            frame_pattern,
            "-y",  # Overwrite
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=FFMPEG_EXTRACT_TIMEOUT,
        )

        if result.returncode != 0:
            error = result.stderr.decode() if result.stderr else "Unknown error"
            raise RuntimeError(f"Frame extraction failed: {error}")

        # Collect extracted frames
        frames = []
        frame_interval_ms = int(1000 / fps)

        for i, frame_path in enumerate(sorted(output_dir.glob("frame_*.jpg"))):
            timestamp_ms = i * frame_interval_ms
            frames.append((timestamp_ms, frame_path))

        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    def analyze(
        self,
        video_path: Path | str,
        fps: float = DEFAULT_VIDEO_FPS,
        analyzers: list[AnalyzerType] | None = None,
        vl_prompt: str | None = None,
        store_thumbnails: bool = True,
        store_results: bool = True,
    ) -> VideoAnalyzeResponse:
        """Analyze a video by extracting and processing frames.

        Args:
            video_path: Path to video file.
            fps: Frames per second to extract (default: 1).
            analyzers: Analyzers to run on each frame.
            vl_prompt: Custom VL prompt.
            store_thumbnails: Whether to save frame thumbnails.
            store_results: Whether to store in database.

        Returns:
            VideoAnalyzeResponse with frame-by-frame results.
        """
        video_path = Path(video_path)
        start_time = time.perf_counter()

        if not self._ffmpeg_available:
            return VideoAnalyzeResponse(
                video_id=str(uuid.uuid4()),
                path=str(video_path),
                duration_seconds=0,
                frames_analyzed=0,
                frames=[],
                processing_time_seconds=0,
            )

        # Get video info
        info = self.get_video_info(video_path)
        duration = info.get("duration", 0) if info else 0

        # Extract frames
        try:
            frames = self.extract_frames(video_path, fps)
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return VideoAnalyzeResponse(
                video_id=str(uuid.uuid4()),
                path=str(video_path),
                duration_seconds=duration,
                frames_analyzed=0,
                frames=[],
                processing_time_seconds=time.perf_counter() - start_time,
            )

        # Initialize pipeline
        pipeline = get_pipeline()
        if not pipeline.is_initialized:
            pipeline.initialize(analyzers)

        # Process each frame
        frame_results: list[VideoFrameResult] = []
        video_id = str(uuid.uuid4())

        for timestamp_ms, frame_path in frames:
            frame_id = str(uuid.uuid4())

            try:
                image = Image.open(frame_path)

                # Analyze frame
                result = pipeline.analyze(
                    image=image,
                    analyzers=analyzers,
                    vl_prompt=vl_prompt,
                    store_results=False,  # We handle video storage separately
                )

                frame_result = VideoFrameResult(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    description=result.description,
                    faces=[],
                )

                # Copy face results
                for face in result.faces:
                    frame_result.faces.append(FaceResult(
                        face_id=face.face_id,
                        bbox=face.bbox,
                        confidence=face.confidence,
                        person_id=face.person_id,
                    ))

                # Save thumbnail
                if store_thumbnails:
                    thumb_path = self._save_thumbnail(
                        image,
                        video_id,
                        timestamp_ms,
                    )
                    frame_result.thumbnail_path = str(thumb_path) if thumb_path else None

                frame_results.append(frame_result)

            except Exception as e:
                logger.error(f"Failed to process frame at {timestamp_ms}ms: {e}")
                frame_results.append(VideoFrameResult(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                ))

            finally:
                # Clean up extracted frame
                frame_path.unlink(missing_ok=True)

        # Store in database
        if store_results:
            self._store_results(video_id, video_path, duration, info, frame_results)

        elapsed = time.perf_counter() - start_time

        return VideoAnalyzeResponse(
            video_id=video_id,
            path=str(video_path),
            duration_seconds=duration,
            frames_analyzed=len(frame_results),
            frames=frame_results,
            processing_time_seconds=elapsed,
        )

    def _save_thumbnail(
        self,
        image: Image.Image,
        video_id: str,
        timestamp_ms: int,
    ) -> Path | None:
        """Save frame thumbnail."""
        try:
            thumb_dir = VISION_THUMBS_DIR / "videos" / video_id
            thumb_dir.mkdir(parents=True, exist_ok=True)

            thumb = image.copy()
            thumb.thumbnail(THUMB_SIZE, Image.Resampling.LANCZOS)

            thumb_path = thumb_dir / f"{timestamp_ms}.jpg"
            thumb.save(thumb_path, "JPEG", quality=THUMB_QUALITY)

            return thumb_path

        except Exception as e:
            logger.error(f"Failed to save thumbnail: {e}")
            return None

    def _store_results(
        self,
        video_id: str,
        video_path: Path,
        duration: float,
        info: dict[str, Any] | None,
        frames: list[VideoFrameResult],
    ) -> None:
        """Store video analysis results in SQLite."""
        session = get_session()
        try:
            # Create video record
            video = Video(
                id=video_id,
                path=str(video_path),
                duration_secs=duration,
                width=info.get("width") if info else None,
                height=info.get("height") if info else None,
                fps=info.get("fps") if info else None,
            )
            session.add(video)

            # Create frame records
            for frame in frames:
                frame_record = VideoFrame(
                    id=frame.frame_id,
                    video_id=video_id,
                    timestamp_ms=frame.timestamp_ms,
                    thumbnail_path=frame.thumbnail_path,
                    description=frame.description,
                )
                session.add(frame_record)

            session.commit()
            logger.info(f"Stored video {video_id} with {len(frames)} frames")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store video results: {e}")
            raise
        finally:
            session.close()


# Global video processor
_processor: VideoProcessor | None = None


def get_video_processor() -> VideoProcessor:
    """Get or create global video processor."""
    global _processor
    if _processor is None:
        _processor = VideoProcessor()
    return _processor

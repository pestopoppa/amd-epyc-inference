"""Pydantic models for vision API requests and responses."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AnalyzerType(str, Enum):
    """Available analyzer types."""
    FACE_DETECT = "face_detect"
    FACE_EMBED = "face_embed"
    FACE_ATTRIBUTES = "face_attributes"
    VL_DESCRIBE = "vl_describe"
    VL_OCR = "vl_ocr"
    VL_STRUCTURED = "vl_structured"
    EXIF_EXTRACT = "exif_extract"
    CLIP_EMBED = "clip_embed"
    OBJECT_DETECT = "object_detect"


class JobStatus(str, Enum):
    """Batch job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Request Models

class AnalyzeRequest(BaseModel):
    """Single image analysis request."""
    image_path: str | None = Field(default=None, description="Path to image file")
    image_base64: str | None = Field(default=None, description="Base64-encoded image")
    image_url: str | None = Field(default=None, description="URL to fetch image from")
    analyzers: list[AnalyzerType] = Field(
        default=[AnalyzerType.VL_DESCRIBE],
        description="Analyzers to run"
    )
    vl_prompt: str | None = Field(
        default="Describe this image briefly.",
        description="Prompt for VL description"
    )
    return_crops: bool = Field(default=False, description="Return face crops")
    store_results: bool = Field(default=True, description="Store in database")


class BatchRequest(BaseModel):
    """Batch processing request."""
    input_directory: str | None = Field(default=None, description="Directory to process")
    input_paths: list[str] | None = Field(default=None, description="List of paths")
    recursive: bool = Field(default=True, description="Process subdirectories")
    extensions: list[str] = Field(
        default=["jpg", "jpeg", "png", "heic", "webp"],
        description="File extensions to process"
    )
    analyzers: list[AnalyzerType] = Field(
        default=[AnalyzerType.FACE_DETECT, AnalyzerType.VL_DESCRIBE, AnalyzerType.EXIF_EXTRACT],
        description="Analyzers to run"
    )
    vl_prompt: str | None = Field(default=None, description="Prompt for VL description")
    batch_size: int = Field(default=100, ge=1, le=1000)
    max_workers: int = Field(default=4, ge=1, le=16)


class SearchRequest(BaseModel):
    """Search request for indexed content."""
    query: str = Field(..., description="Text query")
    search_type: str = Field(
        default="description",
        description="Search type: description, face, visual"
    )
    filters: dict[str, Any] | None = Field(default=None, description="Metadata filters")
    limit: int = Field(default=50, ge=1, le=500)


class FaceIdentifyRequest(BaseModel):
    """Face identification request."""
    image_path: str | None = None
    image_base64: str | None = None
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    return_all: bool = Field(default=False, description="Return all faces, not just identified")


class FaceClusterRequest(BaseModel):
    """Request to cluster unlabeled faces."""
    min_cluster_size: int = Field(default=3, ge=2)
    min_samples: int = Field(default=2, ge=1)


class PersonUpdateRequest(BaseModel):
    """Update person information."""
    name: str | None = None
    merge_with: str | None = Field(default=None, description="Person ID to merge into")


class VideoAnalyzeRequest(BaseModel):
    """Video analysis request."""
    video_path: str = Field(..., description="Path to video file")
    fps: float = Field(default=1.0, ge=0.1, le=30.0, description="Frames per second to extract")
    analyzers: list[AnalyzerType] = Field(
        default=[AnalyzerType.VL_DESCRIBE],
        description="Analyzers for each frame"
    )
    vl_prompt: str | None = None
    store_thumbnails: bool = Field(default=True)


# Response Models

class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x: int
    y: int
    width: int
    height: int


class FaceResult(BaseModel):
    """Face detection/identification result."""
    face_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    bbox: BoundingBox
    confidence: float
    person_id: str | None = None
    person_name: str | None = None
    embedding_stored: bool = False
    crop_base64: str | None = None


class ExifData(BaseModel):
    """EXIF metadata extraction result."""
    taken_at: datetime | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    focal_length: str | None = None
    aperture: str | None = None
    iso: int | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    orientation: int | None = None
    width: int | None = None
    height: int | None = None


class AnalyzeResult(BaseModel):
    """Single image analysis result."""
    image_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    path: str | None = None
    faces: list[FaceResult] = Field(default_factory=list)
    description: str | None = None
    ocr_text: str | None = None
    structured_data: dict[str, Any] | None = None
    exif: ExifData | None = None
    clip_embedding_stored: bool = False
    processing_time_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)


class BatchJobResponse(BaseModel):
    """Batch job creation response."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.PENDING
    total_items: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BatchStatusResponse(BaseModel):
    """Batch job status response."""
    job_id: str
    status: JobStatus
    total_items: int
    processed_items: int
    failed_items: int
    elapsed_seconds: float
    estimated_remaining_seconds: float | None = None
    errors: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Single search result."""
    image_id: str
    path: str
    score: float
    description: str | None = None
    thumbnail_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    results: list[SearchResult]
    total_found: int
    search_time_ms: float


class PersonInfo(BaseModel):
    """Person information."""
    person_id: str
    name: str | None = None
    photo_count: int = 0
    created_at: datetime | None = None
    representative_face_id: str | None = None


class PersonListResponse(BaseModel):
    """List of persons."""
    persons: list[PersonInfo]
    total: int


class ClusterResult(BaseModel):
    """Face clustering result."""
    clusters_created: int
    faces_clustered: int
    noise_faces: int
    new_person_ids: list[str]


class VideoFrameResult(BaseModel):
    """Video frame analysis result."""
    frame_id: str
    timestamp_ms: int
    thumbnail_path: str | None = None
    description: str | None = None
    faces: list[FaceResult] = Field(default_factory=list)


class VideoAnalyzeResponse(BaseModel):
    """Video analysis response."""
    video_id: str
    path: str
    duration_seconds: float
    frames_analyzed: int
    frames: list[VideoFrameResult]
    processing_time_seconds: float

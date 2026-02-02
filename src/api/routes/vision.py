"""Vision processing API endpoints."""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends

from src.vision.config import VISION_CACHE_DIR, ensure_directories
from src.vision.models import (
    AnalyzeRequest,
    AnalyzeResult,
    BatchRequest,
    BatchJobResponse,
    BatchStatusResponse,
    SearchRequest,
    SearchResponse,
    FaceIdentifyRequest,
    FaceClusterRequest,
    ClusterResult,
    PersonUpdateRequest,
    PersonListResponse,
    PersonInfo,
    VideoAnalyzeRequest,
    VideoAnalyzeResponse,
    JobStatus,
)
from src.vision.pipeline import VisionPipeline
from src.vision.batch import BatchProcessor
from src.vision.search import VisionSearch
from src.vision.video import VideoProcessor
from src.vision.clustering import cluster_unlabeled_faces
from src.db.chroma_client import get_collection_stats
from src.api.dependencies import (
    dep_vision_pipeline,
    dep_vision_batch_processor,
    dep_vision_search,
    dep_vision_video_processor,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post("/analyze", response_model=AnalyzeResult)
async def analyze_image(
    request: AnalyzeRequest,
    pipeline: VisionPipeline = Depends(dep_vision_pipeline),
) -> AnalyzeResult:
    """Analyze a single image.

    Provide ONE of:
    - image_path: Path to local file
    - image_base64: Base64-encoded image data
    - image_url: URL to fetch (not yet implemented)
    """
    ensure_directories()

    # Get image path
    if request.image_path:
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {image_path}")
        temp_file = None

    elif request.image_base64:
        # Decode base64 to temp file
        VISION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".jpg",
            dir=VISION_CACHE_DIR,
            delete=False,
        )
        try:
            image_data = base64.b64decode(request.image_base64)
            temp_file.write(image_data)
            temp_file.close()
            image_path = Path(temp_file.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    elif request.image_url:
        raise HTTPException(status_code=501, detail="URL fetch not yet implemented")

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide image_path, image_base64, or image_url"
        )

    try:
        if not pipeline.is_initialized:
            pipeline.initialize(request.analyzers)

        result = pipeline.analyze(
            image=image_path,
            analyzers=request.analyzers,
            vl_prompt=request.vl_prompt,
            store_results=request.store_results,
            return_crops=request.return_crops,
        )

        return result

    finally:
        # Clean up temp file
        if temp_file:
            Path(temp_file.name).unlink(missing_ok=True)


@router.post("/batch", response_model=BatchJobResponse)
async def start_batch_job(
    request: BatchRequest,
    processor: BatchProcessor = Depends(dep_vision_batch_processor),
) -> BatchJobResponse:
    """Start a batch processing job.

    Provide either:
    - input_directory: Directory to process
    - input_paths: List of specific file paths
    """
    if not request.input_directory and not request.input_paths:
        raise HTTPException(
            status_code=400,
            detail="Provide input_directory or input_paths"
        )

    job = processor.create_job(
        input_directory=request.input_directory,
        input_paths=request.input_paths,
        recursive=request.recursive,
        extensions=request.extensions,
        analyzers=request.analyzers,
        vl_prompt=request.vl_prompt,
        batch_size=request.batch_size,
    )

    return BatchJobResponse(
        job_id=job.job_id,
        status=job.status,
        total_items=job.total_items,
        created_at=job.created_at,
    )


@router.get("/batch/{job_id}", response_model=BatchStatusResponse)
async def get_batch_status(
    job_id: str,
    processor: BatchProcessor = Depends(dep_vision_batch_processor),
) -> BatchStatusResponse:
    """Get batch job status."""
    status = processor.get_job_status(job_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return status


@router.delete("/batch/{job_id}")
async def cancel_batch_job(
    job_id: str,
    processor: BatchProcessor = Depends(dep_vision_batch_processor),
) -> dict:
    """Cancel a running batch job."""
    if processor.cancel_job(job_id):
        return {"status": "cancelled", "job_id": job_id}
    else:
        raise HTTPException(status_code=400, detail="Job not running or not found")


@router.post("/search", response_model=SearchResponse)
async def search_images(
    request: SearchRequest,
    search: VisionSearch = Depends(dep_vision_search),
) -> SearchResponse:
    """Search indexed images.

    Search types:
    - description: Semantic search on descriptions
    - face: Find images with similar faces (query = face_id)
    - visual: Visual similarity via CLIP
    """
    result = search.search(
        query=request.query,
        search_type=request.search_type,
        filters=request.filters,
        limit=request.limit,
    )

    return result


@router.get("/faces", response_model=PersonListResponse)
async def list_persons(
    search: VisionSearch = Depends(dep_vision_search),
) -> PersonListResponse:
    """List all known persons."""
    persons = search.list_persons()

    return PersonListResponse(
        persons=[PersonInfo(**p) for p in persons],
        total=len(persons),
    )


@router.post("/faces/identify")
async def identify_faces(request: FaceIdentifyRequest) -> AnalyzeResult:
    """Identify faces in an image against known persons."""
    if not request.image_path and not request.image_base64:
        raise HTTPException(
            status_code=400,
            detail="Provide image_path or image_base64"
        )

    # Use analyze endpoint with face analyzers
    from src.vision.models import AnalyzerType

    analyze_request = AnalyzeRequest(
        image_path=request.image_path,
        image_base64=request.image_base64,
        analyzers=[AnalyzerType.FACE_DETECT, AnalyzerType.FACE_EMBED],
        store_results=False,
    )

    return await analyze_image(analyze_request)


@router.post("/faces/cluster", response_model=ClusterResult)
async def cluster_faces(request: FaceClusterRequest) -> ClusterResult:
    """Cluster unlabeled faces into persons.

    Uses HDBSCAN clustering on face embeddings.
    """
    try:
        result = cluster_unlabeled_faces(
            min_cluster_size=request.min_cluster_size,
            min_samples=request.min_samples,
        )

        return ClusterResult(
            clusters_created=result.clusters_created,
            faces_clustered=result.faces_clustered,
            noise_faces=result.noise_faces,
            new_person_ids=result.new_person_ids,
        )

    except ImportError:
        raise HTTPException(status_code=501, detail="hdbscan not installed")


@router.put("/faces/{person_id}")
async def update_person(
    person_id: str,
    request: PersonUpdateRequest,
    search: VisionSearch = Depends(dep_vision_search),
) -> dict:
    """Update person name or merge with another person."""
    success = search.update_person(
        person_id=person_id,
        name=request.name,
        merge_with=request.merge_with,
    )

    if not success:
        raise HTTPException(status_code=404, detail=f"Person not found: {person_id}")

    return {"status": "updated", "person_id": person_id}


@router.post("/video/analyze", response_model=VideoAnalyzeResponse)
async def analyze_video(
    request: VideoAnalyzeRequest,
    processor: VideoProcessor = Depends(dep_vision_video_processor),
) -> VideoAnalyzeResponse:
    """Analyze a video by extracting and processing frames."""
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    result = processor.analyze(
        video_path=video_path,
        fps=request.fps,
        analyzers=request.analyzers,
        vl_prompt=request.vl_prompt,
        store_thumbnails=request.store_thumbnails,
    )

    return result


@router.get("/stats")
async def get_stats() -> dict:
    """Get vision pipeline statistics."""
    stats = get_collection_stats()

    # Add database counts
    from src.db.models.vision import Photo, Face, Person, Video, managed_session

    with managed_session() as session:
        stats["photos"] = session.query(Photo).count()
        stats["faces_sqlite"] = session.query(Face).count()
        stats["persons"] = session.query(Person).count()
        stats["videos"] = session.query(Video).count()

    return stats

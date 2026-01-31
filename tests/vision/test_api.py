"""Tests for vision API endpoints."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from src.vision.models import (
    AnalyzerType,
    JobStatus,
    AnalyzeResult,
    BatchJobResponse,
    SearchResponse,
)


@pytest.fixture
def test_client():
    """Create a test client for the API."""
    from src.api import create_app

    app = create_app()
    return TestClient(app)


class TestVisionAnalyzeEndpoint:
    """Tests for /v1/vision/analyze endpoint."""

    @patch("src.api.routes.vision.get_pipeline")
    def test_analyze_missing_input(self, mock_pipeline, test_client):
        """Test error when no image input provided."""
        response = test_client.post(
            "/v1/vision/analyze",
            json={},
        )

        assert response.status_code == 400
        assert "Provide image_path" in response.json()["detail"]

    @patch("src.api.routes.vision.get_pipeline")
    def test_analyze_file_not_found(self, mock_pipeline, test_client):
        """Test error when file doesn't exist."""
        response = test_client.post(
            "/v1/vision/analyze",
            json={"image_path": "/nonexistent/file.jpg"},
        )

        assert response.status_code == 404

    @patch("src.api.routes.vision.get_pipeline")
    def test_analyze_success(self, mock_pipeline, test_client, tmp_path):
        """Test successful analysis."""
        # Create test file
        from PIL import Image

        test_image = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100)).save(test_image)

        # Mock pipeline
        mock_instance = MagicMock()
        mock_instance._initialized = True
        mock_instance.analyze.return_value = AnalyzeResult(
            image_id="test123",
            path=str(test_image),
            description="A test image",
        )
        mock_pipeline.return_value = mock_instance

        response = test_client.post(
            "/v1/vision/analyze",
            json={"image_path": str(test_image)},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["image_id"] == "test123"
        assert data["description"] == "A test image"

    @patch("src.api.routes.vision.get_pipeline")
    def test_analyze_with_base64(self, mock_pipeline, test_client):
        """Test analysis with base64 input."""
        import base64
        from io import BytesIO
        from PIL import Image

        # Create base64 image
        img = Image.new("RGB", (50, 50), color="red")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Mock pipeline
        mock_instance = MagicMock()
        mock_instance._initialized = True
        mock_instance.analyze.return_value = AnalyzeResult(
            image_id="b64test",
            description="Base64 image",
        )
        mock_pipeline.return_value = mock_instance

        response = test_client.post(
            "/v1/vision/analyze",
            json={"image_base64": img_b64},
        )

        assert response.status_code == 200


class TestBatchEndpoint:
    """Tests for /v1/vision/batch endpoint."""

    @patch("src.api.routes.vision.get_batch_processor")
    def test_batch_missing_input(self, mock_processor, test_client):
        """Test error when no input provided."""
        response = test_client.post(
            "/v1/vision/batch",
            json={},
        )

        assert response.status_code == 400
        assert "Provide input_directory" in response.json()["detail"]

    @patch("src.api.routes.vision.get_batch_processor")
    def test_batch_create_job(self, mock_processor, test_client, tmp_path):
        """Test creating a batch job."""
        from src.vision.batch import BatchJob

        mock_job = BatchJob(
            job_id="job123",
            status=JobStatus.PENDING,
            total_items=10,
        )

        mock_instance = MagicMock()
        mock_instance.create_job.return_value = mock_job
        mock_processor.return_value = mock_instance

        response = test_client.post(
            "/v1/vision/batch",
            json={"input_directory": str(tmp_path)},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job123"
        assert data["status"] == "pending"

    @patch("src.api.routes.vision.get_batch_processor")
    def test_batch_get_status(self, mock_processor, test_client):
        """Test getting batch job status."""
        from src.vision.models import BatchStatusResponse

        mock_status = BatchStatusResponse(
            job_id="job123",
            status=JobStatus.RUNNING,
            total_items=100,
            processed_items=50,
            failed_items=2,
            elapsed_seconds=30.5,
        )

        mock_instance = MagicMock()
        mock_instance.get_job_status.return_value = mock_status
        mock_processor.return_value = mock_instance

        response = test_client.get("/v1/vision/batch/job123")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job123"
        assert data["processed_items"] == 50

    @patch("src.api.routes.vision.get_batch_processor")
    def test_batch_not_found(self, mock_processor, test_client):
        """Test 404 for unknown job."""
        mock_instance = MagicMock()
        mock_instance.get_job_status.return_value = None
        mock_processor.return_value = mock_instance

        response = test_client.get("/v1/vision/batch/unknown_job")

        assert response.status_code == 404


class TestSearchEndpoint:
    """Tests for /v1/vision/search endpoint."""

    @patch("src.api.routes.vision.get_search")
    def test_search_description(self, mock_search, test_client):
        """Test description search."""
        from src.vision.models import SearchResult

        mock_response = SearchResponse(
            query="beach photos",
            results=[
                SearchResult(
                    image_id="img1",
                    path="/photos/beach.jpg",
                    score=0.85,
                    description="Beach at sunset",
                ),
            ],
            total_found=1,
            search_time_ms=50.5,
        )

        mock_instance = MagicMock()
        mock_instance.search.return_value = mock_response
        mock_search.return_value = mock_instance

        response = test_client.post(
            "/v1/vision/search",
            json={"query": "beach photos"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "beach photos"
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.85


class TestFacesEndpoint:
    """Tests for /v1/vision/faces endpoints."""

    @patch("src.api.routes.vision.get_search")
    def test_list_persons(self, mock_search, test_client):
        """Test listing known persons."""
        mock_instance = MagicMock()
        mock_instance.list_persons.return_value = [
            {
                "person_id": "p1",
                "name": "John Doe",
                "photo_count": 25,
            },
        ]
        mock_search.return_value = mock_instance

        response = test_client.get("/v1/vision/faces")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["persons"][0]["name"] == "John Doe"

    @patch("src.api.routes.vision.get_search")
    def test_update_person(self, mock_search, test_client):
        """Test updating person name."""
        mock_instance = MagicMock()
        mock_instance.update_person.return_value = True
        mock_search.return_value = mock_instance

        response = test_client.put(
            "/v1/vision/faces/p1",
            json={"name": "Jane Doe"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "updated"

    @patch("src.api.routes.vision.get_search")
    def test_update_person_not_found(self, mock_search, test_client):
        """Test updating nonexistent person."""
        mock_instance = MagicMock()
        mock_instance.update_person.return_value = False
        mock_search.return_value = mock_instance

        response = test_client.put(
            "/v1/vision/faces/unknown",
            json={"name": "Test"},
        )

        assert response.status_code == 404


class TestVideoEndpoint:
    """Tests for /v1/vision/video/analyze endpoint."""

    @patch("src.api.routes.vision.get_video_processor")
    def test_video_not_found(self, mock_processor, test_client):
        """Test error when video doesn't exist."""
        response = test_client.post(
            "/v1/vision/video/analyze",
            json={"video_path": "/nonexistent/video.mp4"},
        )

        assert response.status_code == 404

    @patch("src.api.routes.vision.get_video_processor")
    def test_video_analyze(self, mock_processor, test_client, tmp_path):
        """Test video analysis."""
        from src.vision.models import VideoAnalyzeResponse, VideoFrameResult

        # Create dummy video file
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        mock_response = VideoAnalyzeResponse(
            video_id="vid123",
            path=str(video_path),
            duration_seconds=60.0,
            frames_analyzed=10,
            frames=[
                VideoFrameResult(
                    frame_id="f1",
                    timestamp_ms=0,
                    description="Opening scene",
                ),
            ],
            processing_time_seconds=5.5,
        )

        mock_instance = MagicMock()
        mock_instance.analyze.return_value = mock_response
        mock_processor.return_value = mock_instance

        response = test_client.post(
            "/v1/vision/video/analyze",
            json={"video_path": str(video_path)},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["video_id"] == "vid123"
        assert data["frames_analyzed"] == 10


class TestStatsEndpoint:
    """Tests for /v1/vision/stats endpoint."""

    @patch("src.api.routes.vision.get_collection_stats")
    @patch("src.db.models.vision.managed_session")
    def test_get_stats(self, mock_managed_session, mock_chroma_stats, test_client):
        """Test getting pipeline statistics."""
        mock_chroma_stats.return_value = {
            "faces": 100,
            "descriptions": 500,
            "images": 500,
        }

        # Mock managed_session context manager returning a session with query()
        mock_query = MagicMock()
        mock_query.count.return_value = 500
        mock_session_instance = MagicMock()
        mock_session_instance.query.return_value = mock_query
        mock_managed_session.return_value.__enter__ = MagicMock(return_value=mock_session_instance)
        mock_managed_session.return_value.__exit__ = MagicMock(return_value=False)

        response = test_client.get("/v1/vision/stats")

        assert response.status_code == 200
        data = response.json()
        assert "faces" in data
        assert "descriptions" in data

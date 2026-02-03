"""Tests for vision API endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.vision.models import (
    JobStatus,
    AnalyzeResult,
    SearchResponse,
)


@pytest.fixture
def test_client():
    """Create a test client for the API."""
    from src.api import create_app

    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_vision_pipeline():
    """Create a mock vision pipeline."""
    mock = MagicMock()
    mock.is_initialized = True
    return mock


@pytest.fixture
def mock_batch_processor():
    """Create a mock batch processor."""
    return MagicMock()


@pytest.fixture
def mock_vision_search():
    """Create a mock vision search."""
    return MagicMock()


@pytest.fixture
def mock_video_processor():
    """Create a mock video processor."""
    return MagicMock()


class TestVisionAnalyzeEndpoint:
    """Tests for /v1/vision/analyze endpoint."""

    def test_analyze_missing_input(self, test_client, mock_vision_pipeline):
        """Test error when no image input provided."""
        from src.api.dependencies import dep_vision_pipeline
        from src.api import create_app

        app = create_app()
        app.dependency_overrides[dep_vision_pipeline] = lambda: mock_vision_pipeline
        client = TestClient(app)

        response = client.post(
            "/v1/vision/analyze",
            json={},
        )

        assert response.status_code == 400
        assert "Provide image_path" in response.json()["detail"]

    def test_analyze_file_not_found(self, test_client, mock_vision_pipeline):
        """Test error when file doesn't exist."""
        from src.api.dependencies import dep_vision_pipeline
        from src.api import create_app

        app = create_app()
        app.dependency_overrides[dep_vision_pipeline] = lambda: mock_vision_pipeline
        client = TestClient(app)

        response = client.post(
            "/v1/vision/analyze",
            json={"image_path": "/nonexistent/file.jpg"},
        )

        assert response.status_code == 404

    def test_analyze_success(self, test_client, mock_vision_pipeline, tmp_path):
        """Test successful analysis."""
        from src.api.dependencies import dep_vision_pipeline
        from src.api import create_app
        from PIL import Image

        # Create test file
        test_image = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100)).save(test_image)

        # Configure mock
        mock_vision_pipeline.analyze.return_value = AnalyzeResult(
            image_id="test123",
            path=str(test_image),
            description="A test image",
        )

        app = create_app()
        app.dependency_overrides[dep_vision_pipeline] = lambda: mock_vision_pipeline
        client = TestClient(app)

        response = client.post(
            "/v1/vision/analyze",
            json={"image_path": str(test_image)},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["image_id"] == "test123"
        assert data["description"] == "A test image"

    def test_analyze_with_base64(self, test_client, mock_vision_pipeline):
        """Test analysis with base64 input."""
        import base64
        from io import BytesIO
        from PIL import Image
        from src.api.dependencies import dep_vision_pipeline
        from src.api import create_app

        # Create base64 image
        img = Image.new("RGB", (50, 50), color="red")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Configure mock
        mock_vision_pipeline.analyze.return_value = AnalyzeResult(
            image_id="b64test",
            description="Base64 image",
        )

        app = create_app()
        app.dependency_overrides[dep_vision_pipeline] = lambda: mock_vision_pipeline
        client = TestClient(app)

        response = client.post(
            "/v1/vision/analyze",
            json={"image_base64": img_b64},
        )

        assert response.status_code == 200


class TestBatchEndpoint:
    """Tests for /v1/vision/batch endpoint."""

    def test_batch_missing_input(self, test_client, mock_batch_processor):
        """Test error when no input provided."""
        from src.api.dependencies import dep_vision_batch_processor
        from src.api import create_app

        app = create_app()
        app.dependency_overrides[dep_vision_batch_processor] = lambda: mock_batch_processor
        client = TestClient(app)

        response = client.post(
            "/v1/vision/batch",
            json={},
        )

        assert response.status_code == 400
        assert "Provide input_directory" in response.json()["detail"]

    def test_batch_create_job(self, test_client, mock_batch_processor, tmp_path):
        """Test creating a batch job."""
        from src.vision.batch import BatchJob
        from src.api.dependencies import dep_vision_batch_processor
        from src.api import create_app

        mock_job = BatchJob(
            job_id="job123",
            status=JobStatus.PENDING,
            total_items=10,
        )

        mock_batch_processor.create_job.return_value = mock_job

        app = create_app()
        app.dependency_overrides[dep_vision_batch_processor] = lambda: mock_batch_processor
        client = TestClient(app)

        response = client.post(
            "/v1/vision/batch",
            json={"input_directory": str(tmp_path)},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job123"
        assert data["status"] == "pending"

    def test_batch_get_status(self, test_client, mock_batch_processor):
        """Test getting batch job status."""
        from src.vision.models import BatchStatusResponse
        from src.api.dependencies import dep_vision_batch_processor
        from src.api import create_app

        mock_status = BatchStatusResponse(
            job_id="job123",
            status=JobStatus.RUNNING,
            total_items=100,
            processed_items=50,
            failed_items=2,
            elapsed_seconds=30.5,
        )

        mock_batch_processor.get_job_status.return_value = mock_status

        app = create_app()
        app.dependency_overrides[dep_vision_batch_processor] = lambda: mock_batch_processor
        client = TestClient(app)

        response = client.get("/v1/vision/batch/job123")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job123"
        assert data["processed_items"] == 50

    def test_batch_not_found(self, test_client, mock_batch_processor):
        """Test 404 for unknown job."""
        from src.api.dependencies import dep_vision_batch_processor
        from src.api import create_app

        mock_batch_processor.get_job_status.return_value = None

        app = create_app()
        app.dependency_overrides[dep_vision_batch_processor] = lambda: mock_batch_processor
        client = TestClient(app)

        response = client.get("/v1/vision/batch/unknown_job")

        assert response.status_code == 404


class TestSearchEndpoint:
    """Tests for /v1/vision/search endpoint."""

    def test_search_description(self, test_client, mock_vision_search):
        """Test description search."""
        from src.vision.models import SearchResult
        from src.api.dependencies import dep_vision_search
        from src.api import create_app

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

        mock_vision_search.search.return_value = mock_response

        app = create_app()
        app.dependency_overrides[dep_vision_search] = lambda: mock_vision_search
        client = TestClient(app)

        response = client.post(
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

    def test_list_persons(self, test_client, mock_vision_search):
        """Test listing known persons."""
        from src.api.dependencies import dep_vision_search
        from src.api import create_app

        mock_vision_search.list_persons.return_value = [
            {
                "person_id": "p1",
                "name": "John Doe",
                "photo_count": 25,
            },
        ]

        app = create_app()
        app.dependency_overrides[dep_vision_search] = lambda: mock_vision_search
        client = TestClient(app)

        response = client.get("/v1/vision/faces")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["persons"][0]["name"] == "John Doe"

    def test_update_person(self, test_client, mock_vision_search):
        """Test updating person name."""
        from src.api.dependencies import dep_vision_search
        from src.api import create_app

        mock_vision_search.update_person.return_value = True

        app = create_app()
        app.dependency_overrides[dep_vision_search] = lambda: mock_vision_search
        client = TestClient(app)

        response = client.put(
            "/v1/vision/faces/p1",
            json={"name": "Jane Doe"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "updated"

    def test_update_person_not_found(self, test_client, mock_vision_search):
        """Test updating nonexistent person."""
        from src.api.dependencies import dep_vision_search
        from src.api import create_app

        mock_vision_search.update_person.return_value = False

        app = create_app()
        app.dependency_overrides[dep_vision_search] = lambda: mock_vision_search
        client = TestClient(app)

        response = client.put(
            "/v1/vision/faces/unknown",
            json={"name": "Test"},
        )

        assert response.status_code == 404


class TestVideoEndpoint:
    """Tests for /v1/vision/video/analyze endpoint."""

    def test_video_not_found(self, test_client, mock_video_processor):
        """Test error when video doesn't exist."""
        from src.api.dependencies import dep_vision_video_processor
        from src.api import create_app

        app = create_app()
        app.dependency_overrides[dep_vision_video_processor] = lambda: mock_video_processor
        client = TestClient(app)

        response = client.post(
            "/v1/vision/video/analyze",
            json={"video_path": "/nonexistent/video.mp4"},
        )

        assert response.status_code == 404

    def test_video_analyze(self, test_client, mock_video_processor, tmp_path):
        """Test video analysis."""
        from src.vision.models import VideoAnalyzeResponse, VideoFrameResult
        from src.api.dependencies import dep_vision_video_processor
        from src.api import create_app

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

        mock_video_processor.analyze.return_value = mock_response

        app = create_app()
        app.dependency_overrides[dep_vision_video_processor] = lambda: mock_video_processor
        client = TestClient(app)

        response = client.post(
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

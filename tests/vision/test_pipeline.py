"""Tests for the vision pipeline.

Note: Requires sqlalchemy for database models.
"""

import pytest

# Skip entire module if sqlalchemy is not available (required by vision pipeline)
pytest.importorskip("sqlalchemy", reason="sqlalchemy required for vision pipeline tests")

from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from src.vision.models import AnalyzerType, AnalyzeResult
from src.vision.pipeline import VisionPipeline, get_pipeline


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    img = Image.new("RGB", (100, 100), color="red")
    return img


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a test image file."""
    img = Image.new("RGB", (100, 100), color="blue")
    path = tmp_path / "test_image.jpg"
    img.save(path)
    return path


class TestVisionPipeline:
    """Tests for VisionPipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        pipeline = VisionPipeline()
        assert not pipeline._initialized

        pipeline.initialize([AnalyzerType.EXIF_EXTRACT])
        assert pipeline._initialized
        assert AnalyzerType.EXIF_EXTRACT in pipeline._analyzers

    def test_pipeline_cleanup(self):
        """Test pipeline cleanup releases resources."""
        pipeline = VisionPipeline()
        pipeline.initialize([AnalyzerType.EXIF_EXTRACT])
        pipeline.cleanup()

        assert not pipeline._initialized
        assert len(pipeline._analyzers) == 0

    def test_get_pipeline_singleton(self):
        """Test get_pipeline returns same instance."""
        p1 = get_pipeline()
        p2 = get_pipeline()
        assert p1 is p2

    def test_analyze_with_exif_only(self, sample_image_path):
        """Test analysis with just EXIF extraction."""
        pipeline = VisionPipeline()
        pipeline.initialize([AnalyzerType.EXIF_EXTRACT])

        result = pipeline.analyze(
            image=sample_image_path,
            analyzers=[AnalyzerType.EXIF_EXTRACT],
            store_results=False,
        )

        assert isinstance(result, AnalyzeResult)
        assert result.path == str(sample_image_path)
        assert result.image_id  # Should have computed hash

    @patch("src.vision.analyzers.face_detect.FaceDetectAnalyzer")
    def test_analyze_with_mocked_face_detect(self, mock_analyzer_class, sample_image):
        """Test analysis with mocked face detection."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.name = "face_detect"
        mock_instance.analyze.return_value = MagicMock(
            success=True,
            data={
                "faces": [
                    {
                        "bbox": {"x": 10, "y": 10, "width": 50, "height": 50},
                        "confidence": 0.95,
                    }
                ],
                "face_count": 1,
            },
        )
        mock_analyzer_class.return_value = mock_instance

        pipeline = VisionPipeline()
        # Manually set the mocked analyzer
        pipeline._analyzers[AnalyzerType.FACE_DETECT] = mock_instance
        pipeline._initialized = True

        result = pipeline.analyze(
            image=sample_image,
            analyzers=[AnalyzerType.FACE_DETECT],
            store_results=False,
        )

        assert len(result.faces) == 1
        assert result.faces[0].confidence == 0.95

    def test_preprocess_oversized_image(self):
        """Test that oversized images are resized."""
        pipeline = VisionPipeline()

        # Create large image
        large_img = Image.new("RGB", (5000, 5000), color="green")

        processed, error = pipeline._preprocess_image(large_img)

        assert max(processed.size) <= 4096
        assert error is not None  # Should have warning about resize

    def test_compute_hash(self, sample_image_path):
        """Test file hashing for deduplication."""
        pipeline = VisionPipeline()

        hash1 = pipeline._compute_hash(sample_image_path)
        hash2 = pipeline._compute_hash(sample_image_path)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length


class TestAnalyzers:
    """Tests for individual analyzers."""

    def test_exif_analyzer_with_pil(self, sample_image, sample_image_path):
        """Test EXIF extraction with PIL fallback."""
        from src.vision.analyzers.exif import ExifAnalyzer

        analyzer = ExifAnalyzer(use_exiftool=False)
        analyzer.initialize()

        result = analyzer.analyze(sample_image, sample_image_path)

        assert result.success
        assert "exif" in result.data
        assert result.data["exif"]["width"] == 100
        assert result.data["exif"]["height"] == 100

    def test_analyzer_path_convenience(self, sample_image_path):
        """Test analyze_path convenience method."""
        from src.vision.analyzers.exif import ExifAnalyzer

        analyzer = ExifAnalyzer(use_exiftool=False)
        analyzer.initialize()

        result = analyzer.analyze_path(sample_image_path)

        assert result.success

    def test_analyzer_path_not_found(self):
        """Test error handling for missing file."""
        from src.vision.analyzers.exif import ExifAnalyzer

        analyzer = ExifAnalyzer(use_exiftool=False)
        analyzer.initialize()

        result = analyzer.analyze_path("/nonexistent/path.jpg")

        assert not result.success
        assert result.error


class TestModels:
    """Tests for Pydantic models."""

    def test_analyze_request_defaults(self):
        """Test AnalyzeRequest default values."""
        from src.vision.models import AnalyzeRequest

        req = AnalyzeRequest(image_path="/test/path.jpg")

        assert req.analyzers == [AnalyzerType.VL_DESCRIBE]
        assert req.store_results is True
        assert req.return_crops is False

    def test_bounding_box_model(self):
        """Test BoundingBox model."""
        from src.vision.models import BoundingBox

        bbox = BoundingBox(x=10, y=20, width=100, height=50)

        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_job_status_enum(self):
        """Test JobStatus enum values."""
        from src.vision.models import JobStatus

        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"


class TestBatchProcessor:
    """Tests for batch processing."""

    def test_collect_files_by_extension(self, tmp_path):
        """Test file collection with extension filtering."""
        from src.vision.batch import BatchProcessor

        # Create test files
        (tmp_path / "test1.jpg").touch()
        (tmp_path / "test2.png").touch()
        (tmp_path / "test3.txt").touch()

        processor = BatchProcessor()

        files = list(
            processor._collect_files(
                input_directory=tmp_path,
                input_paths=None,
                recursive=False,
                extensions=["jpg", "png"],
            )
        )

        assert len(files) == 2
        assert all(f.suffix in [".jpg", ".png"] for f in files)

    def test_collect_files_recursive(self, tmp_path):
        """Test recursive file collection."""
        from src.vision.batch import BatchProcessor

        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "test1.jpg").touch()
        (subdir / "test2.jpg").touch()

        processor = BatchProcessor()

        files = list(
            processor._collect_files(
                input_directory=tmp_path,
                input_paths=None,
                recursive=True,
                extensions=["jpg"],
            )
        )

        assert len(files) == 2


class TestSearch:
    """Tests for search functionality."""

    @patch("src.vision.search.search_descriptions")
    @patch("src.vision.search.TextEmbedAnalyzer")
    def test_search_by_description(self, mock_embedder_class, mock_search):
        """Test description-based search."""
        from src.vision.search import VisionSearch

        # Setup mocks
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384
        mock_embedder_class.return_value = mock_embedder

        mock_search.return_value = {
            "ids": [["img1", "img2"]],
            "metadatas": [[{"path": "/p1"}, {"path": "/p2"}]],
            "distances": [[0.1, 0.2]],
            "documents": [["desc1", "desc2"]],
        }

        search = VisionSearch()
        search._text_embedder = mock_embedder

        result = search.search("test query", search_type="description", limit=10)

        assert len(result.results) == 2
        assert result.results[0].score == pytest.approx(0.9, rel=0.01)


class TestVideoProcessor:
    """Tests for video processing."""

    def test_video_processor_ffmpeg_check(self):
        """Test ffmpeg availability check."""
        from src.vision.video import VideoProcessor

        processor = VideoProcessor()
        # Just check it doesn't crash
        assert isinstance(processor._ffmpeg_available, bool)

    @patch("subprocess.run")
    def test_get_video_info(self, mock_run):
        """Test video info extraction."""
        from src.vision.video import VideoProcessor

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"format": {"duration": "60.0"}, "streams": [{"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30/1"}]}',
        )

        processor = VideoProcessor()
        processor._ffmpeg_available = True

        info = processor.get_video_info(Path("/test/video.mp4"))

        assert info is not None
        assert info["duration"] == 60.0
        assert info["width"] == 1920
        assert info["fps"] == 30.0

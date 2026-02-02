"""Unit tests for PromptCompressor (LLMLingua-2 wrapper)."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.services.prompt_compressor import PromptCompressor, CompressionResult


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    def test_compression_result_fields(self):
        """Test that CompressionResult has all expected fields."""
        result = CompressionResult(
            compressed_text="test compressed",
            original_chars=100,
            compressed_chars=50,
            original_tokens=20,
            compressed_tokens=10,
            actual_ratio=0.5,
            latency_ms=25.5,
        )

        assert result.compressed_text == "test compressed"
        assert result.original_chars == 100
        assert result.compressed_chars == 50
        assert result.original_tokens == 20
        assert result.compressed_tokens == 10
        assert result.actual_ratio == 0.5
        assert result.latency_ms == 25.5


class TestPromptCompressorInit:
    """Tests for PromptCompressor initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        compressor = PromptCompressor()

        assert compressor.model_name == "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        assert compressor.device == "cpu"
        assert compressor._model is None
        assert not compressor.is_loaded

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        compressor = PromptCompressor(
            model_name="custom-model",
            device="cuda",
        )

        assert compressor.model_name == "custom-model"
        assert compressor.device == "cuda"

    def test_singleton_instance(self):
        """Test get_instance returns singleton."""
        # Reset singleton
        PromptCompressor._instance = None

        instance1 = PromptCompressor.get_instance()
        instance2 = PromptCompressor.get_instance()

        assert instance1 is instance2

        # Clean up
        PromptCompressor._instance = None


class TestPromptCompressorCompress:
    """Tests for PromptCompressor.compress method."""

    @pytest.fixture
    def mock_llmlingua(self):
        """Create a mock LLMLingua compressor."""
        with patch("src.services.prompt_compressor.PromptCompressor._ensure_model_loaded") as mock:
            yield mock

    def test_compress_returns_result(self, mock_llmlingua):
        """Test compress returns CompressionResult."""
        compressor = PromptCompressor()

        # Mock the internal model
        mock_model = MagicMock()
        mock_model.compress_prompt.return_value = {
            "compressed_prompt": "reduced text here",
        }
        compressor._model = mock_model

        test_text = "This is a long test text that needs compression"
        result = compressor.compress(test_text, target_ratio=0.5)

        assert isinstance(result, CompressionResult)
        assert result.compressed_text == "reduced text here"
        assert result.original_chars == len(test_text)
        assert result.compressed_chars == 17
        assert result.actual_ratio == pytest.approx(17 / len(test_text), rel=0.01)

    def test_compress_with_force_tokens(self, mock_llmlingua):
        """Test compress with force_tokens parameter."""
        compressor = PromptCompressor()

        mock_model = MagicMock()
        mock_model.compress_prompt.return_value = {
            "compressed_prompt": "important term preserved",
        }
        compressor._model = mock_model

        result = compressor.compress(
            "The important term should be preserved in the output",
            target_ratio=0.5,
            force_tokens=["important term"],
        )

        # Verify force_tokens was passed to model
        mock_model.compress_prompt.assert_called_once()
        call_kwargs = mock_model.compress_prompt.call_args.kwargs
        assert call_kwargs.get("force_tokens") == ["important term"]

    def test_compress_empty_string(self, mock_llmlingua):
        """Test compress handles empty string."""
        compressor = PromptCompressor()

        mock_model = MagicMock()
        mock_model.compress_prompt.return_value = {"compressed_prompt": ""}
        compressor._model = mock_model

        result = compressor.compress("", target_ratio=0.5)

        assert result.compressed_text == ""
        assert result.actual_ratio == 1.0  # Division by zero fallback


class TestPromptCompressorCompressIfNeeded:
    """Tests for compress_if_needed method."""

    def test_skip_compression_below_threshold(self):
        """Test that compression is skipped for small texts."""
        compressor = PromptCompressor()

        # Text below threshold
        small_text = "This is a short text under the threshold."
        result_text, result_obj = compressor.compress_if_needed(
            small_text,
            min_chars=1000,
            target_ratio=0.5,
        )

        assert result_text == small_text
        assert result_obj is None

    def test_compress_above_threshold(self):
        """Test that compression is applied for texts above threshold."""
        compressor = PromptCompressor()

        # Mock internal model
        mock_model = MagicMock()
        mock_model.compress_prompt.return_value = {"compressed_prompt": "compressed version"}
        compressor._model = mock_model

        # Skip _ensure_model_loaded since model is already set
        with patch.object(compressor, "_ensure_model_loaded"):
            long_text = "x" * 500  # Above threshold
            result_text, result_obj = compressor.compress_if_needed(
                long_text,
                min_chars=100,
                target_ratio=0.5,
            )

        assert result_text == "compressed version"
        assert result_obj is not None
        assert isinstance(result_obj, CompressionResult)


class TestPromptCompressorModelLoading:
    """Tests for model loading behavior."""

    def test_lazy_loading(self):
        """Test that model is not loaded until first compress call."""
        compressor = PromptCompressor()

        assert compressor._model is None
        assert not compressor.is_loaded
        assert compressor.load_time_ms is None

    @patch("src.services.prompt_compressor.PromptCompressor._ensure_model_loaded")
    def test_model_loaded_on_compress(self, mock_ensure):
        """Test that model is loaded on first compress call."""
        compressor = PromptCompressor()

        # Mock model after load
        mock_model = MagicMock()
        mock_model.compress_prompt.return_value = {"compressed_prompt": "result"}

        def set_model():
            compressor._model = mock_model

        mock_ensure.side_effect = set_model

        compressor.compress("test text", target_ratio=0.5)

        mock_ensure.assert_called_once()


class TestPromptCompressorIntegration:
    """Integration tests that use real LLMLingua-2.

    These tests are marked slow and require the llmlingua package.
    Run with: pytest -v -m integration tests/unit/test_prompt_compressor.py
    """

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_compression(self):
        """Test compression with real LLMLingua-2 model."""
        try:
            from llmlingua import PromptCompressor as LLMLinguaCompressor  # noqa: F401
        except ImportError:
            pytest.skip("llmlingua not installed")

        compressor = PromptCompressor()

        # Long text for compression
        long_text = """
        This is a comprehensive document about machine learning and artificial intelligence.
        The field of AI has grown significantly over the past decade.
        Machine learning models can be trained on large datasets to perform various tasks.
        Deep learning uses neural networks with many layers to learn complex patterns.
        Natural language processing enables computers to understand human language.
        Computer vision allows machines to interpret and analyze images and videos.
        Reinforcement learning involves training agents through rewards and penalties.
        The future of AI holds great promise for solving complex problems.
        """

        result = compressor.compress(long_text, target_ratio=0.4)

        # Verify compression occurred
        assert result.compressed_chars < result.original_chars
        assert result.actual_ratio < 0.8  # Some meaningful compression
        assert result.latency_ms > 0

        # Model should be loaded now
        assert compressor.is_loaded
        assert compressor.load_time_ms is not None

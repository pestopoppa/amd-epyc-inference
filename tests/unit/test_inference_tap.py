#!/usr/bin/env python3
"""Tests for the streaming inference tap."""

import threading
from unittest.mock import MagicMock

from src.inference_tap import (
    TapWriter,
    _NullWriter,
    _read_sentinel,
    is_active,
    tap_section,
)


class TestIsActive:
    """Tests for tap activation check."""

    def test_inactive_by_default(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_TAP_FILE", raising=False)
        assert is_active() is False

    def test_active_with_env_var(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_TAP_FILE", "/tmp/tap.log")
        assert is_active() is True

    def test_inactive_with_empty_env_var(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_TAP_FILE", "")
        assert is_active() is False


class TestTapWriter:
    """Tests for TapWriter output format."""

    def test_output_format(self, tmp_path):
        path = str(tmp_path / "tap.log")
        w = TapWriter(path)

        w.write_header("coder_escalation")
        w.write_prompt("Hello world")
        w.write_chunk("def foo():")
        w.write_chunk("\n    pass")
        w.write_timings(10, 100.0, 500.0, 20.0)

        with open(path) as f:
            content = f.read()

        assert "ROLE=coder_escalation" in content
        assert "PROMPT:" in content
        assert "Hello world" in content
        assert "RESPONSE:" in content
        assert "def foo():" in content
        assert "\n    pass" in content
        assert "TIMINGS: 10 tokens in 0.60s" in content
        assert "prompt=100ms" in content
        assert "gen=500ms" in content
        assert "20.0 t/s" in content
        # Verify structure markers
        assert "=" * 72 in content
        assert "-" * 72 in content

    def test_prompt_truncation(self, tmp_path):
        path = str(tmp_path / "tap.log")
        w = TapWriter(path)
        w.write_header("coder")
        long_prompt = "x" * 3000
        w.write_prompt(long_prompt, max_chars=2000)

        with open(path) as f:
            content = f.read()

        assert "1000 chars truncated" in content

    def test_concurrent_writes(self, tmp_path):
        """4 threads writing simultaneously — no corruption."""
        path = str(tmp_path / "tap.log")
        errors = []

        def writer_fn(thread_id):
            try:
                w = TapWriter(path)
                for i in range(20):
                    w.write_chunk(f"[T{thread_id}:{i}]")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer_fn, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        with open(path) as f:
            content = f.read()

        # Each thread wrote 20 chunks
        for t in range(4):
            for i in range(20):
                assert f"[T{t}:{i}]" in content


class TestNullWriter:
    """Verify _NullWriter is a silent no-op."""

    def test_all_methods_noop(self):
        w = _NullWriter()
        # Should not raise
        w.write_header("x")
        w.write_prompt("x")
        w.write_chunk("x")
        w.write_timings(0, 0.0, 0.0, 0.0)


class TestTapSection:
    """Tests for tap_section context manager."""

    def test_yields_null_writer_when_inactive(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_TAP_FILE", raising=False)
        with tap_section("coder", "prompt") as w:
            assert isinstance(w, _NullWriter)

    def test_yields_tap_writer_when_active(self, tmp_path, monkeypatch):
        path = str(tmp_path / "tap.log")
        monkeypatch.setenv("INFERENCE_TAP_FILE", path)
        with tap_section("coder", "prompt text") as w:
            assert isinstance(w, TapWriter)
            w.write_chunk("hello")
            w.write_timings(5, 50.0, 200.0, 25.0)

        with open(path) as f:
            content = f.read()

        assert "ROLE=coder" in content
        assert "prompt text" in content
        assert "hello" in content
        assert "TIMINGS:" in content


class TestCachingBackendIntegration:
    """Tests verifying _call_caching_backend uses correct path."""

    def test_uses_streaming_when_tap_active(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_TAP_FILE", "/dev/null")

        mock_result = MagicMock()
        mock_result.tokens_generated = 10
        mock_result.prompt_eval_ms = 100.0
        mock_result.generation_ms = 500.0
        mock_result.predicted_per_second = 20.0
        mock_result.success = True
        mock_result.output = "hello"
        mock_result.http_overhead_ms = 0.0

        backend = MagicMock()
        backend.infer_stream_text = MagicMock(return_value=mock_result)
        backend.infer = MagicMock(return_value=mock_result)

        from src.prefix_cache import CachingBackend

        cb = CachingBackend(backend)
        role_config = MagicMock()
        request = MagicMock()
        request.prompt = "test prompt"

        cb.infer_stream_text(role_config, request, on_chunk=lambda c: None)

        backend.infer_stream_text.assert_called_once()
        backend.infer.assert_not_called()

    def test_uses_batch_when_tap_inactive(self, monkeypatch):
        monkeypatch.delenv("INFERENCE_TAP_FILE", raising=False)

        mock_result = MagicMock()
        mock_result.success = True

        backend = MagicMock()
        backend.infer = MagicMock(return_value=mock_result)

        from src.prefix_cache import CachingBackend

        cb = CachingBackend(backend)
        role_config = MagicMock()
        request = MagicMock()
        request.prompt = "test prompt"

        cb.infer(role_config, request)

        backend.infer.assert_called_once()


class TestInferStreamTextSSE:
    """Test SSE parsing in LlamaServerBackend.infer_stream_text()."""

    def test_parses_sse_chunks(self):
        """Mock httpx streaming to verify chunk extraction and InferenceResult."""
        from src.backends.llama_server import LlamaServerBackend, ServerConfig

        # Build SSE lines that iter_lines() would yield
        sse_lines = [
            'data: {"content": "Hello"}',
            'data: {"content": " world"}',
            'data: {"content": "!", "stop": true, "tokens_predicted": 3, '
            '"tokens_evaluated": 5, "tokens_cached": 2, '
            '"timings": {"prompt_ms": 50.0, "predicted_ms": 100.0, '
            '"predicted_per_second": 30.0}}',
        ]

        # Create a mock streaming response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines = MagicMock(return_value=iter(sse_lines))
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        config = ServerConfig(base_url="http://localhost:8080")
        backend = LlamaServerBackend(config)
        backend.client = mock_client

        # Minimal role_config mock
        role_config = MagicMock()
        role_config.name = "coder"
        role_config.acceleration.temperature = 0.0

        request = MagicMock()
        request.prompt = "test"
        request.n_tokens = 100
        request.timeout = 60
        request.temperature = 0.0
        request.cache_prompt = None
        request.stop_sequences = None

        received_chunks = []
        result = backend.infer_stream_text(
            role_config, request, on_chunk=lambda c: received_chunks.append(c)
        )

        assert received_chunks == ["Hello", " world", "!"]
        assert result.output == "Hello world!"
        assert result.tokens_generated == 3
        assert result.success is True
        assert result.prompt_eval_ms == 50.0
        assert result.generation_ms == 100.0
        assert result.predicted_per_second == 30.0


class TestSentinelFallback:
    """Tests for sentinel file fallback in is_active() / _tap_path()."""

    def _reset_cache(self):
        """Force-expire the sentinel cache so reads are fresh."""
        import src.inference_tap as _mod
        _mod._sentinel_cache = ("", 0.0)

    def test_is_active_true_with_sentinel(self, tmp_path, monkeypatch):
        """is_active() returns True when sentinel file exists."""
        sentinel = tmp_path / "sentinel"
        sentinel.write_text("/tmp/tap.log")

        monkeypatch.delenv("INFERENCE_TAP_FILE", raising=False)
        monkeypatch.setattr("src.inference_tap._SENTINEL", str(sentinel))
        self._reset_cache()

        assert is_active() is True

    def test_is_active_false_after_removal(self, tmp_path, monkeypatch):
        """is_active() returns False after sentinel is removed (cache expired)."""
        sentinel = tmp_path / "sentinel"
        sentinel.write_text("/tmp/tap.log")

        monkeypatch.delenv("INFERENCE_TAP_FILE", raising=False)
        monkeypatch.setattr("src.inference_tap._SENTINEL", str(sentinel))
        self._reset_cache()

        assert is_active() is True

        # Remove sentinel and expire cache
        sentinel.unlink()
        self._reset_cache()

        assert is_active() is False

    def test_sentinel_cache_is_used(self, tmp_path, monkeypatch):
        """Reads within 5 seconds return cached value (no re-read)."""
        sentinel = tmp_path / "sentinel"
        sentinel.write_text("/tmp/tap.log")

        monkeypatch.delenv("INFERENCE_TAP_FILE", raising=False)
        monkeypatch.setattr("src.inference_tap._SENTINEL", str(sentinel))
        self._reset_cache()

        # First read populates cache
        assert _read_sentinel() == "/tmp/tap.log"

        # Remove file — cached value should still be returned
        sentinel.unlink()
        assert _read_sentinel() == "/tmp/tap.log"  # cached!

        # After expiring cache, read returns empty
        self._reset_cache()
        assert _read_sentinel() == ""

    def test_env_var_takes_precedence(self, tmp_path, monkeypatch):
        """Env var is checked before sentinel."""
        sentinel = tmp_path / "sentinel"
        sentinel.write_text("/tmp/sentinel_tap.log")

        monkeypatch.setenv("INFERENCE_TAP_FILE", "/tmp/env_tap.log")
        monkeypatch.setattr("src.inference_tap._SENTINEL", str(sentinel))
        self._reset_cache()

        assert is_active() is True
        # _tap_path should return env var, not sentinel
        from src.inference_tap import _tap_path
        assert _tap_path() == "/tmp/env_tap.log"

    def test_nonexistent_sentinel(self, tmp_path, monkeypatch):
        """Missing sentinel file returns empty string without error."""
        monkeypatch.delenv("INFERENCE_TAP_FILE", raising=False)
        monkeypatch.setattr("src.inference_tap._SENTINEL", str(tmp_path / "nope"))
        self._reset_cache()

        assert is_active() is False
        assert _read_sentinel() == ""

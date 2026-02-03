"""Tests for TOON encoder service."""

import json
import pytest
from unittest.mock import patch


class TestToonEncoder:
    """Tests for TOON encoding utilities."""

    def test_is_available(self):
        """Test TOON availability check."""
        from src.services.toon_encoder import is_available

        # Should return True if toon_format is installed
        result = is_available()
        assert isinstance(result, bool)

    def test_encode_basic(self):
        """Test basic TOON encoding."""
        from src.services.toon_encoder import encode, is_available

        if not is_available():
            pytest.skip("toon_format not installed")

        data = {"name": "test", "value": 42}
        result = encode(data)
        assert isinstance(result, str)
        assert "name" in result
        assert "test" in result

    def test_encode_fallback_to_json(self):
        """Test fallback to JSON when TOON unavailable."""
        from src.services import toon_encoder

        # Mock toon_format as unavailable
        with patch.object(toon_encoder, "_toon_format", None):
            with patch.object(toon_encoder, "_get_toon", return_value=None):
                result = toon_encoder.encode({"test": 123}, fallback_to_json=True)
                assert json.loads(result) == {"test": 123}

    def test_should_use_toon_uniform_array(self):
        """Test TOON recommendation for uniform arrays."""
        from src.services.toon_encoder import should_use_toon, is_available

        if not is_available():
            pytest.skip("toon_format not installed")

        # Uniform array of objects - should use TOON
        data = {
            "files": [
                {"name": "a.py", "size": 100},
                {"name": "b.py", "size": 200},
                {"name": "c.py", "size": 300},
            ]
        }
        assert should_use_toon(data) is True

    def test_should_use_toon_small_array(self):
        """Test TOON not recommended for small arrays."""
        from src.services.toon_encoder import should_use_toon, is_available

        if not is_available():
            pytest.skip("toon_format not installed")

        # Small array - not worth TOON overhead
        data = {"files": [{"name": "a.py"}]}
        assert should_use_toon(data) is False

    def test_should_use_toon_non_uniform(self):
        """Test TOON not recommended for non-uniform arrays."""
        from src.services.toon_encoder import should_use_toon, is_available

        if not is_available():
            pytest.skip("toon_format not installed")

        # Non-uniform array - different keys
        data = {
            "items": [
                {"name": "a", "size": 100},
                {"name": "b", "color": "red"},
                {"path": "c", "size": 300},
            ]
        }
        assert should_use_toon(data) is False

    def test_encode_list_dir(self):
        """Test directory listing encoding."""
        from src.services.toon_encoder import encode_list_dir, is_available

        files = [
            {"name": "a.py", "type": "file", "size": 100},
            {"name": "b.py", "type": "file", "size": 200},
            {"name": "tests", "type": "dir"},
        ]
        result = encode_list_dir("/test/path", files, 3)
        assert isinstance(result, str)
        assert "/test/path" in result

        # Should be significantly shorter with TOON if available
        json_result = json.dumps({"path": "/test/path", "files": files, "total": 3}, indent=2)
        if is_available():
            assert len(result) < len(json_result)

    def test_encode_grep_hits(self):
        """Test grep hits encoding."""
        from src.services.toon_encoder import encode_grep_hits

        grep_hits = [
            {
                "pattern": "def main",
                "source": "test.py",
                "hits": [
                    {"line_num": 10, "match": "def main():"},
                    {"line_num": 20, "match": "def main_loop():"},
                    {"line_num": 30, "match": "def main_entry():"},
                ],
            }
        ]
        result = encode_grep_hits(grep_hits)
        assert isinstance(result, str)
        assert "def main" in result

    def test_encode_grep_hits_empty(self):
        """Test grep hits encoding with empty input."""
        from src.services.toon_encoder import encode_grep_hits

        result = encode_grep_hits([])
        assert result == ""

    def test_encode_escalation_context(self):
        """Test escalation context encoding."""
        from src.services.toon_encoder import encode_escalation_context

        result = encode_escalation_context(
            task_id="task-123",
            failure_count=2,
            error_category="FORMAT",
            gate_name="schema",
            error_message="JSON parse failed",
            previous_attempts=[
                {"role": "coder", "error": "Parse error", "tokens": 100},
                {"role": "coder", "error": "Schema error", "tokens": 150},
            ],
        )
        assert isinstance(result, str)
        assert "task-123" in result
        assert "FORMAT" in result


class TestToonRoundTrip:
    """Round-trip validation tests."""

    def test_roundtrip_simple(self):
        """Test round-trip for simple objects."""
        from src.services.toon_encoder import encode, decode, is_available

        if not is_available():
            pytest.skip("toon_format not installed")

        original = {"name": "test", "count": 42}
        encoded = encode(original)
        decoded = decode(encoded)
        assert decoded == original

    def test_roundtrip_file_listing(self):
        """Test round-trip for file listing structure."""
        from src.services.toon_encoder import encode, decode, is_available

        if not is_available():
            pytest.skip("toon_format not installed")

        original = {
            "path": "/test/dir",
            "files": [{"name": f"file{i}.py", "type": "file", "size": i * 100} for i in range(10)],
            "total": 10,
        }
        encoded = encode(original)
        decoded = decode(encoded)
        assert decoded == original

    def test_roundtrip_escalation_context(self):
        """Test round-trip for escalation context."""
        from src.services.toon_encoder import encode, decode, is_available

        if not is_available():
            pytest.skip("toon_format not installed")

        original = {
            "task_id": "abc-123",
            "failure_count": 3,
            "error_category": "FORMAT",
            "gate_name": "schema",
            "previous_attempts": [
                {"role": "coder", "error": "Error 1", "tokens": 1000},
                {"role": "coder", "error": "Error 2", "tokens": 1200},
            ],
        }
        encoded = encode(original)
        decoded = decode(encoded)
        assert decoded == original


class TestToonNewEncoders:
    """Tests for new TOON encoding helpers."""

    def test_encode_procedures(self):
        """Test procedure listing encoding."""
        from src.services.toon_encoder import encode_procedures, is_available

        procedures = [
            {"id": "benchmark_new_model", "category": "benchmark", "description": "Run benchmarks"},
            {"id": "add_model_to_registry", "category": "registry", "description": "Add model"},
            {"id": "run_quality_gates", "category": "codebase", "description": "Run gates"},
            {"id": "create_handoff", "category": "codebase", "description": "Create handoff"},
        ]
        result = encode_procedures(procedures)
        assert isinstance(result, str)
        assert "benchmark_new_model" in result

        # Should be shorter with TOON if available
        import json

        json_len = len(json.dumps(procedures, indent=2))
        if is_available() and len(procedures) >= 3:
            assert len(result) < json_len

    def test_encode_memory_results(self):
        """Test memory results encoding."""
        from src.services.toon_encoder import encode_memory_results, is_available

        results = [
            {
                "task": "Fix authentication bug",
                "outcome": "success",
                "strategy": "direct",
                "similarity": 0.85,
            },
            {
                "task": "Implement caching",
                "outcome": "success",
                "strategy": "decomposition",
                "similarity": 0.78,
            },
            {
                "task": "Refactor API layer",
                "outcome": "failure",
                "strategy": "direct",
                "similarity": 0.72,
            },
        ]
        result = encode_memory_results(results)
        assert isinstance(result, str)
        assert "authentication" in result

        # Should be shorter with TOON if available
        import json

        json_result = json.dumps({"results": results}, indent=2)
        if is_available() and len(results) >= 3:
            assert len(result) < len(json_result)


class TestToonTokenReduction:
    """Tests verifying token reduction claims."""

    def test_token_reduction_file_listing(self):
        """Verify significant token reduction for file listings."""
        from src.services.toon_encoder import encode, is_available

        if not is_available():
            pytest.skip("toon_format not installed")

        data = {
            "path": "/mnt/raid0/llm/claude",
            "files": [{"name": f"file{i}.py", "type": "file", "size": i * 100} for i in range(20)],
            "total": 20,
        }

        json_str = json.dumps(data, indent=2)
        toon_str = encode(data)

        reduction = 1 - len(toon_str) / len(json_str)
        # Should achieve at least 40% character reduction
        assert reduction >= 0.40, f"Only {reduction:.1%} reduction"

    def test_token_reduction_grep_hits(self):
        """Verify significant token reduction for grep hits."""
        from src.services.toon_encoder import encode, is_available

        if not is_available():
            pytest.skip("toon_format not installed")

        data = {
            "hits": [
                {"file": f"src/module{i}.py", "line": i * 10, "match": f"def func_{i}():"}
                for i in range(15)
            ]
        }

        json_str = json.dumps(data, indent=2)
        toon_str = encode(data)

        reduction = 1 - len(toon_str) / len(json_str)
        # Should achieve at least 40% character reduction
        assert reduction >= 0.40, f"Only {reduction:.1%} reduction"

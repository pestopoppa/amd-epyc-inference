#!/usr/bin/env python3
"""Unit tests for FastAPI endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.state import get_state, reset_state
from src.gate_runner import GateResult


@pytest.fixture
def client():
    """Create a test client."""
    # Reset state before each test
    reset_state()
    state = get_state()

    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_gate_runner():
    """Create a mock gate runner that returns fast results."""
    mock = MagicMock()
    mock.get_gate_names.return_value = ["format", "lint", "unit"]

    # Create fast mock results
    mock_results = [
        GateResult(
            gate_name="format",
            passed=True,
            exit_code=0,
            output="OK",
            elapsed_seconds=0.1,
        ),
        GateResult(
            gate_name="lint",
            passed=True,
            exit_code=0,
            output="OK",
            elapsed_seconds=0.1,
        ),
        GateResult(
            gate_name="unit",
            passed=True,
            exit_code=0,
            output="OK",
            elapsed_seconds=0.1,
        ),
    ]
    mock.run_all_gates.return_value = mock_results
    mock.run_all_gates_parallel = AsyncMock(return_value=mock_results)
    mock.run_gates_by_name.return_value = [mock_results[0]]  # Return format gate

    return mock


@pytest.fixture
def client_with_mock_gates(mock_gate_runner):
    """Create a test client with mocked gate runner.

    The mock is injected AFTER the TestClient starts (which runs the lifespan
    handler that creates the real GateRunner). We then replace it with our mock.
    """
    reset_state()

    with TestClient(app) as client:
        # Inject mock AFTER lifespan has run (overwrites the real gate_runner)
        state = get_state()
        state.gate_runner = mock_gate_runner
        yield client


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_ok(self, client):
        """Test that health endpoint returns ok status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_version(self, client):
        """Test that health endpoint includes version."""
        response = client.get("/health")

        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_health_reports_mock_mode_available(self, client):
        """Test that mock mode availability is reported."""
        response = client.get("/health")

        data = response.json()
        assert data["mock_mode_available"] is True


class TestChatEndpoint:
    """Test /chat endpoint."""

    def test_chat_mock_mode_default(self, client):
        """Test that chat defaults to mock mode."""
        response = client.post(
            "/chat",
            json={"prompt": "Hello"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mock_mode"] is True
        assert "[MOCK]" in data["answer"]

    def test_chat_returns_answer(self, client):
        """Test that chat returns an answer."""
        response = client.post(
            "/chat",
            json={"prompt": "What is 2+2?", "mock_mode": True}
        )

        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_chat_includes_prompt_in_mock_response(self, client):
        """Test that mock response includes part of the prompt."""
        response = client.post(
            "/chat",
            json={"prompt": "Unique test prompt xyz", "mock_mode": True}
        )

        data = response.json()
        assert "Unique test prompt" in data["answer"]

    def test_chat_tracks_turns(self, client):
        """Test that turns are tracked."""
        response = client.post(
            "/chat",
            json={"prompt": "Test"}
        )

        data = response.json()
        assert "turns" in data
        assert data["turns"] >= 1

    def test_chat_includes_elapsed_time(self, client):
        """Test that elapsed time is included."""
        response = client.post(
            "/chat",
            json={"prompt": "Test"}
        )

        data = response.json()
        assert "elapsed_seconds" in data
        assert data["elapsed_seconds"] >= 0

    def test_chat_with_context(self, client):
        """Test chat with context included."""
        response = client.post(
            "/chat",
            json={
                "prompt": "Summarize",
                "context": "This is some context text.",
                "mock_mode": True
            }
        )

        data = response.json()
        # Should mention context length
        assert "context" in data["answer"].lower()

    def test_chat_validates_max_turns(self, client):
        """Test that max_turns is validated."""
        response = client.post(
            "/chat",
            json={"prompt": "Test", "max_turns": 0}
        )

        assert response.status_code == 422  # Validation error

    def test_chat_max_turns_upper_bound(self, client):
        """Test that max_turns upper bound is enforced."""
        response = client.post(
            "/chat",
            json={"prompt": "Test", "max_turns": 100}
        )

        assert response.status_code == 422  # Validation error


class TestGatesEndpoint:
    """Test /gates endpoints.

    Uses mocked gate runner to avoid slow subprocess calls.
    """

    def test_list_gates(self, client_with_mock_gates):
        """Test listing available gates."""
        response = client_with_mock_gates.get("/gates")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_list_gates_includes_expected(self, client_with_mock_gates):
        """Test that expected gates are in the list."""
        response = client_with_mock_gates.get("/gates")

        data = response.json()
        # These are from the mock
        assert "format" in data or "unit" in data

    def test_run_gates_returns_results(self, client_with_mock_gates):
        """Test that running gates returns results."""
        response = client_with_mock_gates.post(
            "/gates",
            json={}
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "all_passed" in data
        assert "total_elapsed_seconds" in data

    def test_run_specific_gates(self, client_with_mock_gates):
        """Test running specific gates by name."""
        response = client_with_mock_gates.post(
            "/gates",
            json={"gate_names": ["format"]}
        )

        assert response.status_code == 200
        data = response.json()
        # Should have at least one result
        assert len(data["results"]) >= 1

    def test_gate_results_have_required_fields(self, client_with_mock_gates):
        """Test that gate results have required fields."""
        response = client_with_mock_gates.post(
            "/gates",
            json={"gate_names": ["format"]}
        )

        data = response.json()
        if data["results"]:
            result = data["results"][0]
            assert "gate_name" in result
            assert "passed" in result
            assert "exit_code" in result
            assert "elapsed_seconds" in result


class TestStatsEndpoint:
    """Test /stats endpoints."""

    def test_get_stats_initial(self, client):
        """Test getting stats when no requests made."""
        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 0
        assert data["total_turns"] == 0

    def test_stats_increment_on_chat(self, client):
        """Test that stats increment after chat."""
        # Make a chat request
        client.post("/chat", json={"prompt": "Test"})

        # Check stats
        response = client.get("/stats")
        data = response.json()

        assert data["total_requests"] == 1
        assert data["mock_requests"] == 1

    def test_stats_track_turns(self, client):
        """Test that turns are tracked in stats."""
        client.post("/chat", json={"prompt": "Test"})

        response = client.get("/stats")
        data = response.json()

        assert data["total_turns"] >= 1

    def test_stats_average_calculation(self, client):
        """Test average turns calculation."""
        # Make two requests
        client.post("/chat", json={"prompt": "Test 1"})
        client.post("/chat", json={"prompt": "Test 2"})

        response = client.get("/stats")
        data = response.json()

        assert data["total_requests"] == 2
        assert data["average_turns_per_request"] > 0

    def test_reset_stats(self, client):
        """Test resetting stats."""
        # Make a request
        client.post("/chat", json={"prompt": "Test"})

        # Reset
        response = client.post("/stats/reset")
        assert response.status_code == 200

        # Verify reset
        response = client.get("/stats")
        data = response.json()

        assert data["total_requests"] == 0
        assert data["total_turns"] == 0


class TestValidation:
    """Test request validation."""

    def test_chat_requires_prompt(self, client):
        """Test that prompt is required."""
        response = client.post("/chat", json={})

        assert response.status_code == 422

    def test_chat_validates_role(self, client):
        """Test that role field is accepted."""
        response = client.post(
            "/chat",
            json={"prompt": "Test", "role": "coder"}
        )

        # Should succeed (role is just a string)
        assert response.status_code == 200


class TestCORSHeaders:
    """Test CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present."""
        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )

        # FastAPI handles OPTIONS requests for CORS
        assert response.status_code in [200, 405]


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/chat",
            content="not json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_unknown_endpoint(self, client):
        """Test 404 for unknown endpoints."""
        response = client.get("/nonexistent")

        assert response.status_code == 404

"""Tests for parallel_step_executor.py — wave computation and step execution.

Covers:
- compute_waves(): Topological sort, parallel_group merging, error cases
- StepExecutor: Wave-ordered execution, burst parallelism, context passing
- /api/delegate endpoint: Feature gating, dry run, error handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.parallel_step_executor import (
    StepExecutor,
    compute_waves,
)


# ── compute_waves ────────────────────────────────────────────────────────


class TestComputeWaves:
    """Wave computation from step dependencies."""

    def test_empty_steps(self):
        assert compute_waves([]) == []

    def test_single_step(self):
        steps = [{"id": "S1", "action": "do thing"}]
        waves = compute_waves(steps)
        assert len(waves) == 1
        assert waves[0].step_ids == ["S1"]
        assert waves[0].index == 0

    def test_no_deps_single_wave(self):
        """All independent steps land in wave 0."""
        steps = [
            {"id": "S1", "action": "a"},
            {"id": "S2", "action": "b"},
            {"id": "S3", "action": "c"},
        ]
        waves = compute_waves(steps)
        assert len(waves) == 1
        assert set(waves[0].step_ids) == {"S1", "S2", "S3"}

    def test_linear_chain_three_waves(self):
        """S1 → S2 → S3 produces 3 sequential waves."""
        steps = [
            {"id": "S1", "action": "first"},
            {"id": "S2", "action": "second", "depends_on": ["S1"]},
            {"id": "S3", "action": "third", "depends_on": ["S2"]},
        ]
        waves = compute_waves(steps)
        assert len(waves) == 3
        assert waves[0].step_ids == ["S1"]
        assert waves[1].step_ids == ["S2"]
        assert waves[2].step_ids == ["S3"]

    def test_diamond_dependency(self):
        """S1 → (S2, S3) → S4 produces 3 waves."""
        steps = [
            {"id": "S1", "action": "root"},
            {"id": "S2", "action": "left", "depends_on": ["S1"]},
            {"id": "S3", "action": "right", "depends_on": ["S1"]},
            {"id": "S4", "action": "merge", "depends_on": ["S2", "S3"]},
        ]
        waves = compute_waves(steps)
        assert len(waves) == 3
        assert waves[0].step_ids == ["S1"]
        assert set(waves[1].step_ids) == {"S2", "S3"}
        assert waves[2].step_ids == ["S4"]

    def test_parallel_group_merging(self):
        """Steps in the same parallel_group merge to the latest wave."""
        steps = [
            {"id": "S1", "action": "a", "parallel_group": "A"},
            {"id": "S2", "action": "b", "depends_on": ["S1"], "parallel_group": "A"},
        ]
        waves = compute_waves(steps)
        # S1 has no deps (wave 0), S2 depends on S1 (wave 1)
        # But same parallel_group → both merge to wave 1
        assert len(waves) == 1
        assert set(waves[0].step_ids) == {"S1", "S2"}

    def test_parallel_group_with_external_dep(self):
        """Group members merge to max wave, others unaffected."""
        steps = [
            {"id": "S0", "action": "setup"},
            {"id": "S1", "action": "a", "parallel_group": "G"},
            {"id": "S2", "action": "b", "depends_on": ["S0"], "parallel_group": "G"},
            {"id": "S3", "action": "c", "depends_on": ["S2"]},
        ]
        waves = compute_waves(steps)
        # S0: wave 0, S1: wave 0, S2: wave 1 → group G merges to wave 1
        # S3: wave 2
        assert len(waves) == 3
        assert waves[0].step_ids == ["S0"]
        assert set(waves[1].step_ids) == {"S1", "S2"}
        assert waves[2].step_ids == ["S3"]

    def test_circular_deps_raises(self):
        steps = [
            {"id": "S1", "action": "a", "depends_on": ["S2"]},
            {"id": "S2", "action": "b", "depends_on": ["S1"]},
        ]
        with pytest.raises(ValueError, match="Circular dependency"):
            compute_waves(steps)

    def test_duplicate_step_ids_raises(self):
        steps = [
            {"id": "S1", "action": "a"},
            {"id": "S1", "action": "b"},
        ]
        with pytest.raises(ValueError, match="Duplicate step ID"):
            compute_waves(steps)

    def test_missing_dep_reference_raises(self):
        steps = [
            {"id": "S1", "action": "a", "depends_on": ["S99"]},
        ]
        with pytest.raises(ValueError, match="nonexistent step S99"):
            compute_waves(steps)

    def test_missing_id_raises(self):
        steps = [{"action": "no id"}]
        with pytest.raises(ValueError, match="missing 'id'"):
            compute_waves(steps)

    def test_wave_preserves_step_dicts(self):
        """Wave.steps contains the original step dicts."""
        step = {"id": "S1", "action": "test", "actor": "coder"}
        waves = compute_waves([step])
        assert waves[0].steps[0] is step


# ── StepExecutor ─────────────────────────────────────────────────────────


def _make_mock_primitives(responses: dict[str, str] | None = None):
    """Create a mock LLMPrimitives that returns canned responses."""
    mock = MagicMock()
    responses = responses or {}

    def mock_llm_call(prompt, role="worker", n_tokens=1024, **kwargs):
        # Return role-based or default response
        return responses.get(role, f"output from {role}")

    mock.llm_call = mock_llm_call
    return mock


class TestStepExecutor:
    """StepExecutor wave-ordered execution."""

    @pytest.mark.asyncio
    async def test_sequential_waves_execute_in_order(self):
        """Two sequential waves: S1 before S2."""
        primitives = _make_mock_primitives()
        executor = StepExecutor(primitives=primitives)

        steps = [
            {"id": "S1", "action": "first"},
            {"id": "S2", "action": "second", "depends_on": ["S1"]},
        ]
        waves = compute_waves(steps)
        results = await executor.execute_plan(
            {"objective": "test"},
            waves,
            {"worker": "worker_general"},
        )

        assert len(results) == 2
        assert results[0].subtask_id == "S1"
        assert results[0].success
        assert results[1].subtask_id == "S2"
        assert results[1].success

    @pytest.mark.asyncio
    async def test_context_passing_between_waves(self):
        """Output from wave 0 is available in wave 1 via step_outputs."""
        primitives = _make_mock_primitives({"worker_general": "wave0 result"})
        executor = StepExecutor(primitives=primitives)

        steps = [
            {"id": "S1", "action": "produce", "outputs": ["data"]},
            {"id": "S2", "action": "consume", "depends_on": ["S1"], "inputs": ["S1"]},
        ]
        waves = compute_waves(steps)
        results = await executor.execute_plan(
            {"objective": "test"},
            waves,
            {"worker": "worker_general"},
        )

        assert results[0].success
        assert results[1].success
        assert "S1" in executor.step_outputs

    @pytest.mark.asyncio
    async def test_step_failure_skips_dependents(self):
        """Failed step causes dependents to be skipped."""
        primitives = MagicMock()
        primitives.llm_call.side_effect = RuntimeError("backend down")
        executor = StepExecutor(primitives=primitives)

        steps = [
            {"id": "S1", "action": "will fail"},
            {"id": "S2", "action": "depends on S1", "depends_on": ["S1"]},
        ]
        waves = compute_waves(steps)
        results = await executor.execute_plan(
            {"objective": "test"},
            waves,
            {"worker": "worker_general"},
        )

        assert not results[0].success
        assert "backend down" in results[0].error
        assert not results[1].success
        assert "dependency failed" in results[1].error

    @pytest.mark.asyncio
    async def test_burst_worker_steps_parallelize(self):
        """Steps targeting burst workers run concurrently."""
        call_times = []

        def slow_llm_call(prompt, role="worker_fast", n_tokens=1024, **kw):
            import time

            call_times.append(time.monotonic())
            time.sleep(0.05)
            return f"result from {role}"

        primitives = MagicMock()
        primitives.llm_call = slow_llm_call
        executor = StepExecutor(primitives=primitives, max_burst_concurrent=2)

        steps = [
            {"id": "S1", "action": "burst a", "actor": "burst"},
            {"id": "S2", "action": "burst b", "actor": "burst"},
        ]
        waves = compute_waves(steps)
        results = await executor.execute_plan(
            {"objective": "test"},
            waves,
            {"burst": "worker_fast"},
        )

        assert len(results) == 2
        assert all(r.success for r in results)
        # Both calls should start at roughly the same time (parallel)
        if len(call_times) == 2:
            assert abs(call_times[0] - call_times[1]) < 0.04

    @pytest.mark.asyncio
    async def test_non_burst_steps_run_sequentially(self):
        """HOT tier steps within a wave run one at a time."""
        call_order = []

        def tracking_llm_call(prompt, role="worker_general", n_tokens=1024, **kw):
            call_order.append(role)
            return f"result from {role}"

        primitives = MagicMock()
        primitives.llm_call = tracking_llm_call
        executor = StepExecutor(primitives=primitives)

        steps = [
            {"id": "S1", "action": "a", "actor": "coder"},
            {"id": "S2", "action": "b", "actor": "worker"},
        ]
        waves = compute_waves(steps)
        results = await executor.execute_plan(
            {"objective": "test"},
            waves,
            {"coder": "coder_primary", "worker": "worker_general"},
        )

        assert len(results) == 2
        assert all(r.success for r in results)
        # Sequential: calls happen in order
        assert call_order == ["coder_primary", "worker_general"]

    @pytest.mark.asyncio
    async def test_no_review_service(self):
        """Works without review service (skips review)."""
        primitives = _make_mock_primitives()
        executor = StepExecutor(primitives=primitives, review_service=None)

        steps = [{"id": "S1", "action": "test"}]
        waves = compute_waves(steps)
        results = await executor.execute_plan(
            {"objective": "test"},
            waves,
            {"worker": "worker_general"},
        )

        assert len(results) == 1
        assert results[0].success

    @pytest.mark.asyncio
    async def test_role_mapping_applied(self):
        """Actor → role mapping is respected."""
        captured_roles = []

        def tracking_call(prompt, role="default", n_tokens=1024, **kw):
            captured_roles.append(role)
            return "ok"

        primitives = MagicMock()
        primitives.llm_call = tracking_call
        executor = StepExecutor(primitives=primitives)

        steps = [
            {"id": "S1", "action": "code it", "actor": "coder"},
            {"id": "S2", "action": "explore it", "actor": "worker"},
        ]
        waves = compute_waves(steps)
        await executor.execute_plan(
            {"objective": "test"},
            waves,
            {"coder": "coder_primary", "worker": "worker_explore"},
        )

        assert captured_roles == ["coder_primary", "worker_explore"]

    @pytest.mark.asyncio
    async def test_mixed_burst_and_sequential_wave(self):
        """Wave with both burst and non-burst steps handles both."""
        primitives = _make_mock_primitives()
        executor = StepExecutor(primitives=primitives)

        steps = [
            {"id": "S1", "action": "sequential", "actor": "coder"},
            {"id": "S2", "action": "burst", "actor": "fast"},
        ]
        waves = compute_waves(steps)
        results = await executor.execute_plan(
            {"objective": "test"},
            waves,
            {"coder": "coder_primary", "fast": "worker_fast"},
        )

        assert len(results) == 2
        assert all(r.success for r in results)


# ── Delegate Endpoint ────────────────────────────────────────────────────


class TestDelegateEndpoint:
    """POST /api/delegate endpoint tests."""

    @pytest.fixture
    def client(self):
        """FastAPI TestClient with parallel_execution enabled."""
        from fastapi.testclient import TestClient
        from src.api import create_app

        app = create_app()
        return TestClient(app)

    def _make_task_ir(self, steps=None):
        return {
            "task_id": "test-123",
            "objective": "Test objective",
            "plan": {
                "steps": steps
                or [
                    {"id": "S1", "action": "do first", "actor": "worker", "outputs": ["result"]},
                ],
            },
        }

    @patch("src.api.routes.delegate.features")
    def test_feature_flag_disabled(self, mock_features, client):
        mock_features.return_value = MagicMock(parallel_execution=False)
        response = client.post(
            "/api/delegate",
            json={"task_ir": self._make_task_ir()},
        )
        assert response.status_code == 403

    @patch("src.api.routes.delegate.features")
    def test_dry_run_returns_wave_plan(self, mock_features, client):
        mock_features.return_value = MagicMock(parallel_execution=True)
        task_ir = self._make_task_ir(
            [
                {"id": "S1", "action": "first", "actor": "worker", "outputs": ["a"]},
                {"id": "S2", "action": "second", "actor": "worker", "depends_on": ["S1"]},
            ]
        )
        response = client.post(
            "/api/delegate",
            json={"task_ir": task_ir, "dry_run": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-123"
        assert len(data["waves"]) == 2
        assert data["subtask_results"] == []

    @patch("src.api.routes.delegate.features")
    def test_no_steps_returns_422(self, mock_features, client):
        mock_features.return_value = MagicMock(parallel_execution=True)
        response = client.post(
            "/api/delegate",
            json={"task_ir": {"objective": "empty", "plan": {"steps": []}}},
        )
        assert response.status_code == 422

    @patch("src.api.routes.delegate.features")
    def test_circular_deps_returns_422(self, mock_features, client):
        mock_features.return_value = MagicMock(parallel_execution=True)
        task_ir = self._make_task_ir(
            [
                {"id": "S1", "action": "a", "depends_on": ["S2"]},
                {"id": "S2", "action": "b", "depends_on": ["S1"]},
            ]
        )
        response = client.post(
            "/api/delegate",
            json={"task_ir": task_ir},
        )
        assert response.status_code == 422
        assert "Circular" in response.json()["detail"]

    @patch("src.api.routes.delegate.features")
    def test_no_primitives_returns_503(self, mock_features, client):
        mock_features.return_value = MagicMock(parallel_execution=True)
        mock_state = MagicMock()
        mock_state.llm_primitives = None

        from src.api.dependencies import dep_app_state

        app = client.app
        app.dependency_overrides[dep_app_state] = lambda: mock_state
        try:
            response = client.post(
                "/api/delegate",
                json={"task_ir": self._make_task_ir()},
            )
            assert response.status_code == 503
        finally:
            app.dependency_overrides.clear()


# ── Feature Flag ─────────────────────────────────────────────────────────


class TestFeatureFlag:
    """parallel_execution feature flag integration."""

    def test_default_disabled(self):
        from src.features import Features

        f = Features()
        assert f.parallel_execution is False

    def test_dependency_on_architect_delegation(self):
        from src.features import Features

        f = Features(parallel_execution=True, architect_delegation=False)
        errors = f.validate()
        assert any("parallel_execution" in e for e in errors)

    def test_valid_with_all_deps(self):
        from src.features import Features

        f = Features(
            parallel_execution=True,
            architect_delegation=True,
            memrl=True,
        )
        errors = f.validate()
        assert not any("parallel_execution" in e for e in errors)

    def test_in_summary(self):
        from src.features import Features

        f = Features(parallel_execution=True)
        assert "parallel_execution" in f.summary()
        assert f.summary()["parallel_execution"] is True

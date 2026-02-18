#!/usr/bin/env python3
"""Unit tests for API request models."""

import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Load the models file directly to avoid src.api.__init__ which transitively
# imports pydantic_graph and other heavy dependencies not needed for unit tests.
_ROOT = Path(__file__).resolve().parents[2] / "src" / "api" / "models"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f"{name.split('.')[-1]}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module("src.api.models.requests")
ChatRequest = _mod.ChatRequest
GateRequest = _mod.GateRequest
RewardRequest = _mod.RewardRequest


class TestChatRequest:
    """Test ChatRequest field validation."""

    def test_minimal(self):
        req = ChatRequest(prompt="Hello")
        assert req.prompt == "Hello"

    def test_defaults(self):
        req = ChatRequest(prompt="x")
        assert req.context == ""
        assert req.mock_mode is True
        assert req.real_mode is False
        assert req.max_turns == 10
        assert req.role == ""
        assert req.force_role is None
        assert req.force_mode is None
        assert req.allow_delegation is None
        assert req.server_urls is None
        assert req.image_path is None
        assert req.image_base64 is None
        assert req.files is None
        assert req.cache_prompt is None
        assert req.thinking_budget == 0
        assert req.permission_mode == "normal"
        assert req.session_id is None

    def test_prompt_required(self):
        with pytest.raises(ValidationError):
            ChatRequest()  # type: ignore[call-arg]

    def test_max_turns_lower_bound(self):
        req = ChatRequest(prompt="x", max_turns=1)
        assert req.max_turns == 1

    def test_max_turns_upper_bound(self):
        req = ChatRequest(prompt="x", max_turns=50)
        assert req.max_turns == 50

    def test_max_turns_zero_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(prompt="x", max_turns=0)

    def test_max_turns_over_limit_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(prompt="x", max_turns=51)

    def test_thinking_budget_lower_bound(self):
        req = ChatRequest(prompt="x", thinking_budget=0)
        assert req.thinking_budget == 0

    def test_thinking_budget_upper_bound(self):
        req = ChatRequest(prompt="x", thinking_budget=32000)
        assert req.thinking_budget == 32000

    def test_thinking_budget_negative_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(prompt="x", thinking_budget=-1)

    def test_thinking_budget_over_limit_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(prompt="x", thinking_budget=32001)

    def test_force_mode_values(self):
        for mode in ("direct", "react", "repl", "delegated"):
            req = ChatRequest(prompt="x", force_mode=mode)
            assert req.force_mode == mode

    def test_session_id_optional(self):
        req = ChatRequest(prompt="x", session_id="sess_123")
        assert req.session_id == "sess_123"


class TestRewardRequest:
    """Test RewardRequest field validation."""

    def test_minimal(self):
        req = RewardRequest(
            task_description="test task", action="frontdoor:direct", reward=0.5
        )
        assert req.reward == 0.5

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            RewardRequest(task_description="t", action="a")  # type: ignore[call-arg]

    def test_reward_lower_bound(self):
        req = RewardRequest(task_description="t", action="a", reward=-1.0)
        assert req.reward == -1.0

    def test_reward_upper_bound(self):
        req = RewardRequest(task_description="t", action="a", reward=1.0)
        assert req.reward == 1.0

    def test_reward_below_minus_one_rejected(self):
        with pytest.raises(ValidationError):
            RewardRequest(task_description="t", action="a", reward=-1.1)

    def test_reward_above_one_rejected(self):
        with pytest.raises(ValidationError):
            RewardRequest(task_description="t", action="a", reward=1.1)

    def test_defaults(self):
        req = RewardRequest(task_description="t", action="a", reward=0.0)
        assert req.context is None
        assert req.embedding is None

    def test_with_embedding(self):
        req = RewardRequest(
            task_description="t",
            action="a",
            reward=0.5,
            embedding=[0.1, 0.2, 0.3],
        )
        assert len(req.embedding) == 3


class TestGateRequest:
    """Test GateRequest defaults and fields."""

    def test_defaults(self):
        req = GateRequest()
        assert req.gate_names is None
        assert req.stop_on_first_failure is True
        assert req.required_only is False

    def test_specific_gates(self):
        req = GateRequest(gate_names=["shellcheck", "format"])
        assert req.gate_names == ["shellcheck", "format"]

    def test_override_defaults(self):
        req = GateRequest(stop_on_first_failure=False, required_only=True)
        assert req.stop_on_first_failure is False
        assert req.required_only is True

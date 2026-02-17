#!/usr/bin/env python3
"""Unit tests for OpenAI-compatible API models."""

import importlib.util
import sys
import time
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


_mod = _load_module("src.api.models.openai")
OpenAIChatRequest = _mod.OpenAIChatRequest
OpenAIChatResponse = _mod.OpenAIChatResponse
OpenAIChoice = _mod.OpenAIChoice
OpenAIMessage = _mod.OpenAIMessage
OpenAIModelInfo = _mod.OpenAIModelInfo
OpenAIModelsResponse = _mod.OpenAIModelsResponse
OpenAIUsage = _mod.OpenAIUsage


class TestOpenAIMessage:
    """Test OpenAIMessage validation."""

    def test_valid_message(self):
        msg = OpenAIMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_role_required(self):
        with pytest.raises(ValidationError):
            OpenAIMessage(content="Hello")  # type: ignore[call-arg]

    def test_content_required(self):
        with pytest.raises(ValidationError):
            OpenAIMessage(role="user")  # type: ignore[call-arg]

    def test_system_role(self):
        msg = OpenAIMessage(role="system", content="You are a helper")
        assert msg.role == "system"

    def test_assistant_role(self):
        msg = OpenAIMessage(role="assistant", content="Sure!")
        assert msg.role == "assistant"


class TestOpenAIChatRequest:
    """Test OpenAIChatRequest field constraints."""

    def _messages(self):
        return [OpenAIMessage(role="user", content="Hi")]

    def test_defaults(self):
        req = OpenAIChatRequest(messages=self._messages())
        assert req.model == "orchestrator"
        assert req.temperature == 0.0
        assert req.max_tokens == 1024
        assert req.stream is False
        assert req.x_orchestrator_role is None
        assert req.x_show_routing is False

    def test_temperature_lower_bound(self):
        req = OpenAIChatRequest(messages=self._messages(), temperature=0.0)
        assert req.temperature == 0.0

    def test_temperature_upper_bound(self):
        req = OpenAIChatRequest(messages=self._messages(), temperature=2.0)
        assert req.temperature == 2.0

    def test_temperature_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            OpenAIChatRequest(messages=self._messages(), temperature=-0.1)

    def test_temperature_above_two_rejected(self):
        with pytest.raises(ValidationError):
            OpenAIChatRequest(messages=self._messages(), temperature=2.1)

    def test_max_tokens_lower_bound(self):
        req = OpenAIChatRequest(messages=self._messages(), max_tokens=1)
        assert req.max_tokens == 1

    def test_max_tokens_upper_bound(self):
        req = OpenAIChatRequest(messages=self._messages(), max_tokens=32768)
        assert req.max_tokens == 32768

    def test_max_tokens_zero_rejected(self):
        with pytest.raises(ValidationError):
            OpenAIChatRequest(messages=self._messages(), max_tokens=0)

    def test_max_tokens_over_limit_rejected(self):
        with pytest.raises(ValidationError):
            OpenAIChatRequest(messages=self._messages(), max_tokens=32769)

    def test_messages_required(self):
        with pytest.raises(ValidationError):
            OpenAIChatRequest()  # type: ignore[call-arg]


class TestOpenAIChatResponse:
    """Test OpenAIChatResponse factory defaults."""

    def test_id_format(self):
        resp = OpenAIChatResponse(choices=[])
        assert resp.id.startswith("chatcmpl-")
        assert len(resp.id) == len("chatcmpl-") + 8

    def test_unique_ids(self):
        r1 = OpenAIChatResponse(choices=[])
        r2 = OpenAIChatResponse(choices=[])
        assert r1.id != r2.id

    def test_created_timestamp(self):
        before = int(time.time())
        resp = OpenAIChatResponse(choices=[])
        after = int(time.time())
        assert before <= resp.created <= after

    def test_object_type(self):
        resp = OpenAIChatResponse(choices=[])
        assert resp.object == "chat.completion"

    def test_with_choice_and_usage(self):
        choice = OpenAIChoice(
            index=0,
            message=OpenAIMessage(role="assistant", content="Hi"),
            finish_reason="stop",
        )
        usage = OpenAIUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp = OpenAIChatResponse(choices=[choice], usage=usage)
        assert len(resp.choices) == 1
        assert resp.usage.total_tokens == 15


class TestOpenAIUsage:
    """Test OpenAIUsage defaults."""

    def test_defaults(self):
        usage = OpenAIUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestOpenAIModelsResponse:
    """Test OpenAIModelsResponse construction."""

    def test_empty_list(self):
        resp = OpenAIModelsResponse(data=[])
        assert resp.object == "list"
        assert resp.data == []

    def test_with_models(self):
        m1 = OpenAIModelInfo(id="model-a")
        m2 = OpenAIModelInfo(id="model-b")
        resp = OpenAIModelsResponse(data=[m1, m2])
        assert len(resp.data) == 2
        assert resp.data[0].id == "model-a"
        assert resp.data[0].owned_by == "orchestrator"

    def test_model_info_defaults(self):
        info = OpenAIModelInfo(id="test")
        assert info.object == "model"
        assert info.owned_by == "orchestrator"
        assert isinstance(info.created, int)

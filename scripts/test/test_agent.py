from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import scripts.utilities.agent as agent_module
from scripts.utilities.agent import AgentBackend, OpenAICompatibleAgent, build_agent
from scripts.utilities.config import AppConfig
from scripts.utilities.models import ChatMessage


def _config(**overrides: Any) -> AppConfig:
    values = {
        "api_base_url": "http://localhost:11434/v1",
        "api_key": None,
        "model": "local-model",
        "data_dir": Path("data"),
        "session_dir": Path(".local/sessions"),
        "ranker": "none",
        "rank_model": "rank-model",
        "top_k": 5,
        "candidate_model": "candidate-model",
    }
    values.update(overrides)
    return AppConfig(**values)


@dataclass
class _Delta:
    content: str | None


@dataclass
class _Choice:
    delta: _Delta


@dataclass
class _Chunk:
    choices: list[_Choice]


class _Completions:
    def __init__(self, chunks: list[_Chunk], error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> list[_Chunk]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.chunks


@dataclass
class _FakeClient:
    completions: _Completions

    @property
    def chat(self) -> Any:
        return self


def test_stream_yields_nonempty_chunk_content_in_order_and_adds_grounded_prompt() -> None:
    completions = _Completions(
        [
            _Chunk([_Choice(_Delta("first"))]),
            _Chunk([_Choice(_Delta(None))]),
            _Chunk([_Choice(_Delta(" second"))]),
        ]
    )
    agent = OpenAICompatibleAgent(
        _config(), _FakeClient(completions)  # type: ignore[arg-type]
    )

    assert list(agent.stream([ChatMessage("user", "Context: [doc:d1] facts")])) == [
        "first",
        " second",
    ]
    request = completions.calls[0]
    assert request["model"] == "local-model"
    assert request["stream"] is True
    assert request["messages"][0]["role"] == "system"
    system_prompt = request["messages"][0]["content"]
    assert "supplied context" in system_prompt
    assert "[doc:<id>]" in system_prompt
    assert "insufficient evidence" in system_prompt
    assert "unsupported agronomic advice" in system_prompt


def test_agent_backend_protocol_is_implemented() -> None:
    assert isinstance(OpenAICompatibleAgent(_config(), _FakeClient(_Completions([]))), AgentBackend)


def test_build_agent_uses_nullable_local_key_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)

    result = build_agent(_config())

    assert isinstance(result, OpenAICompatibleAgent)
    assert captured == {
        "base_url": "http://localhost:11434/v1",
        "api_key": agent_module._LOCAL_API_KEY_PLACEHOLDER,
    }
    assert "secret" not in repr(captured)


def test_build_agent_passes_cloud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)

    build_agent(_config(api_base_url="https://api.example/v1", api_key="cloud-secret"))

    assert captured == {"base_url": "https://api.example/v1", "api_key": "cloud-secret"}


@pytest.mark.parametrize("field", ["api_base_url", "model"])
def test_agent_rejects_missing_endpoint_or_model(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        OpenAICompatibleAgent(_config(**{field: "   "}), _FakeClient(_Completions([])))


def test_build_agent_does_not_make_network_request(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = object()

    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)
    agent = build_agent(_config())

    assert isinstance(agent, OpenAICompatibleAgent)


def test_provider_error_is_surfaced_without_secret_exposure() -> None:
    secret = "cloud-secret"
    completions = _Completions([], RuntimeError(f"provider rejected key {secret}"))
    agent = OpenAICompatibleAgent(
        _config(api_key=secret), _FakeClient(completions)  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="agent provider request failed") as exc_info:
        list(agent.stream([ChatMessage("user", "question")]))

    assert secret not in str(exc_info.value)

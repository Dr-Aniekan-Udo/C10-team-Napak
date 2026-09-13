"""OpenAI-compatible streaming agent backend."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

from openai import OpenAI

from scripts.utilities.config import AppConfig
from scripts.utilities.models import ChatMessage


_LOCAL_API_KEY_PLACEHOLDER = "agro-rag-local"
_GROUNDED_SYSTEM_PROMPT = """You are an agricultural information assistant.

Answer using only the supplied context and conversation. Cite every
claim grounded in a retrieved document with [doc:<id>], using the document ID
exactly as supplied. If there is insufficient evidence, say so clearly instead
of guessing. Do not provide unsupported agronomic advice or
invent sources, facts, citations, diagnoses, or recommendations.
"""


@runtime_checkable
class AgentBackend(Protocol):
    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        """Yield response text fragments in provider order."""
        ...


class OpenAICompatibleAgent:
    """Small adapter around OpenAI-compatible chat-completion endpoints."""

    def __init__(self, config: AppConfig, client: OpenAI | None = None) -> None:
        if not isinstance(config.api_base_url, str) or not config.api_base_url.strip():
            raise ValueError("api_base_url must be non-empty")
        if not isinstance(config.model, str) or not config.model.strip():
            raise ValueError("model must be non-empty")
        self._config = config
        self._client = client if client is not None else _make_client(config)

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        request_messages = [
            {"role": "system", "content": _GROUNDED_SYSTEM_PROMPT},
            *(
                {"role": message.role, "content": message.content}
                for message in messages
            ),
        ]
        try:
            chunks = self._client.chat.completions.create(
                model=self._config.model,
                messages=request_messages,
                stream=True,
            )
            for chunk in chunks:
                content = _chunk_content(chunk)
                if content:
                    yield content
        except Exception as exc:
            raise RuntimeError("agent provider request failed") from exc


def _make_client(config: AppConfig) -> OpenAI:
    api_key = config.api_key or _LOCAL_API_KEY_PLACEHOLDER
    return OpenAI(base_url=config.api_base_url, api_key=api_key)


def _chunk_content(chunk: Any) -> str | None:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    content = getattr(getattr(choices[0], "delta", None), "content", None)
    return content if isinstance(content, str) else None


def build_agent(config: AppConfig) -> AgentBackend:
    """Construct an agent without making a provider request."""
    return OpenAICompatibleAgent(config)

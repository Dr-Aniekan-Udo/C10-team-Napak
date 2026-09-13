"""Validated immutable domain objects for the Agro-RAG application."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


_ROLES = {"system", "user", "assistant"}


def _as_tuple(value: object, name: str, item_type: type) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of {item_type.__name__}")
    items = tuple(value)
    if not all(isinstance(item, item_type) for item in items):
        raise ValueError(f"{name} must contain only {item_type.__name__} values")
    return items


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in _ROLES:
            raise ValueError(f"role must be one of {sorted(_ROLES)}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be non-empty")


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    title: str
    text: str
    score: float
    rank: int
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str):
            raise ValueError("document_id must be a string")
        if not isinstance(self.title, str):
            raise ValueError("title must be a string")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        document_id = self.document_id.strip()
        title = self.title.strip()
        text = self.text.strip()
        if not document_id:
            raise ValueError("document_id must be non-empty")
        if not title:
            raise ValueError("title must be non-empty")
        if not text:
            raise ValueError("text must be non-empty")
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "text", text)
        if self.source is not None:
            if not isinstance(self.source, str):
                raise ValueError("source must be a string or None")
            source = self.source.strip()
            object.__setattr__(self, "source", source or None)
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("score must be a number")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be a positive integer")


@dataclass(frozen=True)
class SessionTurn:
    user: ChatMessage
    assistant: ChatMessage
    sources: tuple[RetrievedDocument, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.user, ChatMessage):
            raise ValueError("user must be a ChatMessage")
        if not isinstance(self.assistant, ChatMessage):
            raise ValueError("assistant must be a ChatMessage")
        object.__setattr__(self, "sources", tuple(_as_tuple(self.sources, "sources", RetrievedDocument)))


@dataclass(frozen=True)
class Session:
    session_id: str
    turns: tuple[SessionTurn, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be non-empty")
        object.__setattr__(self, "turns", tuple(_as_tuple(self.turns, "turns", SessionTurn)))


@dataclass(frozen=True)
class RagResponse:
    answer: str
    sources: tuple[RetrievedDocument, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer must be non-empty")
        object.__setattr__(self, "sources", tuple(_as_tuple(self.sources, "sources", RetrievedDocument)))

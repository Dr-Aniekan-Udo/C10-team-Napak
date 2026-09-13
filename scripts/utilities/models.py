"""Validated immutable domain objects for the Agro-RAG application."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"role must be one of {sorted(_ROLES)}")
        if not self.content.strip():
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
            source = self.source.strip()
            object.__setattr__(self, "source", source or None)
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be a positive integer")


@dataclass(frozen=True)
class SessionTurn:
    user: ChatMessage
    assistant: ChatMessage
    sources: tuple[RetrievedDocument, ...] = ()


@dataclass(frozen=True)
class Session:
    session_id: str
    turns: tuple[SessionTurn, ...] = ()

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must be non-empty")


@dataclass(frozen=True)
class RagResponse:
    answer: str
    sources: tuple[RetrievedDocument, ...] = ()

    def __post_init__(self) -> None:
        if not self.answer.strip():
            raise ValueError("answer must be non-empty")

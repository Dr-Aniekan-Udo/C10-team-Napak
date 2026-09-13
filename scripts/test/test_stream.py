from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from scripts.utilities.models import ChatMessage, RetrievedDocument, Session, SessionTurn
from scripts.utilities.stream import RagPipeline


SESSION_ID = "00000000-0000-4000-8000-000000000001"


def document(document_id: str, title: str = "Title") -> RetrievedDocument:
    return RetrievedDocument(
        document_id=document_id,
        title=title,
        text=f"Facts from {document_id}",
        score=1.0,
        rank=1,
        source="Extension service",
        source_url=f"https://example.test/{document_id}",
    )


class FakeProcessor:
    def __init__(self) -> None:
        self.context_calls: list[tuple[RetrievedDocument, ...]] = []

    def context(self, documents: tuple[RetrievedDocument, ...]) -> str:
        self.context_calls.append(documents)
        return "\n".join(f"[{item.document_id}] {item.text}" for item in documents)


class FakeRanker:
    def __init__(self, results: list[RetrievedDocument]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def rank(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        self.calls.append((query, top_k))
        return self.results


class FakeAgent:
    def __init__(self, chunks: list[str], error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.calls: list[tuple[ChatMessage, ...]] = []

    def stream(self, messages: tuple[ChatMessage, ...]) -> Any:
        self.calls.append(messages)
        yield from self.chunks
        if self.error is not None:
            raise self.error


@dataclass
class FakeSessions:
    session: Session

    def __post_init__(self) -> None:
        self.loads: list[str] = []
        self.saved: list[Session] = []

    def load(self, session_id: str) -> Session:
        self.loads.append(session_id)
        if session_id != self.session.session_id:
            raise FileNotFoundError(session_id)
        return self.session

    def save(self, session: Session) -> None:
        self.saved.append(session)
        self.session = session


def pipeline(
    *,
    turns: tuple[SessionTurn, ...] = (),
    chunks: list[str] | None = None,
    results: list[RetrievedDocument] | None = None,
    top_k: int = 2,
    error: Exception | None = None,
) -> tuple[RagPipeline, FakeAgent, FakeRanker, FakeSessions, FakeProcessor]:
    sessions = FakeSessions(Session(SESSION_ID, turns))
    processor = FakeProcessor()
    ranker = FakeRanker(results or [document("d1"), document("d2"), document("d3")])
    agent = FakeAgent(["first", " second"] if chunks is None else chunks, error)
    return RagPipeline(processor, ranker, agent, sessions, top_k), agent, ranker, sessions, processor


def test_stream_chat_retrieves_top_k_streams_in_order_and_propagates_sources() -> None:
    rag, agent, ranker, sessions, processor = pipeline()

    events = list(rag.stream_chat(SESSION_ID, "What grows?"))

    sources = (document("d1"), document("d2"))
    assert events == [("first", sources), (" second", sources)]
    assert ranker.calls == [("What grows?", 2)]
    assert processor.context_calls == [sources]
    assert len(agent.calls) == 1
    assert agent.calls[0][-1].role == "user"
    assert "[d1] Facts from d1" in agent.calls[0][-1].content
    assert "What grows?" in agent.calls[0][-1].content
    assert sessions.saved[0].turns[-1].sources == sources


def test_respond_consumes_stream_and_returns_same_sources() -> None:
    rag, _, _, sessions, _ = pipeline(chunks=["answer"])

    response = rag.respond(SESSION_ID, "question")

    assert response.answer == "answer"
    assert response.sources == sessions.session.turns[-1].sources
    assert len(sessions.saved) == 1


def test_history_is_bounded_to_recent_messages_and_characters() -> None:
    turns = tuple(
        SessionTurn(ChatMessage("user", f"old-user-{index}"), ChatMessage("assistant", f"old-answer-{index}"))
        for index in range(20)
    )
    rag, agent, _, _, _ = pipeline(turns=turns, chunks=["answer"])

    list(rag.stream_chat(SESSION_ID, "new question"))

    messages = agent.calls[0]
    assert len(messages) <= RagPipeline.MAX_HISTORY_MESSAGES + 1
    history = messages[:-1]
    assert len("".join(item.content for item in history)) <= RagPipeline.MAX_HISTORY_CHARS
    assert "old-user-0" not in {item.content for item in history}
    assert messages[-1].content.endswith("new question")


@pytest.mark.parametrize("message", ["", "   ", 42])
def test_rejects_empty_or_non_string_input_without_loading_session(message: object) -> None:
    rag, _, _, sessions, _ = pipeline()

    with pytest.raises(ValueError, match="message"):
        list(rag.stream_chat(SESSION_ID, message))  # type: ignore[arg-type]

    assert sessions.loads == []


def test_provider_failure_does_not_save_partial_turn() -> None:
    rag, _, _, sessions, _ = pipeline(chunks=["partial"], error=RuntimeError("provider failed"))

    with pytest.raises(RuntimeError, match="provider failed"):
        list(rag.stream_chat(SESSION_ID, "question"))

    assert sessions.saved == []
    assert sessions.session.turns == ()


def test_session_load_failure_is_preserved() -> None:
    rag, _, _, _, _ = pipeline()

    with pytest.raises(FileNotFoundError):
        list(rag.stream_chat("00000000-0000-4000-8000-000000000002", "question"))


def test_empty_agent_answer_is_not_persisted() -> None:
    rag, _, _, sessions, _ = pipeline(chunks=[])

    with pytest.raises(ValueError, match="answer"):
        list(rag.stream_chat(SESSION_ID, "question"))

    assert sessions.saved == []

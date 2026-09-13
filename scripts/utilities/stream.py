"""Composition boundary for retrieval-grounded chat sessions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from .agent import AgentBackend
from .document_processor import DocumentProcessor
from .models import ChatMessage, RagResponse, RetrievedDocument, Session, SessionTurn
from .ranker import Ranker
from .sessions import SessionStore


class RagPipeline:
    """Retrieve bounded context, stream an answer, then atomically save its turn."""

    MAX_HISTORY_MESSAGES = 10
    MAX_HISTORY_CHARS = 4000

    def __init__(
        self,
        processor: DocumentProcessor,
        ranker: Ranker,
        agent: AgentBackend,
        sessions: SessionStore,
        top_k: int = 5,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.processor = processor
        self.ranker = ranker
        self.agent = agent
        self.sessions = sessions
        self.top_k = top_k

    def respond(self, session_id: str, message: str) -> RagResponse:
        fragments = []
        sources: tuple[RetrievedDocument, ...] = ()
        for fragment, sources in self.stream_chat(session_id, message):
            fragments.append(fragment)
        return RagResponse("".join(fragments), sources)

    def stream_chat(
        self, session_id: str, message: str
    ) -> Iterator[tuple[str, tuple[RetrievedDocument, ...]]]:
        self._validate_message(message)
        session = self._load_session(session_id)
        retrieved = tuple(self.ranker.rank(message, self.top_k))[: self.top_k]
        context = self.processor.context(retrieved)
        messages = self._agent_messages(session, context, message)

        fragments: list[str] = []
        for fragment in self.agent.stream(messages):
            if not isinstance(fragment, str):
                raise ValueError("agent returned a non-string fragment")
            if not fragment:
                continue
            fragments.append(fragment)
            yield fragment, retrieved

        answer = "".join(fragments)
        if not answer.strip():
            raise ValueError("agent returned an empty answer")
        turn = SessionTurn(
            user=ChatMessage("user", message),
            assistant=ChatMessage("assistant", answer),
            sources=retrieved,
        )
        self.sessions.save(Session(session.session_id, session.turns + (turn,)))

    @staticmethod
    def _validate_message(message: object) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

    def _load_session(self, session_id: str) -> Session:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        session = self.sessions.load(session_id)
        if not isinstance(session, Session) or session.session_id != session_id:
            raise ValueError("session store returned an invalid session")
        return session

    def _agent_messages(
        self, session: Session, context: str, message: str
    ) -> tuple[ChatMessage, ...]:
        history: list[ChatMessage] = []
        characters = 0
        for turn in reversed(session.turns):
            turn_messages = (turn.user, turn.assistant)
            if len(history) + len(turn_messages) > self.MAX_HISTORY_MESSAGES:
                break
            turn_characters = sum(len(item.content) for item in turn_messages)
            if characters + turn_characters > self.MAX_HISTORY_CHARS:
                break
            history[0:0] = turn_messages
            characters += turn_characters
        prompt = f"Context:\n{context}\n\nQuestion:\n{message}"
        return tuple(history) + (ChatMessage("user", prompt),)

import math

import pytest

from scripts.utilities.models import (
    ChatMessage,
    RagResponse,
    RetrievedDocument,
    Session,
    SessionTurn,
)


def test_chat_message_requires_supported_role_and_content() -> None:
    assert ChatMessage(role="user", content="Question").role == "user"

    for role in (None, 1, object()):
        with pytest.raises(ValueError, match="role"):
            ChatMessage(role=role, content="output")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="role"):
        ChatMessage(role="tool", content="output")
    for content in (None, 1, object()):
        with pytest.raises(ValueError, match="content"):
            ChatMessage(role="user", content=content)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="content"):
        ChatMessage(role="user", content="   ")


def test_retrieved_document_rejects_non_finite_scores_and_invalid_ranks() -> None:
    for field, value in (("document_id", None), ("title", 1), ("text", object())):
        with pytest.raises(ValueError, match=field):
            RetrievedDocument(
                document_id=value if field == "document_id" else "d1",  # type: ignore[arg-type]
                title=value if field == "title" else "Title",  # type: ignore[arg-type]
                text=value if field == "text" else "Text",  # type: ignore[arg-type]
                score=0.5,
                rank=1,
            )
    for score in (None, "0.5", object()):
        with pytest.raises(ValueError, match="score"):
            RetrievedDocument("d1", "Title", "Text", score, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="score"):
        RetrievedDocument(
            document_id="d1",
            title="Title",
            text="Text",
            score=math.nan,
            rank=1,
        )
    for rank in (None, 1.5, True, 0):
        with pytest.raises(ValueError, match="rank"):
            RetrievedDocument("d1", "Title", "Text", 0.5, rank)  # type: ignore[arg-type]


def test_nested_models_validate_types_and_normalize_sequences() -> None:
    source = RetrievedDocument("d1", "Title", "Text", 0.5, 1)
    user = ChatMessage("user", "Question")
    assistant = ChatMessage("assistant", "Answer")

    turn = SessionTurn(user, assistant, [source])  # type: ignore[arg-type]
    session = Session("session-1", [turn])  # type: ignore[arg-type]
    response = RagResponse("Answer", [source])  # type: ignore[arg-type]

    assert turn.sources == (source,)
    assert session.turns == (turn,)
    assert response.sources == (source,)
    with pytest.raises(AttributeError):
        turn.sources.append(source)  # type: ignore[attr-defined]

    for user_value, assistant_value in (("user", assistant), (user, "assistant")):
        with pytest.raises(ValueError, match="user|assistant"):
            SessionTurn(user_value, assistant_value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sources"):
        SessionTurn(user, assistant, ["not a document"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="session_id"):
        Session(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="turns"):
        Session("session-1", ["not a turn"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="answer"):
        RagResponse(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sources"):
        RagResponse("Answer", ["not a document"])  # type: ignore[list-item]


def test_session_turn_and_response_are_frozen_domain_objects() -> None:
    source = RetrievedDocument(
        document_id="d1",
        title="Title",
        text="Text",
        score=0.5,
        rank=1,
    )
    turn = SessionTurn(
        user=ChatMessage(role="user", content="Question"),
        assistant=ChatMessage(role="assistant", content="Answer [doc:d1]"),
        sources=(source,),
    )
    session = Session(session_id="session-1", turns=(turn,))
    response = RagResponse(answer="Answer [doc:d1]", sources=(source,))

    assert session.turns[0].sources == (source,)
    assert response.answer.startswith("Answer")
    with pytest.raises((AttributeError, TypeError)):
        response.answer = "changed"  # type: ignore[misc]

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

    with pytest.raises(ValueError, match="role"):
        ChatMessage(role="tool", content="output")
    with pytest.raises(ValueError, match="content"):
        ChatMessage(role="user", content="   ")


def test_retrieved_document_rejects_non_finite_scores_and_invalid_ranks() -> None:
    with pytest.raises(ValueError, match="score"):
        RetrievedDocument(
            document_id="d1",
            title="Title",
            text="Text",
            score=math.nan,
            rank=1,
        )
    with pytest.raises(ValueError, match="rank"):
        RetrievedDocument(
            document_id="d1",
            title="Title",
            text="Text",
            score=0.5,
            rank=0,
        )


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

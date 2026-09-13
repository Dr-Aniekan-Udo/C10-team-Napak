import json
import os
from pathlib import Path

import pytest

from scripts.utilities.models import ChatMessage, Session, SessionTurn
from scripts.utilities.sessions import SessionStore


def make_session(session_id: str = "00000000-0000-4000-8000-000000000001") -> Session:
    return Session(
        session_id=session_id,
        turns=(
            SessionTurn(
                user=ChatMessage("user", "Question"),
                assistant=ChatMessage("assistant", "Answer [doc:d1]"),
            ),
        ),
    )


def test_create_save_load_and_clear_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    session = store.create()
    assert len(session.session_id) == 36
    assert store.load(session.session_id) == session

    saved = make_session(session.session_id)
    store.save(saved)
    assert store.load(session.session_id) == saved
    store.clear(session.session_id)
    with pytest.raises(FileNotFoundError):
        store.load(session.session_id)


@pytest.mark.parametrize("session_id", ["../escape", "nested/id", "not-a-uuid", ""])
def test_rejects_path_traversal_and_invalid_session_ids(tmp_path: Path, session_id: str) -> None:
    store = SessionStore(tmp_path)

    with pytest.raises(ValueError):
        store.load(session_id)


def test_rejects_corrupt_json_without_overwriting_it(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = make_session()
    path = tmp_path / f"{session.session_id}.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        store.load(session.session_id)
    with pytest.raises(ValueError, match="JSON"):
        store.save(session)
    assert path.read_text(encoding="utf-8") == "{not json"


def test_rejects_malformed_payload(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = make_session()
    path = tmp_path / f"{session.session_id}.json"
    path.write_text(json.dumps({"session_id": session.session_id, "turns": "bad"}), encoding="utf-8")

    with pytest.raises(ValueError, match="payload"):
        store.load(session.session_id)


def test_save_replaces_atomically_and_leaves_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SessionStore(tmp_path)
    session = make_session()
    calls: list[tuple[str, str]] = []
    original_replace = os.replace

    def record_replace(source: str, destination: str) -> None:
        calls.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", record_replace)
    store.save(session)

    assert len(calls) == 1
    assert calls[0][1] == tmp_path / f"{session.session_id}.json"
    assert store.load(session.session_id) == session
    assert list(tmp_path.glob("*.tmp")) == []

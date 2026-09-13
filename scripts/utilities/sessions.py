"""Validated, atomic JSON session persistence."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

from .models import ChatMessage, RetrievedDocument, Session, SessionTurn


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def create(self) -> Session:
        session = Session(str(uuid.uuid4()))
        self.save(session)
        return session

    def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid session JSON") from exc
        return self._from_payload(payload, session_id)

    def save(self, session: Session) -> None:
        if not isinstance(session, Session):
            raise ValueError("session must be a Session")
        path = self._path(session.session_id)
        self.root.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.load(session.session_id)
        payload = json.dumps(self._to_payload(session), indent=2, sort_keys=True) + "\n"
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.root, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary_path = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)

    def clear(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)

    def _path(self, session_id: str) -> Path:
        if not isinstance(session_id, str):
            raise ValueError("session_id must be a UUID")
        try:
            parsed = uuid.UUID(session_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("session_id must be a UUID") from exc
        if str(parsed) != session_id or parsed.version != 4:
            raise ValueError("session_id must be a canonical UUID4")
        root = self.root.resolve()
        path = root / f"{session_id}.json"
        if path.is_symlink():
            raise ValueError("session path must not be a symlink")
        try:
            path.resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise ValueError("session path must remain within session root") from exc
        return path

    @staticmethod
    def _to_payload(session: Session) -> dict[str, object]:
        return {
            "version": 1,
            "session_id": session.session_id,
            "turns": [
                {
                    "user": {"role": turn.user.role, "content": turn.user.content},
                    "assistant": {"role": turn.assistant.role, "content": turn.assistant.content},
                    "sources": [
                        {
                            "document_id": source.document_id,
                            "title": source.title,
                            "text": source.text,
                            "score": source.score,
                            "rank": source.rank,
                            "source": source.source,
                            "source_url": source.source_url,
                        }
                        for source in turn.sources
                    ],
                }
                for turn in session.turns
            ],
        }

    def _from_payload(self, payload: object, session_id: str) -> Session:
        try:
            if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("session_id") != session_id:
                raise ValueError
            turns = []
            for item in payload["turns"]:
                user = ChatMessage(**item["user"])
                assistant = ChatMessage(**item["assistant"])
                sources = tuple(RetrievedDocument(**source) for source in item.get("sources", []))
                turns.append(SessionTurn(user, assistant, sources))
            return Session(session_id, tuple(turns))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid session payload") from exc

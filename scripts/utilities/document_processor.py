"""Load and format the corpus documents used by the application."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .models import RetrievedDocument


class DocumentProcessor:
    """Immutable-by-convention document lookup with deterministic context output."""

    MAX_CONTEXT_CHARS = 6000
    REQUIRED_COLUMNS = frozenset({"document_id", "title", "text", "source", "source_url"})

    def __init__(self, documents: Sequence[RetrievedDocument]) -> None:
        self._documents = {document.document_id: document for document in documents}

    @classmethod
    def from_csv(cls, path: Path) -> "DocumentProcessor":
        frame = pd.read_csv(path)
        missing = cls.REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"documents CSV missing required columns: {sorted(missing)}")

        documents: list[RetrievedDocument] = []
        identifiers: set[str] = set()
        for row_number, row in frame.iterrows():
            values = {column: row[column] for column in cls.REQUIRED_COLUMNS}
            if any(pd.isna(value) for value in values.values()):
                if pd.isna(values["source_url"]):
                    values["source_url"] = ""
                else:
                    raise ValueError(f"documents CSV row {row_number + 2} contains null values")
            document_id = str(values["document_id"]).strip()
            if not document_id:
                raise ValueError("document_id must be non-empty")
            if document_id in identifiers:
                raise ValueError("document_id values must be unique")
            identifiers.add(document_id)
            documents.append(
                RetrievedDocument(
                    document_id=document_id,
                    title=str(values["title"]),
                    text=str(values["text"]),
                    score=0.0,
                    rank=len(documents) + 1,
                    source=str(values["source"]),
                    source_url=str(values["source_url"]),
                )
            )
        return cls(documents)

    def get(self, document_id: str) -> RetrievedDocument:
        try:
            return self._documents[document_id]
        except KeyError as exc:
            raise KeyError(f"unknown document_id: {document_id}") from exc

    def context(self, documents: Sequence[RetrievedDocument]) -> str:
        documents = tuple(documents)
        if not documents:
            return ""
        headers = [self._citation_header(index, document) for index, document in enumerate(documents, 1)]
        fixed_length = sum(len(header) + 1 for header in headers) + max(0, len(documents) - 1)
        text_budget = max(0, self.MAX_CONTEXT_CHARS - fixed_length)
        base, remainder = divmod(text_budget, len(documents))
        blocks = []
        for index, (header, document) in enumerate(zip(headers, documents)):
            limit = base + (1 if index < remainder else 0)
            blocks.append(f"{header}\n{document.text[:limit]}")
        return "\n\n".join(blocks)[: self.MAX_CONTEXT_CHARS]

    @staticmethod
    def _citation_header(index: int, document: RetrievedDocument) -> str:
        source = document.source or "Unknown"
        url = document.source_url or "Unavailable"
        return f"[{index}] {document.title}\nSource: {source}\nURL: {url}"
